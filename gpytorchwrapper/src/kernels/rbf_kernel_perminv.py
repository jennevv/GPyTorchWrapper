from typing import Optional

import torch
from gpytorch.functions import RBFCovariance
from gpytorch.kernels import Kernel
from gpytorch.kernels.rbf_kernel import postprocess_rbf
from torch import Tensor
from gpytorch.settings import trace_mode

from gpytorchwrapper.src.utils.input_transformer import xyz_to_invdist_torch
from gpytorchwrapper.src.utils.permutational_invariance import (
    generate_permutations,
    generate_unique_distances,
    generate_ard_expansion,
    generate_interatomic_distance_indices,
)


class RBFKernelPermInv(Kernel):
    has_lengthscale = True

    def __init__(
        self,
        n_atoms: int,
        idx_equiv_atoms: list[list[int]],
        select_dims: Tensor = None,
        ard: bool = False,
        **kwargs,
    ):
        if not ard:
            super().__init__(**kwargs)
            if self.ard_num_dims is not None:
                raise NotImplementedError(
                    "Regular ARD is not supported for RBFKernelPermInv. Set 'ard=True' instead and specify ard_expansion."
                )
        else:
            num_dist = n_atoms * (n_atoms - 1) // 2  # Number of interatomic distances
            ard_num_dims = num_dist if not select_dims else len(select_dims)
            num_unique_distances = generate_unique_distances(
                n_atoms, idx_equiv_atoms
            )  # permutationally unique!
            distance_idx = generate_interatomic_distance_indices(n_atoms)
            if select_dims:
                distance_idx = [distance_idx[i] for i in select_dims]
                ard_expansion = generate_ard_expansion(distance_idx, idx_equiv_atoms)
            else:
                ard_expansion = generate_ard_expansion(distance_idx, idx_equiv_atoms)
                if num_unique_distances != len(set(ard_expansion)):
                    raise ValueError(
                        "The permutationally invariant ARD expansion failed."
                        f"Expected number of unique distances {num_unique_distances} != {len(set(ard_expansion))}"
                        f"ARD expansion: {ard_expansion}"
                    )

            super().__init__(ard_num_dims=ard_num_dims, **kwargs)
            self.ard_expansion = ard_expansion
            self.idx_equiv_atoms = idx_equiv_atoms

        if self.active_dims is not None:
            raise NotImplementedError(
                "Keyword active_dims is not supported for RBFKernelPermInv. Please use select_dims instead."
            )

        self.select_dims = select_dims
        self.ard = ard

        dims = torch.arange(0, n_atoms * 3).reshape(n_atoms, 3)
        self.dims = dims
        self.permutations = generate_permutations(idx_equiv_atoms)

    def forward(
        self, x1, x2, diag=False, last_dim_is_batch: Optional[bool] = False, **params
    ):
        k_sum = 0
        num_perms = len(self.permutations)
        init_perm = self.permutations[0]

        for p in self.permutations:
            x2_perm = x2.clone()
            x2_perm[:, self.dims[init_perm, :].flatten()] = x2[
                :, self.dims[p, :].flatten()
            ]

            # Transform xyz coordinates to internuclear distances
            x1_interdist = xyz_to_invdist_torch(x1)
            x2_perm_interdist = xyz_to_invdist_torch(x2_perm)

            if self.select_dims is not None:
                x1_interdist = torch.index_select(
                    x1_interdist, 1, torch.tensor(self.select_dims)
                )
                x2_perm_interdist = torch.index_select(
                    x2_perm_interdist, 1, torch.tensor(self.select_dims)
                )

            mean = x1_interdist.mean(dim=-2, keepdim=True)

            if self.ard:
                perminv_ard_lengthscale = self.lengthscale.clone()[0][
                    self.ard_expansion
                ].unsqueeze(0)
                if self.select_dims:
                    x1_ = (x1_interdist - mean).div(
                        perminv_ard_lengthscale
                    )
                    x2_ = (x2_perm_interdist - mean).div(
                        perminv_ard_lengthscale
                    )
                else:
                    x1_ = (x1_interdist - mean).div(perminv_ard_lengthscale)
                    x2_ = (x2_perm_interdist - mean).div(perminv_ard_lengthscale)
            else:
                x1_ = (x1_interdist - mean).div(self.lengthscale)
                x2_ = (x2_perm_interdist - mean).div(self.lengthscale)

            if (
                x1.requires_grad
                or x2.requires_grad
                or diag
                or last_dim_is_batch
                or trace_mode.on()
            ):
                k_sum += postprocess_rbf(
                    self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)
                )
            else:
                k_sum += RBFCovariance.apply(
                    x1_,
                    x2_,
                    self.lengthscale,
                    lambda x1_, x2_: self.covar_dist(
                        x1_,
                        x2_,
                        square_dist=True,
                        diag=False,
                        **params,
                    ),
                )

        return 1 / num_perms * k_sum
