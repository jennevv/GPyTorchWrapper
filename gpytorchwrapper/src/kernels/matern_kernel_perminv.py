import math
from typing import Optional

import torch
from torch import Tensor

from gpytorchwrapper.src.kernels.perminv_kernel import PermInvKernel
from gpytorchwrapper.src.utils.input_transformer import xyz_to_dist_torch


class MaternKernelPermInv(PermInvKernel):
    has_lengthscale = True

    def __init__(
        self,
        n_atoms: int,
        idx_equiv_atoms: list[list[int]],
        select_dims: Tensor = None,
        nu: float = 2.5,
        ard: bool = False,
        representation: str = "invdist",
        **kwargs,
    ):
        super().__init__(
            n_atoms=n_atoms,
            idx_equiv_atoms=idx_equiv_atoms,
            select_dims=select_dims,
            ard=ard,
            **kwargs,
        )

        if nu not in {0.5, 1.5, 2.5}:
            raise NotImplementedError(
                "Please select one of the following nu values: {0.5, 1.5, 2.5}"
            )

        if self.active_dims is not None:
            raise NotImplementedError(
                "Keyword active_dims is not supported for MaternKernelPermInv. Please use select_dims instead."
            )

        self.nu = nu
        self.representation = representation

    def matern_kernel(self, x1, x2, diag, **params):
        mean = x1.mean(dim=-2, keepdim=True)

        if self.ard:
            perminv_ard_lengthscale = self.lengthscale.clone()[0][
                self.ard_expansion
            ].unsqueeze(0)
            x1_ = (x1 - mean).div(perminv_ard_lengthscale)
            x2_ = (x2 - mean).div(perminv_ard_lengthscale)
        else:
            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)

        distance = self.covar_dist(x1_, x2_, diag=diag, **params)

        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (
                (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
            )
        else:
            raise NotImplementedError(
                "Please select one of the following nu values: {0.5, 1.5, 2.5}"
            )

        return constant_component * exp_component

    def forward(
        self, x1, x2, diag=False, last_dim_is_batch: Optional[bool] = False, **params
    ):
        k_sum = 0
        num_perms = len(self.permutations)
        init_perm = self.permutations[0]

        x1_dist = xyz_to_dist_torch(x1, representation=self.representation)
        x2_dist = (
            xyz_to_dist_torch(x2, representation=self.representation)
            if not torch.equal(x1, x2)
            else x1_dist
        )

        for perm in self.permutations:
            x2_dist_perm = x2_dist.clone()
            x2_dist_perm[:, init_perm] = x2_dist[:, perm]

            if self.select_dims is not None:
                select_dims_tensor = torch.tensor(self.select_dims)
                x1_dist = torch.index_select(x1_dist, 1, select_dims_tensor)
                x2_dist_perm = torch.index_select(x2_dist_perm, 1, select_dims_tensor)

            k_sum += self.matern_kernel(x1_dist, x2_dist_perm, diag, **params)

        return 1 / num_perms * k_sum
