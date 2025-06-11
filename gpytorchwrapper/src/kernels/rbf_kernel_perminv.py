from typing import Optional

import torch
from gpytorch.kernels.rbf_kernel import postprocess_rbf

from gpytorchwrapper.src.kernels.perminv_kernel import PermInvKernel
from gpytorchwrapper.src.utils.input_transformer import xyz_to_dist_torch


class RBFKernelPermInv(PermInvKernel):
    has_lengthscale = True

    def __init__(
        self,
        n_atoms: int,
        idx_equiv_atoms: list[list[int]],
        select_dims: list[int] = None,
        ard: bool = False,
        representation: str = "invdist",
        **kwargs,
    ):
        """
        Initialize the RBFKernelPermInv kernel.

        Parameters
        ----------
        n_atoms : int
            Number of atoms in the molecule or structure.
        idx_equiv_atoms : list of list of int
            Groups of indices indicating equivalent atoms under permutations.
        select_dims : list of int, optional
            Dimensions to select from the distance representation.
        ard : bool, default=False
            If True, use automatic relevance determination (ARD).
        representation : str, default="invdist"
            The type of representation to use for distances, choose from:
                `invdist` for inverse distances
                `morse` for features exp(-r_ij)
        **kwargs
            Additional keyword arguments for the base class.

        Raises
        ------
        NotImplementedError
            If `active_dims` is provided in `kwargs`, which is not supported.
        """
        super().__init__(
            n_atoms=n_atoms,
            idx_equiv_atoms=idx_equiv_atoms,
            select_dims=select_dims,
            ard=ard,
            **kwargs,
        )
        if self.active_dims is not None:
            raise NotImplementedError(
                "Keyword active_dims is not supported for this kernel. Please use select_dims instead."
            )

        self.representation = representation

    def rbf_kernel(self, x1, x2, diag, **params):
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

        return postprocess_rbf(
            self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)
        )

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
            else x1_dist.clone()
        )
        if self.select_dims is not None:
            select_dims_tensor = torch.tensor(self.select_dims)
            x1_dist = torch.index_select(x1_dist, 1, select_dims_tensor)

        for perm in self.permutations:
            x2_dist_perm = x2_dist.clone()
            x2_dist_perm[:, init_perm] = x2_dist[:, perm]

            if self.select_dims is not None:
                x2_dist_perm = torch.index_select(x2_dist_perm, 1, select_dims_tensor)

            k_sum += self.rbf_kernel(x1_dist, x2_dist_perm, diag=diag, **params)
        return 1 / num_perms * k_sum
