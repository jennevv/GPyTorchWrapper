import math

import torch
from gpytorch.kernels import Kernel
from torch import Tensor

from src.utils.input_transformer import xyz_to_invdist_torch
from src.utils.permutational_invariance import generate_permutations


class MaternKernelPermInv(Kernel):
    has_lengthscale = True

    def __init__(self, n_atoms: int, idx_equiv_atoms: list[list[int]], select_dims: Tensor = None, nu: float = 2.5,
                 **kwargs):
        super().__init__(**kwargs)

        if nu not in {0.5, 1.5, 2.5}:
            raise NotImplementedError('Please select one of the following nu values: {0.5, 1.5, 2.5}')
        if self.ard_num_dims is not None:
            raise NotImplementedError(
                'ARD is not supported for MaternKernelPermInv. This will lead to an ill-conditioned covariance matrix.')
        if self.active_dims is not None:
            raise NotImplementedError(
                'Active dimensions are not supported for MaternKernelPermInv. Please use select_dims instead.')


        self.select_dims = select_dims
        self.nu = nu
        self.idx_equiv_atoms = idx_equiv_atoms

        dims = torch.arange(0, n_atoms * 3).reshape(n_atoms, 3)
        self.dims = dims

        self.permutations = generate_permutations(idx_equiv_atoms)

    def forward(self, x1, x2, diag=False, **params):
        k_sum = 0
        num_perms = len(self.permutations)
        init_perm = self.permutations[0]

        for p in self.permutations:
            x2_perm = x2.clone()
            x2_perm[:, self.dims[init_perm, :].flatten()] = x2[:, self.dims[p, :].flatten()]

            # Transform xyz coordinates to internuclear distances
            x1_interdist = xyz_to_invdist_torch(x1)
            x2_perm_interdist = xyz_to_invdist_torch(x2_perm)

            if self.select_dims is not None:
                x1_interdist = x1_interdist.index_select(-1, self.select_dims)
                x2_perm_interdist = x2_perm_interdist.index_select(-1, self.select_dims)

            mean = x1_interdist.mean(dim=-2, keepdim=True)

            x1_ = (x1_interdist - mean).div(self.lengthscale)
            x2_ = (x2_perm_interdist - mean).div(self.lengthscale)

            distance = self.covar_dist(x1_, x2_, diag=diag, **params)

            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            else:
                raise NotImplementedError('Please select one of the following nu values: {0.5, 1.5, 2.5}')

            k_sum += (constant_component * exp_component)

        return 1 / num_perms * k_sum
