import itertools

import torch
from gpytorch.kernels import Kernel
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
from gpytorch.kernels.rbf_kernel import postprocess_rbf

class RBFKernelPermInv(Kernel):
    has_lengthscale = True

    def __init__(self, n_atoms: int, idx_equiv_atoms: list[list[int]], **kwargs):
        super(RBFKernelPermInv, self).__init__(**kwargs)
        self.idx_equiv_atoms = idx_equiv_atoms

        dims_cart = torch.arange(0, n_atoms * 3).reshape(n_atoms, 3)
        self.dims = dims_cart

        self.permutations = self.generate_permutations(idx_equiv_atoms)

    @staticmethod
    def generate_permutations(idx_equiv_atoms: list[list[int]]) -> torch.Tensor:
        all_perms = []
        for group in idx_equiv_atoms:
            all_perms.append([list(p) for p in itertools.permutations(group)])

        perms = [list(tup) for tup in itertools.product(*all_perms)]

        perms = [sum((sublist for sublist in item), []) for item in perms]

        perms = torch.tensor(perms)

        return perms

    @staticmethod
    def xyz_to_invdist_torch(x: torch.Tensor, index: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x is a tensor of shape (n, m) where m is the number of individual x, y, z coordinates
        and n is the number of data points.

        The x, y, z coordinates must be ordered as x1, y1, z1, x2, y2, z2, ... , xn, yn, zn

        The final tensor containing the interatomic distances will have the shape (n, m/3) where m/3 is the number of atoms.
        The order of the distances is d01, d02, ..., d12, d13, ..., d(m/3-2)(m/3-1)

        :param x: torch.Tensor
        :param index: bool, returns unique atom indices per distance
        """

        n, m = x.shape
        num_atoms = m // 3

        coords = x.reshape(n, num_atoms, 3)

        # Calculate pairwise distances
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)

        # Create a mask to zero out the diagonal (self-distances)
        mask = torch.eye(num_atoms, dtype=torch.bool)
        dist = dist.masked_fill(mask, 0)

        # Upper triangular indices
        triu_indices = torch.triu_indices(num_atoms, num_atoms, offset=1)

        # Get the upper triangular part of the distance matrix
        interdist = dist[:, triu_indices[0], triu_indices[1]]

        if index:
            return torch.pow(interdist, -1), torch.transpose(triu_indices, -1, -2)

        return torch.pow(interdist, -1)

    def forward(self, x1, x2, diag=False, **params):
        k_sum = 0
        num_perms = len(self.permutations)
        init_perm = self.permutations[0]

        for p in self.permutations:
            x2_perm = x2.clone()
            x2_perm[:, self.dims[init_perm, :].flatten()] = x2[:, self.dims[p, :].flatten()]

            # Transform xyz coordinates to internuclear distances
            x1_interdist = self.xyz_to_invdist_torch(x1)
            x2_perm_interdist = self.xyz_to_invdist_torch(x2_perm)

            if (
                    x1.requires_grad
                    or x2.requires_grad
                    or (self.ard_num_dims is not None and self.ard_num_dims > 1)
                    or diag
                    or params.get("last_dim_is_batch", False)
                    or trace_mode.on()
            ):
                x1_ = x1_interdist.div(self.lengthscale)
                x2_ = x2_perm_interdist.div(self.lengthscale)
                k_sum += postprocess_rbf(self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params))
            else:
                k_sum += RBFCovariance.apply(
                    x1_interdist,
                    x2_perm_interdist,
                    self.lengthscale,
                    lambda x1_interdist, x2_perm_interdist: self.covar_dist(x1_interdist, x2_perm_interdist, square_dist=True, diag=False, **params),
                )

        return 1 / num_perms * k_sum