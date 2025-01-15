import torch
from functorch.dim import Tensor
from gpytorch.functions import RBFCovariance
from gpytorch.kernels import Kernel
from gpytorch.kernels.rbf_kernel import postprocess_rbf
from gpytorch.settings import trace_mode

from gpytorchwrapper.src.utils.input_transformer import xyz_to_invdist_torch
from gpytorchwrapper.src.utils.permutational_invariance import generate_permutations


class RBFKernelPermInv(Kernel):
    has_lengthscale = True

    def __init__(self, n_atoms: int, idx_equiv_atoms: list[list[int]], select_dims: Tensor = None, **kwargs):
        super().__init__(**kwargs)

        if self.ard_num_dims is not None:
            raise NotImplementedError(
                'ARD is not supported for RBFKernelPermInv. This will lead to an ill-conditioned covariance matrix.')
        if self.active_dims is not None:
            raise NotImplementedError(
                'Active dimensions are not supported for RBFKernelPermInv. Please use select_dims instead.')

        self.idx_equiv_atoms = idx_equiv_atoms
        self.select_dims = select_dims
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

            if self.select_dims:
                x1_interdist = x1_interdist[:, self.select_dims].clone()
                x2_perm_interdist = x2_perm_interdist[:, self.select_dims].clone()

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
                    lambda x1_interdist, x2_perm_interdist: self.covar_dist(x1_interdist, x2_perm_interdist,
                                                                            square_dist=True, diag=False, **params),
                )

        return 1 / num_perms * k_sum
