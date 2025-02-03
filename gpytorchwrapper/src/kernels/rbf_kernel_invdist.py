from gpytorch.kernels import RBFKernel
from torch import Tensor

from gpytorchwrapper.src.utils.input_transformer import xyz_to_invdist_torch


class RBFKernelInvDist(RBFKernel):
    def __init__(self, select_dims: list | Tensor = None, **kwargs):
        super().__init__(**kwargs)
        if self.active_dims is not None:
            raise NotImplementedError(
                "Active dimensions are not supported for RBFKernelInvDist. Please use select_dims instead."
            )

        self.select_dims = select_dims

    def forward(self, x1, x2, diag=False, **params):
        x1_interdist, x2_interdist = xyz_to_invdist_torch(x1), xyz_to_invdist_torch(x2)

        if self.select_dims is not None:
            x1_interdist = x1_interdist.index_select(-1, self.select_dims)
            x2_interdist = x2_interdist.index_select(-1, self.select_dims)

        return super().forward(x1_interdist, x2_interdist, diag=diag, **params)
