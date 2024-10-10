from gpytorch.kernels import MaternKernel
from torch import Tensor

from src.utils.input_transformer import xyz_to_invdist_torch


class MaternKernelInvDist(MaternKernel):
    def __init__(self, select_dims: list | Tensor = None, **kwargs):
        super().__init__(**kwargs)
        if self.active_dims is not None:
            raise NotImplementedError(
                'Active dimensions are not supported for MaternKernelInvDist. Please use select_dims instead.')

        self.select_dims = select_dims

    def forward(self, x1, x2, diag=False, **params):
        x1_transformed = xyz_to_invdist_torch(x1)
        x2_transformed = xyz_to_invdist_torch(x2) if x2 is not None else x1_transformed

        if self.select_dims is not None:
            x1_transformed = x1_transformed.index_select(-1, self.select_dims)
            x2_transformed = x2_transformed.index_select(-1, self.select_dims)

        return super().forward(x1_transformed, x2_transformed, diag=diag, **params)
