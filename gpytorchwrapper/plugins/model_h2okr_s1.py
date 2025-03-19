from gpytorch import kernels, means, models, distributions, constraints

from gpytorchwrapper.src.kernels.linearxmatern_kernel_perminv import (
    LinearxMaternKernelPermInv,
)
from botorch.models.gpytorch import GPyTorchModel


class H2OKrS1(models.ExactGP, GPyTorchModel):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        n_atoms = 4
        idx_equiv_atoms = [[1, 2]]
        dims = [2, 4, 5]

        l_constraint = constraints.GreaterThan(0.075)

        self.mean_module = means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(
            LinearxMaternKernelPermInv(
                n_atoms=n_atoms,
                idx_equiv_atoms=idx_equiv_atoms,
                select_dims=dims,
            )
        )

        self.mean_module.raw_constant.requires_grad = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultivariateNormal(mean_x, covar_x)
