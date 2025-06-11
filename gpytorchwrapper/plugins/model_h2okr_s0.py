from gpytorch import kernels, means, models, distributions

from gpytorchwrapper.src.kernels import (
    PolyxMaternKernelPermInv,
    LinearxMaternKernelPermInv,
)
from botorch.models.gpytorch import GPyTorchModel


class H2OKrS0(models.ExactGP, GPyTorchModel):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        n_atoms = 4
        idx_equiv_atoms = [[1, 2]]
        dims = [2, 4, 5]

        # l_constraint = constraints.GreaterThan(0.075)

        self.mean_module = means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(
            PolyxMaternKernelPermInv(
                n_atoms=n_atoms,
                idx_equiv_atoms=idx_equiv_atoms,
                select_dims=dims,
                ard=True,
                nu=2.5,
                power=1,
                representation="morse",
            )
        )

        self.mean_module.constant = 0.0
        self.mean_module.raw_constant.requires_grad = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultivariateNormal(mean_x, covar_x)
