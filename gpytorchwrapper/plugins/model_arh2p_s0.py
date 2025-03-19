import torch
from gpytorch import kernels, means, models, distributions, constraints, priors

from gpytorchwrapper.src.kernels.polyxmatern_kernel_perminv import PolyxMaternKernelPermInv

from botorch.models.gpytorch import GPyTorchModel

class ArH2pS0(models.ExactGP, GPyTorchModel):

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        outputscale_prior = priors.NormalPrior(5.0, 2.0)
        lengthscale_prior = priors.NormalPrior(0.5, 0.4)

        n_atoms = 3
        idx_equiv_atoms = [[0, 1]]

        self.mean_module = means.ZeroMean()
        self.covar_module = kernels.ScaleKernel(
                PolyxMaternKernelPermInv(
                    n_atoms=n_atoms,
                    idx_equiv_atoms=idx_equiv_atoms,
                    ard=True,
                    nu=2.5,
                    lengthscale_prior=lengthscale_prior,
                    power=1,
                    representation="morse"
                )
                )
        self.covar_module.base_kernel.lengthscale = [lengthscale_prior.mean] * 3
        self.covar_module.outputscale = outputscale_prior.mean

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultivariateNormal(mean_x, covar_x)
