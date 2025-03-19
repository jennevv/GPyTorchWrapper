import torch
from gpytorch import kernels, means, models, distributions


from gpytorchwrapper.src.kernels.linearxmatern_kernel_perminv import (
    LinearxMaternKernelPermInv,
)


class H2OKrMultitaskGPModel(models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(H2OKrMultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        n_atoms = 4
        idx_equiv_atoms = [[1, 2]]
        dims = [2, 4, 5]

        self.mean_module = means.MultitaskMean(
            means.ConstantMean(),
            num_tasks=3,
        )

        self.covar_module = kernels.MultitaskKernel(
            LinearxMaternKernelPermInv(
                n_atoms=n_atoms,
                idx_equiv_atoms=idx_equiv_atoms,
                select_dims=dims,
                # lengthscale_constraint=l_constraint
            ),
            num_tasks=3,
            rank=1,
        )

        self.mean_module.base_mean = torch.tensor([0, 0.99640255494, 0.99640255494])
        self.mean_module.base_means[0].raw_constant.requires_grad = False
        self.mean_module.base_means[1].raw_constant.requires_grad = False
        self.mean_module.base_means[2].raw_constant.requires_grad = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultitaskMultivariateNormal(mean_x, covar_x)
