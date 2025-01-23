from gpytorch import kernels, means, models, distributions
import torch


##### SINGLE OUTPUT MODELS #####
class SingleGPRBF(models.ExactGP):
    """Simple isotropic GP with RBF kernel"""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = means.ZeroMean()
        self.covar_module = (
            kernels.ScaleKernel(kernels.RBFKernel()) + kernels.ConstantKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultivariateNormal(mean_x, covar_x)


class SingleGPRBFARD(models.ExactGP):
    """Learn different lengthscales for each feature"""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        n_features = train_x.shape()[1]

        self.mean_module = means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(
            kernels.RBFKernel(ard_num_dims=n_features)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultivariateNormal(mean_x, covar_x)


class TestModel(models.ExactGP):
    """Test model for debugging"""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultivariateNormal(mean_x, covar_x)


##### MULTIPLE OUTPUT MODELS #####
class BIMultitaskRBF(models.ExactGP):
    """Multi-output GP with RBF kernel"""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        n_output = 4

        self.mean_module = means.MultitaskMean(means.ConstantMean(), num_tasks=n_output)
        self.covar_module = kernels.MultitaskKernel(
            kernels.RBFKernel(), num_tasks=n_output
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class TestMultitaskGPModel(models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(TestMultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = means.MultitaskMean(means.ConstantMean(), num_tasks=2)
        self.covar_module = kernels.MultitaskKernel(
            kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultitaskMultivariateNormal(mean_x, covar_x)
