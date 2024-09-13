from gpytorch import kernels, means, models, distributions
import torch


##### SINGLE OUTPUT MODELS #####
class SingleGPRBF(models.ExactGP):
    """Simple isotropic GP with RBF kernel"""
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = means.ZeroMean()
        self.covar_module = kernels.ScaleKernel(kernels.RBFKernel()) 

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
        self.covar_module = kernels.ScaleKernel(kernels.RBFKernel(ard_num_dims=n_features))

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
        
        self.mean_module = means.ConstantMean(batch_shape=torch.Size([n_output]))
        self.covar_module = kernels.ScaleKernel(kernels.RBFKernel(batch_shape=torch.Size([n_output])))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return distributions.MultitaskMultivariateNormal.from_batch_mvn(
            distributions.MultivariateNormal(mean_x, covar_x)
        )
