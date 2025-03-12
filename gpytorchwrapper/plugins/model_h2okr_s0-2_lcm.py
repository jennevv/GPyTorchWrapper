import gpytorch
import torch

from gpytorchwrapper.src.kernels import LinearxMaternKernelPermInv, LinearxRBFKernelPermInv
from botorch.models.gpytorch import GPyTorchModel

class EnergyErrorLoss(gpytorch.mlls.AddedLossTerm):
    def __init__(self, train_y, pred_y):
        if not train_y.shape == pred_y.shape:
            raise ValueError("Tensor containing targets and predictions have a different shape.")
        self.train_y, self.pred_y = train_y, pred_y

    def loss(self, *args, **kwargs):
        # Calculate non-redundant, unique energy differences between each state
        train_delta = self.train_y[:, None, :] - self.train_y[:, : , None]
        pred_delta = self.pred_y[:, None, :] - self.pred_y[:, : , None]
        train_delta_vec = train_delta[
            torch.triu(torch.ones(self.train_y.shape[0],self.train_y.shape[1], self.train_y.shape[1]), diagonal=1) == 1 # Non-zero upper triangle elements, no diagonal
        ].reshape(self.train_y.shape[0], -1)
        pred_delta_vec = pred_delta[
            torch.triu(torch.ones(self.train_y.shape[0], self.train_y.shape[1], self.train_y.shape[1]), diagonal=1) == 1
        ].reshape(self.train_y.shape[0], -1)

        abs_error = torch.abs(train_delta_vec - pred_delta_vec)

        return torch.mean(abs_error)

class LCMModel(gpytorch.models.ExactGP, GPyTorchModel):
    def __init__(self, train_x, train_y, likelihood):
        super(LCMModel, self).__init__(train_x, train_y, likelihood)
        num_atoms = 4
        idx_equiv_atoms = [[1,2]]
        ard = True
        select_dims = [2, 4, 5]

        base_kernels = [
            LinearxMaternKernelPermInv(
            n_atoms=num_atoms,
            idx_equiv_atoms=idx_equiv_atoms,
            select_dims=select_dims,
            ard=ard,
            nu=2.5),
            LinearxRBFKernelPermInv(
            n_atoms=num_atoms,
            idx_equiv_atoms=idx_equiv_atoms,
            select_dims=select_dims,
            ard=ard)
        ]

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=3
        )
        self.covar_module = gpytorch.kernels.LCMKernel(
            base_kernels=base_kernels, num_tasks=3, rank=1
        )

        self.register_added_loss_term("energy_error_loss")

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        output = gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

        new_added_loss_term = EnergyErrorLoss(self.train_targets, output.mean)
        self.update_added_loss_term("energy_error_loss", new_added_loss_term)

        return output
