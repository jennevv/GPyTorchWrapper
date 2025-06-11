import gpytorch.models
import pytest
import torch
from gpytorchwrapper.src.kernels import (
    MaternKernelPermInv,
    RBFKernelPermInv,
    LinearxRBFKernelPermInv,
    LinearxMaternKernelPermInv,
    PolyxMaternKernelPermInv,
)


@pytest.fixture
def sample_training_data(n_inputs: int = 3):
    return torch.rand((n_inputs, 9)), torch.rand(n_inputs)


@pytest.fixture
def sample_test_data(n_inputs: int = 3) -> torch.Tensor:
    return torch.rand((n_inputs, 9), requires_grad=True)


class MaternModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            MaternKernelPermInv(n_atoms=3, idx_equiv_atoms=[[0, 1]], ard=True)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RBFModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            RBFKernelPermInv(n_atoms=3, idx_equiv_atoms=[[0, 1]], ard=True)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LinearxRBFModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            LinearxRBFKernelPermInv(n_atoms=3, idx_equiv_atoms=[[0, 1]], ard=True)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LinearxMaternModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            LinearxMaternKernelPermInv(n_atoms=3, idx_equiv_atoms=[[0, 1]], ard=True)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PolyxMaternModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            PolyxMaternKernelPermInv(
                n_atoms=3, idx_equiv_atoms=[[0, 1]], ard=True, power=1
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def numerical_gradient(x: torch.Tensor, f: callable) -> torch.Tensor:
    eps = 1e-6
    grad = torch.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += eps
            f_plus = f(x)
            x[i, j] -= 2 * eps
            f_minus = f(x)
            x[i, j] += eps
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    return grad


def test_matern_model(sample_training_data, sample_test_data):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MaternModel(*sample_training_data, likelihood)
    likelihood.double()
    model.double()
    model.eval()
    likelihood.eval()

    test_x = sample_test_data
    X = torch.autograd.Variable(test_x)

    def mean_f(x):
        return likelihood(model(x)).mean.sum()

    jac = torch.autograd.functional.jacobian(mean_f, X)

    assert torch.allclose(
        jac.squeeze(), numerical_gradient(test_x.clone().detach(), mean_f)
    )


def test_rbf_model(sample_training_data, sample_test_data):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = RBFModel(*sample_training_data, likelihood)
    likelihood.double()
    model.double()
    model.eval()
    likelihood.eval()

    test_x = sample_test_data
    X = torch.autograd.Variable(test_x)

    def mean_f(x):
        return likelihood(model(x)).mean.sum()

    jac = torch.autograd.functional.jacobian(mean_f, X)

    assert torch.allclose(
        jac.squeeze(), numerical_gradient(test_x.clone().detach(), mean_f)
    )


def test_linearxrbf_model(sample_training_data, sample_test_data):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = LinearxRBFModel(*sample_training_data, likelihood)
    likelihood.double()
    model.double()
    model.eval()
    likelihood.eval()

    test_x = sample_test_data
    X = torch.autograd.Variable(test_x)

    def mean_f(x):
        return likelihood(model(x)).mean.sum()

    jac = torch.autograd.functional.jacobian(mean_f, X)

    assert torch.allclose(
        jac.squeeze(), numerical_gradient(test_x.clone().detach(), mean_f)
    )


def test_linearxmatern_model(sample_training_data, sample_test_data):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = LinearxMaternModel(*sample_training_data, likelihood)
    likelihood.double()
    model.double()
    model.eval()
    likelihood.eval()

    test_x = sample_test_data
    X = torch.autograd.Variable(test_x)

    def mean_f(x):
        return likelihood(model(x)).mean.sum()

    jac = torch.autograd.functional.jacobian(mean_f, X)

    assert torch.allclose(
        jac.squeeze(), numerical_gradient(test_x.clone().detach(), mean_f)
    )


def test_polyxmatern_model(sample_training_data, sample_test_data):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = PolyxMaternModel(*sample_training_data, likelihood)
    likelihood.double()
    model.double()
    model.eval()
    likelihood.eval()

    test_x = sample_test_data
    X = torch.autograd.Variable(test_x)

    def mean_f(x):
        return likelihood(model(x)).mean.sum()

    jac = torch.autograd.functional.jacobian(mean_f, X)

    assert torch.allclose(
        jac.squeeze(), numerical_gradient(test_x.clone().detach(), mean_f)
    )
