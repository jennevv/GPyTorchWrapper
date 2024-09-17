import pytest
import torch
import gpytorch

from src.models.model_evaluate import ModelEvaluator


class MockModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood):
        super().__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiOutputMockModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood):
        super().__init__(None, None, likelihood)

        num_outputs = 2

        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=num_outputs)
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks=num_outputs)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


@pytest.fixture
def single_output_model_evaluator():
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MockModel(likelihood)
    return ModelEvaluator(model, likelihood)

@pytest.fixture
def multi_output_model_evaluator():
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultiOutputMockModel(likelihood)
    return ModelEvaluator(model, likelihood)


def test_predict(single_output_model_evaluator):
    x = torch.randn(10, 1)
    predictions = single_output_model_evaluator._predict(x)
    assert isinstance(predictions, gpytorch.distributions.MultivariateNormal)
    assert predictions.mean.shape == (10,)


def test_rmse(single_output_model_evaluator):
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.1, 2.1, 3.1])
    rmse = single_output_model_evaluator._rmse(a, b)
    assert isinstance(rmse, float)
    assert pytest.approx(rmse, 0.1) == 0.1


def test_check_if_tensor(single_output_model_evaluator):
    tensor = torch.tensor([1.0, 2.0, 3.0])
    single_output_model_evaluator._check_if_tensor(tensor)  # Should not raise an exception

    with pytest.raises(NotImplementedError):
        single_output_model_evaluator._check_if_tensor([1.0, 2.0, 3.0])


def test_compare_mean_and_output_dimensions(single_output_model_evaluator):
    output = torch.randn(10)
    mean = torch.randn(10)
    single_output_model_evaluator._compare_mean_and_output_dimensions(output, mean)

    with pytest.raises(ValueError):
        single_output_model_evaluator._compare_mean_and_output_dimensions(output, torch.randn(10,2))


def test_evaluate_rmse(single_output_model_evaluator):
    x = torch.randn(10, 3)
    y = torch.randn(10)
    rmse = single_output_model_evaluator.evaluate_rmse(x, y)
    assert isinstance(rmse, list)
    assert len(rmse) == 1
    assert isinstance(rmse[0], float)


def test_evaluate_rmse_multidimensional(multi_output_model_evaluator):
    x = torch.randn(10, 2)
    y = torch.randn(10, 2)
    rmse = multi_output_model_evaluator.evaluate_rmse(x, y)
    assert isinstance(rmse, list)
    assert len(rmse) == 2
    assert all(isinstance(r, float) for r in rmse)


def test_evaluate_correlation(single_output_model_evaluator):
    x = torch.randn(10, 1)
    y = torch.randn(10)
    corr = single_output_model_evaluator.evaluate_correlation(x, y)
    assert isinstance(corr, list)
    assert len(corr) == 1
    assert isinstance(corr[0], float)

def test_evaluate_correlation_multidimensional(multi_output_model_evaluator):
    x = torch.randn(10, 2)
    y = torch.randn(10, 2)
    corr = multi_output_model_evaluator.evaluate_correlation(x, y)
    assert isinstance(corr, list)
    assert len(corr) == 2
    assert all(isinstance(c, float) for c in corr)

def test_input_validation(single_output_model_evaluator):
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)

    with pytest.raises(NotImplementedError):
        single_output_model_evaluator.evaluate_rmse([1, 2, 3], y)

    with pytest.raises(NotImplementedError):
        single_output_model_evaluator.evaluate_correlation(x, [1, 2, 3])


def test_dimension_mismatch(single_output_model_evaluator):
    # Expect a single output dimension
    x = torch.randn(10, 1)
    y = torch.randn(10, 2)

    with pytest.raises(ValueError):
        single_output_model_evaluator.evaluate_rmse(x, y)

    with pytest.raises(ValueError):
        single_output_model_evaluator.evaluate_correlation(x, y)