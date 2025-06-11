import logging

import gpytorch
import torch

from torch import Tensor

from gpytorch.models import ExactGP
from gpytorch.likelihoods import (
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
    FixedNoiseGaussianLikelihood,
    Likelihood,
)

logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.float64)


class ModelEvaluator:
    """
    Class for evaluating the rmse and correlation of the model predictions on the selected dataset.
    """

    def __init__(
        self,
        model: ExactGP,
        likelihood: GaussianLikelihood | MultitaskGaussianLikelihood,
        output_transformer: object = None,
    ):
        """

        Parameters
        ----------
        model : ExactGP
            The GP model object
        likelihood : GaussianLikelihood or MultitaskGaussianLikelihood
            The likelihood object
        output_transformer : object, optional
            A scikit-learn transformer object
        """
        self.model = model
        self.likelihood = likelihood
        self.output_transformer = output_transformer

    def _predict(self, x: Tensor) -> gpytorch.distributions.Distribution:
        self.model.eval()
        self.likelihood.eval()

        with (
            torch.no_grad(),
            gpytorch.settings.fast_computations(
                covar_root_decomposition=False, log_prob=False, solves=False
            ),
        ):
            if isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
                predictions = self.likelihood(
                    self.model(x),
                    noise=Tensor([self.likelihood.noise[0].item()] * x.shape[0]),
                )
            else:
                predictions = self.likelihood(self.model(x))
        return predictions

    def _rmse(self, a: Tensor, b: Tensor) -> float:
        return torch.sqrt(torch.mean(torch.square(a - b))).item()

    def _check_if_tensor(self, tensor):
        if not torch.is_tensor(tensor):
            raise NotImplementedError("The input should be a PyTorch tensor.")

    def _compare_mean_and_output_dimensions(
        self, output: Tensor, mean: Tensor
    ) -> None:
        if output.squeeze().dim() != mean.squeeze().dim():
            raise ValueError(
                "The number of output dimensions does not match the number of prediction dimensions."
            )

    def evaluate_rmse(self, x: Tensor, y: Tensor) -> list[float]:
        self._check_if_tensor(x)
        self._check_if_tensor(y)

        predictions = self._predict(x)

        self._compare_mean_and_output_dimensions(y, predictions.mean)

        rmse = []

        if self.output_transformer is not None:
            if y.dim() == 1:
                y = torch.as_tensor(
                    self.output_transformer.inverse_transform(y.numpy().reshape(-1, 1))
                )
                mean = torch.as_tensor(
                    self.output_transformer.inverse_transform(
                        predictions.mean.numpy().reshape(-1, 1)
                    )
                )
            else:
                y = torch.as_tensor(
                    self.output_transformer.inverse_transform(y.numpy())
                )
                mean = torch.as_tensor(
                    self.output_transformer.inverse_transform(predictions.mean.numpy())
                )

        else:
            mean = predictions.mean

        if y.dim() > 1:
            for i in range(y.shape[1]):
                rmse.append(self._rmse(mean[:, i], y[:, i]))
        else:
            rmse.append(self._rmse(mean, y))
        return rmse

    def evaluate_correlation(self, x: Tensor, y: Tensor) -> list[float]:
        self._check_if_tensor(x)
        self._check_if_tensor(y)

        predictions = self._predict(x)

        self._compare_mean_and_output_dimensions(y, predictions.mean)

        corr = []

        if y.dim() > 1:
            for i in range(y.shape[1]):
                stack = torch.stack([predictions.mean[:, i], y[:, i]])
                corr_matrix = torch.corrcoef(stack)
                corr.append(float(corr_matrix[0, 1]))
        else:
            stack = torch.stack([predictions.mean, y])
            corr_matrix = torch.corrcoef(stack)
            corr.append(float(corr_matrix[0, 1]))

        return corr


def evaluate_model(
    model: ExactGP,
    likelihood: Likelihood,
    output_transformer: object,
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
) -> tuple[list[float], list[float], list[float]] | tuple[list[float], None, None]:
    """
    Evaluate the model on the training and test sets

    Parameters
    -----------
    model : ExactGP
            The trained model
    likelihood : Likelihood
                 The trained likelihood of the model
    output_transformer : object
        The output transformer 
    train_x : Tensor
              The input training data
    train_y : Tensor
              The output training data
    test_x : Tensor
             The input test data
    test_y : Tensor
             The output test data

    Returns
    --------
    train_rmse : list
                 List containing the RMSE values for the training set
    test_rmse : list or None
                List containing the RMSE values for the test set
    test_corr : list or None
                List containing the correlation values for the test set
    """
    logger.info("Evaluating the model.")

    evaluator = ModelEvaluator(model, likelihood, output_transformer)

    train_rmse = evaluator.evaluate_rmse(train_x, train_y)
    logger.info(f"train_rmse: {train_rmse}")

    if test_x is not None:
        test_rmse = evaluator.evaluate_rmse(test_x, test_y)
        test_corr = evaluator.evaluate_correlation(test_x, test_y)
        logger.info(f"test_rmse: {test_rmse}")
        logger.info(f"test_corr: {test_corr}")
        logger.info("Model evaluation complete.\n")
        return train_rmse, test_rmse, test_corr

    else:
        return train_rmse, None, None
