import gpytorch
import torch


class ModelEvaluator:
    """
    Class for evaluating the rmse and correlation of the model predictions on the selected dataset
    """

    def __init__(self, model: object, likelihood: object):
        self.model = model
        self.likelihood = likelihood

    def _predict(self, x: torch.Tensor) -> gpytorch.distributions.Distribution:
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(x))
        return predictions

    def _rmse(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.sqrt(torch.mean(torch.square(a - b))).item()

    def _check_if_tensor(self, tensor):
        if not torch.is_tensor(tensor):
            raise NotImplementedError("The input should be a PyTorch tensor.")

    def _compare_mean_and_output_dimensions(self, output, mean) -> None:
        if output.squeeze().dim() != mean.squeeze().dim():
            raise ValueError('The number of output dimensions does not match the number of prediction dimensions.')

    def evaluate_rmse(self, x: torch.Tensor, y: torch.Tensor) -> list[float]:
        self._check_if_tensor(x)
        self._check_if_tensor(y)

        predictions = self._predict(x)

        self._compare_mean_and_output_dimensions(y, predictions.mean)

        rmse = []

        if y.dim() > 1:
            for i in range(y.shape[1]):
                rmse.append(self._rmse(predictions.mean[:, i], y[:, i]))
        else:
            rmse.append(self._rmse(predictions.mean, y))
        return rmse

    def evaluate_correlation(self, x: torch.Tensor, y: torch.Tensor) -> list[float]:
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
