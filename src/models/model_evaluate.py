import gpytorch
import torch
import logging

logger = logging.getLogger(__name__)

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


def evaluate_model(model, likelihood, train_x, train_y, test_x, test_y):
    """
    Evaluate the model on the training and test sets

    Parameters
    -----------
    model : object
            The trained model
    likelihood : object
                 The trained likelihood of the model
    train_x : torch.Tensor
              The input training data
    train_y : torch.Tensor
              The output training data
    test_x : torch.Tensor
             The input test data
    test_y : torch.Tensor
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
    logger.info('Evaluating the model.')

    evaluator = ModelEvaluator(model, likelihood)

    train_rmse = evaluator.evaluate_rmse(train_x, train_y)
    logger.info(f'train_rmse: {train_rmse}')
    if test_x is not None:
        test_rmse = evaluator.evaluate_rmse(test_x, test_y)
        test_corr = evaluator.evaluate_correlation(test_x, test_y)
        logger.info(f'test_rmse" {test_rmse}\ntest_corr: {test_corr}')
        return train_rmse, test_rmse, test_corr

    else:
        return train_rmse, None, None
