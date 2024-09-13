import pathlib
from pathlib import Path
from typing import Union
import subprocess


import gpytorch
import gpytorch.likelihoods as likelihood_module
import joblib
import pandas as pd
import sklearn.preprocessing as transformer_module
import torch
import yaml

import src.models.gp_models as model_module


class DataReader:
    """
    Class for reading data from either a pickle or csv file.
    The data processed dataset must be a pandas DataFrame.
    """

    def __init__(self):
        pass

    def _read_csv(self, file):
        data = pd.read_csv(file)
        return data

    def _read_pickle(self, file, **kwargs):
        pickle = joblib.load(file)
        data = pickle['processed_dataset']
        return data

    def read_data(self, file, file_type, **kwargs):
        if file_type == 'csv':
            try:
                data = self._read_csv(file)
            except:
                raise ImportError(f'The file is not of the file type {file_type}.')
        elif file_type == 'pickle':
            try:
                data = self._read_pickle(file)
            except:
                raise ImportError(f'The file is not of the file type {file_type}.')
        else:
            raise NotImplementedError(f'The file type {file_type} is not implemented. Choose either csv or pickle.')

        return data


class ModelEvaluator:
    """
    Class for evaluating the rmse and correlation of the model predictions on the selected dataset
    """

    def __init__(self, model, likelihood):
        self.model = model
        self.likelihood = likelihood

    def _predict(self, X):
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X))
        return predictions

    def _rmse(self, a, b):
        return torch.sqrt(torch.mean(torch.square(a - b))).item()

    def _check_if_tensor(self, tensor):
        if not torch.is_tensor(tensor):
            raise NotImplementedError("The input should be a PyTorch tensor.")

    def evaluate_rmse(self, X, y) -> list:
        self._check_if_tensor(X)
        self._check_if_tensor(y)

        predictions = self._predict(X)
        assert predictions.mean.dim() == y.dim(), "Dimensions of the predictions are different from y."

        rmse = []

        if y.dim() > 1:
            for i in range(y.shape[1]):
                rmse.append(self._rmse(predictions.mean[:, i], y[:, i]).item())
        else:
            rmse.append(self._rmse(predictions.mean, y))
        return rmse

    def evaluate_correlation(self, X, y) -> list:
        self._check_if_tensor(X)
        self._check_if_tensor(y)

        predictions = self._predict(X)
        assert predictions.mean.dim() == y.dim(), "Dimensions of the predictions are different from y."

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



def get_transformer(transformer_spec: dict) -> object:
    """
    Get the transformer class and options

    Parameters
    -----------
    transformer_spec : dict
                       Dictionary containing the transformer specifications

    Returns
    --------
    selected_transformer_class : object
                                 The selected transformer class
    """
    selected_transformer = transformer_spec['transformerClass']
    selected_transformer_class = getattr(transformer_module, selected_transformer)
    if transformer_spec['transformerOptions']:
        selected_transformer_class = selected_transformer_class(**transformer_spec['transformerOptions'])
    else:
        selected_transformer_class = selected_transformer_class()
    return selected_transformer_class


def get_likelihood(training_spec: dict, num_tasks: int) -> object:
    """
    Get the likelihood class and options

    Parameters
    -----------
    training_spec : dict
                    Dictionary containing the training specifications
    num_tasks : int
                Number of tasks in the dataset

    Returns
    --------
    selected_likelihood_class : object
                                The selected likelihood class
    """
    selected_likelihood = training_spec['likelihoodClass']
    selected_likelihood_class = getattr(likelihood_module, selected_likelihood)

    if num_tasks > 1:
        selected_likelihood_class = selected_likelihood_class(num_tasks=num_tasks)
    return selected_likelihood_class


def get_model(training_spec: dict) -> object:
    """
    Get the model class and options

    Parameters
    -----------
    training_spec : dict
                    Dictionary containing the training specifications

    Returns
    --------
    selected_model_class : object
                           The selected model class
    """
    selected_model = training_spec['modelClass']
    selected_model_class = getattr(model_module, selected_model)
    return selected_model_class


def input_output_split(data: pd.DataFrame, data_spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into input and output

    Parameters
    -----------
    data : pd.DataFrame
           The data to be split
    data_spec : dict
                Dictionary containing the data specifications

    Returns
    --------
    X : pd.DataFrame
        The input data
    y : pd.DataFrame
        The output data
    """
    n_inputs = data_spec['nInputs']
    n_outputs = data_spec['output']['nOutputs']

    if (n_inputs + n_outputs) > len(data.columns):
        raise RuntimeError("The number of inputs and outputs specified exceeds the number of columns in the data file.")

    if n_inputs + n_outputs < len(data.columns):
        if data_spec['output']['outputIndex']:
            output_idx = data_spec['output']['outputIndex']
        else:
            raise KeyError("The output index is not specified.")
    else:
        output_idx = [i + 1 for i in range(n_outputs)]

    if isinstance(output_idx, int):
        output_idx = [output_idx]

    input_columns = data.columns[0:n_inputs]
    output_columns = []

    for idx in output_idx:
        column_i = data.columns[n_inputs + (idx - 1)]
        output_columns.append(column_i)

    X = data[input_columns]
    y = data[[str(i) for i in output_columns]]
    return X, y


def transform_data(X: pd.DataFrame, transformer: object, columns:list=None) -> Union[pd.DataFrame,tuple[pd.DataFrame, object]]:
    """
    Transform the input data using the selected transformer

    Parameters
    -----------
    X : pd.DataFrame
        The input data
    transformer : object
                  The selected transformer

    Returns
    --------
    X : pd.DataFrame
        The transformed input data
    transformer : object
                  The fitted transformer
    """

    def is_fit_called(obj):
        return hasattr(obj, "n_features_in_")

    if columns is not None:
        columns_to_transform = X.columns[columns]
        X_transformed = X[columns_to_transform].copy()

        if is_fit_called(transformer):
            X_transformed = transformer.transform(X_transformed.values)
            X.loc[:, columns_to_transform] = X_transformed
            return X
        else:
            X_transformed = transformer.fit_transform(X_transformed.values)
            X.loc[:, columns_to_transform] = X_transformed
            return X, transformer
    else:
        if is_fit_called(transformer):
            X = pd.DataFrame(transformer.transform(X.values))
            return X
        else:
            X = pd.DataFrame(transformer.fit_transform(X.values))
            return X, transformer


