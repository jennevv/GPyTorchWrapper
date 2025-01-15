import logging
from typing import Optional

import pandas as pd

from gpytorchwrapper.src.config.config_classes import TransformConf
from gpytorchwrapper.src.config.model_factory import get_transformer

logger = logging.getLogger(__name__)


def transform_data(x: pd.DataFrame, transformer: object, columns: list = None) -> pd.DataFrame | tuple[
    pd.DataFrame, object]:
    """
    Transform the input data using the selected transformer

    Parameters
    -----------
    x : pd.DataFrame
        The input data
    transformer : object
                The selected transformer
    columns : list, optional
            The columns on which the transformer has to operate

    Returns
    --------
    x : pd.DataFrame
        The transformed input data
    transformer : object
                The fitted transformer
    """

    # scikit-learn transformers have this attribute n_features_in_ when they are fitted
    # check to see if the transformer already has been fitted
    def is_fit_called(obj):
        return hasattr(obj, "n_features_in_")

    if not isinstance(x, pd.DataFrame) and not isinstance(x, pd.Series):
        raise NotImplementedError(f'{type(x)} is not a pandas DataFrame or Series.')

    if columns is not None:
        if all(isinstance(column, str) for column in columns):
            columns_to_transform = columns
        elif all(isinstance(column, int) for column in columns):
            columns_to_transform = x.columns[columns]
        else:
            raise NotImplementedError('All items in the columns list should be the same type (int or string).')

        X_transformed = x[columns_to_transform].copy()

        if is_fit_called(transformer):
            X_transformed = transformer.transform(X_transformed.values)
            x.loc[:, columns_to_transform] = X_transformed
            return x
        else:
            X_transformed = transformer.fit_transform(X_transformed.values)
            x.loc[:, columns_to_transform] = X_transformed
            return x, transformer

    else:
        if is_fit_called(transformer):
            x = pd.DataFrame(transformer.transform(x.values))
            return x
        else:
            x = pd.DataFrame(transformer.fit_transform(x.values))
            return x, transformer


def transform(train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame, test_y: pd.DataFrame,
              transform_conf: TransformConf) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[object], Optional[object]]:
    logging.info('Transforming data.')

    # Transform the input
    transform_input = transform_conf.transform_input
    transform_output = transform_conf.transform_output

    if transform_input.transform_data:
        input_transformer = get_transformer(transform_input)

        train_x, input_transformer = transform_data(train_x, input_transformer, transform_input.columns)

        if test_x is not None:
            test_x = transform_data(test_x, input_transformer, transform_input.columns)
        logging.info(f'Transforming input values using {input_transformer}.')

    else:
        input_transformer = None
        logging.info('Using raw input values.')

    # Transform the output
    if transform_output.transform_data:
        output_transformer = get_transformer(transform_output)

        train_y, transformer = transform_data(train_y, output_transformer, columns=None)

        if test_y is not None:
            test_y = transform_data(test_y, output_transformer, columns=None)

        logging.info(f'Transforming output values using {output_transformer}.')
    else:
        output_transformer = None
        logging.info('Using raw output values.')

    return train_x, test_x, train_y, test_y, input_transformer, output_transformer
