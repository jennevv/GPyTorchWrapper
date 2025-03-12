import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split

from gpytorchwrapper.src.config.config_classes import (
    DataConf,
    TestingConf,
    TrainingConf,
    TransformConf,
)
from gpytorchwrapper.src.data.data_transform import transform
from gpytorchwrapper.src.models.model_evaluate import evaluate_model
from gpytorchwrapper.src.models.model_train import train_model, print_model_parameters
from gpytorchwrapper.src.utils import dataframe_to_tensor

logger = logging.getLogger(__name__)


def input_output_split(
    data: pd.DataFrame, data_conf: DataConf
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into input and output

    Parameters
    -----------
    data : pd.DataFrame
        The data to be split
    data_conf : DataConf
            dataclass containing the data specifications

    Returns
    --------
    x : pd.DataFrame
        The input data
    y : pd.DataFrame
        The output data
    """

    n_inputs = data_conf.num_inputs
    n_outputs = data_conf.num_outputs

    if (n_inputs + n_outputs) > len(data.columns):
        raise RuntimeError(
            "The number of inputs and outputs specified exceeds the number of columns in the data file."
        )

    if n_inputs + n_outputs < len(data.columns):
        if isinstance(data_conf.output_index, int) and n_outputs == 1:
            output_idx = [data_conf.output_index]
        elif isinstance(data_conf.output_index, list):
            output_idx = data_conf.output_index
        else:
            raise KeyError(
                f"The output index is not specified or not properly specified. Expected int or list[int], got {type(data_conf.output_index)}"
            )
    else:
        output_idx = [i for i in range(n_outputs)]

    input_columns = data.columns[0:n_inputs]
    output_columns = []

    for idx in output_idx:
        column_i = data.columns[n_inputs + idx]
        output_columns.append(column_i)

    x = data[input_columns]
    y = data[output_columns]
    return x, y


def k_fold_split(
    x: pd.DataFrame,
    y: pd.DataFrame,
    training_conf: TrainingConf,
    transform_conf: TransformConf,
    data_conf: DataConf,
    directory: Path,
    split_size: float = 0.2,
) -> None:
    """
    Split the data using k-fold cross-validation

    Parameters
    -----------
    x : pd.DataFrame
        The input data
    y : pd.DataFrame
        The output data
    training_conf : dict
                    Dictionary containing the training specifications
    transform_conf : dict
                    Dictionary containing the transformer specifications
    data_conf : dict
                Dictionary containing the data specifications
    directory : pathlib.Path
                The output directory
    split_size : float
                The size of the test set

    Returns
    --------
    None
    """
    logger.info("Starting k-fold split testing.")

    n_splits = int(1 / split_size)

    kf = KFold(n_splits=n_splits, shuffle=True)

    train_rmse_arr = np.zeros(n_splits)
    test_rmse_arr = np.zeros(n_splits)
    test_corr_arr = np.zeros(n_splits)
    fold_parameters = []

    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        logger.info(f"Fold {fold}")
        train_x = x.iloc[train_index]
        train_y = y.iloc[train_index]
        test_x = x.iloc[test_index]
        test_y = y.iloc[test_index]

        train_x, test_x, train_y, test_y, input_transformer, output_transformer = (
            transform(train_x, train_y, test_x, test_y, transform_conf)
        )
        train_x, test_x, train_y, test_y = map(
            dataframe_to_tensor, [train_x, test_x, train_y, test_y]
        )

        model, likelihood, fold_params = train_model(
            train_x,
            train_y,
            training_conf,
            test_x,
            test_y
        )

        fold_parameters.append(fold_params)

        # Evaluate the model on the training and test sets
        train_rmse, test_rmse, test_corr = evaluate_model(
            model, likelihood, output_transformer, train_x, train_y, test_x, test_y
        )

        train_rmse_arr[fold] = train_rmse[0]
        test_rmse_arr[fold] = test_rmse[0]
        test_corr_arr[fold] = test_corr[0]

    best_fold = np.argmin(test_rmse_arr)

    logger.info(f"Best fold: Fold {best_fold}")
    logger.info(
        f"Best model parameters: \n{print_model_parameters(fold_parameters[best_fold].keys(), fold_parameters[best_fold].values())}\n"
    )

    kfold_data = np.stack([train_rmse_arr, test_rmse_arr, test_corr_arr], axis=1)

    kfold_df = pd.DataFrame(
        kfold_data, columns=["TRAIN_RMSE", "TEST_RMSE", "TEST_CORR"]
    )
    kfold_df.to_csv(directory / Path("kfold_data.csv"), index=False)

    kfold_results = calculate_kfold_results(kfold_data)

    write_kfold_results(kfold_results, directory)


def calculate_kfold_results(kfold_data):
    avg_train_rmse = np.average(kfold_data[:, 0])
    avg_test_rmse = np.average(kfold_data[:, 1])
    avg_test_corr = np.average(kfold_data[:, 2])

    kfold_results = {
        "avg_train_rmse": avg_train_rmse,
        "avg_test_rmse": avg_test_rmse,
        "avg_test_corr": avg_test_corr,
    }

    return kfold_results


def write_kfold_results(kfold_results, out_dir):
    file_path = out_dir / Path("kfold_results.txt")

    with file_path.open(mode="w") as f:
        f.write(f"Average Train RMSE: {kfold_results['avg_train_rmse']}")
        f.write(f"Average Test RMSE: {kfold_results['avg_test_rmse']}")
        f.write(f"Average Test R2: {kfold_results['avg_test_corr']}")


def stratified_shuffle_split(
    x: pd.DataFrame, y: pd.DataFrame, n_bins: int | None = 5, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and test sets using stratified shuffle split

    Parameters
    -----------
    x : pd.DataFrame
        The input data
    y : pd.DataFrame
        The output data
    n_bins : int
             The number of bins to split the data into
    test_size : float
                The size of the test set

    Returns
    --------
    train_x : pd.DataFrame
              The input training set
    train_y : pd.DataFrame
              The output training set
    test_x : pd.DataFrame
             The input test set
    test_y : pd.DataFrame
             The output test set
    """
    logging.info("Performing stratified shuffle split.")

    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y)

    # Join the input and output data back together
    output_columns = y.columns
    data = x.join(y)

    # Add n bins along the selected column
    data["CAT"] = pd.cut(data[y.columns[0]], bins=n_bins)  # Create category column

    # Split data into test and training set using categories
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=26)

    # Define the stratified training and test sets
    for train_index, test_index in split.split(data, data["CAT"]):
        strat_train_set = data.iloc[train_index]
        strat_test_set = data.iloc[test_index]

        assert len(strat_train_set) + len(strat_test_set) == len(data)

    # Remove the category columns in the training and test sets
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("CAT", axis=1, inplace=True)

    # Transform the input data
    train_x: pd.DataFrame = strat_train_set[x.columns]
    test_x: pd.DataFrame = strat_test_set[x.columns]

    train_y: pd.DataFrame = strat_train_set[output_columns]
    test_y: pd.DataFrame = strat_test_set[output_columns]

    return train_x, test_x, train_y, test_y


def exit_program():
    sys.exit(0)


def split_data(
    x: pd.DataFrame,
    y: pd.DataFrame,
    data_conf: DataConf,
    transform_conf: TransformConf,
    training_conf: TrainingConf,
    testing_conf: TestingConf,
    directory: Path,
) -> (
    None
    | tuple[DataFrame, DataFrame, DataFrame, DataFrame]
    | tuple[DataFrame, None, DataFrame, None]
):
    """
    Split the data into training and test sets.
    If neither kFold nor stratified shuffle split is selected, perform a random split.

    Parameters
    -----------
    x : pd.DataFrame
        The input data
    y : pd.DataFrame
        The output data
    data_conf : DataConf
                Dictionary containing the data specifications
    transform_conf : TransformConf
                       Dictionary containing the transformer specifications
    training_conf : TrainingConf
                    Dictionary containing the training specifications
    testing_conf : TestingConf
                     Dictionary containing the testing specifications
    directory : pathlib.Path
                The output directory
    """
    if testing_conf.test:
        if testing_conf.kfold:
            k_fold_split(
                x,
                y,
                training_conf=training_conf,
                transform_conf=transform_conf,
                data_conf=data_conf,
                directory=directory,
                split_size=testing_conf.test_size,
            )
            exit_program()
        elif testing_conf.strat_shuffle_split:
            train_x, test_x, train_y, test_y = stratified_shuffle_split(
                x, y, n_bins=testing_conf.kfold_bins, test_size=testing_conf.test_size
            )
            return train_x, test_x, train_y, test_y
        else:
            train_x, test_x, train_y, test_y = train_test_split(
                x, y, test_size=testing_conf.test_size
            )
            return train_x, test_x, train_y, test_y
    else:
        return x, None, y, None
