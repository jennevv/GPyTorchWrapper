import argparse
import logging
import pathlib
from dataclasses import asdict
from pathlib import Path
from sys import platform
import gpytorch
import torch

from gpytorchwrapper.src.config.config_reader import read_yaml
from gpytorchwrapper.src.data.data_reader import DataReader
from gpytorchwrapper.src.data.data_splitter import input_output_split, split_data
from gpytorchwrapper.src.data.data_transform import transform
from gpytorchwrapper.src.models.model_train import train_model
from gpytorchwrapper.src.models.model_evaluate import evaluate_model
from gpytorchwrapper.src.models.model_save import save_model
from gpytorchwrapper.src.utils import metadata_dict, dataframe_to_tensor

__author__ = "Jenne Van Veerdeghem"
__version__ = "0.0.1"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.float64)

gpytorch.settings.fast_computations(
    covar_root_decomposition=True, # False = Cholesky decomp., True = Lanczos
    log_prob=True, # False = Cholesky decomp., True = modified conjugate gradients algorithm
    solves=True # False = Cholesky decomp., True = preconditioned conjugate gradients
)


# Needed for training on HPC cluster
if platform == "linux":
    pathlib.WindowsPath = pathlib.PosixPath


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GPR Training",
        description="Train a Gaussian Process Regressor using GPytorch.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="File containing the data",
    )
    parser.add_argument(
        "-f",
        "--file-type",
        type=str,
        required=True,
        help="Format of the data file. Can be either csv or pickle.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="The config file containing the script options.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Name of the output file containing the model and its metadata.",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "-t",
        "--test-set",
        type=str,
        required=False,
        help="File containing the test data. Not usable when cross-validation is selected in the config file.",
    )

    args = parser.parse_args()
    args.input, args.config, args.directory = map(
        Path, [args.input, args.config, args.directory]
    )
    if args.test_set:
        args.test_set = Path(args.test_set)

    # Allow for the creation of the output directory if it does not exist
    args.directory.mkdir(parents=True, exist_ok=True)

    return args


def main():
    args = parse_args()

    reader = DataReader()
    data = reader.read_data(file=args.input, file_type=args.file_type)
    logger.info(f"Data loaded from {args.input}.")

    # Read the input files and split the specifications
    config = read_yaml(args.config)
    data_conf = config.data_conf
    transform_conf = config.transform_conf
    training_conf = config.training_conf
    testing_conf = config.testing_conf
    logger.info(f"Input file {args.config} read.")

    # Data processing
    x, y = input_output_split(data, data_conf)
    if not args.test_set:
        train_x, test_x, train_y, test_y = split_data(
            x, y, data_conf, transform_conf, training_conf, testing_conf, args.directory
        )
        train_x, test_x, train_y, test_y, input_transformer, output_transformer = transform(
            train_x, train_y, test_x, test_y, transform_conf
        )
    else:
        train_x, _, train_y, _ = split_data(
            x, y, data_conf, transform_conf, training_conf, testing_conf, args.directory
        )
        train_x, _, train_y, _, input_transformer, output_transformer = transform(
            train_x, train_y, None, None, transform_conf
        )

        test_data = reader.read_data(file=args.test_set, file_type=args.file_type)
        logger.info(f"Test data loaded from {args.test_set}.")

        x, y = input_output_split(test_data, data_conf)
        test_x, _, test_y, _ = split_data(
            x, y, data_conf, transform_conf, training_conf, testing_conf, args.directory
        )
        test_x, _, test_y, _, input_transformer, output_transformer = transform(
            test_x, test_y, None, None, transform_conf
        )

    train_x, train_y = map(dataframe_to_tensor, [train_x, train_y])
    if test_x is not None:
        test_x, test_y = map(dataframe_to_tensor, [test_x, test_y])

    # Model training
    model, likelihood, _ = train_model(
        train_x, train_y, training_conf, data_conf.num_outputs
    )

    # Evaluate the model on the training and test sets
    train_rmse, test_rmse, test_corr = evaluate_model(
        model, likelihood, output_transformer, train_x, train_y, test_x, test_y
    )

    # Save metadata to dictionaries
    training_metadata = metadata_dict(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        input_transformer=input_transformer,
        output_transformer=output_transformer,
    )
    metrics_metadata = metadata_dict(
        train_rmse=train_rmse, test_rmse=test_rmse, test_corr=test_corr
    )

    # Save the model to a .pth file
    save_model(
        model.state_dict(),
        asdict(config),
        training_metadata,
        metrics_metadata,
        args.output,
        args.directory,
    )


if __name__ == "__main__":
    main()
