import argparse
import pathlib
import sys
from pathlib import Path
from sys import platform

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from botorch import fit_gpytorch_mll

from src.gp_utils import (
    input_output_split,
    transform_data,
    get_model,
    get_likelihood,
    get_transformer,
    read_yaml,
    metadata_dict,
    DataReader,
    ModelEvaluator,
    git_hash,
    dataframe_to_tensor
)

__author__ = 'Jenne Van Veerdeghem'
__version__ = '0.0.1'


# Needed for training on HPC cluster
if platform == "linux":
    pathlib.WindowsPath = pathlib.PosixPath


def print_to_stderr(*a):
    """Useful when using Slurm to log outputs to a file."""
    print(*a, file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GPR Training for H2O-Kr",
        description="Train a Gaussian Process Regressor for the H2O-Kr system using GPytorch."
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
        "--file_type",
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
        help="Name of the output file containing the model and its metadata."
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Output directory",
    )

    args = parser.parse_args()

    args.input, args.config, args.directory = map(Path, [args.input, args.config, args.directory])

    # Allow for the creation of the output directory if it does not exist
    args.directory.mkdir(parents=True, exist_ok=True)

    if args.file_type not in ['csv', 'pickle']:
        raise NotImplementedError(
            f'File type {args.file_type} is not implemented. Please specify either csv or pickle with the --file_type flag.')

    return args


def train_test_split(x, y, data_conf, transformer_conf, training_conf, testing_conf, directory):
    """
    Split the data into training and test sets.

    Parameters
    -----------
    x : pd.DataFrame
        The input data
    y : pd.DataFrame
        The output data
    data_conf : dict
                Dictionary containing the data specifications
    transformer_conf : dict
                       Dictionary containing the transformer specifications
    training_conf : dict
                    Dictionary containing the training specifications
    testing_conf : dict
                     Dictionary containing the testing specifications
    directory : pathlib.Path
                The output directory
    """
    if testing_conf['test']:
        if testing_conf['kFold']:
            k_fold_split(x, y, training_conf=training_conf, transformer_conf=transformer_conf, data_conf=data_conf,
                         directory=directory, split_size=testing_conf['testSize'])
            return x, y, None, None
        elif testing_conf['stratSplit']:
            train_x, train_y, test_x, test_y = stratified_shuffle_split(x, y, n_bins=5,
                                                                        test_size=testing_conf['testSize'])
            return train_x, train_y, test_x, test_y
    else:
        return x, y, None, None


def k_fold_split(x, y, training_conf, transformer_conf, data_conf, directory, split_size=0.2):
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
    transformer_conf : dict
                        Dictionary containing the transformer specifications
    data_conf : dict
                Dictionary containing the data specifications
    directory : pathlib.Path
                The output directory
    split_size : float
                 The size of the test set
    """
    print_to_stderr("\n Starting k-fold split testing.")

    n_splits = int(1 / split_size)

    kf = KFold(n_splits=n_splits, shuffle=True)

    train_rmse_arr = np.zeros(n_splits)
    test_rmse_arr = np.zeros(n_splits)
    test_corr_arr = np.zeros(n_splits)

    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        print_to_stderr(f"\nFold {fold}")
        train_x = x.iloc[train_index]
        train_y = y.iloc[train_index]
        test_x = x.iloc[test_index]
        test_y = y.iloc[test_index]

        train_x, train_y, test_x, test_y, input_transformer, output_transformer = transform(train_x, train_y, test_x,
                                                                                            test_y, transformer_conf)
        train_x, train_y, test_x, test_y = map(dataframe_to_tensor, [train_x, train_y, test_x, test_y])

        model, likelihood = train_model(train_x, train_y, training_conf=training_conf,
                                        num_tasks=data_conf['output']['nOutputs'], input_transformer=input_transformer)

        # Evaluate the model on the training and test sets
        train_rmse, test_rmse, test_corr = evaluate_model(model, likelihood, train_x, train_y, test_x, test_y)

        train_rmse_arr[fold] = train_rmse[0]
        test_rmse_arr[fold] = test_rmse[0]
        test_corr_arr[fold] = test_corr[0]

    kfold_data = np.stack([train_rmse_arr, test_rmse_arr, test_corr_arr], axis=1)

    kfold_df = pd.DataFrame(kfold_data, columns=['TRAIN_RMSE', 'TEST_RMSE', 'TEST_CORR'])
    kfold_df.to_csv(directory / Path('kfold_data.csv'), index=False)

    kfold_results = calculate_kfold_results(kfold_data)

    write_kfold_results(kfold_results, directory)


def calculate_kfold_results(kfold_data):
    avg_train_rmse = np.average(kfold_data[:, 0])
    avg_test_rmse = np.average(kfold_data[:, 1])
    avg_test_corr = np.average(kfold_data[:, 2])

    kfold_results = {
        'avg_train_rmse': avg_train_rmse,
        'avg_test_rmse': avg_test_rmse,
        'avg_test_corr': avg_test_corr
    }

    return kfold_results


def write_kfold_results(kfold_results, out_dir):
    file_path = out_dir / Path("kfold_results.txt")

    with file_path.open(mode='w') as f:
        f.write(f"Average Train RMSE: {kfold_results['avg_train_rmse']}")
        f.write(f"Average Test RMSE: {kfold_results['avg_test_rmse']}")
        f.write(f"Average Test R2: {kfold_results['avg_test_corr']}")


def stratified_shuffle_split(x, y: pd.DataFrame, n_bins=5, test_size=0.2):
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
    print_to_stderr('\nPerforming stratified shuffle split.')

    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y)

    # Join the input and output data back together
    output_columns = y.columns
    data = x.join(y)

    # Add n bins along the selected column
    data['CAT'] = pd.cut(x[x.columns[0]], bins=n_bins)  # Create category column

    # Split data into test and training set using categories
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=26)

    # Define the stratified training and test sets
    for train_index, test_index in split.split(data, data["CAT"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

        assert len(strat_train_set) + len(strat_test_set) == len(data)

    # Remove the category columns in the training and test sets
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("CAT", axis=1, inplace=True)

    # Transform the input data
    train_x = strat_train_set['R THETA PHI'.split()]
    test_x = strat_test_set['R THETA PHI'.split()]

    train_y = strat_train_set[output_columns]
    test_y = strat_test_set[output_columns]

    return train_x, train_y, test_x, test_y


def transform(train_x, train_y, test_x, test_y, transformer_conf):
    print_to_stderr('\nTransforming data.')

    # Transform the input
    if transformer_conf['transformInput']['transformData']:
        input_transformer = get_transformer(transformer_conf['transformInput'])

        train_x, input_transformer = transform_data(train_x, input_transformer, transformer_conf['transformInput']['columns'])

        if test_x is not None:
            test_x = transform_data(test_x, input_transformer, transformer_conf['transformInput']['columns'])
        print_to_stderr(f'Transforming input values using {input_transformer}.')

    else:
        input_transformer = None
        print_to_stderr('Using raw input values.')

    # Transform the output
    if transformer_conf['transformOutput']['transformData']:
        output_transformer = get_transformer(transformer_conf['transformOutput'])

        train_y, transformer = transform_data(train_y, output_transformer, columns=None)

        if test_y is not None:
            test_y = transform_data(test_y, output_transformer, columns=None)

        print_to_stderr(f'Transforming output values using {output_transformer}.')
    else:
        output_transformer = None
        print_to_stderr('Using raw output values.')

    return train_x, train_y, test_x, test_y, input_transformer, output_transformer


def define_optimizer(model, learning_rate, mean_conf, num_tasks):
    """
    Define the optimizer for the model

    Parameters
    -----------
    model : object
            The model to be optimized
    learning_rate : int
                    The learning rate for the optimizer
    mean_conf : dict
                         Dictionary containing the mean constant specifications
    num_tasks : int
                The number of tasks the model is expected to train on

    Returns
    --------
    optimizer : object
                The optimizer for the model
    """

    # Set the mean constant
    if mean_conf['constant_value'] is not None:
        print_to_stderr(f'Assigning constant value {mean_conf["constant_value"]} to the mean.')
        assert str(model.mean_module) == 'ConstantMean()', \
            "The mean module should be a constant mean if trying to assign a constant value."

        if num_tasks > 1:
            model.mean_module.constant = torch.tensor(mean_conf['constant_value'], dtype=torch.float64)
        else:
            model.mean_module.constant = mean_conf['constant_value']

        if not mean_conf['optimize']:
            model.mean_module.raw_constant.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return optimizer


def loss_figure(loss, iteration):
    """
    Plot the change of loss during training

    Parameters
    -----------
    loss : list
           List containing the loss values

    iteration : list
                List containing the iteration values

    Returns
    --------
    None
    """
    plt.scatter(iteration, loss)
    plt.title('Change of loss during training')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss.png", dpi=300)


def set_noiseless(likelihood, num_tasks):
    # Set the likelihood noise to a small value to simulate noiseless data.
    # Not zero for performance reasons.

    if num_tasks > 1:
        likelihood.task_noises = torch.tensor([1e-8] * num_tasks)
        likelihood.raw_task_noises.detach_()
    else:
        likelihood.noise = 1e-8
        likelihood.raw_noise.detach_()
    return likelihood


def print_model_parameters(model):
    for param_name, param in model.named_parameters():
        if param.size() == 1:
            print_to_stderr(f'Parameter name: {param_name:42} value = {param.item()}')
        else:
            print_to_stderr(f'Parameter name: {param_name:42} value = {param.tolist()}')


def training_loop(train_x, train_y, model, mll, optimizer, learning_iterations):
    loss_hash = {'loss': [], 'iteration': []}

    for iteration in range(learning_iterations):
        if (iteration + 1) % 10 == 0:
            print_to_stderr(f'Iteration {iteration + 1}/{learning_iterations}')
            loss_figure(loss_hash['loss'], loss_hash['iteration'])

        optimizer.zero_grad()

        output = model(train_x)

        loss = -mll(output, train_y)

        loss_hash['loss'].append(loss.item())

        loss_hash['iteration'].append(iteration)
        loss.backward()
        optimizer.step()

    # Plot the loss
    loss_figure(loss_hash['loss'], loss_hash['iteration'])


def train_model(train_x, train_y, training_conf, num_tasks, input_transformer):
    """
    Train the model using the training data

    Parameters
    -----------
    train_x : torch.Tensor
              The input training data
    train_y : torch.Tensor
              The output training data
    training_conf : dict
                    Dictionary containing the training specifications
    num_tasks : int
                The number of tasks the model is expected to train on
    input_transformer : object
                        The transformer object for the input data
    Returns
    --------
    model : object
            The trained model
    likelihood : object
                 The trained likelihood of the model
    """
    print_to_stderr('\nDefining the model specifications.')

    # Load the training specifications
    mean_spec = training_conf['meanSpec']
    learning_iterations = training_conf['learningIterations']
    learning_rate = training_conf['learningRate']
    botorch = training_conf['botorch']
    noiseless = training_conf['noiseless']

    # Define likelihood and model
    likelihood_class = get_likelihood(training_conf, num_tasks)
    model_class = get_model(training_conf)

    likelihood = likelihood_class(noise_constraint=gpytorch.constraints.GreaterThan(1e-5))  # Like sklearn WhiteKernel
    if noiseless:
        likelihood = set_noiseless(likelihood=likelihood, num_tasks=num_tasks)

    if mean_spec['transformer']:  # If the mean requires the transformer function
        model = model_class(train_x, train_y, likelihood, input_transformer)
    else:
        model = model_class(train_x, train_y, likelihood)

    print_to_stderr('\nParameters before training:')
    print_model_parameters(model)

    # Training in double precision
    model.double()
    likelihood.double()

    # Define the marginal likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print_to_stderr("\nStart training the model.")
    if botorch:
        fit_gpytorch_mll(mll, max_attempts=100)
    else:
        # Optimize model hyperparameters
        optimizer = define_optimizer(model, learning_rate, mean_spec, num_tasks)
        training_loop(train_x, train_y, model, mll, optimizer, learning_iterations)

    print_to_stderr('\nParameters after training:')
    print_model_parameters(model)

    return model, likelihood


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
    print_to_stderr('Evaluating the model.')

    evaluator = ModelEvaluator(model, likelihood)

    train_rmse = evaluator.evaluate_rmse(train_x, train_y)
    print_to_stderr(f'\ntrain_rmse: {train_rmse}')
    if test_x is not None:
        test_rmse = evaluator.evaluate_rmse(test_x, test_y)
        test_corr = evaluator.evaluate_correlation(test_x, test_y)
        print_to_stderr(f'test_rmse" {test_rmse}\ntest_corr: {test_corr}')
        return train_rmse, test_rmse, test_corr

    else:
        return train_rmse, None, None


def save_model(model_state, model_metadata, training_data, metrics, output, directory):
    """
    Save the model to a .pth file

    Parameters
    -----------
    model_state : object
                  The state of the model
    model_metadata : dict
                     Dictionary containing the metadata of the model
    training_data : dict
                    Dictionary containing the training data and the transformer
    metrics : dict
              Dictionary containing the metrics on the training and test sets
    output : str
           The name of the output file
    directory : pathlib.Path
              The output directory

    Returns
    --------
    None
    """
    if isinstance(directory, str):
        directory = Path(directory)

    savestate = {
        'model_state': model_state,
        'model_metadata': model_metadata,
        'training_data': training_data,
        'metrics': metrics}

    torch.save(savestate, directory / f'{output}_{git_hash()}.pth')


def main():
    args = parse_args()

    reader = DataReader()
    data = reader.read_data(file=args.input, file_type=args.file_type)
    print_to_stderr(f'Data loaded from {args.input}.')

    # Read the input files and split the specifications
    config = read_yaml(args.config)
    data_conf = config['dataSpec']
    transformer_conf = config['transformerSpec']
    training_conf = config['trainingSpec']
    testing_conf = config['testingSpec']
    print_to_stderr(f'Input file {args.config} read.')

    # Data processing
    x, y = input_output_split(data, data_conf)
    train_x, train_y, test_x, test_y = train_test_split(x, y, data_conf, transformer_conf, training_conf, testing_conf,
                                                        args.directory)
    train_x, train_y, test_x, test_y, \
        input_transformer, output_transformer = transform(train_x, train_y, test_x, test_y, transformer_conf)

    train_x, train_y = map(dataframe_to_tensor, [train_x, train_y])
    if test_x is not None:
        test_x, test_y = map(dataframe_to_tensor, [test_x, test_y])

    # Train the model
    model, likelihood = train_model(train_x, train_y, training_conf=training_conf,
                                    num_tasks=data_conf['output']['nOutputs'],
                                    input_transformer=input_transformer)

    # Evaluate the model on the training and test sets
    train_rmse, test_rmse, test_corr = evaluate_model(model, likelihood, train_x, train_y, test_x, test_y)

    # Save metadata to dictionaries
    model_metadata = metadata_dict(model=model, training_spec=training_conf)
    training_metadata = metadata_dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                                      input_transformer=input_transformer, output_transformer=output_transformer)
    metrics_metadata = metadata_dict(train_rmse=train_rmse, test_rmse=test_rmse, test_corr=test_corr)

    # Save the model to a .pth file
    save_model(model.state_dict(), model_metadata, training_metadata, metrics_metadata, args.output, args.directory)


if __name__ == "__main__":
    main()
