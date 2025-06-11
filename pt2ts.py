import argparse
import warnings
from pathlib import Path

import gpytorch
import numpy as np
import torch
from numpy.typing import NDArray
import logging
from gpytorchwrapper.src.config.config_classes import create_config
from gpytorchwrapper.src.models.model_load import load_model

warnings.filterwarnings("ignore")  # Ignore warnings from the torch.jit.trace function

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments for PyTorch to TorchScript model conversion.

    Parses command-line arguments required for converting a trained PyTorch
    Gaussian Process model to TorchScript format for deployment.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with the following attributes:
        - input : pathlib.Path
            Path to the input PyTorch model (.pth file)
        - output : str
            Name of the output TorchScript model file (default: 'model.ts')
        - directory : pathlib.Path
            Directory path where TorchScript model will be saved (created if needed)

    Notes
    -----
    The function automatically creates the output directory if it doesn't exist.
    Input validation ensures the input file path is converted to a Path object.

    Examples
    --------
    >>> args = parse_args()
    >>> print(args.input)
    PosixPath('/path/to/model.pth')
    >>> print(args.output)
    'model.ts'
    """

    parser = argparse.ArgumentParser(
        prog="pt2ts", description="Convert a PyTorch model to a TorchScript model"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the model .pth file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default="model.ts",
        help="Name of the output TorchScript model",
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=False,
        default="./",
        help="Directory where the TorchScript model is saved.",
    )

    args = parser.parse_args()

    args.input, args.directory = map(Path, [args.input, args.directory])

    args.directory.mkdir(parents=True, exist_ok=True)

    return args


class MeanVarModelWrapper(torch.nn.Module):
    """
    Wrapper class for GPyTorch models to extract mean and variance.

    Wraps a GPyTorch Gaussian Process model to provide a simplified interface
    that returns both mean and variance predictions, making it suitable for
    TorchScript tracing.

    Parameters
    ----------
    gp : gpytorch.models.GP
        The GPyTorch Gaussian Process model to wrap

    Attributes
    ----------
    gp : gpytorch.models.GP
        The wrapped Gaussian Process model

    Notes
    -----
    This wrapper is necessary because GPyTorch models return distribution
    objects that are not directly compatible with TorchScript tracing.
    The wrapper extracts the mean and variance components which are
    tensor objects suitable for tracing.

    Examples
    --------
    >>> model = SomeGPyTorchModel()
    >>> wrapped_model = MeanVarModelWrapper(model)
    >>> mean, var = wrapped_model(test_x)
    """

    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.variance


def trace_model(model, len_training_data, transformer, num_inputs):
    """
    Trace a GPyTorch model for TorchScript conversion.

    Creates a TorchScript-compatible traced version of the GPyTorch model
    using torch.jit.trace with appropriate GPyTorch settings for optimal
    performance and compatibility.

    Parameters
    ----------
    model : gpytorch.models.GP
        The trained GPyTorch Gaussian Process model to trace
    len_training_data : int
        Number of training data points, used for kernel size optimization
    transformer : sklearn.preprocessing transformer or None
        Input data transformer, or None if no transformation is applied
    num_inputs : int
        Number of input features/dimensions

    Returns
    -------
    torch.jit.ScriptModule
        Traced TorchScript model that can be saved and deployed

    Notes
    -----
    The tracing process uses several GPyTorch-specific performance enhancing settings:
    - fast_pred_var(): Enables LOVE method for efficient predictive variance
    - fast_pred_samples(): Enables LOVE method for predictive samples
    - trace_mode(): Disables GPyTorch features incompatible with tracing
    - max_eager_kernel_size(): Disables lazy evaluation for better tracing

    Test data is generated randomly and transformed if a transformer is provided.
    The model is set to evaluation mode before tracing.

    Examples
    --------
    >>> traced_model = trace_model(model, 1000, transformer, 5)
    >>> traced_model.save('model.ts')

    See Also
    --------
    create_test_data : Generate random test data for tracing
    MeanVarModelWrapper : Wrapper class for GPyTorch models
    """

    test_x = create_test_data(num_inputs)

    if transformer is not None:
        test_x = transformer.transform(test_x)

    test_x = torch.tensor(test_x, dtype=torch.float64, requires_grad=True)

    with (
        gpytorch.settings.fast_pred_var(),  # LOVE method for predictive variance
        gpytorch.settings.fast_pred_samples(),  # LOVE method for predictive samples
        gpytorch.settings.trace_mode(),  # Required for tracing, turns off some exclusive GPyTorch features
        gpytorch.settings.max_eager_kernel_size(
            len_training_data + len(test_x)
        ),  # Disables lazy evaluation
    ):
        model.eval()
        model(test_x)  # Do precomputation
        traced_model = torch.jit.trace(MeanVarModelWrapper(model), test_x)

    return traced_model


def create_test_data(num_inputs: int) -> NDArray:
    """
    Creates test data for checking the validity of the traced model.
    """
    return np.random.rand(500, num_inputs)


def test_traced_model(model, traced_model, input_transformer, num_inputs):
    """
    Verify integrity of traced model against original model.

    Compares outputs of the original GPyTorch model with the traced
    TorchScript model to ensure conversion accuracy.
    Uses randomly generated test data for comparison.

    Parameters
    ----------
    model : gpytorch.models.GP
        Original GPyTorch Gaussian Process model
    traced_model : torch.jit.ScriptModule
        Traced TorchScript version of the model
    input_transformer : sklearn.preprocessing transformer or None
        Input data transformer, or None if no transformation is applied
    num_inputs : int
        Number of input features/dimensions

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If traced model outputs don't match original model outputs
        within specified tolerance (1e-14)

    Notes
    -----
    The test compares both mean and variance predictions from both models.
    Uses torch.allclose() with absolute tolerance of 1e-14 for numerical
    precision validation. Both models are evaluated in no_grad() mode
    with fast_pred_var() setting for consistency.

    Examples
    --------
    >>> test_traced_model(original_model, traced_model, transformer, 5)
    # Passes silently if models match, raises AssertionError if not

    See Also
    --------
    create_test_data : Generate test data for comparison
    """
    test_x = create_test_data(num_inputs)

    if input_transformer is not None:
        test_x = input_transformer.transform(test_x)

    test_x = torch.tensor(test_x)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        traced_mean, traced_var = traced_model(test_x)
        pred = model(test_x)

    assert torch.allclose(traced_mean, pred.mean, atol=1e-14)
    assert torch.allclose(traced_var, pred.variance, atol=1e-14)

def main():
    """
    Main conversion pipeline from PyTorch model to TorchScript.

    Executes the complete conversion workflow including model loading,
    tracing, integrity testing, and saving the TorchScript model.
    Handles both transformed and non-transformed input data scenarios.

    Returns
    -------
    None
        Function performs conversion and saves TorchScript model to disk

    Raises
    ------
    FileNotFoundError
        If input model file cannot be found
    torch.jit.TracingError
        If model tracing fails due to control flow issues
    AssertionError
        If traced model integrity test fails

    Notes
    -----
    The conversion process includes:
    1. Load PyTorch model and configuration from .pth file
    2. Extract training data and input transformers
    3. Trace the model using torch.jit.trace with test data
    4. Verify traced model produces identical outputs to original
    5. Save traced model in TorchScript format

    The function uses GPyTorch-specific settings for optimal tracing:
    - fast_pred_var() for efficient predictive variance
    - trace_mode() to disable incompatible GPyTorch features
    - max_eager_kernel_size() to disable lazy evaluation

    Examples
    --------
    Command line usage:
    >>> # python pt2ts.py -i model.pth -o converted_model.ts -d output/
    >>> main()

    See Also
    --------
    trace_model : Perform model tracing
    test_traced_model : Verify traced model integrity
    """
    args = parse_args()

    model_dump = torch.load(args.input)

    config = create_config(model_dump["config"])

    train_x, train_y = (
        model_dump["training_data"]["train_x"],
        model_dump["training_data"]["train_y"],
    )
    num_inputs = config.data_conf.num_inputs

    if config.transform_conf.transform_input.transform_data:
        input_transformer = model_dump["training_data"]["input_transformer"]
    else:
        input_transformer = None

    logger.info("Loading model definition.")
    model, likelihood = load_model(config, model_dump, train_x, train_y)

    logger.info("Start tracing model.")
    traced_model = trace_model(model, len(train_x), input_transformer, num_inputs)
    logger.info("Finished tracing model.")

    logger.info("Test integrity traced model.")
    test_traced_model(model, traced_model, input_transformer, num_inputs)
    logger.info("Model integrity is good.")

    logger.info(f"Saving traced model to {args.directory / args.output}.")
    traced_model.save(f"{args.directory / args.output}")

if __name__ == "__main__":
    main()
