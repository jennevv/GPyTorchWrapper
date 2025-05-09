import argparse
import warnings
from pathlib import Path

import gpytorch
import numpy as np
import torch
import logging
from gpytorchwrapper.src.config.config_classes import create_config
from gpytorchwrapper.src.models.model_load import load_model

warnings.filterwarnings("ignore")  # Ignore warnings from the torch.jit.trace function

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def parse_args():
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


def main():
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


class MeanVarModelWrapper(torch.nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.variance

def trace_model(model, len_training_data, transformer, num_inputs):
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


def create_test_data(num_inputs):
    return np.random.rand(500, num_inputs)


def test_traced_model(model, traced_model, input_transformer, num_inputs):
    test_x = create_test_data(num_inputs)

    if input_transformer is not None:
        test_x = input_transformer.transform(test_x)

    test_x = torch.tensor(test_x)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        traced_mean, traced_var = traced_model(test_x)
        pred = model(test_x)

    assert torch.allclose(traced_mean, pred.mean, atol=1e-14)
    assert torch.allclose(traced_var, pred.variance, atol=1e-14)


if __name__ == "__main__":
    main()
