import argparse
import warnings
from pathlib import Path

import gpytorch
import numpy as np
import torch

from src.gp_utils import get_model, read_yaml, get_likelihood

warnings.filterwarnings("ignore") # Ignore warnings from the torch.jit.trace function


def parse_args():
    parser = argparse.ArgumentParser(
        prog="pt2ts",
        description="Convert a PyTorch model to a TorchScript model"
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
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config file containing the script options which was used to train the model.",
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=False,
        default="./",
        help="Directory where the TorchScript model is saved."
    )

    args = parser.parse_args()

    args.input, args.config, args.directory = map(Path, [args.input, args.config, args.directory])
    
    args.directory.mkdir(parents=True, exist_ok=True)
    
    return args



def main():
    args = parse_args()

    input_file = read_yaml(args.config)
    input_data = input_file['dataSpec']

    model_dump = torch.load(args.input)

    train_x, train_y = model_dump['training_data']['train_x'], model_dump['training_data']['train_y']
    num_inputs = input_data['nInputs']
    num_tasks = input_data['output']['nOutputs']

    if input_file['transformerSpec']['transformInput']['transformData']:
        input_transformer = model_dump['training_data']['input_transformer']
    else:
        input_transformer = None

    model_path, likelihood = load_model(input_file, model_dump, train_x, train_y, num_tasks)

    traced_model = trace_model(model_path, len(train_x), input_transformer, num_inputs)

    test_traced_model(model_path, traced_model, input_transformer, num_inputs)

    traced_model.save(f"{args.directory / args.output}")


class MeanVarModelWrapper(torch.nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.variance


def load_model(input_file, model_dump, train_x, train_y, num_tasks):
    likelihood_class = get_likelihood(input_file['trainingSpec'], num_tasks)
    model_class = get_model(input_file['trainingSpec'])

    likelihood = likelihood_class()
    model = model_class(train_x, train_y, likelihood)

    model.double()
    likelihood.double()

    model.load_state_dict(model_dump['model_state'])

    return model, likelihood


def trace_model(model, len_training_data, transformer, num_inputs):
    test_x = create_test_data(num_inputs)

    if transformer is not None:
        test_x = transformer.transform(test_x)

    test_x = torch.tensor(test_x, dtype=torch.float64, requires_grad=True)

    with gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode(), gpytorch.settings.max_eager_kernel_size(
            len_training_data + len(test_x)):
        model.eval()
        pred = model(test_x)  # Do precomputation
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