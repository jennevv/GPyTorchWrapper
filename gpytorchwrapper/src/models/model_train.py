import torch
import gpytorch
import matplotlib.pyplot as plt
import logging

from botorch import fit_gpytorch_mll

from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood, Likelihood, \
    FixedNoiseGaussianLikelihood
from gpytorch.mlls import MarginalLogLikelihood
from torch import Tensor

from gpytorchwrapper.src.config.config_classes import TrainingConf, OptimizerConf, LikelihoodConf
from gpytorchwrapper.src.config.model_factory import get_likelihood, get_model, get_optimizer

logger = logging.getLogger(__name__)


def define_optimizer(model: ExactGP, optimizer_conf: OptimizerConf) -> torch.optim.Optimizer:
    """
    Define the optimizer for the model

    Parameters
    -----------
    model : object
            The model to be optimized
    learning_rate : float
                The learning rate for the optimizer

    Returns
    --------
    optimizer : object
            The optimizer for the model
    """
    optimizer_class = get_optimizer(optimizer_conf)
    optimizer = optimizer_class(model.parameters(), **optimizer_conf.optimizer_options)
    return optimizer

def define_likelihood(likelihood_conf: LikelihoodConf, likelihood_class: Likelihood, train_x: Tensor):
    if likelihood_class is FixedNoiseGaussianLikelihood:
        if "noise" not in likelihood_conf.likelihood_options.keys():
            raise KeyError("The noise parameter is not specified in the likelihood options.")
        elif type(likelihood_conf.likelihood_options["noise"]) == str:
            likelihood_conf.likelihood_options["noise"] = eval(likelihood_conf.likelihood_options["noise"])
            likelihood_conf.likelihood_options["noise"] = torch.tensor([likelihood_conf.likelihood_options["noise"]] * train_x.shape[0], dtype=torch.float64)
        elif isinstance(likelihood_conf.likelihood_options["noise"], list):
            likelihood_conf.likelihood_options["noise"] = torch.tensor(likelihood_conf.likelihood_options["noise"], dtype=torch.float64)
        elif isinstance(likelihood_conf.likelihood_options["noise"], float):
            likelihood_conf.likelihood_options["noise"] = torch.tensor([likelihood_conf.likelihood_options["noise"]] * train_x.shape[0], dtype=torch.float64)
        elif isinstance(likelihood_conf.likelihood_options["noise"], Tensor):
            likelihood_conf.likelihood_options["noise"] = torch.tensor([likelihood_conf.likelihood_options["noise"][0].item()] * train_x.shape[0], dtype=torch.float64)

    if likelihood_conf.likelihood_options:
        likelihood = likelihood_class(**likelihood_conf.likelihood_options)
        print(likelihood.noise)
    else:
        likelihood = likelihood_class()

    return likelihood

def define_model(model_conf, model_class, train_x, train_y, likelihood):
    if model_conf.model_options:
        model = model_class(train_x, train_y, likelihood, **model_conf.model_options)
    else:
        model = model_class(train_x, train_y, likelihood)
    return model


def loss_figure(loss: list[float], iteration: list[int]) -> None:
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
    plt.title("Change of loss during training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss.png", dpi=300)


def training_loop(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    model: ExactGP,
    mll: MarginalLogLikelihood,
    optimizer: torch.optim.Optimizer,
    learning_iterations: int,
    debug: bool,
) -> None:
    """
    The training loop for the model

    Parameters
    -----------
    train_x : torch.Tensor
              The input training data
    train_y : torch.Tensor
            The output training data
    model : object
            The model to be trained
    mll : MarginalLogLikelihood
        The marginal likelihood of the model
    optimizer : object
            The optimizer for the model
    learning_iterations : int
                        The number of iterations to train the model
    debug : bool
            Whether to use the debug mode in GPyTorch. Turn off if unwanted debug exceptions interfere with training.

    Returns
    --------
    None
    """
    loss_hash = {"loss": [], "iteration": []}

    with gpytorch.settings.debug(debug):
        for iteration in range(learning_iterations):
            if (iteration + 1) % 10 == 0:
                logger.info(f"Iteration {iteration + 1}/{learning_iterations}")
                loss_figure(loss_hash["loss"], loss_hash["iteration"])

            optimizer.zero_grad()

            output = model(train_x)

            loss = -mll(output, train_y)

            loss_hash["loss"].append(loss.item())

            loss_hash["iteration"].append(iteration)
            loss.backward()
            optimizer.step()

    # Plot the loss one last time
    loss_figure(loss_hash["loss"], loss_hash["iteration"])


def model_parameters(
    model: ExactGP, transformed: bool = True
) -> tuple[list[str, list[float]]] | tuple[list[str], list[list[float]]]:
    parameters, parameter_names = [], []
    if transformed:
        constraints = []

        for _, constraint in model.named_constraints():
            constraints.append(constraint)

        for constraint, (parameter_name, param) in zip(
            constraints, model.named_parameters()
        ):
            parameter_names.append(parameter_name)
            if param.size() == 1:
                parameters.append(constraint.transform(param).item())
            else:
                parameters.append(constraint.transform(param).tolist())
    else:
        for parameter_name, param in model.named_parameters():
            parameter_names.append(parameter_name)
            if param.size() == 1:
                parameters.append(param.item())
            else:
                parameters.append(param.tolist())

    return parameter_names, parameters


def print_model_parameters(parameter_names: list, parameters: list) -> str:
    parameter_strings = []
    for parameter_name, parameter in zip(parameter_names, parameters):
        parameter_strings.append(
            f"Transformed parameter name: {parameter_name:42} value = {parameter}\n"
        )

    return "".join(parameter_strings)

def train_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    training_conf: TrainingConf,
    num_tasks: int,
) -> tuple[
    ExactGP, GaussianLikelihood | MultitaskGaussianLikelihood, dict[str, list[float]]
]:
    """
    Train the model using the training data

    Parameters
    -----------
    train_x : torch.Tensor
              The input training data
    train_y : torch.Tensor
              The output training data
    training_conf : TrainingConf
                    Dictionary containing the training specifications
    num_tasks : int
                The number of tasks the model is expected to train on

    Returns
    --------
    model : object
            The trained model
    likelihood : object
                 The likelihood of the trained model
    """
    logger.info("Defining the model specifications.")

    # Load the training specifications
    learning_iterations = training_conf.learning_iterations
    botorch = training_conf.botorch
    debug = training_conf.debug
    optimizer_conf = training_conf.optimizer

    # Define likelihood and model
    likelihood_class = get_likelihood(training_conf.likelihood)
    model_class = get_model(training_conf.model)

    likelihood = define_likelihood(training_conf.likelihood, likelihood_class, train_x)
    model = define_model(training_conf.model, model_class, train_x, train_y, likelihood)

    parameter_names, parameters = model_parameters(model)
    logger.info(
        f"Parameters before training: \n{print_model_parameters(parameter_names, parameters)}"
    )

    # Training in double precision
    model.double()
    likelihood.double()

    # Define the marginal likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    logger.info("Start training the model.")
    with (
        gpytorch.settings.debug(debug),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ),
    ):
        if botorch:
            fit_gpytorch_mll(mll)
        else:
            # Optimize model hyperparameters
            optimizer = define_optimizer(model, optimizer_conf)
            training_loop(
                train_x, train_y, model, mll, optimizer, learning_iterations, debug
            )

    parameter_names, parameters = model_parameters(model)
    logger.info(
        f"Parameters after training: \n{print_model_parameters(parameter_names, parameters)}"
    )

    parameter_dict = dict(zip(parameter_names, parameters))

    return model, likelihood, parameter_dict
