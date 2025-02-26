import torch
import gpytorch
import matplotlib.pyplot as plt
import logging

from botorch import fit_gpytorch_mll

from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.mlls import MarginalLogLikelihood

from gpytorchwrapper.src.config.config_classes import TrainingConf
from gpytorchwrapper.src.config.model_factory import get_likelihood, get_model

logger = logging.getLogger(__name__)


def define_optimizer(model: ExactGP, learning_rate: float) -> torch.optim.Optimizer:
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

    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def set_noiseless(
    likelihood: GaussianLikelihood | MultitaskGaussianLikelihood, num_tasks: int
) -> object:
    """
    Set the likelihood noise to a small value to simulate noiseless data
    The noise is set to 1e-8 for performance reasons

    Parameters
    -----------
    likelihood : object
                 The likelihood object
    num_tasks : int
                The number of tasks the model is expected to train on

    Returns
    --------
    likelihood : object
                 The likelihood object with the noise set to a small value
    """
    if num_tasks > 1:
        likelihood.task_noises = torch.tensor([1e-8] * num_tasks)
        likelihood.raw_task_noises.detach_()
    else:
        likelihood.noise = 1e-8
        likelihood.raw_noise.detach_()
    return likelihood


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
    learning_rate = training_conf.learning_rate
    botorch = training_conf.botorch
    noiseless = training_conf.noiseless
    debug = training_conf.debug

    # Define likelihood and model
    likelihood_class = get_likelihood(training_conf)
    model_class = get_model(training_conf)

    # Set noise constraint like sklearn WhiteKernel for comparison reasons
    if num_tasks > 1:
        if noiseless:
            likelihood = likelihood_class(
                num_tasks=num_tasks,
                noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
            )
            likelihood.raw_task_noises.requires_grad = False
        else:
            likelihood = likelihood_class(
                num_tasks=num_tasks,
                noise_constraint=gpytorch.constraints.GreaterThan(1e-5),
            )
    else:
        if noiseless:
            likelihood = likelihood_class(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
            )
            likelihood.raw_noise.requires_grad = False
        else:
            likelihood = likelihood_class(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-5)
            )

    model = model_class(train_x, train_y, likelihood)
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
    with gpytorch.settings.debug(debug), gpytorch.settings.fast_computations(
    covar_root_decomposition=False,
    log_prob=False,
    solves=False
):
        if botorch:
            fit_gpytorch_mll(mll)
        else:
            # Optimize model hyperparameters
            optimizer = define_optimizer(model, learning_rate)
            training_loop(
                train_x, train_y, model, mll, optimizer, learning_iterations, debug
            )

    parameter_names, parameters = model_parameters(model)
    logger.info(
        f"Parameters after training: \n{print_model_parameters(parameter_names, parameters)}"
    )

    parameter_dict = dict(zip(parameter_names, parameters))

    return model, likelihood, parameter_dict
