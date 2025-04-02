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
    else:
        likelihood = likelihood_class()

    return likelihood

def define_model(model_conf, model_class, train_x, train_y, likelihood):
    if model_conf.model_options:
        model = model_class(train_x, train_y, likelihood, **model_conf.model_options)
    else:
        model = model_class(train_x, train_y, likelihood)
    return model


def loss_figure(train_loss: list[float], iteration: list[int], val_loss: list[float] = None) -> None:
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
    plt.scatter(iteration, train_loss, label="Train loss")
    if len(val_loss) > 0:
        plt.scatter(iteration, val_loss, label="Validation loss")
    plt.title("Change of loss during training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png", dpi=300)
    plt.close()


def training_loop(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    model: ExactGP,
    mll: MarginalLogLikelihood,
    optimizer: torch.optim.Optimizer,
    learning_iterations: int,
    debug: bool,
    test_x: torch.Tensor = None,
    test_y: torch.Tensor = None
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
    test_x
    test_y
    Returns
    --------
    None
    """
    loss_hash = {"train_loss": [], "val_loss": [], "iteration": []}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1)

    with gpytorch.settings.debug(debug):
        for iteration in range(learning_iterations):
            if (iteration + 1) % 10 == 0:
                logger.info(f"Iteration {iteration + 1}/{learning_iterations}")
                loss_figure(loss_hash["train_loss"], loss_hash["iteration"], loss_hash["val_loss"])

            optimizer.zero_grad()

            output = model(train_x)

            loss = -mll(output, train_y)

            loss_hash["train_loss"].append(loss.item())

            loss_hash["iteration"].append(iteration)
            loss.backward()
            optimizer.step()

            if test_x is not None:
                model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    val_output = model(test_x)
                    val_loss = -mll(val_output, test_y)
                    loss_hash["val_loss"].append(val_loss.item())

                scheduler.step(val_loss)
                model.train()  # Switch back to training mode
            else:
                scheduler.step(loss)

    # Plot the loss one last time
    loss_figure(loss_hash["train_loss"], loss_hash["iteration"], loss_hash["val_loss"])

def model_parameters(model, transformed: bool = True):
    parameters, parameter_names = [], []

    for full_param_name, param in model.named_parameters():
        parameter_names.append(full_param_name)

        if transformed:
            # Split the full parameter name to get the module and parameter name.
            module_path, param_name = full_param_name.rsplit(".", 1)
            module = model
            for attr in module_path.split("."):
                module = getattr(module, attr)

            # If the parameter is a "raw" parameter and the module defines a property, use that.
            if param_name.startswith("raw_"):
                prop_name = param_name.replace("raw_", "")
                if hasattr(module, prop_name):
                    t_value = getattr(module, prop_name)
                else:
                    # Fall back to manually applying the registered constraint (if any)
                    constraint = module._constraints.get(param_name, None)
                    if constraint is not None:
                        t_value = constraint.transform(param)
                    else:
                        t_value = param
            else:
                t_value = param

            # Convert the value to a Python float or list, as appropriate.
            if t_value.numel() == 1:
                parameters.append(t_value.item())
            else:
                parameters.append(t_value.tolist())
        else:
            # If not transforming, just use the raw parameter.
            parameters.append(param.tolist() if param.numel() > 1 else param.item())

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
    test_x: torch.Tensor = None,
    test_y: torch.Tensor = None
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

    Returns
    --------
    model : object
            The trained model
    likelihood : object
                 The likelihood of the trained model
    parameter_dict : dict
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
                train_x, train_y, model, mll, optimizer, learning_iterations, debug, test_x, test_y
            )

    parameter_names, parameters = model_parameters(model)
    logger.info(
        f"Parameters after training: \n{print_model_parameters(parameter_names, parameters)}"
    )

    parameter_dict = dict(zip(parameter_names, parameters))

    return model, likelihood, parameter_dict
