import torch
import gpytorch
import matplotlib.pyplot as plt
import logging

from botorch import fit_gpytorch_mll

from src.config.config_classes import TrainingConf
from src.config.model_factory import get_likelihood, get_model

logger = logging.getLogger(__name__)


def define_optimizer(model: object, learning_rate: float) -> torch.optim.Optimizer:
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


def set_noiseless(likelihood: object, num_tasks: int) -> object:
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
    plt.title('Change of loss during training')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("loss.png", dpi=300)



def training_loop(train_x: torch.Tensor, train_y: torch.Tensor, model, mll, optimizer: torch.optim.Optimizer, learning_iterations: int, debug: bool) -> None:
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
    mll : object
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
    loss_hash = {'loss': [], 'iteration': []}

    with gpytorch.settings.debug(debug):
        for iteration in range(learning_iterations):
            if (iteration + 1) % 10 == 0:
                logger.info(f'Iteration {iteration + 1}/{learning_iterations}')
                loss_figure(loss_hash['loss'], loss_hash['iteration'])

            optimizer.zero_grad()

            output = model(train_x)

            loss = -mll(output, train_y)

            loss_hash['loss'].append(loss.item())

            loss_hash['iteration'].append(iteration)
            loss.backward()
            optimizer.step()

    # Plot the loss one last time
    loss_figure(loss_hash['loss'], loss_hash['iteration'])

def model_parameters(model: object) -> str:
    parameters = []

    for param_name, param in model.named_parameters():
        if param.size() == 1:
            parameters.append(f'Parameter name: {param_name:42} value = {param.item()}\n')
        else:
            parameters.append(f'Parameter name: {param_name:42} value = {param.tolist()}\n')

    return ''.join(parameters)


def train_model(train_x: torch.Tensor, train_y: torch.Tensor, training_conf: TrainingConf, num_tasks: int) -> tuple[object, object]:
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
    logger.info('Defining the model specifications.')

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
        likelihood = likelihood_class(num_tasks=num_tasks,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-5))
    else:
        likelihood = likelihood_class(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-5))

    if noiseless:
        likelihood = set_noiseless(likelihood=likelihood, num_tasks=num_tasks)

    model = model_class(train_x, train_y, likelihood)

    logger.info(f'Parameters before training: \n{model_parameters(model)}')

    # Training in double precision
    model.double()
    likelihood.double()

    # Define the marginal likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    logger.info("Start training the model.")
    if botorch:
        fit_gpytorch_mll(mll)
    else:
        # Optimize model hyperparameters
        optimizer = define_optimizer(model, learning_rate)
        training_loop(train_x, train_y, model, mll, optimizer, learning_iterations, debug)

    logger.info(f'Parameters after training: \n{model_parameters(model)}')

    return model, likelihood
