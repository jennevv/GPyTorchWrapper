from gpytorch.likelihoods import Likelihood
from gpytorch.models import GP
from torch import Tensor

from gpytorchwrapper.src.config.config_classes import Config
from gpytorchwrapper.src.config.model_factory import get_likelihood, get_model
from gpytorchwrapper.src.models.model_train import define_likelihood, define_model


def load_model(
    config: Config,
    model_dump: dict,
    train_x: Tensor,
    train_y: Tensor,
) -> tuple[GP, Likelihood]:
    """
    Load a model from a config file and the dumped model.
    The model and likelihood objects are set to training mode with double precision.

    Parameters
    ----------
    config : Config
        Object specifying the configuration used for training.
    model_dump : dict
        The unpickled dumped model file
    train_x : Tensor
        The input tensor used for training
    train_y : Tensor
        The output tensor used for training

    Returns
    -------
    return : tuple of GP and Likelihood
        The loaded GP and likelihood objects


    """
    likelihood_class = get_likelihood(config.training_conf.likelihood)
    model_class = get_model(config.training_conf.model)

    likelihood = define_likelihood(
        config.training_conf.likelihood, likelihood_class, train_x
    )
    model = define_model(
        config.training_conf.model, model_class, train_x, train_y, likelihood
    )

    model.double()
    likelihood.double()

    model.load_state_dict(model_dump["state_dict"])

    return model, likelihood
