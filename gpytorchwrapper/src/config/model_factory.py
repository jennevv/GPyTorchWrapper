import importlib
import os
import pkgutil

import sklearn.preprocessing as transformer_module
import gpytorch.likelihoods as likelihood_module
from gpytorch.likelihoods import Likelihood
from gpytorch.models import ExactGP
from torch.optim import Optimizer

import gpytorchwrapper.src.models.gp_models as model_module
import torch.optim as optimizer_module
from .config_classes import TransformerConf, TrainingConf, OptimizerConf
import logging
import sys

logger = logging.getLogger(__name__)


def get_transformer(transformer_conf: TransformerConf) -> object:
    """
    Get the transformer class and options

    Parameters
    -----------
    transformer_conf : TransformerConf
                    dataclass containing the transformer specifications

    Returns
    --------
    selected_transformer_class : object
                                The selected scikit-learn transformer class
    """
    selected_transformer = transformer_conf.transformer_class
    selected_transformer_class = getattr(transformer_module, selected_transformer)
    options = transformer_conf.transformer_options

    if options:
        selected_transformer_class = selected_transformer_class(**options)
    else:
        selected_transformer_class = selected_transformer_class()
    return selected_transformer_class


def get_likelihood(training_conf: TrainingConf) -> Likelihood:
    """
    Get the likelihood class and options

    Parameters
    -----------
    training_conf : dict
                    Dictionary containing the training specifications

    Returns
    --------
    selected_likelihood_class : object
                                The selected likelihood class
    """
    selected_likelihood = training_conf.likelihood_class
    return getattr(likelihood_module, selected_likelihood)


def get_plugins(path: str | None = None):
    if path is None:
        # Dynamically find the plugins directory relative to this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(current_dir, "../../plugins"))

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Plugins directory not found at {path}")

    sys.path.insert(0, path)

    discovered_plugins = {
        name: importlib.import_module(name)
        for finder, name, ispkg in pkgutil.iter_modules()
        if name.startswith("model_")
    }

    return discovered_plugins


def get_model(training_conf: TrainingConf) -> ExactGP:
    """
    Get the model class and options

    Parameters
    -----------
    training_conf : TrainingConf
                    dataclass containing the training specifications

    Returns
    --------
    selected_model_class : object
                           The selected model class
    """
    selected_model = training_conf.model_class

    plugin_modules = get_plugins()
    if hasattr(model_module, selected_model):
        logger.info(f"Loading model class {selected_model} from {model_module}.")
        return getattr(model_module, selected_model)
    elif plugin_modules != {}:
        for module in plugin_modules.values():
            if hasattr(module, selected_model):
                logger.info(f"Loading model class {selected_model} from {module}.")
                return getattr(module, selected_model)
        raise NotImplementedError(
            f"The specified model class, {selected_model}, is not available in gp_models.py or the plugins folder."
        )
    else:
        raise NotImplementedError(
            f"The specified model class, {selected_model}, is not available in gp_models.py or the plugins folder."
        )

def get_optimizer(optimizer_conf: OptimizerConf) -> Optimizer:
    """
    Get the optimizer class and options

    Parameters
    -----------
    optimizer_conf : OptimizerConf
                    dataclass containing the optimizer specifications

    Returns
    --------
    selected_optimizer_class : object
                                The selected optimizer class
    """
    selected_optimizer = optimizer_conf.optimizer_class
    return getattr(optimizer_module, selected_optimizer)
