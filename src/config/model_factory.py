import sklearn.preprocessing as transformer_module
import gpytorch.likelihoods as likelihood_module
import src.models.gp_models as model_module
from src.config import TransformerConf, TrainingConf


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


def get_likelihood(training_conf: TrainingConf) -> object:
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



def get_model(training_conf: TrainingConf) -> object:
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
    return getattr(model_module, selected_model)



