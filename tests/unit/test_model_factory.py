import gpytorch
import pytest
import sklearn
import torch

import gpytorchwrapper.src.models.gp_models as model_module
from gpytorchwrapper.src.config.config_classes import (
    TransformerConf,
    TrainingConf,
    OptimizerConf,
    ModelConf,
    LikelihoodConf,
)
from gpytorchwrapper.src.config.model_factory import (
    get_transformer,
    get_likelihood,
    get_model,
    get_optimizer,
)


def test_get_transformer():
    # Test with PowerTransformer and options
    transformer_conf = TransformerConf(
        transform_data=True,
        transformer_class="PowerTransformer",
        transformer_options={"method": "yeo-johnson", "standardize": True},
    )
    transformer = get_transformer(transformer_conf)
    assert isinstance(transformer, sklearn.preprocessing.PowerTransformer)
    assert transformer.method == "yeo-johnson"
    assert transformer.standardize is True

    # Test with StandardScaler and no options
    transformer_conf = TransformerConf(
        transform_data=True,
        transformer_class="StandardScaler",
        transformer_options=None,
    )
    transformer = get_transformer(transformer_conf)
    assert isinstance(transformer, sklearn.preprocessing.StandardScaler)

    # Test with non-existent transformer
    transformer_conf = TransformerConf(
        transform_data=True,
        transformer_class="NonExistentTransformer",
        transformer_options=None,
    )
    with pytest.raises(AttributeError):
        get_transformer(transformer_conf)


def test_get_likelihood():
    # Test with GaussianLikelihood
    training_conf = TrainingConf(
        model=ModelConf(model_class="TestModel"),
        likelihood=LikelihoodConf("GaussianLikelihood"),
        learning_iterations=100,
    )
    likelihood = get_likelihood(training_conf.likelihood)
    assert likelihood is gpytorch.likelihoods.GaussianLikelihood

    # Test with MultitaskGaussianLikelihood
    training_conf = TrainingConf(
        model=ModelConf(model_class="TestMultitaskGPModel"),
        likelihood=LikelihoodConf("MultitaskGaussianLikelihood"),
        learning_iterations=100,
    )
    likelihood = get_likelihood(training_conf.likelihood)
    assert likelihood is gpytorch.likelihoods.MultitaskGaussianLikelihood

    # Test with non-existent likelihood
    training_conf = TrainingConf(
        model=ModelConf(model_class="TestModel"),
        likelihood=LikelihoodConf("NonexistentLikelihood"),
        learning_iterations=100,
    )
    with pytest.raises(AttributeError):
        get_likelihood(training_conf.likelihood)


def test_get_model():
    training_conf = TrainingConf(
        model=ModelConf(model_class="TestModel"),
        likelihood=LikelihoodConf("GaussianLikelihood"),
        learning_iterations=100,
    )
    model_class = get_model(training_conf.model)
    assert model_class == model_module.TestModel

    # Test with non-existent model
    training_conf = TrainingConf(
        model=ModelConf(model_class="NonexistentModel"),
        likelihood=LikelihoodConf("GaussianLikelihood"),
        learning_iterations=100,
    )

    with pytest.raises(NotImplementedError):
        get_model(training_conf.model)


def test_get_optimizer():
    training_conf = TrainingConf(
        model=ModelConf(model_class="TestModel"),
        likelihood=LikelihoodConf("GaussianLikelihood"),
        learning_iterations=100,
        optimizer=OptimizerConf(
            optimizer_class="Adam",
            optimizer_options={"lr": 0.1},
        ),
    )
    optimizer = get_optimizer(training_conf.optimizer)
    assert optimizer is torch.optim.Adam

    # Test with non-existent optimizer
    training_conf = TrainingConf(
        model=ModelConf(model_class="TestModel"),
        likelihood=LikelihoodConf("GaussianLikelihood"),
        learning_iterations=100,
        optimizer=OptimizerConf(
            optimizer_class="NonExistentOptimizer",
            optimizer_options={"lr": 0.1},
        ),
    )
    with pytest.raises(AttributeError):
        get_optimizer(training_conf.optimizer)
