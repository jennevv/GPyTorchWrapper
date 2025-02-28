import pytest
import pathlib

from gpytorchwrapper.src.config.config_reader import read_yaml


@pytest.fixture
def sample_yaml_file() -> pathlib.Path:
    content = """
    data_conf:
      num_inputs: 3
      num_outputs: 1
      output_index: 1

    transform_conf:
      transform_input:
        transform_data: true
        transformer_class: PowerTransformer
        transformer_options:
          method: yeo-johnson
          standardize: true
        columns: null
      transform_output:
        transform_data: false
        transformer_class: MinMaxScaler
        transformer_options: null
        columns: null

    training_conf:
      model_class: SingleGPRBF
      likelihood_class: GaussianLikelihood
      learning_iterations: 100
      botorch: false
      debug: True
      optimizer:
        optimizer_class: Adam
        optimizer_options:
          lr: 0.1

    testing_conf:
      test: false
      test_size: 0.2
      strat_shuffle_split: false
      kfold: false
      kfold_bins: null
      """
    file = pathlib.Path("./test_config.yml")
    file.write_text(content)
    return file


def test_read_yaml_valid_file(sample_yaml_file):
    config = read_yaml(sample_yaml_file)
    assert config.data_conf.num_inputs == 3
    assert config.data_conf.num_outputs == 1
    assert config.data_conf.output_index == 1

    assert config.transform_conf.transform_input.transform_data is True
    assert config.transform_conf.transform_input.transformer_class == "PowerTransformer"
    assert config.transform_conf.transform_input.transformer_options == {
        "method": "yeo-johnson",
        "standardize": True,
    }
    assert config.transform_conf.transform_input.columns is None

    assert config.transform_conf.transform_output.transform_data is False
    assert config.transform_conf.transform_output.transformer_class == "MinMaxScaler"
    assert config.transform_conf.transform_output.transformer_options is None
    assert config.transform_conf.transform_output.columns is None

    assert config.training_conf.model_class == "SingleGPRBF"
    assert config.training_conf.likelihood_class == "GaussianLikelihood"
    assert config.training_conf.learning_iterations == 100
    assert config.training_conf.botorch is False
    assert config.training_conf.debug is True
    assert config.training_conf.optimizer.optimizer_class == "Adam"
    assert config.training_conf.optimizer.optimizer_options == {"lr": 0.1}

    assert config.testing_conf.test is False
    assert config.testing_conf.test_size == 0.2
    assert config.testing_conf.strat_shuffle_split is False
    assert config.testing_conf.kfold is False
    assert config.testing_conf.kfold_bins is None

    sample_yaml_file.unlink()


def test_read_yaml_invalid_file():
    with pytest.raises(FileNotFoundError):
        read_yaml("non_existent_file.yml")


def test_read_yaml_type_error():
    with pytest.raises(TypeError):
        read_yaml(123)


def test_read_yaml_invalid_yaml():
    content = """
    key: value
    """

    file = pathlib.Path("./invalid.yml")
    file.write_text(content)
    with pytest.raises(NotImplementedError):
        read_yaml(file)
    file.unlink()
