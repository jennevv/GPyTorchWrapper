import pandas as pd
import pytest

from gpytorchwrapper.src.config.config_classes import DataConf
from gpytorchwrapper.src.data.data_splitter import input_output_split


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'input1': [1, 2, 3],
        'input2': [4, 5, 6],
        'output1': [7, 8, 9],
        'output2': [10, 11, 12],
        'extra': [13, 14, 15]
    })


@pytest.fixture
def data_conf():
    return DataConf(num_inputs=2, num_outputs=2, output_index=None)


def test_input_output_split_basic(sample_data, data_conf):
    X, y = input_output_split(sample_data[['input1', 'input2', 'output1', 'output2']], data_conf)

    assert list(X.columns) == ['input1', 'input2']
    assert list(y.columns) == ['output1', 'output2']
    assert X.shape == (3, 2)
    assert y.shape == (3, 2)


def test_input_output_split_with_output_index(sample_data):
    data_conf = DataConf(num_inputs=2, num_outputs=2, output_index=[0, 2])
    X, y = input_output_split(sample_data, data_conf)

    assert list(X.columns) == ['input1', 'input2']
    assert list(y.columns) == ['output1', 'extra']
    assert X.shape == (3, 2)
    assert y.shape == (3, 2)


def test_input_output_split_single_output(sample_data):
    data_conf = DataConf(num_inputs=3, num_outputs=1, output_index=0)
    X, y = input_output_split(sample_data, data_conf)

    assert list(X.columns) == ['input1', 'input2', 'output1']
    assert list(y.columns) == ['output2']
    assert X.shape == (3, 3)
    assert y.shape == (3, 1)


def test_input_output_split_error_too_many_columns(sample_data):
    data_conf = DataConf(num_inputs=3, num_outputs=3, output_index=None)

    with pytest.raises(RuntimeError,
                       match="The number of inputs and outputs specified exceeds the number of columns in the data file."):
        input_output_split(sample_data, data_conf)


def test_input_output_split_error_missing_output_index(sample_data):
    data_conf = DataConf(num_inputs=2, num_outputs=2, output_index=None)

    with pytest.raises(KeyError):
        input_output_split(sample_data, data_conf)


def test_input_output_split_wrong_index_type(sample_data):
    data_conf = DataConf(num_inputs=2, num_outputs=2, output_index='output1')
    with pytest.raises(KeyError):
        X, y = input_output_split(sample_data, data_conf)
