import pytest
import pandas as pd
import torch

from gpytorchwrapper.src.utils import dataframe_to_tensor


@pytest.fixture
def sample_dataframe():
    data = {
        'a': [1, 2, 3, 4],
        'b': [5, 6, 7, 8],
        'c': [9, 10, 11, 12]
    }
    return pd.DataFrame(data)

def test_dataframe_to_tensor(sample_dataframe):
    tensor = dataframe_to_tensor(sample_dataframe)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == torch.Size([4, 3])

    tensor = dataframe_to_tensor(sample_dataframe['a'])
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == torch.Size([4])

def test_dataframe_to_tensor_type_error(sample_dataframe):
    np_array = sample_dataframe.to_numpy()
    with pytest.raises(TypeError):
        dataframe_to_tensor(np_array)

    list = sample_dataframe.values.tolist()
    with pytest.raises(TypeError):
        dataframe_to_tensor(list)


