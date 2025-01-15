import numpy as np
import pytest
from gpytorchwrapper.src.data.data_reader import DataReader
import joblib
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_data():
    data = {
        'a': [1,2,3],
        'b': [4,5,6],
        'c': [7,8,9]
    }
    return pd.DataFrame(data=data)

@pytest.fixture
def data_reader():
    return DataReader()

def test_read_csv(sample_data, data_reader):
    tmp_file = Path('tmp.csv')
    sample_data.to_csv(tmp_file, index=False)

    data = data_reader.read_data(tmp_file, 'csv')

    assert np.allclose(data, sample_data)

    tmp_file.unlink()

def test_read_pickle(sample_data, data_reader):
    tmp_file = Path('tmp.pkl')

    dict = {'processed_dataset': sample_data}

    joblib.dump(dict, tmp_file)

    data = data_reader.read_data(tmp_file, 'pickle')

    assert np.allclose(data, sample_data)

    tmp_file.unlink()

def test_wrong_type_file(sample_data, data_reader):
    tmp_file = Path('tmp.csv')
    sample_data.to_csv(tmp_file, index=False)

    with pytest.raises(ImportError):
        data = data_reader.read_data(tmp_file, 'pickle')

    tmp_file.unlink()

    tmp_file = Path('tmp.pkl')
    dict = {'processed_dataset': sample_data}
    joblib.dump(dict, tmp_file)
    with pytest.raises(ImportError):
        data = data_reader.read_data(tmp_file, 'csv')

    tmp_file.unlink()

def test_not_implemented_type_file(data_reader):
    with pytest.raises(NotImplementedError):
        data_reader.read_data('file', file_type='parquet')


def test_not_dataframe_in_pickle(sample_data, data_reader):
    tmp_file = Path('tmp.pkl')

    dict = {'processed_dataset': sample_data.values}

    joblib.dump(dict, tmp_file)

    with pytest.raises(NotImplementedError):
        data = data_reader._read_pickle(tmp_file)

    tmp_file.unlink()
