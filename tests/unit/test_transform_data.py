import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from gpytorchwrapper.src.data.data_transform import transform_data


@pytest.fixture
def sample_dataframe():
    data = {
        'a': [1, 2, 3, 4],
        'b': [5, 6, 7, 8],
        'c': [9, 10, 11, 12]
    }

    return pd.DataFrame(data=data)

@pytest.fixture
def sample_fitted_transformer(sample_dataframe):
    transformer = StandardScaler()
    transformer.fit(sample_dataframe.values)  # .values is used to get the numpy array, so no column names are used in the fit

    return transformer


def test_transform_data_unfitted_1(sample_dataframe):
    unfitted_transformer = StandardScaler()

    df, transformer = transform_data(sample_dataframe, unfitted_transformer)

    assert np.allclose(df, StandardScaler().fit_transform(sample_dataframe))
    assert np.allclose(transformer.mean_, StandardScaler().fit(sample_dataframe).mean_)
    assert np.allclose(transformer.var_, StandardScaler().fit(sample_dataframe).var_)


def test_transform_data_unfitted_2(sample_dataframe):
    unfitted_transformer = StandardScaler()

    df, transformer = transform_data(sample_dataframe, unfitted_transformer, columns=['a'])

    assert np.allclose(df['a'], pd.DataFrame(StandardScaler().fit_transform(sample_dataframe[['a']])).squeeze())
    assert np.allclose(df[['b', 'c']], sample_dataframe[['b', 'c']])
    assert(hasattr(transformer, 'mean_'))

def test_transform_data_unfitted_3(sample_dataframe):
    unfitted_transformer = StandardScaler()

    df, transformer = transform_data(sample_dataframe, unfitted_transformer, columns=[0])

    assert np.allclose(df['a'], pd.DataFrame(StandardScaler().fit_transform(sample_dataframe[['a']])).squeeze())
    assert np.allclose(df[['b', 'c']], sample_dataframe[['b', 'c']])
    assert(hasattr(transformer, 'mean_'))


def test_transform_data_unfitted_4(sample_dataframe):
    unfitted_transformer = StandardScaler()

    df, transformer = transform_data(sample_dataframe, unfitted_transformer, columns=[0, 1])

    assert np.allclose(df[['a', 'b']], pd.DataFrame(StandardScaler().fit_transform(sample_dataframe[['a', 'b']])))
    assert np.allclose(df[['c']], sample_dataframe[['c']])
    assert(hasattr(transformer, 'mean_'))

def test_transform_data_fitted(sample_dataframe, sample_fitted_transformer):
    with pytest.raises(ValueError):
        df, transformer = transform_data(sample_dataframe, sample_fitted_transformer)

    df = transform_data(sample_dataframe, sample_fitted_transformer)
    assert np.allclose(df, StandardScaler().fit_transform(sample_dataframe))

    with pytest.raises(ValueError):
        df = transform_data(sample_dataframe, sample_fitted_transformer, columns=['a'])


def test_transform_data_invalid_input(sample_dataframe, sample_fitted_transformer):
    with pytest.raises(NotImplementedError):
        transform_data(sample_dataframe.values, sample_fitted_transformer)
