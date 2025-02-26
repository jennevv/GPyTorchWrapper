import subprocess

import numpy as np
import torch
import pandas as pd

def metadata_dict(**kwargs):
    """Save the metadata to a dictionary"""
    return kwargs


def dataframe_to_tensor(df):
    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series):
        raise TypeError("Input must be a Pandas DataFrame or Series")

    tensor = torch.from_numpy(df.to_numpy(dtype=np.float64))

    return tensor.squeeze()
