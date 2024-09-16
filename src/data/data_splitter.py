import pandas as pd
from src.config import DataConf

def input_output_split(data: pd.DataFrame, data_conf: DataConf) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into input and output

    Parameters
    -----------
    data : pd.DataFrame
        The data to be split
    data_conf : DataConf
            dataclass containing the data specifications

    Returns
    --------
    X : pd.DataFrame
        The input data
    y : pd.DataFrame
        The output data
    """

    n_inputs = data_conf.num_inputs
    n_outputs = data_conf.num_outputs

    if (n_inputs + n_outputs) > len(data.columns):
        raise RuntimeError("The number of inputs and outputs specified exceeds the number of columns in the data file.")

    if n_inputs + n_outputs < len(data.columns):
        if isinstance(data_conf.output_index, int) or isinstance(data_conf.output_index, list):
            output_idx = data_conf.output_index
        else:
            raise KeyError(f"The output index is not specified or not properly specified. Expected int or list[int], got {type(data_conf.output_index)}")
    else:
        output_idx = [i for i in range(n_outputs)]

    if isinstance(output_idx, int):
        output_idx = [output_idx]

    input_columns = data.columns[0:n_inputs]
    output_columns = []

    for idx in output_idx:
        column_i = data.columns[n_inputs + (idx)]
        output_columns.append(column_i)

    X = data[input_columns]
    y = data[output_columns]
    return X, y
