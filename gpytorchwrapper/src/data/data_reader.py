from pathlib import Path

import joblib
import pandas as pd


class DataReader:
    """
    Class for reading data from either a pickle or csv file.
    The processed dataset must be a pandas DataFrame.
    """

    def __init__(self):
        pass

    def _read_csv(self, file: str | Path) -> pd.DataFrame | pd.Series:
        if check_df_header(file):
            data = pd.read_csv(file)
        else:
            data = pd.read_csv(file, header=None)
        return data

    def _read_pickle(self, file: str | Path) -> pd.DataFrame:
        pickle = joblib.load(file)
        data = pickle["processed_dataset"]

        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
            raise NotImplementedError(
                f"The processed dataset in the pickle file is not a pandas DataFrame. Instead got {type(data)}"
            )

        return data

    def read_data(self, file: str | Path, file_type: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        file : str
            The path to the file to read.
        file_type : str
            The type of file to read. Either 'csv' or 'pickle'.

        Returns
        -------
        pd.DataFrame
            The data read from the file

        Raises
        ------
        ImportError
            If the file is not of the file type specified.
        NotImplementedError
            If the file type specified is not implemented or the processed dataset in the pickle file is not a pandas DataFrame.
        """
        if file_type == "csv":
            try:
                data = self._read_csv(file)
            except Exception:
                raise ImportError(
                    f"The file is likely not of the file type {file_type}."
                )
        elif file_type == "pickle":
            try:
                data = self._read_pickle(file)
            except Exception:
                raise ImportError(
                    f"The file is likely not of the file type {file_type}."
                )
        else:
            raise NotImplementedError(
                f"The file type {file_type} is not implemented. Choose either csv or pickle."
            )

        return data

def check_df_header(file: str | Path) -> bool:
    """
    Returns True if the file has a header and False if the file doesn't have one.
    """
    df = pd.read_csv(file, header=None, nrows=5)
    df_header = pd.read_csv(file, nrows=5)
    return tuple(df.dtypes) != tuple(df_header.dtypes)
