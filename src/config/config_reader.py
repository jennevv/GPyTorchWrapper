import pathlib
import pyyaml
from src.config.config_reader import config_reader, Config
from typing import Optional

def read_yaml(input_path: pathlib.Path | str) -> Config:
    """
    Read the input file and return a dictionary with the specifications.

    Parameters
    -----------
    input_path : pathlib.Path
            Path to the input file

    Returns
    --------
    input_dict : Config
            Dictionary containing the specifications

    Raises
    ------

    """
    if isinstance(input_path, str):
        input_path = pathlib.Path(input_path)

    with input_path.open('r') as f:
        input_dict = yaml.safe_load(f)
    return input_dict


