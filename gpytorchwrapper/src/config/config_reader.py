import pathlib
import yaml
from .config_classes import create_config, Config


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
    TypeError
        If the input_path variable is not of Path type
    FileNotFoundError
        If the file does not exist
    NotImplementedError
        If the file structure of the YAML file is incorrect
    """
    if isinstance(input_path, str):
        input_path = pathlib.Path(input_path)

    if not isinstance(input_path, pathlib.Path):
        raise TypeError(f"Expected pathlib.Path, got {type(input_path)}")

    if not input_path.exists():
        raise FileNotFoundError(f"The file {input_path} does not exist")

    with input_path.open("r") as f:
        input_dict = yaml.safe_load(f)

    try:
        config = create_config(input_dict)
    except:
        raise NotImplementedError("Incorrect YAML file structure.")

    return config
