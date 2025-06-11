from pathlib import Path
import torch


def save_model(
    state_dict: dict,
    config: dict,
    training_data: dict,
    metrics: dict,
    output: str,
    directory: Path,
) -> None:
    """
    Save the model to a .pth file

    Parameters
    -----------
    state_dict : dict
        The state of the model
    config : dict
        The configuration dictionary to setup the training
    training_data : dict
        Contains the training inputs and outputs used
    metrics : dict
        Dictionary containing the metrics on the training and test sets
    output : str
        The name of the output file
    directory : pathlib.Path
        The output directory

    Returns
    --------
    None
    """
    if isinstance(directory, str):
        directory = Path(directory)

    savestate = {
        "state_dict": state_dict,
        "config": config,
        "training_data": training_data,
        "metrics": metrics,
    }

    torch.save(savestate, directory / f"{output}.pth")
