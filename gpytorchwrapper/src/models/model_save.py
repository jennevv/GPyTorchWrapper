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
    model_state : object
                  The state of the model
    model_metadata : dict
                     Dictionary containing the metadata of the model
    training_data : dict
                    Dictionary containing the training data and the transformer
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
