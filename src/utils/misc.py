import subprocess
import torch
import pandas as pd


def metadata_dict(**kwargs):
    """Save the metadata to a dictionary"""
    return kwargs


def git_hash():
    # Check if the current directory is a git repository
    command = "git rev-parse --is-inside-work-tree"

    # Get the result as text
    is_git_repo = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if is_git_repo.returncode != 0:
        short_hash = 'no_git_repo'
        return short_hash
    else:
        command = "git rev-parse --short HEAD"

        short_hash = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return short_hash.stdout.strip('\n')


def dataframe_to_tensor(df):
    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series):
        raise TypeError("Input must be a Pandas DataFrame or Series")

    tensor = torch.from_numpy(df.to_numpy())

    return tensor
