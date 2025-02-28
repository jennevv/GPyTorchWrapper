import os
import pathlib
import subprocess
import pytest
import sys
import fnmatch
import shutil

sys.path.insert(0, "../../../GPyTorchWrapper")


@pytest.fixture(scope="function")
def run_script_help():
    result = subprocess.run(
        ["python", "training_gpytorch.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result


@pytest.fixture(scope="module")
def run_script_single_output():
    subprocess.run(
        [
            "python",
            "training_gpytorch.py",
            "-i",
            "./tests/end2end/data.csv",
            "-f",
            "csv",
            "-c",
            "./tests/end2end/single_output_config.yml",
            "-o",
            "test_model_single_output",
            "-d",
            "./test_single_model",
        ],
        check=True,
    )


@pytest.fixture(scope="module")
def run_script_single_output_with_external_test():
    subprocess.run(
        [
            "python",
            "training_gpytorch.py",
            "-i",
            "./tests/end2end/data.csv",
            "-f",
            "csv",
            "-c",
            "./tests/end2end/single_output_config.yml",
            "-o",
            "test_model_single_output",
            "-d",
            "./test_single_model",
            "-t",
            "./tests/end2end/data.csv",
        ],
        check=True,
    )


@pytest.fixture(scope="module")
def run_script_multi_output():
    subprocess.run(
        [
            "python",
            "training_gpytorch.py",
            "-i",
            "./tests/end2end/data.csv",
            "-f",
            "csv",
            "-c",
            "./tests/end2end/multi_output_config.yml",
            "-o",
            "test_model_multi_output",
            "-d",
            "./test_multi_model",
        ],
        check=True,
    )


@pytest.fixture(scope="module")
def run_script_multi_output_with_external_test():
    subprocess.run(
        [
            "python",
            "training_gpytorch.py",
            "-i",
            "./tests/end2end/data.csv",
            "-f",
            "csv",
            "-c",
            "./tests/end2end/multi_output_config.yml",
            "-o",
            "test_model_multi_output",
            "-d",
            "./test_multi_model",
            "-t",
            "./tests/end2end/data.csv",
        ],
        check=True,
    )


def test_run(run_script_help):
    result = run_script_help
    assert "usage:" in result.stdout, "Help should contain 'usage'"
    assert "options:" in result.stdout, "Help should contain 'options'"


def test_loss_figure_exists(run_script_single_output):
    loss_fig = pathlib.Path("loss.png")
    assert loss_fig.exists(), FileNotFoundError
    loss_fig.unlink()


def test_loss_figure_exists_multiple(run_script_multi_output):
    loss_fig = pathlib.Path("loss.png")
    assert loss_fig.exists(), FileNotFoundError
    loss_fig.unlink()


def test_single_output_model_exists(run_script_single_output):
    for file in os.listdir("./test_single_model"):
        file = pathlib.Path(file)
        match = fnmatch.fnmatch(file.name, "test_model_single_output.pth")
        if match:
            break
    else:
        raise FileNotFoundError("No model file found.")
    shutil.rmtree("./test_single_model")


def test_multi_output_model_exists(run_script_multi_output):
    for file in os.listdir("./test_multi_model"):
        file = pathlib.Path(file)
        match = fnmatch.fnmatch(file.name, "test_model_multi_output.pth")
        if match:
            break
    else:
        raise FileNotFoundError("No model file found.")
    shutil.rmtree("./test_multi_model")


def test_single_output_model_exists_with_external_test(
    run_script_single_output_with_external_test,
):
    for file in os.listdir("./test_single_model"):
        file = pathlib.Path(file)
        match = fnmatch.fnmatch(file.name, "test_model_single_output.pth")
        if match:
            break
    else:
        raise FileNotFoundError("No model file found.")
    shutil.rmtree("./test_single_model")


def test_multi_output_model_exists_with_external_test(
    run_script_multi_output_with_external_test,
):
    for file in os.listdir("./test_multi_model"):
        file = pathlib.Path(file)
        match = fnmatch.fnmatch(file.name, "test_model_multi_output.pth")
        if match:
            break
    else:
        raise FileNotFoundError("No model file found.")
    shutil.rmtree("./test_multi_model")
