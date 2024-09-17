import subprocess

subprocess.run([
        "python", "training_gpytorch.py",
        "-i", "./tests/end2end/data.csv",
        "-f", "csv",
        "-c", "./tests/end2end/multi_output_config.yml",
        "-o", "test_model_multi_output",
        "-d", "./test_models"
    ], check=True)