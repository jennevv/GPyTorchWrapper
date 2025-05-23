# GPyTorchWrapper
This repository contains a wrapper for training GP models using GPyTorch.
I have created this to simplify modeling potential energy surface models for small molecular systems in GPyTorch, but it is not limited to that application.

For the previously stated purpose, custom permutationally invariant kernel functions have been developed based on the definiton of Bartók and Cśanyi [1], however implemented for a global descriptor.
These kernel functions are fully differentiable using autograd to allow the models to be used in dynamics simulations.
## Installation

Use the `environment.yml` file to install the necessary dependencies on a local machine.
If you want to use an HPC cluster for training, the `environment_hpc.yml` contains the necessary packages optimized for intel CPUs available in the [Intel Software Development Tools]([https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-0/install-intel-ai-analytics-toolkit-via-conda.html](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-0/overview.html)).
```bash
conda env create -f environment.yml
```

Activate the environment and install the package in development mode using pip.

```bash
conda activate gpytorchwrapper
pip install -e . --use-pep517
```
Currently, GPU support is not available in the wrapper so there is no GPU specific environment. 

## Run example in Docker
The repository contains a Dockerfile that allows you to easily train a model based on the data and configuration in the example directory.

Run the following commands inside of the repo directory.
```bash
docker build -t image .
```
```bash
./run-example-in-docker.sh
```
A file called `3d_plot.png` is now present in the local directory and shows a 3D plot of the fit against the noisy training data. 

## Run on cluster with Slurm scheduler
The `bash/` directory contains an example Slurm-based submit script called `sub_gp_training.sh`. The `gp_training.sh` script is a CLI tool to change the submit script for certain input parameters and it submits the job afterwards.  

## Usage
### Training a model
To train a model, you need to provide the training data and the model configuration. 
The model configuration is a yaml file that specifies the model architecture and the training parameters.

```yaml
data_conf:
  num_inputs: int
  num_outputs: int
  output_index: int | list[int]

transform_conf:
  transform_input:
    transform_data: bool
    transformer_class: str
    transformer_options:
      method: str
      standardize: bool
    columns: list | null
  transform_output:
    transform_data: bool
    transformer_class: str
    transformer_options: dict | null
    columns: list | null

training_conf:
  model:
    model_class: str
  likelihood:
    likelihood_class: str
    likelihood_options:
      option1: str | float
  optimizer:
    optimizer_class: str
    learning_rate: float
  learning_iterations: int
  botorch: bool
  debug: bool

testing_conf:
  test: bool
  test_size: float
  strat_shuffle_split: bool
  kfold: bool
  kfold_bins: int | null
```
#### data_conf
The `data_conf` section specifies the input and output dimensions of the data.    

`num_inputs`: the number of input columns in your dataset, these are the first `num_inputs` columns 
`num_outputs`: the number of columns in your dataset containing outputs

If more than one output is present in the dataset but you want to train on only one or a certain number of those outputs, 
you can use the `output_index` variable.    
The column index this refers to starts at 0 which is the index of column [total_columns - num_inputs].    
For example, if there are 4 output values and you want the second and third one to be trained on, you would write    
`output_index: [1,2]`

#### transform_conf
The `transform_conf` section specifies the data transformation to be applied to the input and output data before training.    
The transformer classes are defined by the class names of the scikit-learn `sklearn.preprocessing` transformers.    
After training, the fitted transformers are available in the model pickle file. 

#### training_conf
The `training_conf` section specifies the model architecture and the training parameters.  
The `model_class` must be specified as one of the available models in the `gp_models.py` file.  

Regular hyperparameter optimization in GPyTorch makes use of the Adam optimizer for maximizng the log of the marginal likelihood (LML).
Alternatively, the model can be trained using the BoTorch's Bayesian-based optimization of the LML.
See more information about BoTorch [here](https://botorch.org/).

**Note**: training is always performed in double precision.    

#### testing_conf
A GP model learns its hyperparameters without requiring a validation set.
However, it might be interesting to compare the performance between different models or to have some vague idea of the model's actual performance without using a test set.
For this reason, it is possible to split the dataset into a training and test test using three different methods:

- Random split
- Stratified shuffle split
- K-fold split

The `testing_conf` section specifies the testing parameters.  
If `test` is `True` and the `test_size` is specified, the split will be random using scikit-learn's `train_test_split`.     

### Loading the model
How to load the model is shown in the example below.

```python
import torch
import gpytorch
from gpytorchwrapper.src.config.config_classes import create_config
from gpytorchwrapper.src.models.model_load import load_model

model_dump = torch.load("model.pth")

config = create_config(model_dump["config"])

train_x, train_y = (
    model_dump["training_data"]["train_x"],
    model_dump["training_data"]["train_y"],
)

if config.transform_conf.transform_input.transform_data:
    input_transformer = model_dump["training_data"]["input_transformer"]
else:
    input_transformer = None

model, likelihood = load_model(config, model_dump, train_x, train_y)
```

### Saving the model in TorchScript format
The model can be saved in TorchScript format for deployment purposes using the pt2ts.py script.  
```bash
python pt2ts.py -i model.pth -o model.ts -c config.yml 
```

This model will be able to output the predictions and the uncertainty of the model for a given set of inputs x.  
```python
import torch

model = torch.jit.load('model.ts')

pred, pred_var = model(x)
```
### Custom models
Custom models can be added in the `gpytorchwrapper/plugins` directory.  
The module's name of the module containing the custom model must start with `model_`

# References
[1] Bartók, A. P.; Csányi, G. Gaussian Approximation Potentials: A Brief Tutorial Introduction. International Journal of Quantum Chemistry 2015, 115 (16), 1051–1057. https://doi.org/10.1002/qua.24927.

