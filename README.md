# GPyTorchWrapper
This repository contains a wrapper for training GP models using GPyTorch.
I have created this to simplify modeling potential energy surface models for small molecular systems in GPyTorch, but it is not limited to that application.
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
Currently, GPU support, although supported by GPyTorch, is not available in the wrapper. When it is made available, a GPU specific conda environment will be added. 

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
  model_class: str
  likelihood_class: str
  learning_rate: float
  learning_iterations: int
  noiseless: bool
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
from src.gp_utils import get_model, get_likelihood, read_yaml

metadata = torch.load('model.pth')
config = read_yaml('config.yaml')

train_x, train_y = metadata['training_data']['train_x'], metadata['training_data']['train_y']

likelihood_class = get_likelihood(config['trainingSpec'], num_tasks=config['dataSpec']['output']['nOutputs'])  
model_class = get_model(config['trainingSpec'])

likelihood = likelihood_class()
model = model_class(train_x, train_y, likelihood)

model.double()
likelihood.double()

model.load_state_dict(metadata['model_state'])
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
