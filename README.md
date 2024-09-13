# GPyTorchWrapper
## Installation

Use the `environment.yaml` file to install the necessary dependencies. 

```bash
conda env create -f environment.yaml
```

Activate the environment and install the package in development mode.

```bash
conda activate gpytorchwrapper
pip install -e .
```

## Usage
The package provides a wrapper around the `gpytorch` library to simplify the process of training and evaluating Gaussian Process models. 
The wrapper provides a simple interface to train and evaluate models, and to perform hyperparameter optimization.

### Training a model
To train a model, you need to provide the training data and the model configuration. 
The model configuration is a yaml file that specifies the model architecture and the training parameters.

```yaml
dataSpec:
    nInputs: int
    output: 
        nOutputs: int
        outputIndex: int
transformerSpec:
    transformInput:
        transformData: boolean
        transformerClass: string
        transformerOptions:
          option1: value1
          option2: value2
        columns: list
    transformOutput:
        transformData: boolean
        transformerClass: string
        transformerOptions: 
            option1: value1
            option2: value2
trainingSpec:
    modelClass: SingleGPRBF
    likelihoodClass: GaussianLikelihood
    learningRate: float
    learningIterations: int
    meanSpec:
        constant_value: float
        optimize: boolean
        transformer: boolean
    noiseless: boolean
    botorch: boolean
testingSpec:
    test: boolean
    testSize: 0.2
    stratSplit: boolean
    kFold: boolean
```

The `dataSpec` section specifies the input and output dimensions of the data.
The input columns are expected to be the first `nInputs` columns of the input data.  
The indices of the output columns are specified in the `outputIndex` field   
This is because the output data may contain multiple columns.  
The column index starts at 1 which is the index of column [totalColumns - nInputs].  

The `transformerSpec` section specifies the data transformation to be applied to the input and output data.  
The transformer classes are defined by the class names of the scikit-learn sklearn.preprocessing transformers.  

The `trainingSpec` section specifies the model architecture and the training parameters.  
The modelClass must be specified as one of the available models in the gp_models.py file.  
Instead of using normal training loop, the model can be trained using the BoTorch library.
See more information about BoTorch [here](https://botorch.org/).

The `testingSpec` section specifies the testing parameters.  

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
python pt2ts.py -i model.pth -o model.ts -c config.yaml 
```

This model will be able to output the predictions and the uncertainty of the model for a given set of inputs x.  
```python
import torch

model = torch.jit.load('model.ts')

pred, pred_var = model(x)
```
