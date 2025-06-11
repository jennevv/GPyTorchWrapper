import pandas as pd
import numpy as np
import torch
from gpytorchwrapper.src.config.config_classes import create_config
from gpytorchwrapper.src.models.model_load import load_model
import matplotlib.pyplot as plt

def plot_3d_surface(X, y, y_pred):
    """
    Plot a 3D surface of data with 2D inputs and 1D output.

    Parameters:
    - X: (n_samples, 2) input features
    - y: (n_samples,) output values
    """
    # Reshape for surface plotting
    x1 = np.unique(X[:, 0])
    x2 = np.unique(X[:, 1])
    X1, X2 = np.meshgrid(x1, x2)
    Y_pred = y_pred.reshape(len(x2), len(x1))  # reshape y to match the meshgrid

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y_pred, cmap='viridis', edgecolor='k', alpha=0.8)

    ax.scatter(X[:,0], X[:,1], y, label="original data")

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('3D Surface Plot of Function')
    ax.legend()

    fig.savefig("3d_plot.png")
    plt.show()

model_dump = torch.load("example/model.pth")

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

model.eval()

df = pd.read_csv("example/example_data.csv", header=None)
x = df[df.columns[:2]].values
y = df[df.columns[2]].values

x_tensor = torch.as_tensor(x)
with torch.no_grad():
    dist = model(x_tensor)
    mean = dist.mean

y_pred = mean.detach().numpy()

plot_3d_surface(x, y, y_pred)
