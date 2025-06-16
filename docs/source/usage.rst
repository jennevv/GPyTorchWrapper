Usage
=====

Training a Model
----------------

To train a model, you need to provide:

- **Training data** (CSV or pickle format)
- **Model configuration**: a YAML file specifying the architecture and training parameters.

YAML Configuration
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

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

data_conf (Required)
^^^^^^^^^^^^^^^^^^^^

Specifies input/output structure:

- ``num_inputs``: Number of input columns (first N columns).
- ``num_outputs``: Number of output columns.
- ``output_index``: (Optional) Index or list of indices for outputs to train on. Index 0 corresponds to the first output column (i.e., column ``total_columns - num_inputs``).

Example:

.. code-block:: yaml

    output_index: [1, 2]

This selects the second and third output variables.

training_conf (Required)
^^^^^^^^^^^^^^^^^^^^^^^^

Specifies model details and training parameters:

- ``model_class`` must match a model defined in ``gp_models.py``
- ``likelihood_class`` is the likelihood used by GPyTorch
- ``optimizer_class`` is typically Adam, but can be any PyTorch optimizer
- ``botorch: true`` enables Bayesian optimization using BoTorch instead of Adam, ignores the other optimizer options

.. note::
    Training is always performed using **double precision (float64)**.

For more on BoTorch, visit: https://botorch.org/

transform_conf (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^

Configures preprocessing steps for input and output data. Supports any valid ``sklearn.preprocessing`` transformer class.

Transformed and fitted transformers are stored in the final model pickle file.


testing_conf (Optional)
^^^^^^^^^^^^^^^^^^^^^^^

While GP models do not require a validation set, testing options are provided for evaluation and comparison purposes:

Three testing modes are available:

- **Random split**
- **Stratified shuffle split**
- **K-fold cross-validation**

If ``test: true`` and ``test_size`` is set, a random train/test split is applied using ``scikit-learn.train_test_split``.

Running the Training Script
---------------------------

The training process is initiated via a command-line interface or programmatically.

Command-line Example:

.. code-block:: bash

    python train.py \
        --input data.csv \
        --file-type csv \
        --config config.yaml \
        --output model.pth \
        --directory results/ \
        --test-set test.csv

Command-line Arguments:

- ``--input``: Path to training dataset
- ``--file-type``: Format of input data (``csv`` or ``pickle``)
- ``--config``: Path to YAML configuration file
- ``--output``: Name of output model file
- ``--directory``: Output directory (created if it does not exist)
- ``--test-set``: Optional test set path (incompatible with cross-validation)

Main Workflow
-------------

.. code-block:: text

 1. Load and validate input data
 2. Parse YAML configuration
 3. Split into inputs and outputs
 4. Apply optional data transformations
 5. Train GPR model with GPyTorch
 6. Evaluate (optional)
 7. Save model and metadata to disk

.. note::

    Data is always converted to float64 tensors. Model training and evaluation are logged and optionally saved.

Programmatic Example
--------------------

.. code-block:: python

    from train import main

    args = {
        "input": "data.csv",
        "file_type": "csv",
        "config": "config.yaml",
        "output": "model.pth",
        "directory": "results/",
        "test_set": None
    }

    main(args)


