Deployment
==========

Saving the Model in TorchScript Format
--------------------------------------

For deployment purposes, the model can be converted and saved in **TorchScript** format using the ``pt2ts.py`` script.
This means that the inference can be ran without needing the model class to be specified.

Command-line usage:

.. code-block:: bash

    python pt2ts.py -i model.pth -o model.ts -d ./

This will produce a ``model.ts`` file that can be loaded and used in production environments.

Command-line Arguments:

- ``-i, --input``: Path to trained model pickle file
- ``-o, --output``: Name of output torcschript file
- ``-d, --directory``: Output directory (created if it does not exist)

TorchScript Model Usage
------------------------

The resulting TorchScript model can be loaded with PyTorch's JIT interface. It outputs both the **predictions** and the **uncertainty** (predictive variance) for a given set of inputs ``x``.

.. code-block:: python

    import torch

    model = torch.jit.load('model.ts')

    pred, pred_var = model(x)

``pred`` and ``pred_var`` are both tensors of shape ``(n_samples, output_dim)``.
``pred_var`` represents the model's uncertainty (variance) at each input point.

.. note::

   Make sure that the input tensor ``x`` has the correct shape and dtype (``float64``) expected by the model.

