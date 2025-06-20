GPyTorchWrapper
===============

**GPyTorchWrapper** is a lightweight Python package designed to streamline the training of Gaussian Process (GP) models using `GPyTorch <https://gpytorch.ai/>`_. While it was developed for modeling potential energy surfaces (PES) in small molecular systems using custom permutationally invariant kernels (based on Bartók & Csányi [1]), it is general enough for other regression tasks.

Features
--------

- Modular YAML-based configuration
- Fully differentiable, permutationally invariant kernels
- TorchScript export for deployment

Read m

Installation
------------

Set up the environment locally:

.. code-block:: bash

   conda env create -f environment.yml

.. code-block:: bash

   conda activate gpytorchwrapper

.. code-block:: bash
   
   pip install -e . --use-pep517

For Intel-optimized HPC environments, use ``environment_hpc.yml``.

Quickstart
----------

Run the example in Docker:

.. code-block:: bash

   docker build -t gpytorchwrapper .
   ./run-example-in-docker.sh

After training, ``3d_plot.png`` will appear in the working directory.

HPC Support
-----------

A SLURM-based submission script is available in ``bash/sub_gp_training.sh``. Use ``gp_training.sh`` to configure and launch training jobs with CLI options.

Documentation
-------------

Documentation is available on `https://jennevv.github.io/GPyTorchWrapper/`

Reference
---------

[1] Bartók, A. P.; Csányi, G. *Gaussian Approximation Potentials: A Brief Tutorial Introduction*. Int. J. Quantum Chem. 2015, 115 (16), 1051–1057. https://doi.org/10.1002/qua.24927


