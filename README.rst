GPyTorchWrapper
===============

**GPyTorchWrapper** is a lightweight Python package designed to streamline the training of Gaussian Process (GP) models using `GPyTorch <https://gpytorch.ai/>`_. While it was developed for modeling potential energy surfaces (PES) in small molecular systems using custom permutationally invariant kernels (based on Bartók & Csányi [1]), it is general enough for other regression tasks.

Features
--------

- Modular YAML-based configuration
- Fully differentiable custom kernels
- Support for SLURM-based HPC environments
- Docker-based example training
- TorchScript export for deployment

Installation
------------

Set up the environment locally:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate gpytorchwrapper
   pip install -e . --use-pep517

For Intel-optimized HPC environments, use `environment_hpc.yml`.

Quickstart
----------

Run the example in Docker:

.. code-block:: bash

   docker build -t gpytorchwrapper .
   ./run-example-in-docker.sh

After training, `3d_plot.png` will appear in the working directory.

HPC Support
-----------

A SLURM-based submission script is available in `bash/sub_gp_training.sh`. Use `gp_training.sh` to configure and launch training jobs with CLI options.

Documentation
-------------

- :doc:`usage`: Model training and YAML configuration
- :doc:`deployment`: TorchScript model export
- :doc:`custom_models`: Adding your own models and kernels
- :doc:`example`: Running the included training example

Reference
---------

[1] Bartók, A. P.; Csányi, G. *Gaussian Approximation Potentials: A Brief Tutorial Introduction*. Int. J. Quantum Chem. 2015, 115 (16), 1051–1057. https://doi.org/10.1002/qua.24927


