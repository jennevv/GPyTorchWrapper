.. custom_models:

Custom Models
=============

The ``gpytorchwrapper`` package provides an simple way to use custom Gaussian Process (GP) models and kernels using GPyTorch and BoTorch.
This guide explains how to add and use your own models and describes the supported custom kernel functions.

Directory Structure for Plugins
-------------------------------

All custom models must be placed inside the ``gpytorchwrapper/plugins/`` directory.

**Naming Convention**

Custom model modules **must start with** ``model_`` to be discovered and loaded
automatically by the wrapper.

**Example:**

.. code-block:: text

    gpytorchwrapper/
    └── plugins/
        ├── model_custom_gp.py     ✅ Picked up automatically
        └── my_model.py            ❌ Ignored

Each module should define at least one class that subclasses
``gpytorch.models.ExactGP`` and ``botorch.models.gpytorch.GPyTorchModel``.

Writing a Custom Model
-----------------------

Here’s an example using a custom kernel (``PolyxMaternKernelPermInv``):

.. code-block:: python

    from gpytorch import kernels, means, models, distributions, constraints, priors
    from gpytorchwrapper.src.kernels.polyxmatern_kernel_perminv import (
        PolyxMaternKernelPermInv,
    )
    from botorch.models.gpytorch import GPyTorchModel

    class ArH2pS0(models.ExactGP, GPyTorchModel):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)

            outputscale_prior = priors.NormalPrior(5.0, 2.0)
            lengthscale_prior = priors.NormalPrior(0.5, 0.4)
            variance_prior = priors.NormalPrior(0.5, 0.4)

            n_atoms = 3
            idx_equiv_atoms = [[0, 1]]  # Permutational symmetry

            self.mean_module = means.ConstantMean()
            self.covar_module = kernels.ScaleKernel(
                PolyxMaternKernelPermInv(
                    n_atoms=n_atoms,
                    idx_equiv_atoms=idx_equiv_atoms,
                    ard=True,
                    nu=2.5,
                    lengthscale_prior=lengthscale_prior,
                    power=1,
                    representation="morse",
                    variance_constraint=constraints.Positive(),
                )
            )

            # Optional: manually initialize kernel parameters
            self.covar_module.base_kernel.lengthscale = [lengthscale_prior.mean] * 3
            self.covar_module.base_kernel.variance = [variance_prior.mean] * 3
            self.covar_module.outputscale = outputscale_prior.mean

            # Optional: fix the constant mean
            self.mean_module.constant = 4.0
            self.mean_module.raw_constant.requires_grad = False

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return distributions.MultivariateNormal(mean_x, covar_x)

Save this file as ``gpytorchwrapper/plugins/model_arh2ps0.py``. The wrapper will detect it automatically.

Custom Kernel Functions
------------------------

The wrapper supports several custom kernel functions, including:

PolyxMaternKernelPermInv
^^^^^^^^^^^^^^^^^^^^^^^^^

A permutation-invariant kernel that combines polynomial transformations with
Matérn-type behavior.

**Location:**  
``gpytorchwrapper/src/kernels/polyxmatern_kernel_perminv.py``

**Constructor Arguments:**

- ``n_atoms`` (``int``): Number of atoms in the system.
- ``idx_equiv_atoms`` (``List[List[int]]``): Groups of indices for symmetric atoms.
- ``ard`` (``bool``): Use automatic relevance determination (ARD).
- ``nu`` (``float``): Smoothness parameter of the Matérn kernel.
- ``lengthscale_prior`` (``gpytorch.priors.Prior``): Prior on lengthscales.
- ``power`` (``int``): Degree of the polynomial transformation.
- ``representation`` (``str``): Type of descriptor (e.g. ``"morse"``).
- ``variance_constraint`` (``gpytorch.constraints.Constraint``, optional): Constraint on kernel variance.

See :ref:`permutationally_invariant_kernels` for the theory behind these kernel functions.

Adding More Custom Kernels
----------------------------

To define and use new custom kernels:

1. Add your kernel implementation to ``gpytorchwrapper/src/kernels/``.
2. Import and instantiate the kernel in a plugin model file under ``gpytorchwrapper/plugins/model_*.py``.
3. Follow the GPyTorch API for compatibility.

Summary
--------

- Place models in ``gpytorchwrapper/plugins/``, named ``model_*.py``.
- Subclass ``ExactGP`` and ``GPyTorchModel``.
- Use custom kernels like ``PolyxMaternKernelPermInv`` for molecular systems containing invariant atoms.
- Optionally use priors and constraints for fine control over parameters.


