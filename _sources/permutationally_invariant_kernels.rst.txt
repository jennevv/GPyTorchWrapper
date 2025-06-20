.. permutationally_invariant_kernels:

=================================
Permutationally Invariant Kernels
=================================

Permutational invariance is both a physical constraint and an inherent property of the potential energy surface: exchanging two identical atoms does not change the potential energy. Several strategies exist to enforce this property. For example, sorting the internuclear distances of like bonds works, but that approach is not analytically differentiable [Dral2017]_. In contrast, permutationally invariant kernels, as described by Bartók and Csányi, are fully differentiable [Bartok2015]_. These kernels are therefore well suited for automatic differentiation frameworks like PyTorch to model a potential energy surface for molecular dynamics.

To enforce permutational symmetry in a kernel function, start with any kernel :math:`K(\mathbf{x}, \mathbf{x'})` with vector inputs :math:`\mathbf{x}` and :math:`\mathbf{x'}`. Define a new kernel by summing over all permutations :math:`\hat{P}` of :math:`\mathbf{x'}`:

.. math::

    K'(\mathbf{x}, \mathbf{x'}) = \sum_{\hat{P}} K(\mathbf{x}, \hat{P}\mathbf{x'})\ .

In the implementation under ``src/kernels/``, this idea is followed but with a simplified normalization step. Instead of the protocol suggested by Bartók and Csányi, the result is divided by the number of permutations. This ensures that hyperparameter searches yield magnitudes comparable to those of non-permutational kernels.

Derivatives with Respect to Atomic Coordinates
----------------------------------------------

These kernels allow automatic computation of derivatives via AutoGrad. When the model output represents energies, those derivatives are the forces needed for molecular dynamics simulations.

However, using atomic coordinates directly as input features is suboptimal: their absolute values do not correlate directly with the potential energy, and they do not respect translational or rotational invariance. Instead, we train on interatomic distances or a related descriptor, which does respect those invariances.

Transforming to distance-based features introduces a challenge because it obscures the direct relationship to atomic coordinates and makes analytical derivatives harder. To address this, the transformation is performed inside the kernel’s ``forward`` method. This way, we train on internuclear distances but still rely on AutoGrad to produce exact forces in atomic coordinates — the best of both worlds.

Anisotropic Lengthscales (ARD)
------------------------------

Each input dimension’s lengthscale must respect the same permutational symmetry as the output. In practice, this means any bonds involving permutationally equivalent atoms must share the same lengthscale for the model to remain valid.

This behavior is implemented in the ``PermInvKernel`` base class, from which all other permutationally invariant kernels inherit. Note that GPyTorch’s built-in ``ard_num_dims`` option does not work with these kernels. Instead, ARD must be enabled by setting the Boolean-valued ``ard`` argument when constructing the kernel.

.. warning::

    If you select a ``PermInvKernel`` in your model definition, make sure to set ``debug: False`` in the ``training_conf`` section of your configuration YAML.

References
----------

.. [Dral2017] Dral, P. O.; Owens, A.; Yurchenko, S. N.; Thiel, W. Structure-Based Sampling and Self-Correcting Machine Learning for Accurate Calculations of Potential Energy Surfaces and Vibrational Levels. *J. Chem. Phys.* **2017**, *146* (24), 244108. https://doi.org/10.1063/1.4989536

.. [Bartok2015] Bartók, A. P.; Csányi, G. Gaussian Approximation Potentials: A Brief Tutorial Introduction. *Int. J. Quantum Chem.* **2015**, *115* (16), 1051–1057. https://doi.org/10.1002/qua.24927

