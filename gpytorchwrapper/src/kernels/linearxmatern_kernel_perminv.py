import math
from typing import Optional, Union

import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from linear_operator.operators import MatmulLinearOperator, RootLinearOperator
from torch import Tensor

from gpytorchwrapper.src.utils.input_transformer import xyz_to_invdist_torch
from gpytorchwrapper.src.utils.permutational_invariance import generate_permutations, generate_unique_distances


class LinearxMaternKernelPermInv(Kernel):
    has_lengthscale = True

    def __init__(
        self,
        n_atoms: int,
        idx_equiv_atoms: list[list[int]],
        select_dims: Tensor = None,
        nu: float = 2.5,
        variance_prior: Optional[Prior] = None,
        variance_constraint: Optional[Interval] = None,
        ard: bool = False,
        ard_expansion: list = None,
        **kwargs,
    ):
        if not ard:
            super().__init__(**kwargs)

            if self.ard_num_dims is not None:
                raise NotImplementedError(
                    "Regular ARD is not supported for LinearxMaternKernelPermInv. Set 'ard=True' instead and specify ard_expansion."
                )
        else:
            if ard_expansion is None:
                raise NotImplementedError("Please specify the expansion list for the ard lengthscale tensor.")

            #num_unique_distances = generate_unique_distances(n_atoms, idx_equiv_atoms)

            super().__init__(**kwargs)

        if nu not in {0.5, 1.5, 2.5}:
            raise NotImplementedError(
                "Please select one of the following nu values: {0.5, 1.5, 2.5}"
            )

        if self.active_dims is not None:
            raise NotImplementedError(
                "active_dims is not supported for LinearxMaternKernelPermInv. Please use select_dims instead."
            )

        if variance_constraint is None:
            variance_constraint = Positive()

        self.register_parameter(
            name="raw_variance",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        if variance_prior is not None:
            if not isinstance(variance_prior, Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got "
                    + type(variance_prior).__name__
                )
            self.register_prior(
                "variance_prior",
                variance_prior,
                lambda m: m.variance,
                lambda m, v: m._set_variance(v),
            )

        self.register_constraint("raw_variance", variance_constraint)

        self.select_dims = select_dims
        self.nu = nu
        self.idx_equiv_atoms = idx_equiv_atoms
        self.ard = ard
        self.ard_expansion = ard_expansion

        dims = torch.arange(0, n_atoms * 3).reshape(n_atoms, 3)
        self.dims = dims
        self.permutations = generate_permutations(idx_equiv_atoms)

    @property
    def variance(self) -> torch.Tensor:
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value: Union[float, torch.Tensor]):
        self._set_variance(value)

    def _set_variance(self, value: Union[float, torch.Tensor]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        self.initialize(
            raw_variance=self.raw_variance_constraint.inverse_transform(value)
        )

    def matern_kernel(self, x1, x2, diag, idx, **params):
        mean = x1.mean(dim=-2, keepdim=True)

        if self.ard:
            ard_lengthscale = self.lengthscale[0][self.ard_expansion]

            x1_ = (x1 - mean).div(ard_lengthscale)
            x2_ = (x2 - mean).div(ard_lengthscale)
        else:
            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)

        distance = self.covar_dist(x1_, x2_, diag=diag, **params)

        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (
                (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
            )
        else:
            raise NotImplementedError(
                "Please select one of the following nu values: {0.5, 1.5, 2.5}"
            )

        return constant_component * exp_component

    def linear_kernel(self, x1, x2, diag, last_dim_is_batch, **params):
        x1_ = x1 * self.variance.sqrt()
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLinearOperator when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLinearOperator(x1_)

        else:
            x2_ = x2 * self.variance.sqrt()
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

            prod = MatmulLinearOperator(x1_, x2_.transpose(-2, -1))

        if diag:
            return prod.diagonal(dim1=-1, dim2=-2)
        else:
            return prod

    def forward(
        self, x1, x2, diag=False, last_dim_is_batch: Optional[bool] = False, **params
    ):
        k_sum = 0
        num_perms = len(self.permutations)
        init_perm = self.permutations[0]

        for p in self.permutations:
            x2_perm = x2.clone()
            x2_perm[:, self.dims[init_perm, :].flatten()] = x2[
                :, self.dims[p, :].flatten()
            ]

            # Transform xyz coordinates to internuclear distances
            x1_interdist, idx = xyz_to_invdist_torch(x1, index=True)
            x2_perm_interdist = xyz_to_invdist_torch(x2_perm)

            if self.select_dims is not None:
                x1_interdist = torch.index_select(
                    x1_interdist, 1, torch.tensor(self.select_dims)
                )
                x2_perm_interdist = torch.index_select(
                    x2_perm_interdist, 1, torch.tensor(self.select_dims)
                )
                idx = idx[self.select_dims]

            k_linear = self.linear_kernel(
                x1_interdist, x2_perm_interdist, diag, last_dim_is_batch, **params
            )
            k_matern = self.matern_kernel(
                x1_interdist, x2_perm_interdist, diag, idx=idx, **params
            )

            k_sum += k_linear * k_matern
        return 1 / num_perms * k_sum
