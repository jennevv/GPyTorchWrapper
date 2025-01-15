from typing import Optional, Union

import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from linear_operator.operators import RootLinearOperator, MatmulLinearOperator
from torch import Tensor

from gpytorchwrapper.src.utils.input_transformer import xyz_to_invdist_torch
from gpytorchwrapper.src.utils.permutational_invariance import generate_permutations


class LinearKernelPermInv(Kernel):
    def __init__(
            self, n_atoms: int, idx_equiv_atoms: list[list[int]], select_dims: Tensor = None,
            variance_prior: Optional[Prior] = None, variance_constraint: Optional[Interval] = None, **kwargs):

        super().__init__(**kwargs)

        if variance_constraint is None:
            variance_constraint = Positive()

        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        if variance_prior is not None:
            if not isinstance(variance_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(variance_prior).__name__)
            self.register_prior("variance_prior", variance_prior, lambda m: m.variance, lambda m, v: m._set_variance(v))

        self.register_constraint("raw_variance", variance_constraint)

        dims = torch.arange(0, n_atoms * 3).reshape(n_atoms, 3)
        self.dims = dims
        self.idx_equiv_atoms = idx_equiv_atoms
        self.select_dims = select_dims
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
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: Optional[bool] = False,
                last_dim_is_batch: Optional[bool] = False, **params) -> torch.Tensor:
        k_sum = 0
        num_perms = len(self.permutations)
        init_perm = self.permutations[0]

        for p in self.permutations:
            x2_perm = x2.clone()
            x2_perm[:, self.dims[init_perm, :].flatten()] = x2[:, self.dims[p, :].flatten()]

            # Transform xyz coordinates to internuclear distances
            x1_interdist = xyz_to_invdist_torch(x1)
            x2_perm_interdist = xyz_to_invdist_torch(x2_perm)

            if self.select_dims is not None:
                x1_interdist = x1_interdist.index_select(-1, self.select_dims)
                x2_perm_interdist = x2_perm_interdist.index_select(-1, self.select_dims)

            x1_ = x1_interdist
            x2_ = x2_perm_interdist

            if last_dim_is_batch:
                x1_ = x1_interdist.transpose(-1, -2).unsqueeze(-1)
                x2_ = x2_perm_interdist.transpose(-1, -2).unsqueeze(-1)

            if x1.size() == x2.size() and torch.equal(x1, x2):
                prod = RootLinearOperator(x1_)
            else:
                prod = MatmulLinearOperator(x1_, x2_.transpose(-2, -1))

            if diag:
                k_sum += prod.diagonal(dim1=-1, dim2=-2)
            else:
                k_sum += prod

        return 1 / num_perms * k_sum
