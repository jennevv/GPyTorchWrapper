from typing import Optional, Union

import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels.rbf_kernel import postprocess_rbf
from gpytorch.priors import Prior
from linear_operator.operators import MatmulLinearOperator, RootLinearOperator

from gpytorchwrapper.src.kernels.perminv_kernel import PermInvKernel
from gpytorchwrapper.src.utils.input_transformer import xyz_to_dist_torch


class LinearxRBFKernelPermInv(PermInvKernel):
    has_lengthscale = True

    def __init__(
        self,
        n_atoms: int,
        idx_equiv_atoms: list[list[int]],
        select_dims: list[int] = None,
        ard: bool = False,
        representation: str = "invdist",
        variance_prior: Optional[Prior] = None,
        variance_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        """
        Initialize the LinearxRBFKernelPermInv kernel, a product kernel of a linear kernel and an RBF kernel.

        Parameters
        ----------
        n_atoms : int
            Number of atoms in the molecule or structure.
        idx_equiv_atoms : list of list of int
            Groups of indices indicating equivalent atoms under permutations.
        select_dims : list of int, optional
            Dimensions to select from the distance representation.
        ard : bool, default=False
            If True, use automatic relevance determination (ARD).
        representation : str, default="invdist"
            The type of representation to use for distances, choose from:
                `invdist` for inverse distances
                `morse` for features exp(-r_ij)
        variance_prior : gpytorch.priors.Prior, optional
            Prior distribution for the variance parameter.
        variance_constraint : gpytorch.constraints.Interval, optional
            Constraint for the variance parameter.
        **kwargs
            Additional keyword arguments for the base class.

        Raises
        ------
        NotImplementedError
            If `active_dims` is provided in `kwargs`, which is not supported.
        TypeError
            If `variance_prior` is not an instance of `gpytorch.priors.Prior`.
        """
        super().__init__(
            n_atoms=n_atoms,
            idx_equiv_atoms=idx_equiv_atoms,
            select_dims=select_dims,
            ard=ard,
            **kwargs,
        )

        if self.active_dims is not None:
            raise NotImplementedError(
                "Keyword active_dims is not supported for LinearxRBFKernelPermInv. Please use select_dims instead."
            )

        if variance_constraint is None:
            variance_constraint = Positive()

        self.register_parameter(
            name="raw_variance",
            parameter=torch.nn.Parameter(
                torch.zeros(
                    *self.batch_shape,
                    1,
                    1 if self.ard_num_dims is None else self.ard_num_dims,
                )
            ),
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

        self.representation = representation

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

    def rbf_kernel(self, x1, x2, diag, **params):
        mean = x1.mean(dim=-2, keepdim=True)

        if self.ard:
            perminv_ard_lengthscale = self.lengthscale.clone()[0][
                self.ard_expansion
            ].unsqueeze(0)
            x1_ = (x1 - mean).div(perminv_ard_lengthscale)
            x2_ = (x2 - mean).div(perminv_ard_lengthscale)
        else:
            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)

        return postprocess_rbf(
            self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)
        )

    def linear_kernel(self, x1, x2, diag, last_dim_is_batch, **params):
        if self.ard:
            perminv_ard_variance = self.variance.clone()[0][
                self.ard_expansion
            ].unsqueeze(0)
            x1_ = x1 * perminv_ard_variance.sqrt()
        else:
            x1_ = x1 * self.variance.sqrt()

        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLinearOperator when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLinearOperator(x1_)

        else:
            if self.ard:
                x2_ = x2 * perminv_ard_variance.sqrt()
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

        x1_dist = xyz_to_dist_torch(x1, representation=self.representation)
        x2_dist = (
            xyz_to_dist_torch(x2, representation=self.representation)
            if not torch.equal(x1, x2)
            else x1_dist.clone()
        )
        if self.select_dims is not None:
            select_dims_tensor = torch.tensor(self.select_dims)
            x1_dist = torch.index_select(x1_dist, 1, select_dims_tensor)

        for perm in self.permutations:
            x2_dist_perm = x2_dist.clone()
            x2_dist_perm[:, init_perm] = x2_dist[:, perm]

            if self.select_dims is not None:
                x2_dist_perm = torch.index_select(x2_dist_perm, 1, select_dims_tensor)

            k_linear = self.linear_kernel(
                x1_dist, x2_dist_perm, diag, last_dim_is_batch, **params
            )
            k_rbf = self.rbf_kernel(x1_dist, x2_dist_perm, diag, **params)

            k_sum += k_linear * k_rbf
        return 1 / num_perms * k_sum
