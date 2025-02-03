from typing import Optional, Any

import torch
from gpytorch.constraints import Interval
from gpytorch.means import Mean
from gpytorch.priors import Prior
from scipy.cluster.hierarchy import weighted
from torch import Tensor

from gpytorchwrapper.src.utils.input_transformer import xyz_to_dist_torch


class SwitchingConstantMean(Mean):
    r"""
    A (non-zero) constant prior mean function, i.e.:

    .. math::
        \mu(\mathbf x) = C

    where :math:`C` is a learned constant.

    :param constant_prior: Prior for constant parameter :math:`C`.
    :type constant_prior: ~gpytorch.priors.Prior, optional
    :param constant_constraint: Constraint for constant parameter :math:`C`.
    :type constant_constraint: ~gpytorch.priors.Interval, optional
    :param batch_shape: The batch shape of the learned constant(s) (default: []).
    :type batch_shape: torch.Size, optional

    :var torch.Tensor constant: :math:`C` parameter
    """

    def __init__(
        self,
        num_atoms ,
        constant_prior: Optional[Prior] = None,
        constant_constraint: Optional[Interval] = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ):
        super(SwitchingConstantMean, self).__init__()

        num_dist = num_atoms * (num_atoms - 1) // 2

        self.batch_shape = batch_shape
        self.register_parameter(name="raw_constant", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_dist)))
        if constant_prior is not None:
            self.register_prior("mean_prior", constant_prior, self._constant_param, self._constant_closure)
        if constant_constraint is not None:
            self.register_constraint("raw_constant", constant_constraint)

    @property
    def constant(self):
        return self._constant_param(self)

    @constant.setter
    def constant(self, value):
        self._constant_closure(self, value)

    # We need a getter of this form so that we can pickle ConstantMean modules with a mean prior, see PR #1992
    def _constant_param(self, m):
        if hasattr(m, "raw_constant_constraint"):
            return m.raw_constant_constraint.transform(m.raw_constant)
        return m.raw_constant

    # We need a setter of this form so that we can pickle ConstantMean modules with a mean prior, see PR #1992
    def _constant_closure(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_constant)

        if hasattr(m, "raw_constant_constraint"):
            m.initialize(raw_constant=m.raw_constant_constraint.inverse_transform(value))
        else:
            m.initialize(raw_constant=value)

    @staticmethod
    def sigmoid(x):
        return x / (1 + torch.exp(-x))

    def weighted_constant(self, dist):
        num_channels = dist.shape[1]

        weights = self.sigmoid(dist)

        total_weight = torch.sum(weights, dim=1)
        if total_weight > 0:
            weights /= total_weight.reshape(-1,1)
        else:
            # Fallback: if all weights are zero (should not happen with sigmoid), assign equal weight.
            weights = torch.ones(num_channels) / num_channels

        # Compute the weighted mean using the channel_means
        weighted_constant = torch.sum(self.constants * weights, dim=1)

        return weighted_constant

    def forward(self, x):
        dist = xyz_to_dist_torch(x)
        w_constant = self.weighted_constant(dist)

        constant = self.w_constant.unsqueeze(-1)  # *batch_shape x 1
        return constant.expand(torch.broadcast_shapes(w_constant.shape, dist.shape[:-1]))
