from typing import Optional

import numpy as np
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, MaternKernel
from gpytorch.priors import Prior


class MixtureKernel(Kernel):
    has_lengthscale = True

    def __init__(
        self,
        discrete_dims: list[int],
        continuous_dims: list[int],
        lamda: Optional[float] = 0.5,
        discrete_lengthscale_prior: Prior = None,
        continuous_lengthscale_prior: Prior = None,
        discrete_lengthscale_constraint: Interval = None,
        continuous_lengthscale_constraint: Interval = None,
        discrete_ard: bool = False,
        continuous_ard: bool = True,
        **kwargs
    ):
        super(MixtureKernel, self).__init__(has_lengthscale=True, **kwargs)
        # check discrete and continuous dims are disjoint
        assert (
            len(set(discrete_dims).intersection(set(continuous_dims))) == 0
        ), "Discrete and continuous dims must be disjoint."
        self.discrete_dims = discrete_dims
        self.continuous_dims = continuous_dims
        self.optimize_lamda = lamda is None
        self.fixed_lamda = lamda if not self.optimize_lamda else None
        self.discrete_ard = discrete_ard
        self.continuous_ard = continuous_ard

        self.discrete_dims_np = np.asarray(discrete_dims)
        self.continuous_dims_np = np.asarray(continuous_dims)

        self.register_parameter("raw_lamda", torch.nn.Parameter(torch.ones(1)))
        self.register_constraint("raw_lamda", Interval(0, 1))

        self.discrete_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=len(discrete_dims) if discrete_ard else None,
            lengthscale_constraint=discrete_lengthscale_constraint,
            lengthscale_prior=discrete_lengthscale_prior,
        )
        self.continuous_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=len(continuous_dims) if continuous_ard else None,
            lengthscale_constraint=continuous_lengthscale_constraint,
            lengthscale_prior=continuous_lengthscale_prior,
        )

    @property
    def lamda(self):
        if self.optimize_lamda:
            return self.raw_lamda_constraint.transform(self.raw_lamda)
        else:
            return self.fixed_lamda

    @lamda.setter
    def lamda(self, value: float):
        self._set_lamda(value)

    def _set_lamda(self, value: float):
        if self.optimize_lamda:
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value).to(self.raw_lamda)
            self.initialize(
                raw_lamda=self.raw_lamda_constraint.inverse_transform(value)
            )
        else:
            # Manually restrict the value of lamda between 0 and 1.
            if value <= 0:
                self.fixed_lamda = 0.0
            elif value >= 1:
                self.fixed_lamda = 1.0
            else:
                self.fixed_lamda = value

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        x1_continuous: Optional[torch.Tensor] = None,
        x2_continuous: Optional[torch.Tensor] = None,
        **params
    ) -> torch.Tensor:
        if x1_continuous is None and x2_continuous is None:
            assert x1.shape[-1] == len(self.discrete_dims) + len(
                self.continuous_dims
            ), "Input dimension mismatch. Expected {}, got {}.".format(
                len(self.discrete_dims) + len(self.continuous_dims), x1.shape[-1]
            )

            x1_discrete = x1[..., self.discrete_dims_np]
            x2_discrete = x2[..., self.discrete_dims_np]
            x1_continuous = x1[..., self.continuous_dims_np]
            x2_continuous = x2[..., self.continuous_dims_np]
        else:
            assert x1.shape[1] == len(
                self.discrete_dims
            ), "Input dimension mismatch. Expected {}, got {}.".format(
                len(self.discrete_dims), x1.shape[1]
            )
            assert x1_continuous.shape[1] == len(
                self.continuous_dims
            ), "Input dimension mismatch. Expected {}, got {}.".format(
                len(self.continuous_dims), x1_continuous.shape[1]
            )
            assert x2.shape[1] == len(
                self.discrete_dims
            ), "Input dimension mismatch. Expected {}, got {}.".format(
                len(self.discrete_dims), x2.shape[1]
            )
            assert x2_continuous.shape[1] == len(
                self.continuous_dims
            ), "Input dimension mismatch. Expected {}, got {}.".format(
                len(self.continuous_dims), x2_continuous.shape[1]
            )
            x1_discrete = x1
            x2_discrete = x2

        k_discrete = self.discrete_kernel(x1_discrete, x2_discrete, diag=diag, **params)
        k_continuous = self.continuous_kernel(
            x1_continuous, x2_continuous, diag=diag, **params
        )

        return (1 - self.lamda) * (
            k_discrete + k_continuous
        ) + self.lamda * k_discrete * k_continuous
