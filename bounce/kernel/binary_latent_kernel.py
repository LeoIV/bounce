from random import shuffle

import gin
import torch
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, AdditiveKernel
from torch.quasirandom import SobolEngine

from bounce.projection import Parameter


@gin.configurable
class BinaryLatentSpaceKernel(Kernel):
    has_lengthscale = True

    def __init__(
            self,
            parameters: list[Parameter],
            n_latent_dims: int = 2,
            parameters_per_bin: int = 3,
            **kwargs
    ):
        super(BinaryLatentSpaceKernel, self).__init__(**kwargs)

        for parameter in parameters:
            assert parameter.is_binary, "BinaryLatentSpaceKernel only supports binary parameters."

        self.discrete_indices = torch.tensor(
            [i for i, p in enumerate(parameters) if p.is_discrete],
            dtype=torch.long
        )
        self.discrete_parameters = [p for p in parameters if p.is_discrete]

        self.continuous_indices = torch.tensor(
            [i for i, p in enumerate(parameters) if p.is_continuous],
            dtype=torch.long
        )
        self.continuous_parameters = [p for p in parameters if p.is_continuous]

        params_and_indices = [(p, i) for i, p in enumerate(parameters)]
        # shuffle
        shuffle(params_and_indices)

        params = [p for p, _ in params_and_indices]
        indices = [i for _, i in params_and_indices]

        # combine parameters in chunks of size parameters_per_bin
        self.parameters_chunks = [params[i:i + parameters_per_bin] for i in
                                  range(0, len(params), parameters_per_bin)]
        self.parameter_index_chunks = [indices[i:i + parameters_per_bin] for i in
                                       range(0, len(indices), parameters_per_bin)]

        self.n_realizations = list()

        for chunk in self.parameters_chunks:
            self.n_realizations.append(2 ** len(chunk))

        self.n_realizations = torch.tensor(self.n_realizations)

        self.n_latent_dims = n_latent_dims
        self.n_continuous_dims = len(self.continuous_indices)
        self.n_discrete_dims = len(self.discrete_indices)
        self.register_constraint("raw_lengthscale", GreaterThan(1e-6))

        active_dims = torch.split(torch.arange(self.n_latent_dims * len(self.parameters_chunks)), self.n_latent_dims)

        base_kernels = [MaternKernel(nu=2.5, active_dims=active_dim, lengthscale=False)
                        for active_dim in active_dims]
        for bk in base_kernels:
            bk.lengthscale = 1
            bk.raw_lengthscale.requires_grad = False

        # TODO necessary and what about continuous dims?
        self.base_kernel_additive = AdditiveKernel(
            *base_kernels
        )
        self.base_kernel_multiplicative = ProductKernel(
            *base_kernels
        )

        # check discrete and continuous dims are disjoint
        assert len(
            set(self.discrete_indices).intersection(set(self.continuous_indices))
        ) == 0, "Discrete and continuous dims must be disjoint."

        self.params = dict()

        for bin_idx, n_realizations in enumerate(self.n_realizations):
            constraint = Interval(0.0, 1.0)

            # register parameters for free latent dimensions
            self.register_parameter(
                name=f"raw_free_latent_dims_{bin_idx}",
                parameter=torch.nn.Parameter(
                    constraint.inverse_transform(
                        SobolEngine(dimension=self.n_latent_dims, scramble=True).draw(n_realizations)
                    )
                )
            )
            # register constraint for free latent dimensions
            self.register_constraint(f"raw_free_latent_dims_{bin_idx}", constraint)

    def get_free_latent_dims(self, bin_idx: int):
        constraint = getattr(self, f"raw_free_latent_dims_{bin_idx}_constraint")
        parameter = getattr(self, f"raw_free_latent_dims_{bin_idx}")
        return constraint.transform(parameter)

    def get_non_zero_latent_dims(self, bin_idx: int):
        constraint = getattr(self, f"raw_non_zero_latent_dims_{bin_idx}_constraint")
        parameter = getattr(self, f"raw_non_zero_latent_dims_{bin_idx}")
        return constraint.transform(parameter)

    def map_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map categorical variables to latent dimensions.

        Parameters:
            x: torch.Tensor (..., n_dims) - Input data.

        Returns:
            torch.Tensor (..., n_continuous_dims + latent_dim * n_categorical_dims) - Mapped data.
        """
        # map categorical variables to latent dimensions
        x_discrete = x[..., self.discrete_indices]
        x_continuous = x[..., self.continuous_indices]

        x_latent = torch.empty(
            (*x.shape[:-1], self.n_continuous_dims + self.n_latent_dims * len(self.parameters_chunks)),
            dtype=torch.float64,
            device=x.device
        )
        x_latent[..., :self.n_continuous_dims] = x_continuous

        latent_mapping = self.latent_mapping
        latent_mapping = [lm.to(x.device) for lm in latent_mapping]

        for i_discrete in range(len(self.n_realizations)):
            _latent_mapping = latent_mapping[i_discrete]

            # get discrete indices
            _x_discrete = torch.sum(
                x_discrete[..., self.parameter_index_chunks[i_discrete]].long() * torch.pow(2,
                                                                                            torch.arange(len(
                                                                                                self.parameter_index_chunks[
                                                                                                    i_discrete]),
                                                                                                device=x.device)),
                dim=-1)
            # get latent dimensions
            _x_latent = _latent_mapping[_x_discrete, ...]

            # set latent dimensions in x_latent
            x_latent[..., self.n_continuous_dims + i_discrete * self.n_latent_dims:self.n_continuous_dims + (
                    i_discrete + 1) * self.n_latent_dims] = _x_latent
            pass

        return x_latent

    @property
    def latent_mapping(self) -> list[torch.Tensor]:
        latent_dims = [torch.empty(
            (n_realizations, self.n_latent_dims),
            dtype=torch.float64,
        ) for n_realizations in self.n_realizations]

        for i_discrete in range(len(latent_dims)):
            # get free latent dimensions
            free_latent_dims = self.get_free_latent_dims(i_discrete)
            # add to raw discrete
            latent_dims[i_discrete] = free_latent_dims
        return latent_dims

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            diag: bool = False,
            **params
    ) -> torch.Tensor:
        assert x1.shape[
                   -1] == self.n_discrete_dims + self.n_continuous_dims, "Input dimension mismatch. Expected {}, got {}.".format(
            self.n_discrete_dims + self.n_continuous_dims, x1.shape[-1]
        )

        x1_latent = self.map_to_latent(x1)
        x1_ = x1_latent.div(self.lengthscale)

        if x1 is x2:
            x2_ = x1_
        else:
            x2_latent = self.map_to_latent(x2)
            x2_ = x2_latent.div(self.lengthscale)

        return 0 * self.base_kernel_additive.forward(x1_, x2_, diag=diag, **params) + \
            1 * self.base_kernel_multiplicative.forward(x1_, x2_, diag=diag, **params)
