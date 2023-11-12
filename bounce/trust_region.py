import math

import gin
import numpy as np
import torch


@gin.configurable
class TrustRegion:
    """
    Trust region object for the trust region algorithm.
    """

    def __init__(
        self,
        dimensionality: int,
        length_init_discrete: int = 40,
        length_init_continuous: float = 0.8,
    ):
        """

        Args:
            dimensionality: the dimensionality of the trust region
            length_init_discrete: the initial trust region length for discrete variables
            length_init_continuous: the initial trust region length for continuous variables
        """
        self.dimensionality = dimensionality

        self.length_init_discrete = min(length_init_discrete, dimensionality)
        self.length_init_continuous = length_init_continuous

        self.length_min_discrete = 1
        self.length_min_continuous = 0.5**7

        self.length_max_discrete = dimensionality
        self.length_max_continuous = 1.6

        self.length_discrete = self.length_init_discrete
        self.length_discrete_continuous = float(self.length_init_discrete)
        self.length_continuous = self.length_init_continuous

        self.terminated = False

    def reset(self):
        """
        Reset the trust region to its initial state.

        Returns:
            None

        """
        self.length_discrete = self.length_init_discrete
        self.length_discrete_continuous = float(self.length_init_discrete)
        self.length_continuous = self.length_init_continuous
        self.terminated = False


def update_tr_state(
    trust_region: TrustRegion,
    fx_next: torch.Tensor,
    fx_incumbent: torch.Tensor,
    adjustment_factor: np.double,
) -> None:
    """
    Update the trust region state based on the current function value and the incumbent function value.

    Args:
        trust_region: the trust region object
        fx_next: the function value of the next point
        fx_incumbent: the incumbent function value
        adjustment_factor: the adjustment factor for the trust region length

    Returns:
        None

    """
    if fx_next >= fx_incumbent - 1e-3 * math.fabs(fx_incumbent):
        trust_region.length_discrete_continuous = (
            trust_region.length_discrete_continuous * adjustment_factor.item()
        )
        trust_region.length_discrete = max(
            1, math.floor(trust_region.length_discrete_continuous)
        )

        trust_region.length_continuous = (
            trust_region.length_continuous * adjustment_factor.item()
        )
    else:
        trust_region.length_discrete_continuous = min(
            trust_region.length_discrete_continuous / adjustment_factor.item(),
            trust_region.length_max_discrete,
        )
        trust_region.length_discrete = max(
            1, math.floor(trust_region.length_discrete_continuous)
        )

        trust_region.length_continuous = min(
            trust_region.length_continuous / adjustment_factor.item(),
            trust_region.length_max_continuous,
        )

    discrete_terminated = (
        trust_region.length_discrete_continuous - 1e-4
        <= trust_region.length_min_discrete
    )
    continuous_terminated = (
        trust_region.length_continuous - 1e-4 <= trust_region.length_min_continuous
    )

    if discrete_terminated or continuous_terminated:
        trust_region.terminated = True
