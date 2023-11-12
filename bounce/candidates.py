import logging
import warnings
from typing import Optional

import gin
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels import MaternKernel

from bounce.kernel.categorical_mixture import MixtureKernel
from bounce.neighbors import hamming_distance, hamming_neighbors_within_tr
from bounce.projection import AxUS
from bounce.trust_region import TrustRegion
from bounce.util.benchmark import ParameterType


@gin.configurable
def create_candidates_discrete(
    x_scaled: torch.Tensor,
    fx_scaled: torch.Tensor,
    acquisition_function: Optional[ExpectedImprovement],
    model: SingleTaskGP,
    axus: AxUS,
    trust_region: TrustRegion,
    device: str,
    batch_size: int = 1,
    x_bests: Optional[list[torch.Tensor]] = None,
    add_spray_points: bool = True,
    sampler: Optional[SobolQMCNormalSampler] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Create candidate points for the next batch.

    Args:
        model: The current GP model
        batch_size: The number of candidate points to create
        x_scaled: The current points in the trust region
        fx_scaled: The function values at the current points
        acquisition_function: The approximate posterior samples
        axus: The current AxUS embedding for the trust region
        trust_region: The current trust region state
        device: The device to use ('cpu' or 'cuda')
        x_bests: The center of the trust region, should be in [0, 1]^d
        add_spray_points: Whether to add spray points (points within hamming distance 1 of the center)
        sampler: The sampler to use for the acquisition function


    Returns:
        The candidate points, the function values at the candidate points, the new GP hyperparameters, and the new trust region state

    """

    # Get the indices of the continuous parameters
    indices_not_to_optimize = torch.tensor(
        [i for b, i in axus.bins_and_indices_of_type(ParameterType.CONTINUOUS)]
    )

    # Find the center of the trust region
    x_centers = torch.clone(x_scaled[fx_scaled.argmin(), :]).detach()
    # x_center should be in [0, 1]^d at this point
    x_centers = torch.repeat_interleave(x_centers.unsqueeze(0), batch_size, dim=0)
    if x_bests is not None:
        # replace
        x_centers[:, indices_not_to_optimize] = (
            x_bests[:, indices_not_to_optimize] + 1
        ) / 2

    # define the number of candidates as in the TuRBO paper
    n_candidates = min(5000, max(2000, 200 * axus.target_dim))

    x_batch_return = torch.zeros(
        (batch_size, axus.target_dim), dtype=x_scaled.dtype, device=x_scaled.device
    )
    fx_batch_return = torch.zeros(
        (batch_size, 1), dtype=fx_scaled.dtype, device=fx_scaled.device
    )

    for batch_index in range(batch_size):
        _acquisition_function = acquisition_function
        if acquisition_function is None:
            assert (
                sampler is not None
            ), "Either acquisition_function or sampler must be provided"
            x_pending = x_batch_return[:batch_index, :] if batch_index > 0 else None
            _acquisition_function = qExpectedImprovement(
                model=model,
                best_f=(-fx_scaled).max().item(),
                sampler=sampler,
                X_pending=x_pending,
            )

        def ts(x: torch.Tensor, batch_index: int):
            """
            Get the approximate posterior sample of a specific batch index.

            Args:
                x: The points to evaluate the posterior sample at
                batch_index: The index of the batch to evaluate the posterior sample for

            Returns:
                The approximate posterior sample at the given points

            """

            return -_acquisition_function(x.unsqueeze(1))

        x_candidates = sample_initial_points_discrete(
            x_center=x_centers[batch_index],
            axus=axus,
            tr_length=trust_region.length_discrete,
            n_initial_points=n_candidates,
        )

        if add_spray_points:
            x_spray = hamming_neighbors_within_tr(
                x_center=x_centers[batch_index],
                x=x_centers[batch_index],
                tr_length=trust_region.length_discrete,
                axus=axus,
            )
            x_candidates = torch.vstack((x_candidates, x_spray))

        # Evaluate the acquisition function for all candidates
        with torch.no_grad():
            candidate_acquisition_values = ts(x_candidates, batch_index=batch_index)
        # Find the top k candidates with the highest acquisition function value
        top_k_candidate_indices = torch.topk(
            candidate_acquisition_values,
            k=min(3, len(candidate_acquisition_values)),
            largest=False,
        )[1]
        # Start local search
        best_posterior_value = torch.inf
        x_best = None

        for top_index in top_k_candidate_indices:
            x_candidate = x_candidates[top_index, :].clone().unsqueeze(0)

            posterior_value_k = candidate_acquisition_values[top_index].item()

            if posterior_value_k < best_posterior_value:
                best_posterior_value = posterior_value_k
                x_best = x_candidate
            while True:
                x_start_neighbors = hamming_neighbors_within_tr(
                    x_center=x_centers[batch_index],
                    x=x_candidate,
                    tr_length=trust_region.length_discrete,
                    axus=axus,
                )

                # remove rows from x_start_neighbors that are already in self.x (which is a 2d tensor of shape (n, d))
                for x_eval in x_scaled.to(device=device):
                    x_start_neighbors = x_start_neighbors[
                        ~torch.all(x_start_neighbors == x_eval, dim=1)
                    ]

                if x_start_neighbors.numel() == 0:
                    # no neighbors left, continue with next top candidate
                    break

                with torch.no_grad():
                    neighbors_acq_val = ts(x_start_neighbors, batch_index=batch_index)

                if (
                    len(neighbors_acq_val) > 0
                    and torch.min(neighbors_acq_val) < posterior_value_k
                ):
                    x_candidate = x_start_neighbors[torch.argmin(neighbors_acq_val)]
                    posterior_value_k = torch.min(neighbors_acq_val).item()
                else:
                    # could not find a better neighbor, continue with next top candidate
                    break
                if posterior_value_k < best_posterior_value:
                    best_posterior_value = posterior_value_k
                    x_best = x_candidate.unsqueeze(0)
        if x_best is None:
            warnings.warn(
                "Could not find a better point than the center of the trust region"
            )
            # choose random point
            x_best = x_centers[batch_index].unsqueeze(0)
        # repeat x_cand batch_size many times
        x_batch_return[batch_index, :] = x_best.squeeze()
        fx_batch_return[batch_index, :] = best_posterior_value

    assert len(indices_not_to_optimize) == 0 or torch.any(
        x_centers[:, indices_not_to_optimize].squeeze()
        == x_batch_return[:, indices_not_to_optimize].squeeze()
    ), "x_ret should not be optimized at indices_not_to_optimize"

    # transform to [-1, 1], was [0, 1]
    x_batch_return = x_batch_return * 2 - 1

    tr_state = {
        "center": x_scaled[fx_scaled.argmin(), :].detach().cpu().numpy().reshape(1, -1),
        "length": np.array([trust_region.length_discrete]),
    }

    return x_batch_return, fx_batch_return.reshape(batch_size), tr_state


def create_candidates_continuous(
    x_scaled: torch.Tensor,
    fx_scaled: torch.Tensor,
    acquisition_function: Optional[ExpectedImprovement],
    model: SingleTaskGP,
    axus: AxUS,
    trust_region: TrustRegion,
    device: str,
    batch_size: int,
    indices_to_optimize: Optional[torch.Tensor] = None,
    x_bests: Optional[list[torch.Tensor]] = None,
    sampler: Optional[SobolQMCNormalSampler] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Create candidate points for the next batch.

    Args:
        x_scaled: The current points in the trust region
        fx_scaled: The function values at the current points
        acquisition_function: The acquisition function to use
        model: The current GP model
        axus: The current AxUS embedding for the trust region
        trust_region: The current trust region state
        device: The device to use ('cpu' or 'cuda')
        indices_to_optimize: The indices of the candidate points to optimize (in case of mixed spaces)
        x_bests: The center of the trust region
        batch_size: int

    Returns:
        The candidate points, the function values at the candidate points, the new GP hyperparameters, and the new trust region state

    """

    if indices_to_optimize is None:
        indices_to_optimize = torch.arange(axus.target_dim)
    indices_not_to_optimize = torch.arange(axus.target_dim)[
        ~torch.isin(torch.arange(axus.target_dim), indices_to_optimize)
    ]

    x_centers = torch.clone(x_scaled[fx_scaled.argmin(), :]).detach()
    # repeat x_centers batch_size many times
    x_centers = torch.repeat_interleave(x_centers.unsqueeze(0), batch_size, dim=0)

    if x_bests is not None:
        x_centers[:, indices_not_to_optimize] = (
            x_bests[:, indices_not_to_optimize] + 1
        ) / 2

    assert len(x_centers.shape) == 2, "x_center should be a 2d tensor"

    fx_argmins = torch.zeros(batch_size, dtype=torch.long, device=device)
    fx_mins = torch.zeros(batch_size, dtype=torch.double, device=device)
    x_cand_downs = torch.zeros(
        (batch_size, axus.target_dim), dtype=torch.double, device=device
    )
    for batch_index in range(batch_size):
        x_center = x_centers[batch_index, :]

        if isinstance(model.covar_module.base_kernel, MixtureKernel):
            weights = model.covar_module.base_kernel.continuous_kernel.lengthscale.detach().squeeze(
                0
            )
        elif isinstance(model.covar_module.base_kernel, MaternKernel):
            weights = model.covar_module.base_kernel.lengthscale.detach().squeeze(0)
        else:
            raise NotImplementedError(
                "Only MixtureKernel and MaternKernel are supported"
            )
        weights /= weights.mean()
        weights /= torch.prod(torch.pow(weights, 1 / len(weights)))
        _x_center = x_center[indices_to_optimize]
        _tr_lb = torch.clip(
            _x_center - trust_region.length_continuous * weights / 2, 0, 1
        )
        _tr_ub = torch.clip(
            _x_center + trust_region.length_continuous * weights / 2, 0, 1
        )
        tr_lb = torch.zeros(axus.target_dim, dtype=torch.double, device=device)
        tr_ub = torch.ones(axus.target_dim, dtype=torch.double, device=device)
        tr_lb[indices_to_optimize] = _tr_lb
        tr_ub[indices_to_optimize] = _tr_ub

        _acquisition_function = acquisition_function
        if acquisition_function is None:
            assert (
                sampler is not None
            ), "Either acquisition_function or sampler must be provided"
            x_pending = x_cand_downs[:batch_index, :] if batch_index > 0 else None
            _acquisition_function = qExpectedImprovement(
                model=model,
                best_f=(-fx_scaled).max().item(),
                sampler=sampler,
                X_pending=x_pending,
            )

        # EI-based acquisition function
        x_cand_down = optimize_acqf(
            acq_function=_acquisition_function,
            bounds=torch.stack([tr_lb, tr_ub], dim=0),
            q=1,
            fixed_features={
                i: x_center[i].item() for i in indices_not_to_optimize.tolist()
            },
            num_restarts=10,
            raw_samples=512,
        )
        x_cand_down, y_cand_down = x_cand_down
        x_cand_downs[batch_index, :] = x_cand_down
        fx_argmins[batch_index] = -y_cand_down

    tr_state = {
        "center": x_scaled[fx_scaled.argmin(), :].detach().cpu().numpy().reshape(1, -1),
        "length": np.array([trust_region.length_continuous]),
        "lb": tr_lb.detach().cpu().numpy(),
        "ub": tr_ub.detach().cpu().numpy(),
    }

    return x_cand_downs * 2 - 1, fx_mins.reshape(batch_size), tr_state


def sample_initial_points_discrete(
    x_center: torch.Tensor,
    tr_length: torch.Tensor,
    axus: AxUS,
    n_initial_points: int,
) -> torch.Tensor:
    """
    Sample initial points for the discrete parameters

    Args:
        x_center: the center of the trust region
        tr_length: the length of the trust region
        axus: the AxUS embedding
        n_initial_points: the number of initial points to sample

    Returns:
        x_cand: the sampled initial points

    """
    discrete_parameter_types = [
        pt for pt in ParameterType if pt != ParameterType.CONTINUOUS
    ]

    # copy x_center n_initial_points times
    x_cand = torch.repeat_interleave(x_center.unsqueeze(0), n_initial_points, dim=0)

    for parameter_type in discrete_parameter_types:
        if axus.n_bins_of_type(parameter_type) == 0:
            # No parameters of this type
            continue
        if parameter_type == ParameterType.BINARY:
            indices = torch.tensor(
                [i for b, i in axus.bins_and_indices_of_type(parameter_type)]
            )
            # draw min(tr_length, len(indices)) indices for each candidate
            indices_for_cand = torch.tensor(
                np.array(
                    [
                        np.random.choice(
                            indices, min(tr_length - 1, len(indices)), replace=False
                        )
                        for _ in range(n_initial_points)
                    ]
                ),
                dtype=torch.long,
                device=x_cand.device,
            )
            # draw values for each index
            values_for_cand = torch.randint(
                0,
                2,
                (n_initial_points, len(indices_for_cand[0])),
                dtype=x_cand.dtype,
                device=x_cand.device,
            )
            # set values for each candidate
            x_cand = x_cand.scatter_(1, indices_for_cand, values_for_cand)
        elif parameter_type == ParameterType.CATEGORICAL:
            indicess = [i for b, i in axus.bins_and_indices_of_type(parameter_type)]
            if len(indicess) > tr_length:
                index_setss = [
                    np.random.choice(
                        np.arange(len(indicess)),
                        min(tr_length, len(indicess)),
                        replace=False,
                    )
                    for _ in range(n_initial_points)
                ]
                for i, index_sets in enumerate(index_setss):
                    index_sets = [indicess[i] for i in index_sets]
                    # set x_cand to 0 for each index
                    x_cand[i, torch.cat(index_sets)] = 0
                    if True:  # else:
                        # this is the expensive part
                        for indices in index_sets:
                            # set one index to 1
                            x_cand[i, np.random.choice(indices)] = 1
            else:
                for indices in indicess:
                    # set x_cand to 0 for each index
                    x_cand[:, indices] = 0
                    # sample n_initial_points indices
                    indices_for_cand = np.random.choice(indices, n_initial_points)
                    # set one index to 1
                    x_cand[torch.arange(n_initial_points), indices_for_cand] = 1
            pass

        elif parameter_type == ParameterType.ORDINAL:
            raise NotImplementedError("Ordinal parameters are not supported yet")
        else:
            raise ValueError(f"Unknown parameter type {parameter_type}")

    # remove duplicates
    x_cand = torch.unique(x_cand, dim=0)
    # remove points that coincide with x_center
    x_cand = x_cand[torch.any(x_cand != x_center, dim=1), :]
    # remove candidates that are not within the trust region
    x_cand_in_tr = x_cand[hamming_distance(x_cand, x_center) <= tr_length, :]
    if len(x_cand_in_tr) == 0:
        logging.debug(f"No initial points in trust region, returning all candidates")
    return x_cand_in_tr if len(x_cand_in_tr) > 0 else x_cand
