from typing import Optional

import numpy as np
import torch
from botorch.utils.transforms import normalize, unnormalize
from numpy.random import RandomState
from torch.quasirandom import SobolEngine

from bounce.util.benchmark import Parameter, ParameterType


def join_data(
        x: torch.Tensor,
        index_mapping: dict[torch.Tensor, list[torch.Tensor]]
) -> torch.Tensor:
    """
    Update data after increasing the target dimensionality.

    Args:
        x: data
        index_mapping: dictionary of indices before splitting and indices of equal data after splitting

    Returns:

    """

    # find max index before splitting
    max_index_before_splitting = max(torch.concat(list(index_mapping.keys())))
    # find non-empty index lists
    index_lists_after_splitting = [index_list_after_splitting for index_list_after_splitting in
                                   index_mapping.values() if len(index_list_after_splitting) > 0]
    # find max index after splitting, set to max index before splitting if no splitting was done
    max_index_after_splitting = max(
        torch.cat([torch.cat(index_list_after_splitting) for index_list_after_splitting in index_lists_after_splitting])
    ) if len(index_lists_after_splitting) > 0 else max_index_before_splitting
    # create new tensor with correct shape
    new_x = torch.empty((x.shape[0], max_index_after_splitting + 1), dtype=x.dtype)
    # copy old data
    new_x[:, :max_index_before_splitting + 1] = x
    # fill new data with equal data
    for indcs_old, indcss_new in index_mapping.items():
        _x = x[:, indcs_old]
        for indcs_new in indcss_new:
            new_x[:, indcs_new] = _x

    return new_x


def to_unit_cube(
        x,
        lb,
        ub
):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = normalize(x, torch.stack([lb, ub], dim=0))
    return xx


def to_1_around_origin(
        x,
        lb,
        ub
):
    """Project to [-1, 1]^d from hypercube with bounds lb and ub"""
    assert lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    x = to_unit_cube(x, lb, ub)
    xx = x * 2 - 1
    # xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(
        x,
        lb,
        ub
):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = unnormalize(x, torch.stack([lb, ub], dim=0))
    return xx


def from_1_around_origin(
        x,
        lb,
        ub
):
    xx = (x + 1) / 2
    return from_unit_cube(xx, lb, ub)


def sample_binary(
        number_of_samples: int,
        bins: list['Bin'],
        dtype: torch.dtype = torch.double,
        seed: Optional[int] = None,
):
    """

    Args:
        number_of_samples: the number of samples
        bins: the dimensionality of the samples
        dtype: the dtype of the samples
        seed: the seed for the random number generator

    Returns:
        the samples

    """

    for bin in bins:
        assert bin.parameter_type == ParameterType.BINARY, f"Parameter {bin.parameter} is not binary."
    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)
    x_init = RandomState(seed).choice(
        [-1, 1],
        size=(number_of_samples, len(bins))
    )
    x_init = torch.tensor(x_init, dtype=dtype)
    return x_init


def sample_continuous(
        number_of_samples: int,
        bins: list['Bin'],
        dtype: torch.dtype = torch.double,
        seed: Optional[int] = None,
):
    """

    Args:
        number_of_samples: the number of samples
        bins: the dimensionality of the samples
        dtype: the dtype of the samples
        seed: the seed for the random number generator

    Returns:
        the samples

    """
    for bin in bins:
        assert bin.parameter_type == ParameterType.CONTINUOUS, f"Parameter {bin.parameter} is not continuous."

    sobol = SobolEngine(len(bins), scramble=True, seed=seed)
    x_init = sobol.draw(number_of_samples).to(dtype=dtype) * 2 - 1
    return x_init


def sample_categorical(
        number_of_samples: int,
        bins: list['Bin'],
        dtype: torch.dtype = torch.double,
        seed: Optional[int] = None,
):
    """

    Args:
        number_of_samples: the number of samples
        bins: the dimensionality of the samples
        dtype: the dtype of the samples
        seed: the seed for the random number generator

    Returns:
        the samples

    """
    for bin in bins:
        assert bin.parameter_type == ParameterType.CATEGORICAL, f"Parameter {bin.parameter} is not categorical."

    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)
    dim = sum(bin.dims_required for bin in bins)
    x_init = torch.zeros((number_of_samples, dim), dtype=dtype)
    start = 0
    for i, bin in enumerate(bins):
        end = start + bin.dims_required
        idxs = RandomState(seed + i).choice(
            bin.dims_required,
            size=number_of_samples
        )

        mask = torch.zeros((number_of_samples, bin.dims_required), dtype=x_init.dtype)
        mask[torch.arange(number_of_samples), idxs] = 1

        x_init[:, start:end] = mask
        start = end

    return x_init * 2 - 1


def construct_mixed_point(
        size: int,
        binary_indices: Optional[list] = None,
        continuous_indices: Optional[list] = None,
        categorical_indices: Optional[list] = None,
        ordinal_indices: Optional[list] = None,
        x_binary: Optional[torch.Tensor] = None,
        x_continuous: Optional[torch.Tensor] = None,
        x_categorical: Optional[torch.Tensor] = None,
        x_ordinal: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Construct a mixed point from the different types of points

    Args:
        size: the number of points
        binary_indices: indices of the binary parameters
        continuous_indices: indices of the continuous parameters
        categorical_indices: indices of the categorical parameters
        ordinal_indices: indices of the ordinal parameters
        x_binary: the binary point
        x_continuous: the continuous point
        x_categorical: the categorical point
        x_ordinal: the ordinal point

    Returns:
        the mixed point

    """

    # any of the x_* can be None, but not all of them
    assert x_binary is not None or x_continuous is not None or x_categorical is not None or x_ordinal is not None, 'All x_* are None'
    if x_binary is not None:
        assert x_binary.size(0) == size, 'x_binary has wrong size'
    if x_continuous is not None:
        assert x_continuous.size(0) == size, 'x_continuous has wrong size'
    if x_categorical is not None:
        assert x_categorical.size(0) == size, 'x_categorical has wrong size'
    if x_ordinal is not None:
        assert x_ordinal.size(0) == size, 'x_ordinal has wrong size'

    if x_binary is not None:
        assert len(binary_indices) > 0, 'Benchmark does not have binary parameters but x_binary is not None'
        assert x_binary.size(1) == len(binary_indices), 'x_binary has wrong size'
    if x_continuous is not None:
        assert len(continuous_indices) > 0, 'Benchmark does not have continuous parameters but x_continuous is not None'
        assert x_continuous.size(1) == len(continuous_indices), 'x_continuous has wrong size'
    if x_categorical is not None:
        assert len(
            categorical_indices
        ) > 0, 'Benchmark does not have categorical parameters but x_categorical is not None'
        assert x_categorical.size(1) == len(categorical_indices), 'x_categorical has wrong size'
    if x_ordinal is not None:
        assert len(ordinal_indices) > 0, 'Benchmark does not have ordinal parameters but x_ordinal is not None'
        assert x_ordinal.size(1) == len(ordinal_indices), 'x_ordinal has wrong size'

    total_n_params = sum(
        [
            len(binary_indices) if binary_indices is not None else 0,
            len(continuous_indices) if continuous_indices is not None else 0,
            len(categorical_indices) if categorical_indices is not None else 0,
            len(ordinal_indices) if ordinal_indices is not None else 0,
        ]
    )
    x = torch.zeros((size, total_n_params), dtype=torch.double)

    if x_binary is not None:
        x[:, binary_indices] = x_binary
    if x_continuous is not None:
        x[:, continuous_indices] = x_continuous
    if x_categorical is not None:
        x[:, categorical_indices] = x_categorical
    if x_ordinal is not None:
        x[:, ordinal_indices] = x_ordinal

    return x


def parameter_types(
        parameters: list[Parameter]
) -> list[ParameterType]:
    """

        Args:
            parameters: the parameters

        Returns:
            the unique parameter types

        """
    return list(set([p.type for p in parameters]))
