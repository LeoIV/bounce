import numpy as np
import pytest
import torch

from bounce.projection import Bin, BinSizing
from bounce.util.benchmark import Parameter, ParameterType
from bounce.util.data_handling import (
    join_data,
    to_unit_cube,
    to_1_around_origin,
    from_unit_cube,
    from_1_around_origin, sample_binary, sample_continuous, sample_categorical, construct_mixed_point,
)


def test_join_data():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    index_mapping = {torch.tensor([0]): [torch.tensor([3])], torch.tensor([1]): [torch.tensor([4])],
                     torch.tensor([2]): [torch.tensor([5, 6])]}
    new_x = join_data(x, index_mapping)
    assert torch.all(x[:, 0] == new_x[:, 0])
    assert torch.all(x[:, 0] == new_x[:, 3])
    assert torch.all(x[:, 1] == new_x[:, 1])
    assert torch.all(x[:, 1] == new_x[:, 4])
    assert torch.all(x[:, 2] == new_x[:, 2])
    assert torch.all(x[:, 2] == new_x[:, 5])
    assert torch.all(x[:, 2] == new_x[:, 6])


def test_to_unit_cube():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double)
    lb = torch.tensor([0, 0, 0])
    ub = torch.tensor([6, 6, 6])
    xx = to_unit_cube(x, lb, ub)
    assert torch.allclose(xx, x / 6)


def test_from_unit_cube():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double)
    lb = torch.tensor([0, 0, 0])
    ub = torch.tensor([6, 6, 6])
    xx = from_unit_cube(x / 6, lb, ub)
    assert torch.allclose(xx, x)


def test_to_1_around_origin():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double)
    lb = torch.tensor([0, 0, 0])
    ub = torch.tensor([6, 6, 6])
    xx = to_1_around_origin(x, lb, ub)
    assert torch.allclose(xx, x / 3 - 1)


def test_from_1_around_origin():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double)
    lb = torch.tensor([0, 0, 0])
    ub = torch.tensor([6, 6, 6])
    xx = from_1_around_origin((x / 3) - 1, lb, ub)
    assert torch.allclose(xx, x)


def test_sample_binary():
    N_BINS = 5
    bins = [
        Bin(
            bin_sizing=BinSizing.MIN,
            parameters=[
                Parameter(
                    lower_bound=0,
                    upper_bound=1,
                    name='x',
                    type=ParameterType.BINARY
                )
            ]
        ) for _ in range(N_BINS)
    ]

    samples1 = sample_binary(
        number_of_samples=45,
        bins=bins,
    )

    samples2 = sample_binary(
        number_of_samples=78,
        bins=bins,
    )

    samples3 = sample_binary(
        number_of_samples=78,
        bins=bins,
    )

    assert samples1.shape == (45, N_BINS)
    assert samples2.shape == (78, N_BINS)
    assert samples3.shape == (78, N_BINS)
    assert not torch.allclose(samples2, samples3)

    samples4 = sample_binary(
        number_of_samples=78,
        bins=bins,
        seed=42,
    )

    samples5 = sample_binary(
        number_of_samples=78,
        bins=bins,
        seed=42,
    )

    assert torch.allclose(samples4, samples5)

    # check that error if thrown for non-binary parameters

    for pt in [ParameterType.ORDINAL, ParameterType.CATEGORICAL, ParameterType.CONTINUOUS]:
        bins = [
            Bin(
                bin_sizing=BinSizing.MIN,
                parameters=[
                    Parameter(
                        lower_bound=0,
                        upper_bound=1,
                        name='x',
                        type=pt,
                        random_sign=1
                    )
                ]
            ) for _ in range(N_BINS)
        ]
        with pytest.raises(Exception):
            sample_binary(
                number_of_samples=78,
                bins=bins,
            )


def test_sample_continuous():
    N_BINS = 5
    bins = [
        Bin(
            bin_sizing=BinSizing.MIN,
            parameters=[
                Parameter(
                    lower_bound=0,
                    upper_bound=1,
                    name='x',
                    type=ParameterType.CONTINUOUS
                )
            ]
        ) for _ in range(N_BINS)
    ]

    samples1 = sample_continuous(
        number_of_samples=45,
        bins=bins,
    )

    samples2 = sample_continuous(
        number_of_samples=78,
        bins=bins,
    )

    samples3 = sample_continuous(
        number_of_samples=78,
        bins=bins,
    )

    assert samples1.shape == (45, N_BINS)
    assert samples2.shape == (78, N_BINS)
    assert samples3.shape == (78, N_BINS)
    assert not torch.allclose(samples2, samples3)

    samples4 = sample_continuous(
        number_of_samples=78,
        bins=bins,
        seed=42,
    )

    samples5 = sample_continuous(
        number_of_samples=78,
        bins=bins,
        seed=42,
    )

    assert torch.allclose(samples4, samples5)

    # check that error if thrown for non-continuous parameters

    for pt in [ParameterType.ORDINAL, ParameterType.CATEGORICAL, ParameterType.BINARY]:
        bins = [
            Bin(
                bin_sizing=BinSizing.MIN,
                parameters=[
                    Parameter(
                        lower_bound=0,
                        upper_bound=1,
                        name='x',
                        type=pt,
                        random_sign=1
                    )
                ]
            ) for _ in range(N_BINS)
        ]
        with pytest.raises(Exception):
            sample_continuous(
                number_of_samples=78,
                bins=bins,
            )


def test_sample_categorical():
    N_BINS = 5
    bins = [
        Bin(
            bin_sizing=BinSizing.MIN,
            parameters=[
                Parameter(
                    lower_bound=0,
                    upper_bound=3,
                    name='x',
                    type=ParameterType.CATEGORICAL,
                )
            ]
        ) for _ in range(N_BINS)
    ]

    samples1 = sample_categorical(
        number_of_samples=45,
        bins=bins,
    )

    samples2 = sample_categorical(
        number_of_samples=78,
        bins=bins,
    )

    samples3 = sample_categorical(
        number_of_samples=78,
        bins=bins,
    )

    assert samples1.shape == (45, N_BINS * 4)
    assert samples2.shape == (78, N_BINS * 4)
    assert samples3.shape == (78, N_BINS * 4)
    assert not torch.allclose(samples2, samples3)

    samples4 = sample_categorical(
        number_of_samples=78,
        bins=bins,
        seed=42,
    )

    samples5 = sample_categorical(
        number_of_samples=78,
        bins=bins,
        seed=42,
    )

    assert torch.allclose(samples4, samples5)

    # check that error if thrown for non-categorical parameters

    for pt in [ParameterType.ORDINAL, ParameterType.BINARY, ParameterType.CONTINUOUS]:
        bins = [
            Bin(
                bin_sizing=BinSizing.MIN,
                parameters=[
                    Parameter(
                        lower_bound=0,
                        upper_bound=1,
                        name='x',
                        type=pt,
                        random_sign=1
                    )
                ]
            ) for _ in range(N_BINS)
        ]
        with pytest.raises(Exception):
            sample_categorical(
                number_of_samples=78,
                bins=bins,
            )


def test_construct_mixed_point():
    ## BINARY

    N_BINS = 5
    bins = [
        Bin(
            bin_sizing=BinSizing.MIN,
            parameters=[
                Parameter(
                    lower_bound=0,
                    upper_bound=1,
                    name='x',
                    type=ParameterType.BINARY
                )
            ]
        ) for _ in range(N_BINS)
    ]

    binary_point = sample_binary(
        number_of_samples=45,
        bins=bins,
    )

    bp = construct_mixed_point(
        size=45,
        binary_indices=np.arange(5).tolist(),
        x_binary=binary_point,
    )

    assert bp.shape == (45, N_BINS)

    # check that error if size does not fit point shape
    with pytest.raises(Exception):
        construct_mixed_point(
            size=46,
            binary_indices=np.arange(5).tolist(),
            x_binary=binary_point,
        )

    ## CONTINUOUS

    N_BINS = 5
    bins = [
        Bin(
            bin_sizing=BinSizing.MIN,
            parameters=[
                Parameter(
                    lower_bound=0,
                    upper_bound=1,
                    name='x',
                    type=ParameterType.CONTINUOUS
                )
            ]
        ) for _ in range(N_BINS)
    ]

    cp = continuous_point = sample_continuous(
        number_of_samples=45,
        bins=bins,
    )

    assert cp.shape == (45, N_BINS)

    construct_mixed_point(
        size=45,
        continuous_indices=np.arange(5).tolist(),
        x_continuous=continuous_point,
    )

    # check that error if size does not fit point shape
    with pytest.raises(Exception):
        construct_mixed_point(
            size=46,
            continuous_indices=np.arange(5).tolist(),
            x_continuous=continuous_point,
        )

    ## CATEGORICAL

    N_BINS = 5
    bins = [
        Bin(
            bin_sizing=BinSizing.MIN,
            parameters=[
                Parameter(
                    lower_bound=0,
                    upper_bound=3,
                    name='x',
                    type=ParameterType.CATEGORICAL,
                )
            ]
        ) for _ in range(N_BINS)
    ]

    categorical_point = sample_categorical(
        number_of_samples=45,
        bins=bins,
    )

    cp = construct_mixed_point(
        size=45,
        categorical_indices=np.arange(20).tolist(),
        x_categorical=categorical_point,
    )

    assert cp.shape == (45, N_BINS * 4)

    # check that error if size does not fit point shape

    with pytest.raises(Exception):
        construct_mixed_point(
            size=46,
            categorical_indices=np.arange(5).tolist(),
            x_categorical=categorical_point,
        )

    ## MIXED

    mixed_point = construct_mixed_point(
        size=45,
        binary_indices=np.arange(5).tolist(),
        x_binary=binary_point,
        continuous_indices=np.arange(5, 10).tolist(),
        x_continuous=continuous_point,
        categorical_indices=np.arange(10, 30).tolist(),
        x_categorical=categorical_point,
    )
