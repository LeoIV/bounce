import torch

from bounce.projection import Bin, BinSizing
from bounce.util.benchmark import Parameter, ParameterType


def test_project_up_binary():
    bin = Bin(
        parameters=[
            Parameter(
                type=ParameterType.BINARY,
                lower_bound=0,
                upper_bound=1,
                name="x",
                random_sign=-1,
            ),
            Parameter(
                type=ParameterType.BINARY,
                lower_bound=0,
                upper_bound=1,
                name="y",
                random_sign=1,
            ),
        ]
    )

    low = torch.tensor([[0], [1]], dtype=torch.double)

    up = bin.project_up(low)

    assert up.shape == (2, 2)
    assert up[1, 0] == -1
    assert up[1, 1] == 1


def test_project_up_continuous():
    bin = Bin(
        parameters=[
            Parameter(
                type=ParameterType.CONTINUOUS,
                lower_bound=0,
                upper_bound=1,
                name="x",
                random_sign=-1,
            ),
            Parameter(
                type=ParameterType.CONTINUOUS,
                lower_bound=0,
                upper_bound=1,
                name="y",
                random_sign=1,
            ),
        ]
    )

    low = torch.tensor([[0], [1]], dtype=torch.double)

    up = bin.project_up(low)

    assert up.shape == (2, 2)
    assert up[1, 0] == -1
    assert up[1, 1] == 1


def test_project_up_categorical():
    bin = Bin(
        parameters=[
            Parameter(
                type=ParameterType.CATEGORICAL,
                lower_bound=0,
                upper_bound=2,
                name="x",
            ),
            Parameter(
                type=ParameterType.CATEGORICAL,
                lower_bound=0,
                upper_bound=3,
                name="y",
            ),
        ],
        bin_sizing=BinSizing.MIN,
    )

    low = torch.tensor([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.double)


# TODO Continue here
