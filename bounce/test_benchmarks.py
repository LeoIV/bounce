import pytest
import torch

from bounce.benchmarks import (
    AckleyEffectiveDim,
    RosenbrockEffectiveDim,
    HartmannEffectiveDim,
    BraninEffectiveDim,
    LevyEffectiveDim,
    DixonPriceEffectiveDim,
    GriewankEffectiveDim,
    MichalewiczEffectiveDim,
    RastriginEffectiveDim,
    ShiftedAckley10,
    SVMMixed,
    Ackley53,
    MaxSat60,
    MaxSat125,
    Labs,
)
from bounce.util.benchmark import Parameter, ParameterType


@pytest.mark.parametrize(
    "test_function,fixed_edim",
    [
        (AckleyEffectiveDim, False),
        (RosenbrockEffectiveDim, False),
        (HartmannEffectiveDim, True),
        (BraninEffectiveDim, True),
        (LevyEffectiveDim, False),
        (DixonPriceEffectiveDim, False),
        (GriewankEffectiveDim, False),
        (MichalewiczEffectiveDim, False),
        (RastriginEffectiveDim, False),
    ],
)
def test_ackley_effective_dimension(test_function, fixed_edim):
    benchmark = test_function()
    assert benchmark.dim == 200, "Default dimension is 200"

    if not fixed_edim:
        benchmark = test_function(effective_dim=5)
        assert benchmark.dim == 200, "Default dimension is 200"
        assert benchmark.effective_dim == 5, "Effective dimension can be set to 5"
    else:
        assert benchmark.dim == 200, "Default dimension is 200"
        with pytest.raises(AssertionError):
            benchmark = test_function(effective_dim=5)
            benchmark = test_function(effective_dim=6)

    if not fixed_edim:
        benchmark = test_function(effective_dim=15)
        assert benchmark.dim == 200, "Default dimension is 200"
        assert benchmark.effective_dim == 15, "Effective dimension can be set to 15"
    else:
        assert benchmark.dim == 200, "Default dimension is 200"
        with pytest.raises(AssertionError):
            benchmark = test_function(effective_dim=15)
            benchmark = test_function(effective_dim=16)

    benchmark = test_function(dim=100)
    assert benchmark.dim == 100, "Dimension can be set to 100"

    # test reproducatibility
    benchmark = test_function()
    x = torch.rand(1, 200).repeat(20, 1)
    fx = benchmark(x)
    assert torch.all(fx == fx[0]), "Reproducibility test failed"


@pytest.mark.parametrize(
    "benchmark",
    [
        AckleyEffectiveDim,
        ShiftedAckley10,
        RosenbrockEffectiveDim,
        HartmannEffectiveDim,
        BraninEffectiveDim,
        LevyEffectiveDim,
        DixonPriceEffectiveDim,
        GriewankEffectiveDim,
        MichalewiczEffectiveDim,
        RastriginEffectiveDim,
    ],
)
def test_continuous_benchmarks(benchmark):
    benchmark = benchmark()
    assert benchmark.is_continuous, "Continuous benchmark"
    assert not benchmark.is_discrete, "Continuous benchmark"
    assert not benchmark.is_mixed, "Continuous benchmark"

    parameters: list[Parameter] = benchmark.parameters
    for param in parameters:
        assert param.type == ParameterType.CONTINUOUS
        assert param.n_realizations == float("inf")


@pytest.mark.parametrize(
    "benchmark",
    [
        SVMMixed,
        Ackley53,
    ],
)
def test_mixed_benchmarks(benchmark):
    benchmark = benchmark()
    assert benchmark.is_mixed, "Mixed benchmark"
    assert not benchmark.is_continuous, "Mixed benchmark"
    assert not benchmark.is_discrete, "Mixed benchmark"

    parameters: list[Parameter] = benchmark.parameters
    continuous_parameters = [
        param for param in parameters if param.type == ParameterType.CONTINUOUS
    ]
    discrete_parameters = [
        param for param in parameters if param.type != ParameterType.CONTINUOUS
    ]

    assert len(continuous_parameters) > 0, "Mixed benchmark has continuous parameters"
    assert len(discrete_parameters) > 0, "Mixed benchmark has discrete parameters"


@pytest.mark.parametrize(
    "benchmark",
    [
        MaxSat60,
        MaxSat125,
        Labs,
    ],
)
def test_discrete_benchmarks(benchmark):
    benchmark = benchmark()
    assert benchmark.is_discrete, "Discrete benchmark"
    assert not benchmark.is_continuous, "Discrete benchmark"
    assert not benchmark.is_mixed, "Discrete benchmark"

    parameters: list[Parameter] = benchmark.parameters
    for param in parameters:
        assert param.type != ParameterType.CONTINUOUS
