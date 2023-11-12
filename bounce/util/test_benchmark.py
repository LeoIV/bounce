import pytest

from bounce.util.benchmark import Parameter, ParameterType


def test_binary_parameter():
    param = Parameter(
        lower_bound=0,
        upper_bound=1,
        type=ParameterType.BINARY,
        name="test",
    )

    assert param.n_realizations == 2
    assert param.dims_required == 1
    assert param.random_sign in [-1, 1]


def test_binary_parameter_wrong_bounds():
    with pytest.raises(Exception):
        # upper bound must be 1
        Parameter(
            lower_bound=0,
            upper_bound=2,
            type=ParameterType.BINARY,
            name="test",
        )

    with pytest.raises(Exception):
        # upper bound must be 1
        Parameter(
            lower_bound=1,
            upper_bound=2,
            type=ParameterType.BINARY,
            name="test",
        )


def test_binary_parameter_wrong_random_sign():
    with pytest.raises(Exception):
        # random sign must be -1 or 1
        Parameter(
            lower_bound=0,
            upper_bound=1,
            type=ParameterType.BINARY,
            name="test",
            random_sign=0,
        )

    with pytest.raises(Exception):
        # random sign must be -1 or 1
        Parameter(
            lower_bound=0,
            upper_bound=1,
            type=ParameterType.BINARY,
            name="test",
            random_sign=2,
        )


def test_categorical_parameter():
    param = Parameter(
        lower_bound=0,
        upper_bound=2,
        type=ParameterType.CATEGORICAL,
        name="test",
    )

    assert param.n_realizations == 3
    assert param.dims_required == 3
    assert param.random_sign in [0, 1, 2]


def test_categorical_parameter_wrong_bounds():
    with pytest.raises(Exception):
        # upper bound must be larger than lower bound
        Parameter(
            lower_bound=2,
            upper_bound=1,
            type=ParameterType.CATEGORICAL,
            name="test",
        )

    with pytest.raises(Exception):
        # lower bound must be integer
        Parameter(
            lower_bound=0.5,
            upper_bound=1,
            type=ParameterType.CATEGORICAL,
            name="test",
        )


def test_ordinal_parameter():
    param = Parameter(
        lower_bound=0,
        upper_bound=2,
        type=ParameterType.ORDINAL,
        name="test",
        random_sign=1
    )

    assert param.n_realizations == 3
    assert param.dims_required == 1


def test_ordinal_parameter_wrong_bounds():
    with pytest.raises(Exception):
        # upper bound must be larger than lower bound
        Parameter(
            lower_bound=2,
            upper_bound=1,
            type=ParameterType.ORDINAL,
            name="test",
        )

    with pytest.raises(Exception):
        # lower bound must be integer
        Parameter(
            lower_bound=0.5,
            upper_bound=1,
            type=ParameterType.ORDINAL,
            name="test",
        )


def test_ordinal_parameter_random_sign_not_implemented():
    with pytest.raises(NotImplementedError):
        # random sign must be -1 or 1
        Parameter(
            lower_bound=0,
            upper_bound=1,
            type=ParameterType.ORDINAL,
            name="test",
        )


def test_real_parameter():
    param = Parameter(
        lower_bound=0,
        upper_bound=1,
        type=ParameterType.CONTINUOUS,
        name="test",
    )
    # realization is inf
    assert param.n_realizations == float("inf")
    assert param.dims_required == 1


def test_real_parameter_wrong_bounds():
    with pytest.raises(Exception):
        # upper bound must be larger than lower bound
        Parameter(
            lower_bound=2,
            upper_bound=1,
            type=ParameterType.CONTINUOUS,
            name="test",
        )

    # lower bound can be float
    Parameter(
        lower_bound=0.5,
        upper_bound=1,
        type=ParameterType.CONTINUOUS,
        name="test",
    )

    # upper bound can be float
    Parameter(
        lower_bound=0,
        upper_bound=1.5,
        type=ParameterType.CONTINUOUS,
        name="test",
    )

    # both bounds can be float
    Parameter(
        lower_bound=0.5,
        upper_bound=1.5,
        type=ParameterType.CONTINUOUS,
        name="test",
    )
