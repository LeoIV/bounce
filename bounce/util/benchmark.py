import dataclasses
import json
import os
import random
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional


def eval_singularity_benchmark(
    eval_points: list[list[float]], singularity_image_path: str, name: str
) -> float:
    """
    Evaluate a benchmark function in a singularity image

    Args:
        eval_points: the points to evaluate
        singularity_image_path: the path to the singularity image
        name: the name of the benchmark function

    Returns:
        the result of the evaluation

    """
    cmd = (
        f"$( cd {singularity_image_path} ; poetry env info --path)/bin/python3 {os.path.join(singularity_image_path, 'main.py')} --name {name} "
        f"-x {' '.join(list(map(lambda _x: str(_x), eval_points)))}"
    )
    process = subprocess.check_output(
        cmd,
        shell=True,
        env={
            **os.environ,
            **{
                "LD_LIBRARY_PATH": f"{singularity_image_path}/data/mujoco210/bin:/usr/lib/nvidia",
                "MUJOCO_PY_MUJOCO_PATH": f"{singularity_image_path}/data/mujoco210",
            },
        },
    )
    res = process.decode().split("\n")
    return float(res[-2])


@dataclass
class BenchmarkRequest:
    """
    A request to evaluate a benchmark function

    Args:
        function: the name of the benchmark function
        dim: the dimension of the benchmark function
        eval_points: the points to evaluate
        effective_dim: the effective dimension of the benchmark function
        noise_std: the noise standard deviation of the benchmark function
        max_steps: the maximum number of steps to evaluate the benchmark function
    """

    function: str
    dim: int
    eval_points: list[list[float]]
    effective_dim: Optional[int] = None
    noise_std: Optional[float] = None
    max_steps: Optional[int] = 2000

    def as_json(self) -> str:
        """
        Convert the request to a json string

        Returns:
            the json string

        """
        return json.dumps(dataclasses.asdict(self))


class ParameterType(Enum):
    """
    The type of a parameter
    """

    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


@dataclass
class Parameter:
    """
    A parameter of a benchmark function

    Args:
        name: the name of the parameter
        type: the type of the parameter
        lower_bound: the lower bound of the parameter
        upper_bound: the upper bound of the parameter
        random_sign: the random sign of the parameter (will be randomly chosen if None)
        n_realizations: the number of realizations of the parameter (will be computed automatically if None)
    """

    name: str
    type: ParameterType
    lower_bound: float
    upper_bound: float
    random_sign: int = None
    n_realizations: float | int = dataclasses.field(init=False)

    def __post_init__(self):
        if self.random_sign is None:
            if (
                self.type == ParameterType.BINARY
                or self.type == ParameterType.CONTINUOUS
            ):
                self.random_sign = random.choice([-1, 1])
            elif self.type == ParameterType.CATEGORICAL:
                # random sign random int in [0, n_realizations]
                n_realizations = int(self.upper_bound - self.lower_bound + 1)
                self.random_sign = random.randint(0, n_realizations - 1)
            elif self.type == ParameterType.ORDINAL:
                raise NotImplementedError(
                    "Random sign for ordinal parameters not implemented"
                )
            else:
                raise ValueError(f"Unknown parameter type {self.type}")
        if self.type == ParameterType.CATEGORICAL:
            assert float(
                self.lower_bound
            ).is_integer(), "Categorical parameters must have integer lower bound"
            assert float(
                self.upper_bound
            ).is_integer(), "Categorical parameters must have integer upper bound"
            assert (
                self.lower_bound < self.upper_bound
            ), "Categorical parameters must have lower bound < upper bound"
            self.n_realizations = int(self.upper_bound - self.lower_bound + 1)
        elif self.type == ParameterType.ORDINAL:
            assert float(
                self.lower_bound
            ).is_integer(), "Ordinal parameters must have integer lower bound"
            assert float(
                self.upper_bound
            ).is_integer(), "Ordinal parameters must have integer upper bound"
            assert (
                self.lower_bound < self.upper_bound
            ), "Ordinal parameters must have lower bound < upper bound"
            self.n_realizations = int(self.upper_bound - self.lower_bound + 1)
        elif self.type == ParameterType.BINARY:
            assert self.lower_bound == 0, "Binary parameters must have lower bound 0"
            assert self.upper_bound == 1, "Binary parameters must have upper bound 1"
            assert self.random_sign in [
                -1,
                1,
            ], "Binary parameters must have random sign in [-1, 1]"
            self.n_realizations = 2
        elif self.type == ParameterType.CONTINUOUS:
            assert (
                self.lower_bound < self.upper_bound
            ), "Continuous parameters must have lower bound < upper bound"
            self.n_realizations = float("inf")

    @property
    def dims_required(self) -> int:
        """
        The number of dimensions required to represent the parameter

        Returns:
            the number of dimensions required to represent the parameter

        """
        match self.type:
            case ParameterType.CONTINUOUS:
                return 1
            case ParameterType.CATEGORICAL:
                return self.n_realizations
            case ParameterType.ORDINAL:
                return 1
            case ParameterType.BINARY:
                return 1
            case _:
                raise ValueError(f"Unknown parameter type {self.type}")
