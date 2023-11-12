import logging
import math
import multiprocessing
import os
import pathlib
from abc import ABC, abstractmethod
from functools import partial
from logging import info
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Type, Union

import gin
import numpy as np
import torch
from botorch.test_functions import (
    Ackley as BotorchAckley,
    Branin as BotorchBranin,
    DixonPrice as BotorchDixonPrice,
    Griewank as BotorchGriewank,
    Hartmann as BotorchHartmann,
    Levy as BotorchLevy,
    Michalewicz as BotorchMichalewicz,
    Rastrigin as BotorchRastrigin,
    Rosenbrock as BotorchRosenbrock,
    SyntheticTestFunction,
)
from numpy.random import RandomState
from sklearn.svm import SVR

from bounce.util.benchmark import Parameter, ParameterType, eval_singularity_benchmark
from bounce.util.data_loading import (
    load_uci_data,
    download_maxsat60_data,
    download_maxsat125_data,
)
from bounce.util.pest_control import _pest_control_score
from bounce.util.sat import WCNF


@gin.configurable
class Benchmark(ABC):
    """
    Abstract benchmark function.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        noise_std: Optional[float],
        flip: bool = False,
    ):
        """
        Initialize the benchmark function.

        Args:
            parameters: the parameters of the benchmark (list of `Parameter` objects)
            noise_std: the standard deviation of the noise (0 and `None` mean no noise)
            flip: whether to randomly change the structure of the optimum
        """
        # assert all parameters have unique names
        assert len(set([p.name for p in parameters])) == len(
            parameters
        ), "Duplicate parameter names are not allowed."

        lb = torch.tensor([p.lower_bound for p in parameters])
        ub = torch.tensor([p.upper_bound for p in parameters])

        if not lb.shape == ub.shape or not lb.ndim == 1 or not ub.ndim == 1:
            raise RuntimeError("bounds mismatch")
        if not torch.all(lb < ub):
            raise RuntimeError("out of bounds")
        self.noise_std = noise_std
        """
        the standard deviation of the noise (0 and `None` mean no noise)
        """
        self.parameters: list[Parameter] = parameters
        """
        the parameters of the benchmark (list of `Parameter` objects)
        """
        self._lb_vec = lb
        self._ub_vec = ub

        self.flip = flip
        """
        whether to randomly change the structure of the optimum
        """

    @property
    def dim(self) -> int:
        """

        Returns:
            int: the benchmark dimensionality (in terms of the number of parameters)
        """
        return len(self.parameters)

    @property
    def representation_dim(self) -> int:
        """

        Returns:
            int: the representation dimensionality (in terms of the shape of the representation vector)
        """
        return sum(
            [
                1 if p.type != ParameterType.CATEGORICAL else p.n_realizations
                for p in self.parameters
            ]
        )

    @property
    def lb_vec(self) -> torch.Tensor:
        """

        Returns:
            np.ndarray: the lower bound of the search space of this benchmark (length = benchmark dim)

        """

        # check if benchmark contains only non-categorical parameters
        if not any([p.type == ParameterType.CATEGORICAL for p in self.parameters]):
            return self._lb_vec
        else:
            vec = []
            for p in self.parameters:
                if p.type == ParameterType.CATEGORICAL:
                    vec.extend([0] * p.n_realizations)
                else:
                    vec.append(p.lower_bound)
            return torch.tensor(vec, dtype=torch.double)

    @property
    def ub_vec(self) -> torch.Tensor:
        """

        Returns:
            np.ndarray: the upper bound of the search space of this benchmark (length = benchmark dim)

        """

        # check if benchmark contains only non-categorical parameters
        if not any([p.type == ParameterType.CATEGORICAL for p in self.parameters]):
            return self._ub_vec
        else:
            vec = []
            for p in self.parameters:
                if p.type == ParameterType.CATEGORICAL:
                    vec.extend([1] * p.n_realizations)
                else:
                    vec.append(p.upper_bound)
            return torch.tensor(vec, dtype=torch.double)

    @property
    def fun_name(self) -> str:
        """

        Returns:
            str: the name of this function

        """
        return self.__class__.__name__

    @property
    def is_continuous(self) -> bool:
        """

        Returns:
            bool: whether the benchmark is continuous

        """
        # check if all parameters are continuous
        return all([p.type == ParameterType.CONTINUOUS for p in self.parameters])

    @property
    def is_discrete(self) -> bool:
        """

        Returns:
            bool: whether the benchmark is discrete

        """
        # check if all parameters are non-continuous
        return all([p.type != ParameterType.CONTINUOUS for p in self.parameters])

    @property
    def is_categorical(self) -> bool:
        """

        Returns:
            bool: whether the benchmark is categorical

        """
        # check if all parameters are categorical
        return all([p.type == ParameterType.CATEGORICAL for p in self.parameters])

    @property
    def is_ordinal(self) -> bool:
        """

        Returns:
            bool: whether the benchmark is ordinal

        """
        # check if all parameters are ordinal
        return all([p.type == ParameterType.ORDINAL for p in self.parameters])

    @property
    def is_binary(self) -> bool:
        """

        Returns:
            bool: whether the benchmark is binary

        """
        # check if all parameters are binary
        return all([p.type == ParameterType.BINARY for p in self.parameters])

    @property
    def is_mixed(self) -> bool:
        """

        Returns:
            bool: whether the benchmark is mixed

        """
        # check if all parameters are mixed
        return not self.is_continuous and not self.is_discrete

    @property
    def is_mixed_binary(self) -> bool:
        """

        Returns:
            bool: whether the benchmark is mixed and non-continuous parameters are binary

        """
        # check if benchmark is mixed and non-continuous parameters are binary
        return self.is_mixed and [
            p.type == ParameterType.BINARY
            for p in self.parameters
            if p.type != ParameterType.CONTINUOUS
        ]

    @property
    def n_binary(self) -> int:
        """

        Returns:
            int: the number of binary parameters

        """
        return len([p for p in self.parameters if p.type == ParameterType.BINARY])

    @property
    def n_categorical(self) -> int:
        """

        Returns:
            int: the number of categorical parameters

        """
        return len([p for p in self.parameters if p.type == ParameterType.CATEGORICAL])

    @property
    def n_ordinal(self) -> int:
        """

        Returns:
            int: the number of ordinal parameters

        """
        return len([p for p in self.parameters if p.type == ParameterType.ORDINAL])

    @property
    def n_continuous(self) -> int:
        """

        Returns:
            int: the number of continuous parameters

        """
        return len([p for p in self.parameters if p.type == ParameterType.CONTINUOUS])

    @property
    def n_discrete(self) -> int:
        """

        Returns:
            int: the number of discrete parameters

        """
        return len([p for p in self.parameters if p.type != ParameterType.CONTINUOUS])

    @property
    def binary_indices(self) -> torch.Tensor:
        """

        Returns:
            torch.Tensor: the indices of the binary parameters

        """
        return torch.tensor(
            [i for i, p in enumerate(self.parameters) if p.type == ParameterType.BINARY]
        )

    @property
    def categorical_indices(self) -> torch.Tensor:
        """

        Returns:
            torch.Tensor: the indices of the categorical parameters

        """
        return torch.tensor(
            [
                i
                for i, p in enumerate(self.parameters)
                if p.type == ParameterType.CATEGORICAL
            ]
        )

    @property
    def ordinal_indices(self) -> torch.Tensor:
        """

        Returns:
            torch.Tensor: the indices of the ordinal parameters

        """
        return torch.tensor(
            [
                i
                for i, p in enumerate(self.parameters)
                if p.type == ParameterType.ORDINAL
            ]
        )

    @property
    def continuous_indices(self) -> torch.Tensor:
        """

        Returns:
            torch.Tensor: the indices of the continuous parameters

        """
        return torch.tensor(
            [
                i
                for i, p in enumerate(self.parameters)
                if p.type == ParameterType.CONTINUOUS
            ]
        )

    @property
    def unique_parameter_types(self) -> list[ParameterType]:
        """

        Returns:
            list[ParameterType]: the unique parameter types of this benchmark

        """
        return list(set([p.type for p in self.parameters]))

    def number_of_parameters_of_type(self, parameter_type: ParameterType) -> int:
        """

        Returns:
            int: the number of parameters of a given type

        """
        return len([p for p in self.parameters if p.type == parameter_type])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class SyntheticBenchmark(Benchmark):
    """
    Abstract class for synthetic benchmarks
    """

    @abstractmethod
    def __init__(
        self,
        parameters: list[Parameter],
        noise_std: float,
        *args,
        **kwargs,
    ):
        """
        Initialize the benchmark function.

        Args:
            parameters: the parameters of the benchmark (list of `Parameter` objects)
            noise_std: the standard deviation of the noise (0 and `None` mean no noise)
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        super().__init__(parameters=parameters, noise_std=noise_std)

    @abstractmethod
    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Call the benchmark function for one or multiple points.

        Args:
            x: Union[np.ndarray, list[float], list[list[float]]]: the x-value(s) to evaluate. numpy array can be 1 or 2-dimensional

        Returns:
            np.ndarray: The function values.


        """
        raise NotImplementedError()

    @property
    def optimal_value(self) -> Optional[np.ndarray]:
        """

        Returns:
            Optional[Union[float, np.ndarray]]: the optimal value if known

        """
        return None


class EffectiveDimBenchmark(SyntheticBenchmark):
    """
    Abstract class for synthetic benchmarks with an effective dimensionality
    """

    def __init__(
        self,
        dim: int,
        effective_dim: int,
        parameters: list[Parameter],
        noise_std: float,
        *args,
        **kwargs,
    ):
        """
        Initialize the benchmark function.

        Args:
            dim: the benchmark dimensionality
            effective_dim: the effective dimensionality of the benchmark
            parameters: the parameters of the benchmark (list of `Parameter` objects)
            noise_std: the standard deviation of the noise (0 and `None` mean no noise)
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        super().__init__(dim=dim, parameters=parameters, noise_std=noise_std)
        self.effective_dim: int = effective_dim
        """
        the effective dimensionality of the benchmark
        """

    @abstractmethod
    def __call__(
        self, x: np.ndarray | list[float] | list[list[float]], *args, **kwargs
    ):
        raise NotImplementedError()


class BoTorchFunctionBenchmark(SyntheticBenchmark):
    """
    Abstract class for synthetic benchmarks that are implemented in BoTorch

    Args:
        dim: the benchmark dimensionality
        parameters: the parameters of the benchmark
        noise_std: the standard deviation of the noise (0 means no noise)
        lb: the lower bound of the search space
        ub: the upper bound of the search space
        benchmark_func: the BoTorch benchmark function
    """

    def __init__(
        self,
        dim: int,
        noise_std: Optional[float],
        ub: torch.Tensor,
        lb: torch.Tensor,
        benchmark_func: Type[SyntheticTestFunction],
        *args,
        **kwargs,
    ):
        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=lb[i],
                upper_bound=ub[i],
            )
            for i in range(dim)
        ]
        super().__init__(parameters=parameters, noise_std=noise_std)
        try:
            self._benchmark_func = benchmark_func(dim=dim, noise_std=noise_std)
        except:
            self._benchmark_func = benchmark_func(noise_std=noise_std)

    @property
    def effective_dim(self) -> int:
        return self._dim

    @property
    def optimal_value(self) -> float:
        """
        Get the optimal value of the benchmark function

        Returns:
            the optimal value of the benchmark function

        """
        return self._benchmark_func.optimal_value

    def __call__(self, x, *args, **kwargs):
        super(BoTorchFunctionBenchmark, self).__call__(x)
        if x.ndim in [0, 1]:
            x = x.expand(1, -1)
        assert x.ndim == 2
        res = self._benchmark_func.forward(
            torch.clip(x, self._lb_vec, self._ub_vec)
        ).squeeze()
        return res


class EffectiveDimBoTorchBenchmark(BoTorchFunctionBenchmark):
    """
    A benchmark class for synthetic benchmarks with a known effective dimensionality that are based on a BoTorch
    implementation.

    Args:
        dim: int: the ambient dimensionality of the benchmark
        noise_std: float: standard deviation of the noise of the benchmark function
        effective_dim: int: the desired effective dimensionality of the benchmark function
        ub: np.ndarray: the upper bound of the benchmark search space. length = dim
        lb: np.ndarray: the lower bound of the benchmark search space. length = dim
        benchmark_func: Type[SyntheticTestFunction]: the BoTorch benchmark function to use
    """

    def __init__(
        self,
        dim: int,
        noise_std: Optional[float],
        effective_dim: int,
        ub: torch.Tensor,
        lb: torch.Tensor,
        benchmark_func: Type[SyntheticTestFunction],
        *args,
        **kwargs,
    ):
        super().__init__(
            dim,
            noise_std,
            ub=ub,
            lb=lb,
            benchmark_func=benchmark_func,
            *args,
            **kwargs,
        )
        # override the botoch benchmark function with the effective dim
        try:
            self._benchmark_func = benchmark_func(
                dim=effective_dim, noise_std=noise_std
            )
        except:
            self._benchmark_func = benchmark_func(noise_std=noise_std)

        if effective_dim > dim:
            raise RuntimeError("effective dim too large")
        self._fake_dim = dim
        self._effective_dim = effective_dim
        self.effective_dims = np.arange(dim)[:effective_dim]
        """
        the effective dimensionality of the benchmark
        """
        info(f"effective dims: {list(self.effective_dims)}")

    def __call__(
        self, x: Union[np.ndarray, list[float], list[list[float]]], *args, **kwargs
    ) -> np.ndarray:
        """
        Call the benchmark function for one or multiple points.

        Args:
            x: Union[np.ndarray, list[float], list[list[float]]]: the x-value(s) to evaluate. numpy array can be 1 or 2-dimensional
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            np.ndarray: The function values.

        """
        if x.ndim in [0, 1]:
            x = torch.unsqueeze(x, 0)
        assert x.ndim == 2

        res = self._benchmark_func.forward(x[:, : self.effective_dim]).squeeze()
        return res

    @property
    def dim(self):
        """

        Returns:
            int: the ambient dimensionality of the benchmark

        """
        return self._fake_dim

    @property
    def effective_dim(self) -> int:
        """

        Returns:
            int: the effective dimensionality of the benchmark

        """
        return self._effective_dim


@gin.configurable
class AckleyEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    A benchmark function with many local minima (see https://www.sfu.ca/~ssurjano/ackley.html)

    .. warning:: This function has its optimum at the origin. This might lead to overly optimistic results for `Bounce`.
    """

    def __init__(
        self, dim: int = 200, noise_std=None, effective_dim: int = 10, *args, **kwargs
    ):
        """
        Initialize the benchmark function.

        Args:
            dim: the ambient dimensionality of the benchmark
            noise_std: the standard deviation of the noise (0 and `None` mean no noise)
            effective_dim: the effective dimensionality of the benchmark
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        super(AckleyEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=torch.ones(dim) * -32.768,
            ub=torch.ones(dim) * 32.768,
            benchmark_func=BotorchAckley,
        )


@gin.configurable
class ShiftedAckley10(EffectiveDimBoTorchBenchmark):
    """
    A benchmark function with many local minima (see https://www.sfu.ca/~ssurjano/ackley.html)

    .. note:: The optimizer for this function is shifted to avoid the problems with `AckleyEffectiveDim`.
    """

    def __init__(self, dim: int = 200, noise_std=None, *args, **kwargs):
        """
        Initialize the benchmark function.

        Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.offsets = torch.tensor(
            [
                -14.15468831,
                -17.35934204,
                4.93227439,
                30.68108305,
                -20.94097318,
                -9.68946759,
                11.23919487,
                4.93101114,
                2.87604112,
                -31.0805155,
            ]
        )
        """
        The offsets that are used to shift the optimizer
        """

        lb = torch.ones(size=(dim,)) * (-32.768)
        lb[:10] = lb[:10] - self.offsets
        ub = torch.ones(size=(dim,)) * 32.768
        ub[:10] = ub[:10] - self.offsets

        super(ShiftedAckley10, self).__init__(
            dim=dim,
            effective_dim=10,
            noise_std=noise_std,
            lb=lb,
            ub=ub,
            benchmark_func=BotorchAckley,
        )

    def __call__(
        self, x: Union[np.ndarray, list[float], list[list[float]]], *args, **kwargs
    ) -> np.ndarray:
        """
        Call the benchmark function for one or multiple points.

        Args:
            x: Union[np.ndarray, list[float], list[list[float]]]: the x-value(s) to evaluate. numpy array can be 1 or 2-dimensional
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: The function values.

        """
        return super().__call__(x)


@gin.configurable
class RosenbrockEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    A valley-shape benchmark function (see https://www.sfu.ca/~ssurjano/rosen.html)
    """

    def __init__(
        self,
        dim: int = 200,
        noise_std: Optional[float] = None,
        effective_dim: int = 10,
        *args,
        **kwargs,
    ):
        """
        Initialize the benchmark function.

        Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            ub=torch.ones(dim) * 10,
            lb=torch.ones(dim) * (-5),
            benchmark_func=BotorchRosenbrock,
        )


@gin.configurable
class HartmannEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    A valley-shape benchmark function (see https://www.sfu.ca/~ssurjano/rosen.html)
    """

    def __init__(
        self,
        dim: int = 200,
        noise_std: Optional[float] = None,
        effective_dim: int = 6,
        *args,
        **kwargs,
    ):
        """
        Initialize the benchmark function.

        Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        assert effective_dim == 6
        super().__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            ub=torch.ones(dim),
            lb=torch.zeros(dim),
            benchmark_func=BotorchHartmann,
        )


@gin.configurable
class BraninEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Branin function with three local minima (see https://www.sfu.ca/~ssurjano/branin.html)
    """

    def __init__(
        self,
        dim: int = 200,
        noise_std: Optional[float] = None,
        effective_dim: int = 2,
        *args,
        **kwargs,
    ):
        """
        Initialize the benchmark function.

        Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        assert (
            effective_dim == 2
        ), "Branin function only supports effective dimensionality of 2"
        lb = torch.ones(dim) * (-5)
        lb[1] = 0
        ub = torch.ones(dim) * (15)
        ub[0] = 10

        super().__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=lb,
            ub=ub,
            benchmark_func=BotorchBranin,
        )


@gin.configurable
class LevyEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Levy function with many local minima (see https://www.sfu.ca/~ssurjano/levy.html)
    """

    def __init__(
        self, dim: int = 200, noise_std=None, effective_dim: int = 2, *args, **kwargs
    ):
        """

        Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super(LevyEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=torch.ones(dim) * (-10),
            ub=torch.ones(dim) * 10,
            benchmark_func=BotorchLevy,
        )


@gin.configurable
class DixonPriceEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The valley shaped Dixon-Price function (see https://www.sfu.ca/~ssurjano/dixonpr.html)
    """

    def __init__(
        self, dim: int = 200, noise_std=None, effective_dim: int = 2, *args, **kwargs
    ):
        """

        Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super(DixonPriceEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=torch.ones(dim) * (-10),
            ub=torch.ones(dim) * 10,
            benchmark_func=BotorchDixonPrice,
        )


@gin.configurable
class GriewankEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Griewank function with many local minima (see https://www.sfu.ca/~ssurjano/griewank.html)

    .. warning:: This function has its optimum at the origin. This might lead to overly optimistic results for `Bounce`.
    """

    def __init__(
        self, dim: int = 200, noise_std=None, effective_dim: int = 2, *args, **kwargs
    ):
        """
        Initialize the benchmark function.

        Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super(GriewankEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=torch.ones(dim) * (-600),
            ub=torch.ones(dim) * 600,
            benchmark_func=BotorchGriewank,
        )


@gin.configurable
class MichalewiczEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Michalewicz function with steep drops (see https://www.sfu.ca/~ssurjano/michal.html)
    """

    def __init__(
        self, dim: int = 200, noise_std=None, effective_dim: int = 2, *args, **kwargs
    ):
        """
        Initialize the benchmark function.

        Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super(MichalewiczEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=torch.zeros(dim),
            ub=torch.ones(dim) * math.pi,
            benchmark_func=BotorchMichalewicz,
        )


@gin.configurable
class RastriginEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Rastrigin function with many local minima (see https://www.sfu.ca/~ssurjano/rastr.html)

    .. warning:: This function has its optimum at the origin. This might lead to overly optimistic results for `Bounce`.

    """

    def __init__(
        self, dim: int = 200, noise_std=None, effective_dim: int = 2, *args, **kwargs
    ):
        """
        Initialize the benchmark function.

        Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super(RastriginEffectiveDim, self).__init__(
            dim=dim,
            effective_dim=effective_dim,
            noise_std=noise_std,
            lb=torch.ones(dim) * (-5.12),
            ub=torch.ones(dim) * 5.12,
            benchmark_func=BotorchRastrigin,
        )


@gin.configurable
class SingularityBenchmark(Benchmark):
    """
    A benchmark function that is implemented in the Singularity Benchmark Suite (requires cloning the repository
    parallel to the Bounce repository, see https://github.com/LeoIV/BenchSuite)
    """

    def __init__(
        self,
        dim: int,
        name: str,
        parameters: list[Parameter],
        singularity_image_path: Optional[str] = None,
        n_workers: Optional[int] = None,
    ):
        """
        Initialize the benchmark function.

        Args:
            dim: the benchmark dimensionality
            name: the name of the benchmark function
            parameters: the parameters of the benchmark (list of `Parameter` objects)
            singularity_image_path: the path to the singularity image of the Singularity Benchmark Suite
            n_workers: the number of workers to use for parallel evaluation
        """
        super().__init__(parameters=parameters, noise_std=None)

        if singularity_image_path is None:
            singularity_image_path = os.path.join(
                Path(__file__).parent.parent.parent.resolve(), "BenchSuite"
            )

        self.singularity_image_path = singularity_image_path
        """
        the path to the singularity image of the Singularity Benchmark Suite
        """
        self.name = name
        """
        the name of the benchmark function
        """
        self.n_workers = n_workers
        """
        the number of workers to use for parallel evaluation
        """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with Pool(
            multiprocessing.cpu_count() if self.n_workers is None else self.n_workers
        ) as p:
            func = partial(
                eval_singularity_benchmark,
                singularity_image_path=self.singularity_image_path,
                name=self.name,
            )
            results = p.map(func, x.detach().cpu().numpy().tolist())
        # TODO make dtype dynamic
        results = torch.tensor(results, dtype=torch.float64)
        return results


@gin.configurable
class SVM(SingularityBenchmark):
    """
    The SVM benchmark from the Singularity Benchmark Suite
    """

    def __init__(
        self,
    ):
        dim: int = 388
        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for i in range(dim)
        ]
        super().__init__(
            dim=dim,
            name="svm",
            parameters=parameters,
        )


@gin.configurable
class SVMMixed(Benchmark):
    """
    The mixed variable-type SVM benchmark
    """

    def __init__(
        self,
        n_features: int = 50,
    ):
        """
        Initialize the benchmark function.

        Args:
            n_features: the number of features to optimize over (the `n` most important parameters are chosen using an
                `XGBoost` feature importance analysis)
        """
        self.n_features = n_features
        """
        the number of features to optimize over (the `n` most important parameters are chosen using an
        `XGBoost` feature importance analysis)
        """
        discrete_parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.BINARY,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for i in range(n_features)
        ]
        continuous_parameters = [
            Parameter(
                name=f"x{i + n_features}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for i in range(3)
        ]
        parameters = discrete_parameters + continuous_parameters
        super().__init__(parameters=parameters, noise_std=None)
        self.train_x, self.train_y, self.test_x, self.test_y = load_uci_data(
            n_features=n_features
        )

        self.flip_tensor = torch.tensor(RandomState(0).choice([0, 1], n_features))
        """
        the tensor used to flip the binary parameters
        """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim in [0, 1]:
            x = torch.unsqueeze(x, 0)
        assert x.ndim == 2
        assert x.shape[1] == self.n_features + 3

        rmses = []

        for _x in x:
            assert (
                len(_x) == self.n_features + 3
            ), f"Expected {self.n_features + 3} dimensions, got {len(_x)}"
            if self.flip:
                _x[self.binary_indices] = torch.abs(
                    _x[self.binary_indices] - self.flip_tensor
                )
            epsilon = 0.01 * 10 ** (2 * _x[-3])  # Default = 0.1
            C = 0.01 * 10 ** (4 * _x[-2])  # Default = 1.0
            gamma = (
                (1 / self.n_binary) * 0.1 * 10 ** (2 * _x[-1])
            )  # Default = 1.0 / self.n_features
            model = SVR(C=C.item(), epsilon=epsilon.item(), gamma=gamma.item())
            inds_selected = np.where(_x[np.arange(self.n_binary)].cpu().numpy() == 1)[0]
            if len(inds_selected) == 0:  # Silly corner case with no features
                rmses.append(1.0)
            else:
                model.fit(self.train_x[:, inds_selected], self.train_y)
                y_pred = model.predict(self.test_x[:, inds_selected])
                rmse = math.sqrt(((y_pred - self.test_y) ** 2).mean(axis=0).item())
                rmses.append(rmse)
        return torch.tensor(rmses, dtype=torch.float64)


@gin.configurable
class LassoDNA(SingularityBenchmark):
    """
    The 180D Lasso DNA benchmark from the Singularity Benchmark Suite
    """

    def __init__(
        self,
    ):
        """
        Initialize the benchmark function.
        """
        dim = 180
        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for i in range(dim)
        ]
        super().__init__(
            dim=dim,
            name="lasso_dna",
            parameters=parameters,
        )


@gin.configurable
class LassoSimple(SingularityBenchmark):
    """
    The 60D Lasso Simple benchmark from the Singularity Benchmark Suite
    """

    def __init__(
        self,
    ):
        """
        Initialize the benchmark function.
        """
        dim = 60
        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for i in range(dim)
        ]
        super().__init__(
            dim=dim,
            name="lasso_simple",
            parameters=parameters,
        )


@gin.configurable
class LassoMedium(SingularityBenchmark):
    """
    The 180D Lasso Medium benchmark from the Singularity Benchmark Suite
    """

    def __init__(
        self,
    ):
        """
        Initialize the benchmark function.
        """
        dim = 100
        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for i in range(dim)
        ]
        super().__init__(
            dim=dim,
            name="lasso_simple",
            parameters=parameters,
        )


@gin.configurable
class LassoHigh(SingularityBenchmark):
    """
    The 300D Lasso High benchmark from the Singularity Benchmark Suite
    """

    def __init__(
        self,
    ):
        """
        Initialize the benchmark function.
        """
        dim = 300
        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for i in range(dim)
        ]
        super().__init__(
            dim=dim,
            name="lasso_hard",
            parameters=parameters,
        )


@gin.configurable
class LassoHard(SingularityBenchmark):
    """
    The 1000D Lasso Hard benchmark from the Singularity Benchmark Suite
    """

    def __init__(
        self,
    ):
        """
        Initialize the benchmark function.
        """
        dim = 1000
        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for i in range(dim)
        ]
        super().__init__(
            dim=dim,
            name="lasso_hard",
            parameters=parameters,
        )


@gin.configurable
class Mopta08(SingularityBenchmark):
    """
    The 124D Mopta08 benchmark from the Singularity Benchmark Suite
    """

    def __init__(
        self,
    ):
        """
        Initialize the benchmark function.
        """
        dim = 124
        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for i in range(dim)
        ]
        super().__init__(
            dim=dim,
            name="mopta08",
            parameters=parameters,
        )


@gin.configurable
class MaxSat(Benchmark):
    """
    A MaxSat benchmark (this class will be subclassed for each MaxSat instance)
    """

    def __init__(
        self,
        instance_filename: str,
        normalize_weights: bool = True,
        negative_weights: bool = False,
    ):
        """
        Initialize the benchmark function.

        Args:
            instance_filename: the filename of the MaxSat instance
            normalize_weights: whether to normalize the weights to zero mean and unit standard deviation
            negative_weights: whether to use the negative weights (i.e. the weights of the unsatisfied clauses)
        """

        wcnf = WCNF(
            os.path.join(
                Path(__file__).parent.parent, "data", "maxsat", instance_filename
            )
        )
        dim = wcnf.nv
        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.BINARY,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            for i in range(dim)
        ]
        super().__init__(
            noise_std=None,
            parameters=parameters,
        )
        self.normalize_weights = normalize_weights
        """
        whether to normalize the weights to zero mean and unit standard deviation
        """
        self.negative_weights = negative_weights
        """
        whether to use the negative weights (i.e. the weights of the unsatisfied clauses)
        """
        self.weights = np.array(wcnf.weights, dtype=np.float64)
        """
        the weights of the clauses
        """
        self.total_weight = self.weights.sum()
        """
        the total weight of the clauses
        """
        # normalize weights to zero mean and unit standard deviation
        if self.normalize_weights:
            self.weights = (self.weights - self.weights.mean()) / self.weights.std()
        self.clauses = np.zeros((len(wcnf.clauses), self.dim), dtype=np.bool_)
        """
        the clauses of the MaxSat instance (one-hot encoded)
        """
        self.clause_idxs = []
        """
        the indices of the clauses
        """
        for i, clause in enumerate(wcnf.clauses):
            clause_idxs = np.abs(np.array(clause)) - 1
            self.clauses[i, clause_idxs] = np.array(clause) > 0
            self.clause_idxs.append(clause_idxs)
        self.random_flip = None
        """
        the random bit array used for flipping the bits
        """
        if self.flip:
            self.random_flip = RandomState(43).choice([False, True], size=self.dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().numpy().squeeze()
        if x.ndim == 1:
            x = x[np.newaxis, :]

        x = x.astype(np.bool_)

        fxs = []
        for _x in x:
            if self.random_flip is not None:
                _x = np.logical_xor(_x, self.random_flip)

            weights_sum = np.sum(
                self.weights
                * [
                    np.any(np.equal(_x[ci], self.clauses[i, ci]))
                    for i, ci in enumerate(self.clause_idxs)
                ]
            )
            if self.negative_weights:
                # weights of unsatisfied clauses
                weight_diff = self.total_weight - weights_sum
                fx = torch.tensor(weight_diff).unsqueeze(-1)
            else:
                fx = -torch.tensor(weights_sum).unsqueeze(-1)
            fxs.append(fx)
        return torch.cat(fxs, dim=0)


@gin.configurable
class MaxSat60(MaxSat):
    """
    The 60D MaxSat benchmark (see http://www.maxsat.udl.cat/11/benchmarks/index.html for more information)
    """

    def __init__(self, *args, **kwargs):
        if not pathlib.Path("data/maxsat/frb10-6-4.wcnf").exists():
            download_maxsat60_data()
        super().__init__(instance_filename="frb10-6-4.wcnf", *args, **kwargs)


@gin.configurable
class MaxSat125(MaxSat):
    """
    The 125D MaxSat benchmark (see https://maxsat-evaluations.github.io/2018/benchmarks.html for more information)
    """

    def __init__(self, *args, **kwargs):
        if not pathlib.Path(
            "data/maxsat/cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf"
        ).exists():
            download_maxsat125_data()
        super().__init__(
            instance_filename="cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf",
            *args,
            **kwargs,
        )


def generate_contamination_dynamics(
    random_seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the contamination dynamics for the Contamination benchmark.

    Args:
        random_seed: the random seed to use for the random number generator

    Returns:
        the initial contamination, the contamination lambdas, and the restoration gammas

    """
    n_stages = 25
    n_simulations = 100

    init_alpha = 1.0
    init_beta = 30.0
    contam_alpha = 1.0
    contam_beta = 17.0 / 3.0
    restore_alpha = 1.0
    restore_beta = 3.0 / 7.0
    init_Z = np.random.RandomState(random_seed).beta(
        init_alpha, init_beta, size=(n_simulations,)
    )
    lambdas = np.random.RandomState(random_seed).beta(
        contam_alpha, contam_beta, size=(n_stages, n_simulations)
    )
    gammas = np.random.RandomState(random_seed).beta(
        restore_alpha, restore_beta, size=(n_stages, n_simulations)
    )

    return init_Z, lambdas, gammas


def sample_init_points(n_vertices: int, n_points: int, random_seed: int = None):
    """
    Sample initial points for the Contamination benchmark.

    Args:
        n_vertices:
        n_points:
        random_seed:

    Returns:
        torch.Tensor: the initial points

    """
    if random_seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(random_seed)
    init_points = torch.empty(0).long()
    for _ in range(n_points):
        init_points = torch.cat(
            [
                init_points,
                torch.cat(
                    [torch.randint(0, int(elm), (1, 1)) for elm in n_vertices], dim=1
                ),
            ],
            dim=0,
        )
    if random_seed is not None:
        torch.set_rng_state(rng_state)
    return init_points


def _contamination(x: np.ndarray, cost, init_Z, lambdas, gammas, U, epsilon):
    assert x.size == 25

    rho = 1.0
    n_simulations = 100

    Z = np.zeros((x.size, n_simulations))
    Z[0] = (
        lambdas[0] * (1.0 - x[0]) * (1.0 - init_Z) + (1.0 - gammas[0] * x[0]) * init_Z
    )
    for i in range(1, 25):
        Z[i] = (
            lambdas[i] * (1.0 - x[i]) * (1.0 - Z[i - 1])
            + (1.0 - gammas[i] * x[i]) * Z[i - 1]
        )

    below_threshold = Z < U
    constraints = np.mean(below_threshold, axis=1) - (1.0 - epsilon)

    return np.sum(x * cost - rho * constraints)


@gin.configurable
class Ackley53(Benchmark):
    def __init__(self, lamda=1e-6, *args, **kwargs):
        """
        Initialize the benchmark function.

        Args:
            lamda: the lambda parameter of the Ackley function (random noise)
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        binary_parameters = [
            Parameter(
                name=f"x_{i}", type=ParameterType.BINARY, lower_bound=0, upper_bound=1
            )
            for i in range(50)
        ]
        continuous_parameters = [
            Parameter(
                name=f"x_{i + 50}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0,
                upper_bound=1,
            )
            for i in range(3)
        ]
        parameters = binary_parameters + continuous_parameters
        super().__init__(parameters=parameters, noise_std=None)

        self.binary_inds = list(range(len(binary_parameters)))
        self.continuous_inds = [50, 51, 52]
        self.n_vertices = 2 * torch.ones(len(binary_parameters), dtype=torch.long)
        self.config = self.n_vertices
        self.lamda = lamda
        self.random_flips = np.array(
            [
                1,
                1,
                1,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
            ]
        )

        self.feature_idxs = torch.arange(50)

    @staticmethod
    def _ackley(x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(x), axis=1) / 53))
        cos_term = -1 * np.exp(np.sum(np.cos(c * np.copy(x)) / 53, axis=1))
        result = a + np.exp(1) + sum_sq_term + cos_term
        return result

    def __call__(
        self,
        X,
    ):
        if type(X) == torch.Tensor:
            X = X.cpu().numpy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # To make sure there is no cheating, round the discrete variables before calling the function
        X[:, self.binary_inds] = np.round(X[:, self.binary_inds])
        if self.flip:
            X[:, self.binary_inds] = self.random_flips - X[:, self.binary_inds]
        X[:, self.continuous_inds] = -1 + 2 * X[:, self.continuous_inds]
        result = self._ackley(X)
        return torch.tensor(result + self.lamda * np.random.rand(*result.shape))


@gin.configurable
class Contamination(Benchmark):
    """
    Contamination Control Problem with the simplest graph
    """

    def __init__(self, effective_dim: int = 25, ambient_dim: int = 25, *args, **kwargs):
        """
        Initialize the benchmark function.

        Args:
            effective_dim: the effective dimensionality of the function
            ambient_dim: the ambient dimensionality of the function
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        self.ambient_dim = ambient_dim
        """
        The ambient dimensionality of the function
        """
        self.effective_dim = effective_dim
        """
        The effective dimensionality of the function
        """

        parameters = [
            Parameter(
                name=f"x_{i}", type=ParameterType.BINARY, lower_bound=0, upper_bound=1
            )
            for i in range(self.ambient_dim)
        ]
        super().__init__(parameters=parameters, noise_std=None, *args, **kwargs)
        self.lamda = 1e-2
        """
        The lambda parameter of the function
        """
        self.n_vertices = np.array([2] * self.effective_dim)
        """
        The number of vertices in the graph
        """
        self.suggested_init = torch.empty(0).long()
        """
        The suggested initial points
        """
        self.suggested_init = torch.cat(
            [
                self.suggested_init,
                sample_init_points(
                    self.n_vertices, 20 - self.suggested_init.size(0), random_seed=42
                ),
            ],
            dim=0,
        )
        self.adjacency_mat = []
        """
        The adjacency matrix of the graph
        """
        self.fourier_freq = []
        """
        The Fourier frequencies of the graph
        """
        self.fourier_basis = []
        """
        The Fourier basis of the graph
        """
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(
                torch.ones(n_v - 1), 1
            )
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.linalg.eigh(laplacian)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)
        # In all evaluation, the same sampled values are used.
        self.init_Z, self.lambdas, self.gammas = generate_contamination_dynamics(
            random_seed=42
        )

        if self.flip:
            self.random_flip = torch.tensor(
                RandomState(43).choice([True, False], size=(self.effective_dim,))
            )
        else:
            self.random_flip = None

    def __call__(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == self.ambient_dim
        return torch.cat(
            [self._evaluate_single(x[i]) for i in range(x.size(0))], dim=0
        ).to(dtype=torch.double)

    def _evaluate_single(self, x):
        assert x.dim() == 1
        assert x.numel() == self.ambient_dim
        if x.dim() == 2:
            x = x.squeeze(0)
        x = x[: self.effective_dim]
        if self.random_flip is not None:
            x = torch.logical_xor(x, self.random_flip).to(dtype=torch.float)
        evaluation = _contamination(
            x=(x.cpu() if x.is_cuda else x).numpy(),
            cost=np.ones(x.numel()),
            init_Z=self.init_Z,
            lambdas=self.lambdas,
            gammas=self.gammas,
            U=0.1,
            epsilon=0.05,
        )
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()


@gin.configurable
class Labs(Benchmark):
    """
    The low-autocorrelation binary sequence (LABS) benchmark
    """

    def __init__(self, dim: int = 50, *args, **kwargs):
        """
        Initialize the benchmark function.

        Args:
            dim: the dimensionality of the function
            *args: additional arguments
            **kwargs: additional keyword arguments
        """
        parameters = [
            Parameter(
                name=f"x{i}", type=ParameterType.BINARY, lower_bound=0, upper_bound=1
            )
            for i in range(dim)
        ]
        super().__init__(
            noise_std=None,
            parameters=parameters,
        )

        self.random_flip = None
        """
        the random bit array used for flipping the bits
        """
        if self.flip:
            self.random_flip = RandomState(43).choice([-1, 1], size=dim, replace=True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert x.ndim == 2
        fxs = []
        for _x in x:
            # set the 0s to -1
            _x[_x == 0] = -1
            if self.random_flip is not None:
                _x = _x * self.random_flip
            e = 0
            for k in range(1, self.dim):
                e += torch.square(torch.sum(_x[:-k] * _x[k:]))
            fx = -(self.dim**2 / (2 * e)).to(dtype=torch.double).unsqueeze(-1)
            fxs.append(fx)
        return torch.cat(fxs, dim=0)


@gin.configurable
class PestControl(Benchmark):
    """
    The pest control benchmark
    """

    def __init__(
        self,
        seed: int = 0,
        n_stages: int = 25,
        n_choice: int = 5,
        ambient_dim: int = 25,
    ):
        """
        Initialize the benchmark function.

        Args:
            seed: the random seed to use for the random number generator
            n_stages: the number of stages in the contamination chain
            n_choice: the number of possible interventions (0 is no intervention)
            ambient_dim: the ambient dimensionality of the function
        """
        assert (
            n_stages <= ambient_dim
        ), f"n_stages must be less than or equal to ambient_dim. n_stages: {n_stages}, ambient_dim: {ambient_dim}"

        self.n_stages = n_stages
        """
        The number of stages in the contamination chain
        """
        self.n_choice = n_choice
        """
        The number of possible interventions (0 is no intervention)
        """
        self.ambient_dim = ambient_dim
        """
        The ambient dimensionality of the function
        """

        parameters = [
            Parameter(
                name=f"x{i}",
                type=ParameterType.CATEGORICAL,
                lower_bound=0,
                upper_bound=self.n_choice - 1,
            )
            for i in range(self.ambient_dim)
        ]
        self.seed = seed
        super().__init__(
            parameters=parameters,
            noise_std=None,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert x.ndim == 2
        return torch.tensor([self._call(_x) for _x in x], dtype=torch.double)

    def _call(self, x):
        assert x.ndim == 1
        start = 0
        _x = []
        for parameter in self.parameters:
            end = start + parameter.dims_required
            one_hot = x[start:end]
            # transform onehot to categorical
            cat = torch.argmax(one_hot)
            _x.append(cat)
            start = end
        _x = np.array(_x)
        # only use the first n_stages
        _x = _x[: self.n_stages]
        if self.flip:
            _x = (
                _x + RandomState(self.seed).choice(self.n_choice, self.n_stages)
            ) % self.n_choice

        logging.info("evaluating PestControl with", _x)
        eval = _pest_control_score(_x, seed=self.seed)
        return eval
