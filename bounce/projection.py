import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import gin
import numpy as np
import torch

from bounce.util.benchmark import Parameter, ParameterType
from bounce.util.data_handling import parameter_types


class BinSizing(Enum):
    """
    The way to bin the parameters. If "min", ordinal and categorical parameters use a bin with min(p_1, ..., p_n)
    realizations. If "max", ordinal and categorical parameters use a bin with max(p_1, ..., p_n) realizations.
    """

    MIN = "min"
    MAX = "max"


@dataclass
class Bin:
    """
    A bin is a collection of parameters that are binned together. Parameters within a bin are assumed to be of
    the same type. The bin is responsible for projecting a tensor of shape (n_samples, dims_required_for_bin) to a
    tensor of shape (n_samples, sum(bins_required_for_parameter(p) for p in parameters)).

    Args:
        parameters: the parameters to bin
        bin_sizing: the way to bin the parameters. If "min", ordinal and categorical parameters use a bin with min(p_1, ..., p_n)
            realizations. If "max", ordinal and categorical parameters use a bin with max(p_1, ..., p_n) realizations.
        forced_n_dims: if not None, the number of dimensions required for the bin. If None, the number of dimensions
            required for the bin is determined by the parameter type and bin_sizing.
        parameter_type: the type of the parameters in the bin. This is set automatically and should not be set manually.
    """

    parameters: list[Parameter]
    parameter_type: ParameterType = field(init=False)
    # the way to bin the parameters. if "min", ordinal and categorical parameters use a bin with min(p_1, ..., p_n)
    # realizations. if "max", ordinal and categorical parameters use a bin with max(p_1, ..., p_n) realizations.
    bin_sizing: BinSizing = BinSizing.MIN
    forced_n_dims: Optional[int] = None

    def __post_init__(self):
        # assert all parameters have the same type
        assert (
            len(set([p.type for p in self.parameters])) == 1
        ), "all parameters must have the same type"
        self.parameter_type = self.parameters[0].type

    @property
    def dims_required(self) -> int:
        """
        Returns the number of dimensions required for the bin.
        """
        if self.forced_n_dims is not None:
            return self.forced_n_dims
        match self.parameter_type:
            case ParameterType.CATEGORICAL:
                match self.bin_sizing:
                    case BinSizing.MIN:
                        f = min
                    case BinSizing.MAX:
                        f = max
                    case _:
                        raise ValueError(f"Unknown bin sizing {self.bin_sizing}")
                return f([p.dims_required for p in self.parameters])
            case ParameterType.CONTINUOUS:
                return 1
            case ParameterType.ORDINAL:
                return 1
            case ParameterType.BINARY:
                return 1
            case _:
                raise ValueError(f"Unknown parameter type {self.parameter_type}")

    def project_up(self, x: torch.Tensor, low_sequency: bool = False) -> torch.Tensor:
        """
        Projects a tensor of shape (n_samples, dims_required_for_bin) to a tensor of shape
        (n_samples, sum(bins_required_for_parameter(p) for p in parameters))

        Args:
            x: the tensor to project
            low_sequency: if True, the projection does not use random signs

        Returns:
            the projected tensor

        """
        # assert x is 2d
        assert len(x.shape) == 2, "x must be 2d"
        # assert x has the correct number of dimensions
        assert (
            x.shape[1] == self.dims_required
        ), "x must have the correct number of dimensions"
        # define output tensor
        output = -torch.ones(
            (x.shape[0], sum(p.dims_required for p in self.parameters)), dtype=x.dtype
        )
        # fill output tensor
        start = 0
        for parameter in self.parameters:
            end = start + parameter.dims_required
            match parameter.type:
                case ParameterType.CONTINUOUS:
                    # for continuous parameters, we just copy the input and possibly flip the sign
                    output[:, start:end] = x * (
                        1 if low_sequency else parameter.random_sign
                    )
                case ParameterType.BINARY:
                    # for binary parameters, we just copy the input and possibly flip the sign
                    output[:, start:end] = x * (
                        1 if low_sequency else parameter.random_sign
                    )
                case ParameterType.CATEGORICAL:
                    # assert only one non-``zero'' index per sample
                    assert (
                        torch.sum(x != -1, dim=1).max() == 1
                    ), "Exactly one non-``zero'' index per sample is required"
                    # get indices of non-zero elements
                    k = torch.argmax(x, dim=1)
                    # get output indices
                    v = torch.ceil(k * parameter.dims_required / self.dims_required).to(
                        dtype=torch.long
                    )
                    # add random sign unless low_sequency
                    v = (
                        v + (0 if low_sequency else parameter.random_sign)
                    ) % parameter.dims_required
                    # create output tensor
                    output[torch.arange(output.shape[0]), start + v] = 1
                case ParameterType.ORDINAL:
                    raise NotImplementedError("ordinal parameters not yet implemented")
            start = end
        return output

    def split(self, n_new_bins: int) -> list["Bin"]:
        """
        Splits the bin into 1+new_bins bins.

        Args:
            n_new_bins: the number of new bins to create
        """
        # assert n_new_bins > 0
        assert n_new_bins > 0, "n_new_bins must be positive"
        # assert n_new_bins is an integer
        assert n_new_bins == int(n_new_bins), "n_new_bins must be an integer"
        # assert n_new_bins is not larger than the number of parameters
        assert n_new_bins < len(
            self.parameters
        ), "n_new_bins must be smaller than the number of parameters"

        parameter_indices = np.arange(len(self.parameters))
        np.random.shuffle(parameter_indices)
        # split into n_new_bins + 1 bins
        bins = np.array_split(parameter_indices, n_new_bins + 1)
        # create new bins
        new_bins = [Bin([self.parameters[i] for i in b]) for b in bins]
        if (
            self.parameter_type == ParameterType.CATEGORICAL
            or self.parameter_type == ParameterType.ORDINAL
        ):
            # if the parameter type is categorical or ordinal, we need to set the number of dimensions for the new bins
            # to keep the same mapping as before
            for bi in new_bins:
                bi.forced_n_dims = self.dims_required
        return new_bins


@gin.configurable
class AxUS:
    def __init__(
        self,
        parameters: list[Parameter],
        n_bins: int,
        bin_sizing: BinSizing = BinSizing.MIN,
        low_sequency: bool = False,
    ):
        """
        Initializes the AxUS class.

        Args:
            parameters:the parameters to use
            n_bins: the number of bins to use
            bin_sizing: the way to bin the parameters. if "min", ordinal and categorical parameters use a bin with
                min(p_1, ..., p_n), if "max", ordinal and categorical parameters use a bin with max(p_1, ..., p_n)
                realizations.
            low_sequency: if True, bins do not use random signs
        """
        self.parameters = parameters
        self.n_bins = n_bins
        self.bin_sizing = bin_sizing
        self.low_sequency = low_sequency

        self.bins: list[Bin] = []

        assert n_bins >= len(
            parameter_types(parameters)
        ), "n_bins must be at least as large as the number of parameter types"

        self._reset()

        self._param_indices = dict()
        start = 0
        for i, p in enumerate(self.parameters):
            end = start + p.dims_required
            self._param_indices[p.name] = torch.arange(start, end)
            start = end

    def parameter_indices(self, parameter: Parameter) -> torch.Tensor:
        """
        Returns the indices of the parameters in the input tensor.

        Args:
            parameter: the parameter to get the indices for

        Returns:
            the indices of the parameter in the input tensor

        """
        return self._param_indices[parameter.name]

    @property
    def input_dim(self) -> int:
        """

        Returns:
            the number of dimensions required for the input tensor

        """
        return sum(p.dims_required for p in self.parameters)

    @property
    def target_dim(self) -> int:
        """

        Returns:
            the number of dimensions required for the output tensor

        """
        assert len(self.bins) > 0, "bins have not been initialized yet"
        return sum([b.dims_required for b in self.bins])

    def _reset(self):
        """
        Resets the bins to the initial state.

        Returns:
            None

        """
        # find indices for every parameter type
        self.bins = []
        parameter_type_indices = dict()
        bins_per_type = dict()
        for parameter_type in ParameterType:
            _parameter_type_indices = [
                i for i, p in enumerate(self.parameters) if p.type == parameter_type
            ]
            # if there are parameters of this type, add them to the dictionary containing the indices for every type
            if len(_parameter_type_indices) > 0:
                parameter_type_indices[parameter_type] = _parameter_type_indices
                bins_per_type[parameter_type] = max(
                    1,
                    math.floor(
                        (len(_parameter_type_indices) / (len(self.parameters)))
                        * self.n_bins
                    ),
                )

        if sum(bins_per_type.values()) < self.n_bins:
            # increase the number of bins for a random parameter type
            random_parameter_type = np.random.choice(
                [t for t in parameter_type_indices.keys()]
            )
            bins_per_type[random_parameter_type] += 1

        while sum(bins_per_type.values()) > self.n_bins:
            # decrease the number of bins for a random parameter type
            random_parameter_type = np.random.choice(
                [t for t in parameter_type_indices.keys() if bins_per_type[t] > 1]
            )
            bins_per_type[random_parameter_type] -= 1

        # find the number of bins for every parameter type
        for parameter_type, _parameter_type_indices in parameter_type_indices.items():
            n_bins = bins_per_type[parameter_type]
            logging.debug(
                f"Parameter type {parameter_type} gets {n_bins}/{self.n_bins} bins."
            )
            if (
                parameter_type == ParameterType.CONTINUOUS
                or parameter_type == ParameterType.BINARY
            ):
                index_permutation = np.random.permutation(
                    torch.tensor(_parameter_type_indices),
                )
            elif (
                parameter_type == ParameterType.CATEGORICAL
                or parameter_type == ParameterType.ORDINAL
            ):
                # TODO this should be changed for benchmarks with varying number of categories
                # so that variables with similar number of categories are in the same bin
                index_permutation = np.random.permutation(
                    torch.tensor(_parameter_type_indices),
                )
            else:
                raise ValueError(f"Unknown parameter type {parameter_type}")
            index_permutation = torch.tensor(index_permutation, dtype=torch.long)

            input_dim_bins = torch.tensor_split(index_permutation, n_bins)
            for input_dim_bin in input_dim_bins:
                self.bins.append(
                    Bin(
                        parameters=[self.parameters[i] for i in input_dim_bin],
                        bin_sizing=self.bin_sizing,
                    )
                )
        logging.debug(f"Total number of bins: {len(self.bins)}")

    def project_up(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects a tensor of shape (n_samples, dims_required_for_bin) to a tensor of shape
        (n_samples, sum(bins_required_for_parameter(p) for p in parameters))

        Returns:
            the projected tensor

        """
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        # assert x is 2d
        assert len(x.shape) == 2, "x must be 2d"
        x = x.t()
        # assert x has the correct number of dimensions
        assert (
            x.shape[1] == self.target_dim
        ), "x must have the correct number of dimensions"
        # define output tensor
        output = torch.zeros((x.shape[0], self.input_dim), dtype=x.dtype)
        # fill output tensor
        input_start = 0
        for bin in self.bins:
            input_end = input_start + bin.dims_required
            indices = torch.concat([self.parameter_indices(p) for p in bin.parameters])
            bp = bin.project_up(
                x[:, input_start:input_end], low_sequency=self.low_sequency
            )
            output[:, indices] = bp
            input_start = input_end
        return output.t()

    @property
    def bin_indices(self) -> list[torch.Tensor]:
        """

        Returns:
            a list of tensors containing the indices of the parameters in each bin

        """
        bin_indices = []
        start = 0
        for bin in self.bins:
            end = start + bin.dims_required
            bin_indices.append(torch.arange(start, end))
            start = end
        return bin_indices

    def n_bins_of_type(self, parameter_type: ParameterType) -> int:
        """
        Returns the number of bins of a certain type.

        Args:
            parameter_type: the type of the parameters

        Returns:
            the number of bins of that type

        """
        return sum([1 for b in self.bins if b.parameters[0].type == parameter_type])

    def bins_of_type(self, parameter_type: ParameterType) -> list[Bin]:
        """
        Returns the bins of a certain type.

        Args:
            parameter_type: the type of the parameters

        Returns:
            a list of bins

        """
        return [b for b in self.bins if b.parameters[0].type == parameter_type]

    def bins_and_indices_of_type(
        self, parameter_type: ParameterType
    ) -> list[tuple[Bin, torch.Tensor]]:
        """
        Returns the bins and their indices of a certain type.

        Args:
            parameter_type: the type of the parameters

        Returns:
            a list of tuples containing the bins and their indices

        """
        return [
            (b, i)
            for b, i in zip(self.bins, self.bin_indices)
            if b.parameters[0].type == parameter_type
        ]

    def split(self, n_new_bins: int) -> dict[torch.Tensor, list[torch.Tensor]]:
        """
        Splits the bins into n_new_bins bins.

        Returns:
            a dictionary that maps old (and current) bin indices to new bin indices

        """
        # assert n_new_bins is an integer
        assert n_new_bins == int(n_new_bins), "n_new_bins must be an integer"

        b_old = []
        b_new = []

        # define a mapping that maps old bin indices to new bin indices
        index_mapping: dict[torch.Tensor, list[torch.Tensor]] = dict()

        # keep track of the number of new indices added
        indices_added = 0

        # target dim before splitting
        target_dim_old = self.target_dim

        for bin, bin_indcs in zip(self.bins, self.bin_indices):
            n_new = min(n_new_bins, len(bin.parameters) - 1)
            if n_new == 0:
                b_old.append(bin)
                logging.debug(
                    f"Bin of type {bin.parameter_type} is not split because it only contains one parameter."
                )
                index_mapping[bin_indcs] = []
                continue
            bs = bin.split(n_new)
            b_old.append(bs[0])
            b_new.extend(bs[1:])
            index_mapping[bin_indcs] = []
            for _new in range(n_new):
                index_mapping[bin_indcs].append(
                    torch.arange(
                        target_dim_old + indices_added,
                        target_dim_old + indices_added + bs[_new + 1].dims_required,
                    )
                )
                indices_added += bs[_new + 1].dims_required
        self.bins = b_old + b_new
        self.n_bins = len(self.bins)
        return index_mapping
