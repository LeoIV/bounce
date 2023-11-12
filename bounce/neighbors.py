import torch

from bounce.projection import AxUS
from bounce.util.benchmark import ParameterType


def hamming_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Hamming distance between a set of points (x) and a vector (y)

    Args:
        x: The set of points
        y: The second vector

    Returns:
        The Hamming distance between the points and the vector
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    assert len(x.shape) == 2, "x must be a matrix"
    if len(y.shape) == 2:
        y = y.squeeze()
    assert len(y.shape) == 1, "y must be a vector"

    return torch.sum(x != y, dim=1)


def hamming_neighbors_within_tr(
    x: torch.Tensor,
    x_center: torch.Tensor,
    tr_length: torch.Tensor,
    axus: AxUS,
) -> torch.Tensor:
    """
    Find the neighbors of the points in x that are within Hamming distance 1 and still the trust region

    Args:
        x: The points to compute the neighbors for
        x_center: The center of the trust region
        tr_length: The length of the trust region
        axus: The AxUS embedding

    Returns:
        The neighbors of the points in x that are within Hamming distance 1 and still the trust region
    """
    x = torch.clone(x)
    if len(x.shape) == 2:
        x = x.squeeze()
    assert len(x.shape) == 1, "x must be a vector"

    discrete_parameter_types = [
        pt for pt in ParameterType if pt != ParameterType.CONTINUOUS
    ]

    neighbors_for_type = dict()

    for parameter_type in discrete_parameter_types:
        if axus.n_bins_of_type(parameter_type) == 0:
            # No parameters of this type
            continue
        if parameter_type == ParameterType.BINARY:
            indices = torch.tensor(
                [i for b, i in axus.bins_and_indices_of_type(parameter_type)]
            )
            diagonal = torch.zeros_like(x)
            diagonal[indices] = 1
            diag_nonzero = diagonal != 0

            type_neighbors = torch.abs(torch.diag(diagonal) - x.unsqueeze(0))[
                diag_nonzero, :
            ]
        elif parameter_type == ParameterType.CATEGORICAL:
            indicess = [i for b, i in axus.bins_and_indices_of_type(parameter_type)]
            type_neighbors = torch.zeros((0, len(x)), device=x.device)
            for indices in indicess:
                # find inactive indices
                inactive_indices = [i for i in indices if x[i] == 0]
                # create len(inactive_index) copies of x
                x_copies = torch.repeat_interleave(
                    x.unsqueeze(0), len(inactive_indices), dim=0
                )
                x_copies[:, indices] = 0
                for i, inactive_index in enumerate(inactive_indices):
                    x_copies[i, inactive_index] = 1
                # vstack x_copies to type_neighbors
                type_neighbors = torch.vstack((type_neighbors, x_copies))
        elif parameter_type == ParameterType.ORDINAL:
            raise NotImplementedError("Ordinal parameters are not supported yet")
        else:
            raise ValueError(f"Unknown parameter type {parameter_type}")

        # add type_neighbors to neighbors_for_type
        neighbors_for_type[parameter_type] = type_neighbors

    # stack all neighbors
    neighbors = torch.vstack(
        [type_neighbors for type_neighbors in neighbors_for_type.values()]
    )
    # remove duplicates
    neighbors = torch.unique(neighbors, dim=0)
    # remove the original point
    neighbors = neighbors[torch.any(neighbors != x, dim=1), :]
    # remove the neighbors that are not within the trust region
    neighbors = neighbors[hamming_distance(neighbors, x_center) <= tr_length, :]
    return neighbors
