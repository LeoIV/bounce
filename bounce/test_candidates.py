import pytest
import torch

from bounce.candidates import hamming_distance


@pytest.fixture
def example_data():
    x = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 1, 0]])
    y = torch.tensor([1, 0, 1])
    expected_output = torch.tensor([3, 0, 2])
    return x, y, expected_output


def test_hamming_distance(example_data):
    x, y, expected_output = example_data
    output = hamming_distance(x, y)
    assert torch.all(output == expected_output)


def test_hamming_distance_with_unsqueeze():
    x = torch.tensor([0, 1, 0])
    y = torch.tensor([1, 0, 1])
    expected_output = torch.tensor([3])
    output = hamming_distance(x, y)
    assert torch.all(output == expected_output)


def test_hamming_distance_with_squeeze(example_data):
    x, y, expected_output = example_data
    output = hamming_distance(x, y)
    assert torch.all(output == expected_output)
