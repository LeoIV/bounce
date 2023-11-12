import pytest
import torch

from bounce.kernel.categorical_mixture import MixtureKernel


def test_mixture_kernel():
    kern = MixtureKernel(discrete_dims=[0, 1], continuous_dims=[2, 3])

    with pytest.raises(AssertionError):
        # Discrete and continuous dims must be disjoint.
        kern = MixtureKernel(discrete_dims=[0, 1], continuous_dims=[1, 2])


def test_mixture_kernel_lamda():
    kern = MixtureKernel(discrete_dims=[0, 1], continuous_dims=[2, 3], lamda=0.5)

    # Check that the lengthscale is set correctly.
    assert kern.lamda == 0.5

    # Check that the lengthscale can be set.
    kern.lamda = 0.1
    assert kern.lamda == 0.1

    # Check that lambda is set to 1 if larger than 1.
    kern.lamda = 1.1
    assert kern.lamda == 1.0

    # Check that lambda is set to 0 if smaller than 0.
    kern.lamda = -0.1
    assert kern.lamda == 0.0


def test_call():
    kern = MixtureKernel(discrete_dims=[0, 1], continuous_dims=[2, 3], lamda=0.5)

    # Check that the kernel can be called.
    x = torch.randn(10, 4)
    kern(x, x)

    # Check that the kernel can be called with a single input.
    kern(x)
