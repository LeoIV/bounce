import gpytorch
import torch
from botorch.models import SingleTaskGP

from bounce.gaussian_process import fit_mll, get_gp
from bounce.projection import AxUS
from bounce.util.benchmark import Parameter, ParameterType


def test_get_gp():
    # create some dummy data
    x = torch.randn(10, 2)
    fx = torch.randn(10)
    axus = AxUS(
        parameters=[
            Parameter(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                lower_bound=0,
                upper_bound=1,
            )
            for i in range(10)
        ],
        n_bins=3,
    )

    # test that the function returns a tuple
    gp, train_x, train_fx = get_gp(axus, x, fx)
    assert isinstance(gp, gpytorch.models.ExactGP)
    assert isinstance(train_x, torch.Tensor)
    assert isinstance(train_fx, torch.Tensor)

    # test that the function returns a GP model with the correct input and output dimensions
    assert gp.train_inputs[0].shape == (10, 2)
    assert len(gp.train_targets.shape) == 1
    assert gp.train_targets.numel() == 10

    # test that the function returns a GP model with the correct kernel
    assert isinstance(gp.covar_module, gpytorch.kernels.ScaleKernel)
    assert isinstance(gp.covar_module.base_kernel, gpytorch.kernels.MaternKernel)

    # test that the function returns a GP model with the correct likelihood
    assert isinstance(gp.likelihood, gpytorch.likelihoods.GaussianLikelihood)

    # test that the function returns the correct input and output points
    assert torch.all(train_x == x)
    assert torch.all(train_fx == fx.unsqueeze(1))


def test_fit_mll():
    # create some dummy data
    x = torch.randn(10, 2)
    fx = torch.randn(10).reshape(-1, 1)
    gp = SingleTaskGP(x, fx)

    # test that the function fits the GP model without errors
    fit_mll(gp, x, fx)

    # test that the function fits the GP model with the correct optimizer
    assert isinstance(gp.covar_module, gpytorch.kernels.ScaleKernel)
    assert isinstance(gp.covar_module.base_kernel, gpytorch.kernels.MaternKernel)
    assert isinstance(gp.likelihood, gpytorch.likelihoods.GaussianLikelihood)

    # test that the function fits the GP model with the correct input and output dimensions
    assert gp.train_inputs[0].shape == (10, 2)
    assert len(gp.train_targets.shape) == 1
    assert gp.train_targets.numel() == 10
