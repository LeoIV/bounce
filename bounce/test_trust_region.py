import numpy as np
import torch

from bounce.trust_region import TrustRegion, update_tr_state


def test_reset_trust_region():
    tr = TrustRegion(
        dimensionality=80,
        length_init_discrete=40,
        length_init_continuous=0.8,
    )

    tr.length_discrete = 10
    tr.length_discrete_continuous = 10
    tr.length_continuous = 10

    tr.reset()

    assert tr.length_discrete == 40
    assert tr.length_discrete_continuous == 40
    assert tr.length_continuous == 0.8
    assert not tr.terminated


def test_tr_length_limited_to_dimensionality():
    tr = TrustRegion(
        dimensionality=2,
        length_init_discrete=40,
        length_init_continuous=0.8,
    )

    assert tr.length_discrete == 2
    assert tr.length_discrete_continuous == 2
    assert tr.length_continuous == 0.8
    assert not tr.terminated


def test_update_tr_state_improve():
    tr = TrustRegion(
        dimensionality=80,
        length_init_discrete=40,
        length_init_continuous=0.8,
    )

    length_discrete_continuous_before = tr.length_discrete_continuous
    length_continuous_before = tr.length_continuous

    fx_next = torch.tensor(
        [
            0.5,
        ]
    ).reshape(1, 1)

    fx_incumbent = torch.tensor(
        [
            1,
        ]
    ).reshape(1, 1)

    update_tr_state(
        trust_region=tr,
        fx_next=fx_next,
        fx_incumbent=fx_incumbent,
        adjustment_factor=np.array(0.5),
    )

    assert tr.length_discrete_continuous == 2 * length_discrete_continuous_before
    assert tr.length_continuous == 2 * length_continuous_before


def test_update_tr_state_worsen():
    tr = TrustRegion(
        dimensionality=80,
        length_init_discrete=40,
        length_init_continuous=0.8,
    )

    length_discrete_continuous_before = tr.length_discrete_continuous
    length_continuous_before = tr.length_continuous

    fx_next = torch.tensor(
        [
            1,
        ]
    ).reshape(1, 1)

    fx_incumbent = torch.tensor(
        [
            0.5,
        ]
    ).reshape(1, 1)

    update_tr_state(
        trust_region=tr,
        fx_next=fx_next,
        fx_incumbent=fx_incumbent,
        adjustment_factor=np.array(0.5),
    )

    assert tr.length_discrete_continuous == 0.5 * length_discrete_continuous_before
    assert tr.length_continuous == 0.5 * length_continuous_before
