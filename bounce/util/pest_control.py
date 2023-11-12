# Source: https://github.com/aryandeshwal/BODi/blob/main/bodi/pestcontrol.py
import logging

import numpy as np


# PESTCONTROL_N_CHOICE = 5
# PESTCONTROL_N_STAGES = 10


def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
    """
    Compute the spread of the pest in the next stage.

    Args:
        curr_pest_frac: the current fraction of the pest
        spread_rate: the rate of the spread
        control_rate: the rate of the control
        apply_control: whether to apply control

    Returns:

    """
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(x, seed=None):
    """
    Compute the pest control score.

    Args:
        x: the control actions for each stage
        seed: the seed for the random number generator

    Returns:
        the pest control score

    """
    logging.debug(f"running pest w/ seed {seed}")
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    if seed is not None:
        init_pest_frac = np.random.RandomState(seed).beta(
            init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,)
        )
    else:
        init_pest_frac = np.random.beta(
            init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,)
        )
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        if seed is not None:
            spread_rate = np.random.RandomState(seed).beta(
                spread_alpha, spread_beta, size=(n_simulations,)
            )
        else:
            spread_rate = np.random.beta(
                spread_alpha, spread_beta, size=(n_simulations,)
            )
        do_control = x[i] > 0
        if do_control:
            if seed is not None:
                control_rate = np.random.RandomState(seed).beta(
                    control_alpha, control_beta[x[i]], size=(n_simulations,)
                )
            else:
                control_rate = np.random.beta(
                    control_alpha, control_beta[x[i]], size=(n_simulations,)
                )
            next_pest_frac = _pest_spread(
                curr_pest_frac, spread_rate, control_rate, True
            )
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                1.0
                - control_price_max_discount[x[i]]
                / float(n_stages)
                * float(np.sum(x == x[i]))
            )
        else:
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold
