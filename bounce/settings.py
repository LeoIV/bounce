from gpytorch.constraints import Interval

# GP SETTINGS
LENGTHSCALE_CONSTRAINT = Interval(0.005, 20.0)
NOISE_CONSTRAINT = Interval(5e-4, 0.2)

# ALGORITHM SETTINGS

MLL_FITTING_ITERATIONS: int = 100
ADJUST_NUMBER_OF_NEW_BINS: bool = True
NAME: str = "Bounce"
