import logging
import lzma
import math
import os.path
import zlib
from datetime import datetime
from typing import Optional, Union

import gin
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from torch import Size
from tqdm import tqdm

from bounce import settings
from bounce.benchmarks import Benchmark
from bounce.candidates import create_candidates_continuous, create_candidates_discrete
from bounce.gaussian_process import fit_mll, get_gp
from bounce.projection import AxUS, Bin
from bounce.trust_region import TrustRegion, update_tr_state
from bounce.util.benchmark import ParameterType
from bounce.util.data_handling import (
    construct_mixed_point,
    from_1_around_origin,
    join_data,
    sample_binary,
    sample_categorical,
    sample_continuous,
)
from bounce.util.printing import BColors


@gin.configurable
class Bounce:
    """
    Bounce class: implements the Bounce algorithm.

    The main method is `run()` which runs the algorithm.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        number_initial_points: int,
        initial_target_dimensionality: int,
        number_new_bins_on_split: int,
        maximum_number_evaluations: int,
        batch_size: int,
        results_dir: str,
        desired_final_dimensionality: Optional[int] = None,
        maximum_number_evaluations_until_input_dim: Optional[int] = None,
        max_cholesky_size: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: Optional[str] = None,
        use_scipy_lbfgs: bool = True,
        max_lbfgs_iters: Optional[int] = None,
        min_cuda: int = 10,
        n_interleaved: int = 5,
    ):
        """
        Init

        Args:
            benchmark: the benchmark to be used
            number_initial_points: the number of initial points to be sampled
            initial_target_dimensionality: the dimensionality in which `Bounce` starts the optimization
            number_new_bins_on_split: the number of new bins to be created on each split (if applicable)
            maximum_number_evaluations: the maximum number of function evaluations
            batch_size: the batch size to be used
            results_dir: the directory where the results will be stored
            desired_final_dimensionality: the dimensionality in which `Bounce` terminates the optimization
            maximum_number_evaluations_until_input_dim: the maximum number of function evaluations until the input
            max_cholesky_size: the maximum size of the Cholesky decomposition
            device: the device to be used (cpu or cuda)
            dtype: the dtype to be used (float32 or float64)
            use_scipy_lbfgs: whether to use scipy's LBFGS implementation or the backup Adam optimizer for the GP fitting
            max_lbfgs_iters: maximum iterations until we run LBFGS, after that use Adam
            min_cuda: the minimum number of data points to use cuda
            n_interleaved: the number of interleaved steps when optimizing mixed benchmarks

        """
        self.benchmark = benchmark
        """
        The benchmark to be used
        """
        self.number_initial_points = number_initial_points
        """
        the number of initial points to be sampled
        """
        self.initial_target_dimensionality = initial_target_dimensionality
        """
        the dimensionality in which `Bounce` starts the optimization
        """
        self.number_new_bins_on_split = number_new_bins_on_split
        """
        the number of new bins to be created on each split (if applicable)
        """
        self.maximum_number_evaluations = maximum_number_evaluations
        """
        the maximum number of function evaluations
        """
        self.batch_size = batch_size
        """
        the batch size to be used
        """
        self.max_cholesky_size = max_cholesky_size
        """
        the maximum size of the Cholesky decomposition
        """
        self.use_scipy_lbfgs = use_scipy_lbfgs
        """
        whether to use scipy's LBFGS implementation or the backup Adam optimizer for the GP fitting
        """
        self.device = device
        """
        the device to be used (cpu or cuda)
        """
        self.min_cuda = min_cuda
        """
        the minimum number of data points to use cuda
        """
        self.max_lbfgs_iters = max_lbfgs_iters  # maximum iterations until we run LBFGS, after that use Adam
        """
        maximum iterations until we run LBFGS, after that use Adam
        """
        self.n_interleaved = n_interleaved
        """
        the number of interleaved steps when optimizing mixed benchmarks
        """
        self.dtype = None
        """
        the dtype to be used (float32 or float64)
        """
        if dtype is None:
            self.dtype = torch.float64
        else:
            match dtype:
                case "float32":
                    self.dtype = torch.float32
                case "float64":
                    self.dtype = torch.float64
                case _:
                    raise ValueError(f"Unknown dtype {dtype}")

        # defining results directory
        now = datetime.now()
        gin_config_str = gin.config_str()
        adler = zlib.adler32(gin_config_str.encode("utf-8"))

        fname = now.strftime("%m-%d-%Y-%H:%M:%S:%f")
        self.results_dir = os.path.join(results_dir, str(adler), fname)
        """
        the directory where the results will be stored
        """
        os.makedirs(self.results_dir, exist_ok=True)
        # save gin config to file
        with open(os.path.join(self.results_dir, "gin_config.txt"), "w") as f:
            f.write(gin.config_str())

        self.desired_final_dimensionality = None
        """
        the dimensionality in which `Bounce` terminates the optimization
        """
        if desired_final_dimensionality is None:
            self.desired_final_dimensionality = self.benchmark.dim
        else:
            self.desired_final_dimensionality = desired_final_dimensionality

        self.maximum_number_evaluations_until_input_dim = None
        """
        the maximum number of function evaluations until the input dimensionality is reached
        """
        if maximum_number_evaluations_until_input_dim is None:
            self.maximum_number_evaluations_until_input_dim = (
                self.maximum_number_evaluations
            )
        else:
            assert (
                maximum_number_evaluations_until_input_dim
                <= self.maximum_number_evaluations
            )
            self.maximum_number_evaluations_until_input_dim = (
                maximum_number_evaluations_until_input_dim
            )

        self.tr_splits = 0
        """
        the number of splits that have been performed (the target dimensionality has been increased)
        """

        # PRIVATE ATTRIBUTES

        self._n_evals = 0
        self._n_splits = math.ceil(
            math.log(
                self.desired_final_dimensionality / self.initial_target_dimensionality,
                self.number_new_bins_on_split + 1,
            )
        )

        if settings.ADJUST_NUMBER_OF_NEW_BINS:
            self._adjust_number_bins_on_split()
        logging.info(f"🤖 {settings.NAME} will split at most {self._n_splits} times.")
        self.split_budget = self._split_budget(self.initial_target_dimensionality)
        """
        the budget for the current target dimensionality
        """

        self.random_embedding = AxUS(
            parameters=self.benchmark.parameters,
            n_bins=self.initial_target_dimensionality,
        )
        """
        the random embedding used for the low-dimensional target space (only `AxUS` used)
        """

        self.trust_region = TrustRegion(dimensionality=self.random_embedding.target_dim)
        """
        a `TrustRegion` instance to model the trust region
        """

        # saving global data
        self.x_global = torch.empty(
            0, self.random_embedding.target_dim, dtype=self.dtype
        )
        """
        the input points in the low-dimensional target space
        """
        self.x_up_global = torch.empty(
            0, self.benchmark.representation_dim, dtype=self.dtype
        )
        """
        the input points in the high-dimensional representation space
        """
        self.fx_global = torch.empty(0, dtype=self.dtype)
        """
        the function values at the input points
        """

        # tr local data
        self._reset_local_data()

        all_target_dims = self.initial_target_dimensionality * torch.pow(
            1 + self.number_new_bins_on_split, torch.arange(0, self._n_splits + 1)
        )
        self._all_split_budgets = {
            i.item(): self._split_budget(i.item()) for i in all_target_dims
        }

    def _reset_local_data(self):
        # saving tr local data
        self.x_tr = torch.empty(0, self.random_embedding.target_dim, dtype=self.dtype)
        self.x_up_tr = torch.empty(
            0, self.benchmark.representation_dim, dtype=self.dtype
        )
        self.fx_tr = torch.empty(0, dtype=self.dtype)

    def _adjust_number_bins_on_split(self):
        """
        Adjusts the number of new bins on split to the number of new bins that minimizes the difference between the
        true final target dimensionality and the desired final dimensionality.

        Returns:
            None

        """
        possible_bin_sizes = torch.arange(
            1,
            torch.floor(
                torch.log2(torch.tensor(self.desired_final_dimensionality))
            ).item(),
        )

        if possible_bin_sizes.numel() == 0:
            possible_bin_sizes = torch.tensor([1])

        best_bin_size = (
            torch.argmin(
                torch.abs(
                    self.initial_target_dimensionality
                    * (1 + possible_bin_sizes) ** self._n_splits
                    - self.desired_final_dimensionality
                )
            )
            + 1
        )
        if best_bin_size != self.number_new_bins_on_split:
            logging.debug(
                f"Updating number of new bins from {self.number_new_bins_on_split} to {best_bin_size}"
            )
            self.number_new_bins_on_split = best_bin_size.item()
            self._n_splits = math.ceil(
                math.log(
                    self.desired_final_dimensionality
                    / self.initial_target_dimensionality,
                    self.number_new_bins_on_split + 1,
                )
            )

    def _split_budget(self, target_dimensionality: int) -> int:
        """
        Calculates the number of evaluations to be used for the split with target_dimensionality.

        Args:
            target_dimensionality: the target dimensionality of the split

        Returns:
            the number of evaluations to be used for the split with target_dimensionality

        """
        total_budget = (
            self.maximum_number_evaluations_until_input_dim - self.number_initial_points
        )

        if target_dimensionality >= self.benchmark.dim:
            return min(
                10 * target_dimensionality,
                self.maximum_number_evaluations - self._n_evals,
            )
        split_budget = round(
            -(self.number_new_bins_on_split * total_budget * target_dimensionality)
            / (
                self.initial_target_dimensionality
                * (1 - (self.number_new_bins_on_split + 1) ** (self._n_splits + 1))
            )
        )
        return min(2**target_dimensionality, split_budget)

    def sample_init(self):
        """
        Samples the initial points, evaluates them, and adds them to the observations.
        Increases the number of evaluations by the number of initial points.

        Returns:
            None

        """
        types_points_and_indices = {pt: (None, None) for pt in ParameterType}
        # sample initial points for each parameter type present in the benchmark
        for parameter_type in self.benchmark.unique_parameter_types:
            # find number of parameters of type parameter_type
            bins_of_type: list[Bin] = self.random_embedding.bins_of_type(parameter_type)
            indices_of_type = torch.concat(
                [
                    self.random_embedding.bins_and_indices_of_type(parameter_type)[i][1]
                    for i in range(len(bins_of_type))
                ]
            )
            match parameter_type:
                case ParameterType.BINARY:
                    _x_init = sample_binary(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                    )
                case ParameterType.CONTINUOUS:
                    _x_init = sample_continuous(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                    )
                case ParameterType.CATEGORICAL:
                    _x_init = sample_categorical(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                    )
                case ParameterType.ORDINAL:
                    raise NotImplementedError(
                        "Ordinal parameters are not supported yet."
                    )
                case _:
                    raise ValueError(f"Unknown parameter type {parameter_type}.")
            types_points_and_indices[parameter_type] = (_x_init, indices_of_type)

        x_init = construct_mixed_point(
            size=self.number_initial_points,
            binary_indices=types_points_and_indices[ParameterType.BINARY][1],
            continuous_indices=types_points_and_indices[ParameterType.CONTINUOUS][1],
            categorical_indices=types_points_and_indices[ParameterType.CATEGORICAL][1],
            ordinal_indices=types_points_and_indices[ParameterType.ORDINAL][1],
            x_binary=types_points_and_indices[ParameterType.BINARY][0],
            x_continuous=types_points_and_indices[ParameterType.CONTINUOUS][0],
            x_categorical=types_points_and_indices[ParameterType.CATEGORICAL][0],
            x_ordinal=types_points_and_indices[ParameterType.ORDINAL][0],
        )

        x_init_up = from_1_around_origin(
            x=self.random_embedding.project_up(x_init.T).T,
            lb=self.benchmark.lb_vec,
            ub=self.benchmark.ub_vec,
        )
        fx_init = self.benchmark(x_init_up)

        self._add_data_to_tr_observations(
            xs_down=x_init,
            xs_up=x_init_up,
            fxs=fx_init,
        )

        self._n_evals += self.number_initial_points

    def run(self):
        """
        Runs the algorithm.

        Returns:
            None

        """

        self.sample_init()

        while self._n_evals <= self.maximum_number_evaluations:
            axus = self.random_embedding
            x = self.x_tr
            fx = self.fx_tr

            # normalize data
            mean = torch.mean(fx)
            std = torch.std(fx)
            if std == 0:
                std += 1
            fx_scaled = (fx - mean) / std
            x_scaled = (x + 1) / 2

            if self.device == "cuda":
                fx_scaled = fx_scaled.to(self.device)
                x_scaled = x_scaled.to(self.device)

            # Select the kernel
            model, train_x, train_fx = get_gp(
                axus=axus,
                x=x_scaled,
                fx=-fx_scaled,
            )

            use_scipy_lbfgs = self.use_scipy_lbfgs and (
                self.max_lbfgs_iters is None or len(train_x) <= self.max_lbfgs_iters
            )
            fit_mll(
                model=model,
                train_x=train_x,
                train_fx=-train_fx,
                max_cholesky_size=self.max_cholesky_size,
                use_scipy_lbfgs=use_scipy_lbfgs,
            )
            acquisition_function = None
            sampler = None

            if self.batch_size > 1:
                # we don't set the acquisition function here, because it needs to be redefined
                # for each batch item to be able to condition on the earlier batch items
                # note that this is the only place where we don't use the acquisition function
                sampler = SobolQMCNormalSampler(Size([1024]), seed=self._n_evals)
            else:
                # use analytical EI for batch size 1
                acquisition_function = ExpectedImprovement(
                    model=model, best_f=(-fx_scaled).max().item()
                )

            if self.benchmark.is_discrete:
                x_best, fx_best, tr_state = create_candidates_discrete(
                    x_scaled=x_scaled,
                    fx_scaled=fx_scaled,
                    model=model,
                    axus=axus,
                    trust_region=self.trust_region,
                    device=self.device,
                    batch_size=self.batch_size,
                    acquisition_function=acquisition_function,
                    sampler=sampler,
                )
                fx_best = fx_best * std + mean
            elif self.benchmark.is_continuous:
                x_best, fx_best, tr_state = create_candidates_continuous(
                    x_scaled=x_scaled,
                    fx_scaled=fx_scaled,
                    acquisition_function=acquisition_function,
                    model=model,
                    axus=axus,
                    trust_region=self.trust_region,
                    device=self.device,
                    batch_size=self.batch_size,
                    sampler=sampler,
                )
                fx_best = fx_best * std + mean
            # TODO don't use elif True here but check for the exact type
            elif True:
                # Scale the function values

                continuous_indices = torch.tensor(
                    [
                        i
                        for b, i in axus.bins_and_indices_of_type(
                            ParameterType.CONTINUOUS
                        )
                    ]
                )
                x_best = None
                for _ in tqdm(range(self.n_interleaved), desc="☯ Interleaved steps"):
                    x_best, fx_best, tr_state = create_candidates_discrete(
                        x_scaled=x_scaled,
                        fx_scaled=fx_scaled,
                        axus=axus,
                        model=model,
                        trust_region=self.trust_region,
                        device=self.device,
                        batch_size=self.batch_size,
                        x_bests=x_best,  # expects [-1, 1],
                        acquisition_function=acquisition_function,
                        sampler=sampler,
                    )
                    x_best = x_best.reshape(-1, axus.target_dim)
                    true_center = x[fx.argmin()]
                    x_best[:, continuous_indices] = true_center[continuous_indices].to(
                        device=x_best.device
                    )
                    x_best, fx_best, tr_state = create_candidates_continuous(
                        x_scaled=x_scaled,
                        fx_scaled=fx_scaled,
                        axus=axus,
                        trust_region=self.trust_region,
                        device=self.device,
                        indices_to_optimize=continuous_indices,
                        x_bests=x_best,  # expects [-1, 1]
                        acquisition_function=acquisition_function,
                        model=model,
                        batch_size=self.batch_size,
                        sampler=sampler,
                    )
                    fx_best = fx_best * std + mean
                    x_best = x_best.reshape(-1, axus.target_dim)
                x_best = x_best
            else:
                raise NotImplementedError(
                    "Only binary and continuous benchmarks are supported."
                )
            # get the GP hyperparameters as a dictionary
            self.save_tr_state(tr_state)
            minimum_xs = x_best.detach().cpu()
            minimum_fxs = fx_best.detach().cpu()

            fx_batches = minimum_fxs

            cand_batch = torch.empty(
                (self.batch_size, self.benchmark.representation_dim), dtype=self.dtype
            )

            xs_low_dim = list()
            xs_high_dim = list()

            for batch_index in range(self.batch_size):
                # Find the row (tr index) and column (batch index) of the minimum
                col = torch.where(fx_batches == fx_batches.min())[0]
                # Find the point that gave the minimum
                x_elect = minimum_xs[col[0]]
                if len(x_elect.shape) == 1:
                    # avoid transpose warnings
                    x_elect = x_elect.unsqueeze(0)
                # Add the point to the lower-dimensional observations
                xs_low_dim.append(x_elect)

                # Project the point up to the high dimensional space
                x_elect_up = from_1_around_origin(
                    self.random_embedding.project_up(x_elect.T).T,
                    lb=self.benchmark.lb_vec,
                    ub=self.benchmark.ub_vec,
                )
                # Add the point to the high-dimensional observations
                xs_high_dim.append(x_elect_up)
                # Add the point to the batch to be evaluated
                cand_batch[batch_index, :] = x_elect_up.squeeze()
                # Set the value of the minimum to infinity so that it is not selected again
                fx_batches[col[0]] = torch.inf

            # Sample on the candidate points
            y_next = self.benchmark(cand_batch)

            best_fx = self.fx_tr.min()
            if torch.min(y_next) < best_fx:
                logging.info(
                    f"✨ Iteration {self._n_evals}: {BColors.OKGREEN}New incumbent function value {y_next.min().item():.3f}{BColors.ENDC}"
                )
            else:
                logging.debug(
                    f"🚀 Iteration {self._n_evals}: No improvement. Best function value {best_fx.item():.3f}"
                )

            # Calculate the estimated trust region dimensionality
            tr_dim = self._forecasted_tr_dim
            # Number of times this trust region has been selected
            # Remaining budget for this trust region
            remaining_budget = self._all_split_budgets[tr_dim]
            remaining_budget = min(
                remaining_budget, self.maximum_number_evaluations - self._n_evals
            )
            remaining_budget = max(remaining_budget, 1)
            tr = self.trust_region
            factor = (tr.length_min_discrete / tr.length_discrete_continuous) ** (
                1 / remaining_budget
            )
            factor **= self.batch_size
            factor = np.clip(factor, a_min=1e-10, a_max=None)
            logging.debug(
                f"🔎 Adjusting trust region by factor {factor.item():.3f}. Remaining budget: {remaining_budget}"
            )
            update_tr_state(
                trust_region=self.trust_region,
                fx_next=y_next.min(),
                fx_incumbent=self.fx_tr.min(),
                adjustment_factor=factor,
            )

            logging.debug(
                f"📏 Trust region has length {tr.length_discrete_continuous:.3f} and minium l {tr.length_min_discrete:.3f}"
            )

            self._all_split_budgets[tr_dim] = (
                self._all_split_budgets[tr_dim] - self.batch_size
            )
            self._n_evals += self.batch_size

            self._add_data_to_tr_observations(
                xs_down=torch.vstack(xs_low_dim),
                xs_up=torch.vstack(xs_high_dim),
                fxs=y_next.reshape(-1),
            )

            # Splitting trust regions that terminated
            if self.trust_region.terminated:
                if self.random_embedding.target_dim < self.benchmark.representation_dim:
                    # Full dim is not reached yet
                    logging.info(f"✂️ Splitting trust region")

                    index_mapping = self.random_embedding.split(
                        self.number_new_bins_on_split
                    )

                    # move data to higher-dimensional space
                    self.x_tr = join_data(self.x_tr, index_mapping)
                    self.x_global = join_data(self.x_global, index_mapping)

                    self.trust_region = TrustRegion(
                        dimensionality=self.random_embedding.target_dim
                    )
                    if self.tr_splits < self._n_splits:
                        self.tr_splits += 1

                    self.split_budget = self._split_budget(
                        self.initial_target_dimensionality
                        * (self.number_new_bins_on_split + 1) ** self.tr_splits
                    )
                else:
                    # Full dim is reached
                    logging.info(
                        f"🏁 Reached full dimensionality. Restarting with new random samples."
                    )
                    self.split_budget = self._split_budget(
                        self.random_embedding.input_dim
                    )
                    # Reset the last split budget
                    self._all_split_budgets[self._forecasted_tr_dim] = self.split_budget

                    # empty tr data, does not delete the global data
                    self._reset_local_data()

                    # reset the trust region
                    self.trust_region.reset()

                    self.sample_init()
        with lzma.open(os.path.join(self.results_dir, f"results.csv.xz"), "w") as f:
            np.savetxt(
                f,
                np.hstack(
                    (
                        self.x_up_global.detach().cpu().numpy(),
                        self.fx_global.detach().cpu().numpy().reshape(-1, 1),
                    )
                ),
                delimiter=",",
            )

    @property
    def _forecasted_tr_dim(self) -> int:
        """
        Calculate the estimated trust region dimensionality.

        Returns:
            the estimated trust region dimensionality

        """

        return self.initial_target_dimensionality * (
            1 + self.number_new_bins_on_split
        ) ** (self.tr_splits)

    def _add_data_to_tr_observations(
        self,
        xs_down: torch.Tensor,
        xs_up: torch.Tensor,
        fxs: torch.Tensor,
    ):
        """
        Add data to the tr local observations and save the selected trust regions to disk.

        Args:
            xs_down: the low-dimensional points that were evaluated in the trust regions
            xs_up:  the high-dimensional points that were evaluated in the trust regions
            fxs:  the function values of the high-dimensional points that were evaluated in the trust regions

        Returns:
            None

        """

        self.fx_tr = torch.cat(
            (
                self.fx_tr,
                fxs.reshape(-1).detach().cpu(),
            )
        )
        self.x_tr = torch.vstack(
            (
                self.x_tr,
                xs_down.detach().cpu(),
            )
        )
        self.x_up_tr = torch.vstack(
            (
                self.x_up_tr,
                xs_up.detach().cpu(),
            )
        )

        self._add_data_to_global_observations(
            xs_down=xs_down,
            xs_up=xs_up,
            fxs=fxs,
        )

    def _add_data_to_global_observations(
        self,
        xs_down: torch.Tensor,
        xs_up: torch.Tensor,
        fxs: torch.Tensor,
    ):
        """
        Add data to the global observations and save the selected trust regions to disk.

        Args:
            xs_down: the low-dimensional points that were evaluated in the trust regions
            xs_up:  the high-dimensional points that were evaluated in the trust regions
            fxs:  the function values of the high-dimensional points that were evaluated in the trust regions

        Returns:
            None

        """

        self.fx_global = torch.cat(
            (
                self.fx_global,
                fxs.reshape(-1).detach().cpu(),
            )
        )
        self.x_global = torch.vstack(
            (
                self.x_global,
                xs_down.detach().cpu(),
            )
        )
        self.x_up_global = torch.vstack(
            (
                self.x_up_global,
                xs_up.detach().cpu(),
            )
        )

    def save_tr_state(
        self,
        tr_state: dict[str, Union[float, np.ndarray]],
    ):
        """
        Save the trust region state to disk.

        Args:
            tr_state: the trust region state

        Returns:
            None

        """
        for key, value in tr_state.items():
            with lzma.open(os.path.join(self.results_dir, f"{key}.csv.xz"), "a") as f:
                np.savetxt(f, value, delimiter=",")
