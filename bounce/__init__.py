r"""
# What is this?
This is the documentation for the accompanying Python package for the paper [*"Bounce: a Reliable Bayesian Optimization Algorithm for Combinatorial and Mixed Spaces"*](https://arxiv.org/pdf/2307.00618.pdf)  by Leonard Papenmeier, Luigi Nardi, and Matthias Poloczek.

If you use this package, please cite the paper as follows:
```bibtex
@inproceedings{
papenmeier2023bounce,
    title = {Bounce: a Reliable Bayesian Optimization Algorithm for Combinatorial and Mixed Spaces},
    author = {Leonard Papenmeier and Luigi Nardi and Matthias Poloczek},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2023},
    url = {https://arxiv.org/abs/2307.00618}
}
```

The repository for the paper can be found [here](https://github.com/LeoIV/bounce).

# Getting started

## Installation

Bounce uses `poetry` for dependency management.
`Bounce` requires `python>=3.10`. To install `poetry`, see [here](https://python-poetry.org/docs/#installation).
To install the dependencies, run

```bash
poetry install
```

This installation will not install all benchmarks.
In particular, all benchmarks inherited from `SingularityBenchmark` will not be installed.
The easiest way to install all benchmarks is to install the singularity container (see below).
To install all benchmarks manually, clone the [benchmark repository](https://github.com/LeoIV/BenchSuite) parallel to
this repository and follow the instructions in the README.

.. note::
    If you don't want to use the `Singularity` container, the easiest way is to install `BenchSuite` in a separate
    `poetry` environment as code calling `BenchSuite` will
    automatically take care of executing in the right environment.

### Singularity container

To install Singularity,
see [here](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html#quick-installation-steps).
To build the singularity container, run

```bash
sudo singularity build bounce.sif singularity_container
```

## Usage

`Bounce` uses [`gin`](https://github.com/google/gin-config) for configuration.
The default configuration can be found in `configs/default.gin`.
You can override the default configuration by passing a different `gin` file via the `--gin-files` flag.
Alternatively, you can override the default configuration by passing `gin` bindings via the `--gin-bindings` flag.

### Using the Singularity container

Below, we show an example of how to run `Bounce` using the singularity container.
```bash
singularity run bounce.sif --gin-files configs/default.gin \
--gin-bindings \"Bounce.benchmark = @Labs()\"
```
This call will run `Bounce` on the `Labs` benchmark with the default configuration.

### Usage without the Singularity container

Below, we show an example of how to run `Bounce` without the singularity container.
```bash
poetry run python main.py --gin-files configs/default.gin \
--gin-bindings "Bounce.benchmark = @Labs()"
```

### Reproducing the results in the paper

We provide a script to run the experiments in the paper.
The script assumes that you installed `singularity` and built the singularity container.
The singularity container must be in the same directory as the script and be named `bounce.sif`.
To run the experiments in the paper, run

```bash
mkdir results # create a directory for the results (necessary!)
sh ./reproduce_results.sh # bounce.sif and results/ must be in the same directory
```

The results will be stored in the `results` directory.
"""
