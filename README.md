[![pytest](https://github.com/LeoIV/bounce/actions/workflows/pytest.yml/badge.svg)](https://github.com/LeoIV/bounce/actions/workflows/pytest.yml)
[![singularity build](https://github.com/LeoIV/bounce/actions/workflows/singularity.yml/badge.svg)](https://github.com/LeoIV/bounce/actions/workflows/singularity.yml)
![python version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)

# Bounce

This is the implementation of the Bounce algorithm for the paper ["Bounce: a Reliable Bayesian Optimization Algorithm for
Combinatorial and Mixed Spaces"](https://arxiv.org/abs/2307.00618).

### Citation

Please cite our paper if you use Bounce in your work:
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

## Installation

Bounce uses `poetry` for dependency management.
`Bounce` requires `python>=3.10`. To install `poetry`, see [here](https://python-poetry.org/docs/#installation).
To install the dependencies, run

```bash
poetry install
```

This installation will not install all benchmarks.
Especially, all benchmarks inherited from `SingularityBenchmark` will not be installed.
The easiest way to install all benchmarks is to install the singularity container (see below).
To install all benchmarks manually, clone the [benchmark repository](https://github.com/LeoIV/BenchSuite) parallel to
this repository and follow the instructions in the README.

### Singularity container

To install Singularity,
see [here](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html#quick-installation-steps).
To build the singularity container, run

```bash
sudo singularity build bounce.sif singularity_container
```

## Usage

We provide a script to run the experiments in the paper.
The script assumes that you installed `singularity` and built the singularity container.
The singularity container must be in the same directory as the script and be named `bounce.sif`.
To run the experiments in the paper, run

```bash
mkdir results # create a directory for the results (necessary!)
sh ./reproduce_results.sh # bounce.sif and results/ must be in the same directory
```

The results will be stored in the `results` directory.