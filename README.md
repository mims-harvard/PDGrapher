# PDGrapher

## Project structure

The project consists of next folders:
- [data](data/) contains all of the data on which our models were built. On how to obtain this data, refer to [Data](#data) section,
- [docs](docs/) contains documentation, built with 'sphinx',
- [src/pdgrapher](src/pdgrapher/) contains the source code for PDGrapher library,
- [examples](examples/) contains some examples that demonstrate the use of PDGrapher library, read the [README.md](examples/README.md) in examples folder for more instructions,
- [tests](tests/) contains unit and integration tests.

## Requirements

Requirements can be installed via [requirements.txt](requirements.txt) file. For development purposes, see [requirements-dev.txt](requirements-dev.txt). We recommend using [venv](https://docs.python.org/3/library/venv.html) or [Conda](https://docs.conda.io/en/latest/) for an environment.

Main requirements can also be installed manually to select more specific versions (like CUDA):
1. Install [PyTorch](https://pytorch.org/get-started/locally/)
2. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) along with its optional dependencies.
3. Install [PyTorch Lightning](https://github.com/Lightning-AI/lightning)

## Data

For already processed data, download the next files and place them in [data/rep-learning-approach-3/processed/](data/rep-learning-approach-3/processed/) folder:
- [TODO](.)

For manual data processing please refer to the [README.md](data/README.md) in the data folder.

## Building

This project can be build as a Python library by running `pip install -e .` in the root of this repository.

## Documentation

Documentation can be build with next two commands:
- `sphinx-apidoc -fe -o docs/source/ src/pdgrapher/` updates the source files from which the documentation is built,
- `docs/make html` builds the documentation

All of the settings along with links to instructions can be found and modified in [docs/source/conf.py](docs/source/conf.py). 