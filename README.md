# Combinatorial prediction of therapeutic perturbations using causally-inspired neural networks
[![ProjectPage](https://img.shields.io/badge/Project-PDGrapher-red)](https://zitniklab.hms.harvard.edu/projects/PDGrapher/) [![CodePage](https://img.shields.io/badge/Code-GitHub-orange)](https://github.com/mims-harvard/PDGrapher) [![Paper](https://img.shields.io/badge/Paper-BioRxiv-green)](https://www.biorxiv.org/content/10.1101/2024.01.03.573985v2)![License](https://img.shields.io/badge/license-MIT-blue) 

[Guadalupe Gonzalez*](https://www.gene.com/scientists/our-scientists/guadalupe-gonzalez), [Xiang Lin*](https://scholar.google.com/citations?user=SKdT80YAAAAJ&hl=en), [Isuru Herath](https://scholar.google.com/citations?user=F-RC5k0AAAAJ&hl=en), [Kirill Veselkov](https://scholar.google.com/citations?user=0n-5UGYAAAAJ&hl=en),
[Michael Bronstein](https://scholar.google.com/citations?user=UU3N6-UAAAAJ&hl=en), and [Marinka Zitnik](https://dbmi.hms.harvard.edu/people/marinka-zitnik)

![](https://github.com/mims-harvard/PDGrapher/blob/main/figures/figure_1.jpg)
## Project structure

The project consists of next folders:
- [data](data/) contains all of the data on which our models were built. On how to obtain this data, refer to [Data](#data) section,
- [docs](docs/) contains documentation, built with 'sphinx',
- [src/pdgrapher](src/pdgrapher/) contains the source code for PDGrapher,
- [examples](examples/) contains some examples that demonstrate the use of PDGrapher library, read the [README.md](examples/README.md) in examples folder for more instructions,
- [tests](tests/) contains unit and integration tests.

## Virtual environment

```
conda env create -f conda-env.yml
conda activate pdgrapher
pip install pip==23.2.1
pip install -r requirements.txt

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
pip install torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
pip install torchmetrics==0.9.3
pip install lightning==1.9.5

```


## Data

For processed data, download the compressed folders and place them in `data/processed/` with the following commands:

```
cd data
mkdir processed
cd processed
wget https://figshare.com/ndownloader/files/43624557
tar -xzvf torch_data.tar.gz
cd ../
wget https://figshare.com/ndownloader/files/43632327
tar -xzvf splits.tar.gz
```


## Building

This project can be build as a Python library by running `pip install -e .` in the root of this repository.


## Documentation

Documentation can be build with next two commands:
- `sphinx-apidoc -fe -o docs/source/ src/pdgrapher/` updates the source files from which the documentation is built,
- `docs/make html` builds the documentation

Then, the documentation can be accessed locally by going to `docs/build/html/index.html`
All of the settings along with links to instructions can be found and modified in [docs/source/conf.py](docs/source/conf.py). 


