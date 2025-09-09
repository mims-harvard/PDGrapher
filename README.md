# Combinatorial prediction of therapeutic perturbations using causally-inspired neural networks
[![ProjectPage](https://img.shields.io/badge/Project-PDGrapher-red)](https://zitniklab.hms.harvard.edu/projects/PDGrapher/) [![CodePage](https://img.shields.io/badge/Code-GitHub-orange)](https://github.com/mims-harvard/PDGrapher) [![Paper](https://img.shields.io/badge/Paper-BioRxiv-green)](https://www.biorxiv.org/content/10.1101/2024.01.03.573985v5) [![Paper](https://img.shields.io/badge/Paper-NBME-green)](https://www.nature.com/articles/s41551-025-01481-x) [![Data](https://img.shields.io/badge/Data-Links-purple)](https://github.com/mims-harvard/PDGrapher/tree/main/data) ![License](https://img.shields.io/badge/license-MIT-blue) 



[Guadalupe Gonzalez*](https://www.guadalupegonzalez.io/), [Xiang Lin*](https://xianglin226.github.io/), [Isuru Herath](https://scholar.google.com/citations?user=F-RC5k0AAAAJ&hl=en), [Kirill Veselkov](https://scholar.google.com/citations?user=0n-5UGYAAAAJ&hl=en),
[Michael Bronstein](https://scholar.google.com/citations?user=UU3N6-UAAAAJ&hl=en), and [Marinka Zitnik](https://dbmi.hms.harvard.edu/people/marinka-zitnik)

![](https://github.com/mims-harvard/PDGrapher/blob/main/figures/figure1.jpg)
## Project structure

The project consists of next folders:
- [data](data/) contains all of the data on which our models were built. On how to obtain this data, refer to [Data](#data) section,
- [docs](docs/) contains documentation, built with 'sphinx',
- [src/pdgrapher](src/pdgrapher/) contains the source code for PDGrapher,
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

### Download genetic and splits data from Zenodo
```bash
cd data/processed
# Download splits and genetic data
wget -O splits.tar.gz "https://zenodo.org/api/records/15375990/files/splits.tar.gz/content"
wget -O torch_data_genetic.tar.gz "https://zenodo.org/api/records/15375990/files/torch_data_genetic.tar.gz/content"

# Extract splits data
tar -xzvf splits.tar.gz

# Create torch_data directory and extract genetic data into it
mkdir -p torch_data
cd torch_data
tar -xzvf ../torch_data_genetic.tar.gz
cd ..
```

### Download chemical data from Zenodo
```bash
# Download chemical data (run from data/processed directory)
wget -O torch_data_chemical.tar.gz "https://zenodo.org/api/records/15390483/files/torch_data_chemical.tar.gz/content"

# Extract chemical torch data into torch_data directory
cd torch_data
tar -xzvf ../torch_data_chemical.tar.gz
cd ..
```

### Data Sources
- **Genetic data and splits**: [https://zenodo.org/records/15375990](https://zenodo.org/records/15375990)
- **Chemical data**: [https://zenodo.org/records/15390483](https://zenodo.org/records/15390483)


## Building

This project can be build as a Python library by running `pip install -e .` in the root of this repository.


## Documentation

Documentation can be built with the following commands:

### Prerequisites
First, install the required documentation dependencies:
```bash
pip install myst-parser
```

### Build Documentation
1. `sphinx-apidoc -fe -o docs/source/ src/pdgrapher/` updates the source files from which the documentation is built
2. `cd docs && make html` builds the documentation

Then, the documentation can be accessed locally by going to `docs/build/html/index.html`
All of the settings along with links to instructions can be found and modified in [docs/source/conf.py](docs/source/conf.py). 

## Notebooks
|  Tutorials       | Links                     |
|----------------|---------------------------------|
| Train PDGrapher on chemical dataset         | [notebook](./notebooks/train_chemical.ipynb)            |
| Train PDGrapher on genetic dataset     | [notebook](./notebooks/train_genetic.ipynb)    | 
| Test PDGrapher on chemical/genetic dataset  | [notebook](./notebooks/test_PDG.ipynb)      | 

## Additional Resources
* [Paper](https://www.nature.com/articles/s41551-025-01481-x)  
* [HMS News & Research](https://hms.harvard.edu/news/new-ai-tool-pinpoints-genes-drug-combos-restore-health-diseased-cells)

@article{gonzalez2025combinatorial,  
  title={Combinatorial Prediction of Therapeutic Perturbations Using Causally-Inspired Neural Networks},  
  author={Gonzalez, Guadalupe and Lin, Xiang and Herath, Isuru and Veselkov, Kirill and Bronstein, Michael and Zitnik, Marinka},  
  journal={Nature Biomedical Engineering},  
  url={https://www.nature.com/articles/s41551-025-01481-x},  
  year={2025}  
}  

## License
The code in this package is licensed under the MIT License.

## Questions
Please leave a Github issue or contact [Guadalupe Gonzalez](mailto:ggonzalezp16@gmail.com), [Xiang Lin](mailto:xianglin226@gmail.com), or [Marinka Zitnik](mailto:marinka@zitnik.si)  

