



# Downloading and expanding processed datasets:

## Download genetic and splits data from Zenodo
```bash
mkdir -p processed
cd processed
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

## Download chemical data from Zenodo
```bash
# Download chemical data (run from data/processed directory)
wget -O torch_data_chemical.tar.gz "https://zenodo.org/api/records/15390483/files/torch_data_chemical.tar.gz/content"

# Extract chemical torch data into torch_data directory
cd torch_data
tar -xzvf ../torch_data_chemical.tar.gz
cd ..
```

## Data Sources
- **Genetic data and splits**: [https://zenodo.org/records/15375990](https://zenodo.org/records/15375990)
- **Chemical data**: [https://zenodo.org/records/15390483](https://zenodo.org/records/15390483)