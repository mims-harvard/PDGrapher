#!/bin/bash

# Process LINCS data
cd lincs/scripts
python3 process_data.py
python3 process_data_healthy.py
cd ../..

# Process COSMIC data
cd cosmic/scripts
python3 generate_cell_line_mutations_files.py
cd ../..

# Process PPI data
cd ppi/scripts
pyton3 ppi_lincs_perturbed.py
cd ../..

# Data preparation
cd rep-learning-approach-3/scripts
python3 export_data_for_torch_geometric.py
cd ../..