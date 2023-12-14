#!/bin/bash

# Process COSMIC data
cd scripts/cosmic
python3 generate_cell_line_mutations_files.py
cd ../..


# Process COSMIC data
cd scripts/drugbank
python3 get_drug_targets.py
cd ../..

# Process PPI data
cd scripts/ppi
python3 export_ppi_all_genes.py
cd ../..


# Process LINCS data
cd scripts/lincs
python3 process_data.py
python3 process_data_healthy.py
python3 process_data_chemical_1.py
python3 process_data_chemical_2.py
python3 process_data_healthy_chemical.py
cd ../..



# Data preparation
cd scripts/rep-learning-approach-3
python3 export_data_for_torch_geometric.py
python3 export_data_for_torch_geometric_chemical.py
cd ../..