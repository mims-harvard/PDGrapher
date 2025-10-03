## Scripts to build GRN for PDGrapher
---

### 1. Generate Expression Matrices
Use `xpr_matrix_generator_datatypesplit.py` to build the `xpr_matrices` files.  

**Inputs:**  
- `level3_beta_ctl_n188708x12328.gctx` (download from [C-map](https://clue.io/releases/data-dashboard))  
- `ppi_all_genes_edgelist.txt` (see `data/scripts/ppi` for instructions on generating this file)  

**Command Example:**  
python xpr_matrix_generator_datatypesplit.py


### 2. Generate GENIE3 edge list
Use `GENIEppi-run.py` to build the `{cell_line}_{pert_type}_edgelist.txt` files.  

**Inputs:**  
- `xpr_matrices` (from step 1)  

**Command Example:**  
python GENIEppi-run.py --data_type cmp --cell_line A549

### 3. Filter edge lists
Use `filter_edge_list.py` to build the `{cell_line}_{pert_type}_edgelist_filtered.txt` files.  

**Inputs:**  
- `{cell_line}_{pert_type}_edgelist.txt` files (from step 2)  

**Command Example:**  
python filter_edge_list.py --data_type cmp --cell_line A549
