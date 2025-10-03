import networkx as nx
import pandas as pd
import numpy as np
import os
import os.path as osp
import h5py

#Loading Data
DATA_ROOT = "data/raw/lincs/"
#log_handle = open(osp.join(outdir, 'process_data_lognorm_log.txt'), 'w')
#log_handle = open(osp.join(outdir, 'test_log.txt'), 'w')

inst_info = pd.read_csv(os.path.join(DATA_ROOT, 'instinfo_beta.txt'), sep="\t", low_memory=False)
inst_info_xpr = inst_info[np.logical_and(inst_info['pert_type'] == 'trt_xpr', inst_info['failure_mode'].isna())].reset_index(inplace=False, drop=True) 
inst_info_ctl = inst_info[np.logical_and(np.logical_or(inst_info['pert_type'] == 'ctl_vector',inst_info['pert_type'] == 'ctl_vehicle'), inst_info['failure_mode'].isna())].reset_index(inplace=False, drop=True)
gene_info = pd.read_csv(os.path.join(DATA_ROOT, 'geneinfo_beta.txt'), sep="\t", low_memory=False)
df_xpr=pd.DataFrame(inst_info_xpr[['sample_id','pert_id', 'pert_dose','pert_dose_unit','pert_time','cell_iname']])


f = h5py.File(os.path.join(DATA_ROOT, 'level3_beta_ctl_n188708x12328.gctx'), 'r')
matrix_xpr = f['0']['DATA']['0']['matrix'][:].transpose()
gene_ids_xpr = f['0']['META']['ROW']['id'][:]
sample_ids_xpr = f['0']['META']['COL']['id'][:]
matrix_xpr = pd.DataFrame(matrix_xpr, columns = sample_ids_xpr.astype(str), index = gene_ids_xpr.astype(int))
del f

#Decoding sample_ids_xpr array
decode=np.vectorize(np.char.decode)
sample_ids_xpr=decode(sample_ids_xpr)

#Creating dict of cell_line:sample_ids
cell_line_sids={}
for cn in inst_info_ctl.cell_iname.unique():
	cell_line_sids[cn]=list(inst_info_ctl.loc[inst_info_ctl['cell_iname']==cn]['sample_id'])

#Lists of sample_ids for ctl_vector and ctl_vehicle samples
xpr_samples_unique = list(inst_info_ctl.loc[inst_info_ctl['pert_type']=="ctl_vector"].sample_id.unique())
cmp_samples_unique = list(inst_info_ctl.loc[inst_info_ctl['pert_type']=="ctl_vehicle"].sample_id.unique())
    
    

#Renaming columns of matrix_xpr_trans to gene symbols
matrix_xpr_trans=matrix_xpr.T
new_cols = []
for i in matrix_xpr_trans.columns: 
	new_cols.append(gene_info.loc[gene_info['gene_id']==i]['gene_symbol'].item())
matrix_xpr_trans.columns = new_cols


#Subsetting matrix_xpr_trans column genes to include only those in the ppi, but not those that are perturbed.
path_edge_list = 'data/raw/ppi/ppi_all_genes_edgelist.txt'
ppi = nx.read_edgelist(path_edge_list)
ppi = ppi.subgraph(gene_info['gene_symbol'].tolist())


#pert_genes = dict()
#for cell_line in list(set(inst_info_xpr['cell_iname'])):
	#pert_genes[cell_line] = set(inst_info_xpr[inst_info_xpr['cell_iname']==cell_line]['cmap_name'].tolist())

outdir = 'data/raw/grn/xpr_matrices/'
os.makedirs(outdir, exist_ok=True)

#chemical cell lines
for cn in cell_line_sids:
	if cn in ['A549', 'MCF7', 'PC3', 'VCAP', 'MDAMB231', 'BT20', 'HT29', 'A375', 'HELA']:
		print(cn)
		cellline_xpr_matrix_chem = matrix_xpr_trans[np.logical_and(matrix_xpr_trans.index.isin(cell_line_sids[cn]),matrix_xpr_trans.index.isin(cmp_samples_unique))]
		gene_intersection_list=list(set(ppi.nodes).intersection(set(list(gene_info.gene_symbol.unique()))))
		cellline_xpr_matrix_chem=cellline_xpr_matrix_chem[gene_intersection_list]
		cellline_xpr_matrix_chem.to_csv(osp.join(outdir,'{}_xpr_matrix_cmp_nonpertsubset.txt'.format(cn)), sep='	', index=False)

#genetic cell lines
for cn in cell_line_sids:
	if cn in ['BICR6', 'YAPC', 'AGS', 'U251MG', 'ES2', 'MCF7', 'PC3', 'A375', 'HT29', 'A549']:
		print(cn)
		cellline_xpr_matrix_gen = matrix_xpr_trans[np.logical_and(matrix_xpr_trans.index.isin(cell_line_sids[cn]),matrix_xpr_trans.index.isin(xpr_samples_unique))]
		gene_intersection_list=list(set(ppi.nodes).intersection(set(list(gene_info.gene_symbol.unique()))))
		cellline_xpr_matrix_gen=cellline_xpr_matrix_gen[gene_intersection_list]
		cellline_xpr_matrix_gen.to_csv(osp.join(outdir,'{}_xpr_matrix_gen_nonpertsubset.txt'.format(cn)), sep='	', index=False)


    
