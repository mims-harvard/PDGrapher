''' 
Normalizes perturbed (treated) and control (diseased) data of genetic interventions
Saves data as npz

1. Reads data (load_data)
2. Filters to keep only the cell lines that we use in experiments (filter_cell_lines_custom)
3. Filters to keep only the samples in metadata (filter_data_metadata)
4. Filters out samples for which the drug targets are not in LINCS genes (filter_samples_with_unknown_perturbed_genes)
5. Normalizes data between (0,1) and save (normalize_and_save)
'''
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math
from random import sample
from sklearn.preprocessing import MinMaxScaler


outdir = '../../processed/lincs'
os.makedirs(outdir, exist_ok=True)

	#LOG
	

################
# Data loading
################
def stats_data(inst_info_xpr, matrix_xpr, matrix_ctl, gene_info):
	dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))

		####Data exploration -- GE values of genes that are perturbed (!)
	#Get GE value for each gene perturbed by CRISPR
	values_pert = {}
	values_control = {}
	for i in range(len(inst_info_xpr)):
		gene_symbol = inst_info_xpr.at[i, 'cmap_name']
		if gene_symbol in dict_symbol_id:					#if the cmap_name of gene is in the gene_info
			sample_id = inst_info_xpr.at[i, 'sample_id']
			gene_id = dict_symbol_id[gene_symbol]
			if gene_id in values_pert:
				values_pert[gene_id].append(matrix_xpr.at[gene_id, sample_id])
			else:
				values_pert[gene_id] = [matrix_xpr.at[gene_id, sample_id]]


	for gene_symbol in list(set(inst_info_xpr['cmap_name'])):
		if gene_symbol in dict_symbol_id:				#if the cmap_name of gene is in the gene_info
			gene_id = dict_symbol_id[gene_symbol]
			values_control[gene_id] = [matrix_ctl.loc[gene_id]]

	for key in values_pert:
		values_pert[key] = np.mean(values_pert[key])

	for key in values_control:
		values_control[key] = np.mean(values_control[key])

	fig, (ax1, ax2) = plt.subplots(2, figsize=(16,6))
	ax1.hist(values_pert.values())
	ax2.hist(values_control.values())
	ax1.set_title('Values of perturbed genes (avg) - CRISPR')
	ax2.set_title('Values of genes in control (avg)')
	fig.savefig(osp.join(outdir,'exploration_ge_crispr.png'))


	return



def loads_data(DATA_ROOT, log_handle):
	#Loads metadata
	inst_info = pd.read_csv(os.path.join(DATA_ROOT, 'instinfo_beta.txt'), sep="\t", low_memory=False)
	inst_info_xpr = inst_info[np.logical_and(inst_info['pert_type'] == 'trt_xpr',  inst_info['failure_mode'].isna())].reset_index(inplace=False, drop=True)
	inst_info_ctl = inst_info[np.logical_and(inst_info['pert_type'] == 'ctl_vector', inst_info['failure_mode'].isna()) ].reset_index(inplace=False, drop=True)
	gene_info = pd.read_csv(os.path.join(DATA_ROOT, 'geneinfo_beta.txt'), sep="\t", low_memory=False)


	####################
	#Loads data matrices
	### CRISPR
	f = h5py.File(os.path.join(DATA_ROOT, 'level3_beta_trt_xpr_n420583x12328.gctx'), 'r')
	matrix_xpr = f['0']['DATA']['0']['matrix'][:].transpose()
	gene_ids_xpr = f['0']['META']['ROW']['id'][:]
	sample_ids_xpr = f['0']['META']['COL']['id'][:]
	matrix_xpr = pd.DataFrame(matrix_xpr, columns = sample_ids_xpr.astype(str), index = gene_ids_xpr.astype(int))
	del f


	#re-order gene_info based on the order in gene_ids_xpr (rows of data)
	gene_info.index = gene_info['gene_id']
	gene_info = gene_info.loc[gene_ids_xpr.astype(int)].reset_index(inplace=False, drop=True)
	gene_info.to_csv(osp.join(outdir, 'gene_info.txt'), index=False)

	#Stats
	log_handle.write('CRISPR\n------\n')
	log_handle.write('CRISPR entries in inst_info metadata:\t{}\n'.format(len(inst_info_xpr)))
	log_handle.write('CRISPR entries in data matrix:\t{}\n'.format(len(sample_ids_xpr)))
	log_handle.write('Overlap between inst_info metadata and sample ids in data matrix:\t{}\n'.format(len(set(inst_info_xpr['sample_id']).intersection(set(sample_ids_xpr.astype(str))))))



	### Control data
	f = h5py.File(os.path.join(DATA_ROOT, 'level3_beta_ctl_n188708x12328.gctx'), 'r')
	matrix_ctl = f['0']['DATA']['0']['matrix'][:].transpose()
	gene_ids_ctl = f['0']['META']['ROW']['id'][:]					#not in the same order as gene_ids_xpr
	sample_ids_ctl = f['0']['META']['COL']['id'][:]
	matrix_ctl = pd.DataFrame(matrix_ctl, columns = sample_ids_ctl.astype(str), index = gene_ids_ctl.astype(int))
	del f


	#Stats
	log_handle.write('CONTROL\n------\n')
	log_handle.write('Control entries in inst_info metadata:\t{}\n'.format(len(inst_info_ctl)))
	log_handle.write('Control entries in data matrix:\t{}\n'.format(len(sample_ids_ctl)))
	log_handle.write('Overlap between inst_info metadata and sample ids in data matrix:\t{}\n'.format(len(set(inst_info_ctl['sample_id']).intersection(set(sample_ids_ctl.astype(str))))))
	log_handle.write('\n------\n')


	stats_data(inst_info_xpr, matrix_xpr, matrix_ctl, gene_info)

	return inst_info_xpr, inst_info_ctl, gene_info, matrix_xpr, matrix_ctl




################
# Processing
################

################
#1. Filter column metadata and data matrix to keep only those in metadata

def filter_data_metadata(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, log_handle):
	log_handle.write('Filtering to keep only those in metadata\n------\n')
	#CRISPR
	list_ids = list(inst_info_xpr['sample_id'])	#in metadata
	matrix_xpr = matrix_xpr[list_ids]	#Filtered data matrix
	log_handle.write('CRISPR:\t{} datapoints\n'.format(matrix_xpr.shape[1]))
	#CONTROL
	list_ids = list(inst_info_ctl['sample_id'])	#in metadata
	#extra steps
	#--
	list_ids = list(set(list_ids).intersection(set(matrix_ctl.columns.astype(str))))	#in metadata and in data matrix (some of metadata are not in data matrix)
	inst_info_ctl.index = inst_info_ctl['sample_id']; inst_info_ctl = inst_info_ctl.loc[list_ids].reset_index(inplace=False, drop=True) #remove entries from metadata that are not in data matrix
	#--
	matrix_ctl = matrix_ctl[list_ids]	#Filtered data matrix
	log_handle.write('CONTROL:\t{} datapoints\n\n\n'.format(matrix_ctl.shape[1]))
	return inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl



################
#2. Filter to keep only cell lines with more perturbations

def stats_control(inst_info_ctl, log_handle):
	#Stats unique cell lines
	log_handle.write('Unique cell lines:\t{}:\n'.format(len(set(inst_info_ctl['cell_iname']))))
	for c in list(set(inst_info_ctl['cell_iname'])):
		log_handle.write('\t{}\n'.format(c))

	log_handle.write('\n\n')

	#Stats on dosages
	df_ctl = pd.DataFrame(inst_info_ctl[['cmap_name', 'cell_iname', 'pert_idose']].groupby(['cmap_name', 'cell_iname']).apply(lambda x: x['pert_idose'].unique()))
	df_ctl = pd.DataFrame([(i, len(df_ctl.loc[i][0])) for i in df_ctl.index], columns =['cmap_name-cell_line', 'n_doses'])
	log_handle.write('Stats on dosages and timepoints\n')
	log_handle.write('\n------\nHOW MANY DOSES ARE THERE FOR CMAP_NAME-CELL LINE PAIRS?\n------\n')
	for index,value in pd.Series.iteritems(pd.DataFrame(df_ctl['n_doses'])['n_doses'].describe()):
	    log_handle.write('{}:\t{}\n'.format(index, value))

	log_handle.write('\n')
	log_handle.write('Number of pairs with more than 1 dose:\t{}/{}\n'.format(sum(df_ctl['n_doses']>1), len(df_ctl)))
	log_handle.write('Number of pairs with more than 2 doses:\t{}/{}\n\n'.format(sum(df_ctl['n_doses']>2), len(df_ctl)))

	#Stats on timepoints
	df_ctl = pd.DataFrame(inst_info_ctl[['cmap_name', 'cell_iname', 'pert_time']].groupby(['cmap_name', 'cell_iname']).apply(lambda x: x['pert_time'].unique()))
	df_ctl = pd.DataFrame([(i, len(df_ctl.loc[i][0])) for i in df_ctl.index], columns =['cmap_name-cell_line', 'n_times'])

	log_handle.write('\n------\nHOW MANY TIMEPOINTS ARE THERE FOR CMAP_NAME-CELL LINE PAIRS?\n------\n')
	for index,value in pd.Series.iteritems(pd.DataFrame(df_ctl['n_times'])['n_times'].describe()):
	    log_handle.write('{}:\t{}\n'.format(index, value))

	log_handle.write('\n')
	log_handle.write('Number of pairs with more than 1 timepoint:\t{}/{}\n'.format(sum(df_ctl['n_times']>1), len(df_ctl['n_times'])))
	log_handle.write('Number of pairs with more than 2 timepoints:\t{}/{}\n\n'.format(sum(df_ctl['n_times']>2), len(df_ctl['n_times'])))


	log_handle.write('\nUSING THEM ALL FOR NOW\n')

	#Types of vectors
	log_handle.write('Number of vectors:\t{}:\n'.format(len(set(inst_info_ctl['cmap_name']))))
	df=pd.DataFrame.from_dict(Counter(inst_info_ctl['cmap_name']), orient='index')
	df = df.sort_values(by=0)
	for i, v in enumerate(zip(df.index, df[0])):
		log_handle.write('{}:\t{}\n'.format(v[0], v[1]))

	#Number of controls per cell line
	replicates = inst_info_ctl.groupby(['cell_iname']).size()
	log_handle.write('\n\n------\nNUMBER OF REPLICATES PER CELL LINE (different doses, times, vectors)\n-----------\n')
	df=pd.DataFrame.from_dict(Counter(inst_info_ctl['cell_iname']), orient='index')
	df = df.sort_values(by=0)
	for i, v in enumerate(zip(df.index, df[0])):
		log_handle.write('{}:\t{}\n'.format(v[0], v[1]))

	return


def filter_cell_lines(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, log_handle):
	log_handle.write('Filtering to keep only cell lines with highest mumber of perturbed genes\n------\n')

	#####CRISPR
	#Obtain cell lines with the most perturbations (> 90th-percentile)
	df_xpr = pd.DataFrame(inst_info_xpr[['cmap_name', 'cell_iname']].groupby('cell_iname', as_index=True).apply(lambda x: x['cmap_name'].unique()))
	df_xpr = pd.DataFrame([(i, len(df_xpr.loc[i][0])) for i in df_xpr.index], columns =['cell_line', 'n_cmap_names'])
	df_xpr = df_xpr.sort_values(by='n_cmap_names')
	keep_cell_lines = df_xpr[df_xpr['n_cmap_names']>np.percentile(df_xpr['n_cmap_names'], 60)]['cell_line'].tolist()

	#Find indices of samples that are on the desired cell lines
	keep_index = []
	for i in range(len(inst_info_xpr)):
		if inst_info_xpr.at[i, 'cell_iname'] in keep_cell_lines:
			keep_index.append(i)


	inst_info_xpr = inst_info_xpr.loc[keep_index].reset_index(inplace=False, drop=True) #filter from metadata
	list_ids = list(inst_info_xpr['sample_id'])	#obtain sample ID from metadata
	matrix_xpr = matrix_xpr[list_ids]	#Filtered data matrix
	log_handle.write('CRISPR:\t{} datapoints\n'.format(matrix_xpr.shape[1]))



	#####CONTROL
	keep_index = []
	for i in range(len(inst_info_ctl)):
		if inst_info_ctl.at[i, 'cell_iname'] in keep_cell_lines:
			keep_index.append(i)


	inst_info_ctl = inst_info_ctl.loc[keep_index].reset_index(inplace=False, drop=True)  #filter from metadata
	list_ids = list(inst_info_ctl['sample_id'])	#obtain sample ID from metadata
	matrix_ctl = matrix_ctl[list_ids]	#Filtered data matrix
	log_handle.write('CONTROL:\t{} datapoints\n'.format(matrix_ctl.shape[1]))

	# Stats
	# stats_control(inst_info_ctl, log_handle)

	return inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, keep_cell_lines


def filter_cell_lines_custom(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, log_handle):
	log_handle.write('Filtering to keep only cell lines: A549, PC3, MCF7\n------\n')
	#####CRISPR
	#Obtain cell lines with the most perturbations (> 4K genes perturbed)
	keep_cell_lines = ['A549', 'PC3', 'MCF7', 'A375', 'HT29', 'ES2', 'BICR6', 'YAPC', 'AGS', 'U251MG']
	#Find indices of samples that are on the desired cell lines
	keep_index = []
	for i in range(len(inst_info_xpr)):
		if inst_info_xpr.at[i, 'cell_iname'] in keep_cell_lines:
			keep_index.append(i)
	inst_info_xpr = inst_info_xpr.loc[keep_index].reset_index(inplace=False, drop=True) #filter from metadata
	list_ids = list(inst_info_xpr['sample_id'])	#obtain sample ID from metadata
	matrix_xpr = matrix_xpr[list_ids]	#Filtered data matrix
	log_handle.write('CRISPR:\t{} datapoints\n'.format(matrix_xpr.shape[1]))
	#####CONTROL
	keep_index = []
	for i in range(len(inst_info_ctl)):
		if inst_info_ctl.at[i, 'cell_iname'] in keep_cell_lines:
			keep_index.append(i)
	inst_info_ctl = inst_info_ctl.loc[keep_index].reset_index(inplace=False, drop=True)  #filter from metadata
	list_ids = list(inst_info_ctl['sample_id'])	#obtain sample ID from metadata
	matrix_ctl = matrix_ctl[list_ids]	#Filtered data matrix
	log_handle.write('CONTROL:\t{} datapoints\n'.format(matrix_ctl.shape[1]))
	# Stats
	#stats_control(inst_info_ctl, log_handle)
	return inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, keep_cell_lines

################
#3. Concatenate perturbation and control data, normalize (binarize), and save
def filter_samples_with_unknown_perturbed_genes(inst_info_xpr, matrix_xpr, gene_info, log_handle):
	########################################################################################
	#First filter perturbation samples to remove those with genes not mapping to genes_info
	#Will need to remove this once I get the mapping file from CLUE
	known_genes = list(set(gene_info['gene_symbol']))
	keep_index = []
	for i in range(len(inst_info_xpr)):
		if inst_info_xpr.at[i, 'cmap_name'] in known_genes:
			keep_index.append(i)


	inst_info_xpr = inst_info_xpr.loc[keep_index].reset_index(inplace=False, drop=True) #filter from metadata
	list_ids = list(inst_info_xpr['sample_id'])	#obtain sample ID from metadata
	matrix_xpr = matrix_xpr[list_ids]	#Filtered data matrix
	log_handle.write('Filtering samples with perturbed genes not mapped to gene_info -- TEMPORARY STEP THAT SHOULD BE REMOVED AFTER MAPPING GENES\n')
	log_handle.write('CRISPR:\t{} datapoints\n'.format(matrix_xpr.shape[1]))
	return inst_info_xpr, matrix_xpr


def binarize_genewise_ranking_all(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, gene_info, keep_cell_lines, log_handle, outdir):
	log_handle.write('\n\n------\nBINARIZING GENEWISE RANKING ALL SAMPLES\n------\n')
	outdir = osp.join(outdir, 'binarize_genewise_ranking_all')
	os.makedirs(outdir, exist_ok= True)
	########################################################################################
	#All data
	metadata = pd.concat([inst_info_xpr, inst_info_ctl], axis=0).reset_index(inplace=False, drop=True)
	matrix = pd.concat([matrix_xpr, matrix_ctl], 1)
	metadata.to_csv(osp.join(outdir, 'all_metadata.txt'))



	#Normalization 
	#0. create dictionary of gene_id: sample_id
		#gene symbol -> gene id
	dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))
		#gene symbol -> sample id
	dict_gene_id_sample_id= dict()
	for i in range(len(inst_info_xpr)):
		gene_symbol = inst_info_xpr.at[i, 'cmap_name']
		gene_id = dict_symbol_id[gene_symbol]
		if gene_id in dict_gene_id_sample_id:
			dict_gene_id_sample_id[gene_id].append(inst_info_xpr.at[i, 'sample_id'])
		else:
			dict_gene_id_sample_id[gene_id] = [inst_info_xpr.at[i, 'sample_id']]


	#1. Iterate through each gene, mask out the samples in which it was perturbed, and normalize --> take top 2% of samples as = 1
	#for the perturbed genes = 0
	#Create matrix of NGenes x NExperiments (add column name as sample_id)
	matrix_binarized = pd.DataFrame(np.zeros_like(matrix), index = matrix.index, columns = matrix.columns)
	i = 1
	higher_than_thr = []
	higher_than_thr_ids = []
	for gene_id in list(matrix.index):
		print('{}/{}'.format(i, len(matrix)))
		#mask of elements to ignore (perturbed samples --> =0)
		if gene_id in dict_gene_id_sample_id:
			mask_pert = dict_gene_id_sample_id[gene_id]
			mask_norm = list(set(list(matrix.columns)) - set(mask_pert))
		else:
			mask_pert = None
			mask_norm = list(matrix.columns)
		#normalize
		threshold = np.percentile(matrix.loc[gene_id][mask_norm].values, 98)
		matrix_binarized.loc[gene_id][mask_norm] = (matrix.loc[gene_id][mask_norm] >= threshold).astype(int).values
		i+=1
		#some stats
		if mask_pert is not None:
			higher_than_thr += (matrix.loc[gene_id][mask_pert] >= threshold).values.astype(int).tolist()
			higher_than_thr_ids += matrix.loc[gene_id][mask_pert].index[np.where(matrix.loc[gene_id][mask_pert] >= threshold)].tolist()

	###Filter columns (samples) in which the perturbed gene has an expression value that is >= the threshold used to binarize
	log_handle.write('Filtering:\t{} columns/samples because the perturbed gene has an expression value >= the threshold used to binarize -- TEMPORARY STEP THAT SHOULD BE REMOVED LATER ON\n'.format(len(higher_than_thr_ids)))
	for c in higher_than_thr_ids:
		del matrix_binarized[c]


	#plot higher than thr
	fig, ax1 = plt.subplots(1, figsize=(16,6))
	ax1.hist(higher_than_thr)
	ax1.set_title('Perturbed genes values >= threshold')
	fig.savefig(osp.join(outdir,'exploration_ge_crispr_higher_than_thr.png'))
	log_handle.write('After binarizing, stats, perturbed gene values would be higher than threshold in :\t{} samples\n'.format(np.sum(higher_than_thr)))


	#2. Save data and metadata for each condition and cell line
	#CRISPR + cell lines
	#Control + cell lines
	log_handle.write('----------------\n----------------\nDATA MATRICES\n')
	log_handle.write('CELL\tPERT\t\tSIZE\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
	metadata.index = metadata['sample_id']
	metadata = metadata.loc[matrix_binarized.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
	for cell_line in keep_cell_lines:
		for pert_type in ['trt_xpr', 'ctl_vector']:
			metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
			data_i = matrix_binarized[metadata_i.index]
			metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
			filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
			np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
			log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), len(set(metadata_i['cmap_name'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
	log_handle.write('\n\n------\nSTATS\n------\n')		

	return



def binarize_genewise_comparing_to_control_all_controls_joint(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, gene_info, keep_cell_lines, log_handle, outdir, use_log):

	log_handle.write('\n\n------\nBINARIZING GENEWISE COMPARING TO CONTROL\n------\n')
	if use_log:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control_all_controls_joint_lognorm')
	else:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control_all_controls_joint')
	os.makedirs(outdir, exist_ok= True)

	########################################################################################
	#All data
	metadata = pd.concat([inst_info_xpr, inst_info_ctl], axis=0).reset_index(inplace=False, drop=True)
	matrix = pd.concat([matrix_xpr, matrix_ctl], 1)
	metadata.to_csv(osp.join(outdir, 'all_metadata.txt'))



	#Normalization 
	#0. create dictionary of gene_id: sample_id
		#gene symbol -> gene id
	dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))
		#gene id -> sample id
	dict_gene_id_sample_id= dict()
	for i in range(len(inst_info_xpr)):
		gene_symbol = inst_info_xpr.at[i, 'cmap_name']
		gene_id = dict_symbol_id[gene_symbol]
		if gene_id in dict_gene_id_sample_id:
			dict_gene_id_sample_id[gene_id].append(inst_info_xpr.at[i, 'sample_id'])
		else:
			dict_gene_id_sample_id[gene_id] = [inst_info_xpr.at[i, 'sample_id']]


	#1. Iterate through each gene, mask out the samples in which it was perturbed, and normalize --> set to 1 those genes that have value of average + 2std higher than control
	#for the perturbed genes = 0
	#Create matrix of NGenes x NExperiments (add column name as sample_id)
	matrix_binarized = pd.DataFrame(np.zeros_like(matrix), index = matrix.index, columns = matrix.columns)
	i = 1
	higher_than_thr = []
	higher_than_thr_ids = []


	mask_norm = list(matrix_ctl.columns) #mask_norm is controls only

	# averages = np.mean(matrix[mask_norm], 1)
	# stds = np.std(matrix[mask_norm], 1)
	# thresholds = averages + (2*stds)
	if use_log:
		matrix = np.log2(matrix + 1)

	averages = np.mean(matrix[mask_norm], 1)
	stds = np.std(matrix[mask_norm], 1)
	thresholds = averages + (2*stds)

	#hist of values
	mv = matrix.values.flatten()
	sampling = sample(range(len(mv)), int(0.1*len(mv)))
	mv = mv[sampling]

	fig, ax = plt.subplots(figsize=(16,6))
	ax.hist(mv)
	ax.set_title('Histogram of values')
	fig.savefig(osp.join(outdir,'histogram.png'))
	plt.close()


	for gene_id in list(matrix.index):
		print('{}/{}'.format(i, len(matrix)))
		#mask of elements to use for normalization: the ones in control samples only
		#mask of elements to binarize to 0 (samples in which gene_id is perturbed)
		if gene_id in dict_gene_id_sample_id:
			mask_pert = dict_gene_id_sample_id[gene_id]		
		else:
			mask_pert = None
		#normalize
		threshold = thresholds.loc[gene_id]
		matrix_binarized.loc[gene_id] = (matrix.loc[gene_id] >= threshold).astype(int).values
		matrix_binarized.loc[gene_id][mask_pert] = 0
		i+=1
		#some stats
		if mask_pert is not None:
			gte = matrix.loc[gene_id][mask_pert] >= threshold
			higher_than_thr += gte.values.astype(int).tolist()
			higher_than_thr_ids += matrix.loc[gene_id][mask_pert].index[np.where(gte)].tolist()




	###Filter columns (samples) in which the perturbed gene has an expression value that is >= the threshold used to binarize
	# log_handle.write('Filtering:\t{} columns/samples because the perturbed gene has an expression value >= the threshold used to binarize -- TEMPORARY STEP THAT SHOULD BE REMOVED LATER ON\n'.format(len(higher_than_thr_ids)))

	# print('Filtering columns from data...')
	# keep_columns = list(set(matrix_binarized.columns) - set(higher_than_thr_ids))
	# matrix_binarized = matrix_binarized[keep_columns]



	#plot higher than thr
	fig, ax1 = plt.subplots(1, figsize=(16,6))
	ax1.hist(higher_than_thr)
	ax1.set_title('Perturbed genes values >= threshold')
	fig.savefig(osp.join(outdir,'exploration_ge_crispr_higher_than_thr.png'))
	log_handle.write('After binarizing, stats, perturbed gene values would be higher than threshold in :\t{} samples\n'.format(np.sum(higher_than_thr)))


	#2. Save data and metadata for each condition and cell line
	#CRISPR + cell lines
	#Control + cell lines
	log_handle.write('----------------\n----------------\nDATA MATRICES\n')
	log_handle.write('CELL\tPERT\t\tSIZE\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
	metadata.index = metadata['sample_id']
	metadata = metadata.loc[matrix_binarized.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
	for cell_line in keep_cell_lines:
		for pert_type in ['trt_xpr', 'ctl_vector']:
			metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
			data_i = matrix_binarized[metadata_i.index]
			metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
			filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
			np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
			log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), len(set(metadata_i['cmap_name'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
	log_handle.write('\n\n------\nSTATS\n------\n')		

	return


def binarize_genewise_comparing_to_control(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, gene_info, keep_cell_lines, log_handle, outdir, use_log):
	log_handle.write('\n\n------\nBINARIZING GENEWISE COMPARING TO CONTROL\n------\n')
	if use_log:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control_lognorm')
	else:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control')
	os.makedirs(outdir, exist_ok= True)

	########################################################################################
	#All data
	metadata = pd.concat([inst_info_xpr, inst_info_ctl], axis=0).reset_index(inplace=False, drop=True)
	metadata.to_csv(osp.join(outdir, 'all_metadata.txt'))
	matrix = pd.concat([matrix_xpr, matrix_ctl], 1)


	#0. create dictionary of gene_id: sample_id
		#gene symbol -> gene id
	dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))
		#gene id -> sample id
	dict_gene_id_sample_id= dict()
	for i in range(len(inst_info_xpr)):
		gene_symbol = inst_info_xpr.at[i, 'cmap_name']
		gene_id = dict_symbol_id[gene_symbol]
		if gene_id in dict_gene_id_sample_id:
			dict_gene_id_sample_id[gene_id].append(inst_info_xpr.at[i, 'sample_id'])
		else:
			dict_gene_id_sample_id[gene_id] = [inst_info_xpr.at[i, 'sample_id']]


	matrix_binarized = pd.DataFrame(np.zeros_like(matrix), index = matrix.index, columns = matrix.columns)

	i = 1
	higher_than_thr = []
	higher_than_thr_ids = []
	control_corrected = []
	pert_corrected = []

	#lognorm
	if use_log:
		matrix = np.log2(matrix + 1)

	#hist of values
	mv = matrix.values.flatten()
	sampling = sample(range(len(mv)), int(0.1*len(mv)))
	mv = mv[sampling]

	fig, ax = plt.subplots(figsize=(16,6))
	ax.hist(mv)
	ax.set_title('Histogram of values')
	fig.savefig(osp.join(outdir,'histogram.png'))
	plt.close()

	for cell_line in list(set(metadata['cell_iname'])):
		matrix_i = matrix[metadata[metadata['cell_iname']==cell_line]['sample_id']]
		#Normalization 
		#1. Iterate through each gene, mask out the samples in which it was perturbed, and normalize --> set to 1 those genes that have value of average + 2std higher than control
		#for the perturbed genes = 0
		#Create matrix of NGenes x NExperiments (add column name as sample_id)
		mask_norm = list(set(matrix_ctl.columns).intersection(set(matrix_i.columns))) #mask_norm is controls only (for specific cell line 'cell_line')
		control_corrected += mask_norm
		averages = np.mean(matrix[mask_norm], 1)
		stds = np.std(matrix[mask_norm], 1)
		thresholds = averages + (2*stds)
		for gene_id in list(matrix_i.index):
			#mask of elements to use for normalization: the ones in control samples only
			#mask of elements to binarize to 0 (samples in which gene_id is perturbed)
			if gene_id in dict_gene_id_sample_id:
				mask_pert = list(set(dict_gene_id_sample_id[gene_id]).intersection(set(matrix_i.columns)))	#mask_pert for specific cell line 'cell_line'
				pert_corrected += mask_pert
			else:
				mask_pert = []
			#normalize
			threshold = thresholds.loc[gene_id]
			matrix_binarized.loc[gene_id][matrix_i.columns] = (matrix_i.loc[gene_id] >= threshold).astype(int).values
			matrix_binarized.loc[gene_id][mask_pert] = 0
			#some stats
			if mask_pert != []:
				gte = matrix_i.loc[gene_id][mask_pert] >= threshold
				higher_than_thr += gte.values.astype(int).tolist()
				higher_than_thr_ids += matrix_i.loc[gene_id][mask_pert].index[np.where(gte)].tolist()
		print('{}/{}'.format(i, len(list(set(metadata['cell_iname'])))))
		i+=1

	print('Controls covered:{}/{}\n'.format(len(control_corrected), len(inst_info_ctl)))
	print('Perturbed covered:{}/{}\n'.format(len(pert_corrected), len(inst_info_xpr)))

	###Filter columns (samples) in which the perturbed gene has an expression value that is >= the threshold used to binarize
	# log_handle.write('Filtering:\t{} columns/samples because the perturbed gene has an expression value >= the threshold used to binarize -- TEMPORARY STEP THAT SHOULD BE REMOVED LATER ON\n'.format(len(higher_than_thr_ids)))

	# print('Filtering columns from data...')
	# keep_columns = list(set(matrix_binarized.columns) - set(higher_than_thr_ids))
	# matrix_binarized = matrix_binarized[keep_columns]



	#plot higher than thr
	fig, ax1 = plt.subplots(1, figsize=(16,6))
	ax1.hist(higher_than_thr)
	ax1.set_title('Perturbed genes values >= threshold')
	fig.savefig(osp.join(outdir,'exploration_ge_crispr_higher_than_thr.png'))
	log_handle.write('After binarizing, stats, perturbed gene values would be higher than threshold in :\t{} samples\n'.format(np.sum(higher_than_thr)))


	#2. Save data and metadata for each condition and cell line
	#CRISPR + cell lines
	#Control + cell lines
	log_handle.write('----------------\n----------------\nDATA MATRICES\n')
	log_handle.write('CELL\tPERT\t\tSIZE\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
	metadata.index = metadata['sample_id']
	metadata = metadata.loc[matrix_binarized.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
	for cell_line in keep_cell_lines:
		for pert_type in ['trt_xpr', 'ctl_vector']:
			metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
			data_i = matrix_binarized[metadata_i.index]
			metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
			filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
			np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
			log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), len(set(metadata_i['cmap_name'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
	log_handle.write('\n\n------\nSTATS\n------\n')		

	return





def binarize_genewise_comparing_to_control_augmented(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, gene_info, keep_cell_lines, log_handle, outdir, use_log):
	log_handle.write('\n\n------\nBINARIZING GENEWISE COMPARING TO CONTROL\n------\n')
	if use_log:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control_lognorm/augmented')
	else:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control/augmented')
	os.makedirs(outdir, exist_ok= True)

	########################################################################################
	#All data
	metadata = pd.concat([inst_info_xpr, inst_info_ctl], axis=0).reset_index(inplace=False, drop=True)
	metadata.to_csv(osp.join(outdir, 'all_metadata.txt'))
	matrix = pd.concat([matrix_xpr, matrix_ctl], 1)


	#0. create dictionary of gene_id: sample_id
		#gene symbol -> gene id
	dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))
		#gene id -> sample id
	dict_gene_id_sample_id= dict()
	for i in range(len(inst_info_xpr)):
		gene_symbol = inst_info_xpr.at[i, 'cmap_name']
		gene_id = dict_symbol_id[gene_symbol]
		if gene_id in dict_gene_id_sample_id:
			dict_gene_id_sample_id[gene_id].append(inst_info_xpr.at[i, 'sample_id'])
		else:
			dict_gene_id_sample_id[gene_id] = [inst_info_xpr.at[i, 'sample_id']]

	#lognorm
	if use_log:
		matrix = np.log2(matrix + 1)

	matrix_augmented = matrix.copy()
	###Data augmentation using Gaussian noise
	AUG_PROPORTION = 5
	columns = matrix.columns
	for i in range(AUG_PROPORTION):
		columns_i = [e+'___{}'.format(i) for e in columns]
		noise = np.random.normal(0,1,matrix.shape)
		to_add = pd.DataFrame(matrix.values + noise, columns = columns_i, index = matrix.index)
		matrix_augmented = pd.concat([matrix_augmented, to_add], 1)


	matrix = matrix_augmented

	matrix_binarized = pd.DataFrame(np.zeros_like(matrix), index = matrix.index, columns = matrix.columns)

	i = 1
	higher_than_thr = []
	higher_than_thr_ids = []
	control_corrected = []
	pert_corrected = []



	#hist of values
	# mv = matrix.values.flatten()
	# sampling = sample(range(len(mv)), int(0.005*len(mv)))
	# mv = mv[sampling]

	# fig, ax = plt.subplots(figsize=(16,6))
	# ax.hist(mv)
	# ax.set_title('Histogram of values')
	# fig.savefig(osp.join(outdir,'histogram.png'))
	# plt.close()

	for cell_line in list(set(metadata['cell_iname'])):
		columns = metadata[metadata['cell_iname']==cell_line]['sample_id'].tolist()
		columns_augmented = [e+'___{}'.format(i) for i in range(AUG_PROPORTION) for e in columns] + columns
		columns = columns_augmented
		matrix_i = matrix[columns]
		#Normalization 
		#1. Iterate through each gene, mask out the samples in which it was perturbed, and normalize --> set to 1 those genes that have value of average + 2std higher than control
		#for the perturbed genes = 0
		#Create matrix of NGenes x NExperiments (add column name as sample_id)
		columns_control = list(matrix_ctl.columns)
		columns_control = [e+'___{}'.format(i) for i in range(AUG_PROPORTION) for e in columns_control] + columns_control
		columns_i = list(matrix_i.columns)
		mask_norm = list(set(columns_control).intersection(set(columns_i))) #mask_norm is controls only (for specific cell line 'cell_line')
		control_corrected += mask_norm
		averages = np.mean(matrix[mask_norm], 1)
		stds = np.std(matrix[mask_norm], 1)
		thresholds = averages + (2*stds)

		for gene_id in list(matrix_i.index):
			#mask of elements to use for normalization: the ones in control samples only
			#mask of elements to binarize to 0 (samples in which gene_id is perturbed)
			if gene_id in dict_gene_id_sample_id:
				samples_perturbed = dict_gene_id_sample_id[gene_id]
				samples_perturbed = [e+'___{}'.format(i) for i in range(AUG_PROPORTION) for e in samples_perturbed] + samples_perturbed
				mask_pert = list(set(samples_perturbed).intersection(set(matrix_i.columns)))	#mask_pert for specific cell line 'cell_line'
				pert_corrected += mask_pert
			else:
				mask_pert = []
			#normalize
			threshold = thresholds.loc[gene_id]
			matrix_binarized.loc[gene_id][matrix_i.columns] = (matrix_i.loc[gene_id] >= threshold).astype(int).values
			matrix_binarized.loc[gene_id][mask_pert] = 0
			#some stats
			if mask_pert != []:
				gte = matrix_i.loc[gene_id][mask_pert] >= threshold
				higher_than_thr += gte.values.astype(int).tolist()
				higher_than_thr_ids += matrix_i.loc[gene_id][mask_pert].index[np.where(gte)].tolist()
		print('{}/{}'.format(i, len(list(set(metadata['cell_iname'])))))
		i+=1

	# print('Controls covered:{}/{}\n'.format(len(control_corrected), len(inst_info_ctl)))
	# print('Perturbed covered:{}/{}\n'.format(len(pert_corrected), len(inst_info_xpr)))

	###Filter columns (samples) in which the perturbed gene has an expression value that is >= the threshold used to binarize
	# log_handle.write('Filtering:\t{} columns/samples because the perturbed gene has an expression value >= the threshold used to binarize -- TEMPORARY STEP THAT SHOULD BE REMOVED LATER ON\n'.format(len(higher_than_thr_ids)))

	# print('Filtering columns from data...')
	# keep_columns = list(set(matrix_binarized.columns) - set(higher_than_thr_ids))
	# matrix_binarized = matrix_binarized[keep_columns]



	#plot higher than thr
	fig, ax1 = plt.subplots(1, figsize=(16,6))
	ax1.hist(higher_than_thr)
	ax1.set_title('Perturbed genes values >= threshold')
	fig.savefig(osp.join(outdir,'exploration_ge_crispr_higher_than_thr.png'))
	log_handle.write('After binarizing, stats, perturbed gene values would be higher than threshold in :\t{} samples\n'.format(np.sum(higher_than_thr)))


	#2. Save data and metadata for each condition and cell line
	#CRISPR + cell lines
	#Control + cell lines
	log_handle.write('----------------\n----------------\nDATA MATRICES\n')
	log_handle.write('CELL\tPERT\t\tSIZE\tAUGMENTED SIZE\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
	metadata.index = metadata['sample_id']
	# metadata = metadata.loc[matrix_binarized.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
	for cell_line in keep_cell_lines:
		for pert_type in ['trt_xpr', 'ctl_vector']:
			metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
			columns = list(metadata_i.index)
			to_add = []
			for i in range(AUG_PROPORTION):
				to_add += [e+'___{}'.format(i) for e in columns]
			columns = columns + to_add
			data_i = matrix_binarized[columns]
			metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
			filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
			np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
			log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), data_i.shape[1], len(set(metadata_i['cmap_name'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
	log_handle.write('\n\n------\nSTATS\n------\n')		

	return




def normalize_and_save(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, gene_info, keep_cell_lines, log_handle, outdir, use_log):
	log_handle.write('\n\n------\nNORMALIZE DATA AND SAVE\n------\n')
	if use_log:
		outdir = osp.join(outdir, 'real_lognorm')
	else:
		outdir = osp.join(outdir, 'real')
	os.makedirs(outdir, exist_ok= True)
	

	#All data
	metadata = pd.concat([inst_info_xpr, inst_info_ctl], axis=0).reset_index(inplace=False, drop=True)
	metadata.to_csv(osp.join(outdir, 'all_metadata.txt'))
	matrix = pd.concat([matrix_xpr, matrix_ctl], 1)
	del(matrix_xpr) 

	#hist of values before lognorm
	mv = matrix.values.flatten()
	sampling = sample(range(len(mv)), int(0.001*len(mv)))
	mv = mv[sampling]

	fig, ax = plt.subplots(figsize=(16,6))
	ax.hist(mv)
	ax.set_title('Histogram of values')
	fig.savefig(osp.join(outdir,'histogram_raw.png'))
	plt.close()

	#NORMALIZATION
	#lognorm + minmax
	if use_log:
		matrix = np.log2(matrix + 1)

	scaler = MinMaxScaler((0,1))
	matrix = matrix.transpose()
	matrix = pd.DataFrame(scaler.fit_transform(matrix), columns = matrix.columns, index = matrix.index)
	matrix = matrix.transpose()



	#hist of values after lognorm
	mv = matrix.values.flatten()
	sampling = sample(range(len(mv)), int(0.001*len(mv)))
	mv = mv[sampling]

	fig, ax = plt.subplots(figsize=(16,6))
	ax.hist(mv)
	ax.set_title('Histogram of values')
	fig.savefig(osp.join(outdir,'histogram_lognorm.png'))
	plt.close()

	#2. Save data and metadata for each condition and cell line
	#Compound + cell lines
	#Control + cell lines
	log_handle.write('----------------\n----------------\nDATA MATRICES\n')
	log_handle.write('CELL\tPERT\t\tSIZE\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
	metadata.index = metadata['sample_id']
	metadata = metadata.loc[matrix.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
	for cell_line in keep_cell_lines:
		for pert_type in ['trt_xpr', 'ctl_vector']:
			metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
			data_i = matrix[metadata_i.index]
			metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
			filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
			np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
			log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), len(set(metadata_i['pert_id'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
	log_handle.write('\n\n------\nSTATS\n------\n')		

	return




def main():  

	DATA_ROOT = "../../raw/lincs/2022-02-LINCS_Level3/data/"
	log_handle = open(osp.join(outdir, 'process_data_lognorm.txt'), 'w')
	inst_info_xpr, inst_info_ctl, gene_info, matrix_xpr, matrix_ctl = loads_data(DATA_ROOT, log_handle)
	inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl = filter_data_metadata(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, log_handle)
	inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, keep_cell_lines = filter_cell_lines_custom(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, log_handle)
	# inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, keep_cell_lines = filter_cell_lines(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, log_handle)
	inst_info_xpr, matrix_xpr = filter_samples_with_unknown_perturbed_genes(inst_info_xpr, matrix_xpr, gene_info, log_handle)
	use_log=True
	normalize_and_save(inst_info_xpr, matrix_xpr, inst_info_ctl, matrix_ctl, gene_info, keep_cell_lines, log_handle, outdir, use_log)
	log_handle.close()


if __name__ == "__main__":
    main()


