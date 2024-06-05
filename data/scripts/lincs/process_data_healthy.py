'''
Process LINCS data of healhty cell lines
MCF10A, NL20, RWPE1
Will do some processing first and then rely on the functions in process_data.py
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
DATA_ROOT = "../../raw/lincs/2022-02-LINCS_Level3/data/"


################
# Data loading
################

#function updated from the one in process_data to load the cell lines of interest
def loads_data(DATA_ROOT, log_handle):
	healhty_cell_lines = ['MCF10A', 'NL20', 'RWPE1']

	#Loads metadata
	inst_info = pd.read_csv(os.path.join(DATA_ROOT, 'instinfo_beta.txt'), sep="\t", low_memory=False)

	inst_info_ctl_mcf10a = inst_info[np.logical_and(inst_info['cell_iname'] == 'MCF10A',np.logical_and(inst_info['pert_type'] == 'ctl_untrt', inst_info['failure_mode'].isna())) ].reset_index(inplace=False, drop=True)
	inst_info_ctl_nl20 = inst_info[np.logical_and(inst_info['cell_iname'] == 'NL20',np.logical_and(inst_info['pert_type'] == 'ctl_vehicle', inst_info['failure_mode'].isna())) ].reset_index(inplace=False, drop=True)
	inst_info_ctl_rwpe1 = inst_info[np.logical_and(inst_info['cell_iname'] == 'RWPE1',np.logical_and(inst_info['pert_type'] == 'ctl_vector', inst_info['failure_mode'].isna())) ].reset_index(inplace=False, drop=True)


	inst_info_ctl = pd.concat([inst_info_ctl_mcf10a, inst_info_ctl_nl20, inst_info_ctl_rwpe1])

	gene_info = pd.read_csv(os.path.join(DATA_ROOT, 'geneinfo_beta.txt'), sep="\t", low_memory=False)


	####################
	#Loads data matrices


	### Control data -- filter to keep only those in my metadata
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

	return inst_info_ctl, gene_info, matrix_ctl




################
# Processing
################

################
#1. Filter column metadata and data matrix to keep only those in metadata

def filter_data_metadata(inst_info_ctl, matrix_ctl, log_handle):
	log_handle.write('Filtering to keep only those in metadata\n------\n')
	#CONTROL
	list_ids = list(inst_info_ctl['sample_id'])	#in metadata
	#extra steps
	#--
	list_ids = list(set(list_ids).intersection(set(matrix_ctl.columns.astype(str))))	#in metadata and in data matrix (some of metadata are not in data matrix)
	inst_info_ctl.index = inst_info_ctl['sample_id']; inst_info_ctl = inst_info_ctl.loc[list_ids].reset_index(inplace=False, drop=True) #remove entries from metadata that are not in data matrix
	#--
	matrix_ctl = matrix_ctl[list_ids]	#Filtered data matrix
	log_handle.write('CONTROL:\t{} datapoints\n\n\n'.format(matrix_ctl.shape[1]))
	return inst_info_ctl, matrix_ctl


################
#3. Normalize (binarize), and save


def binarize_genewise_comparing_to_control(inst_info_ctl, matrix_ctl, gene_info, log_handle, outdir, use_log):
	log_handle.write('\n\n------\nBINARIZING GENEWISE COMPARING TO CONTROL\n------\n')
	if use_log:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control_lognorm')
	else:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control')
	os.makedirs(outdir, exist_ok= True)




	########################################################################################
	#All data
	metadata = inst_info_ctl
	metadata.to_csv(osp.join(outdir, 'all_metadata_healthy.txt'))
	matrix = matrix_ctl


	matrix_binarized = pd.DataFrame(np.zeros_like(matrix), index = matrix.index, columns = matrix.columns)

	i = 1

	control_corrected = []


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
	fig.savefig(osp.join(outdir,'histogram_healthy.png'))
	plt.close()

	for cell_line in list(set(metadata['cell_iname'])):
		matrix_i = matrix[metadata[metadata['cell_iname']==cell_line]['sample_id']]
		#Normalization 
		#Create matrix of NGenes x NExperiments (add column name as sample_id)
		mask_norm = list(set(matrix_ctl.columns).intersection(set(matrix_i.columns))) #mask_norm is controls only (for specific cell line 'cell_line')
		control_corrected += mask_norm
		averages = np.mean(matrix[mask_norm], 1)
		stds = np.std(matrix[mask_norm], 1)
		thresholds = averages + (2*stds)
		for gene_id in list(matrix_i.index):
			#normalize
			threshold = thresholds.loc[gene_id]
			matrix_binarized.loc[gene_id][matrix_i.columns] = (matrix_i.loc[gene_id] >= threshold).astype(int).values
			
		print('{}/{}'.format(i, len(list(set(metadata['cell_iname'])))))
		i+=1




	print('Controls covered:{}/{}\n'.format(len(control_corrected), len(inst_info_ctl)))

	#2. Save data and metadata for each condition and cell line
	#CRISPR + cell lines
	#Control + cell lines
	log_handle.write('----------------\n----------------\nDATA MATRICES\n')
	log_handle.write('CELL\tPERT\t\tSIZE\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
	metadata.index = metadata['sample_id']
	metadata = metadata.loc[matrix_binarized.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
	for cell_line, pert_type in zip(['MCF10A', 'NL20', 'RWPE1'],['ctl_untrt', 'ctl_vehicle', 'ctl_vector'] ):
		metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
		data_i = matrix_binarized[metadata_i.index]
		metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
		filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
		np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
		log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), len(set(metadata_i['cmap_name'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
	log_handle.write('\n\n------\nSTATS\n------\n')		

	return



def binarize_genewise_comparing_to_control_augmented(inst_info_ctl, matrix_ctl, gene_info, log_handle, outdir, use_log):
	log_handle.write('\n\n------\nBINARIZING GENEWISE COMPARING TO CONTROL\n------\n')
	if use_log:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control_lognorm/augmented')
	else:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control/augmented')
	os.makedirs(outdir, exist_ok= True)




	########################################################################################
	#All data
	metadata = inst_info_ctl
	metadata.to_csv(osp.join(outdir, 'all_metadata_healthy.txt'))
	matrix = matrix_ctl
	

	#lognorm
	if use_log:
		matrix = np.log2(matrix + 1)


	matrix_augmented = matrix.copy()
	###Data augmentation using Gaussian noise
	AUG_PROPORTION = 10
	columns = matrix.columns
	for i in range(AUG_PROPORTION):
		columns_i = [e+'___{}'.format(i) for e in columns]
		noise = np.random.normal(0,1,matrix.shape)
		to_add = pd.DataFrame(matrix.values + noise, columns = columns_i, index = matrix.index)
		matrix_augmented = pd.concat([matrix_augmented, to_add], 1)


	matrix = matrix_augmented
	matrix_binarized = pd.DataFrame(np.zeros_like(matrix), index = matrix.index, columns = matrix.columns)

	i = 1

	control_corrected = []




	#hist of values
	mv = matrix.values.flatten()
	sampling = sample(range(len(mv)), int(0.1*len(mv)))
	mv = mv[sampling]

	fig, ax = plt.subplots(figsize=(16,6))
	ax.hist(mv)
	ax.set_title('Histogram of values')
	fig.savefig(osp.join(outdir,'histogram_healthy.png'))
	plt.close()

	for cell_line in list(set(metadata['cell_iname'])):
		columns = metadata[metadata['cell_iname']==cell_line]['sample_id'].tolist()
		columns_augmented = [e+'___{}'.format(i) for i in range(AUG_PROPORTION) for e in columns] + columns
		columns = columns_augmented
		matrix_i = matrix[columns]
		#Normalization 
		#Create matrix of NGenes x NExperiments (add column name as sample_id)
		#Binarization
		mask_norm = list(set(matrix_i.columns)) #mask_norm is controls only (for specific cell line 'cell_line')
		control_corrected += mask_norm
		averages = np.mean(matrix[mask_norm], 1)
		stds = np.std(matrix[mask_norm], 1)
		thresholds = averages + (2*stds)
		for gene_id in list(matrix_i.index):
			#normalize
			threshold = thresholds.loc[gene_id]
			matrix_binarized.loc[gene_id][matrix_i.columns] = (matrix_i.loc[gene_id] >= threshold).astype(int).values
		print('{}/{}'.format(i, len(list(set(metadata['cell_iname'])))))
		i+=1




	print('Controls covered:{}/{}\n'.format(len(control_corrected), matrix.shape[1]))

	#2. Save data and metadata for each condition and cell line
	#Control + cell lines
	log_handle.write('----------------\n----------------\nDATA MATRICES\n')
	log_handle.write('CELL\tPERT\t\tSIZE\tAUGMENTED SIZE\t\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
	metadata.index = metadata['sample_id']
	# metadata = metadata.loc[matrix_binarized.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
	for cell_line, pert_type in zip(['MCF10A', 'NL20', 'RWPE1'],['ctl_untrt', 'ctl_vehicle', 'ctl_vector'] ):
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


def normalize_and_save(inst_info_ctl, matrix_ctl, gene_info, log_handle, outdir, use_log):
	log_handle.write('\n\n------\nBINARIZING GENEWISE COMPARING TO CONTROL\n------\n')
	if use_log:
		outdir = osp.join(outdir, 'real_lognorm')
	else:
		outdir = osp.join(outdir, 'real')
	os.makedirs(outdir, exist_ok= True)


	########################################################################################
	#All data
	metadata = inst_info_ctl
	metadata.to_csv(osp.join(outdir, 'all_metadata_healthy.txt'))
	matrix = matrix_ctl


	#lognorm
	if use_log:
		matrix = np.log2(matrix + 1)
	scaler = MinMaxScaler((0,1))
	matrix = matrix.transpose()
	matrix = pd.DataFrame(scaler.fit_transform(matrix), columns = matrix.columns, index = matrix.index)
	matrix = matrix.transpose()

	#hist of values
	mv = matrix.values.flatten()
	sampling = sample(range(len(mv)), int(0.1*len(mv)))
	mv = mv[sampling]

	fig, ax = plt.subplots(figsize=(16,6))
	ax.hist(mv)
	ax.set_title('Histogram of values')
	fig.savefig(osp.join(outdir,'histogram_healthy.png'))
	plt.close()



	#2. Save data and metadata for each condition and cell line
	#CRISPR + cell lines
	#Control + cell lines
	log_handle.write('----------------\n----------------\nDATA MATRICES\n')
	log_handle.write('CELL\tPERT\t\tSIZE\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
	metadata.index = metadata['sample_id']
	metadata = metadata.loc[matrix.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
	for cell_line, pert_type in zip(['MCF10A', 'NL20', 'RWPE1'],['ctl_untrt', 'ctl_vehicle', 'ctl_vector'] ):
		metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
		data_i = matrix[metadata_i.index]
		metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
		filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
		np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
		log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), len(set(metadata_i['cmap_name'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
	log_handle.write('\n\n------\nSTATS\n------\n')		

	return





def main():  
	from process_data import stats_control


	DATA_ROOT = "../../raw/lincs/2022-02-LINCS_Level3/data/"
	log_handle = open(osp.join(outdir, 'process_data_healthy_lognorm.txt'), 'w')
	inst_info_ctl, gene_info, matrix_ctl = loads_data(DATA_ROOT, log_handle)
	inst_info_ctl, matrix_ctl = filter_data_metadata(inst_info_ctl, matrix_ctl, log_handle)
	#stats_control(inst_info_ctl, log_handle)
	use_log=True
	normalize_and_save(inst_info_ctl, matrix_ctl, gene_info, log_handle, outdir, use_log)
	log_handle.close()


if __name__ == "__main__":
    main()






