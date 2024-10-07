''' 
Normalizes perturbed (treated) and control (diseased) data of chemical interventions
Saves data as npz

1. Reads data (load_data)
2. Filters to keep only the cell lines that we use in experiments (filter_cell_lines_custom)
3. Filters to keep only the samples in metadata (filter_data_metadata)
4. Maps drug gene targets to LINCS genes (map_gene_targets_to_lincs)
5. Filters out samples for which the drug targets are not in LINCS genes (filter_samples_with_unknown_perturbed_genes)
6. Normalizes data between (0,1) and save (normalize_and_save)
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
from sklearn.preprocessing import MinMaxScaler
import math
from random import sample


import networkx as nx
import csrgraph as cg
import os
import itertools
import json
import operator
import swifter
from tqdm import tqdm
import time
import pickle


	#LOG
	

################
# Data loading
################

def get_gene_names(x, dict_id_names):
	# Returns a list of gene names for each protein ID (node ID of protein). 
	# If there is more than one gene name for a given ID, a list of all of them are given.
	gene_names = []
	for gene_id in x:
		gname = dict_id_names[gene_id]
		if gname ==[]:
			continue
		else:
			gene_names.append(dict_id_names[gene_id])
	return gene_names



def targetList(pid,dict_pid_target_names):
	if pid not in dict_pid_target_names:
		return []  
	else:
		return dict_pid_target_names[pid]
		



def loads_data(DATA_ROOT, log_handle):
	
	#Loads metadata
	inst_info = pd.read_csv(os.path.join(DATA_ROOT, 'instinfo_beta.txt'), sep="\t", low_memory=False)
	inst_info_cp = inst_info[np.logical_and(inst_info['pert_type'] == 'trt_cp',  inst_info['failure_mode'].isna())].reset_index(inplace=False, drop=True)
	inst_info_ctl = inst_info[np.logical_and(inst_info['pert_type'] == 'ctl_vehicle', inst_info['failure_mode'].isna()) ].reset_index(inplace=False, drop=True)
	gene_info = pd.read_csv(os.path.join(DATA_ROOT, 'geneinfo_beta.txt'), sep="\t", low_memory=False)
	
	with open('../../processed/lincs/chemical/dataframes/df_targets.pickle', 'rb') as f:
		df_targets = pickle.load(f)
	

	
	drugbank_targets=pd.read_csv("../../processed/drugbank/targets.txt")
	
	dict_id_names = dict()
	for i in range(len(drugbank_targets)):
		name = drugbank_targets.at[i, 'gene_name']
		if name != '-':
			dict_id_names[drugbank_targets.at[i,'target_id']] = [drugbank_targets.at[i,'gene_name']]
		else:
			dict_id_names[drugbank_targets.at[i,'target_id']] = []
		synonyms = drugbank_targets.at[i, 'gene_synonyms']
		if synonyms != '-' and str(synonyms) != 'nan':
			dict_id_names[drugbank_targets.at[i,'target_id']] += synonyms.split('||')

	
	df_targets['target_names']=df_targets['targets'].apply(lambda x: get_gene_names(x, dict_id_names))
	dict_pid_target_names = dict(zip(df_targets['pert_id'], df_targets['target_names']))
	inst_info_cp['drugbank_target_names']=inst_info_cp['pert_id'].apply(lambda x: targetList(x,dict_pid_target_names))

	
	####################
	#Loads data matrices
	### Compound
	f = h5py.File(os.path.join(DATA_ROOT, 'level3_beta_trt_cp_n1805898x12328.gctx'), 'r')
	matrix_cp = f['0']['DATA']['0']['matrix'][:].transpose()
	gene_ids_cp = f['0']['META']['ROW']['id'][:]
	sample_ids_cp = f['0']['META']['COL']['id'][:]
	matrix_cp = pd.DataFrame(matrix_cp, columns = sample_ids_cp.astype(str), index = gene_ids_cp.astype(int))
	del f

	
	#re-order gene_info based on the order in gene_ids_cp (rows of data)
	gene_info.index = gene_info['gene_id']
	gene_info = gene_info.loc[gene_ids_cp.astype(int)].reset_index(inplace=False, drop=True)
	gene_info.to_csv(osp.join(outdir, 'gene_info.txt'), index=False)

	#Stats
	log_handle.write('Compounds\n------\n')
	log_handle.write('Compound entries in inst_info metadata:\t{}\n'.format(len(inst_info_cp)))
	log_handle.write('Compound entries in data matrix:\t{}\n'.format(len(sample_ids_cp)))
	log_handle.write('Overlap between inst_info metadata and sample ids in data matrix:\t{}\n'.format(len(set(inst_info_cp['sample_id']).intersection(set(sample_ids_cp.astype(str))))))



	### Control data
	f = h5py.File(os.path.join(DATA_ROOT, 'level3_beta_ctl_n188708x12328.gctx'), 'r')
	matrix_ctl = f['0']['DATA']['0']['matrix'][:].transpose()
	gene_ids_ctl = f['0']['META']['ROW']['id'][:]					#not in the same order as gene_ids_cp
	sample_ids_ctl = f['0']['META']['COL']['id'][:]
	matrix_ctl = pd.DataFrame(matrix_ctl, columns = sample_ids_ctl.astype(str), index = gene_ids_ctl.astype(int))
	del f


	#Stats
	log_handle.write('CONTROL\n------\n')
	log_handle.write('Control entries in inst_info metadata:\t{}\n'.format(len(inst_info_ctl)))
	log_handle.write('Control entries in data matrix:\t{}\n'.format(len(sample_ids_ctl)))
	log_handle.write('Overlap between inst_info metadata and sample ids in data matrix:\t{}\n'.format(len(set(inst_info_ctl['sample_id']).intersection(set(sample_ids_ctl.astype(str))))))
	log_handle.write('\n------\n')


	# stats_data(inst_info_cp, matrix_cp, matrix_ctl, gene_info, df_targets)

	return sample_ids_cp, inst_info_cp, inst_info_ctl, gene_info, matrix_cp, matrix_ctl, df_targets





def stats_data(inst_info_cp, matrix_cp, matrix_ctl, gene_info, df_targets):
	dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))
	dict_pert_id = dict(zip(df_targets['pert_id'], df_targets['target_names']))

		####Data exploration -- GE values of genes that are perturbed (!)
	#Get GE value for each gene perturbed by Compound
	values_pert = {}
	values_control = {}
	for i in range(len(inst_info_cp)):
		pert_id = inst_info_cp.at[i, 'pert_id']
		if pert_id in dict_pert_id:
			gene_symbols=dict_pert_id[pert_id]
			if type(gene_symbols)==list:
				for j in range(len(gene_symbols)): 
					
					if gene_symbols[j] in dict_symbol_id:					#if the cmap_name of gene is in the gene_info
						sample_id = inst_info_cp.at[i, 'sample_id']
						gene_id = dict_symbol_id[gene_symbols[j]]
						if gene_id in values_pert:
							values_pert[gene_id].append(matrix_cp.at[gene_id, sample_id])
						else:
							values_pert[gene_id] = [matrix_cp.at[gene_id, sample_id]]        
				


	for pert_id in list(set(inst_info_cp['pert_id'])):
		if pert_id in dict_pert_id:
			gene_symbols=dict_pert_id[pert_id]
			if type(gene_symbols)==list:
				for z in range(len(gene_symbols)):
		
				
				
					if gene_symbols[z] in dict_symbol_id:				#if the cmap_name of gene is in the gene_info
						gene_id = dict_symbol_id[gene_symbols[z]]
						values_control[gene_id] = [matrix_ctl.loc[gene_id]]
	
	for key in values_pert:
		values_pert[key] = np.mean(values_pert[key])

	for key in values_control:
		values_control[key] = np.mean(values_control[key])
	
	fig, (ax1, ax2) = plt.subplots(2, figsize=(16,6))
	ax1.hist(values_pert.values())
	ax2.hist(values_control.values())
	ax1.set_title('Values of perturbed genes (avg) - Compounds')
	ax2.set_title('Values of genes in control (avg)')
	fig.savefig(osp.join(outdir,'exploration_ge_compounds.png'))


	return




################
# Processing
################


################
#1. Filter to keep only cell lines with more perturbations

def stats_control(inst_info_ctl, log_handle):
	log_handle.write('STATS CONTROL DATA\n***********************************\n')
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

	log_handle.write('\n***********************************\n')

	return


def filter_cell_lines(inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, log_handle):
	log_handle.write('Filtering to keep only cell lines with highest mumber of perturbed genes\n------\n')

	#####Compound
	#Obtain cell lines with the most perturbations (> 4K genes perturbed)
	df_cp = pd.DataFrame(inst_info_cp[['cmap_name', 'cell_iname']].groupby('cell_iname', as_index=True).apply(lambda x: x['cmap_name'].unique()))
	df_cp = pd.DataFrame([(i, len(df_cp.loc[i][0])) for i in df_cp.index], columns =['cell_line', 'n_cmap_names'])
	df_cp = df_cp.sort_values(by='n_cmap_names')

	keep_cell_lines = df_cp[df_cp['n_cmap_names']>np.percentile(df_cp['n_cmap_names'], 90)]['cell_line'].tolist()

	#Find indices of samples that are on the desired cell lines
	keep_index = []
	for i in range(len(inst_info_cp)):
		if inst_info_cp.at[i, 'cell_iname'] in keep_cell_lines:
			keep_index.append(i)


	inst_info_cp = inst_info_cp.loc[keep_index].reset_index(inplace=False, drop=True) #filter from metadata
	list_ids = list(inst_info_cp['sample_id'])	#obtain sample ID from metadata
	matrix_cp = matrix_cp[list_ids]	#Filtered data matrix
	log_handle.write('Compounds:\t{} datapoints\n'.format(matrix_cp.shape[1]))



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

	return inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, keep_cell_lines




def filter_cell_lines_custom(inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, log_handle):
	log_handle.write('Filtering to keep custom list of cell lines\n------\n')

	#####Compound
	keep_cell_lines = ['A549', 'PC3', 'MCF7', 'VCAP', 'MDAMB231', 'BT20', 'HA1E', 'HT29', 'A375', 'HELA', 'YAPC']

	#Find indices of samples that are on the desired cell lines
	keep_index = []
	for i in range(len(inst_info_cp)):
		if inst_info_cp.at[i, 'cell_iname'] in keep_cell_lines:
			keep_index.append(i)


	inst_info_cp = inst_info_cp.loc[keep_index].reset_index(inplace=False, drop=True) #filter from metadata
	list_ids = list(inst_info_cp['sample_id'])	#obtain sample ID from metadata
	matrix_cp = matrix_cp[list_ids]	#Filtered data matrix
	log_handle.write('Compounds:\t{} datapoints\n'.format(matrix_cp.shape[1]))



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

	return inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, keep_cell_lines


def maxSample(w,x,y,z, max_list):
	if (w,x,y,z) in max_list: 
		return 1
	else: 
		return 0


def filter_dosage_timepoints(sample_ids_cp, inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, log_handle):
	log_handle.write('Filtering to keep only samples with the highest dosage and longest timepoint per drug-cell line combination\n------\n')
	#####Compounds   

	df_cp=pd.DataFrame(inst_info_cp[['sample_id','pert_id', 'pert_dose','pert_dose_unit','pert_time','cell_iname']])
	df_cp_max=df_cp.sort_values(by=['pert_dose','pert_time']).drop_duplicates(["pert_id","cell_iname"],keep="last")
	max_list=list(zip(df_cp_max['pert_dose'], df_cp_max['pert_time'],df_cp_max['pert_id'],df_cp_max['cell_iname']))

	#Decoding sample_ids_cp array
	decode=np.vectorize(np.char.decode)
	sample_ids_cp_dec=decode(sample_ids_cp)
	tqdm.pandas()
	df_cp['max'] = df_cp.swifter.apply(lambda row : maxSample(row['pert_dose'],row['pert_time'], row['pert_id'],row['cell_iname'], max_list), axis = 1)
	df_cp_filtered=df_cp.loc[df_cp['max']== 1]
	df_cp_filtered.to_csv("../../processed/chemical/dataframes/df_cp_filtered.csv")

	sid_index=np.intersect1d(sample_ids_cp_dec,df_cp_filtered.sample_id.to_numpy(), return_indices=True)[1]
	f1 = operator.itemgetter(*sid_index)
	sample_ids=f1(sample_ids_cp_dec)
	keep_index=inst_info_cp[inst_info_cp['sample_id'].isin(sample_ids)].index.tolist()

	inst_info_cp = inst_info_cp.loc[keep_index].reset_index(inplace=False, drop=True) #filter from metadata
	list_ids = list(inst_info_cp['sample_id'])	#obtain sample ID from metadata
	matrix_cp = matrix_cp[list_ids]	#Filtered data matrix
	log_handle.write('Compounds:\t{} datapoints\n'.format(matrix_cp.shape[1]))
	return inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl




################
#2. Filter column metadata and data matrix to keep only those in metadata

def filter_data_metadata(inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, log_handle):
	log_handle.write('Filtering to keep only those in metadata\n------\n')
	#Compund
	list_ids = list(inst_info_cp['sample_id'])	#in metadata
	matrix_cp = matrix_cp[list_ids]	#Filtered data matrix
	log_handle.write('Compounds:\t{} datapoints\n'.format(matrix_cp.shape[1]))
	#CONTROL
	list_ids = list(inst_info_ctl['sample_id'])	#in metadata
	#extra steps
	#--
	list_ids = list(set(list_ids).intersection(set(matrix_ctl.columns.astype(str))))	#in metadata and in data matrix (some of metadata are not in data matrix)
	inst_info_ctl.index = inst_info_ctl['sample_id']; inst_info_ctl = inst_info_ctl.loc[list_ids].reset_index(inplace=False, drop=True) #remove entries from metadata that are not in data matrix
	#--
	matrix_ctl = matrix_ctl[list_ids]	#Filtered data matrix
	log_handle.write('CONTROL:\t{} datapoints\n\n\n'.format(matrix_ctl.shape[1]))
	return inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl




################
#3. Map drug target names to symbols from LINCS
def map_gene_targets_to_lincs(inst_info_cp, gene_info):
	genes_in_lincs = set(gene_info.gene_symbol)
	inst_info_cp['target_names'] = ''
	for i in range(len(inst_info_cp)):
		drugbank_target_names = inst_info_cp.at[i, 'drugbank_target_names']
		if drugbank_target_names ==[]:
			continue
		else:
			target_names = []
			for target_list in drugbank_target_names:
				target_name = list(set(target_list).intersection(genes_in_lincs))
				if len(target_name) == 1:
					target_names.append(target_name[0])
			inst_info_cp.at[i, 'target_names'] = target_names

	return inst_info_cp


################
#3. Filter samples with unknown perturbed genes (keeping those with at least 1 known perturbed gene)
def filter_samples_with_unknown_perturbed_genes(inst_info_cp, matrix_cp, gene_info, log_handle):
	########################################################################################
	#First filter perturbation samples to remove those with genes not mapping to genes_info
	keep_index = []
	for i in range(len(inst_info_cp)):
		if len(inst_info_cp.at[i, 'target_names']) > 0:
			keep_index.append(i)

	inst_info_cp = inst_info_cp.loc[keep_index].reset_index(inplace=False, drop=True) #filter from metadata
	list_ids = list(inst_info_cp['sample_id'])	#obtain sample ID from metadata
	matrix_cp = matrix_cp[list_ids]	#Filtered data matrix
	log_handle.write('Filtering samples without protein targets in drugbank and lincs\n')
	log_handle.write('Compounds:\t{} datapoints\n'.format(matrix_cp.shape[1]))
	return inst_info_cp, matrix_cp





################
#4. Cretes a dictionary of gene_symbol:sample_ids
def genesymb2sampleiddict(inst_info_cp):
	#Returns a dictionary of gene_symbol: [sampleids]
	genesymb_list=list(inst_info_cp.target_names)
	genesymb_list=list(itertools.chain(*genesymb_list))
	genesymb_list_unique=list(set(genesymb_list))
			
	l=[ [] for _ in range(len(genesymb_list_unique)) ]
	d=dict(zip(genesymb_list_unique,l))
	
	for sample in range(len(inst_info_cp)):
		sid=inst_info_cp.at[sample,'sample_id']
		for target in inst_info_cp.at[sample,'target_names']:
			if target in d:
				d[target].append(sid)
	return d



def binarize_genewise_comparing_to_control(inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, gene_info, keep_cell_lines, log_handle, outdir, use_log):
	log_handle.write('\n\n------\nBINARIZING GENEWISE COMPARING TO CONTROL\n------\n')
	if use_log:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control_lognorm')
	else:
		outdir = osp.join(outdir, 'binarize_genewise_comparing_to_control')
	os.makedirs(outdir, exist_ok= True)
	

	#All data
	metadata = pd.concat([inst_info_cp, inst_info_ctl], axis=0).reset_index(inplace=False, drop=True)
	metadata.to_csv(osp.join(outdir, 'all_metadata.txt'))
	matrix = pd.concat([matrix_cp, matrix_ctl], 1)
	del(matrix_cp) 

	#Normalization 
	#0. create dictionary of gene_id: sample_id
		#gene symbol -> gene id
	dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))
		#gene symbol -> sample id
	#Passing in dictionary from genesymb2sampleiddict
	d=genesymb2sampleiddict(inst_info_cp)
	#Switching keys to be gene_ids instead of gene_symbols
	dict_gene_id_sample_id= dict((dict_symbol_id[key],value) for (key,value) in d.items())
	matrix_binarized = pd.DataFrame(np.zeros_like(matrix), index = matrix.index, columns = matrix.columns)
	del(d)
	del(dict_symbol_id)
	del(gene_info)    
	i = 1
	higher_than_thr = []
	higher_than_thr_ids = []
	control_corrected = []
	pert_corrected = []

	#lognorm
	if use_log:
		matrix = np.log2(matrix + 1)
	
	
	#hist of values
	# mv_shape=matrix.values.shape
	# num_elements = mv_shape[0]*mv_shape[1]
	# chosenCols=np.random.randint(0, mv_shape[1], size=int(0.001*num_elements))
	# chosenRows=np.random.randint(0, mv_shape[0], size=int(0.001*num_elements))
	# filter_ind=np.array(np.array(list(zip(chosenRows,chosenCols))))
	# mv=list(matrix.values[filter_ind[:,0],filter_ind[:,1]])
	# fig, ax = plt.subplots(figsize=(16,6))
	# ax.hist(mv)
	# ax.set_title('Histogram of values')
	# fig.savefig(osp.join(outdir,'histogram.png'))
	# plt.close()

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
			# matrix_binarized.loc[gene_id][mask_pert] = 0  #not setting to zero here -- it's chemical perturbations not KO
			#some stats
			if mask_pert != []:
				gte = matrix_i.loc[gene_id][mask_pert] >= threshold
				higher_than_thr += gte.values.astype(int).tolist()
				higher_than_thr_ids += matrix_i.loc[gene_id][mask_pert].index[np.where(gte)].tolist()
		print('{}/{}'.format(i, len(list(set(metadata['cell_iname'])))))
		i+=1

	print('Controls covered:{}/{}\n'.format(len(control_corrected), len(inst_info_ctl)))
	print('Perturbed covered:{}/{}\n'.format(len(set(pert_corrected)), len(inst_info_cp)))



	#plot higher than thr
	fig, ax1 = plt.subplots(1, figsize=(16,6))
	ax1.hist(higher_than_thr)
	ax1.set_title('Perturbed genes values >= threshold')
	fig.savefig(osp.join(outdir,'exploration_ge_compounds_higher_than_thr.png'))
	log_handle.write('After binarizing, stats, perturbed gene values would be higher than threshold in :\t{} samples\n'.format(np.sum(higher_than_thr)))


	#2. Save data and metadata for each condition and cell line
	#Compound + cell lines
	#Control + cell lines
	log_handle.write('----------------\n----------------\nDATA MATRICES\n')
	log_handle.write('CELL\tPERT\t\tSIZE\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
	metadata.index = metadata['sample_id']
	metadata = metadata.loc[matrix_binarized.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
	for cell_line in keep_cell_lines:
		for pert_type in ['trt_cp', 'ctl_vehicle']:
			metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
			data_i = matrix_binarized[metadata_i.index]
			metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
			filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
			np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
			log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), len(set(metadata_i['pert_id'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
	log_handle.write('\n\n------\nSTATS\n------\n')		

	return






def normalize_and_save(inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, gene_info, keep_cell_lines, log_handle, outdir, use_log):
	log_handle.write('\n\n------\nNORMALIZE DATA AND SAVE\n------\n')
	if use_log:
		outdir = osp.join(outdir, 'real_lognorm')
	else:
		outdir = osp.join(outdir, 'real')
	os.makedirs(outdir, exist_ok= True)
	

	#All data
	metadata = pd.concat([inst_info_cp, inst_info_ctl], axis=0).reset_index(inplace=False, drop=True)
	metadata.to_csv(osp.join(outdir, 'all_metadata.txt'))
	matrix = pd.concat([matrix_cp, matrix_ctl], 1)
	del(matrix_cp) 

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
		for pert_type in ['trt_cp', 'ctl_vehicle']:
			metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
			data_i = matrix[metadata_i.index]
			metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
			filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
			np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
			log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), len(set(metadata_i['pert_id'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
	log_handle.write('\n\n------\nSTATS\n------\n')		

	return



outdir = '../../processed/lincs/chemical/nofilter_dose_timepoint'
os.makedirs(outdir, exist_ok=True)


def main():  

	DATA_ROOT = "../../raw/lincs/2022-02-LINCS_Level3/data/"
	log_handle = open(osp.join(outdir, 'log_process_data_real_lognorm.txt'), 'w')

	sample_ids_cp, inst_info_cp, inst_info_ctl, gene_info, matrix_cp, matrix_ctl, df_targets = loads_data(DATA_ROOT, log_handle)
	print('loaded data - lognorm', inst_info_cp.shape)

	# inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, keep_cell_lines = filter_cell_lines(inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, log_handle)
	inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, keep_cell_lines = filter_cell_lines_custom(inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, log_handle)
	print('filtered cell lines - lognorm', inst_info_cp.shape)


	inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl = filter_data_metadata(inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, log_handle)
	print('filtered data - lognorm', inst_info_cp.shape)

	inst_info_cp = map_gene_targets_to_lincs(inst_info_cp, gene_info)
	inst_info_cp, matrix_cp = filter_samples_with_unknown_perturbed_genes(inst_info_cp, matrix_cp, gene_info, log_handle)
	print('filtered unknown perturbed genes - lognorm', inst_info_cp.shape)
	
	use_log=True
	normalize_and_save(inst_info_cp, matrix_cp, inst_info_ctl, matrix_ctl, gene_info, keep_cell_lines, log_handle, outdir, use_log)
	print('real values - lognorm')
	log_handle.close()


if __name__ == "__main__":
	startTime=time.time()
	main()
	print(time.time()-startTime)

