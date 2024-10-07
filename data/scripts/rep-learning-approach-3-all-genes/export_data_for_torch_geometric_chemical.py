'''
Exports data for proof of concept model 
Code taken from rep-learning-approach
changed to adapt to new data sources (healthy cell lines + COSMIC)

'''


import pandas as pd
import networkx as nx
import numpy as np
import os
import os.path as osp
import math
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, to_undirected




############
#Data loading
############
def load_ppi(path_edge_list, log_handle):
	#Loads PPI
	ppi = nx.read_edgelist(path_edge_list)
	log_handle.write('----------------\nNumber of nodes in PPI:\t{}\n'.format(ppi.number_of_nodes()))
	log_handle.write('Number of edges in PPI:\t{}\n'.format(ppi.number_of_edges()))
	return ppi


def load_gene_metadata(file, log_handle):
	#Loads gene metadata
	gene_info = pd.read_csv(file)
	# dict_symbol_index = dict(zip(gene_info['gene_symbol'], range(len(gene_info))))	#genes are ordered with the same ordering as rows in data matrices
	dict_entrez_symbol = dict(zip(gene_info['gene_id'], gene_info['gene_symbol']))
	dict_symbol_entrez = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))
	return gene_info, dict_entrez_symbol, dict_symbol_entrez

def load_cosmic(path_cosmic_file, log_handle):
	data = pd.read_csv(path_cosmic_file)
	log_handle.write('Loading COSMIC data. Number of cell lines:\t{}\n'.format(len(set(data['Sample name']))))
	return data


def map_cosmic_to_lincs(cosmic_data, cell_line, gene_info, dict_symbol_entrez, log_handle):
	cosmic_data = cosmic_data[cosmic_data['Sample name']==cell_line]
	log_handle.write('Mapping cosmic genes to lincs. Mapped: {}/{}\n'.format(len(set(cosmic_data['Gene name']).intersection(gene_info['gene_symbol'])), len(set(cosmic_data['Gene name']))))
	#Filter genes not mapped to LINCS
	cosmic_data = cosmic_data[[gene_symbol in dict_symbol_entrez for gene_symbol in cosmic_data['Gene name']]]
	#Save COSMIC mutations as entrez id (dataframe index)
	cosmic_mutations = list(set([dict_symbol_entrez[symbol] for symbol in cosmic_data['Gene name']]))
	return cosmic_mutations



def load_healthy_data(data_root_dir, healthy, log_handle):
	healthy_data_path = osp.join(data_root_dir, 'cell_line_{}_pert_{}.npz'.format(healthy[0], healthy[1]))
	healthy_metadata_path = osp.join(data_root_dir, 'cell_line_{}_pert_{}_metadata.txt'.format(healthy[0], healthy[1]))
	#Loads metadata
	healthy_metadata = pd.read_csv(healthy_metadata_path)
	#Loads data
	with np.load(healthy_data_path, allow_pickle=True) as arr:
		healthy_data =arr['data']
		col_ids = arr['col_ids']
		row_ids = arr['row_ids']
	healthy_data = pd.DataFrame(healthy_data, columns= col_ids, index=row_ids)
	log_handle.write('Loading healthy cell line:\t{} Number of samples:\t{}\n'.format(healthy[0], healthy_data.shape[1]))
	return healthy_data, healthy_metadata





def load_data(cell_line, data_root_dir, log_handle):
	#Loads data matrix (observational)
	file = osp.join(data_root_dir, 'cell_line_{}_pert_ctl_vehicle.npz'.format(cell_line))
	file_metadata = osp.join(data_root_dir, 'cell_line_{}_pert_ctl_vehicle_metadata.txt'.format(cell_line))
	obs_metadata = pd.read_csv(file_metadata)
	with np.load(file, allow_pickle=True) as arr:
		obs_data =arr['data']
		col_ids = arr['col_ids']
		row_ids = arr['row_ids']
	obs_data = pd.DataFrame(obs_data, columns= col_ids, index=row_ids)
	log_handle.write('Number of observational datapoints:\t{}\n'.format(len(obs_metadata)))
	#Loads data matrix (interventional)
	file = osp.join(data_root_dir, 'cell_line_{}_pert_trt_cp.npz'.format(cell_line))
	file_metadata = osp.join(data_root_dir, 'cell_line_{}_pert_trt_cp_metadata.txt'.format(cell_line))
	int_metadata = pd.read_csv(file_metadata)
	with np.load(file, allow_pickle=True) as arr:
		int_data =arr['data']
		col_ids = arr['col_ids']
		row_ids = arr['row_ids']
	int_data = pd.DataFrame(int_data, columns= col_ids, index=row_ids)
	log_handle.write('Number of interventional datapoints:\t{}\n'.format(len(int_metadata)))
	return obs_metadata, obs_data, int_metadata, int_data




############
#Processing
############

def filter_data(healthy_data, healthy_metadata, cosmic_mutations, obs_metadata, obs_data, int_metadata, int_data, ppi, gene_info, log_handle):
	#1.Filter out obs and int data to keep only genes that are in the PPI
	gene_symbols_in_ppi = list(ppi.nodes())
	dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))
	gene_ids_in_ppi = [dict_symbol_id[i] for i in gene_symbols_in_ppi]
	gene_info.index = gene_info['gene_id']; gene_info = gene_info.loc[gene_ids_in_ppi].reset_index(inplace=False, drop=True)
	obs_data = obs_data.loc[gene_ids_in_ppi]
	int_data = int_data.loc[gene_ids_in_ppi]
	if healthy_data is not None:
		healthy_data = healthy_data.loc[gene_ids_in_ppi]
	if cosmic_mutations is not None:
		cosmic_mutations = pd.DataFrame(cosmic_mutations)[[e in gene_ids_in_ppi for e in cosmic_mutations]][0].tolist()
		log_handle.write('Mutations remaining in PPI:\t{}\n'.format(len(cosmic_mutations)))
	#2. Filter out samples whose interventions are not in the remaining genes (those in the PPI)
	keep = []
	for i, gene_symbols in enumerate(int_metadata['target_names']):
		gene_symbols = gene_symbols.replace("[\'", "").replace("\']", "").replace(' ','').replace('\'','').split(',')
		if len(set(gene_symbols).intersection(set(gene_symbols_in_ppi))) > 0:
			keep.append(int_metadata.at[i, 'sample_id'])
	int_metadata.index = int_metadata['sample_id']; int_metadata = int_metadata.loc[keep].reset_index(inplace=False, drop=True)
	int_data = int_data[keep]
	log_handle.write('Number of interventional datapoints after keeping only those with perturbed genes in PPI:\t{}\n'.format(len(int_metadata)))
	return healthy_data, healthy_metadata, cosmic_mutations, obs_metadata, obs_data, int_metadata, int_data, gene_info





############
#Asembling the data
############

def assemble_data_list(healthy_data, healthy_metadata, cosmic_mutations, obs_metadata, obs_data, int_metadata, int_data, ppi, gene_info, log_handle):
	log_handle.write('Assembling data...\n')

	#First, we re-index genes in PPI and data
		#Gene symbol to index to ordered index
	gene_symbol_to_index = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))
	gene_index_to_ordered_index = dict(zip(gene_info['gene_id'], range(len(gene_info))))
	gene_info['ordered_index'] = [gene_index_to_ordered_index[i] for i in gene_info['gene_id']]


		#Reindex genes in PPI, obs_data, int_data, healty_data, and cosmic_mutations
	ppi = nx.relabel_nodes(ppi, gene_symbol_to_index)
	ppi = nx.relabel_nodes(ppi, gene_index_to_ordered_index)
	int_data.index = [gene_index_to_ordered_index[i] for i in int_data.index]
	int_data = int_data.sort_index(inplace=False)
	obs_data.index = [gene_index_to_ordered_index[i] for i in obs_data.index]
	obs_data = obs_data.sort_index(inplace=False)
	if healthy_data is not None:
		healthy_data.index = [gene_index_to_ordered_index[i] for i in healthy_data.index]
		healthy_data = healthy_data.sort_index(inplace=False)
	if cosmic_mutations is not None:
		cosmic_mutations = [gene_index_to_ordered_index[i] for i in cosmic_mutations]
		cosmic_vector = np.zeros(len(healthy_data))
		cosmic_vector[cosmic_mutations] = 1


	#Assembling samples
	edge_index = torch.LongTensor(np.array(ppi.edges()).transpose())
	edge_index = add_remaining_self_loops(edge_index)[0]
	edge_index = to_undirected(edge_index)
	number_of_nodes = ppi.number_of_nodes()

	#Pre-compute distances from each node to each of the nodes mutated and save in a dictionary
	
	G = nx.Graph() 
	G.add_edges_from(tuple(zip(edge_index[0,:].tolist(), edge_index[1,:].tolist())))
	# dict_node_mutation_spl = dict()
	# for node_index in range(G.number_of_nodes()):
	# 	for mutation in cosmic_mutations:
	# 		spl = nx.shortest_path_length(G, node_index, mutation)
	# 		dict_node_mutation_spl[(node_index, mutation)] = spl




	#remove incoming edges to perturbed nodes (mutated nodes)
	# mask = [e not in cosmic_mutations for e in edge_index[1,:]]
	# edge_index_mutilated = edge_index[:, mask]
	# edge_index_mutilated = add_remaining_self_loops(edge_index_mutilated)[0]
	
	dict_forward_sample_and_mutations = dict() #saves the mutation vector used in forward
	forward_data_list = []
	if healthy_data is not None: #Only process forward list if we have healthy data
		#FORWARD DATA - healthy_data, cosmic_vector, obs_data
			#each Data object will be a pairing of a random healthy_data column, the cosmic mutations, and a obs_data column
			#will have as many as obs_data columns
		i = 0
		order = np.array(range(healthy_data.shape[1]))
		np.random.shuffle(order)
		
		for sample_id in obs_data.columns:
			#sample a random healthy GE vector
			i = i % healthy_data.shape[1]
			sample_index = order[i]
			healthy_sample = healthy_data[healthy_data.columns[i]].values
			healthy = torch.Tensor(healthy_sample)
			#mutation
			#randomize mutations. First select the percentage of mutations to include, then select the mutations
		
			perc_to_include = np.random.choice([0.25, 0.50, 0.75, 1], 1).item()
			if int_metadata['cell_mfc_name'][0].split('.')[0] == 'PC3' or int_metadata['cell_mfc_name'][0].split('.')[0] == 'VCAP':
				perc_to_include = 1
			
			cosmic_mutations_i = np.random.choice(cosmic_mutations, int(len(cosmic_mutations)* perc_to_include))
			cosmic_vector = np.zeros(len(healthy_data))
			cosmic_vector[cosmic_mutations_i] = 1
			mutations = torch.Tensor(cosmic_vector)
			#diseased
			diseased = torch.Tensor(obs_data[sample_id])
			#additional features
			mutation_gene_indices = torch.where(mutations)[0].tolist()
			# additional_features = []
			# for node_index in range(len(healthy)):
			# 	spls = [dict_node_mutation_spl[(node_index, e)] for e in mutation_gene_indices]
			# 	additional_features.append(torch.Tensor([np.min(spls), np.max(spls), np.mean(spls)]))
			data = Data(healthy = healthy, mutations=mutations, diseased=diseased, gene_symbols = gene_info['gene_symbol'].tolist())
			data.num_nodes = number_of_nodes
			forward_data_list.append(data)
			#Save 
			i +=1
			dict_forward_sample_and_mutations[sample_id] = mutations

		print('finished data forward')







	#BACKWARD DATA - obs_data, int_data


	#dict sample id: perturbed gene ordered index
	
	dict_sample_id_perturbed_gene_ordered_index = dict()
	for sample_id, gene_symbols in zip(int_metadata['sample_id'], int_metadata['target_names']):
		dict_sample_id_perturbed_gene_ordered_index[sample_id] = []
		for gene in gene_symbols.replace("[\'", "").replace("\']", "").replace(' ','').replace('\'','').split(','):
			if gene in gene_symbol_to_index:
				 dict_sample_id_perturbed_gene_ordered_index[sample_id].append(gene_index_to_ordered_index[gene_symbol_to_index[gene]])
		if len(dict_sample_id_perturbed_gene_ordered_index[sample_id])==0:
			print(sample_id)

	
	#Fill dictionary with remaining spls
	# perturbations = list(dict_sample_id_perturbed_gene_ordered_index.values())
	# perturbations = [item for sublist in perturbations for item in sublist]
	# perturbations = list(set(perturbations))
	# dict_node_mutation_spl = dict()
	# for node_index in range(G.number_of_nodes()):
	# 	for mutation in perturbations:
	# 		if (node_index, mutation) in dict_node_mutation_spl:
	# 			continue
	# 		else:
	# 			spl = nx.shortest_path_length(G, node_index, mutation)
	# 			dict_node_mutation_spl[(node_index, mutation)] = spl


		#these are helpers to sample from obs_data
	order = np.array(range(obs_data.shape[1]))
	np.random.shuffle(order)
	i = 0
	#shuffle obs data columns
	backward_data_list = []
	unique_names_pert = set()
	for sample_id in int_data.columns:
		binary_indicator_perturbation = np.zeros(len(int_data))
		binary_indicator_perturbation[dict_sample_id_perturbed_gene_ordered_index[sample_id]] = 1
		#Get a random pre-intervention sample
		i = i % obs_data.shape[1]
		sample_index = order[i]
		obs_sample_id = obs_data.columns[i]
		obs_sample = obs_data[obs_data.columns[i]].values
		#concat initial node features and perturbation indicator
		diseased = torch.Tensor(obs_sample)
		intervention = torch.Tensor(binary_indicator_perturbation)
		if healthy_data is not None:
			mutations = dict_forward_sample_and_mutations[obs_sample_id]
		else:
			mutations = torch.Tensor(np.zeros(len(diseased)))
		# torch.Tensor(np.stack([obs_sample, binary_indicator_perturbation], 1))
		#post-intervention
		treated = torch.Tensor(int_data[sample_id])
		#remove incoming edges to perturbed node
		# perturbed_node = dict_sample_id_perturbed_gene_ordered_index[sample_id]
		# edge_index_mutilated = edge_index[:, edge_index[1,:] != perturbed_node]
		#additional features
		# additional_features = []
		# for node_index in range(len(diseased)):
		# 	spls = [dict_node_mutation_spl[(node_index, e)] for e in dict_sample_id_perturbed_gene_ordered_index[sample_id]]
		# 	additional_features.append(torch.Tensor([np.min(spls), np.max(spls), np.mean(spls)]))
		drug_name = int_metadata[int_metadata['sample_id'] == sample_id]['cmap_name'].item()
		unique_names_pert.add(drug_name)
		data = Data(perturbagen_name = drug_name, diseased = diseased, intervention=intervention, treated = treated, gene_symbols = gene_info['gene_symbol'].tolist(), mutations = mutations)
		data.num_nodes = number_of_nodes
		backward_data_list.append(data)
		i +=1

	log_handle.write('Samples forward:\t{}\n'.format(len(forward_data_list)))
	log_handle.write('Samples backward:\t{}\n'.format(len(backward_data_list)))
	log_handle.write('Unique perturbagens:\t{}\n'.format(len(unique_names_pert)))
	return forward_data_list, backward_data_list, edge_index




def save_data(forward_data_list, backward_data_list, edge_index, cell_line, log_handle):
	log_handle.write('Saving data {} ...\n\n\n'.format(cell_line))
	torch.save(forward_data_list, osp.join(outdir, 'data_forward_{}.pt'.format(cell_line)))
	torch.save(backward_data_list, osp.join(outdir, 'data_backward_{}.pt'.format(cell_line)))
	torch.save(edge_index, osp.join(outdir, 'edge_index_{}.pt'.format(cell_line)))
	return


binarization = 'real_lognorm'
outdir = '../../processed/torch_data/chemical/{}'.format(binarization)
os.makedirs(outdir, exist_ok=True)




def main():  
	#cell-line wise
	log_handle = open(osp.join(outdir, 'log_export_data.txt'), 'w')
	data_root_dir = '../../processed/lincs/chemical/nofilter_dose_timepoint/{}'.format(binarization)
	data_root_dir_healthy = '../../processed/lincs/chemical/{}'.format(binarization)
	# , ('RWPE1', 'ctl_vector')

	#Cell lines with healthy counterparts
	for cell_line, healthy in zip(['A549', 'MCF7', 'PC3', 'VCAP', 'MDAMB231', 'BT20'], [('NL20', 'ctl_vehicle'), ('MCF10A', 'ctl_vehicle'), ('RWPE1', 'ctl_vector'), ('RWPE1', 'ctl_vector'), ('MCF10A', 'ctl_vehicle'), ('MCF10A', 'ctl_vehicle')]):
		log_handle.write('----------------\n\nCELL LINE:{}\n----------------\n'.format(cell_line))
		#PPI
		ppi = load_ppi('../../processed/ppi/ppi_all_genes_edgelist.txt', log_handle)
		#gene info
		gene_info, dict_entrez_symbol, dict_symbol_entrez = load_gene_metadata('../../processed/lincs/chemical/nofilter_dose_timepoint/gene_info.txt', log_handle)
		#FORWARD DATA
			#healthy GE data
		healthy_data, healthy_metadata = load_healthy_data(data_root_dir_healthy, healthy, log_handle)
			#COSMIC
		cosmic_data = load_cosmic('../../processed/cosmic/CosmicCLP_MutantExport_only_verified_and_curated.csv', log_handle)
		cosmic_mutations = map_cosmic_to_lincs(cosmic_data, cell_line, gene_info, dict_symbol_entrez, log_handle)
		#BACKWARD DATA
			#LINCS
		obs_metadata, obs_data, int_metadata, int_data = load_data(cell_line, data_root_dir, log_handle)
		healthy_data, healthy_metadata, cosmic_mutations, obs_metadata, obs_data, int_metadata, int_data, gene_info = filter_data(healthy_data, healthy_metadata, cosmic_mutations, obs_metadata, obs_data, int_metadata, int_data, ppi, gene_info, log_handle)


		forward_data_list, backward_data_list, edge_index = assemble_data_list(healthy_data, healthy_metadata, cosmic_mutations, obs_metadata, obs_data, int_metadata, int_data, ppi, gene_info, log_handle)
		# save_data(forward_data_list, backward_data_list, edge_index, cell_line, log_handle)


	#Cell lines without healthy counterparts
	for cell_line, healthy in zip(['HA1E', 'HT29', 'A375', 'HELA'], [None, None, None, None]):
		log_handle.write('----------------\n\nCELL LINE:{}\n----------------\n'.format(cell_line))
		#PPI
		ppi = load_ppi('../../processed/ppi/ppi_all_genes_edgelist.txt', log_handle)
		#gene info
		gene_info, dict_entrez_symbol, dict_symbol_entrez = load_gene_metadata('../../processed/lincs/chemical/nofilter_dose_timepoint/gene_info.txt', log_handle)
		#FORWARD DATA
			#healthy GE data
		# healthy_data, healthy_metadata = load_healthy_data(data_root_dir_healthy, healthy, log_handle)
			#COSMIC
		cosmic_data = load_cosmic('../../processed/cosmic/CosmicCLP_MutantExport_only_verified_and_curated.csv', log_handle)
		cosmic_mutations = map_cosmic_to_lincs(cosmic_data, cell_line, gene_info, dict_symbol_entrez, log_handle)
		#BACKWARD DATA
			#LINCS
		obs_metadata, obs_data, int_metadata, int_data = load_data(cell_line, data_root_dir, log_handle)
		healthy_data, healthy_metadata, cosmic_mutations, obs_metadata, obs_data, int_metadata, int_data, gene_info = filter_data(None, None, None, obs_metadata, obs_data, int_metadata, int_data, ppi, gene_info, log_handle)


		forward_data_list, backward_data_list, edge_index = assemble_data_list(None, None, None, obs_metadata, obs_data, int_metadata, int_data, ppi, gene_info, log_handle)
		# save_data(forward_data_list, backward_data_list, edge_index, cell_line, log_handle)

	log_handle.close()







if __name__ == "__main__":
    main()