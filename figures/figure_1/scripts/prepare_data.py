



import torch
import pandas as pd
import sys
import numpy as np
import os
import os.path as osp

import networkx as nx
import matplotlib.pyplot as plt
import random
from torch_geometric.utils import remove_self_loops
import pickle


##Prepare vectors with top K predicted and GT targets of all approved drugs (not in training)
cell_lines = ['A549_corrected_pos_emb', 'BT20_corrected_pos_emb', 'MCF7_corrected_pos_emb']

for cell_line in cell_lines:
	outdir = '../processed/{}'.format(cell_line)
	os.makedirs(outdir, exist_ok=True)


	#################################################
	#Loads data

	#Reads in approved drugs
	drugs = pd.read_csv('../../../data/processed/nci/drugs_and_targets.csv', sep='\t')

	#Selects approved drugs for cell line use case
	drugs_approved = drugs[drugs['cell_line'] == cell_line.split('_')[0]]['drug'].tolist()

	#Loads torch datasets
	forward_dataset = torch.load('../../../data/processed/torch_data/chemical/real_lognorm/data_forward_{}.pt'.format(cell_line.split('_')[0]))
	backward_dataset = torch.load('../../../data/processed/torch_data/chemical/real_lognorm/data_backward_{}.pt'.format(cell_line.split('_')[0]))
	splits = torch.load('../../../data/processed/splits/chemical/{}/random/5fold/splits.pt'.format(cell_line.split('_')[0]))
	gene_symbols = forward_dataset[0].gene_symbols
	edge_index = torch.load('../../../data/processed/torch_data/chemical/real_lognorm/edge_index_{}.pt'.format(cell_line.split('_')[0]))
	edge_index = remove_self_loops(edge_index)[0]
		#Saves edge index
	dict_node_index_to_symbol = dict(zip(range(len(gene_symbols)), gene_symbols))
	new_edges = []
	for edge in edge_index.transpose(1,0):
		new_edges.append((dict_node_index_to_symbol[edge[0].item()], dict_node_index_to_symbol[edge[1].item()]))
		

	edge_index_df = pd.DataFrame(new_edges)
	edge_index_df[2] = 1
	edge_index_df.to_csv(osp.join(outdir, 'edge_index_{}.txt'.format(cell_line.split('_')[0])), header=None, index=False, sep='\t')





	#################################################
	#Loads random and PDGrapher predicted gene list

	####Aggregated ranking
	with open('../../figure_5/{}/aggregated_ranking.pickle'.format(cell_line), 'rb') as f:
		aggregated_ranking = pickle.load(f)

	ranked_list_of_genes = [gene_symbols[e] for e in aggregated_ranking]

	with open('../../figure_5/{}/aggregated_ranking_random.pickle'.format(cell_line), 'rb') as f:
		random_aggregated_ranking = pickle.load(f)

	random_ranked_list_of_genes = [gene_symbols[e] for e in random_aggregated_ranking]






	#################################################
	#Gets info on approved drugs

	#Gets drugs in training, and drugs approved not in training (these are the ones we need to check if we recover)
	#Gets unique drugs in dataset
	perturbagens_in_dataset = []
	for d in backward_dataset:
		perturbagens_in_dataset.append(d.perturbagen_name)
		

	perturbagens_in_dataset = list(set(perturbagens_in_dataset))


	#Overlap between approved drugs and perturbagens in dataset
	drugs_in_both = set(drugs_approved).intersection(perturbagens_in_dataset)


	#Retrieve drugs only approved but not in dataset
	approved_drugs_not_in_dataset = list(set(drugs_approved) - set(perturbagens_in_dataset))


	#Genes of approved drugs (do we recover them?)
	genes_in_approved_drugs = []
	genes_in_approved_drugs_dict = {}
	for d in approved_drugs_not_in_dataset:
		for e in drugs[drugs['drug']==d]['targets']:
			genes_in_approved_drugs += e.split(',')
			if d not in genes_in_approved_drugs_dict:
				genes_in_approved_drugs_dict[d] = []
			genes_in_approved_drugs_dict[d] += e.split(',')

	for drug in genes_in_approved_drugs_dict:
		genes_in_approved_drugs_dict[drug] = list(set(genes_in_approved_drugs_dict[drug]))


	genes_in_approved_drugs = set(genes_in_approved_drugs)	
	genes_in_approved_drugs = list(genes_in_approved_drugs.intersection(gene_symbols))	#taking only those which overlap with the PPI

	# Filter out genes that are not in ranked_list_of_genes
	genes_in_approved_drugs_dict = {
		drug: [gene for gene in target_genes if gene in ranked_list_of_genes]
		for drug, target_genes in genes_in_approved_drugs_dict.items()
	}

	genes_in_approved_drugs_dict = {k: v for k, v in genes_in_approved_drugs_dict.items() if len(v) > 0}




	#################################################
	#Let's build the dataframes we need to plot SAFE

	pd.DataFrame(gene_symbols).to_csv(osp.join(outdir, 'gene_symbols.txt'), index = False)

	ks = [10, 20, 50, 100, 500, 1000]
	for k in ks:
		vector_to_save = np.zeros(len(aggregated_ranking)).astype(int)
		vector_to_save[aggregated_ranking[0:k]] = 1
		vector_to_save_real = np.zeros(len(aggregated_ranking)).astype(int)
		indices_approved_drugs = [gene_symbols.index(item) for item in genes_in_approved_drugs]
		vector_to_save_real[indices_approved_drugs] = 1
		vector_to_save_random = np.zeros(len(aggregated_ranking)).astype(int)
		vector_to_save_random[random_aggregated_ranking[0:k]] = 1
		vector_to_save = pd.DataFrame([gene_symbols, vector_to_save, vector_to_save_real, vector_to_save_random]).transpose()
		vector_to_save.columns = ['Gene', 'Top {} predicted targets'.format(k), 'Targets of approved drugs', 'Top {} random targets'.format(k)]
		vector_to_save.to_csv(osp.join(outdir, '{}_top{}.txt'.format(cell_line, k)), index=False, sep='\t')
















