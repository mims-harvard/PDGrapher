
'''
Random baseline: For each diseased-treated pair, 
picks any N random genes as the intervention (the same number of genes as the correct intervention)
It can be
--totally random
--cancer genes at the top
--cancer targets at the top


'''


import torch
import os.path as osp
import pandas as pd
import numpy as np
from scipy.spatial import distance
import random
import os
import pickle

cell_lines = {'chemical': ["A549", "A375", "BT20", "HELA", "HT29", "MCF7", "MDAMB231", "PC3", "VCAP"],
				'genetic': ["A549", "A375", "AGS", "BICR6", "ES2", "HT29", "MCF7", "PC3", "U251MG", "YAPC"]
	
}

splits_type = 'random'

def compute_idcg(num_correct, num_nodes):
    """
    Computes the Ideal Discounted Cumulative Gain (IDCG) for the given
    number of correct interventions and total number of nodes.

    Args:
        num_correct (int): Number of correct interventions.
        num_nodes (int): Total number of nodes.

    Returns:
        float: The IDCG value.
    """
    idcg = 0
    for rank in range(1, num_correct + 1):  # Ideal ranking: 1 to num_correct
        gain = 1 - (rank / num_nodes)  # Gain function
        discount = 1 / np.log2(rank + 1)  # Logarithmic discount
        idcg += gain * discount
    return idcg

# for dataset_type in ['genetic', 'chemical']:
for dataset_type in ['chemical']:
    # for selection_type in ['random', 'cancer_genes', 'cancer_targets', 'perturbed_genes']:
	for selection_type in ['cancer_genes', 'cancer_targets', 'perturbed_genes']:
		for test_cell_line in cell_lines[dataset_type]:
			for train_cell_line in cell_lines[dataset_type]:
				if test_cell_line == train_cell_line:
					continue
				
				
				train_cell_lines = [train_cell_line]
	
				#outdir
				outdir = './results/mechanistic/baseline_random_{}/new_cell_line/{}/{}_{}/{}'.format(dataset_type, selection_type, train_cell_line, test_cell_line, splits_type)
				os.makedirs(outdir, exist_ok=True)
				log = open(osp.join(outdir, 'log.txt'), 'w')

				#Loads dataset
				
				if dataset_type == 'chemical':
					base_path = "../../data/processed/torch_data/chemical/real_lognorm/"
				else:
					base_path = "../../data/processed/torch_data/real_lognorm/"

				path = osp.join(base_path, 'data_backward_{}.pt'.format(test_cell_line))
				dataset = torch.load(path)




				if selection_type =='cancer_genes':	#loads genes from all other cell lines
					gene_indices_to_choose_from = []
			
					for train_cell in train_cell_lines:
						path = osp.join(base_path, 'data_forward_{}.pt'.format(train_cell))
						dataset_f = torch.load(path)
						
						for d in dataset_f:
							gene_indices_to_choose_from += torch.where(d.mutations)[0].numpy().tolist()

					gene_indices_to_choose_from = list(set(gene_indices_to_choose_from))
				
    
    
				elif selection_type == 'cancer_targets':
					#Gene indices to choose from: those which are targets of approved cancer drugs
					drugs_and_targets = pd.read_csv('../../data/processed/nci/drugs_and_targets.csv', sep='\t')
					gene_names_to_choose_from = []
					
					for train_cell in train_cell_lines:
						for e in drugs_and_targets[drugs_and_targets['cell_line'] == train_cell]['targets'].tolist():
							gene_names_to_choose_from += e.split(',') 

					gene_names_to_choose_from = list(set(gene_names_to_choose_from))
					log.write('Number of cancer target names:\t{}\n'.format(len(gene_names_to_choose_from)))

					dict_name_index = dict(zip(dataset[0].gene_symbols, range(len(dataset[0].gene_symbols))))
					gene_indices_to_choose_from = []
					for gene in gene_names_to_choose_from:
						if gene in dict_name_index:
							gene_indices_to_choose_from.append(dict_name_index[gene])

					log.write('Mapped names to PPI:\t{}/{}\n'.format(len(gene_indices_to_choose_from), len(gene_names_to_choose_from)))

					
					log.write('Cancer targets:\n')
					for gene in gene_names_to_choose_from:
						if gene in dict_name_index:
							log.write('{}\n'.format(gene))

				elif selection_type=='perturbed_genes': #set of genes perturbed in train cell line
					gene_indices_to_choose_from = []
			
					for train_cell in train_cell_lines:
						path = osp.join(base_path, 'data_backward_{}.pt'.format(train_cell))
						dataset_b = torch.load(path)
						
						for d in dataset_b:
							gene_indices_to_choose_from += torch.where(d.intervention)[0].numpy().tolist()

					gene_indices_to_choose_from = list(set(gene_indices_to_choose_from))

					log.write('\n-----\n')


				#Test dataset (different cell line)
				path = osp.join(base_path, 'data_backward_{}.pt'.format(test_cell_line))
				dataset_new_cell_line = torch.load(path)
				splits_file	= f"../../data/processed/splits/{dataset_type}/{test_cell_line.split('_')[0]}/random/5fold/splits.pt"
				splits = torch.load(splits_file)



				#5 rounds
				all_recall_at_1 = {'test':[]}
				all_recall_at_10 = {'test':[]}
				all_recall_at_100 = {'test':[]}
				all_recall_at_1000 = {'test':[]}
				all_rankings = {'test':[]}
				all_rankings_dcg = {'test':[]}
				all_perc_partially_accurate_predictions = {'test':[]}
				real_interventions = {1:[], 2:[], 3:[], 4:[], 5:[]}
				retrieved_interventions = {1:[], 2:[], 3:[], 4:[], 5:[]}
				
				log = open(osp.join(outdir, 'log.txt'), 'w')

				for split_index in range(1,6):
					

					#For each sample in the dataset, predict as intervention a random gene
					recall_at_1 = []
					recall_at_10 = []
					recall_at_100 = []
					recall_at_1000 = []
					ranking = []
					rankings_dcg = []
					n_non_zeros = 0
					
					
					split = splits[split_index]
					test_samples_indices = split['test_index_backward'].tolist()

					for i in test_samples_indices:
						treated = dataset_new_cell_line[i].treated.numpy()
						correct_intervention = torch.where(dataset_new_cell_line[i].intervention)[0].tolist()
						diseased = dataset_new_cell_line[i].diseased.numpy()
						
						if selection_type == 'random':
							random_ranking_of_genes = [e for e in range(len(diseased))]
							random.shuffle(random_ranking_of_genes)
						
						elif selection_type == 'cancer_genes' or selection_type == 'cancer_targets' or selection_type == 'perturbed_genes':
							random.shuffle(gene_indices_to_choose_from)
							remaining_gene_indices = list(set([e for e in range(len(diseased))]) - set(gene_indices_to_choose_from))
							random.shuffle(remaining_gene_indices)
							random_ranking_of_genes = gene_indices_to_choose_from + remaining_gene_indices

						retrieved_intervention_indices = random_ranking_of_genes[0:len(correct_intervention)]
						retrieved_interventions[split_index].append(retrieved_intervention_indices)
						real_interventions[split_index].append(correct_intervention)				

						#Records recall@K
						recall_at_1.append(len(set(random_ranking_of_genes[:1]).intersection(correct_intervention)) / len(correct_intervention))
						recall_at_10.append(len(set(random_ranking_of_genes[:10]).intersection(correct_intervention)) / len(correct_intervention))
						recall_at_100.append(len(set(random_ranking_of_genes[:100]).intersection(correct_intervention)) / len(correct_intervention))
						recall_at_1000.append(len(set(random_ranking_of_genes[:1000]).intersection(correct_intervention)) / len(correct_intervention))
		
						#Records ranking
						num_nodes = len(random_ranking_of_genes)
						for ci in list(correct_intervention):
							ranking.append(1 - (random_ranking_of_genes.index(ci) / num_nodes))
		
					#Ranking metric - DCG-style						


						dcg = 0
						for ci in list(correct_intervention):
							# Get the rank of the current ground-truth intervention
							rank = random_ranking_of_genes.index(ci) + 1 #1-based indexing for CDG
							gain = 1 - (rank / num_nodes)
							discount = 1 / np.log2(rank + 1)
							dcg += gain * discount
							
						#normalize
						idcg = compute_idcg(len(correct_intervention), num_nodes)
						ndcg = dcg / idcg if idcg > 0 else 0
						rankings_dcg.append(ndcg)

						#Records number of partially accurate predictions
						overlap = len(set(correct_intervention).intersection(random_ranking_of_genes[:len(correct_intervention)]))

						if overlap != 0:
							n_non_zeros += 1


					all_recall_at_1['test'].append(np.mean(recall_at_1))
					all_recall_at_10['test'].append(np.mean(recall_at_10))
					all_recall_at_100['test'].append(np.mean(recall_at_100))
					all_recall_at_1000['test'].append(np.mean(recall_at_1000))
					all_rankings['test'].append(np.mean(ranking))
					all_rankings_dcg['test'].append(np.mean(rankings_dcg))
					all_perc_partially_accurate_predictions['test'].append(100 * n_non_zeros/len(dataset_new_cell_line))

					#Log results
				log.write('\n\n----------------------\n')
				log.write('\n\nTEST SET\n')
				log.write('recall@1: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_1['test']), np.std(all_recall_at_1['test'])))
				log.write('recall@10: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_10['test']), np.std(all_recall_at_10['test'])))
				log.write('recall@100: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_100['test']), np.std(all_recall_at_100['test'])))
				log.write('recall@1000: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_1000['test']), np.std(all_recall_at_1000['test'])))
				log.write('percentage of samples with partially accurate predictions: {:.2f}±{:.2f}\n'.format(np.mean(all_perc_partially_accurate_predictions['test']), np.std(all_perc_partially_accurate_predictions['test'])))
				log.write('ranking score: {:.2f}±{:.2f}\n'.format(np.mean(all_rankings['test']), np.std(all_rankings['test'])))
				log.write('ranking score - DCG: {:.2f}±{:.2f}\n'.format(np.mean(all_rankings_dcg['test']), np.std(all_rankings_dcg['test'])))
    
				log.write('--------------------------\n')
				log.write('All metric datapoints:\n')
				log.write('recall@1: {}\n'.format(all_recall_at_1['test']))
				log.write('recall@10: {}\n'.format(all_recall_at_10['test']))
				log.write('recall@100: {}\n'.format(all_recall_at_100['test']))
				log.write('recall@1000: {}\n'.format(all_recall_at_1000['test']))
				log.write('percentage of samples with partially accurate predictions: {}\n'.format(all_perc_partially_accurate_predictions['test']))
				log.write('ranking score: {}\n'.format(all_rankings['test']))
				log.write('ranking score - DCG: {}\n'.format(all_rankings_dcg['test']))

	
				log.close()
    
    
				with open(osp.join(outdir, 'retrieved_interventions.pkl'), 'wb') as f:
					pickle.dump(retrieved_interventions, f)
				with open(osp.join(outdir, 'real_interventions.pkl'), 'wb') as f:
					pickle.dump(real_interventions, f)

