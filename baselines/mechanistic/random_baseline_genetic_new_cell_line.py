
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

for selection_type in ['cancer_genes', 'cancer_targets']:
	for cell_line, test_cell_line in zip(['MCF7', 'A549'], ['A549', 'MCF7']):

			#outdir
			outdir = '../experiments/baseline_random_genetic/{}/train_{}_predict_{}'.format(selection_type, cell_line, test_cell_line)
			os.makedirs(outdir, exist_ok=True)
			log = open(osp.join(outdir, 'log.txt'), 'w')

			#Loads dataset
			
			base_path = "../../data/rep-learning-approach-3-all-genes/processed/real_lognorm/"

			path = osp.join(base_path, 'data_backward_{}.pt'.format(cell_line))
			dataset = torch.load(path)


			if selection_type =='cancer_genes':
				path = osp.join(base_path, 'data_forward_{}.pt'.format(cell_line))
				dataset_f = torch.load(path)

				gene_indices_to_choose_from = []
				for d in dataset_f:
					gene_indices_to_choose_from += torch.where(d.mutations)[0].numpy().tolist()

				gene_indices_to_choose_from = list(set(gene_indices_to_choose_from))
			
			elif selection_type == 'cancer_targets':
				#Gene indices to choose from: those which are targets of approved cancer drugs
				drugs_and_targets = pd.read_csv('../../data/random-baseline-cancer-targets/processed/drugs_and_targets.csv', sep='\t')
				gene_names_to_choose_from = []
				for e in drugs_and_targets[drugs_and_targets['cell_line'] == cell_line]['targets'].tolist():
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

				log.write('\n-----\n')

			#Test dataset (different cell line)
			path = osp.join(base_path, 'data_backward_{}.pt'.format(test_cell_line))
			dataset_new_cell_line = torch.load(path)



			#5 rounds
			k_position_of_correct_list = []
			k_position_of_correct_all = []
			real_interventions = {1:[], 2:[], 3:[], 4:[], 5:[]}
			retrieved_interventions = {1:[], 2:[], 3:[], 4:[], 5:[]}
			
			log = open(osp.join(outdir, 'log.txt'), 'w')

			for split_index in range(1,6):
				

				#For each sample in the dataset, predict as intervention a random gene
				k_position_of_correct = []
				
				for i in range(len(dataset_new_cell_line)):
					treated = dataset_new_cell_line[i].treated.numpy()
					correct_intervention = torch.where(dataset_new_cell_line[i].intervention)[0].tolist()
					diseased = dataset_new_cell_line[i].diseased.numpy()
					
					if selection_type == 'random':
						random_ranking_of_genes = [e for e in range(len(diseased))]
						random.shuffle(random_ranking_of_genes)
					
					elif selection_type == 'cancer_genes' or selection_type == 'cancer_targets':
						random.shuffle(gene_indices_to_choose_from)
						remaining_gene_indices = list(set([e for e in range(len(diseased))]) - set(gene_indices_to_choose_from))
						random.shuffle(remaining_gene_indices)
						random_ranking_of_genes = gene_indices_to_choose_from + remaining_gene_indices

					retrieved_intervention_indices = random_ranking_of_genes[0:len(correct_intervention)]
					retrieved_interventions[split_index].append(retrieved_intervention_indices)
					real_interventions[split_index].append(correct_intervention)				

					##Eval
					#topK
					for c in correct_intervention:
						k_position_of_correct.append(np.where(np.array(random_ranking_of_genes) == c)[0].item())
					print('{}/{}'.format(i+1, len(dataset_new_cell_line)))

				k_position_of_correct_list.append(np.mean(k_position_of_correct))
				k_position_of_correct_all += k_position_of_correct

				#Log results
				print('round {}'.format(split_index))
				print('Average topK that the correct intervention is found in: {}\n'.format(np.mean(k_position_of_correct)))

				
				log.write('Round {}\n'.format(split_index))
				log.write('Average topK that the correct intervention is found in: {}\n\n'.format(np.mean(k_position_of_correct)))
				

				#Log results
			print('Average topK that the correct intervention is found in: {}+-{}\n'.format(np.mean(k_position_of_correct_list), np.std(k_position_of_correct_list)))
			log.write('\n-----------------\n')
			log.write('Average topK that the correct intervention is found in: {}+-{}\n'.format(np.mean(k_position_of_correct_list), np.std(k_position_of_correct_list)))
			log.close()
			pd.DataFrame(k_position_of_correct_all).to_csv(osp.join(outdir, 'k_position_of_correct.txt'), index = False, header=None)
			with open(osp.join(outdir, 'retrieved_interventions.pkl'), 'wb') as f:
				pickle.dump(retrieved_interventions, f)
			with open(osp.join(outdir, 'real_interventions.pkl'), 'wb') as f:
				pickle.dump(real_interventions, f)

