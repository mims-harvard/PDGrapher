import pickle as pkl
import torch
import anndata
import os.path as osp
import os
import numpy as np
from scipy import stats
import pandas as pd
import torch_geometric

cell_selector = {'MCF7': 'A549',
				'A549': 'MCF7'}


all_results = []

for dataset_type in ['chemical', 'genetic']:
	for folder_cell_line in ['A549', 'MCF7']:
		for setting in ['random', 'new_cell_line']:

			if setting == 'random':
				test_cell_line = cell_selector[folder_cell_line]
			else:
				test_cell_line = folder_cell_line

			train_cell_line = cell_selector[folder_cell_line]

			if dataset_type == 'chemical':
				base_path = '../data/rep-learning-approach-3-all-genes/processed/chemical/real_lognorm'
				train = anndata.read_h5ad(osp.join(base_path, 'data_scgen.h5ad'))
				backward_dataset = torch.load(osp.join(base_path, 'data_backward_{}.pt'.format(test_cell_line)))
			else:
				base_path = '../data/rep-learning-approach-3-all-genes/processed/real_lognorm'
				train = anndata.read_h5ad(osp.join(base_path, 'data_scgen.h5ad'))
				backward_dataset = torch.load(osp.join(base_path, 'data_backward_{}.pt'.format(test_cell_line)))

			#perturbagen - targets dictionary
			dict_pert_targets = dict()
			for data in backward_dataset:
				dict_pert_targets[data.perturbagen_name] = set(torch.where(data.intervention)[0].tolist())

			results_r2 = []
			results_scgen_r2 = []
			top_1_perf = []
			acc = []
			acc_all = []
			n_zeros = []

			for split_index in range(1,6):
				highest_r2 = 'results_lincs/{}/{}/split_{}/int_disc_highest_r2_predicted_response_vs_real_response_{}.pickle'.format(dataset_type,folder_cell_line,split_index, setting)
				selected_pert = 'results_lincs/{}/{}/split_{}/int_disc_selected_predicted_perturbagen_{}.pickle'.format(dataset_type,folder_cell_line, split_index, setting)
				selected_pred_resp = 'results_lincs/{}/{}/split_{}/int_disc_selected_predicted_response_{}.pickle'.format(dataset_type,folder_cell_line, split_index, setting)


				with open(highest_r2, 'rb') as f: highest_r2 = pkl.load(f)
				with open(selected_pert, 'rb') as f: selected_pert = pkl.load(f)
				with open(selected_pred_resp, 'rb') as f: selected_pred_resp = pkl.load(f)

				splits = torch.load('../baselines/source/splits/{}/{}/random/5fold/splits.pt'.format(dataset_type, test_cell_line))
				test_indices = splits[split_index]['test_index_backward'].tolist()
				if setting == 'random':
					test_treated = train[(train.obs["cell_type"] == test_cell_line) & (train.obs["condition"] != 'control')][test_indices]
				else:
					test_treated = train[(train.obs["cell_type"] == test_cell_line) & (train.obs["condition"] != 'control')]


				ordered_perturbagens = np.array(test_treated.obs.condition.tolist())
				unique_perturbagens = test_treated.obs.condition.unique().tolist()



				scgen_r2s = []
				for perturbagen in unique_perturbagens:
					idxs = np.where(ordered_perturbagens == perturbagen)[0]
					score_ys = selected_pred_resp[idxs]
					real_ys = test_treated.X[idxs]
					x = np.mean(score_ys, 0 ).ravel()
					y = np.mean(real_ys, 0 ).ravel()
					m, b, forward_r_value, p_value, std_err = stats.linregress(x, y) 
					scgen_r2s.append(forward_r_value**2)


				results_r2.append(np.mean(highest_r2))
				results_scgen_r2.append(np.mean(scgen_r2s))

				top_1_perf.append(sum(selected_pert == ordered_perturbagens) / len(ordered_perturbagens))
				acc_i = []
				acc_i_all = []
				n_zeros_i = 0

				for i in range(len(selected_pert)):
					jaccards = len(dict_pert_targets[selected_pert[i]].intersection(dict_pert_targets[ordered_perturbagens[i]])) / len(dict_pert_targets[selected_pert[i]].union(dict_pert_targets[ordered_perturbagens[i]]))
					if jaccards ==0:
						n_zeros_i += 1
					else:
						acc_i.append(jaccards)
					acc_i_all.append(jaccards)
					
				acc.append(np.mean(acc_i))
				acc_all.append(np.mean(acc_i_all))
				n_zeros.append(n_zeros_i/len(selected_pert))


			all_results.append([dataset_type, 
								test_cell_line, 
								setting, 
								'{:.2f} ± {:.2f}'.format(np.mean(results_r2), np.std(results_r2)), 
								'{:.2f} ± {:.2f}'.format(np.mean(results_scgen_r2), np.std(results_scgen_r2)), 
								'{:.2f} ± {:.2f}'.format(np.mean(top_1_perf), np.std(top_1_perf)), 
								'{:.8f} ± {:.8f}'.format(np.mean(acc), np.std(acc)), 
								'{:.8f} ± {:.8f}'.format(np.mean(acc_all), np.std(acc_all)), 
								'{:.2f} ± {:.2f}'.format(np.mean(n_zeros), np.std(n_zeros)), 
								1 - np.array(n_zeros)    ])

pd.DataFrame(all_results, columns = ['dataset_type', 'cell_line', 'setting', 'R2', 'scgenR2', 'top1_perf', 'jaccard int (only nonzero)', 'jaccard int (all)', 'n zeros', 'all PD accuracy']).to_csv('3_int_discovery_performance.csv')















