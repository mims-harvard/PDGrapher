

import pandas as pd
import torch
torch.set_num_threads(1)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import os.path as osp
import argparse
from glob import glob

import logging
import scanpy as sc
import scgen
import anndata
import numpy as np
import pickle
from scipy.stats import pearsonr
import subprocess
import warnings
from threadpoolctl import threadpool_limits
warnings.filterwarnings("ignore")

def np_pearson_cor(x, y):
	xv = x - x.mean(axis=0)
	yv = y - y.mean(axis=0)
	xvss = (xv * xv).sum(axis=0)
	yvss = (yv * yv).sum(axis=0)
	with threadpool_limits(limits=4, user_api='blas'):
		result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
	# bound the values to -1 to 1 in the event of precision issues
	return np.maximum(np.minimum(result, 1.0), -1.0)

def torch_pearson_cor(x, y, device):
	n_batches = 2
	n_samples_step = round(x.shape[1]/n_batches) + 1
	results = []
	for i in range(n_batches):
		x_i = torch.Tensor(x[:, i*n_samples_step:(i+1)*n_samples_step]).to(device)
		y_i = torch.Tensor(y[:, i*n_samples_step:(i+1)*n_samples_step]).to(device)
		xv = x_i - x_i.mean(axis=0)
		yv = y_i - y_i.mean(axis=0)
		xvss = (xv * xv).sum(axis=0)
		yvss = (yv * yv).sum(axis=0)
		result = torch.matmul(xv.transpose(1,0), yv) / torch.sqrt(torch.outer(xvss, yvss))
		results.append(np.diag(result.detach().cpu().numpy()))
	results = np.concatenate(results)
	# bound the values to -1 to 1 in the event of precision issues
	return np.maximum(np.minimum(results, 1.0), -1.0)**2


# @profile
def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	#Model & dataset args
	parser.add_argument('--dataset_type', default='chemical', type=str, help='genetic/chemical')
	parser.add_argument('--train_cell_type', default='A549', type=str, help='Train cell type')
	parser.add_argument('--gpu_index', default="2", type=str, help='GPU index')
	parser.add_argument('--eval_setting', default="random", type=str, help='Evaluation setting')
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

	test_cell_type_selector = {'A549': 'MCF7',
								'MCF7': 'A549'}
		
	device = torch.device('cuda')

	if args.dataset_type == 'chemical':
		base_path = '../data/rep-learning-approach-3-all-genes/processed/chemical/real_lognorm'
		train = anndata.read_h5ad(osp.join(base_path, 'data_scgen.h5ad'))
	else:
		base_path = '../data/rep-learning-approach-3-all-genes/processed/real_lognorm'
		train = anndata.read_h5ad(osp.join(base_path, 'data_scgen.h5ad'))

	train_cell_type = args.train_cell_type 
	test_cell_type = test_cell_type_selector[args.train_cell_type]


	splits = torch.load('../baselines/source/splits/{}/{}/random/5fold/splits.pt'.format(args.dataset_type, train_cell_type))



	for split_index in splits.keys():
		indir = 'results_lincs/{}/{}/split_{}'.format(args.dataset_type, test_cell_type, split_index)

				#Get subset of data to predict for (TRAIN cell line because setting = random)
		test_indices = splits[split_index]['test_index_backward'].tolist()
		test_diseased = train[(train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] == 'control')][test_indices]
		test_treated = train[(train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] != 'control')][test_indices]


		#Data for scgen model
		train_indices = splits[split_index]['train_index_backward'].tolist() +  splits[split_index]['val_index_backward'].tolist()
		train_subset_perturbed = train[(train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] != 'control')][train_indices]
		train_subset_control = train[(train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] == 'control')][train_indices]
		test = train[(train.obs["cell_type"] == test_cell_type)]
		train_subset = anndata.concat([train_subset_perturbed, train_subset_control, test], join='inner')

		train_data_path = osp.join(indir, 'train_random_tmp.h5ad')
		train_subset.write(train_data_path)
		test_diseased_path = osp.join(indir, 'test_diseased_random_tmp.h5ad')
		test_diseased.write(test_diseased_path)




		#Compute response for each perturbagen in train set
		perturbagens_in_train = train_subset_perturbed.obs['condition'].unique().tolist()
		r2_predicted_response_vs_real_response = pd.DataFrame(np.empty([len(test_diseased), len(perturbagens_in_train)]), columns = perturbagens_in_train)
		vectors_predicted_response = pd.DataFrame(np.empty([len(test_diseased), len(perturbagens_in_train)]), columns = perturbagens_in_train)


		highest_r2_predicted_response_vs_real_response = np.zeros(len(test_diseased))
		vectors_predicted_response = np.empty([len(test_diseased), test_diseased.X.shape[1]])
		vectors_predicted_perturbagen = [-1 for i in  range(len(test_diseased))]



		jj = 1
		
		perturbagens_in_test = test.obs['condition'].unique().tolist()
		perturbagens = list(set(perturbagens_in_train).intersection(set(perturbagens_in_test)))
		perturbagens.sort()
		log = open(osp.join(indir, 'log_2_compute_r2_predicted_response_random.txt'), 'w')	

		for perturbagen in perturbagens:
			

			_ = subprocess.run(['python', 'scgen_single_run_evaluation.py',
                                '--outdir', indir,
                                '--train_data_path', train_data_path,
                                '--train_cell_type', train_cell_type,
                                '--test_cell_type', test_cell_type,
                                '--perturbagen', perturbagen,
                                '--test_diseased_path', test_diseased_path,
                                '--eval_setting', args.eval_setting])


			
			pred_all_samples = anndata.read_h5ad(osp.join(indir, 'response_tmp_{}_{}.h5ad'.format(args.eval_setting, perturbagen))).X

			corrs = np_pearson_cor(pred_all_samples.transpose(), test_treated.X.transpose())
			corrs = np.diag(corrs)**2


			# corrs = torch_pearson_cor(pred_all_samples.transpose(), test_treated.X.transpose(), device)


			#Updated predicted response 
			vectors_predicted_response = np.where(np.repeat(corrs.reshape(-1,1), pred_all_samples.shape[1], 1) > np.repeat(highest_r2_predicted_response_vs_real_response.reshape(-1,1), pred_all_samples.shape[1], 1), pred_all_samples, vectors_predicted_response)
			#Update predicted perturbagens
			vectors_predicted_perturbagen = np.where(corrs > highest_r2_predicted_response_vs_real_response, perturbagen, vectors_predicted_perturbagen)
			#Update R2
			highest_r2_predicted_response_vs_real_response = np.where(corrs > highest_r2_predicted_response_vs_real_response, corrs, highest_r2_predicted_response_vs_real_response)
			pred_all_samples = corrs

			print('{}/{}'.format(jj, len(perturbagens_in_train)))
			jj += 1
			log.write('{}\n'.format(perturbagen))
			log.flush()
			os.fsync(log.fileno())		

			_ = subprocess.run(['rm', '-rf',
                    osp.join(indir,'response_tmp_{}_{}.h5ad'.format(args.eval_setting, perturbagen))])	

		log.close()
		with open(osp.join(indir, 'int_disc_selected_predicted_response_random.pickle'), 'wb') as f:
			pickle.dump(vectors_predicted_response, f)
		
		with open(osp.join(indir, 'int_disc_selected_predicted_perturbagen_random.pickle'), 'wb') as f:
			pickle.dump(vectors_predicted_perturbagen, f)

		with open(osp.join(indir, 'int_disc_highest_r2_predicted_response_vs_real_response_random.pickle'), 'wb') as f:
			pickle.dump(highest_r2_predicted_response_vs_real_response, f)

		_ = subprocess.run(['rm', '-rf',
                    osp.join(indir,'*tmp*')])
		
		

		

















main()