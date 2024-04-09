
import torch
import os
import sys

import logging
import scanpy as sc
import scgen
import anndata
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import os.path as osp
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#Model & dataset args
parser.add_argument('--outdir', default=None, type=str, help='Outdir')
parser.add_argument('--train_data_path', default=None, type=str, help='Train data path')
parser.add_argument('--train_cell_type', default=None, type=str, help='Train cell type')
parser.add_argument('--test_cell_type', default=None, type=str, help='Test cell type')
parser.add_argument('--perturbagen', default=None, type=str, help='Perturbagen')
args = parser.parse_args()


def single_pass():
	train = anndata.read_h5ad(args.train_data_path)
	train_cell_type = args.train_cell_type
	test_cell_type = args.test_cell_type
	perturbagen = args.perturbagen
	outdir = args.outdir

	    
	if not ((train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] == perturbagen)).any():
	    pd.DataFrame(['perturbagen not in training', '', '', '']).transpose().to_csv(osp.join(args.outdir, 'results_tmp_{}.csv'.format(perturbagen)), index=False )
	    return

	#Subselects data for training
	subset_test_treated = train[(train.obs["cell_type"] == test_cell_type) & (train.obs["condition"] != 'control')]
	subset_test_control = train[(train.obs["cell_type"] == test_cell_type) & (train.obs["condition"] == 'control')]
	subset_test_control = subset_test_control[np.where(subset_test_treated.obs["condition"] == perturbagen)[0]]

	train_new = train[(  ((train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] == perturbagen)) |
	                     ((train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] == "control")))]

	train_new = anndata.concat([train_new, subset_test_control])


	#Processing data
	scgen.SCGEN.setup_anndata(train_new, batch_key="condition", labels_key="cell_type")

	#Creating and saving model 
	model = scgen.SCGEN(train_new)
	model.save(osp.join(outdir, "lincs_saved_models/model_perturbation_prediction_lincs_{}.pt".format(perturbagen)), overwrite=True)

	#Training the model
	model.train(
	max_epochs=100,
	batch_size=2048,
	early_stopping=True,
	early_stopping_patience=25
	)

	model.save(osp.join(outdir, "lincs_saved_models/model_perturbation_prediction_lincs_{}.pt".format(perturbagen)), overwrite=True)

	#Prediction
	pred, delta = model.predict(
	ctrl_key='control',
	stim_key=perturbagen,
	celltype_to_predict=test_cell_type
	)
	pred.obs['condition'] = 'pred'


	stim = train[((train.obs['cell_type'] == test_cell_type) & (train.obs['condition'] == perturbagen))]

	from scipy import stats
	x = np.asarray(np.mean(pred.X, axis=0)).ravel()
	y = np.asarray(np.mean(stim.X, axis=0)).ravel()
	m, b, r_value, p_value, std_err = stats.linregress(x, y) 


	x = np.asarray(pred.X)
	y = np.asarray(stim.X)
	forward_pearson = []
	for i in range(len(y)):
	    forward_pearson.append(pearsonr(y[i,:], x[i,:])[0])
	forward_pearson = np.mean(forward_pearson)
	forward_r2 = forward_pearson**2

	n_stim_test = stim.X.shape[0]
	n_stim_train = train[((train.obs['cell_type'] == train_cell_type) & (train.obs['condition'] == perturbagen))].shape[0]
	# pd.DataFrame([r_value**2, forward_r2, n_stim_train, n_stim_test]).transpose().to_csv(osp.join(args.outdir, 'results_tmp_{}.csv'.format(perturbagen)), index=False )

	return 

single_pass()