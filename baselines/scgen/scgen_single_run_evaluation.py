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
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#Model & dataset args
parser.add_argument('--outdir', default=None, type=str, help='Outdir')
parser.add_argument('--train_data_path', default=None, type=str, help='Train data path')
parser.add_argument('--test_diseased_path', default=None, type=str, help='Test diseased data path')
parser.add_argument('--train_cell_type', default=None, type=str, help='Train cell type')
parser.add_argument('--test_cell_type', default=None, type=str, help='Test cell type')
parser.add_argument('--perturbagen', default=None, type=str, help='Perturbagen')
parser.add_argument('--eval_setting', default="random", type=str, help='Evaluation setting')
args = parser.parse_args()


def single_pass():
	train = anndata.read_h5ad(args.train_data_path)
	test_diseased = anndata.read_h5ad(args.test_diseased_path)
	train_cell_type = args.train_cell_type
	test_cell_type = args.test_cell_type
	perturbagen = args.perturbagen
	outdir = args.outdir
	eval_setting = args.eval_setting

		#format dataset so I can load the trained model
	subset_test_treated = train[(train.obs["cell_type"] == test_cell_type) & (train.obs["condition"] != 'control')]
	subset_test_control = train[(train.obs["cell_type"] == test_cell_type) & (train.obs["condition"] == 'control')]
	subset_test_control = subset_test_control[np.where(subset_test_treated.obs["condition"] == perturbagen)[0]]

	train_new = train[(  ((train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] == perturbagen)) |
						 ((train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] == "control")))]

	train_new = anndata.concat([train_new, subset_test_control])

	
	scgen.SCGEN.setup_anndata(train_new, batch_key="condition", labels_key="cell_type")
	model = scgen.SCGEN(train_new)
	model = model.load(osp.join(outdir, "lincs_saved_models/model_perturbation_prediction_lincs_{}.pt".format(perturbagen)), adata = train_new)


	pred_all_samples = model.predict(adata_to_predict= test_diseased, ctrl_key='control', stim_key=perturbagen)[0]	
	pred_all_samples.write(osp.join(outdir, 'response_tmp_{}_{}.h5ad'.format(eval_setting, perturbagen)))


	return





single_pass()