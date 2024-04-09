#!/usr/bin/env python
# coding: utf-8

# # SCGEN:  Perturbation Prediction

# In[ ]:


import torch
torch.set_num_threads(1)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import logging
import scanpy as sc
import scgen
import anndata
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import subprocess

# ### Loading Train Data
import torch
import os.path as osp
import argparse
from glob import glob



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#Model & dataset args
parser.add_argument('--dataset_type', default='genetic', type=str, help='genetic/chemical')
parser.add_argument('--train_cell_type', default='A549', type=str, help='Train cell type')
parser.add_argument('--gpu_index', default="1", type=str, help='GPU index')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

test_cell_type_selector = {'A549': 'MCF7',
                            'MCF7': 'A549'}
    
    

if args.dataset_type == 'chemical':
    base_path = '../data/rep-learning-approach-3-all-genes/processed/chemical/real_lognorm'
    train = anndata.read_h5ad(osp.join(base_path, 'data_scgen.h5ad'))
else:
    base_path = '../data/rep-learning-approach-3-all-genes/processed/real_lognorm'
    train = anndata.read_h5ad(osp.join(base_path, 'data_scgen.h5ad'))


print('Cell types', set(train.obs.cell_type.values.tolist()))
print('Conditions', len(set(train.obs.condition.values.tolist())))
print('Diseased A549:', sum((train.obs["cell_type"] == 'A549') & (train.obs["condition"] == 'control')))
print('Treated A549:', sum((train.obs["cell_type"] == 'A549') & (train.obs["condition"] != 'control')))
print('Diseased MCF7:', sum((train.obs["cell_type"] == 'MCF7') & (train.obs["condition"] == 'control')))
print('Treated MCF7:', sum((train.obs["cell_type"] == 'MCF7') & (train.obs["condition"] != 'control')))


def train_model():
    train_cell_type = args.train_cell_type 
    test_cell_type = test_cell_type_selector[args.train_cell_type]


    splits = torch.load('../baselines/source/splits/{}/{}/random/5fold/splits.pt'.format(args.dataset_type, train_cell_type))

    for split_index in splits.keys():
        outdir = 'results_lincs/{}/{}/split_{}'.format(args.dataset_type, test_cell_type, split_index)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(osp.join(outdir, "lincs_saved_models"), exist_ok=True)
        
        #Get subset of training data
        train_indices = splits[split_index]['train_index_backward'].tolist() +  splits[split_index]['val_index_backward'].tolist()
        train_subset_perturbed = train[(train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] != 'control')][train_indices]
        train_subset_control = train[(train.obs["cell_type"] == train_cell_type) & (train.obs["condition"] == 'control')][train_indices]
        test = train[(train.obs["cell_type"] == test_cell_type)]

        train_subset = anndata.concat([train_subset_perturbed, train_subset_control, test], join='inner')
        train_data_path = osp.join(outdir, 'train_subset_tmp.h5ad')
        train_subset.write(train_data_path)


        perturbagens_test = list(set(train_subset[( ((train_subset.obs["cell_type"] == test_cell_type) & (train_subset.obs["condition"] != "control")))].copy().obs['condition'].tolist()))


        for perturbagen in perturbagens_test:
            from time import time
            tic = time()
            _ = subprocess.run(['python', 'scgen_single_run.py',
                                '--outdir', outdir,
                                '--train_data_path', train_data_path,
                                '--train_cell_type', train_cell_type,
                                '--test_cell_type', test_cell_type,
                                '--perturbagen', perturbagen])
            print('1 model trained: {}secs'.format(time()-tic))
            return


        rows = []
        perturbagens = []
        for path in glob(osp.join(outdir, 'results_tmp_*')):
            perturbagens.append(path.split('/')[-1].split('_')[-1].replace('.csv', ''))
            rows.append(pd.read_csv(path).values.tolist())
        

        #Gather results
        results = pd.DataFrame(np.vstack(rows))
        results.columns = ['r2','scgenr2', 'n_perturbed_train', 'n_perturbed_test']
        results['perturbagen'] = perturbagens


        #Save results
        results.to_csv(osp.join(outdir, 'response_prediction_results.csv'), index=False)

        for path in glob(osp.join(outdir, 'results_tmp_*')):
            _ = subprocess.run(['rm', '-rf', path])

train_model()





