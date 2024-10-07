


import torch
import os.path as osp
import h5py



base_path = '../processed/real_lognorm'

#Loads datasets
#MCF7
# mcf7_forward = torch.load(osp.join(base_path, 'data_forward_MCF7.pt'))
mcf7_backward = torch.load(osp.join(base_path, 'data_backward_MCF7.pt'))

#A549
# a549_forward = torch.load(osp.join(base_path, 'data_forward_A549.pt'))
a549_backward = torch.load(osp.join(base_path, 'data_backward_A549.pt'))


#Builds datasets
mcf7_control = []
mcf7_treated = []
mcf7_perturbagen = []
for data in mcf7_backward:
    mcf7_control.append(data.diseased.numpy().tolist())
    mcf7_treated.append(data.treated.numpy().tolist())
    mcf7_perturbagen.append(data.perturbagen_name)
    

    
a549_treated = []
a549_perturbagen = []
a549_control = []
for data in a549_backward:
    a549_control.append(data.diseased.numpy().tolist())
    a549_treated.append(data.treated.numpy().tolist())
    a549_perturbagen.append(data.perturbagen_name)


#Transforms datasets into pandas
import numpy as np
import pandas as pd
mcf7_control = pd.DataFrame(np.array(list(mcf7_control)))
mcf7_treated = pd.DataFrame(np.array(mcf7_treated))
a549_control = pd.DataFrame(np.array(list(a549_control)))
a549_treated = pd.DataFrame(np.array(a549_treated))


#Creating obs
cell_line = ['MCF7' for i in range(len(mcf7_control) + len(mcf7_treated))] + ['A549' for i in range(len(a549_control) + len(a549_treated))]
condition = ['control' for i in range(len(mcf7_control))] + mcf7_perturbagen +  ['control' for i in range(len(a549_control))] + a549_perturbagen


#Creates annotated data
X = pd.concat([mcf7_control, mcf7_treated, a549_control, a549_treated], 0).reset_index(inplace=False, drop=True)
obs = pd.DataFrame([cell_line, condition]).transpose()
obs.columns = ['cell_type', 'condition']
var = pd.DataFrame(a549_backward[0].gene_symbols, columns = ['gene_symbols'])

import anndata
train = anndata.AnnData(X, obs,var)

import hdf5plugin
train.write(
    osp.join(base_path, 'data_scgen.h5ad')
)


