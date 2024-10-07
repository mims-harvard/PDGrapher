

import torch
import os.path as osp
import h5py
import pandas as pd
import numpy as np
import anndata

base_path = '../../processed/torch_data/chemical/real_lognorm'

cell_lines = ["A549","A375","BT20","HA1E","HELA","HT29","MCF7","MDAMB231","PC3","VCAP"]

for cell_line in cell_lines:
    print("Processing cell line: ", cell_line)
    #Loads datasets
    data_backward = torch.load(osp.join(base_path, 'data_backward_' + cell_line + '.pt'))
    #Builds datasets
    data_treated = []
    data_perturbagen = []
    data_control = []
    for data in data_backward:
        data_control.append(data.diseased.numpy().tolist())
        data_treated.append(data.treated.numpy().tolist())
        data_perturbagen.append(data.perturbagen_name)
    #Transforms datasets into pandas
    data_control = pd.DataFrame(np.array(list(data_control)))
    data_treated = pd.DataFrame(np.array(data_treated))
    print(data_control.shape)
    print(data_treated.shape)
    #Creating obs
    cellline = [cell_line for i in range(len(data_control) + len(data_treated))]
    condition = ['control' for i in range(len(data_control))] + data_perturbagen
    #Creates annotated data
    X = pd.concat([data_control, data_treated], axis=0).reset_index(inplace=False, drop=True)
    obs = pd.DataFrame([cellline, condition]).transpose()
    obs.columns = ['cell_type', 'condition']
    var = pd.DataFrame(data_backward[0].gene_symbols, columns = ['gene_symbols'])
    train = anndata.AnnData(X, obs,var)
    ind = torch.load(f"../../processed/splits/chemical/{cell_line}/random/5fold/splits.pt")
    for j in range(1,6):
        train_ = ind[j]["train_index_backward"]
        test_ = ind[j]["test_index_backward"]
        val_ = ind[j]["val_index_backward"]
        print(len(train_) + len(test_) + len(val_))
        assert len(train_) + len(test_) + len(val_) == len(train) / 2
        print("Pass checking")
        temp = np.array(['data0' for _ in range(len(data_treated))])
        temp[train_] = 'train'
        temp[test_] = 'test'
        temp[val_] = 'val'
        train.obs['split'+str(j)] = np.concatenate([temp, temp], axis=0) # same split for treated and control
        train.write(
        osp.join(base_path, 'data_split_' + cell_line +'.h5ad')
    )
