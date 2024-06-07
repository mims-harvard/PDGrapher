import torch
import pandas as pd
import sys
import numpy as np
import os
import os.path as osp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.manual_seed(0)
np.random.seed(0)

sys.path.append('../../../../source')
sys.path.append('../../../../baselines/source')

from gcn import GCNModelInterventionDiscovery
from utils import get_threshold_diseased, get_threshold_treated
from torch_geometric.loader import DataLoader





import numpy as np
import torch

import os
import os.path as osp
torch.set_num_threads(20)
from datetime import datetime
from pdgrapher import Dataset, PDGrapher, Trainer
import sys
from glob import glob
from _utils import get_thresholds

#PARAMS
cell_line = sys.argv[1]  # Get the cell line from command line arguments
n_layers_nn = 1
global use_forward_data
use_forward_data = True
if cell_line in ['HA1E', 'HT29', 'A375', 'HELA']:
    use_forward_data = False

use_backward_data = True
use_supervision = True #whether to use supervision loss
use_intervention_data = True #whether to use cycle loss
current_date = datetime.now()



#Modify based on folder name
n_layers_gnn = 2
path = 'A549'
outdir = path
path_model = path

dataset = Dataset(
    forward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_forward_{cell_line}.pt",
    backward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_backward_{cell_line}.pt",
    splits_path=f"../data/processed/splits/chemical/{cell_line}/random/5fold/splits.pt"
)

edge_index = torch.load(f"../data/processed/torch_data/chemical/real_lognorm/edge_index_{cell_line}.pt")





top_1_perf = []
int_acc_perf = []
int_acc_perf_all = []
avg_zeros = []



for split_index in range(1,6):
    model = PDGrapher(edge_index, model_kwargs={"n_layers_nn": 1, "n_layers_gnn": n_layers_gnn, "num_vars": dataset.get_num_vars(),
                                                },
                                response_kwargs={'train': True},
                                perturbation_kwargs={'train':True})




    fold = 1
    save_path = osp.join(path, '_fold_{}_response_prediction.pt'.format(fold))
    checkpoint = torch.load(save_path)
    model.response_prediction.load_state_dict(checkpoint["model_state_dict"])
    # restore Perturbation discovery
    save_path = osp.join(path, '_fold_{}_perturbation_discovery.pt'.format(fold))
    checkpoint = torch.load(save_path)
    model.perturbation_discovery.load_state_dict(checkpoint["model_state_dict"])



    dataset.prepare_fold(fold)

    thresholds = get_thresholds(dataset)
    thresholds = {k: model.fabric.to_device(v) for k, v in thresholds.items()} # do we really need them?
    model.response_prediction.edge_index = model.fabric.to_device(model.response_prediction.edge_index)
    model.perturbation_discovery.edge_index = model.fabric.to_device(model.perturbation_discovery.edge_index)
    model.perturbation_discovery.eval()
    model.response_prediction.eval()


    (
                train_loader_forward, train_loader_backward,
                val_loader_forward, val_loader_backward,
                test_loader_forward, test_loader_backward
            ) = dataset.get_dataloaders(num_workers = 20)


    top_1_binary = []
    int_acc = []
    int_acc_all = []
    n_zeros = 0
    for data in test_loader_backward:
        pred_backward_m2 = model.perturbation_discovery(torch.concat([model.fabric.to_device(data.diseased.view(-1, 1)), model.fabric.to_device(data.treated.view(-1, 1))], 1), model.fabric.to_device(model.batch), mutilate_mutations=model.fabric.to_device(model.mutations), threshold_input=thresholds)
        out = pred_backward_m2
                        
        num_nodes = int(data.num_nodes / len(torch.unique(data.batch)))
        correct_interventions = tuple(zip(torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[0].tolist(), torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[1].tolist()))
        
        correct_interventions = set(torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[1].tolist())
        predicted_interventions = set(torch.argsort(out.detach().cpu().view(-1, num_nodes), descending=True)[0, 0:len(correct_interventions)].tolist())
        top_1_binary.append(int(len(correct_interventions.intersection(predicted_interventions)) == len(correct_interventions)))
        jaccards = len(correct_interventions.intersection(predicted_interventions)) / len(correct_interventions.union(predicted_interventions))
        if jaccards ==0:
            n_zeros += 1
        else:
            int_acc.append(jaccards)
        int_acc_all.append(jaccards)

    top_1_perf.append(np.mean(top_1_binary))
    int_acc_perf.append(np.mean(int_acc))
    int_acc_perf_all.append(np.mean(int_acc_all))
    avg_zeros.append(n_zeros/len(test_loader_backward))







outdir_i = osp.join(outdir, path_model.split('/')[-4], path_model.split('/')[-2])
os.makedirs(outdir_i, exist_ok=True)
log = open(osp.join(outdir_i, 'top_performance.txt'), 'w')
log.write('top 1: {} ± {}\n'.format(np.mean(top_1_perf), np.std(top_1_perf)))
log.write('interv acc (only non-zero): {} ± {}\n'.format(np.mean(int_acc_perf), np.std(int_acc_perf)))
log.write('interv acc (all): {} ± {}\n'.format(np.mean(int_acc_perf_all), np.std(int_acc_perf_all)))
log.write('n zeros: {} ± {}\n'.format(np.mean(avg_zeros), np.std(avg_zeros)))
log.write('all non-zeros (final PD accuracy): {}\n'.format(avg_zeros))
log.close()












