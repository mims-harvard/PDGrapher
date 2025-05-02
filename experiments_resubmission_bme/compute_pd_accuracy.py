

import scipy
import scipy.spatial
from pdgrapher import Dataset, PDGrapher, Trainer
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



import numpy as np
import torch

import os
import os.path as osp
torch.set_num_threads(20)
from datetime import datetime

# from pdgrapher.pdgrapher_old import PDGrapherOld

from pdgrapher import PDGrapher

from pdgrapher import Trainer, Dataset
import sys
from glob import glob

from pdgrapher._utils import get_thresholds

# Check if at least one argument is provided
if len(sys.argv) < 3:
    print("Usage: script.py <data_type> <cell_line1> <cell_line2> ... <cell_lineN>")
    sys.exit(1)  # Exit the script with an error code

# Get the list of cell lines from command line arguments (excluding the script name itself)
data_type = sys.argv[1]
cell_lines = sys.argv[2:]

# Example: Print the list of cell lines
print("Received cell lines:", cell_lines)

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
    
# Process each cell line (for demonstration purposes, we'll just print them)
for cell_line in cell_lines:
    print(f"Processing cell line: {cell_line}")
    
    # Some params
    use_backward_data = True
    use_supervision = True #whether to use supervision loss
    use_intervention_data = True #whether to use cycle loss
    current_date = datetime.now() 
    n_layers_nn = 1

    global use_forward_data
    use_forward_data = True





    if data_type == 'chemical':
        if cell_line.split('_')[0] in ['HA1E', 'HT29', 'A375', 'HELA']:
            use_forward_data = False

        #Dataset
        dataset = Dataset(
            forward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_forward_{cell_line.split('_')[0]}.pt",
            backward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_backward_{cell_line.split('_')[0]}.pt",
            splits_path=f"../data/processed/splits/chemical/{cell_line.split('_')[0]}/random/5fold/splits.pt"
        )

        edge_index = torch.load(f"../data/processed/torch_data/chemical/real_lognorm/edge_index_{cell_line.split('_')[0]}.pt")

        #Modify based on folder name
        paths = glob('results/chemical/{}/*'.format(cell_line))
    
    elif data_type == 'genetic':
        if cell_line.split('_')[0] in ['ES2', 'BICR6', 'YAPC', 'AGS', 'U251MG', 'HT29', 'A375']:
            use_forward_data = False

        #Dataset
        dataset = Dataset(
            forward_path=f"../data/processed/torch_data/real_lognorm/data_forward_{cell_line.split('_')[0]}.pt",
            backward_path=f"../data/processed/torch_data/real_lognorm/data_backward_{cell_line.split('_')[0]}.pt",
            splits_path=f"../data/processed/splits/genetic/{cell_line.split('_')[0]}/random/5fold/splits.pt"
        )

        edge_index = torch.load(f"../data/processed/torch_data/real_lognorm/edge_index_{cell_line.split('_')[0]}.pt")

        #Modify based on folder name
        paths = glob('results/genetic/{}/*'.format(cell_line))  

    elif data_type == 'synthetic_missing_components':
        use_forward_data = False

        #Dataset
        dataset = Dataset(
            forward_path=None,
            backward_path="../data/processed/synthetic_lincs/chemical/missing_components/data_backward_synthetic_MDAMB231.pt",
            splits_path="../data/processed/splits/synthetic_lincs/chemical/missing_components/random/5fold/splits.pt"
        )

        

        #Modify based on folder name
        paths = glob('results/synthetic_lincs/data_missing_components/*/*')  
        
    elif data_type == 'synthetic_confounders':
        use_forward_data = False
        edge_index = torch.load(f"../data/processed/synthetic_lincs/chemical/confounders/edge_index_synthetic_MDAMB231.pt")
        
        

        #Modify based on folder name
        paths = glob('results/synthetic_lincs/confounders/*/*')    



    elif data_type == 'synthetic_missing_components_random':
        use_forward_data = False

        #Dataset
        dataset = Dataset(
            forward_path=None,
            backward_path="../data/processed/synthetic_lincs/chemical/missing_components/data_backward_synthetic_MDAMB231.pt",
            splits_path="../data/processed/splits/synthetic_lincs/chemical/missing_components_random/random/5fold/splits.pt"
        )

        

        #Modify based on folder name
        paths = glob('results/synthetic_lincs/data_missing_components_random/*/*')  



    for path in paths:
        
        if data_type == 'synthetic_missing_components':
            fraction = path.split('/')[-2].split('_')[1]
            edge_index = torch.load(f"../data/processed/synthetic_lincs/chemical/missing_components/edge_index_MDAMB231_fraction_{fraction}.pt")

        if data_type == 'synthetic_missing_components_random':
            fraction = path.split('/')[-2].split('_')[1]
            edge_index = torch.load(f"../data/processed/synthetic_lincs/chemical/missing_components_random/edge_index_MDAMB231_fraction_{fraction}.pt")
            
        if data_type == 'synthetic_confounders':
            fraction = path.split('/')[-2].split('_')[1]
            #Dataset
            dataset = Dataset(
                forward_path=None,
                backward_path="../data/processed/synthetic_lincs/chemical/confounders/data_backward_synthetic_MDAMB231_fraction_{}.pt".format(fraction),
                splits_path="../data/processed/splits/synthetic_lincs/chemical/confounders/random/5fold/splits.pt"
            )

        outdir = path
        path_model = path

        n_layers_gnn = int(path.split('/')[-1].split('_')[2])


        all_recall_at_1 = {'test':[], 'val':[]}
        all_recall_at_10 = {'test':[], 'val':[]}
        all_recall_at_100 = {'test':[], 'val':[]}
        all_recall_at_1000 = {'test':[], 'val':[]}
        all_perc_partially_accurate_predictions = {'test':[], 'val':[]}
        all_rankings = {'test':[], 'val':[]}
        all_rankings_dcg = {'test':[], 'val':[]}



        for fold in range(1,6):
            #Instantiates model
            model = PDGrapher(edge_index, model_kwargs={"n_layers_nn": 1, "n_layers_gnn": n_layers_gnn, "num_vars": dataset.get_num_vars(),
                                                        },
                                        response_kwargs={'train': True},
                                        perturbation_kwargs={'train':True})

            # restore response prediction
            save_path = osp.join(path, '_fold_{}_response_prediction.pt'.format(fold))
            checkpoint = torch.load(save_path)
            model.response_prediction.load_state_dict(checkpoint["model_state_dict"])
            # restore Perturbation discovery
            save_path = osp.join(path, '_fold_{}_perturbation_discovery.pt'.format(fold))
            checkpoint = torch.load(save_path)
            model.perturbation_discovery.load_state_dict(checkpoint["model_state_dict"])

            #loads fold-specific dataset
            device = torch.device('cuda')

            dataset.prepare_fold(fold)

            thresholds = get_thresholds(dataset)
            thresholds = {k: v.to(device) if v is not None else v for k, v in thresholds.items()} 


            model.response_prediction.edge_index = model.response_prediction.edge_index.to(device)
            model.perturbation_discovery.edge_index = model.perturbation_discovery.edge_index.to(device)
            model.perturbation_discovery = model.perturbation_discovery.to(device)

            model.perturbation_discovery.eval()
            model.response_prediction.eval()


            (
                        train_loader_forward, train_loader_backward,
                        val_loader_forward, val_loader_backward,
                        test_loader_forward, test_loader_backward
                    ) = dataset.get_dataloaders(num_workers = 20, batch_size = 1)


            recall_at_1 = []
            recall_at_10 = []
            recall_at_100 = []
            recall_at_1000 = []
            perc_partially_accurate_predictions = []
            rankings = []
            rankings_dcg = []
            n_non_zeros = 0



            for data in test_loader_backward:
                pred_backward_m2 = model.perturbation_discovery(torch.concat([data.diseased.view(-1, 1).to(device), data.treated.view(-1, 1).to(device)], 1), data.batch.to(device), mutilate_mutations=data.mutations.to(device), threshold_input=thresholds)
                out = pred_backward_m2
                # import pdb; pdb.set_trace()
                                
                num_nodes = int(data.num_nodes / len(torch.unique(data.batch)))


                correct_interventions = set(torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[1].tolist())
                predicted_interventions = torch.argsort(out.detach().cpu().view(-1, num_nodes), descending=True)[0, :].tolist()

                for ci in list(correct_interventions):
                    rankings.append(1 - (predicted_interventions.index(ci) / num_nodes))
                
                
                #Weighted truncated ranking metric
                
                dcg = 0
                for ci in list(correct_interventions):
                    # Get the rank of the current ground-truth intervention
                    rank = predicted_interventions.index(ci) + 1 #1-based indexing for CDG
                    gain = 1 - (rank / num_nodes)
                    discount = 1 / np.log2(rank + 1)
                    dcg += gain * discount
                    
                #normalize
                idcg = compute_idcg(len(correct_interventions), num_nodes)
                ndcg = dcg / idcg if idcg > 0 else 0
                rankings_dcg.append(ndcg)
                    


                
                recall_at_1.append(len(set(predicted_interventions[:1]).intersection(correct_interventions)) / len(correct_interventions))
                recall_at_10.append(len(set(predicted_interventions[:10]).intersection(correct_interventions)) / len(correct_interventions))
                recall_at_100.append(len(set(predicted_interventions[:100]).intersection(correct_interventions)) / len(correct_interventions))
                recall_at_1000.append(len(set(predicted_interventions[:1000]).intersection(correct_interventions)) / len(correct_interventions))


                jaccards = len(correct_interventions.intersection(predicted_interventions[:len(correct_interventions)])) / len(correct_interventions.union(predicted_interventions))

                if jaccards != 0:
                    n_non_zeros += 1


            all_recall_at_1['test'].append(np.mean(recall_at_1))
            all_recall_at_10['test'].append(np.mean(recall_at_10))
            all_recall_at_100['test'].append(np.mean(recall_at_100))
            all_recall_at_1000['test'].append(np.mean(recall_at_1000))
            all_rankings['test'].append(np.mean(rankings))
            all_rankings_dcg['test'].append(np.mean(rankings_dcg))
            
            all_perc_partially_accurate_predictions['test'].append(100 * n_non_zeros/len(test_loader_backward))
            print('fold {}/5'.format(fold))




            ####VALIDATION SET
            recall_at_1 = []
            recall_at_10 = []
            recall_at_100 = []
            recall_at_1000 = []
            perc_partially_accurate_predictions = []
            rankings = []
            rankings_dcg = []
            # den_100 = []
            n_non_zeros = 0
            
            
            for data in val_loader_backward:
                pred_backward_m2 = model.perturbation_discovery(torch.concat([data.diseased.view(-1, 1).to(device), data.treated.view(-1, 1).to(device)], 1), data.batch.to(device), mutilate_mutations=data.mutations.to(device), threshold_input=thresholds)
                out = pred_backward_m2
                                
                num_nodes = int(data.num_nodes / len(torch.unique(data.batch)))
                
                
                correct_interventions = set(torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[1].tolist())
                predicted_interventions = torch.argsort(out.detach().cpu().view(-1, num_nodes), descending=True)[0, :].tolist()

                for ci in list(correct_interventions):
                    rankings.append(1 - (predicted_interventions.index(ci) / num_nodes))
                
                
                dcg = 0
                for ci in list(correct_interventions):
                    # Get the rank of the current ground-truth intervention
                    rank = predicted_interventions.index(ci) + 1 #1-based indexing for CDG
                    gain = 1 - (rank / num_nodes)
                    discount = 1 / np.log2(rank + 1)
                    dcg += gain * discount
                    
                #normalize
                idcg = compute_idcg(len(correct_interventions), num_nodes)
                ndcg = dcg / idcg if idcg > 0 else 0
                rankings_dcg.append(ndcg)
                
                
                
                recall_at_1.append(len(set(predicted_interventions[:1]).intersection(correct_interventions)) / len(correct_interventions))
                recall_at_10.append(len(set(predicted_interventions[:10]).intersection(correct_interventions)) / len(correct_interventions))
                recall_at_100.append(len(set(predicted_interventions[:100]).intersection(correct_interventions)) / len(correct_interventions))
                recall_at_1000.append(len(set(predicted_interventions[:1000]).intersection(correct_interventions)) / len(correct_interventions))


                jaccards = len(correct_interventions.intersection(predicted_interventions[:len(correct_interventions)])) / len(correct_interventions.union(predicted_interventions))

                if jaccards != 0:
                    n_non_zeros += 1


            all_recall_at_1['val'].append(np.mean(recall_at_1))
            all_recall_at_10['val'].append(np.mean(recall_at_10))
            all_recall_at_100['val'].append(np.mean(recall_at_100))
            all_recall_at_1000['val'].append(np.mean(recall_at_1000))
            all_rankings['val'].append(np.mean(rankings))
            all_rankings_dcg['val'].append(np.mean(rankings_dcg))
            
            all_perc_partially_accurate_predictions['val'].append(100 * n_non_zeros/len(test_loader_backward))
            print('fold {}/5'.format(fold))





        log = open(osp.join(outdir, 'final_performance_metrics.txt'), 'w')
        log.write('\n\nVALIDATION SET\n')
        log.write('recall@1: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_1['val']), np.std(all_recall_at_1['val'])))
        log.write('recall@10: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_10['val']), np.std(all_recall_at_10['val'])))
        log.write('recall@100: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_100['val']), np.std(all_recall_at_100['val'])))
        log.write('recall@1000: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_1000['val']), np.std(all_recall_at_1000['val'])))
        log.write('percentage of samples with partially accurate predictions: {:.2f}±{:.2f}\n'.format(np.mean(all_perc_partially_accurate_predictions['val']), np.std(all_perc_partially_accurate_predictions['val'])))
        log.write('ranking score: {:.2f}±{:.2f}\n'.format(np.mean(all_rankings['val']), np.std(all_rankings['val'])))
        log.write('ranking score - DCG: {:.2f}±{:.2f}\n'.format(np.mean(all_rankings_dcg['val']), np.std(all_rankings_dcg['val'])))

        log.write('--------------------------\n')
        log.write('All metric datapoints:\n')
        log.write('recall@1: {}\n'.format(all_recall_at_1['val']))
        log.write('recall@10: {}\n'.format(all_recall_at_10['val']))
        log.write('recall@100: {}\n'.format(all_recall_at_100['val']))
        log.write('recall@1000: {}\n'.format(all_recall_at_1000['val']))
        log.write('percentage of samples with partially accurate predictions: {}\n'.format(all_perc_partially_accurate_predictions['val']))
        log.write('ranking score: {}\n'.format(all_rankings['val']))
        log.write('ranking score - DCG: {}\n'.format(all_rankings_dcg['val']))

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












