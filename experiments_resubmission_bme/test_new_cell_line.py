'''
This file loads model trained on one cell line and evaluates on all other cell lines

Note: the final performance that needs to be reported is the aggregation of performance by each of the other 9 models
'''



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
from pdgrapher import Dataset, PDGrapher, Trainer
import sys
from glob import glob

from pdgrapher._utils import get_thresholds

# Check if at least one argument is provided
if len(sys.argv) < 2:
    print("Usage: script.py <cell_line1> <cell_line2> ... <cell_lineN>")
    sys.exit(1)  # Exit the script with an error code

# Get the list of cell lines from command line arguments (excluding the script name itself)
cell_lines = sys.argv[1:]

# Example: Print the list of cell lines
print("Received cell lines:", cell_lines)



dict_cell_types = {'A549': 'lung',
                    'A375': 'skin',
                    'BT20': 'breast',
                    'HA1E': 'liver',
                    'HELA': 'cervix',
                    'HT29': 'colon',
                    'MCF7': 'breast',
                    'MDAMB231': 'breast',
                    'PC3': 'prostate',
                    'VCAP': 'prostate'}






# Process each cell line (for demonstration purposes, we'll just print them)
for cell_line in cell_lines:
    print(f"Processing cell line: {cell_line}")
    # Add your processing code here
        
    n_layers_nn = 1

    global use_forward_data
    use_forward_data = True
    if cell_line.split('_')[0] in ['HA1E', 'HT29', 'A375', 'HELA']:
        use_forward_data = False

    use_backward_data = True
    use_supervision = True #whether to use supervision loss
    use_intervention_data = True #whether to use cycle loss
    current_date = datetime.now()

    test_cell_lines = ["A549_corrected_pos_emb", "A375_corrected_pos_emb", "BT20_corrected_pos_emb", "HELA_corrected_pos_emb", "HT29_corrected_pos_emb", "MCF7_corrected_pos_emb", "MDAMB231_corrected_pos_emb", "PC3_corrected_pos_emb", "VCAP_corrected_pos_emb"]
    test_cell_lines.remove(cell_line)
    
    current_cell = dict_cell_types[cell_line.split('_')[0]]
    
    test_cell_lines = [e for e in test_cell_lines if dict_cell_types[e.split('_')[0]] != current_cell]
    
    paths = glob('results/chemical/{}/*'.format(cell_line))
    
    for path in paths:
               
        outdir = path
        path_model = path

        n_layers_gnn = int(path.split('/')[-1].split('_')[2])


        log = open(osp.join(outdir, 'final_performance_metrics_new_cell_lines_filtering_similar.txt'), 'w')

        for test_cell_line in test_cell_lines:
    
            train_dataset = Dataset(
                forward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_forward_{cell_line.split('_')[0]}.pt",
                backward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_backward_{cell_line.split('_')[0]}.pt",
                splits_path=f"../data/processed/splits/chemical/{cell_line.split('_')[0]}/random/5fold/splits.pt"
            )
            test_dataset = Dataset(
                forward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_forward_{test_cell_line.split('_')[0]}.pt",
                backward_path=f"../data/processed/torch_data/chemical/real_lognorm/data_backward_{test_cell_line.split('_')[0]}.pt",
                splits_path=f"../data/processed/splits/chemical/{test_cell_line.split('_')[0]}/random/5fold/splits.pt"
            )
            edge_index = torch.load(f"../data/processed/torch_data/chemical/real_lognorm/edge_index_{test_cell_line.split('_')[0]}.pt")


            all_recall_at_1 = []
            all_recall_at_10 = []
            all_recall_at_100 = []
            all_recall_at_1000 = []
            all_perc_partially_accurate_predictions = []
            all_rankings = []
            all_rankings_dcg = []



            for fold in range(1,6):
                #Instantiates model
                model = PDGrapher(edge_index, model_kwargs={"n_layers_nn": 1, "n_layers_gnn": n_layers_gnn, "num_vars": train_dataset.get_num_vars(),
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

                train_dataset.prepare_fold(fold)
                test_dataset.prepare_fold(fold)

                thresholds = get_thresholds(train_dataset)
                thresholds = {k: v.to(device) for k, v in thresholds.items()} 


                model.response_prediction.edge_index = model.response_prediction.edge_index.to(device)
                model.perturbation_discovery.edge_index = model.perturbation_discovery.edge_index.to(device)
                model.perturbation_discovery = model.perturbation_discovery.to(device)

                model.perturbation_discovery.eval()
                model.response_prediction.eval()


                (
                            train_loader_forward, train_loader_backward,
                            val_loader_forward, val_loader_backward,
                            test_loader_forward, test_loader_backward
                        ) = test_dataset.get_dataloaders(num_workers = 20, batch_size = 1)


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
                        
                        
                    rankings_dcg.append(dcg)                    
                        

                    recall_at_1.append(len(set(predicted_interventions[:1]).intersection(correct_interventions)) / len(correct_interventions))
                    recall_at_10.append(len(set(predicted_interventions[:10]).intersection(correct_interventions)) / len(correct_interventions))
                    recall_at_100.append(len(set(predicted_interventions[:100]).intersection(correct_interventions)) / len(correct_interventions))
                    recall_at_1000.append(len(set(predicted_interventions[:1000]).intersection(correct_interventions)) / len(correct_interventions))


                    jaccards = len(correct_interventions.intersection(predicted_interventions[:len(correct_interventions)])) / len(correct_interventions.union(predicted_interventions))

                    if jaccards != 0:
                        n_non_zeros += 1


                all_recall_at_1.append(np.mean(recall_at_1))
                all_recall_at_10.append(np.mean(recall_at_10))
                all_recall_at_100.append(np.mean(recall_at_100))
                all_recall_at_1000.append(np.mean(recall_at_1000))
                all_rankings.append(np.mean(rankings))
                all_rankings_dcg.append(np.mean(rankings_dcg))
                all_perc_partially_accurate_predictions.append(100 * n_non_zeros/len(test_loader_backward))
                print('fold {}/5'.format(fold))




            log.write('--------------------------\n')
            log.write('--------------------------\n')
            log.write('Train cell line: {}, Test cell line: {}\n'.format(cell_line, test_cell_line))
            log.write('recall@1: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_1), np.std(all_recall_at_1)))
            log.write('recall@10: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_10), np.std(all_recall_at_10)))
            log.write('recall@100: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_100), np.std(all_recall_at_100)))
            log.write('recall@1000: {:.4f}±{:.4f}\n'.format(np.mean(all_recall_at_1000), np.std(all_recall_at_1000)))
            log.write('percentage of samples with partially accurate predictions: {:.2f}±{:.2f}\n'.format(np.mean(all_perc_partially_accurate_predictions), np.std(all_perc_partially_accurate_predictions)))
            log.write('ranking score: {:.2f}±{:.2f}\n'.format(np.mean(all_rankings), np.std(all_rankings)))
            log.write('ranking score - DCG: {:.2f}±{:.2f}\n'.format(np.mean(all_rankings_dcg), np.std(all_rankings_dcg)))

            log.write('--------------------------\n')
            log.write('All metric datapoints:\n')
            log.write('recall@1: {}\n'.format(all_recall_at_1))
            log.write('recall@10: {}\n'.format(all_recall_at_10))
            log.write('recall@100: {}\n'.format(all_recall_at_100))
            log.write('recall@1000: {}\n'.format(all_recall_at_1000))
            log.write('percentage of samples with partially accurate predictions: {}\n'.format(all_perc_partially_accurate_predictions))
            log.write('ranking score: {}\n'.format(all_rankings))
            log.write('ranking score - DCG: {}\n'.format(all_rankings_dcg))
            
            log.write('--------------------------\n\n')

        log.close()












