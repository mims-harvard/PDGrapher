
import random
import numpy as np
import pandas as pd
import torch
from pdgrapher._utils import get_thresholds
from pdgrapher import PDGrapher
from pdgrapher import Trainer, Dataset
from glob import glob
import os.path as osp
import os
import sys 
import pickle



#Selects cell line
cell_line = sys.argv[1]  # Get the cell line from command line arguments


#Outdir
outdir = cell_line
os.makedirs(cell_line, exist_ok = True)
log = open(osp.join(outdir, 'log.txt'), 'w')


#First extract the drugs against cancer types for which we have healthy counterparts
cell_lines = {'A549': 'NL20', 'MCF7': 'MCF10A', 'MDAMB231': 'MCF10A', 'BT20': 'MCF10A', 'PC3': 'RWPE1', 'VCAP': 'RWPE1'}
cancer_types = {'A549': 'lung', 'MCF7': 'breast', 'MDAMB231': 'breast', 'BT20':'breast', 'PC3': 'prostate', 'VCAP': 'prostate'}

#Reads in approved drugs
drugs = pd.read_csv('../../data/processed/nci/drugs_and_targets.csv', sep='\t')


#Selects approved drugs for cell line use case
drugs_approved = drugs[drugs['cell_line'] == cell_line.split('_')[0]]['drug'].tolist()

#Loads torch datasets
forward_dataset = torch.load('../../data/processed/torch_data/chemical/real_lognorm/data_forward_{}.pt'.format(cell_line.split('_')[0]))
backward_dataset = torch.load('../../data/processed/torch_data/chemical/real_lognorm/data_backward_{}.pt'.format(cell_line.split('_')[0]))
splits = torch.load('../../data/processed/splits/chemical/{}/random/5fold/splits.pt'.format(cell_line.split('_')[0]))
gene_symbols = forward_dataset[0].gene_symbols


#Gets drugs in training, and drugs approved not in training (these are the ones we need to check if we recover)
#Gets unique drugs in dataset
perturbagens_in_dataset = []
for d in backward_dataset:
    perturbagens_in_dataset.append(d.perturbagen_name)
    

perturbagens_in_dataset = list(set(perturbagens_in_dataset))


#Overlap between approved drugs and perturbagens in dataset
drugs_in_both = set(drugs_approved).intersection(perturbagens_in_dataset)


#Retrieve drugs only approved but not in dataset
approved_drugs_not_in_dataset = list(set(drugs_approved) - set(perturbagens_in_dataset))


#Genes of approved drugs (do we recover them?)
genes_in_approved_drugs = []
for d in approved_drugs_not_in_dataset:
    for e in drugs[drugs['drug']==d]['targets']:
        genes_in_approved_drugs += e.split(',')


genes_in_approved_drugs = set(genes_in_approved_drugs)



#Builds dataset of diseased and healthy samples. Gets mutated genes, and perturbed genes in dataset
diseased_samples = set()
healthy_samples = set()
mutated_genes = set()
mutated_genes_indices = set()
perturbed_genes_in_dataset = set()


for data in forward_dataset:
    healthy_samples.add(tuple(data.healthy.numpy()))
    diseased_samples.add(tuple(data.diseased.numpy()))
    genes = np.take(gene_symbols, torch.where(data.mutations)[0].numpy())
    mutated_genes.update(genes)
    genes_indices = torch.where(data.mutations)[0].numpy()
    mutated_genes_indices.update(genes_indices)


for data in backward_dataset:
    if data.perturbagen_name in drugs_approved:
        continue
    genes = np.take(gene_symbols, torch.where(data.intervention)[0].numpy())
    perturbed_genes_in_dataset.update(genes)
    
    
diseased_samples = list(diseased_samples)
healthy_samples = list(healthy_samples)


#Makes predictions for all folds
##Some params
use_backward_data = True
use_supervision = True #whether to use supervision loss
use_intervention_data = True #whether to use cycle loss
n_layers_nn = 1

global use_forward_data
use_forward_data = True
if cell_line.split('_')[0] in ['HA1E', 'HT29', 'A375', 'HELA']:
    use_forward_data = False

##Loads dataset
dataset = Dataset(
            forward_path=f"../../data/processed/torch_data/chemical/real_lognorm/data_forward_{cell_line.split('_')[0]}.pt",
            backward_path=f"../../data/processed/torch_data/chemical/real_lognorm/data_backward_{cell_line.split('_')[0]}.pt",
            splits_path=f"../../data/processed/splits/chemical/{cell_line.split('_')[0]}/random/5fold/splits.pt"
        )

edge_index = torch.load(f"../../data/processed/torch_data/chemical/real_lognorm/edge_index_{cell_line.split('_')[0]}.pt")



#Selects best model for this cell line
performance = pd.read_csv('../../results_metrics_aggregated_bme/perturbagen_pred/PDgrapher/within/chemical/val/{}_drugpred_within_best.csv'.format(cell_line.split('_')[0]))
ngnn = performance[performance['Set'] == 'Test']['GNN'].iloc[0]
path = glob('../../experiments_resubmission_bme/results/chemical/{}_corrected_pos_emb/n_gnn_{}*'.format(cell_line.split('_')[0], ngnn))[0]


def check_overlap_and_usecase(k):
    #Get targets of approved drugs not in training
    dict_approved_drugs_and_targets = dict()
    for d in approved_drugs_not_in_dataset:
        for e in drugs[drugs['drug']==d]['targets']:
            dict_approved_drugs_and_targets[d] = e.split(',')
            
    #Genes of approved drugs (do we recover them?)
    genes_in_approved_drugs = []
    for d in approved_drugs_not_in_dataset:
        for e in drugs[drugs['drug']==d]['targets']:
            genes_in_approved_drugs += e.split(',')


    genes_in_approved_drugs = set(genes_in_approved_drugs)

    predicted_genes = np.array(gene_symbols)[aggregated_ranking[0:k]]

    predicted_genes_that_are_in_approved_drugs = list(genes_in_approved_drugs.intersection(predicted_genes))

    #Subdictionary with drugs that have at least one target that was predicted
    subdict = {drug: targets for drug, targets in dict_approved_drugs_and_targets.items() 
            if any(target in predicted_genes_that_are_in_approved_drugs for target in targets)}

    log.write('\nUSE CASES:\n')
    log.write('Of the top {} predicted genes, these ones are targets of approved drugs not in the training set: {}\n'.format(k, predicted_genes_that_are_in_approved_drugs))
    log.write('The approved drugs (not in training) that have at least one of those genes as targets are: {}\n'.format(subdict))
    log.write('--\n')




#This will save the predictions from diseased -> healthy
scores = []
for fold in range(1,6):
    n_layers_gnn = ngnn

    #Instantiate model
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


    (train_loader_forward, train_loader_backward,
    val_loader_forward, val_loader_backward,
    test_loader_forward, test_loader_backward
            ) = dataset.get_dataloaders(num_workers = 20, batch_size = 1)


    #Go through all diseased samples, predict diseased -> healthy
    #Average the scores, and then retrieve the top genes
    for i in range(len(diseased_samples)):
        diseased = torch.Tensor(diseased_samples[i]).to(device)
        healthy = torch.Tensor(healthy_samples[random.randint(0, len(healthy_samples)-1)]).to(device)
        batch = torch.ones(len(diseased))
        mutations = torch.zeros(len(diseased))
        mutations[list(mutated_genes_indices)] = 1
        out = model.perturbation_discovery(torch.concat([diseased.view(-1, 1), healthy.view(-1, 1)], 1), batch.to(device), mutilate_mutations=mutations.to(device), threshold_input=thresholds)
        num_nodes = len(diseased)
        predicted_interventions = out.detach().cpu().view(-1, num_nodes)
        scores.append(predicted_interventions)
        



#Aggregate ranking into single ranking
scores = torch.mean(torch.stack(scores).squeeze(1), 0)
aggregated_ranking = torch.argsort(scores.detach().cpu().view(-1, num_nodes), descending=True)[0, :].tolist()

with open(osp.join(outdir, 'aggregated_ranking.pickle'), 'wb') as f:
    pickle.dump(aggregated_ranking, f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
########################################################################################################################
############Use case - comparing precision and recall of recovering targets of approved drugs not in training set
##Maybe the "perturbed" is not so important, but just the fact we can recover the genes (better than what would be expected by chance) -> I need to check what the metrics would look like for random ranking

#Genes of approved drugs (do we recover them?)
genes_in_approved_drugs = []
for d in approved_drugs_not_in_dataset:
    for e in drugs[drugs['drug']==d]['targets']:
        genes_in_approved_drugs += e.split(',')


genes_in_approved_drugs = set(genes_in_approved_drugs)


#Random ranking
aggregated_ranking_random = [e for e in range(len(diseased_samples[0]))]
random.shuffle(aggregated_ranking_random)

with open(osp.join(outdir, 'aggregated_ranking_random.pickle'), 'wb') as f:
    pickle.dump(aggregated_ranking_random, f)

# Initialize lists to store results
overlap_approved = []
precision_approved = []
recall_approved = []
overlap_random = []
precision_random = []
recall_random = []
# Define k values
ks = [1, 10, 20, 30, 50, 100, 200, 300, 400, 500, 1000]

# Calculate overlap, precision, and recall for each k
for k in ks:
    if k > len(aggregated_ranking):
        break
    predicted_genes = np.array(gene_symbols)[aggregated_ranking[0:k]]
    predicted_genes_random = np.array(gene_symbols)[aggregated_ranking_random[0:k]]
    overlap_approved_k = len(genes_in_approved_drugs.intersection(predicted_genes))
    overlap_random_k = len(genes_in_approved_drugs.intersection(predicted_genes_random))
    overlap_approved.append(overlap_approved_k)
    overlap_random.append(overlap_random_k)
    precision_approved_k = overlap_approved_k / k
    recall_approved_k = overlap_approved_k / len(genes_in_approved_drugs)
    precision_random_k = overlap_random_k / k
    recall_random_k = overlap_random_k / len(genes_in_approved_drugs)
    precision_approved.append(precision_approved_k)
    recall_approved.append(recall_approved_k)
    precision_random.append(precision_random_k)
    recall_random.append(recall_random_k)



# Print the results
log.write('Mutated genes in this cancer type: {}\n\n'.format(mutated_genes))
log.write('Comparing recall of recovering targets of approved drugs not in training set vs random\n')
log.write('K: {}\n'.format(ks))
# log.write("Precision (Approved): {}\n".format(precision_approved))
formatted_recall_approved = [float(f"{x:.4f}") for x in recall_approved]
log.write("Recall (Approved): {}\n".format(formatted_recall_approved))
# log.write("Precision (Random): {}\n".format(precision_random))
formatted_recall_random = [float(f"{x:.4f}") for x in recall_random]
log.write("Recall (Random): {}\n\n".format(formatted_recall_random))




########################################################################################################################
############Use case - comparing precision and recall of recovering targets of approved drugs not in training set (additionally, only considering the targets that are not in the training set!)
genes_in_approved_drugs = genes_in_approved_drugs - perturbed_genes_in_dataset

if len(genes_in_approved_drugs) != 0:
    # Initialize lists to store results
    overlap_approved = []
    precision_approved = []
    recall_approved = []
    overlap_random = []
    precision_random = []
    recall_random = []
    # Define k values
    ks = [1, 10, 20, 30, 50, 100, 200, 300, 400, 500, 1000]

    # Calculate overlap, precision, and recall for each k
    for k in ks:
        if k > len(aggregated_ranking):
            break
        predicted_genes = np.array(gene_symbols)[aggregated_ranking[0:k]]
        predicted_genes_random = np.array(gene_symbols)[aggregated_ranking_random[0:k]]
        overlap_approved_k = len(genes_in_approved_drugs.intersection(predicted_genes))
        overlap_random_k = len(genes_in_approved_drugs.intersection(predicted_genes_random))
        overlap_approved.append(overlap_approved_k)
        overlap_random.append(overlap_random_k)
        precision_approved_k = overlap_approved_k / k
        
        recall_approved_k = overlap_approved_k / len(genes_in_approved_drugs)
        precision_random_k = overlap_random_k / k
        recall_random_k = overlap_random_k / len(genes_in_approved_drugs)
        precision_approved.append(precision_approved_k)
        recall_approved.append(recall_approved_k)
        precision_random.append(precision_random_k)
        recall_random.append(recall_random_k)

    # Print the results
    log.write('Comparing recall of recovering targets of approved drugs not in training set vs random (additionally, only considering the targets that are not in the training set!)\n')
    log.write('K: {}\n'.format(ks))
    # log.write("Precision (Approved): {}\n".format(precision_approved))
    formatted_recall_approved = [float(f"{x:.4f}") for x in recall_approved]
    log.write("Recall (Approved): {}\n".format(formatted_recall_approved))
    # log.write("Precision (Random): {}\n".format(precision_random))
    formatted_recall_random = [float(f"{x:.4f}") for x in recall_random]
    log.write("Recall (Random): {}\n\n".format(formatted_recall_random))
else:
    log.write('Comparing recall of recovering targets of approved drugs not in training set vs random (additionally, only considering the targets that are not in the training set!)\n')
    log.write('Not possible because there are no targets of approved drugs that are not in training set (all targets of approved drugs are in training set)\n\n')





########################################################################################################################
############Use case - precision and recall of recovering perturbed targets in training set that are not targets of approved drugs

#Genes of approved drugs (do we recover them?)
genes_in_approved_drugs = []
for d in approved_drugs_not_in_dataset:
    for e in drugs[drugs['drug']==d]['targets']:
        genes_in_approved_drugs += e.split(',')


genes_in_approved_drugs = set(genes_in_approved_drugs)

genes_in_approved_drugs = perturbed_genes_in_dataset  - genes_in_approved_drugs

# Initialize lists to store results
overlap_approved = []
precision_approved = []
recall_approved = []
overlap_random = []
precision_random = []
recall_random = []
# Define k values
ks = [1, 10, 20, 30, 50, 100, 200, 300, 400, 500, 1000]

# Calculate overlap, precision, and recall for each k
for k in ks:
    if k > len(aggregated_ranking):
        break
    predicted_genes = np.array(gene_symbols)[aggregated_ranking[0:k]]
    predicted_genes_random = np.array(gene_symbols)[aggregated_ranking_random[0:k]]
    overlap_approved_k = len(genes_in_approved_drugs.intersection(predicted_genes))
    overlap_random_k = len(genes_in_approved_drugs.intersection(predicted_genes_random))
    overlap_approved.append(overlap_approved_k)
    overlap_random.append(overlap_random_k)
    precision_approved_k = overlap_approved_k / k
    recall_approved_k = overlap_approved_k / len(genes_in_approved_drugs)
    precision_random_k = overlap_random_k / k
    recall_random_k = overlap_random_k / len(genes_in_approved_drugs)
    precision_approved.append(precision_approved_k)
    recall_approved.append(recall_approved_k)
    precision_random.append(precision_random_k)
    recall_random.append(recall_random_k)

# Print the results
log.write('Comparing recall of recovering targets perturbed in the dataset which are not targets of approved drugs in training set vs random\n')
log.write('K: {}\n'.format(ks))
# log.write("Precision (Perturbed not approved): {}\n".format(precision_approved))
formatted_recall_approved = [float(f"{x:.4f}") for x in recall_approved]
log.write("Recall (Perturbed not approved): {}\n".format(formatted_recall_approved))
# log.write("Precision (Random): {}\n".format(precision_random))
formatted_recall_random = [float(f"{x:.4f}") for x in recall_random]
log.write("Recall (Random): {}\n\n".format(formatted_recall_random))




#########Use case -- pick a few example drugs whose genes are recovered well
#Let's take the top k genes, check if there's any overlap with the genes in approved drugs, and log the drugs that have any of the top genes as target
check_overlap_and_usecase(k=10)
check_overlap_and_usecase(k=20)
check_overlap_and_usecase(k=30)
check_overlap_and_usecase(k=50)
check_overlap_and_usecase(k=100)


log.close()