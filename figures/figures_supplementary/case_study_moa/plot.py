

import torch
import pandas as pd
import sys
import numpy as np
import os
import os.path as osp

torch.manual_seed(0)
np.random.seed(0)
from pdgrapher import Dataset, PDGrapher, Trainer
from torch_geometric.loader import DataLoader
# sys.path.append('../../../../source')
# sys.path.append('../../../../baselines/source')

import sys
from glob import glob

from pdgrapher._utils import get_thresholds
import copy
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt

outdir = 'processed'
os.makedirs(outdir, exist_ok=True)




def create_graph_from_edge_index(edge_index):
    G = nx.Graph()
    for edge in edge_index.t().tolist():
        G.add_edge(edge[0], edge[1])
    return G

def remove_self_loops(G):
    # Get a list of self-loop edges (edges where source and target nodes are the same)
    self_loop_edges = list(nx.selfloop_edges(G))
    # Remove self-loop edges from the graph
    G.remove_edges_from(self_loop_edges)

def find_neighbors(G, nodes, hop=1):
    neighbors = set(nodes)
    for _ in range(hop):
        neighbors = set((n for neighbor in neighbors for n in G.neighbors(neighbor)))
    return neighbors

def plot_subnetwork_and_save(edge_index, predicted, real, output_file, plot_neighbors, gene_symbols):
    G = create_graph_from_edge_index(edge_index)
    remove_self_loops(G)
    # Include 'predicted', 'real', and their 1-hop neighbors in the subgraph
    if plot_neighbors:
        # Find the 1-hop neighbors of 'predicted' and 'real'
        predicted_neighbors = find_neighbors(G, predicted, hop=1)
        real_neighbors = find_neighbors(G, real, hop=1)
        subgraph_nodes = set(predicted + real + list(predicted_neighbors) + list(real_neighbors))
    else:
        subgraph_nodes = set(predicted + real) #+ list(predicted_neighbors) + list(real_neighbors))


    subgraph = G.subgraph(subgraph_nodes)
    node_mapping = dict(zip(range(len(gene_symbols)), gene_symbols))
    # Assign node colors based on their presence in 'predicted' and 'real'
    node_colors = []
    for node in subgraph.nodes():
        if node in predicted and node in real:
            node_colors.append('green')  # Node in both 'predicted' and 'real'
            subgraph.nodes[node]['category'] = 'both'
            subgraph.nodes[node]['Polygon'] = int(5)
            subgraph.nodes[node]['Label'] = node_mapping[node]
            subgraph.nodes[node]['label'] = node_mapping[node]
        elif node in predicted:
            node_colors.append('red')  # Node only in 'predicted'
            subgraph.nodes[node]['category'] = 'predicted'
            subgraph.nodes[node]['Polygon'] = int(3)
            subgraph.nodes[node]['Label'] = node_mapping[node]
            subgraph.nodes[node]['label'] = node_mapping[node]
        elif node in real:
            node_colors.append('orange')  # Node only in 'real'
            subgraph.nodes[node]['category'] = 'real'
            subgraph.nodes[node]['Polygon'] = int(4)
            subgraph.nodes[node]['Label'] = node_mapping[node]
            subgraph.nodes[node]['label'] = node_mapping[node]
        else:
            node_colors.append('grey')  # Node only a neighbor
            subgraph.nodes[node]['category'] = 'other'
            subgraph.nodes[node]['Polygon'] = int(2)
            subgraph.nodes[node]['Label'] = ""
            subgraph.nodes[node]['label'] = ""


    
    # subgraph = nx.relabel_nodes(subgraph, node_mapping)


    # for node in subgraph.nodes():
    #     if node not in predicted or node not in real:
    
    # for node in subgraph.nodes:
    #     subgraph.nodes[node].pop('label', None)


    # nx.write_gexf(subgraph, "{}.gexf".format(output_file))
    nx.write_gexf(subgraph, "{}.gexf".format(output_file), encoding='utf-8', version='1.2draft', prettyprint=True)

    # Draw the subgraph with nodes colored appropriately
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, with_labels=False, node_color=node_colors, font_color='white', node_size=20)
    # Save the plot as a PDF file
    plt.savefig("{}.pdf".format(output_file))
    plt.clf()  # 



# Process each cell line (for demonstration purposes, we'll just print them)
# for cell_line in cell_lines:
    
    
cell_line = 'A549_corrected_pos_emb'
    

print(f"Processing cell line: {cell_line}")

# Some params
use_backward_data = True
use_supervision = True #whether to use supervision loss
use_intervention_data = True #whether to use cycle loss
current_date = datetime.now() 
n_layers_nn = 1

global use_forward_data
use_forward_data = True


if cell_line.split('_')[0] in ['HA1E', 'HT29', 'A375', 'HELA']:
    use_forward_data = False

#Dataset
dataset = Dataset(
    forward_path=f"../../../data/processed/torch_data/chemical/real_lognorm/data_forward_{cell_line.split('_')[0]}.pt",
    backward_path=f"../../../data/processed/torch_data/chemical/real_lognorm/data_backward_{cell_line.split('_')[0]}.pt",
    splits_path=f"../../../data/processed/splits/chemical/{cell_line.split('_')[0]}/random/5fold/splits.pt"
)

edge_index = torch.load(f"../../../data/processed/torch_data/chemical/real_lognorm/edge_index_{cell_line.split('_')[0]}.pt")


#Only do this for best-performing model
performance = pd.read_csv('../../../results_metrics_aggregated_bme/perturbagen_pred/PDgrapher/within/chemical/val/{}_drugpred_within_best.csv'.format(cell_line.split('_')[0]))
ngnn = performance[performance['Set'] == 'Test']['GNN'].iloc[0]



paths = glob('../../../experiments_resubmission_bme/results/chemical/{}/n_gnn_{}_*'.format(cell_line, ngnn))

assert len(paths) == 1, "There should be only one best model"
path = paths[0]



outdir = osp.join(path, 'networks_of_predictions')
os.makedirs(outdir, exist_ok=True)
path_model = path

n_layers_gnn = int(path.split('/')[-1].split('_')[2])


fold = 1



for fold in range(1,6):
    outdir_i = osp.join(outdir, 'split_{}'.format(fold))
    os.makedirs(outdir_i, exist_ok=True)
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
            ) = dataset.get_dataloaders(num_workers = 20, batch_size = 1, shuffle = False)



    correct_interventions_list = []
    top_k_interventions_list = []
    perturbagen_name_list = []

    #Iterates through data and saves predicted targets
    for data in test_loader_backward:
        out  = model.perturbation_discovery(torch.concat([data.diseased.view(-1, 1).to(device), data.treated.view(-1, 1).to(device)], 1), data.batch.to(device), mutilate_mutations=data.mutations.to(device), threshold_input=thresholds)
    
        num_nodes = int(data.num_nodes / len(torch.unique(data.batch)))
        correct_interventions = tuple(zip(torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[0].tolist(), torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[1].tolist()))
        
        correct_interventions_list.append(np.array(correct_interventions)[:,1].tolist())
        top_k_interventions_list.append(torch.argsort(out.detach().cpu().view(-1, num_nodes)[0, :], descending=True).numpy().tolist())
        perturbagen_name_list.append(data.perturbagen_name)



    overlap_correct_vs_topk = []
    for i in range(len(top_k_interventions_list)):
        correct = correct_interventions_list[i]
        predicted = top_k_interventions_list[i][0:len(correct)]
        overlap = len(set(correct).intersection(set(predicted))) / len(correct)
        overlap_correct_vs_topk.append(overlap)

    indices_predicted_more_49percent_overlap = np.where(np.logical_and(np.array(overlap_correct_vs_topk)>0.49, np.array(overlap_correct_vs_topk)<1))[0]

    
    for plot_neighbors in [True]:
        for topn in [1]:
            for i in indices_predicted_more_49percent_overlap:
                output_file = osp.join(outdir_i, "subnetwork_plot_{}_top{}_neighbors_{}".format(perturbagen_name_list[i][0], topn, plot_neighbors))
                if len(correct_interventions_list[i]) > 2 or perturbagen_name_list[i][0]=='ethotoin':
                    plot_subnetwork_and_save(edge_index, top_k_interventions_list[i][0:topn*len(correct_interventions_list[i])], correct_interventions_list[i], output_file, plot_neighbors, data.gene_symbols[0])



