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
import networkx as nx
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



#Argsparse
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_features', default=2, type=int, help='Number of features per node')
parser.add_argument('--embedding_layer_dim', default=16, type=int, help='Dimensionality of embedding layer')
parser.add_argument('--dim_gnn', default=16, type=int, help='Hidden dimensionality of GNN layers')
parser.add_argument('--positional_features_dims', default=16, type=int, help='Dimensionality of positional features for node embeddings')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--gpu_index', default=4, type=int, help='GPU index')
parser.add_argument('--log_train', default=0, type=int, help='Whether to log training set behavior during training' )
parser.add_argument('--log_test', default=0, type=int, help='Whether to log test set behavior during training' )
args = parser.parse_args()


for cell_line, model_paths in zip(['A549', 'MCF7'], [[
                                                        # 'baseline_gcn__3_gcnlayers_1_nnlayers_trainr_1_traini_1_usef_1_useb_1_use_interv_0_usesup_1_0.01mult_split_random__2023_3_3__1_43_37',
                                                      'baseline_gcn__1_gcnlayers_1_nnlayers_trainr_1_traini_1_usef_1_useb_1_use_interv_1_usesup_1_0.01mult_split_random__2023_2_28__11_37_40',
                                                      # 'baseline_gcn__3_gcnlayers_1_nnlayers_trainr_1_traini_1_usef_1_useb_1_use_interv_1_usesup_0_0.01mult_split_random__2023_3_3__8_48_39'
                                                      ],
                                                      [
                                                      # 'baseline_gcn__2_gcnlayers_1_nnlayers_trainr_1_traini_1_usef_1_useb_1_use_interv_1_usesup_0_0.01mult_split_random__2023_3_3__13_54_50',
                                                      'baseline_gcn__1_gcnlayers_1_nnlayers_trainr_1_traini_1_usef_1_useb_1_use_interv_1_usesup_1_0.01mult_split_random__2023_3_1__1_55_27',
                                                      # 'baseline_gcn__3_gcnlayers_1_nnlayers_trainr_1_traini_1_usef_1_useb_1_use_interv_0_usesup_1_0.01mult_split_random__2023_3_5__0_16_31'
                                                      ]]  ):
    
    for model_path in model_paths:
        for split_index in range(1,6):

            ##Model to compute the rankings for
            path_model = '../../../../baselines/experiments/chemical/{}/5_folds/{}/split_{}'.format(cell_line, model_path, split_index)
            args.dataset_type = path_model.split('/')[6]
            args.dataset_name = path_model.split('/')[7]
            args.n_layers_gnn = int(path_model.split('/')[9].split('_')[3])
            args.n_layers_nn = int(path_model.split('/')[9].split('_')[5])
            args.train_response = int(path_model.split('/')[9].split('_')[8])
            args.use_forward_data = int(path_model.split('/')[9].split('_')[12])
            args.use_backward_data = int(path_model.split('/')[9].split('_')[14])
            args.train_intervention =int(path_model.split('/')[9].split('_')[10])
            args.use_interv_data = int(path_model.split('/')[9].split('_')[17])
            args.use_supervision = int(path_model.split('/')[9].split('_')[19])
            args.splits_type = path_model.split('/')[9].split('_')[22]
            args.nfolds = path_model.split('/')[8][:-1].replace('_','')
            args.splits_path = osp.join('../../../../baselines/source/splits/',args.dataset_type, args.dataset_name, args.splits_type, args.nfolds, 'splits.pt')


            outdir_i = osp.join(outdir, cell_line, model_path, 'split_{}'.format(split_index))
            os.makedirs(outdir_i, exist_ok=True)

            #Dataset
            base_path = "../../../../data/rep-learning-approach-3-all-genes/processed/chemical/real_lognorm"
            path_edge_index = osp.join(base_path, 'edge_index_{}.pt'.format(args.dataset_name))
            edge_index = torch.load(path_edge_index)
            path = osp.join(base_path, 'data_backward_{}.pt'.format(args.dataset_name))
            dataset_backward = torch.load(path)


            #Device and extra args
            device = torch.device('cuda', args.gpu_index)
            args.num_vars= dataset_backward[0].num_nodes
            args.out_channels = 1

            #Model instantiation and loading saved model
            model_2 = GCNModelInterventionDiscovery(args, out_fun='perturbation', edge_index=edge_index.to(device)).to(device)
            checkpoint = torch.load(os.path.join(path_model, 'model_2.pt'))
            model_2.load_state_dict(checkpoint['model_state_dict'])
            model_2.build_dictionary_node_to_edge_index_position()

            #Loads splits
            splits = torch.load(args.splits_path)
            split = splits[split_index]
            dataset = [dataset_backward[i] for i in split['test_index_backward']]
            train_dataset_backward = [dataset_backward[i] for i in split['train_index_backward']]
            val_dataset_backward = [dataset_backward[i] for i in split['val_index_backward']]
            test_loader_backward = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

            #Select perturbagens for which our predictions are best
            # perturbagens_in_train = set([e.perturbagen_name for e in train_dataset_backward] + [e.perturbagen_name for e in val_dataset_backward])
            # perturbagens_in_test = set([e.perturbagen_name for e in dataset] )
            # perturbagens_only_in_test = list(perturbagens_in_test - perturbagens_in_train)
            # dataset = [dataset_backward[i] if dataset_backward[i].perturbagen_name in perturbagens_only_in_test else '' for i in split['test_index_backward']]
            # dataset = set(dataset); dataset.remove(''); dataset = list(dataset)
            # test_loader_backward = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)


            diseased_threshold, treated_threshold = get_threshold_diseased(train_dataset_backward), get_threshold_treated(train_dataset_backward),
            thresholds = {'diseased': diseased_threshold.to(device), 'treated': treated_threshold.to(device)}

            correct_interventions_list = []
            top_k_interventions_list = []
            perturbagen_name_list = []

            #Iterates through data and saves predicted targets
            for data in test_loader_backward:
                out  = model_2(torch.concat([data.diseased.view(-1,1).to(device), data.treated.view(-1, 1).to(device)], 1), model_2.edge_index, data.batch.to(device), mutilate_graph=True, mutilate_mutations = data.mutations.to(device), threshold_input=thresholds)
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
                            plot_subnetwork_and_save(edge_index, top_k_interventions_list[i][0:topn*len(correct_interventions_list[i])], correct_interventions_list[i], output_file, plot_neighbors, dataset[0].gene_symbols)






































# #Keep in the test set only those that are not in the training set
# perturbagens_in_train = set([e.perturbagen_name for e in train_dataset_backward] + [e.perturbagen_name for e in val_dataset_backward])
# perturbagens_in_test = set([e.perturbagen_name for e in dataset] )
# perturbagens_only_in_test = list(perturbagens_in_test - perturbagens_in_train)
# dataset = [dataset_backward[i] if dataset_backward[i].perturbagen_name in perturbagens_only_in_test else '' for i in split['test_index_backward']]
# dataset = set(dataset); dataset.remove(''); dataset = list(dataset)
# test_loader_backward = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)


# diseased_threshold, treated_threshold = get_threshold_diseased(train_dataset_backward), get_threshold_treated(train_dataset_backward),
# thresholds = {'diseased': diseased_threshold.to(device), 'treated': treated_threshold.to(device)}

# correct_interventions_list = []
# top_k_interventions_list = []
# perturbagen_name_list = []

# #Iterates through data and saves predicted targets
# for data in test_loader_backward:
#   out  = model_2(torch.concat([data.diseased.view(-1,1).to(device), data.treated.view(-1, 1).to(device)], 1), model_2.edge_index, data.batch.to(device), mutilate_graph=True, mutilate_mutations = data.mutations.to(device), threshold_input=thresholds)
#   num_nodes = int(data.num_nodes / len(torch.unique(data.batch)))
#   correct_interventions = tuple(zip(torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[0].tolist(), torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[1].tolist()))
#   correct_interventions_list.append(np.array(correct_interventions)[:,1].tolist())
#   top_k_interventions_list.append(torch.argsort(out.detach().cpu().view(-1, num_nodes)[0, :], descending=True).numpy().tolist())
#   perturbagen_name_list.append(data.perturbagen_name)


# for plot_neighbors in [False]:
#   for topn in [1,2,3,4,5,10]:
#       for i in range(len(top_k_interventions_list)):
#           output_file = osp.join(outdir_i, "subnetwork_plot_{}_top{}_neighbors_{}.pdf".format(perturbagen_name_list[i][0], topn, plot_neighbors))
#           plot_subnetwork_and_save(edge_index, top_k_interventions_list[i][0:topn*len(correct_interventions_list[i])], correct_interventions_list[i], output_file, plot_neighbors)















