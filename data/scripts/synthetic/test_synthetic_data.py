

import torch
from create_synthetic_with_confounders import create_dag_from_ppi, topological_sort, reindex_dag_nodes, get_parents_for_each_node, simulate_observational_data, intervene_on_gene, generate_weights_and_biases
import networkx as nx
import numpy as np
import pandas as pd
from time import time
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, levene, bartlett
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from create_synthetic_with_confounders import reindex_perturbed_nodes, generate_confounder_biases
from utils import check_similarities
import os
import os.path as osp
outdir = '../../processed/synthetic'
os.makedirs(outdir, exist_ok = True)


############################################################
############################################################
#Generate synthetic data
############################################################
############################################################

tic  = time()
edge_index = torch.load('../../processed/torch_data/chemical/real_lognorm/edge_index_A549.pt')
real_observational_data =  torch.load('../../processed/torch_data/chemical/real_lognorm/data_forward_A549.pt')
real_interventional_data =  torch.load('../../processed/torch_data/chemical/real_lognorm/data_backward_A549.pt')
num_nodes = len(edge_index.unique())
edge_index = edge_index.transpose(1,0).tolist()
toc = time()
print('Time loading data: {:.3f} secs'.format(toc - tic))

tic = time()
ppi_network = nx.Graph()
ppi_network.add_edges_from(edge_index)
toc = time()
print('Time creating PPI network: {:.3f} secs'.format(toc - tic))


tic = time()
# Convert the PPI network to a DAG
dag = create_dag_from_ppi(ppi_network)
sorted_nodes = topological_sort(dag.edges(), num_nodes)
toc = time()
print('Time converting PPI to DAG: {:.3f} secs'.format(toc - tic))


tic = time()
#Reindex so that it's consistent with hierarchical ordering
dag = reindex_dag_nodes(dag, sorted_nodes)
parents_dict = get_parents_for_each_node(dag)
toc = time()
print('Time relabeling nodes: {:.3f} secs'.format(toc - tic))

weights, biases = generate_weights_and_biases(parents_dict)
cofounder_biases = generate_confounder_biases(parents_dict, confounder_probability=0, mean=0.0, std=0)

tic = time()
#Simulate observational data
observational_data = simulate_observational_data(parents_dict, 100, weights, biases, cofounder_biases)

toc = time()
print('Time simulating observational data: {:.3f} secs'.format(toc - tic))





############################################################
############################################################
#Proof of concept
############################################################
############################################################
#Real observational data (let's load it and compare distributions - if not the same as real data, then I need to tweak the causal mechanisms)
real_observational_data = np.stack([e.diseased for e in real_observational_data])
real_interventional_data = np.stack([e.treated for e in real_interventional_data])

print(f'Simulated observational data: {observational_data.shape}')
print(f'Real observational data: {real_observational_data.shape}')

print('Subsampling real data')
real_observational_data = real_observational_data[np.random.randint(0, len(real_observational_data), 100)]
real_interventional_data = real_interventional_data[np.random.randint(0, len(real_interventional_data), 100)]
print(f'Real observational data: {real_observational_data.shape}')

check_similarities(real_observational_data, observational_data, 'poc_observational')














############################################################
############################################################
#Interventional data creation and analyses
############################################################
############################################################
#Lets get unique perturbages from interventional data
backward_data = torch.load('../../processed/torch_data/chemical/real_lognorm/data_backward_A549.pt')

unique_perturbagens = set()
for d in backward_data:
    pert_indices = tuple(torch.where(d.intervention)[0].tolist()) + tuple([d.perturbagen_name])
    unique_perturbagens.add(pert_indices)


n_replicates = 1 #average number of replicates per drug
cofounder_biases = generate_confounder_biases(parents_dict, confounder_probability=0, mean=0.0, std=0)
observational_data = simulate_observational_data(parents_dict, len(unique_perturbagens) * n_replicates, weights, biases, cofounder_biases)

#generates interventional data for different sets of genes intervened on (perturbagens)
interventional_data_list = []
perturbagen_names_list = []
perturbed_gene_indices_list = []
unique_perturbagens = list(unique_perturbagens)

for i in range(len(unique_perturbagens)):
    perturbagen = unique_perturbagens[i]
    gene_indices = reindex_perturbed_nodes(list(perturbagen)[:-1], sorted_nodes)
    intervention_values = len(gene_indices) * [0]
    interventional_data_list.append(intervene_on_gene(observational_data[i*n_replicates:i*n_replicates + n_replicates, :], parents_dict=parents_dict, gene_indices = gene_indices, intervention_values = intervention_values, weights=weights, biases=biases))
    perturbagen_names_list += [perturbagen[-1]] * n_replicates
    perturbed_gene_indices_list += [gene_indices] * n_replicates
    print(f'{i+1}/{len(unique_perturbagens)}')


interventional_data = np.vstack(interventional_data_list)



#Similarities of final simulated datasets and real datasets
forward_data = torch.load('../../processed/torch_data/chemical/real_lognorm/data_forward_A549.pt')
real_observational_data = np.stack([e.diseased for e in forward_data])
real_observational_data = real_observational_data[np.random.randint(0, len(real_observational_data), len(observational_data))]
real_interventional_data = np.stack([e.treated for e in backward_data])
real_interventional_data = real_interventional_data[np.random.randint(0, len(real_interventional_data), len(interventional_data))]
check_similarities(real_observational_data, observational_data, 'observational')
check_similarities(real_interventional_data, interventional_data, 'interventional')





#####################################################################################################################################################################################
#####################################################################################################################################################################################
#Observational and interventional data with missing components
#The only difference here is that the edge_index has some missing edges
#####################################################################################################################################################################################
#####################################################################################################################################################################################

##Now that I have observational and interventional data, let's pack them up in a torch data object
data_list = []
from torch_geometric.data import Data
for i in range(len(observational_data.shape[0])):
    mutations = torch.zeros(observational_data.shape[1])
    binary_indicator_perturbation = np.zeros(observational_data.shape[1])
    binary_indicator_perturbation[perturbed_gene_indices_list[i]] = 1
    data_list.append(Data(diseased = torch.Tensor(observational_data[i,:]), perturbagen_name = perturbagen_names_list[i], intervention=torch.Tensor(binary_indicator_perturbation), 
                        treated = torch.Tensor(interventional_data[i,:]), mutations = mutations, num_nodes = observational_data.shape[1]))
    


torch.save(data_list, osp.join(outdir, 'data_backward_synthetic.pt'))

edge_list = list(dag.edges)
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
torch.save(edge_index, osp.join(outdir, 'edge_index_synthetic.pt'))


##Modified graphs Apply various edge removal strategies
from create_synthetic_with_confounders import remove_random_edges, remove_high_betweenness_edges, remove_bridge_edges

# dag_random_edges_removed = remove_random_edges(dag.copy(), remove_fraction=0.1)
# dag_high_betweenness_edges_removed = remove_high_betweenness_edges(dag.copy(), fraction=0.1)

for fraction in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
    dag_bridge_edges_removed = remove_bridge_edges(dag.copy(), fraction=fraction)
    edge_list = list(dag_bridge_edges_removed.edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    torch.save(edge_index, osp.join(outdir, 'edge_index_synthetic_bridge_removed_frac_{}.pt'.format(fraction)))





#####################################################################################################################################################################################
#####################################################################################################################################################################################
#Observational and interventional data with presence of confounders
#Here the graph is the same but the signal is generated with presence of some extra nodes
#####################################################################################################################################################################################
#####################################################################################################################################################################################
fraction = 0.2
cofounder_biases = generate_confounder_biases(parents_dict, confounder_probability=fraction, mean=0.0, std=0.1)


#Lets get unique perturbages from interventional data
backward_data = torch.load('../../processed/torch_data/chemical/real_lognorm/data_backward_A549.pt')

unique_perturbagens = set()
for d in backward_data:
    pert_indices = tuple(torch.where(d.intervention)[0].tolist()) + tuple([d.perturbagen_name])
    unique_perturbagens.add(pert_indices)


n_replicates = 1 #average number of replicates per drug
observational_data = simulate_observational_data(parents_dict, len(unique_perturbagens) * n_replicates, weights, biases, cofounder_biases)

#generates interventional data for different sets of genes intervened on (perturbagens)
interventional_data_list = []
perturbagen_names_list = []
perturbed_gene_indices_list = []
unique_perturbagens = list(unique_perturbagens)

for i in range(len(unique_perturbagens)):
    perturbagen = unique_perturbagens[i]
    gene_indices = reindex_perturbed_nodes(list(perturbagen)[:-1], sorted_nodes)
    intervention_values = len(gene_indices) * [0]
    interventional_data_list.append(intervene_on_gene(observational_data[i*n_replicates:i*n_replicates + n_replicates, :], parents_dict=parents_dict, gene_indices = gene_indices, intervention_values = intervention_values, weights=weights, biases=biases))
    perturbagen_names_list += [perturbagen[-1]] * n_replicates
    perturbed_gene_indices_list += [gene_indices] * n_replicates
    print(f'{i+1}/{len(unique_perturbagens)}')


interventional_data = np.vstack(interventional_data_list)



#Similarities of final simulated datasets and real datasets
forward_data = torch.load('../../processed/torch_data/chemical/real_lognorm/data_forward_A549.pt')
real_observational_data = np.stack([e.diseased for e in forward_data])
real_observational_data = real_observational_data[np.random.randint(0, len(real_observational_data), len(observational_data))]
real_interventional_data = np.stack([e.treated for e in backward_data])
real_interventional_data = real_interventional_data[np.random.randint(0, len(real_interventional_data), len(interventional_data))]
check_similarities(real_observational_data, observational_data, 'observational')
check_similarities(real_interventional_data, interventional_data, 'interventional')



##Saving dataset
##Now that I have observational and interventional data, let's pack them up in a torch data object
data_list = []
for i in range(len(observational_data.shape[0])):
    mutations = torch.zeros(observational_data.shape[1])
    binary_indicator_perturbation = np.zeros(observational_data.shape[1])
    binary_indicator_perturbation[perturbed_gene_indices_list[i]] = 1
    data_list.append(Data(diseased = torch.Tensor(observational_data[i,:]), perturbagen_name = perturbagen_names_list[i], intervention=torch.Tensor(binary_indicator_perturbation), 
                        treated = torch.Tensor(interventional_data[i,:]), mutations = mutations, num_nodes = observational_data.shape[1]))
    


torch.save(data_list, osp.join(outdir, 'data_backward_synthetic_confounder_frac_{}.pt'.format(fraction)))














# dag_random_edges_removed_parents_dict = get_parents_for_each_node(dag_random_edges_removed)
# dag_high_betweenness_edges_removed_parents_dict = get_parents_for_each_node(dag_high_betweenness_edges_removed)
# dag_bridge_edges_removed_parents_dict = get_parents_for_each_node(dag_bridge_edges_removed)


# data_random_edges = simulate_observational_data(dag_random_edges_removed, num_samples = 100)
# data_high_betweenness_edges = simulate_observational_data(dag_high_betweenness_edges_removed_parents_dict, num_samples = 100)
# data_bridge_edges = simulate_observational_data(dag_bridge_edges_removed_parents_dict, num_samples = 100)



















# ##Fit a Gaussian distribution and compare mean and covariate
# from scipy.stats import multivariate_normal

# # Fit multivariate normal to the data
# mean1, cov1 = np.mean(real_observational_data, axis=0), np.cov(real_observational_data, rowvar=False)
# mean2, cov2 = np.mean(observational_data, axis=0), np.cov(observational_data, rowvar=False)


# plt.figure(figsize=(8, 8))

# plt.scatter(mean1, mean2, alpha=0.6)
# plt.title('Scatter Plot of Vector 1 vs Vector 2')
# plt.xlabel('Real observational')
# plt.ylabel('Simulated observational')
# plt.grid(True)
# plt.show()
# plt.savefig('scatter_plot_mean_gauss.png')
# plt.close()


# plt.figure(figsize=(12, 6))

# plt.hist(mean1, bins=50, alpha=0.5, label='Real observational')
# plt.hist(mean2, bins=50, alpha=0.5, label='Simulated observational')

# plt.title('Histogram of Vectors')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.legend()
# plt.grid(True)
# plt.savefig('histogram_gauss.png')
# plt.close()

# import seaborn as sns
# import numpy as np

# # Creating a combined array for heatmap
# data = np.vstack([mean1, mean2])

# plt.figure(figsize=(10, 2))
# sns.heatmap(data, annot=False, cmap='viridis', cbar=True, yticklabels=['Real observational', 'Simulated observational'])
# plt.title('Heatmap of Vectors')
# plt.xlabel('Index')
# plt.savefig('heatmap_gauss.png')
# plt.close()


############################################################
############################################################
#Observational data analyses
############################################################
############################################################
##Let's check the distribution similarity
#Simple PCA

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming X1 and X2 are your datasets
pca = PCA(n_components=2)

X = np.vstack([real_observational_data, observational_data])
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:100, 0], X_pca[:100, 1], label='Real observational data')
plt.scatter(X_pca[100:, 0], X_pca[100:, 1], label='Simulated observational data')

# plt.scatter(X1_pca[:, 0], X1_pca[:, 1], label='Real observational data')
# plt.scatter(X2_pca[:, 0], X2_pca[:, 1], label='Simulated observational data')
plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Datasets')
plt.savefig('scatter_plot_pca.png')
plt.close()



## t-SNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:100, 0], X_tsne[:100, 1], label='Real observational data')
plt.scatter(X_tsne[100:, 0], X_tsne[100:, 1], label='Simulated observational data')
plt.legend()
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE of Datasets')
plt.savefig('scatter_plot_tsne.png')
plt.close()


