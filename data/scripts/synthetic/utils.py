
import numpy as np
import pandas as pd
from time import time
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, levene, bartlett
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import os.path as osp



def check_similarities(real_data, simulated_data, name, ppi_network, dag, outdir):
    ############################################################
    ############################################################
    #Dataset properties and statistical comparisons
    ############################################################
    ############################################################
    log = open(osp.join(outdir,'{}_log.txt'.format(name)), 'w')
    # Number of genes in generated vs real samples
    num_genes_real = real_data.shape[1]
    num_genes_simulated = simulated_data.shape[1]
    log.write(f"Number of Genes: Real = {num_genes_real}, Simulated = {num_genes_simulated}\n")
    log.write(f"Number of Edges: Real = {ppi_network.number_of_edges()}, Simulated = {dag.number_of_edges()}\n")
    log.write(f"Density: Real = {nx.density(ppi_network)}, Simulated = {nx.density(dag)}\n")
    log.write(f"Number of connected components: Real = {nx.number_connected_components(ppi_network)}, Simulated = {nx.number_weakly_connected_components(dag)}\n")
    # Degree Distribution (Mean Degree)
    degrees_real = [d for n, d in ppi_network.degree()]
    degrees_simulated = [d for n, d in dag.degree()]
    mean_degree_real = np.mean(degrees_real)
    mean_degree_simulated = np.mean(degrees_simulated)
    log.write(f"Mean Degree: Real = {mean_degree_real}, Simulated = {mean_degree_simulated}\n")
    # Clustering Coefficient
    clustering_real = nx.average_clustering(ppi_network)
    clustering_simulated = nx.average_clustering(dag.to_undirected())
    log.write(f"Clustering Coefficient: Real = {clustering_real}, Simulated = {clustering_simulated}\n")
    # Flatten the data for some comparisons
    real_data_flattened = real_data.flatten()
    simulated_data_flattened = simulated_data.flatten()
    # Global mean value of gene expression (average across samples) [t-test or MWU-test]
    global_mean_real = np.mean(real_data_flattened)
    global_mean_simulated = np.mean(simulated_data_flattened)
    # Kolmogorov-Smirnov Test
    ks_stat, ks_p_value = stats.ks_2samp(real_data_flattened, simulated_data_flattened)
    log.write(f"KS Statistic of gene expression values: {ks_stat}, p-value: {ks_p_value}\n\n")
    # Earth Mover's Distance
    emd_distance = stats.wasserstein_distance(real_data_flattened, simulated_data_flattened)
    log.write(f"Earth Mover's Distance of gene expression values: {emd_distance}\n\n")
    # Perform t-test
    t_test_result = ttest_ind(real_data_flattened, simulated_data_flattened)
    # Perform Mann-Whitney U test (if data is not normally distributed)
    mannwhitney_test_result_mean = mannwhitneyu(real_data_flattened, simulated_data_flattened)
    log.write(f"Global Mean Value: Real = {global_mean_real}, Simulated = {global_mean_simulated}\n")
    log.write(f"T-test for Global Mean Value: {t_test_result}\n")
    log.write(f"Mann-Whitney U test for Global Mean Value: {mannwhitney_test_result_mean}\n\n")
    # Global median value of gene expression (average across samples) [MWU-test]
    global_median_real = np.median(real_data_flattened)
    global_median_simulated = np.median(simulated_data_flattened)
    # Perform Mann-Whitney U test
    mannwhitney_test_result_median = mannwhitneyu(real_data_flattened, simulated_data_flattened)
    log.write(f"Global Median Value: Real = {global_median_real}, Simulated = {global_median_simulated}\n")
    log.write(f"Mann-Whitney U test for Global Median Value: {mannwhitney_test_result_median}\n\n")
    # Global standard deviation of gene expression (average across samples) [Levene's test or Bartlett's test]
    global_std_real = np.std(real_data_flattened)
    global_std_simulated = np.std(simulated_data_flattened)
    # Perform Levene's test
    levene_test_result = levene(real_data_flattened, simulated_data_flattened)
    # Perform Bartlett's test
    bartlett_test_result = bartlett(real_data_flattened, simulated_data_flattened)
    log.write(f"Global Standard Deviation: Real = {global_std_real}, Simulated = {global_std_simulated}\n")
    log.write(f"Levene's test for Global Standard Deviation: {levene_test_result}\n")
    log.write(f"Bartlett's test for Global Standard Deviation: {bartlett_test_result}\n")
    # Minimum and maximum expression value
    global_min_real = np.min(real_data_flattened)
    global_min_simulated = np.min(simulated_data_flattened)
    global_max_real = np.max(real_data_flattened)
    global_max_simulated = np.max(simulated_data_flattened)
    log.write(f"Global Min Value: Real = {global_min_real}, Simulated = {global_min_simulated}\n")
    log.write(f"Global Max Value: Real = {global_max_real}, Simulated = {global_max_simulated}\n")
    log.close()
    #################################
    #Histograms
    #################################
    # Calculate the required statistics
    # Gene-level mean
    real_gene_means = np.mean(real_data, axis=0)
    simulated_gene_means = np.mean(simulated_data, axis=0)
    # Gene-level standard deviation
    real_gene_stds = np.std(real_data, axis=0)
    simulated_gene_stds = np.std(simulated_data, axis=0)
    # Expression values (flattened)
    real_data_flattened = real_data.flatten()
    simulated_data_flattened = simulated_data.flatten()
    # Sample mean
    real_sample_means = np.mean(real_data, axis=1)
    simulated_sample_means = np.mean(simulated_data, axis=1)
    # Sample median
    real_sample_medians = np.median(real_data, axis=1)
    simulated_sample_medians = np.median(simulated_data, axis=1)
    # Plot and save the histograms
    plt.figure(figsize=(10, 6))
    sns.histplot(real_gene_means, color='blue', label='Real Gene Means', kde=True, stat="density")
    sns.histplot(simulated_gene_means, color='orange', label='Simulated Gene Means', kde=True, stat="density")
    plt.legend()
    plt.title('Histogram of Gene-Level Means')
    plt.xlabel('Gene-Level Mean')
    plt.ylabel('Density')
    plt.savefig(osp.join(outdir, f'{name}_histogram_gene_level_means.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(real_gene_stds, color='blue', label='Real Gene Stds', kde=True, stat="density")
    sns.histplot(simulated_gene_stds, color='orange', label='Simulated Gene Stds', kde=True, stat="density")
    plt.legend()
    plt.title('Histogram of Gene-Level Standard Deviations')
    plt.xlabel('Gene-Level Standard Deviation')
    plt.ylabel('Density')
    plt.savefig(osp.join(outdir,f'{name}_histogram_gene_level_stds.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(real_data_flattened, color='blue', label='Real Expression Values', kde=True, stat="density")
    sns.histplot(simulated_data_flattened, color='orange', label='Simulated Expression Values', kde=True, stat="density")
    plt.legend()
    plt.title('Histogram of Expression Values')
    plt.xlabel('Expression Value')
    plt.ylabel('Density')
    plt.savefig(osp.join(outdir,f'{name}_histogram_expression_values.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(real_sample_means, color='blue', label='Real Sample Means', kde=True, stat="density")
    sns.histplot(simulated_sample_means, color='orange', label='Simulated Sample Means', kde=True, stat="density")
    plt.legend()
    plt.title('Histogram of Sample Means')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.savefig(f'{name}_histogram_sample_means.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(real_sample_medians, color='blue', label='Real Sample Medians', kde=True, stat="density")
    sns.histplot(simulated_sample_medians, color='orange', label='Simulated Sample Medians', kde=True, stat="density")
    plt.legend()
    plt.title('Histogram of Sample Medians')
    plt.xlabel('Sample Median')
    plt.ylabel('Density')
    plt.savefig(osp.join(outdir,f'{name}_histogram_sample_medians.png'))
    plt.close()



def check_similarities_only_data(real_data, simulated_data, name, outdir):
    ############################################################
    ############################################################
    #Dataset properties and statistical comparisons
    ############################################################
    ############################################################
    log = open(osp.join(outdir,'{}_log_only_data.txt'.format(name)), 'w')
    # Flatten the data for some comparisons
    real_data_flattened = real_data.flatten()
    simulated_data_flattened = simulated_data.flatten()
    # Global mean value of gene expression (average across samples) [t-test or MWU-test]
    global_mean_real = np.mean(real_data_flattened)
    global_mean_simulated = np.mean(simulated_data_flattened)
    # Kolmogorov-Smirnov Test
    ks_stat, ks_p_value = stats.ks_2samp(real_data_flattened, simulated_data_flattened)
    log.write(f"KS Statistic of gene expression values: {ks_stat}, p-value: {ks_p_value}\n\n")
    # Earth Mover's Distance
    emd_distance = stats.wasserstein_distance(real_data_flattened, simulated_data_flattened)
    log.write(f"Earth Mover's Distance of gene expression values: {emd_distance}\n\n")
    # Perform t-test
    t_test_result = ttest_ind(real_data_flattened, simulated_data_flattened)
    # Perform Mann-Whitney U test (if data is not normally distributed)
    mannwhitney_test_result_mean = mannwhitneyu(real_data_flattened, simulated_data_flattened)
    log.write(f"Global Mean Value: Real = {global_mean_real}, Simulated = {global_mean_simulated}\n")
    # log.write(f"T-test for Global Mean Value: {t_test_result}\n")
    # log.write(f"Mann-Whitney U test for Global Mean Value: {mannwhitney_test_result_mean}\n\n")
    # Global median value of gene expression (average across samples) [MWU-test]
    global_median_real = np.median(real_data_flattened)
    global_median_simulated = np.median(simulated_data_flattened)
    # Perform Mann-Whitney U test
    mannwhitney_test_result_median = mannwhitneyu(real_data_flattened, simulated_data_flattened)
    log.write(f"Global Median Value: Real = {global_median_real}, Simulated = {global_median_simulated}\n")
    # log.write(f"Mann-Whitney U test for Global Median Value: {mannwhitney_test_result_median}\n\n")
    # Global standard deviation of gene expression (average across samples) [Levene's test or Bartlett's test]
    global_std_real = np.std(real_data_flattened)
    global_std_simulated = np.std(simulated_data_flattened)
    # Perform Levene's test
    levene_test_result = levene(real_data_flattened, simulated_data_flattened)
    # Perform Bartlett's test
    bartlett_test_result = bartlett(real_data_flattened, simulated_data_flattened)
    log.write(f"Global Standard Deviation: Real = {global_std_real}, Simulated = {global_std_simulated}\n")
    # log.write(f"Levene's test for Global Standard Deviation: {levene_test_result}\n")
    # log.write(f"Bartlett's test for Global Standard Deviation: {bartlett_test_result}\n")
    # Minimum and maximum expression value
    global_min_real = np.min(real_data_flattened)
    global_min_simulated = np.min(simulated_data_flattened)
    global_max_real = np.max(real_data_flattened)
    global_max_simulated = np.max(simulated_data_flattened)
    log.write(f"Global Min Value: Real = {global_min_real}, Simulated = {global_min_simulated}\n")
    log.write(f"Global Max Value: Real = {global_max_real}, Simulated = {global_max_simulated}\n")
    log.close()
    #################################
    #Histograms
    #################################
    # Calculate the required statistics
    # Gene-level mean
    real_gene_means = np.mean(real_data, axis=0)
    simulated_gene_means = np.mean(simulated_data, axis=0)
    # Gene-level standard deviation
    real_gene_stds = np.std(real_data, axis=0)
    simulated_gene_stds = np.std(simulated_data, axis=0)
    # Expression values (flattened)
    real_data_flattened = real_data.flatten()
    simulated_data_flattened = simulated_data.flatten()
    # Sample mean
    real_sample_means = np.mean(real_data, axis=1)
    simulated_sample_means = np.mean(simulated_data, axis=1)
    # Sample median
    real_sample_medians = np.median(real_data, axis=1)
    simulated_sample_medians = np.median(simulated_data, axis=1)
    # Plot and save the histograms
    plt.figure(figsize=(10, 6))
    sns.histplot(real_gene_means, color='blue', label='Real Gene Means', kde=True, stat="density")
    sns.histplot(simulated_gene_means, color='orange', label='Simulated Gene Means', kde=True, stat="density")
    plt.legend()
    plt.title('Histogram of Gene-Level Means')
    plt.xlabel('Gene-Level Mean')
    plt.ylabel('Density')
    plt.savefig(osp.join(outdir, f'{name}_histogram_gene_level_means.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(real_gene_stds, color='blue', label='Real Gene Stds', kde=True, stat="density")
    sns.histplot(simulated_gene_stds, color='orange', label='Simulated Gene Stds', kde=True, stat="density")
    plt.legend()
    plt.title('Histogram of Gene-Level Standard Deviations')
    plt.xlabel('Gene-Level Standard Deviation')
    plt.ylabel('Density')
    plt.savefig(osp.join(outdir,f'{name}_histogram_gene_level_stds.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(real_data_flattened, color='blue', label='Real Expression Values', kde=True, stat="density")
    sns.histplot(simulated_data_flattened, color='orange', label='Simulated Expression Values', kde=True, stat="density")
    plt.legend()
    plt.title('Histogram of Expression Values')
    plt.xlabel('Expression Value')
    plt.ylabel('Density')
    plt.savefig(osp.join(outdir,f'{name}_histogram_expression_values.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(real_sample_means, color='blue', label='Real Sample Means', kde=True, stat="density")
    sns.histplot(simulated_sample_means, color='orange', label='Simulated Sample Means', kde=True, stat="density")
    plt.legend()
    plt.title('Histogram of Sample Means')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.savefig(f'{name}_histogram_sample_means.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(real_sample_medians, color='blue', label='Real Sample Medians', kde=True, stat="density")
    sns.histplot(simulated_sample_medians, color='orange', label='Simulated Sample Medians', kde=True, stat="density")
    plt.legend()
    plt.title('Histogram of Sample Medians')
    plt.xlabel('Sample Median')
    plt.ylabel('Density')
    plt.savefig(osp.join(outdir,f'{name}_histogram_sample_medians.png'))
    plt.close()


def check_similarities_only_graphs(name, ppi_network, dag, outdir):
    ############################################################
    ############################################################
    #Dataset properties and statistical comparisons
    ############################################################
    ############################################################
    log = open(osp.join(outdir,'{}_log_only_graph.txt'.format(name)), 'w')
    # Number of genes in generated vs real samples
    num_genes_real = ppi_network.number_of_nodes()
    num_genes_simulated = dag.number_of_nodes()
    log.write(f"Number of Genes: Real = {num_genes_real}, Simulated = {num_genes_simulated}\n")
    log.write(f"Number of Edges: Real = {ppi_network.number_of_edges()}, Simulated = {dag.number_of_edges()}\n")
    log.write(f"Density: Real = {nx.density(ppi_network)}, Simulated = {nx.density(dag)}\n")
    log.write(f"Number of connected components: Real = {nx.number_connected_components(ppi_network)}, Simulated = {nx.number_weakly_connected_components(dag)}\n")
    # Degree Distribution (Mean Degree)
    degrees_real = [d for n, d in ppi_network.degree()]
    degrees_simulated = [d for n, d in dag.degree()]
    mean_degree_real = np.mean(degrees_real)
    mean_degree_simulated = np.mean(degrees_simulated)
    log.write(f"Mean Degree: Real = {mean_degree_real}, Simulated = {mean_degree_simulated}\n")
    # Clustering Coefficient
    clustering_real = nx.average_clustering(ppi_network)
    clustering_simulated = nx.average_clustering(dag.to_undirected())
    log.write(f"Clustering Coefficient: Real = {clustering_real}, Simulated = {clustering_simulated}\n")
    log.close()