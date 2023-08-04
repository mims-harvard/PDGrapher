#Generate a subset of the PPI
#First checks the overlap between genes in PPI (BRIOGRID) and genes in LINCS
#Removes all genes from PPI that are *not* in LINCS (no GE data available for them)


import networkx as nx
import csrgraph as cg
from collections import Counter
import pandas as pd
import os
import os.path as osp

path_edge_list = '../2020-12-PPI/processed/ppi_edgelist.txt'
log_handle = open('log_ppi_lincs_perturbed.txt', 'w')


#Loads dataset PPI (BIOGRID)
ppi = nx.read_edgelist(path_edge_list)
#Loads gene info LINCS
gene_info = pd.read_csv('../../lincs/2022-02-LINCS_Level3/data/geneinfo_beta.txt', sep="\t", low_memory=False)
#Loads perturbations metadata
pert_metadata = pd.read_csv(osp.join('../../lincs/processed/binarize_genewise_comparing_to_control', 'all_metadata.txt'), low_memory=False)
pert_metadata = pert_metadata[pert_metadata['pert_type'] != 'ctl_vector']
pert_genes = set(pert_metadata['cmap_name'].tolist())

pert_genes = dict()
for cell_line in list(set(pert_metadata['cell_iname'])):
    pert_genes[cell_line] = set(pert_metadata[pert_metadata['cell_iname']==cell_line]['cmap_name'].tolist())





log_handle.write('Overlap of genes from LINCS to PPI:{}/{}\n'.format(len(set(ppi.nodes()).intersection(set(gene_info['gene_symbol']))), len(gene_info)))


#Filter nodes from PPI to keep only the ones in LINCS
ppi = ppi.subgraph(gene_info['gene_symbol'].tolist())
log_handle.write('Keeping only PPI nodes that are in LINCS:{}\n'.format(ppi.number_of_nodes()))



#Filter nodes from PPI to keep only those that have perturbations available
for cell_line in list(set(pert_metadata['cell_iname'])):
    ppi_i = ppi.subgraph(pert_genes[cell_line])
    log_handle.write('Cell line\t{}\n'.format(cell_line))
    log_handle.write('Keeping only PPI nodes that are in LINCS and perturbed in cell line {}:{} nodes, {} edges\n'.format(cell_line, ppi_i.number_of_nodes(), ppi_i.number_of_edges()))


    #Checks how many connected components there are
    ccs = [len(c) for c in sorted(nx.connected_components(ppi_i), key=len, reverse=True)]
    log_handle.write('Number of connected componens:\t{}\n'.format(len(ccs)))
    # log_handle.write('Size of connected components:\n')
    # for c in ccs:
    # 	log_handle.write('{}\n'.format(str(c)))


    # Selects biggest connected component
    Gcc = sorted(nx.connected_components(ppi_i), key=len, reverse=True)
    ppi_i = ppi_i.subgraph(Gcc[0])
    log_handle.write('After keeping only biggest CC:\n')
    log_handle.write('stats: {} nodes, {} edges, {} density, {} diameter\n\n\n'.format(ppi_i.number_of_nodes(), ppi_i.number_of_edges(), nx.density(ppi_i), nx.diameter(ppi_i)))

    #Saves ppi
    outdir = '../processed'
    os.makedirs(outdir, exist_ok=True)
    ppi_f = osp.join(outdir, 'ppi_lincs_perturbed_edgelist_{}.txt'.format(cell_line))

    nx.write_edgelist(ppi_i, ppi_f, data=False) 



log_handle.close()



















