'''
Export PPI with all genes in LINCS (those that overlap)
'''

#Generate a subset of the PPI
#First checks the overlap between genes in PPI (BRIOGRID) and genes in LINCS
#Removes all genes from PPI that are *not* in LINCS (no GE data available for them)


import networkx as nx
import csrgraph as cg
from collections import Counter
import pandas as pd
import os
import os.path as osp

path_edge_list = '../../raw/ppi/2022-03-PPI/processed/ppi_edgelist.txt'
log_handle = open('log_ppi_all_genes.txt', 'w')


#Loads dataset PPI (BIOGRID)
ppi = nx.read_edgelist(path_edge_list)
#Loads gene info LINCS
gene_info = pd.read_csv('../../raw/lincs/2022-02-LINCS_Level3/data/geneinfo_beta.txt', sep="\t", low_memory=False)



log_handle.write('Overlap of genes from LINCS to PPI:{}/{}\n'.format(len(set(ppi.nodes()).intersection(set(gene_info['gene_symbol']))), len(gene_info)))

#Filter nodes from PPI to keep only the ones in LINCS
ppi = ppi.subgraph(gene_info['gene_symbol'].tolist())
log_handle.write('Keeping only PPI nodes that are in LINCS:{}\n'.format(ppi.number_of_nodes()))

ccs = [len(c) for c in sorted(nx.connected_components(ppi), key=len, reverse=True)]
log_handle.write('Number of connected componens:\t{}\n'.format(len(ccs)))
Gcc = sorted(nx.connected_components(ppi), key=len, reverse=True)
ppi = ppi.subgraph(Gcc[0])
log_handle.write('After keeping only biggest CC:\n')
log_handle.write('stats: {} nodes, {} edges, {} density, {} diameter\n\n\n'.format(ppi.number_of_nodes(), ppi.number_of_edges(), nx.density(ppi), nx.diameter(ppi)))

#Saves ppi
outdir = '../../processed/ppi'
os.makedirs(outdir, exist_ok=True)
ppi_f = osp.join(outdir, 'ppi_all_genes_edgelist.txt')

nx.write_edgelist(ppi, ppi_f, data=False) 




log_handle.close()



















