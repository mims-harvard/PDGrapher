import networkx as nx
import csrgraph as cg
from collections import Counter
import pandas as pd
import os
import os.path as osp
import numpy as np
import sys
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cell_line', default=None, nargs='+')
parser.add_argument('--data_type', default='cross', type=str)
args = parser.parse_args()
data_type = args.data_type
celllines = args.cell_line

#chemical
if data_type == "chemical":
    print("Chemical")
    gene_info = pd.read_csv('data/raw/lincs/geneinfo_beta.txt', sep="\t", low_memory=False)
    outdir = 'data/raw/grn/filter/'
    for i in celllines:
        log_handle = open(f'log_grn_all_genes_genetic_{i}.txt', 'w')
        path_edge_list = 'data/raw/grn/processed/{}_cmp_edgelist.txt'.format(i)
        print("Do: ", i)
        grn_0 = nx.read_edgelist(path_edge_list, nodetype=str, data=(('weight', float),))
        grn_1 = [(u, v, d['weight']) for u, v, d in grn_0.edges(data=True)]
        grn_2 = sorted(grn_1, key=lambda x: x[2])
        grn = nx.Graph()
        grn.add_weighted_edges_from(grn_2)
        wl = np.array([float(weight) for u, v, weight in grn_2])
        k = np.quantile(wl, 0.99)
        edges = np.array(grn_2)
        print(f"Do cutoff {k}...")
        edges_filter = edges[wl < k]
        grn.remove_edges_from(edges_filter)
        #Loads gene info LINCS
        log_handle.write('Cutoff:{}, removed edges {}\n'.format(k, len(edges_filter)))
        log_handle.write('Overlap of genes from LINCS to grn:{}/{}\n'.format(len(set(grn.nodes()).intersection(set(gene_info['gene_symbol']))), len(gene_info)))
        print("Flitering...")
        #Filter nodes from grn to keep only the ones in LINCS
        grn = grn.subgraph(gene_info['gene_symbol'].tolist())
        log_handle.write('Keeping only grn nodes that are in LINCS:{}\n'.format(grn.number_of_nodes()))
        print("Find connected components...")
        ccs = [len(c) for c in sorted(nx.connected_components(grn), key=len, reverse=True)]
        log_handle.write('Number of connected componens:\t{}\n'.format(len(ccs)))
        Gcc = sorted(nx.connected_components(grn), key=len, reverse=True)
        grn = grn.subgraph(Gcc[0])
        log_handle.write('After keeping only biggest CC:\n')
        log_handle.write('stats: {} nodes, {} edges, {} density, {} diameter\n\n\n'.format(grn.number_of_nodes(), grn.number_of_edges(), nx.density(grn), nx.diameter(grn)))
        print("Done and save...")
        grn_f = osp.join(outdir, f'{i}_cmp_edgelist_filtered.txt')
        nx.write_edgelist(grn, grn_f, data=False)

elif data_type == "genetic":
    gene_info = pd.read_csv('data/raw/lincs/geneinfo_beta.txt', sep="\t", low_memory=False)
    outdir = 'data/raw/grn/filter/'
    for i in celllines:
        log_handle = open(f'log_grn_all_genes_genetic_{i}.txt', 'w')
        path_edge_list = 'data/raw/grn/processed/{}_gen_edgelist.txt'.format(i)
        print("Do: ", i)
        grn_0 = nx.read_edgelist(path_edge_list, nodetype=str, data=(('weight', float),))
        grn_1 = [(u, v, d['weight']) for u, v, d in grn_0.edges(data=True)]
        grn_2 = sorted(grn_1, key=lambda x: x[2])
        grn = nx.Graph()
        grn.add_weighted_edges_from(grn_2)
        wl = np.array([float(weight) for u, v, weight in grn_2])
        k = np.quantile(wl, 0.99)
        edges = np.array(grn_2)
        print(f"Do cutoff {k}...")
        edges_filter = edges[wl < k]
        grn.remove_edges_from(edges_filter)
        #Loads gene info LINCS
        log_handle.write('Cutoff:{}, removed edges {}\n'.format(k, len(edges_filter)))
        log_handle.write('Overlap of genes from LINCS to grn:{}/{}\n'.format(len(set(grn.nodes()).intersection(set(gene_info['gene_symbol']))), len(gene_info)))
        print("Flitering...")
        #Filter nodes from grn to keep only the ones in LINCS
        grn = grn.subgraph(gene_info['gene_symbol'].tolist())
        log_handle.write('Keeping only grn nodes that are in LINCS:{}\n'.format(grn.number_of_nodes()))
        print("Find connected components...")
        ccs = [len(c) for c in sorted(nx.connected_components(grn), key=len, reverse=True)]
        log_handle.write('Number of connected componens:\t{}\n'.format(len(ccs)))
        Gcc = sorted(nx.connected_components(grn), key=len, reverse=True)
        grn = grn.subgraph(Gcc[0])
        log_handle.write('After keeping only biggest CC:\n')
        log_handle.write('stats: {} nodes, {} edges, {} density, {} diameter\n\n\n'.format(grn.number_of_nodes(), grn.number_of_edges(), nx.density(grn), nx.diameter(grn)))
        print("Done and save...")
        grn_f = osp.join(outdir, f'{i}_gen_edgelist_filtered.txt')
        nx.write_edgelist(grn, grn_f, data=False)
