from GENIE3 import *
import networkx as nx
import numpy as np
import os
import time
root = 'data/raw/'
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cell_line', default=None, nargs='+')
parser.add_argument('--data_type', default='cmp', type=str) #cmp or gen for chemical or genetic
args = parser.parse_args()
data_type = args.data_type
celllines = args.cell_line

def run_GENIE():
    for cn in celllines:
        startTime=time.time()
        d = root + 'xpr_matrices/{}_xpr_matrix_{}_nonpertsubset.txt'.format(cn, data_type)
        data=loadtxt(d, skiprows=1)
        f=open(d)
        gene_names=f.readline()
        f.close()
        gene_names = gene_names.rstrip('\n').split('\t')
        VIM = GENIE3(data, ntrees=100, nthreads=20)
        outdir = root + 'processed/'
        os.makedirs(outdir, exist_ok=True)
        with open(outdir + "{}_{}_GENIE3arr.npy".format(cn, data_type), 'wb') as f:
            np.save(f, VIM)
        print("Cell Line: {} - ".format(cn)+str(time.time()-startTime))

def get_edgelist(): 
    for cn in celllines:
        outdir = root + 'processed/'
        startTime=time.time()
        d = root + 'xpr_matrices/{}_xpr_matrix_{}_nonpertsubset.txt'.format(cn, data_type)
        arr = np.load(outdir + "{}_{}_GENIE3arr.npy".format(cn, data_type))
        f=open(d)
        gene_names=f.readline()
        f.close()
        gene_names = gene_names.rstrip('\n').split('\t')
        
        reg_link_list=get_link_list(arr, gene_names=gene_names, file_name=outdir+"{}_{}_edgelist.txt".format(cn, data_type))
        g = nx.DiGraph((x,y,{'weight': v}) for (x, y, v) in reg_link_list)
        nx.write_weighted_edgelist(g, outdir+'{}_{}_nxEdgelist.txt'.format(cn, data_type), delimiter='    ')
        #print(reg_link_list[0], type(reg_link_list[0]))
        print("Cell Line: {} - ".format(cn)+str(time.time()-startTime))

    
if __name__ == "__main__":
    #startTime=time.time()
    run_GENIE()
    get_edgelist()
    #print(time.time()-startTime)
