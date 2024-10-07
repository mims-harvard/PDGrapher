'''
Processes CosmicCLP_MutantExport.tsv (15th Sept 2022) to create a file with cell line: mutations 
Expert curated list obtained from:https://cancer.sanger.ac.uk/cell_lines/curation#  (18th March 2024)

'''

import pandas as pd
from collections import Counter
import os
import os.path as osp

#Creates out dir
outdir = '../../processed/cosmic'
os.makedirs(outdir, exist_ok=True)


cell_lines = ['A549', 'PC-3', 'MCF7', 'BT-20', 'MDA-MB-231', 'VCaP', 'A375', 'HT-29', 'ES-2', 'BICR6', 'YAPC', 'AGS', 'U251MG']



#Loads data 
data = pd.read_csv('../../raw/cosmic/2022-10-COSMIC/data/CosmicCLP_MutantExport.tsv', sep='\t', encoding="ISO-8859-1")
#Filter to include only the 5 cell lines of interest
mask = [e in cell_lines for e in data['Sample name']]
data = data[mask]



#Explore data
log_handle = open('log_stats.txt','w')
columns = ['Mutation Description', 'Mutation somatic status', 'Mutation verification status']
for cell_line in cell_lines:
	data_i = data[data['Sample name']==cell_line]
	log_handle.write('\nCELL LINE:\t{}\n'.format(cell_line))
	for column in columns:
		log_handle.write(column+'\n')
		log_handle.write(str(Counter(data_i[column])) +'\n\n')
	log_handle.write('Total genes mutated:\t{}\n\n\n\n'.format(len(data_i['Gene name'])))






#As agreed with Marinka, take an overlap of the 'verified' mutations and expert curated genes, for each cell line
	#Filter to keep only verified genes
data = data[data['Mutation verification status'] == 'Verified']


	#Filter to keep only curated genes
curated_genes = pd.read_csv('../../raw/cosmic/2022-10-COSMIC/data/expert_curated_genes_cosmic_2024.csv', sep='\t')['Genes'].tolist()
mask = [gene in curated_genes for gene in data['Gene name']]
data_curated = data[mask]
data_curated['Sample name'] = [e.replace('-','').upper() for e in data_curated['Sample name']]


#Save file
data_curated.to_csv(osp.join(outdir, 'CosmicCLP_MutantExport_only_verified_and_curated.csv'))
log_handle.write(str(data_curated['Sample name'].value_counts()))
log_handle.close()
