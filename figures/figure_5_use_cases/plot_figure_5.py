


import pandas as pd
import os.path as osp
import numpy as np
import os
import re
import torch
import sys
sys.path.append('../')
from constants import *
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import pickle

from matplotlib import font_manager
font_dirs = ['/home/gonzag46/.fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42  # Output Type 42 (TrueType), editable in Illustrator


outdir = 'processed'
os.makedirs(outdir, exist_ok=True)

specific_K_values = [1, 10, 20, 30, 40, 50, 100,200, 300, 400, 500, 1000, 2000, 4000, 6000, 8000, 10000, 10716]
cancer_types = {'A549': 'lung', 'MCF7': 'breast', 'MDAMB231': 'breast', 'BT20':'breast', 'PC3': 'prostate', 'VCAP': 'prostate'}


#Loads all stats
cell_lines = ["A549", "BT20", "MCF7", "MDAMB231", "PC3", "VCAP"]


recall_dict = {e:{} for e in cell_lines}
highlight_points_dict = {e:{} for e in cell_lines}
highlight_points_full_dict = {e:{} for e in cell_lines}



for cell_line in cell_lines:


    #################################################
    #Loads data

    #Reads in approved drugs
    drugs = pd.read_csv('../../data/processed/nci/drugs_and_targets.csv', sep='\t')

    #Selects approved drugs for cell line use case
    drugs_approved = drugs[drugs['cell_line'] == cell_line.split('_')[0]]['drug'].tolist()

    #Loads torch datasets
    forward_dataset = torch.load('../../data/processed/torch_data/chemical/real_lognorm/data_forward_{}.pt'.format(cell_line.split('_')[0]))
    backward_dataset = torch.load('../../data/processed/torch_data/chemical/real_lognorm/data_backward_{}.pt'.format(cell_line.split('_')[0]))
    splits = torch.load('../../data/processed/splits/chemical/{}/random/5fold/splits.pt'.format(cell_line.split('_')[0]))
    gene_symbols = forward_dataset[0].gene_symbols



    #################################################
    #Loads random and PDGrapher predicted gene list

    ####Aggregated ranking
    with open('{}_corrected_pos_emb/aggregated_ranking.pickle'.format(cell_line), 'rb') as f:
        aggregated_ranking = pickle.load(f)
    
    ranked_list_of_genes = [gene_symbols[e] for e in aggregated_ranking]

    with open('{}_corrected_pos_emb/aggregated_ranking_random.pickle'.format(cell_line), 'rb') as f:
        random_aggregated_ranking = pickle.load(f)

    random_ranked_list_of_genes = [gene_symbols[e] for e in random_aggregated_ranking]





    #################################################
    #Gets info on approved drugs

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
    genes_in_approved_drugs_dict = {}
    for d in approved_drugs_not_in_dataset:
        for e in drugs[drugs['drug']==d]['targets']:
            genes_in_approved_drugs += e.split(',')
            if d not in genes_in_approved_drugs_dict:
                genes_in_approved_drugs_dict[d] = []
            genes_in_approved_drugs_dict[d] += e.split(',')

    for drug in genes_in_approved_drugs_dict:
        genes_in_approved_drugs_dict[drug] = list(set(genes_in_approved_drugs_dict[drug]))


    genes_in_approved_drugs = set(genes_in_approved_drugs)


    # Filter out genes that are not in ranked_list_of_genes
    genes_in_approved_drugs_dict = {
        drug: [gene for gene in target_genes if gene in ranked_list_of_genes]
        for drug, target_genes in genes_in_approved_drugs_dict.items()
    }

    genes_in_approved_drugs_dict = {k: v for k, v in genes_in_approved_drugs_dict.items() if len(v) > 0}





    #################################################
    #Computes recall for each K (and highlight points - where the drugs have achieved recall != 0, and recall == 1)


    # Initialize a list to store recall values at each K
    recall_at_k = {}
    recall_at_k_random = {}
    # Initialize a list to store when each drug reaches recall = 1
    highlight_points = []
    highlight_points_full = []

    # Iterate over all values of K from 1 to the length of ranked_list_of_genes
    for k in range(1, len(ranked_list_of_genes) + 1):
        recalls = []
        recalls_random = []
        for drug, target_genes in genes_in_approved_drugs_dict.items():
            #Recall PDGrapher
            recovered_targets = set(ranked_list_of_genes[:k]) & set(target_genes)
            recall = len(recovered_targets) / len(target_genes)
            #Recall random
            recovered_targets = set(random_ranked_list_of_genes[:k]) & set(target_genes)
            recall_random = len(recovered_targets) / len(target_genes)
            if k in specific_K_values:
                recalls.append(recall)
                recalls_random.append(recall_random)
            # Check if the drug reaches recall != 0 at this K
            if recall != 0  and drug not in [x[0] for x in highlight_points]:
                highlight_points.append((drug, k, recall))
            # Check if the drug reaches recall = 1 at this K
            if recall == 1  and drug not in [x[0] for x in highlight_points_full]:
                highlight_points_full.append((drug, k, recall))
        # Calculate average recall at this K
        average_recall = np.mean(recalls)
        average_recall_random = np.mean(recalls_random)
        if k in specific_K_values:
            recall_at_k[k] = average_recall
            recall_at_k_random[k] = average_recall_random
        
        
    recall_dict[cell_line] = {'PDGrapher':recall_at_k, 'Reference': recall_at_k_random}
    highlight_points_dict[cell_line] = highlight_points
    highlight_points_full_dict[cell_line] = highlight_points_full










##################################################################################################
#PANEL AVERAGE RECALL ACROSS CELL LINES (top 0 to 1000)
##################################################################################################

outdir = 'processed'
os.makedirs(outdir, exist_ok=True)

# Extracting data
methods = ['Reference', 'PDGrapher']
ks = [1, 10, 50, 100, 200, 300, 400, 500, 1000]

# Initialize a dictionary to hold average recalls across cell lines
average_recalls = {method: [] for method in methods}

# Calculate average recall for each method at each K value
for method in methods:
    for k_index in ks:
        recalls = []
        for cell_line in recall_dict.keys():
            recalls.append(recall_dict[cell_line][method][k_index])
        average_recalls[method].append(np.mean(recalls))

# Define the color palette
palette = {
    'PDGrapher': '#04C4D9',
    'Reference': '#333333',
}



with sns.plotting_context(plotting_context):
    # Plotting with Seaborn
    fig, ax = plt.subplots(figsize=(3.543, 2.19))
    for method in methods:
        # Plot the line for average recall
        sns.lineplot(x=ks, y=average_recalls[method], color=palette[method], marker='o', label=method, ax=ax, markersize=4)
        # Plot individual points for each cell line
        for cell_line in recall_dict.keys():
            cell_line_recalls = [recall_dict[cell_line][method][k_index] for k_index in ks]
            sns.scatterplot(x=ks, y=cell_line_recalls, color=palette[method], alpha=0.5, ax=ax, s=10)
    # Ensure the legend is correctly formatted without duplication
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(methods)], labels[:len(methods)], loc='upper right', frameon=False, title=None)
    plt.legend(frameon=False, title=None, bbox_to_anchor=(0.3, 1))
    # Formatting the plot
    plt.xlabel('Predicted gene rank')
    plt.ylabel('Recall')
    plt.grid(False)
    sns.despine()
    plt.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.18)
    # Show the plot
    plt.savefig(osp.join(outdir, 'recall_targets_approved_drugs.pdf'))
    plt.close()
    
# import pdb; pdb.set_trace()
    
with open(osp.join(outdir, 'recall_dict.pkl'), 'wb') as f:
    pickle.dump(recall_dict, f)
    
    
    
    
# ##################################################################################################
# ##Panel recall vs K for a single cell line - highlighting the points where recall drug targets != 0 
# ##################################################################################################

# for cell_line in cell_lines:
        
#     #Outdir
#     outdir = cell_line + '_corrected_pos_emb'
#     os.makedirs(outdir, exist_ok = True)


#     specific_K_values = [1, 10, 20, 30, 40, 50, 100, 1000, 2000, 4000, 6000, 8000, 10000, 10716]


#     # Convert to DataFrame for Seaborn
#     df = pd.DataFrame(recall_dict[cell_line]['PDGrapher'], columns=['K', 'Average Recall'])
#     df = df[[e in specific_K_values for e in df['K']]]



#     # Create a dictionary to aggregate drugs by their x position (k value)
#     # Create a dictionary to aggregate drugs by their x position (k value)
#     highlight_dict = defaultdict(list)
#     for drug, k, recall in highlight_points_dict[cell_line]:
#         highlight_dict[k].append(drug.capitalize())
            
            
#     # Define a list of colors that match the aesthetic of your figures
#     color_list = [
#         '#04C4D9',   # Cyan-like blue
#         '#F28E2B',   # Orange
#         '#E15759',   # Red
#         '#76B7B2',   # Teal
#         '#59A14F',   # Green
#         '#EDC948',   # Yellow
#         '#B07AA1',   # Purple
#         '#FF9DA7',   # Pinkish Red
#         '#9C755F',   # Brown
#         '#BAB0AC',   # Gray
#         '#4E79A7',    # Dark Blue
#         '#2B7A0B',
#         '#FFA07A'
#     ]

#     # Generate a dynamic color palette based on the highlight_dict keys
#     color_palette = {k: color_list[i % len(color_list)] for i, k in enumerate(highlight_dict.keys())}




#     with sns.plotting_context(plotting_context):
#         # Plot using Seaborn
#         plt.figure(figsize=(7.35, 3.4))
#         sns.lineplot(data=df, x='K', y='Average Recall', color='#04C4D9', marker='')
#         # Plotting
#         for k, drugs in highlight_dict.items():
#             label = ',\n'.join(drugs)
#             color = color_palette.get(k, '#000000')  # Default to black if k not in palette
#             plt.axvline(x=k, color=color, linestyle='--', label=label)
#         # Plotting details
#         sns.despine()
#         plt.xlabel('K (Number of top predicted genes considered)')
#         plt.ylabel('Average recall across approved drugs')
#         plt.title('')
#         plt.legend(loc='best', frameon=False, title=None, bbox_to_anchor=(1.2, 1))
#         plt.grid(False)
#         plt.subplots_adjust(left=0.08, right=0.8, top=0.9, bottom=0.1)
#         plt.savefig(osp.join(outdir, 'cumulative_recovery_alldrugs_nonzero_overlap.pdf'))
#         plt.close()
#         plt.close()
        
        
        
    
    
    
    
    

# ##################################################################################################
# ##Panel recall vs K for a single cell line - highlighting the points where recall drug targets == 1 
# ##################################################################################################
# for cell_line in cell_lines:
#     #Outdir
#     outdir = cell_line + '_corrected_pos_emb'
#     os.makedirs(outdir, exist_ok = True)


#     specific_K_values = [1, 10, 20, 30, 40, 50, 100, 1000, 2000, 4000, 6000, 8000, 10000, 10716]


#     # Convert to DataFrame for Seaborn
#     df = pd.DataFrame(recall_dict[cell_line]['PDGrapher'], columns=['K', 'Average Recall'])
#     df = df[[e in specific_K_values for e in df['K']]]



#     # Create a dictionary to aggregate drugs by their x position (k value)
#     # Create a dictionary to aggregate drugs by their x position (k value)
#     highlight_dict = defaultdict(list)
#     for drug, k, recall in highlight_points_full_dict[cell_line]:
#         highlight_dict[k].append(drug.capitalize())
            
            
#     # Define a list of colors that match the aesthetic of your figures
#     color_list = [
#         '#04C4D9',   # Cyan-like blue
#         '#F28E2B',   # Orange
#         '#E15759',   # Red
#         '#76B7B2',   # Teal
#         '#59A14F',   # Green
#         '#EDC948',   # Yellow
#         '#B07AA1',   # Purple
#         '#FF9DA7',   # Pinkish Red
#         '#9C755F',   # Brown
#         '#BAB0AC',   # Gray
#         '#4E79A7',    # Dark Blue
#         '#2B7A0B',
#         '#FFA07A'
#     ]

#     # Generate a dynamic color palette based on the highlight_dict keys
#     color_palette = {k: color_list[i % len(color_list)] for i, k in enumerate(highlight_dict.keys())}




#     with sns.plotting_context(plotting_context):
#         # Plot using Seaborn
#         plt.figure(figsize=(7.35, 3.4))
#         sns.lineplot(data=df, x='K', y='Average Recall', color='#04C4D9', marker='')
#         # Plotting
#         for k, drugs in highlight_dict.items():
#             label = ',\n'.join(drugs)
#             color = color_palette.get(k, '#000000')  # Default to black if k not in palette
#             plt.axvline(x=k, color=color, linestyle='--', label=label)
#         # Plotting details
#         sns.despine()
#         plt.xlabel('K (Number of top predicted genes considered)')
#         plt.ylabel('Average recall across approved drugs')
#         plt.title('')
#         plt.legend(loc='best', frameon=False, title=None, bbox_to_anchor=(1.2, 1))
#         plt.grid(False)
#         plt.subplots_adjust(left=0.08, right=0.8, top=0.9, bottom=0.1)
#         plt.savefig(osp.join(outdir, 'cumulative_recovery_alldrugs_full_overlap.pdf'))
#         plt.close()
#         plt.close()