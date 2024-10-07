##Here we have code to replicate figures included in Figure 3 of main paper
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp
from glob import glob
import sys
sys.path.append('../')
from constants import *
import os
import numpy as np
from scipy import stats


outdir = 'processed'
os.makedirs(outdir, exist_ok = True)

base_path = '../../results_metrics_aggregated_bme/perturbagen_pred'


###############################################################################################################
## Panels distance between predicted genes and GT genes in the PPI (PDGrapher vs random)
###############################################################################################################



import torch
import networkx as nx
import pickle

outdir = 'distance_predicted_targets_and_gt'


all_lengths_ours_to_real = {}
all_lengths_random_to_real = {}
cell_line_path_dict = {}

for cell_line in ["A549", "A375", "BT20", "HELA", "HT29", "MCF7", "MDAMB231", "PC3", "VCAP"]:
    #Loads edge index 
    base_path = "../../data/processed/torch_data/chemical/real_lognorm"
    path_edge_index = osp.join(base_path, 'edge_index_{}.pt'.format(cell_line))
    edge_index = torch.load(path_edge_index)
    G = nx.Graph() 
    G.add_edges_from(tuple(zip(edge_index[0,:].tolist(), edge_index[1,:].tolist())))
    performance = pd.read_csv('../../results_metrics_aggregated_bme/perturbagen_pred/PDgrapher/within/chemical/val/{}_drugpred_within_best.csv'.format(cell_line))
    ngnn = performance[performance['Set'] == 'Test']['GNN'].iloc[0]
    model_path = glob('../../experiments_resubmission_bme/results/chemical/{}_corrected_pos_emb/n_gnn_{}*'.format(cell_line, ngnn))[0]
    random_path = '../../baselines/mechanistic/results/mechanistic/baseline_random_chemical/random/{}/random'.format(cell_line)
    our_predicted_interventions = pickle.load(open(osp.join(model_path, 'retrieved_interventions.pkl'), "rb"))
    our_real_interventions = pickle.load(open(osp.join(model_path, 'real_interventions.pkl'), "rb"))
    random_predicted_interventions = pickle.load(open(osp.join(random_path, 'retrieved_interventions.pkl'), "rb"))
    random_real_interventions = pickle.load(open(osp.join(random_path, 'real_interventions.pkl'), "rb"))
    for i in our_real_interventions.keys():
        assert len(our_predicted_interventions[i]) == len(our_real_interventions[i]), "predicted and real interventions should have the same length (our model)"
        assert len(random_predicted_interventions[i]) == len(random_real_interventions[i]), "predicted and real interventions should have the same length (random model)"
        assert len(random_predicted_interventions[i]) == len(our_predicted_interventions[i]), "our model and random model should have the same length"
    #Outpath
    outpath = osp.join(outdir, '/'.join(model_path.split('/')[-2:]))
    os.makedirs(outpath, exist_ok=True)
    cell_line_path_dict[cell_line] = outpath
    #Computes the distances if not pre-saved
    if not os.path.exists(osp.join(outpath, 'random_to_real.txt')):
        length = dict(nx.all_pairs_shortest_path_length(G))
        lengths_ours_to_real = []
        lengths_random_to_real = []
        for split_index in our_predicted_interventions.keys():
            pred_ours = our_predicted_interventions[split_index]
            pred_random = random_predicted_interventions[split_index]
            real = our_real_interventions[split_index]
            for i in range(len(pred_ours)):
                pred_ours_i = pred_ours[i]
                pred_random_i = pred_random[i]
                real_i = real[i]
                for j in range(len(real_i)):
                    for jj in range(len(real_i)):
                        lengths_ours_to_real.append(length[real_i[j]][pred_ours_i[jj]])
                        lengths_random_to_real.append(length[real_i[j]][pred_random_i[jj]])
        pd.DataFrame(lengths_random_to_real).to_csv(osp.join(outpath, 'random_to_real.txt'), index=False, header=None)
        pd.DataFrame(lengths_ours_to_real).to_csv(osp.join(outpath, 'ours_to_real.txt'), index=False, header=None)
    else:
        lengths_random_to_real = pd.read_csv(osp.join(outpath, 'random_to_real.txt'),header=None)[0].tolist()
        lengths_ours_to_real = pd.read_csv(osp.join(outpath, 'ours_to_real.txt'),  header=None)[0].tolist()
    all_lengths_ours_to_real[cell_line] = lengths_ours_to_real
    all_lengths_random_to_real[cell_line] = lengths_random_to_real



##STATS - computation-heavy
#Need to do this in 
def calculate_effect_size(U, n1, n2):
    # Calculate the rank-biserial correlation based on the U statistic
    return 1 - (2*U) / (n1*n2)

def bootstrap_ci(group1, group2, n_bootstrap=1000, ci=95):
    # Calculate the observed effect size
    U_observed, _ = stats.mannwhitneyu(group1, group2, alternative='less')
    observed_effect_size = calculate_effect_size(U_observed, len(group1), len(group2))
    # Generate bootstrap samples and compute effect sizes
    bootstrapped_effect_sizes = []
    for _ in range(n_bootstrap):
        # Resampling with replacement within each group
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        # Calculate the U statistic for the bootstrap sample
        U, _ = stats.mannwhitneyu(sample1, sample2, alternative='less')
        effect_size = calculate_effect_size(U, len(sample1), len(sample2))
        bootstrapped_effect_sizes.append(effect_size)
    # Compute the percentiles for the confidence interval
    lower_bound = np.percentile(bootstrapped_effect_sizes, (100-ci)/2)
    upper_bound = np.percentile(bootstrapped_effect_sizes, 100 - (100-ci)/2)
    return (lower_bound, upper_bound)



#Stats for individual cell lines
for cell_line in all_lengths_ours_to_real:
    lengths_ours_to_real = all_lengths_ours_to_real[cell_line]
    lengths_random_to_real = all_lengths_random_to_real[cell_line]
    log = open(osp.join(cell_line_path_dict[cell_line], 'log_avg_distance.txt'), 'w')
    log.write('avg distance in network\t{:.2f} ± {:.2f}\n'.format(np.mean(lengths_ours_to_real), np.std(lengths_ours_to_real)))
    t_stat, p_value = stats.ttest_ind(lengths_ours_to_real, lengths_random_to_real, alternative='less')
    log.write('Average distance for random:\t{}\n'.format(np.mean(lengths_random_to_real)))
    log.write('Average distance for PDGrapher:\t{}\n'.format(np.mean(lengths_ours_to_real)))
    log.write('P-value t-test of PDGrapher distances being smaller than random:\t{}\n'.format(p_value))
    t_stat, p_value = stats.mannwhitneyu(lengths_ours_to_real, lengths_random_to_real, alternative='less')
    log.write('Median distance for random:\t{}\n'.format(np.median(lengths_random_to_real)))
    log.write('Median distance for PDGrapher:\t{}\n'.format(np.median(lengths_ours_to_real)))
    log.write('Statistic MW-test of PDGrapher distances being smaller than random:\t{}\n'.format(t_stat))
    log.write('P-value MW-test of PDGrapher distances being smaller than random:\t{}\n'.format(p_value))
    n1 = len(lengths_ours_to_real)
    n2 = len(lengths_random_to_real)
    r = 1 - (2 * t_stat) / (n1 * n2)
    log.write('Effect size:\t{}\n'.format(r))
    confidence_interval = bootstrap_ci(lengths_ours_to_real, lengths_random_to_real)
    log.write('Confidence intervanl:\t{}'.format(confidence_interval))
    # #Let's test if my data comes from a normal distribution
    # norm_sample = np.random.normal(np.mean(legths_ours_to_real), np.std(legths_ours_to_real), size=len(legths_ours_to_real))
    # ks_statistic, p_value = stats.ks_2samp(legths_ours_to_real, norm_sample)
    log.close()


#Stats for all cell lines together
lengths_ours_to_real = np.concatenate([all_lengths_ours_to_real[e] for e in all_lengths_ours_to_real]).tolist()
lengths_random_to_real = np.concatenate([all_lengths_random_to_real[e] for e in all_lengths_random_to_real]).tolist()

log = open(osp.join(outpath, '../../log_avg_distance_alltogether.txt'), 'w')
log.write('avg distance in network\t{:.2f} ± {:.2f}\n'.format(np.mean(lengths_ours_to_real), np.std(lengths_ours_to_real)))
t_stat, p_value = stats.ttest_ind(lengths_ours_to_real, lengths_random_to_real, alternative='less')
log.write('Average distance for random:\t{}\n'.format(np.mean(lengths_random_to_real)))
log.write('Average distance for PDGrapher:\t{}\n'.format(np.mean(lengths_ours_to_real)))
log.write('P-value t-test of PDGrapher distances being smaller than random:\t{}\n'.format(p_value))
t_stat, p_value = stats.mannwhitneyu(lengths_ours_to_real, lengths_random_to_real, alternative='less')
log.write('Median distance for random:\t{}\n'.format(np.median(lengths_random_to_real)))
log.write('Median distance for PDGrapher:\t{}\n'.format(np.median(lengths_ours_to_real)))
log.write('Statistic MW-test of PDGrapher distances being smaller than random:\t{}\n'.format(t_stat))
log.write('P-value MW-test of PDGrapher distances being smaller than random:\t{}\n'.format(p_value))
n1 = len(lengths_ours_to_real)
n2 = len(lengths_random_to_real)
r = 1 - (2 * t_stat) / (n1 * n2)
log.write('Effect size:\t{}\n'.format(r))
confidence_interval = bootstrap_ci(lengths_ours_to_real, lengths_random_to_real)
log.write('Confidence intervanl:\t{}'.format(confidence_interval))
# #Let's test if my data comes from a normal distribution
# norm_sample = np.random.normal(np.mean(legths_ours_to_real), np.std(legths_ours_to_real), size=len(legths_ours_to_real))
# ks_statistic, p_value = stats.ks_2samp(legths_ours_to_real, norm_sample)
log.close()






