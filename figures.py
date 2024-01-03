#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:20:46 2024

@author: tanzira
"""
#%%all imports
import pandas as pd
import numpy as np
import seaborn as sns
import anndata
import scanpy as sc
# from scqcut import scQcut_monolith
import matplotlib.pyplot as plt
from scipy.io import loadmat
fig_dir = '/home/tanzira/HDD/single_cell_decomposition/Results/Figures'
#%%Read aces data
data_raw  = loadmat('Data/ACES_Data/ACESExpr.mat')['data']
aces_p_type = loadmat('Data/ACES_Data/ACESLabel.mat')['label']
entrez_id = loadmat('Data/ACES_Data/ACES_EntrezIds.mat')['entrez_ids']
aces_data = pd.DataFrame(data_raw)
aces_data.columns = entrez_id.reshape(-1)


''' Seperating the studies for leave one study out cross validation. '''

cv_train_idx_file = 'Data/ACES_Data/CVIndTrain200.txt'
train_cv_idx = pd.read_csv(cv_train_idx_file, header = None, sep = ' ')
d_map = pd.DataFrame(0, 
                     index = range(train_cv_idx.shape[0]),
                     columns = range(train_cv_idx.shape[1]))

for col in train_cv_idx.columns:
    idx_other = train_cv_idx[col][train_cv_idx[col] > 0]-1
    idx = np.setdiff1d(range(train_cv_idx.shape[0]), idx_other)
    d_map.loc[idx, col] = 1
#%%plot ACES dataset AUC for different clustering algorithm
clustering_algorithms = ['Bulk', 'Kmeans', 'Hierarchical' , 'Spectral']

file = 'Results/ACES_individual_dataset_accuracy_scores.csv'
results_all_dataset = pd.read_csv(file, index_col =[0, 1, 2, 3, 4, 5])

mean_results = results_all_dataset.groupby(level = [0, 1, 2, 3, 5]).mean()
# fig, ax = plt.subplots(4, 3, figsize = (24, 16), sharex=True, sharey=True)
# ax = ax.flat


#%%Sort dataset by size
dataset_names = results_all_dataset.index.get_level_values('dataset').unique()
dataset_size = pd.DataFrame(0, index = dataset_names,
                            columns = ['total', 'negative', 'postive'])

for i, name in enumerate(mean_results.index.get_level_values('dataset').unique()):
    test_index = d_map[d_map[i] == 1].index
    data_test, Y = aces_data.iloc[test_index], aces_p_type[test_index].ravel()
    bad, good, total = np.sum(Y), int(data_test.shape[0]-np.sum(Y)), data_test.shape[0]
    print(name, '\t\t', total, '\t\t', good, '\t\t', bad)
    dataset_size.loc[name, :] = total, good, bad
dataset_size['idx'] = np.arange(0, len(dataset_size))
dataset_size.sort_values(by = 'total', ascending = False, inplace = True)

#%%Bar plot for different cluster type
fig, ax = plt.subplots(2, 1, figsize = (10, 12))
auc_each_dataset = {}

for i, name in enumerate(dataset_size.index):
    auc_and_index = pd.DataFrame(0, index=clustering_algorithms, columns = ['n_cluster', 'AUC'])
    for cluster_name in clustering_algorithms:
        t1 = mean_results.groupby(['dataset', 'metric', 'cluster_type', 'n_cluster']).max().loc[name, 'AUC', cluster_name].idxmax()
        t2 = mean_results.groupby(['dataset', 'metric', 'cluster_type', 'n_cluster']).max().loc[name, 'AUC', cluster_name].max()
        auc_and_index.loc[cluster_name] = [t1.iloc[0], t2.iloc[0]]
    auc_each_dataset[name] = auc_and_index
auc_each_dataset = pd.concat(auc_each_dataset, names = ['dataset', 'cluster_type'])
bar = auc_each_dataset.unstack(level = 1).plot.bar(y = 'AUC', ax = ax[0])
bar = auc_each_dataset.unstack(level = 1).plot.bar(y = 'n_cluster', ax = ax[1], legend = False)
ax[0].legend(bbox_to_anchor = [1, 1], loc = 'upper left')
ax[0].set_ylim([0.3, 0.85])
ax[0].set_ylabel('AUC')
ax[0].set_title('Best AUC for individual datasets')
ax[1].set_title('n_cluster for best result')
ax[1].set_ylabel('# cluster')
plt.subplots_adjust(wspace = 0.05, hspace = 0.3)
plt.savefig(fig_dir + '/best_auc_individual_aces_different_clustering.eps', bbox_inches='tight')
plt.show()
#%%Hierarchical cluster plot analysis ACES
mean_results = results_all_dataset.groupby(level = [0, 1, 2, 3, 5]).mean()
fig, ax = plt.subplots(4, 3, figsize = (16, 12), sharex = True, sharey = True)
ax = ax.flat
cluster_type = 'Hierarchical'
acc_metric = 'Kappa'
print('Dataset \t\t total \t\t Good \t\tBad')
for i, (name, idx) in enumerate(zip(dataset_size.index, dataset_size.idx)):
    test_index = d_map[d_map[idx] == 1].index
    data_test, Y = aces_data.iloc[test_index], aces_p_type[test_index].ravel()
    bad, good, total = np.sum(Y), int(data_test.shape[0]-np.sum(Y)), data_test.shape[0]
    print(name, '\t\t', total, '\t\t', good, '\t\t', bad)
    t2 = mean_results.loc[name, 'RF', cluster_type, :, acc_metric].plot(ax = ax[i],
                                            title = '{0}: {1} -/+ : {2}/{3}'.format(name,total, good, bad),
                                                                   marker = '.', legend = False)
    ax[i].axhline(mean_results.loc[name, 'RF', 'Bulk', :, acc_metric].values, color = 'r')
plt.subplots_adjust(wspace = 0.2, hspace = 0.3)
ax[2].legend(['decomp', 'bulk'], bbox_to_anchor =(1, 1), loc = 'upper left')
plt.suptitle("{0} with {1} clustering for different k".format(acc_metric, cluster_type))
plt.savefig(fig_dir + '/{0}_individual_aces_different_k_{1}_clustering.png'.format(acc_metric, cluster_type),\
            bbox_inches='tight')
plt.show()