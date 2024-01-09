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

#%%Read single cell dataset
'''Normalized files'''
countfile = 'DataSubtypesBRC/E-GEOD-75688-quantification-raw-files/E-GEOD-75688.aggregated_filtered_counts.mtx'
normcountfile = 'DataSubtypesBRC/E-GEOD-75688-normalised-files/E-GEOD-75688.aggregated_filtered_normalised_counts.mtx'
cfile = 'DataSubtypesBRC/E-GEOD-75688-normalised-files/E-GEOD-75688.aggregated_filtered_normalised_counts.mtx_cols'
rfile = 'DataSubtypesBRC/E-GEOD-75688-normalised-files/E-GEOD-75688.aggregated_filtered_normalised_counts.mtx_rows'
exp_design_file = 'DataSubtypesBRC/ExpDesign-E-GEOD-75688.tsv'

adata = anndata.read_mtx(countfile).T

obs = pd.read_csv(cfile, header = None).squeeze()
var = pd.read_csv(rfile, header = None, sep = '\t', index_col = 0, names = ['gene'])
exp_design = pd.read_csv(exp_design_file, sep = '\t',
                         usecols=(0, 3, 5, 7, 9, 11, 13), index_col = 0)
exp_design = exp_design.loc[obs.values, :]
exp_design.columns = ['individual', 'organism_part', 'sampling_site', 
                      'disease', 'histology', 'single_cell_id']

#Sample information file
sample_info = pd.read_csv('Data/GSE75688_final_sample_information.txt',
                          sep = '\t', index_col=0)

adata.obs = exp_design
adata.var = var
adata.layers['normalized'] = anndata.read_mtx(normcountfile).T.X

#Keep only the samples from both file
common_samples = np.intersect1d(adata.obs.single_cell_id , sample_info.index)
adata = adata[adata.obs.single_cell_id.isin(common_samples)]
sample_info = sample_info.loc[common_samples]

patient_type = {'BC01':'Non-metastatic', 'BC02':'Non-metastatic', 'BC03':'Metastatic', 'BC03LN':'Metastatic', 
                'BC04':'Non-metastatic','BC05':'Non-metastatic', 'BC06':'Non-metastatic',
                'BC07':'Metastatic', 'BC07LN':'Metastatic','BC08':'Non-metastatic', 
                'BC09':'Non-metastatic', 'BC09_Re':'Non-metastatic', 'BC10':'Non-metastatic', 'BC11':'Non-metastatic'}


equivalent_subtypes = {'estrogen-receptor positive breast cancer':'LuminalA',
       'HER2 positive breast carcinoma':'HER2',
       'estrogen-receptor positive and HER2 positive breast cancer':'LuminalB',
       'triple-negative breast cancer':'TNBC/Basal-like'}

adata.obs['subtype'] = adata.obs['histology'].map(equivalent_subtypes)
adata.obs['patient_type'] = adata.obs['individual'].map(patient_type)
adata.obs['tumor_status'] = adata.obs['single_cell_id'].map(sample_info['index'])
adata.obs['cell_type'] = adata.obs['single_cell_id'].map(sample_info['index2'])
adata.obs['cell_type_name'] = adata.obs['single_cell_id'].map(sample_info['index3'])


#%%Read ACES, NKI data
'''Reading ACES data'''
aces_raw  = loadmat( 'Data/ACES_Data/ACESExpr.mat')['data']
aces_ptype = loadmat('Data/ACES_Data/ACESLabel.mat')['label']
aces_entrez_id = loadmat('Data/ACES_Data/ACES_EntrezIds.mat')['entrez_ids']
aces_data = pd.DataFrame(aces_raw)
aces_data.columns = aces_entrez_id.reshape(-1)
aces_subtype = np.loadtxt('Data/ACES_Data/ACES_Subtype.txt', dtype = float).astype(int)
    
'''Reading NKI data'''
nki_raw = loadmat('Data/NKI_Data/vijver.mat')['vijver']
nki_ptype = loadmat('Data/NKI_Data/VijverLabel.mat')['label']
nki_entrez_id = loadmat('Data/NKI_Data/vijver_gene_list.mat')['vijver_gene_list']
nki_data = pd.DataFrame(nki_raw)
nki_data.columns = nki_entrez_id.reshape(-1)
nki_subtype = np.loadtxt('Data/NKI_Data/NKI_subtype.txt', dtype = int)

study_names=['Desmedt', 'Hatzis', 'Ivshina', 'Loi', 'Miller', 'Minn',
          'Pawitan', 'Schmidt', 'Symmans', 'WangY', 'WangYE', 'Zhang'] #Study names

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
#%% ACES study wise Subtype plot
subtype_map = {0:'Normal-like', 1:'Basal-like', 2:'Luminal-A', 3:'Luminal-B', 4:'HER2'}
status_map = {0:'Non-metastatic', 1:'Metastatic'}
patient_info = pd.DataFrame(None, index = range(len(aces_data)))
for i, study in enumerate(study_names):
    test_index = d_map[d_map[i] == 1].index
    patient_info.loc[test_index, 'study'] = study
    patient_info.loc[test_index, 'status'] = aces_ptype[test_index].astype(int)
    patient_info.loc[test_index, 'subtype'] = aces_subtype[test_index].astype(int)
patient_info['subtype_name'] = patient_info['subtype'].map(subtype_map)
patient_info['status_name'] = patient_info['status'].map(status_map)

df = patient_info.pivot_table(index='study', columns='subtype_name', 
                              aggfunc='size', fill_value = 0)

df2 = patient_info.pivot_table(index='study', columns='status_name', 
                              aggfunc='size', fill_value = 0)

# plot
fig, ax = plt.subplots(2, 1, figsize = (12, 8), sharex = True)
df.plot(kind='bar' ,ax = ax[0])
ax[0].legend(bbox_to_anchor=(1, 1.02), loc='upper left')
df2.plot(kind='bar' ,ax = ax[1])
ax[1].legend(bbox_to_anchor=(1, 1.02), loc='upper left')
plt.xticks(rotation=45)
plt.suptitle('ACES study wise subtype plot')
plt.savefig(fig_dir + '/aces_study_wise_subtypes_and_status.png', bbox_inches='tight')
plt.show()
#%%plot ACES dataset AUC for different clustering algorithm
clustering_algorithms = ['Bulk', 'Kmeans', 'Hierarchical' , 'Spectral']

file = 'Results/ACES_individual_dataset_accuracy_scores.csv'
results_all_dataset = pd.read_csv(file, index_col =[0, 1, 2, 3, 4, 5])

mean_results = results_all_dataset.groupby(level = [0, 1, 2, 3, 5]).mean()

#%%Sort dataset by size
dataset_names = results_all_dataset.index.get_level_values('dataset').unique()
dataset_size = pd.DataFrame(0, index = dataset_names,
                            columns = ['total', 'negative', 'postive'])

for i, name in enumerate(mean_results.index.get_level_values('dataset').unique()):
    test_index = d_map[d_map[i] == 1].index
    data_test, Y = aces_data.iloc[test_index], aces_ptype[test_index].ravel()
    bad, good, total = np.sum(Y), int(data_test.shape[0]-np.sum(Y)), data_test.shape[0]
    print(name, '\t\t', total, '\t\t', good, '\t\t', bad)
    dataset_size.loc[name, :] = total, good, bad
dataset_size['idx'] = np.arange(0, len(dataset_size))
dataset_size.sort_values(by = 'total', ascending = False, inplace = True)

#%%Bar plot for different cluster type
clustering_algorithms = ['Bulk', 'Kmeans', 'Hierarchical' , 'Spectral']

file = 'Results/ACES_individual_dataset_accuracy_scores.csv'
results_all_dataset = pd.read_csv(file, index_col =[0, 1, 2, 3, 4, 5])

mean_results = results_all_dataset.groupby(level = [0, 1, 2, 3, 5]).mean()

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
    data_test, Y = aces_data.iloc[test_index], aces_ptype[test_index].ravel()
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

#%% ACES NKI Subtype relationship with metastatic patients

nki_subtype_map = {1:'Normal-like', 2:'Basal-like', 3:'Luminal-A', 4:'Luminal-B', 5:'HER2'}
status_map = {0:'Non-metastatic', 1:'Metastatic'}
nki_patient_info = pd.DataFrame(None, index = range(len(nki_data)))
for i in range(len(nki_data)):
    nki_patient_info.loc[i, 'status'] = nki_ptype[i].astype(int)
    nki_patient_info.loc[i, 'subtype'] = nki_subtype[i].astype(int)
nki_patient_info['subtype_name'] = nki_patient_info['subtype'].map(nki_subtype_map)
nki_patient_info['status_name'] = nki_patient_info['status'].map(status_map)



aces_subtype_map = {0:'Normal-like', 1:'Basal-like', 2:'Luminal-A', 3:'Luminal-B', 4:'HER2'}
status_map = {0:'Non-metastatic', 1:'Metastatic'}
aces_patient_info = pd.DataFrame(None, index = range(len(aces_data)))
for i, study in enumerate(study_names):
    test_index = d_map[d_map[i] == 1].index
    aces_patient_info.loc[test_index, 'study'] = study
    aces_patient_info.loc[test_index, 'status'] = aces_ptype[test_index].astype(int)
    aces_patient_info.loc[test_index, 'subtype'] = aces_subtype[test_index].astype(int)
aces_patient_info['subtype_name'] = aces_patient_info['subtype'].map(aces_subtype_map)
aces_patient_info['status_name'] = aces_patient_info['status'].map(status_map)

df = aces_patient_info.pivot_table(index='subtype_name', columns='status_name', 
                              aggfunc='size', fill_value = 0)
df2 = nki_patient_info.pivot_table(index='subtype_name', columns='status_name', 
                              aggfunc='size', fill_value = 0)
# plot
fig, ax = plt.subplots(2, 1, figsize = (10, 8), sharex = True)
df.plot(kind='bar', ax = ax[0])
ax[0].legend(bbox_to_anchor=(1, 1.02), loc='upper left')
df2.plot(kind='bar' ,ax = ax[1], legend = False)
# ax[1].legend(bbox_to_anchor=(1, 1.02), loc='upper left')
plt.xticks(rotation=45)
plt.suptitle('subtype wise metastasis status')
plt.savefig(fig_dir + '/subtype_wise_metastasis_status.png', bbox_inches='tight')
plt.show()
#%% Singlecell subtype vs cluster plot
# 'individual', 'organism_part', 'sampling_site', 'disease', 'histology',
# 'single_cell_id', 'subtype', 'patient_type', 'tumor_status', 'cell_type', 
# 'cell_type_name', 'cell_cluster_type_23'
plt.figure(figsize = (10, 6))
t2 = adata_pam.obs.pivot_table(index='cell_type_name', columns='cell_cluster_type_23', 
                              aggfunc='size', fill_value = 0)
sns.heatmap(t2, annot = True)
plt.title('Patients cells in each cluster')