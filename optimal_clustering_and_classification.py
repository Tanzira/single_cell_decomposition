#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:34:05 2023

@author: tanzira
"""

#%%All imports
import pandas as pd
import numpy as np
import seaborn as sns
import anndata
import scanpy as sc
from scqcut import scQcut_monolith
import matplotlib.pyplot as plt
import io
import csv
from docx import Document
import mygene
from sklearn.metrics import confusion_matrix as conf_mat, adjusted_rand_score as ari
from scipy.io import loadmat
from sklearn import linear_model


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.metrics import matthews_corrcoef


import augment
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering


import time
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

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


#%% Read ACES and TCGA and NKI Data
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
    
'''Reading WANG data'''

wang_raw = loadmat('Data/WangData/wang.mat')['wang']
wang_ptype = loadmat('Data/WangData/wang_label.mat')['label']
wang_entrez_id = loadmat('Data/WangData/wang_gene_list.mat')['wang_gene_list']
wang_data = pd.DataFrame(wang_raw)
wang_data.columns = wang_entrez_id.reshape(-1)

'''Reading GSE/UNC data'''
gse_raw = loadmat('Data/GSE_Data/gse.mat')['gse']
gse_ptype = loadmat('Data/GSE_Data/gse_label.mat')['gse_label']
gse_entrez_id = loadmat('Data/GSE_Data/gse_gene_list.mat')['gse_gene_list']
gse_data = pd.DataFrame(gse_raw)
gse_data.columns = gse_entrez_id.reshape(-1)

'''Reading YAU data'''
# wang_affymetrix, wang_gene_exp
yau_data = pd.read_csv('Data/YAU_gene_exp.csv', index_col=0)
yau_ptype = np.loadtxt('Data/YAU_lables.txt').astype(int)
yau_subtype = np.loadtxt('Data/YAU_subtypes.txt', dtype = str)


'''Reading vantveer data'''
# wang_affymetrix, wang_gene_exp
vantveer_data = pd.read_csv('Data/VantVeerData/VantVeer_exp.csv', index_col = 0)
vantveer_ptype = np.loadtxt('Data/VantVeerData/VantVeer_lables.txt').astype(int)
    
'''Reading TCGA data'''
# wang_affymetrix, wang_gene_exp
tcga_data = pd.read_csv('Data/TCGA_BRCA/TCGA_BRCA_expr.csv', index_col = 0)
tcga_ptype = np.loadtxt('Data/TCGA_BRCA/TCGA_BRCA_lables.txt').astype(int)

'''Reading Desmdt data'''
# wang_affymetrix, wang_gene_exp
desmedt_data = pd.read_csv('Data/DesmedtData/Desmedt_expr.csv', index_col = 0)
desmedt_ptype = np.loadtxt('Data/DesmedtData/Desmedt_label.txt').astype(int)

'''Reading Vijver/NKI data'''
# wang_affymetrix, wang_gene_exp
vijver_data = pd.read_csv('Data/VijverNKI/Vijver_expr.csv', index_col = 0)
vijver_ptype = np.loadtxt('Data/VijverNKI/Vijver_label.txt').astype(int)

#ACES, TCGA, PAM and single cell intersection

pam_50_genes = pd.read_csv('Data/PAM50GenesWitnEnsemble.txt', index_col = 0)
aces_ensemble = pd.read_csv('Data/ACES_entrez_to_ensemble.txt', index_col = 0)
nki_ensemble = pd.read_csv('Data/NKI_entrez_to_ensemble.txt', index_col = 0)
# wang_ensemble = pd.read_csv('Data/Wang_entrez_to_ensemble.txt', index_col = 0)
wang_ensemble = pd.read_csv('Data/WangData/wang_entrez_to_ensemble.txt', index_col = 0)
gse_ensemble = pd.read_csv('Data/GSE_Data/GSE_entrez_to_ensemble.txt', index_col = 0)

yau_ensemble = pd.read_csv('Data/YAU_symbol_to_ensemble.txt', index_col = 0)
vantveer_ensemble = pd.read_csv('Data/VantVeerData/VantVeer_symbol_to_ensemble.txt', index_col = 0)
# tcga_ensemble = pd.read_csv('Data/TCGA_symbol_to_ensemble.txt', index_col = 0)
desmedt_ensemble = pd.read_csv('Data/DesmedtData/Desmedt_symbol_to_ensemble.txt', index_col = 0)
vijver_ensemble = pd.read_csv('Data/VijverNKI/Vijver_symbol_to_ensemble.txt', index_col = 0)
tcga_ensemble = pd.read_csv('Data/TCGA_BRCA/TCGA_brca_symbol_to_ensemble.txt', index_col = 0)

#READ cosmic/EMT and BRCA dataset
cosmic_ensemble = pd.read_csv('Data/COSMIC_symbol_to_ensemble.txt', index_col = 0)
diffbrca_ensemble = pd.read_csv('Data/DIFF_BRCA_symbol_to_ensemble.txt', index_col = 0)
quickgo_emt_ensemble = pd.read_csv('Data/QUICKGO_EMT_symbol_to_ensemble.txt', index_col = 0)


aces_pam_avail = np.intersect1d(aces_ensemble.index, pam_50_genes.index)
nki_pam_avail = np.intersect1d(nki_ensemble.index, pam_50_genes.index)
wang_pam_avail = np.intersect1d(wang_ensemble.index, pam_50_genes.index)
gse_pam_avail = np.intersect1d(gse_ensemble.index, pam_50_genes.index)
yau_pam_avail = np.intersect1d(yau_ensemble.index.astype('str'), pam_50_genes.index)
vantveer_pam_avail = np.intersect1d(vantveer_ensemble.index.astype('str'), pam_50_genes.index)
desmedt_pam_avail = np.intersect1d(desmedt_ensemble.index.astype('str'), pam_50_genes.index)
vijver_pam_avail = np.intersect1d(vijver_ensemble.index.astype('str'), pam_50_genes.index)
tcga_pam_avail = np.intersect1d(tcga_ensemble.index.astype('str'), pam_50_genes.index)


#Number of all common genes in ACES and the single cell database
aces_adata_comm = np.intersect1d(aces_ensemble.index, adata.var)
#Number of all common genes in ACES and the single cell database
nki_adata_comm = np.intersect1d(nki_ensemble.index, adata.var)

wang_adata_comm = np.intersect1d(wang_ensemble.index, adata.var)
#Number of all common genes in TCGA and the single cell database
yau_adata_comm = np.intersect1d(yau_ensemble.index, adata.var)
tcga_adata_comm = np.intersect1d(tcga_ensemble.index, adata.var)

#Number of all common genes in ACES and the single cell database
cosmic_aces_comm = np.intersect1d(cosmic_ensemble.index, aces_adata_comm)
#Number of all common genes in ACES and the single cell database
diffbrca_aces_comm = np.intersect1d(diffbrca_ensemble.index, aces_adata_comm)
#Number of all common genes in TCGA and the single cell database
quickgo_emt_aces_comm = np.intersect1d(quickgo_emt_ensemble.index, aces_adata_comm)


#Number of all common genes in ACES and the single cell database
cosmic_nki_comm = np.intersect1d(cosmic_ensemble.index, nki_adata_comm)
#Number of all common genes in ACES and the single cell database
diffbrca_nki_comm = np.intersect1d(diffbrca_ensemble.index, nki_adata_comm)
#Number of all common genes in TCGA and the single cell database
quickgo_emt_nki_comm = np.intersect1d(quickgo_emt_ensemble.index, nki_adata_comm)

#Number of all common genes in ACES and the single cell database
cosmic_tcga_comm = np.intersect1d(cosmic_ensemble.index, tcga_adata_comm)
#Number of all common genes in ACES and the single cell database
diffbrca_tcga_comm = np.intersect1d(diffbrca_ensemble.index, tcga_adata_comm)
#Number of all common genes in TCGA and the single cell database
quickgo_emt_tcga_comm = np.intersect1d(quickgo_emt_ensemble.index, tcga_adata_comm)

#%%Cluster SC data and Decompose NKI/AECS dataset
n_splits = 10
random_state = 0
niter = 2
cmin, cmax = 5, 45
#all false for aces
#Patient remove or keep
remove_non_tumor = True
remove_lymphnode = False
remove_chemo_patient = True

datasets = ['NKI', 'ACES', 'TCGA', 'Desmedt', 'VantVeer', 'GSE', 'WANG', 'Vijver', 'YAU']

test_data = datasets[1]
print(test_data)

models = {"RF" :lambda random_state: RandomForestClassifier(random_state = random_state),
        # "LR" : lambda random_state:LogisticRegression(solver='liblinear', random_state=random_state),
        #   "DT" : lambda random_state:DecisionTreeClassifier(random_state=random_state),
        #   "KNN" : lambda random_state:KNeighborsClassifier(),
        # "MLP": lambda random_state:MLPClassifier(random_state=random_state),
        # "SVC" : lambda random_state:svm.SVC(kernel = 'rbf', random_state=random_state, probability=True),
        # "ADB" : lambda random_state:AdaBoostClassifier(random_state=random_state),
        # "BAGG" :lambda random_state: BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=random_state)
    }
clustering_algorithms = {'Kmeans' : lambda n_clusters: KMeans(n_clusters=n_clusters, random_state=random_state, n_init = 100),
                          # # 'BisectingKMeans' : lambda n_clusters:BisectingKMeans(n_clusters=n_clusters, random_state=random_state),
                               'Hierarchical' : lambda n_clusters:AgglomerativeClustering(n_clusters=n_clusters, linkage='average'),
                          #  'Birch' : lambda n_clusters:Birch(threshold = 0.5, n_clusters=n_clusters),
                                # 'Spectral' : lambda n_clusters:SpectralClustering(n_clusters=n_clusters,
                                #                 assign_labels='discretize',random_state=random_state)
                          }

def performance_metrics(y_true, y_pred, y_score):
    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return auc, f1, mcc, kappa

def get_X_Y(data, Y, ensemble_id, available, entrez = True):
    if entrez:
        X = data.loc[:, ensemble_id.loc[available].entrezgene.values]
    else:
        X = data.loc[:, ensemble_id.loc[available].symbol.values]
    return X, Y.ravel(), available

def do_cluster(sc_X, adata_pam, n_clusters):
    cluster_alg = clustering_algorithms['Hierarchical'](n_clusters).fit(sc_X)
    membership = cluster_alg.labels_
    membership = np.array(['C{:03d}'.format(i) for i in cluster_alg.labels_])
    #keeping the cluster type of each cell inside observations
    adata_pam.obs['cell_cluster_type_{0}'.format(n_clusters)] = membership    
    sc_data_raw = pd.DataFrame(adata_pam.layers['normalized'].todense(), 
                      index = adata_pam.obs['cell_cluster_type_{0}'.format(n_clusters)], 
                      columns = adata_pam.var.index)
    sc_data = sc_data_raw.groupby(['cell_cluster_type_{0}'.format(n_clusters)]).mean()
    return sc_data


if test_data == 'NKI':
    X, Y, available = get_X_Y(nki_data, nki_ptype, nki_ensemble, nki_pam_avail, True)
    
elif test_data == 'ACES':
    X, Y, available = get_X_Y(aces_data, aces_ptype, aces_ensemble, aces_pam_avail, True)

elif test_data == 'WANG':
    X, Y, available = get_X_Y(wang_data, wang_ptype, wang_ensemble, wang_pam_avail, True)
    
elif test_data == 'GSE':
    X, Y, available = get_X_Y(gse_data, gse_ptype, gse_ensemble, gse_pam_avail, True)

elif test_data == 'VantVeer':
    X, Y, available = get_X_Y(vantveer_data, vantveer_ptype, vantveer_ensemble, vantveer_pam_avail, True)

elif test_data == 'YAU':
    X, Y, available = get_X_Y(yau_data, yau_ptype, yau_ensemble, yau_pam_avail, False)

elif test_data == 'TCGA':
    X, Y, available = get_X_Y(tcga_data, tcga_ptype, tcga_ensemble, tcga_pam_avail, False)
    
elif test_data == 'Desmedt':
    X, Y, available = get_X_Y(desmedt_data, desmedt_ptype, desmedt_ensemble, desmedt_pam_avail, False)
    
elif test_data == 'Vijver':
    X, Y, available = get_X_Y(vijver_data, vijver_ptype, vijver_ensemble, vijver_pam_avail, False)
    
X = X.to_numpy()

adata_pam = adata[:, available]

# sc_tumor_only = np.intersect1d(adata_pam.obs.single_cell_id, sc_tumor_only_all)


combined_sc_clusters = pd.DataFrame()
if remove_non_tumor:
    #Keep only five types
    sc_X =pd.DataFrame(adata_pam.layers['normalized'].todense(), 
                      index = adata_pam.obs['cell_type_name'], 
                      columns = adata_pam.var.index)
    sc_X = sc_X[sc_X.index != 'Tumor']
    sc_data = do_cluster(sc_X, adata_pam[adata_pam.obs.tumor_status != 'Tumor'], 17)
    # sc_data = sc_X.groupby(['cell_type_name']).mean()
    # sc_data = sc_data.iloc[[0, 1, 2, 3], :]
    # print(sc_data)
    combined_sc_clusters = combined_sc_clusters.append(sc_data)
    adata_pam = adata_pam[adata_pam.obs.tumor_status == 'Tumor']
if remove_chemo_patient:
    # adata_temp = adata_pam[adata_pam.obs.individual == 'BC05']
    # sc_X =pd.DataFrame(adata_temp.layers['normalized'].todense(), 
    #                   index = adata_temp.obs['cell_type_name'], 
    #                   columns = adata_pam.var.index)
    # sc_data = sc_X.groupby(['cell_type_name']).mean()
    # combined_sc_clusters = combined_sc_clusters.append(sc_data)
    adata_pam = adata_pam[adata_pam.obs.individual != 'BC05']
   
if remove_lymphnode:
    adata_pam = adata_pam[adata_pam.obs.organism_part != 'lymph node']

#Cluster cells into 20 groups using kmeans
sc_X = adata_pam.layers['normalized'].todense()
results_all_cluster = {}

for cluster_alg_name, cluster_alg_func in clustering_algorithms.items():
    start_time = datetime.now()
    results_cluster_it = {}
    for n_clusters in range(cmin, cmax, 1):
        cluster_alg = cluster_alg_func(n_clusters)
        cluster_alg.fit(sc_X)
        if cluster_alg_name == 'Birch':
            membership = cluster_alg.predict(sc_X)
        else:
            membership = cluster_alg.labels_
        
        n_clusters = len(np.unique(membership))
        print('\n',test_data, cluster_alg_name, 'n_clusters: ', n_clusters, X.shape,  '\n')
        
        membership = np.array(['C{:03d}'.format(i) for i in cluster_alg.labels_])
        #keeping the cluster type of each cell inside observations
        adata_pam.obs['cell_cluster_type'] = membership
        #Encode cells into a size-20 vector of 0 and 1
        encoded_sc_data = pd.DataFrame(0, index = adata_pam.obs['single_cell_id'],
                                       columns = np.unique(membership))
        
        for cluster_id in encoded_sc_data.columns:
            row_idx = adata_pam.obs[adata_pam.obs['cell_cluster_type'] == cluster_id].single_cell_id
            encoded_sc_data.loc[row_idx.values, cluster_id] = 1
        
        #decompose NKI/ACES/TCGA data using single cell data
        sc_data_raw = pd.DataFrame(adata_pam.layers['normalized'].todense(), 
                          index = adata_pam.obs['cell_cluster_type'], 
                          columns = adata_pam.var.index)
        sc_data = sc_data_raw.groupby(['cell_cluster_type']).mean()
        sc_data = sc_data.to_numpy().T
        
        #decomposing bulk dataset using single cell
        decomposed_mat = []
        for i, patient in enumerate(X):
            reg = LinearRegression()
            reg.fit(sc_data, patient)
            decomposed_mat.append(reg.coef_)
        #converting into numpy array
        decomposed_mat = np.array(decomposed_mat)
        results_all = {}
        for model_name, model_func in models.items():
            results_it = {}
            for iteration in range(niter):
                results_metric = pd.DataFrame(0, index = ['AUC', 'F1', 'MCC', 'Kappa'],
                                              columns = [ 'bulk', 'decomp'])
                random_state = iteration
                model = model_func(random_state)
                skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
                
                # pred_probas = []
                # y_true, y_pred = [], []
                # for cv_idx, (train_index, test_index) in enumerate(skf.split(X, Y)):
                #     data_train, c_train, data_test, c_test = X[train_index], Y[train_index], X[test_index], Y[test_index]
                #     model.fit(data_train, c_train)
                #     pred_proba = model.predict_proba(data_test)[:, 1]
                #     pred_probas.extend(pred_proba)
                #     y_pred.extend(model.predict(data_test))
                #     y_true.extend(c_test)
                # auc, f1, mcc, kappa = performance_metrics(y_true, y_pred, pred_probas)
                # results_metric.loc[:, 'bulk'] = [auc, f1, mcc, kappa]
                
                pred_probas = []
                y_true, y_pred = [], []
                
                for cv_idx, (train_index, test_index) in enumerate(skf.split(decomposed_mat, Y)):
                    data_train, c_train, data_test, c_test = decomposed_mat[train_index], Y[train_index],\
                                                    decomposed_mat[test_index], Y[test_index]
                    model.fit(data_train, c_train)
                    pred_proba = model.predict_proba(data_test)[:, 1]
                    pred_probas.extend(pred_proba)
                    y_pred.extend(model.predict(data_test))
                    y_true.extend(c_test)
                auc, f1, mcc, kappa = performance_metrics(y_true, y_pred, pred_probas)
                results_metric.loc[:, 'decomp'] = [auc, f1, mcc, kappa]
                results_it[iteration] = results_metric
            results_it = pd.concat(results_it)
            results_all[model_name] = results_it
        t = pd.concat(results_all, names = ['model', 'iteration', 'metric'])  
        t = t.reorder_levels([0, 2, 1]).sort_index()
        # # print(results_all.groupby(level = [0,1]).mean())
        print( t.groupby(level = [0,1]).mean().loc[:, 'AUC', :])
        results_all = pd.concat(results_all)
        results_cluster_it[n_clusters] = results_all
        end_time = datetime.now()
        
    results_cluster_it = pd.concat(results_cluster_it)
    results_all_cluster[cluster_alg_name] = results_cluster_it
    print('\nDuration for each cluster: {}'.format(end_time - start_time))
results_all_cluster = pd.concat(results_all_cluster, names = ['clust_name', 'n_cluster', 'model', 'iteration', 'metric'])
results_all_cluster = results_all_cluster.reorder_levels([2, 0, 4, 1, 3]).sort_index()

#%%Apply clustering and classification
def do_cluster(sc_X, adata_pam, n_clusters):
    cluster_alg = clustering_algorithms['Hierarchical'](n_clusters).fit(sc_X)
    membership = cluster_alg.labels_
    membership = np.array(['C{:03d}'.format(i) for i in cluster_alg.labels_])
    #keeping the cluster type of each cell inside observations
    adata_pam.obs['cell_cluster_type_{0}'.format(n_clusters)] = membership    
    sc_data_raw = pd.DataFrame(adata_pam.layers['normalized'].todense(), 
                      index = adata_pam.obs['cell_cluster_type_{0}'.format(n_clusters)], 
                      columns = adata_pam.var.index)
    sc_data = sc_data_raw.groupby(['cell_cluster_type_{0}'.format(n_clusters)]).mean()
    return sc_data

models = {"RF" :lambda random_state: RandomForestClassifier(random_state = random_state)}
clustering_algorithms = {'Hierarchical' : lambda n_clusters:AgglomerativeClustering(n_clusters=n_clusters)}

def performance_metrics(y_true, y_pred, y_score):
    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return auc, f1, mcc, kappa

n_splits = 10
random_state = 15
niter = 5

#Patient remove or keep
remove_non_tumor = True
remove_lymphnode = False
remove_chemo_patient = True

test_data = 'ACES'
# n_clusters = 19
clst1, clst2, clst3 = 3, 20, 14

for clst3 in range(5, 30, 1):
    if test_data == 'NKI':
        X, Y, available = get_X_Y(nki_data, nki_ptype, nki_ensemble, nki_pam_avail, True)
        
    elif test_data == 'ACES':
        X, Y, available = get_X_Y(aces_data, aces_ptype, aces_ensemble, aces_pam_avail, True)
        
    elif test_data == 'GSE':
        X, Y, available = get_X_Y(gse_data, gse_ptype, gse_ensemble, gse_pam_avail, True)

    elif test_data == 'TCGA':
        X, Y, available = get_X_Y(tcga_data, tcga_ptype, tcga_ensemble, tcga_pam_avail, False)
            
    adata_pam = adata[:, pam_50_genes.index]
    adata_pam = adata[:, aces_pam_avail]
    
    combined_sc_clusters = pd.DataFrame()
     

    if remove_non_tumor:
        #Keep only five types
        sc_X =pd.DataFrame(adata_pam.layers['normalized'].todense(), 
                          index = adata_pam.obs['cell_type_name'], 
                          columns = adata_pam.var.index)
        sc_X = sc_X[sc_X.index != 'Tumor']
        sc_data = do_cluster(sc_X, adata_pam[adata_pam.obs.tumor_status != 'Tumor'], 17)
        # sc_data = sc_X.groupby(['cell_type_name']).mean()
        # sc_data = sc_data.iloc[[0, 1, 2, 3], :]
        # print(sc_data)
        combined_sc_clusters = combined_sc_clusters.append(sc_data)
        adata_pam = adata_pam[adata_pam.obs.tumor_status == 'Tumor']
    if remove_chemo_patient:
        # adata_temp = adata_pam[adata_pam.obs.individual == 'BC05']
        # sc_X =pd.DataFrame(adata_temp.layers['normalized'].todense(), 
        #                   index = adata_temp.obs['cell_type_name'], 
        #                   columns = adata_pam.var.index)
        # sc_data = sc_X.groupby(['cell_type_name']).mean()
        # combined_sc_clusters = combined_sc_clusters.append(sc_data)
        adata_pam = adata_pam[adata_pam.obs.individual != 'BC05']
       
    if remove_lymphnode:
        adata_pam = adata_pam[adata_pam.obs.organism_part != 'lymph node']
    
    sc_X =pd.DataFrame(adata_pam.layers['normalized'].todense(), 
                      index = adata_pam.obs['organism_part'], 
                      columns = adata_pam.var.index)
    # for n_clusters in range(16, 16, 5):
    sc_data = do_cluster(sc_X, adata_pam, clst3)
    combined_sc_clusters = combined_sc_clusters.append(sc_data)
    
    
    # print(combined_sc_clusters)
    
    combined_sc_clusters = combined_sc_clusters.loc[:, available]
    
    combined_sc_clusters = combined_sc_clusters.to_numpy().T
    
    X = X.to_numpy()
    #decomposing bulk dataset using single cell
    decomposed_mat = []
    for i, patient in enumerate(X):
        # reg = linear_model.Lasso(alpha=.01)
        reg = LinearRegression()
        # reg = HuberRegressor()
        reg.fit(combined_sc_clusters, patient)
        decomposed_mat.append(reg.coef_)
    #converting into numpy array
    decomposed_mat = np.array(decomposed_mat)
    print('done: ', X.shape, clst3, decomposed_mat.shape, adata_pam.shape)
    
    results_all = {}
    for model_name, model_func in models.items():
        results_it = {}
        for iteration in range(niter):
            results_metric = pd.DataFrame(0, index = ['AUC', 'F1', 'MCC', 'Kappa'],
                                          columns = [ 'bulk', 'decomp'])
            # results_metric = pd.DataFrame(0, index = ['AUC', 'F1', 'MCC', 'Kappa'],
            #                               columns = ['bulk', 'decomp', 'sc_train'])
            random_state = iteration
            model = model_func(random_state)
            skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            
            # pred_probas = []
            # y_true, y_pred = [], []
            # for cv_idx, (train_index, test_index) in enumerate(skf.split(X, Y)):
            #     data_train, c_train, data_test, c_test = X[train_index], Y[train_index], X[test_index], Y[test_index]
            #     model.fit(data_train, c_train)
            #     pred_proba = model.predict_proba(data_test)[:, 1]
            #     pred_probas.extend(pred_proba)
            #     y_pred.extend(model.predict(data_test))
            #     y_true.extend(c_test)
            # auc, f1, mcc, kappa = performance_metrics(y_true, y_pred, pred_probas)
            # results_metric.loc[:, 'bulk'] = [auc, f1, mcc, kappa]
            
            pred_probas = []
            y_true, y_pred = [], []
            
            for cv_idx, (train_index, test_index) in enumerate(skf.split(decomposed_mat, Y)):
                data_train, c_train, data_test, c_test = decomposed_mat[train_index], Y[train_index],\
                                                decomposed_mat[test_index], Y[test_index]
                model.fit(data_train, c_train)
                pred_proba = model.predict_proba(data_test)[:, 1]
                pred_probas.extend(pred_proba)
                y_pred.extend(model.predict(data_test))
                y_true.extend(c_test)
            auc, f1, mcc, kappa = performance_metrics(y_true, y_pred, pred_probas)
            results_metric.loc[:, 'decomp'] = [auc, f1, mcc, kappa]
            results_it[iteration] = results_metric
        results_it = pd.concat(results_it)
        results_all[model_name] = results_it
    t = pd.concat(results_all, names = ['model', 'iteration', 'metric'])  
    t = t.reorder_levels([0, 2, 1]).sort_index()
    # # print(results_all.groupby(level = [0,1]).mean())
    print('\n', t.groupby(level = [0,1]).mean().loc[:, 'AUC', :])
    results_all = pd.concat(results_all)
    
    # plt.figure(figsize = (10, 6))
    # t1 = encoded_sc_data.copy()
    # t1['patient'] = t1.index.str.split('_').str[0]
    # t2 = t1.groupby('patient').sum()
    # # t2 = t2 / t2.sum(axis = 1)[:, None]
    # sns.heatmap(t2, annot = True)
    # plt.title('Patients cells in each cluster')
    # plt.show()