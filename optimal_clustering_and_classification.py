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
# from scqcut import scQcut_monolith
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
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.metrics import matthews_corrcoef


# import augment
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
yau_data = pd.read_csv('Data/YAU_Data/YAU_gene_exp.csv', index_col=0)
yau_ptype = np.loadtxt('Data/YAU_Data/YAU_lables.txt').astype(int)
yau_subtype = np.loadtxt('Data/YAU_Data/YAU_subtypes.txt', dtype = str)


'''Reading vantveer data'''
# wang_affymetrix, wang_gene_exp
vantveer_data = pd.read_csv('Data/VantVeerData/VantVeer_exp.csv', index_col = 0)
vantveer_ptype = np.loadtxt('Data/VantVeerData/VantVeer_lables.txt').astype(int)
    
'''Reading TCGA data'''
# wang_affymetrix, wang_gene_exp
tcga_data = pd.read_csv('Data/TCGA_BRCA/expr_brca_only.csv', sep = '\t', index_col = 0)
tcga_survival = pd.read_csv('Data/TCGA_BRCA/survival_brca_only.csv', sep = '\t', index_col = 0)
tcga_ptype = np.array(tcga_survival.PFI)

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

yau_ensemble = pd.read_csv('Data/YAU_Data/YAU_symbol_to_ensemble.txt', index_col = 0)
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
# random_state = 0
niter = 2
cmin, cmax = 18, 21
#all false for aces
#Patient remove or keep
remove_non_tumor = True
remove_lymphnode = False
remove_chemo_patient = True

datasets = ['NKI',  'GSE', 'ACES', 'TCGA', 'WANG', 'Desmedt', 'VantVeer', 'Vijver', 'YAU']

test_data = datasets[0]
print(test_data)

models = {
            "RF" :lambda random_state: RandomForestClassifier(random_state = random_state),
        # "LR" : lambda random_state:LogisticRegression(solver='liblinear', random_state=random_state),
        #   "DT" : lambda random_state:DecisionTreeClassifier(random_state=random_state),
          # "KNN" : lambda random_state:KNeighborsClassifier(),
        # "MLP": lambda random_state:MLPClassifier(random_state=random_state),
        # "SVC" : lambda random_state:svm.SVC(kernel = 'rbf', random_state=random_state, probability=True),
        # "ADB" : lambda random_state:AdaBoostClassifier(random_state=random_state),
        # "BAGG" :lambda random_state: BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=random_state)
    }
clustering_algorithms = {'Kmeans' : lambda n_clusters: KMeans(n_clusters=n_clusters, random_state=random_state, n_init = 100),
                          # # 'BisectingKMeans' : lambda n_clusters:BisectingKMeans(n_clusters=n_clusters, random_state=random_state),
                               # 'Hierarchical' : lambda n_clusters:AgglomerativeClustering(n_clusters=n_clusters, linkage='average'),
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

def do_cluster(sc_X, adata_pam, cluster_alg_name, n_clusters):
    
    cluster_alg = clustering_algorithms[cluster_alg_name](n_clusters).fit(sc_X)
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

# adata_pam = adata[:, pam_50_genes.index]
adata_pam = adata[:, available]

results_all_cluster = {}

for cluster_alg_name, cluster_alg_func in clustering_algorithms.items():
    start_time = datetime.now()
    results_cluster_it = {}
    for n_clusters in range(cmin, cmax, 1):
        adata_pam = adata[:, available]
        combined_sc_clusters = pd.DataFrame()
        if remove_non_tumor:
            adata_pam = adata_pam[adata_pam.obs.tumor_status == 'Tumor']
        if remove_chemo_patient:
            adata_pam = adata_pam[adata_pam.obs.individual != 'BC05']
           
        if remove_lymphnode:
            adata_pam = adata_pam[adata_pam.obs.organism_part != 'lymph node']

        #Cluster cells into 20 groups using kmeans
        sc_X = adata_pam.layers['normalized'].todense()
        sc_data = do_cluster(sc_X, adata_pam, cluster_alg_name, n_clusters)
        sc_data = sc_data.to_numpy().T
        #decomposing bulk dataset using single cell
        decomposed_mat = []
        for i, patient in enumerate(X):
            reg = LinearRegression()
            reg.fit(sc_data, patient)
            decomposed_mat.append(reg.coef_)
        #converting into numpy array
        decomposed_mat = np.array(decomposed_mat)
        print(test_data, n_clusters, decomposed_mat.shape)
        results_all = {}
        for model_name, model_func in models.items():
            results_it = {}
            for iteration in range(niter):
                results_metric = pd.DataFrame(0, index = ['AUC', 'F1', 'MCC', 'Kappa'],
                                              columns = [ 'bulk', 'decomp'])
                random_state = iteration
                model = model_func(random_state)
                skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
                
                pred_probas = []
                y_true, y_pred = [], []
                for cv_idx, (train_index, test_index) in enumerate(skf.split(X, Y)):
                    data_train, c_train, data_test, c_test = X[train_index], Y[train_index], X[test_index], Y[test_index]
                    model.fit(data_train, c_train)
                    pred_proba = model.predict_proba(data_test)[:, 1]
                    pred_probas.extend(pred_proba)
                    y_pred.extend(model.predict(data_test))
                    y_true.extend(c_test)
                auc, f1, mcc, kappa = performance_metrics(y_true, y_pred, pred_probas)
                results_metric.loc[:, 'bulk'] = [auc, f1, mcc, kappa]
                
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

#%% Dataset wise classification (NKI, GSE, WANG, TCGA)
n_splits = 10
niter = 20
cmin, cmax = 5, 51
#all false for aces
#Patient remove or keep
remove_non_tumor = True
remove_lymphnode = False
remove_chemo_patient = True

models = {
    "RF" :lambda random_state: RandomForestClassifier(random_state = random_state),
    }
clustering_algorithms = {
    'Bulk' : lambda n_clusters : None,
    'Kmeans' : lambda n_clusters, random_state : KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1000),
    'Hierarchical' : lambda n_clusters, random_state : AgglomerativeClustering(n_clusters=n_clusters, linkage='average'),
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

def dataset_picker(test_data):
    if test_data == 'NKI':
        X, Y, available = get_X_Y(nki_data, nki_ptype, nki_ensemble, nki_pam_avail, True)
            
    elif test_data == 'GSE':
        X, Y, available = get_X_Y(gse_data, gse_ptype, gse_ensemble, gse_pam_avail, True)
        
    elif test_data == 'ACES':
        X, Y, available = get_X_Y(aces_data, aces_ptype, aces_ensemble, aces_pam_avail, True)
    
    elif test_data == 'TCGA':
        X, Y, available = get_X_Y(tcga_data, tcga_ptype, tcga_ensemble, tcga_pam_avail, False)
        
    elif test_data == 'WANG':
        X, Y, available = get_X_Y(wang_data, wang_ptype, wang_ensemble, wang_pam_avail, True) 
        
    elif test_data == 'VantVeer':
        X, Y, available = get_X_Y(vantveer_data, vantveer_ptype, vantveer_ensemble, vantveer_pam_avail, True)
    X = X.to_numpy() 
    return X, Y, available

def do_cluster(sc_X, adata_pam, cluster_alg_name, n_clusters, random_state):
    
    cluster_alg = clustering_algorithms[cluster_alg_name](n_clusters, random_state).fit(sc_X)
    membership = cluster_alg.labels_
    membership = np.array(['C{:03d}'.format(i) for i in cluster_alg.labels_])
    #keeping the cluster type of each cell inside observations
    adata_pam.obs['cell_cluster_type_{0}'.format(n_clusters)] = membership    
    sc_data_raw = pd.DataFrame(adata_pam.layers['normalized'].todense(), 
                      index = adata_pam.obs['cell_cluster_type_{0}'.format(n_clusters)], 
                      columns = adata_pam.var.index)
    sc_data = sc_data_raw.groupby(['cell_cluster_type_{0}'.format(n_clusters)]).mean()
    return sc_data
# dataset_names=['Desmedt', 'Hatzis', 'Ivshina', 'Loi', 'Miller', 'Minn',
#          'Pawitan', 'Schmidt', 'Symmans', 'WangY', 'WangYE', 'Zhang'] #Study names

# dataset_names = ['NKI',  'GSE', 'ACES', 'TCGA', 'WANG', 'Desmedt', 'VantVeer', 'Vijver', 'YAU']

dataset_names = ['NKI',  'GSE', 'WANG', 'TCGA', 'ACES']

begin_time = datetime.now()
results_all_dataset = {}
for data_idx, dataset_name in enumerate(dataset_names):
    start_time = datetime.now()
    X, Y, available = dataset_picker(dataset_name)
    adata_pam = adata[:, available]
    if remove_non_tumor:
        adata_pam = adata_pam[adata_pam.obs.tumor_status == 'Tumor']
    if remove_chemo_patient:
        adata_pam = adata_pam[adata_pam.obs.individual != 'BC05']
    if remove_lymphnode:
        adata_pam = adata_pam[adata_pam.obs.organism_part != 'lymph node']
    results_it = {}
    for iteration in range(niter):#iterating each model n times
        random_state = iteration
        skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        results_model = {}
        for model_name, model_func in models.items(): #iterating through models
            model = model_func(random_state)
            results_all_cluster = {}
            for cluster_alg_name, cluster_alg_func in clustering_algorithms.items():
                results_cluster_it = {}
                results_metric = pd.DataFrame(0, index = ['AUC', 'F1', 'MCC', 'Kappa'], columns = ['score'])
                if cluster_alg_name == 'Bulk':
                    n_clusters = 1
                    pred_probas, y_true, y_pred = [], [], []
                    for cv_idx, (train_index, test_index) in enumerate(skf.split(X, Y)):
                        data_train, c_train, data_test, c_test = X[train_index], Y[train_index],\
                            X[test_index], Y[test_index]
                        model.fit(data_train, c_train)
                        pred_proba = model.predict_proba(data_test)[:, 1]
                        pred_probas.extend(pred_proba)
                        y_pred.extend(model.predict(data_test))
                        y_true.extend(c_test)
                    auc, f1, mcc, kappa = performance_metrics(y_true, y_pred, pred_probas)
                    results_metric.loc[:, 'score'] = [auc, f1, mcc, kappa]
                    results_cluster_it[n_clusters] = results_metric
                else:
                    sc_X = adata_pam.layers['normalized'].todense()
                    for n_clusters in range(cmin, cmax, 1):
                        results_metric = pd.DataFrame(0, index = ['AUC', 'F1', 'MCC', 'Kappa'], columns = ['score'])
                        sc_data = do_cluster(sc_X, adata_pam, cluster_alg_name, n_clusters, random_state)
                        sc_data = sc_data.to_numpy().T
                        
                        #decomposing bulk dataset using single cell
                        decomposed_mat = []
                        for i, patient in enumerate(X):
                            reg = LinearRegression()
                            reg.fit(sc_data, patient)
                            decomposed_mat.append(reg.coef_)
                        #converting into numpy array
                        decomposed_mat = np.array(decomposed_mat)
                        
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
                        results_metric.loc[:, 'score'] = [auc, f1, mcc, kappa]
                        results_cluster_it[n_clusters] = results_metric
                        
                results_cluster_it = pd.concat(results_cluster_it)
                results_all_cluster[cluster_alg_name] = results_cluster_it
                
            
            results_all_cluster = pd.concat(results_all_cluster)
            results_model[model_name] = results_all_cluster
            
        results_model = pd.concat(results_model)
        results_it[iteration] = results_model
        print(dataset_names[data_idx], 'iteration', iteration)
    t = pd.concat(results_it, names = ['iteration', 'classifier', 'cluster_type', 'n_cluster', 'metric'])  
    t = t.reorder_levels([1, 2, 3, 4, 0]).sort_index()
    # print( dataset_names[data_idx], t.groupby(level = [0,1,3]).mean().loc[:, :, 'AUC', :])
    print( dataset_names[data_idx], t.groupby(level = [0,1, 2, 3]).mean().loc['RF', :, :, 'AUC'])
    results_it = pd.concat(results_it)
    results_all_dataset[dataset_names[data_idx]] = results_it
    end_time = datetime.now()
    print('\nDuration for each dataset: {}'.format(end_time - start_time))
results_all_dataset = pd.concat(results_all_dataset, names = ['dataset', 'iteration', 'classifier', 
                                                              'cluster_type', 'n_cluster', 'metric']) 
results_all_dataset = results_all_dataset.reorder_levels([0, 2, 3, 4, 1, 5]).sort_index()
print('\nTotal Duration for all dataset: {}'.format(end_time - begin_time))
#%%plot ACES dataset AUC for different clustering algorithm
mean_results = results_all_dataset.groupby(level = [0, 1, 2, 3, 5]).mean()
# fig, ax = plt.subplots(4, 3, figsize = (24, 16), sharex=True, sharey=True)
# ax = ax.flat
fig, ax = plt.subplots(2, 1, figsize = (10, 12))
auc_each_dataset = {}
for i, name in enumerate(mean_results.index.get_level_values('dataset').unique()):
    auc_and_index = pd.DataFrame(0, index=clustering_algorithms.keys(), columns = ['n_cluster', 'AUC'])
    for cluster_name in clustering_algorithms.keys():
        t1 = mean_results.groupby(['dataset', 'metric', 'cluster_type', 'n_cluster']).max().loc[name, 'AUC', cluster_name].idxmax()
        t2 = mean_results.groupby(['dataset', 'metric', 'cluster_type', 'n_cluster']).max().loc[name, 'AUC', cluster_name].max()
        auc_and_index.loc[cluster_name] = [t1.iloc[0], t2.iloc[0]]
    auc_each_dataset[name] = auc_and_index
auc_each_dataset = pd.concat(auc_each_dataset, names = ['dataset', 'cluster_type'])
bar = auc_each_dataset.unstack(level = 1).plot.bar(y = 'AUC', ax = ax[0])
bar = auc_each_dataset.unstack(level = 1).plot.bar(y = 'n_cluster', ax = ax[1], legend = False)
ax[0].legend(bbox_to_anchor = [1, 1], loc = 'upper left')
ax[0].set_ylim([0.3, 0.85])
plt.ylabel('AUC')
ax[0].set_title('Best AUC for individual datasets')
ax[1].set_title('n_cluster for best result')
plt.subplots_adjust(wspace = 0.05, hspace = 0.3)
plt.show()

#%%Analysis on hierarchical clustering
# dataset_names = ['NKI',  'GSE', 'WANG', 'TCGA']
mean_results = results_all_dataset.groupby(level = [0, 1, 2, 3, 5]).mean()
fig, ax = plt.subplots(2, 3, figsize = (12, 6), sharex = True, sharey = False)
ax = ax.flat
cluster_type = 'Kmeans'
acc_metric = 'AUC'
print('Dataset \t\t total \t\t Good \t\tBad')
for i, name in enumerate(mean_results.index.get_level_values('dataset').unique()):
    X, Y, available = dataset_picker(name)
    bad, good = np.sum(Y), int(X.shape[0]-np.sum(Y))
    print(name, '\t\t', X.shape[0], '\t\t', good, '\t\t', bad)
    t2 = mean_results.loc[name, 'RF', cluster_type, :, acc_metric].plot(ax = ax[i],
                                            title = '{0}: {1}'.format(name, X.shape[0]),
                                            marker = '.', legend = False)
    ax[i].axhline(mean_results.loc[name, 'RF', 'Bulk', :, 'AUC'].values, color = 'r')
plt.subplots_adjust(wspace = 0.2, hspace = 0.3)
ax[-1].legend(['decomp', 'bulk'], bbox_to_anchor =(1, 1), loc = 'upper left')
plt.suptitle("{0} with {1} clustering for different k".format(acc_metric, cluster_type))
plt.show()

#%%Single dataset evaluation
n_splits = 10
random_state = 4
niter = 2
cmin, cmax = 233, 234

remove_non_tumor = True
remove_lymphnode = False
remove_chemo_patient = True

datasets = ['ACES']

test_data = datasets[0]
print(test_data)

models = {
    "RF" :lambda random_state: RandomForestClassifier(random_state = random_state),
    # "ADB" : lambda random_state:AdaBoostClassifier(random_state=random_state),
    # "BAGG" :lambda random_state: BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=random_state)
    # "xgb" : lambda random_state:XGBClassifier(random_state=random_state),
    # "LR" : lambda random_state:LogisticRegression(solver='liblinear', random_state=random_state),
    #   "DT" : lambda random_state:DecisionTreeClassifier(random_state=random_state),
    #    "KNN" : lambda random_state:KNeighborsClassifier(),
    # "MLP": lambda random_state:MLPClassifier(random_state=random_state),
    # "SVC" : lambda random_state:svm.SVC(kernel = 'rbf', random_state=random_state, probability=True),
    
    
    
    }
clustering_algorithms = {
    # 'Kmeans' : lambda n_clusters, random_state: KMeans(n_clusters=n_clusters, random_state = 4, n_init = 1000),
    'Hierarchical' : lambda n_clusters, random_state : AgglomerativeClustering(n_clusters=n_clusters, linkage='average'),
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

def do_cluster(sc_X, adata_pam, cluster_alg_name, n_clusters, random_state):
    
    cluster_alg = clustering_algorithms[cluster_alg_name](n_clusters, random_state).fit(sc_X)
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
elif test_data == 'GSE':
    X, Y, available = get_X_Y(gse_data, gse_ptype, gse_ensemble, gse_pam_avail, True)
elif test_data == 'ACES':
    X, Y, available = get_X_Y(aces_data, aces_ptype, aces_ensemble, aces_pam_avail, True)
X = X.to_numpy()

adata_pam = adata[:, available]

results_all_cluster = {}

for cluster_alg_name, cluster_alg_func in clustering_algorithms.items():
    start_time = datetime.now()
    results_cluster_it = {}
    for n_clusters in range(cmin, cmax, 1):
        adata_pam = adata[:, available]
        combined_sc_clusters = pd.DataFrame()
        if remove_non_tumor:
            adata_pam = adata_pam[adata_pam.obs.tumor_status == 'Tumor']
            
        if remove_chemo_patient:
            adata_pam = adata_pam[adata_pam.obs.individual != 'BC05']
           
        if remove_lymphnode:
            adata_pam = adata_pam[adata_pam.obs.organism_part != 'lymph node']

        #Cluster cells into 20 groups using kmeans
        sc_X = adata_pam.layers['normalized'].todense()
        sc_data = do_cluster(sc_X, adata_pam, cluster_alg_name, n_clusters, random_state)
        sc_data = sc_data.to_numpy().T
        #decomposing bulk dataset using single cell
        decomposed_mat = []
        for i, patient in enumerate(X):
            reg = LinearRegression()
            reg.fit(sc_data, patient)
            decomposed_mat.append(reg.coef_)
        #converting into numpy array
        # decomposed_mat = np.append(X, decomposed_mat, axis = 1)
        decomposed_mat = np.array(decomposed_mat)
        print(test_data, n_clusters, decomposed_mat.shape)
        results_all = {}
        for model_name, model_func in models.items():
            results_it = {}
            for iteration in range(niter):
                results_metric = pd.DataFrame(0, index = ['AUC', 'F1', 'MCC', 'Kappa'],
                                              columns = [ 'bulk', 'decomp'])
                random_state = iteration
                model = model_func(random_state)
                skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
                
                pred_probas = []
                y_true, y_pred = [], []
                for cv_idx, (train_index, test_index) in enumerate(skf.split(X, Y)):
                    data_train, c_train, data_test, c_test = X[train_index], Y[train_index], X[test_index], Y[test_index]
                    model.fit(data_train, c_train)
                    pred_proba = model.predict_proba(data_test)[:, 1]
                    pred_probas.extend(pred_proba)
                    y_pred.extend(model.predict(data_test))
                    y_true.extend(c_test)
                auc, f1, mcc, kappa = performance_metrics(y_true, y_pred, pred_probas)
                results_metric.loc[:, 'bulk'] = [auc, f1, mcc, kappa]
                
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
        print( t.groupby(level = [0,1]).mean().loc[:, 'AUC', :])
        results_all = pd.concat(results_all)
        results_cluster_it[n_clusters] = results_all
        end_time = datetime.now()
        
    results_cluster_it = pd.concat(results_cluster_it)
    results_all_cluster[cluster_alg_name] = results_cluster_it
    print('\nDuration for each cluster: {}'.format(end_time - start_time))
results_all_cluster = pd.concat(results_all_cluster, names = ['clust_name', 'n_cluster', 'model', 'iteration', 'metric'])
results_all_cluster = results_all_cluster.reorder_levels([2, 0, 4, 1, 3]).sort_index()





#%% Subtype analysis
SEED = 2024
n_clusters = 23
remove_non_tumor = True
remove_lymphnode = False
remove_chemo_patient = True


test_data = 'ACES'
if test_data == 'NKI':
    X, Y, available = get_X_Y(nki_data, nki_ptype, nki_ensemble, nki_pam_avail, True)
    Y = np.array(nki_subtype)
elif test_data == 'GSE':
    X, Y, available = get_X_Y(gse_data, gse_ptype, gse_ensemble, gse_pam_avail, True)
elif test_data == 'ACES':
    X, Y, available = get_X_Y(aces_data, aces_ptype, aces_ensemble, aces_pam_avail, True)
    Y = np.array(aces_subtype)
X = X.to_numpy()

print("__________________________________________________")
print("Subtype accuracy for:", test_data, "\tshape:", X.shape)
print("__________________________________________________")
print('classifier \t\t\tbulk \t decomposed ')
print("___________________________________________________")

for model_name, model in models.items():
    model = model(SEED)
    skf = StratifiedKFold(n_splits=n_splits, random_state=SEED, shuffle=True)
    pred_probas1 = []
    y_true1, y_pred1 = [], []
    for cv_idx, (train_index, test_index) in enumerate(skf.split(X, Y)):
        data_train, c_train, data_test, c_test = X[train_index], Y[train_index], X[test_index], Y[test_index]
        model.fit(data_train, c_train)
        pred_proba1 = model.predict_proba(data_test)[:, 1]
        pred_probas1.extend(pred_proba)
        y_pred1.extend(model.predict(data_test))
        y_true1.extend(c_test)

    pred_probas2 = []
    y_true2, y_pred2 = [], []
    adata_pam = adata[:, available]
    combined_sc_clusters = pd.DataFrame()
    if remove_non_tumor:
        adata_pam = adata_pam[adata_pam.obs.tumor_status == 'Tumor']
        
    if remove_chemo_patient:
        adata_pam = adata_pam[adata_pam.obs.individual != 'BC05']
       
    if remove_lymphnode:
        adata_pam = adata_pam[adata_pam.obs.organism_part != 'lymph node']
    sc_X = adata_pam.layers['normalized'].todense()
    sc_data = do_cluster(sc_X, adata_pam, cluster_alg_name, n_clusters, random_state)
    sc_data = sc_data.to_numpy().T
    #decomposing bulk dataset using single cell
    decomposed_mat = []
    for i, patient in enumerate(X):
        reg = LinearRegression()
        reg.fit(sc_data, patient)
        decomposed_mat.append(reg.coef_)
    #converting into numpy array
    # decomposed_mat = np.append(X, decomposed_mat, axis = 1)
    decomposed_mat = np.array(decomposed_mat)
    for cv_idx, (train_index, test_index) in enumerate(skf.split(decomposed_mat, Y)):
        data_train, c_train, data_test, c_test = decomposed_mat[train_index], Y[train_index], decomposed_mat[test_index], Y[test_index]
        model.fit(data_train, c_train)
        pred_proba2 = model.predict_proba(data_test)[:, 1]
        pred_probas2.extend(pred_proba)
        y_pred2.extend(model.predict(data_test))
        y_true2.extend(c_test)
    
    print('\t', model_name, ' \t\t\t', round(accuracy_score(y_true1, y_pred1, normalize=True), 2), '\t\t',
          round(accuracy_score(y_true2, y_pred2, normalize=True), 2))
    
#%% Dendogram Finding the best K for aggomerative clustering

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Perform agglomerative clustering
linkage_matrix = linkage(adata_pam.layers['normalized'].todense(), method='ward')

# Create a dendrogram
dendrogram(linkage_matrix)

# Display the dendrogram
plt.show()
#%% Silhouette Score Finding the best K for aggomerative clustering

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
max_clusters = 50
silhouette_scores = []
cluster_range = range(10, max_clusters + 1)

for n_clusters in cluster_range:
    # Perform agglomerative clustering
    model = KMeans(n_clusters=n_clusters, n_init = 1000)
    labels = model.fit_predict(adata_pam.layers['normalized'].todense())

    # Calculate silhouette score
    silhouette_avg = silhouette_score(adata_pam.layers['normalized'].todense(), labels)
    silhouette_scores.append(silhouette_avg)
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Agglomerative Clustering')
plt.show()