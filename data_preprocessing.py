#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:55:13 2023

@author: tanzira
"""

#%%All imports
import pandas as pd
import numpy as np
import anndata
import mygene

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score

from datetime import datetime
#%%HNSCC single cell data
df = pd.read_csv('Data/SC_HNCC/HNSCC_all_data.txt',
                       sep = '\t', index_col = 0, header=0, skiprows=(1, 2, 3, 4, 5)).T

observations = pd.read_csv('Data/SC_HNCC/HNSCC_all_data.txt',
                       sep = '\t', index_col = 0, nrows=5).T
print(df.shape)
print(observations)
df.index = df.columns.str.replace('"', '')
adata = anndata.AnnData(df)

adata.obs = observations

#%%Read single cell data
countfile = 'Data/SC_BRCA/Wu_etal_2021_BRCA_scRNASeq/count_matrix_sparse.mtx'
metadata_file = 'Data/SC_BRCA/Wu_etal_2021_BRCA_scRNASeq/metadata.csv'
varfile = 'Data/SC_BRCA/Wu_etal_2021_BRCA_scRNASeq/count_matrix_genes.tsv'
var = pd.read_csv(varfile, header = None, index_col= 0, sep = '\t',  names = ['gene'])

adata = anndata.read_mtx(countfile).T
exp_design = pd.read_csv(metadata_file, index_col = 0)

adata.obs = exp_design
adata.var = var

#%%Convert any gene id to ensemble
#This is the code for converting entrezgene id to ensemble gene id
scopes_list = ['entrezgene', 'symbol']

def geneid_to_ensemble_id(gene_ids, scopes = 'entrezgene'):
    mg = mygene.MyGeneInfo()
    geneSyms = mg.querymany(gene_ids, 
                            scopes = scopes, 
                            fields='ensembl.gene', 
                            species='human')
    
    symbol_dict = {}
    for item in geneSyms:
        k = item['query']
        if scopes == 'entrezgene':
            k = int(item['query'])
            
        if 'notfound' in item or 'ensembl' not in item:
            print('Query not found: ', k)
            continue
            
        elif type(item['ensembl']) == list:
            v = item['ensembl'][-1]['gene']
        else:
            v = item['ensembl']['gene']
        symbol_dict[k] = v
    
    anygene_id_to_ensemble = pd.DataFrame(list(symbol_dict.items()), columns = [scopes, 'ensemble'])
    anygene_id_to_ensemble.set_index('ensemble', inplace = True)
    return anygene_id_to_ensemble

# gene_ids = tcga_data.columns
# gene_ids = tcga_data.columns
# gene_ids = cosmic_genes['Entrez GeneId']
# gene_ids = cosmic_genes['Gene Symbol']
# gene_ids = diff_brca_genes['Gene']
# gene_ids = quickgo_emt_genes['SYMBOL']

# gene_ids = gse_data.columns
# gene_ids = tcga_data.columns
# gene_ids = var['gene']
# gene_ids = yau_data.index
# gene_ids = vijver_data.columns
# geneid_to_ensemble = geneid_to_ensemble_id(gene_ids, scopes = scopes_list[1])#Change scopes for entrezid

#Change file name according to the dataset
# geneid_to_ensemble.to_csv('brca_sc_symbol_to_ensemble.txt')

#%%Entrez to symbol
from Bio import Entrez

def entrez_id_to_symbol(entrez_id):
    try:
        # Provide your email to identify yourself to NCBI
        Entrez.email = "your_email@example.com"
        
        # Fetch gene information using Entrez.efetch
        handle = Entrez.efetch(db="gene", id=entrez_id, rettype="gene_table", retmode="text")
        record = handle.read().strip().split('\n')[1]  # Get the second line (gene symbol)
        
        # Extract gene symbol from the record
        gene_symbol = record.split('\t')[1]
        
        return gene_symbol
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
entrez_id = "12345"  # Replace with your Entrez Gene ID
gene_symbol = entrez_id_to_symbol(entrez_id)

if gene_symbol:
    print(f"Entrez Gene ID {entrez_id} corresponds to Gene Symbol: {gene_symbol}")
else:
    print(f"Unable to retrieve gene symbol for Entrez Gene ID {entrez_id}")
#%%Filter the dataset
pam_50_genes = pd.read_csv('Data/PAM50GenesWitnEnsemble.txt', index_col = 0)
adata_pam_avail = pd.read_csv('Data/brca_sc_symbol_to_ensemble.txt', index_col=(0))
# sc_pam_avail = np.intersect1d(adata_pam_avail.index, adata_pam_avail.index)
intersection_df = pd.merge(adata_pam_avail, pam_50_genes, on='ensemble', how='inner')
adata_pam = adata[:, intersection_df.symbol]


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
#%%Only NKI
n_splits = 10
random_state = 4
niter = 2
cmin, cmax = 20, 45

remove_non_tumor = True
remove_lymphnode = False
remove_chemo_patient = True

datasets = ['NKI']

test_data = datasets[0]
print(test_data)

models = {
    "RF" :lambda random_state: RandomForestClassifier(random_state = random_state),
    }
clustering_algorithms = {
    'Kmeans' : lambda n_clusters, random_state: KMeans(n_clusters=n_clusters, random_state = random_state, n_init = 100),
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
    sc_data_raw = pd.DataFrame(adata_pam.X.todense(), 
                      index = adata_pam.obs['cell_cluster_type_{0}'.format(n_clusters)], 
                      columns = adata_pam.var.index)
    sc_data = sc_data_raw.groupby(['cell_cluster_type_{0}'.format(n_clusters)]).mean()
    return sc_data


if test_data == 'NKI':
    nki_pam_avail = pd.merge(nki_ensemble, intersection_df, on='ensemble', how='inner')
    X, Y, available = get_X_Y(nki_data, nki_ptype, nki_ensemble, nki_pam_avail.index, True)
    available = np.intersect1d(adata_pam_avail.index, available)

    
elif test_data == 'GSE':
    X, Y, available = get_X_Y(gse_data, gse_ptype, gse_ensemble, gse_pam_avail, True)
    

X = X.to_numpy()

# adata_pam = adata[:, available.symbol.values]

results_all_cluster = {}

for cluster_alg_name, cluster_alg_func in clustering_algorithms.items():
    start_time = datetime.now()
    results_cluster_it = {}
    for n_clusters in range(cmin, cmax, 1):
        #Cluster cells into 20 groups using kmeans
        sc_X = adata_pam.X.todense()
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