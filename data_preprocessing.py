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

