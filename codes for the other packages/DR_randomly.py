#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:31:47 2022

@author: khlab
"""

import randomly
import pandas as pd
import numpy as np
from glob import glob
import os
from time import time

tmp_dir = "/home/user/Downloads/noise_data2/preprocessed"
tmp_files = glob(os.path.join(tmp_dir,"*.csv.gz"))
f = tmp_files[4]
for f in tmp_files[5:]:
    df_ = pd.read_csv(f,compression=None)
    # df_ = pd.read_csv(f)
    df = df_.iloc[:,1:]
    b_name_ = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
    model = randomly.Rm()
    model.preprocess(df, min_tp=0, 
                     min_genes_per_cell=0, 
                     min_cells_per_gene=0,
                    refined=True)
    model.refining(min_trans_per_gene=7)
    model.fit()
    # model.plot_mp()
    # model.plot_statistics()
    
    df2 = model.return_cleaned(fdr=0.0001)
    df2.to_csv("/home/khlab/Documents/Data/gene_test_data/comp_dr/tmp_randomly.csv")
    