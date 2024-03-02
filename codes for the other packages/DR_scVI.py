import scanpy as sc
import pandas as pd
import numpy as np
from glob import glob
import os
from time import time

import matplotlib.pyplot as plt
import seaborn as sns

import scvi
# import scib


sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")

tmp_dir = "/home/user/Downloads/tmp_collect_"

tmp_files = glob(os.path.join(tmp_dir,"*.csv.gz"))
f = tmp_files[4]
for f in tmp_files[0:]:
    # s_time = time()
    try:
        adata = sc.read_csv(f)
        b_name_ = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
        adata.layers['counts'] = adata.X.copy() # move count data into a layer
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata) # log the data for better umap visualization later
        adata.raw = adata        
        
        sc.pp.highly_variable_genes(adata,
        n_top_genes=10000,
        subset=True,
        inplace=True,
        flavor="seurat_v3",
        layer='counts',
        )
       
        scvi.model.SCVI.setup_anndata(adata)
        adata.X
        vae = scvi.model.SCVI(adata)
        vae.train()
        pd.DataFrame(vae.get_latent_representation()).to_csv("/home/user/Downloads/tmp3.csv")
        
    
    except:
        continue
    
    # e_time = time()
    # print(e_time-s_time)
    


