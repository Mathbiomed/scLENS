import scanpy as sc
import pandas as pd
import numpy as np
from glob import glob
import os
from time import time

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")

tmp_dir = "/home/user/Downloads/tmp_collect_"

tmp_files = glob(os.path.join(tmp_dir,"*.csv.gz"))
# tmp_i = np.array([s.find("D496_BLD") for s in tmp_files ]) > 0
# f = np.array(tmp_files)[tmp_i][5]
f = tmp_files[4]
for f in tmp_files[0:]:
    # s_time = time()
    try:
        adata = sc.read_csv(f)
        b_name_ = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        adata.raw = adata
        
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata)
        sc.tl.pca(adata, svd_solver='arpack')
        # sc.pp.neighbors(adata)
        # sc.tl.leiden(adata)
        
        # sc.tl.umap(adata)
        # sc.tl.tsne(adata, use_rep="X_pca")
        # sc.pl.umap(adata, color=["leiden"], cmap="tab20")
        pca_df = pd.DataFrame(adata.obsm["X_pca"])
        # umap_df = pd.DataFrame(adata.obsm["X_umap"])
        # dir_path = "/home/user/Documents/Data/gene_test_data/comp_dr/"
        dir_path = "/home/user/Downloads/tmp_collect_"
        pca_file_name = os.path.join(dir_path , b_name_+"_scanpy_pca.csv" )
        pca_df.to_csv(pca_file_name)
        
    except:
        continue
    
    # e_time = time()
    # print(e_time-s_time)
    


