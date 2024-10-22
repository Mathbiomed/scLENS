import Pkg
Pkg.activate(".") # Activate the project environment in the current directory
Pkg.instantiate() # Install dep pacakges

# Load scLENS
using scLENS

# Device selection for GPU or CPU processing
using CUDA:has_cuda
cur_dev = if has_cuda()
    "gpu"
else
    "cpu"
end

# Load the compressed CSV file into a dataframe
ndf = scLENS.read_file("data/Z8eq.csv.gz")
# ndf = scLENS.read_file("data/Z8eq.csv.gz",gid_file="gene_dictionary/gene_id.csv")

# Perform data preprocessing
pre_df = scLENS.preprocess(ndf)

# Create an embedding using scLENS
sclens_embedding = scLENS.sclens(pre_df,device_=cur_dev)

scLENS.plot_mpdist(sclens_embedding)
scLENS.plot_stability(sclens_embedding)

# Apply UMAP transformation
scLENS.apply_umap!(sclens_embedding)
panel_1 = scLENS.plot_embedding(sclens_embedding,pre_df.cell)

# Save the PCA results to a CSV file
using CSV
CSV.write("out/pca.csv",sclens_embedding[:pca_n1])
CSV.write("out/umap.csv",DataFrame(sclens_embedding[:umap],:auto)) # Save the UMAP results to a CSV file

# Save scLENS outcome as anndata
scLENS.save_anndata("out/test_data.h5ad",sclens_embedding)

# the UMAP distribution as an image
using PlotlyJS:savefig
savefig(panel_1,"out/umap_dist.png")