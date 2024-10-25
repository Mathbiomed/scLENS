# scLENS: Data-Driven Signal Detection for Unbiased scRNA-seq Data Analysis

## Introduction
scLENS (Single-cell Low-dimension Embedding using effective Noise Subtraction) is a dimensionality reduction tool specifically developed to address the challenges of signal distortion and manual bias in scRNA-seq data analysis. By incorporating L2 normalization and leveraging random matrix theory (RMT) for noise filtering, scLENS automatically detects biologically meaningful signals without requiring time-consuming parameter tuning. It excels in analyzing highly sparse and variable scRNA-seq datasets, ensuring accurate downstream analysis results.

scLENS supports both GPU and CPU execution, and if CUDA is not available, the package will automatically use the CPU.

## Requirements

To run this project, you will need the following:

- **Julia** (version 1.6 or higher recommended)
- **CUDA** for GPU processing (optional, if GPU support is required)

### GPU Requirements for CUDA
To use CUDA, you must have an NVIDIA GPU with CUDA capability, and the appropriate NVIDIA drivers must be installed on your system. If CUDA is not available, the package will automatically use the CPU.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mathbiomed/scLENS/
   cd scLENS
   ```

2. **Activate the project environment**:
   In the Julia REPL, navigate to the project directory and run:
   ```julia
   import Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

This will set up the required dependencies and activate the environment for the project.

### Converting 10x Data in gz Format to JLD2

If you have 10x data files in `gz` format and want to convert them to JLD2 format, you can use the `tenx2jld2` function. Follow the steps below:

1. **Prepare the 10x Data**:
   
   Make sure your data folder contains the following files:
   
   - `matrix.mtx.gz`
   - `features.tsv.gz`
   - `barcodes.tsv.gz`

2. **Convert the Data**:
   
   Use the `tenx2jld2` function to convert the 10x data into a JLD2 file:

   ```julia
   # Import necessary packages
   using scLENS

   # Example usage
   scLENS.tenx2jld2("/path/to/10x/data", "output_data.jld2")
   ```

   This command will save the converted data as a JLD2 file in the specified output location. The default output is `out_jld2/out.jld2`.

3. **Load the Converted Data**:
   
   To load the saved JLD2 data back into a DataFrame, use the following command:

   ```julia
   using JLD2

   # Load the DataFrame
   df = JLD2.load("output_data.jld2", "data")
   ```

   The `df` variable will now contain the DataFrame stored under the variable name `"data"`.


## Usage Example

Below is an example of how to use the scLENS package, including data loading, quality control (QC), embedding creation, and saving the results.

```julia
import Pkg
Pkg.activate(".")  # Activate the project environment in the current directory
Pkg.instantiate()  # Install required packages

# Load scLENS
using scLENS

# Device selection for GPU or CPU processing
using CUDA: has_cuda
cur_dev = if has_cuda()
    "gpu"
else
    "cpu"
end

# Load the compressed CSV file into a dataframe
ndf = scLENS.read_file("data/Z8eq.csv.gz")
# Alternatively, load with a gene dictionary
# ndf = scLENS.read_file("data/Z8eq.csv.gz", gid_file="gene_dictionary/gene_id.csv")

# Perform Quality Control (QC)
pre_df = scLENS.preprocess(ndf)

# Create an embedding using scLENS
sclens_embedding = scLENS.sclens(pre_df, device_=cur_dev)

# Plot the distribution of eigenvalues and clustering stability
scLENS.plot_mpdist(sclens_embedding)
scLENS.plot_stability(sclens_embedding)

# Apply UMAP transformation to the embedding
scLENS.apply_umap!(sclens_embedding)
panel_1 = scLENS.plot_embedding(sclens_embedding, pre_df.cell)

# Save the PCA and UMAP results to CSV files
using DataFrames
using CSV
CSV.write("out/pca.csv", sclens_embedding[:pca_n1])
CSV.write("out/umap.csv", DataFrame(sclens_embedding[:umap], :auto))

# Save the scLENS outcome as an AnnData file
scLENS.save_anndata("out/test_data.h5ad", sclens_embedding)

# Save the UMAP plot as an image
using PlotlyJS: savefig
savefig(panel_1, "out/umap_dist.png")
```

### Data Loading
scLENS currently supports **CSV** and **jld2** files for data loading.

- **CSV format**: The file should have rows representing cells and columns representing genes. The first row must contain gene names or IDs, and the first column must contain cell IDs.
  
- **jld2 format**: The file must contain a variable named `"data"`, which should be a DataFrame. The first column of the DataFrame should be named `:cell` and must represent cell IDs.

If you wish to change gene names, you can provide a second input to the `scLENS.read_file` function, as shown below:

```julia
ndf = scLENS.read_file("data/Z8eq.csv.gz", gene_id_file="path/to/gene_id.csv")
```

The `gene_id_file` must be in **CSV format** and should contain two columns: 
- `"gene"`: Original gene names
- `"gene_ID"`: Corresponding new gene names

This file should follow the structure of the `gene_dictionary/gene_id.csv` file.

### Quality Control (QC)
The **`preprocess`** function handles quality control. During this process, cells with fewer than 200 expressed genes and genes expressed in fewer than 15 cells are filtered out to ensure data quality.

### Plotting
scLENS supports the following plots:
- **MP distribution**: `scLENS.plot_mpdist(sclens_embedding)`
- **Signal stability**: `scLENS.plot_stability(sclens_embedding)`
- **UMAP embedding**: `scLENS.plot_embedding(sclens_embedding, pre_df.cell)`

To save these plots as images, you can use `PlotlyJS.savefig`.

### UMAP Application
You can apply UMAP using the `apply_umap!(sclens_embedding)` function. 

- To modify UMAP settings, such as the number of neighbors (`k`), output dimensions (`nc`), or minimum distance (`md`), you can adjust these parameters. Changing the metric is not recommended.

### Saving Results
The current version of scLENS supports saving results in:
- **CSV format** using `CSV.write()`
- **AnnData format** (compatible with Scanpy) using `scLENS.save_anndata()`

A function for saving results as a **Seurat object** will be available in a future version. If you need to convert the results to a Seurat object, please use version 1.0.0.

## Citation
If you use scLENS in your research, please cite the following paper:

**Kim, H., Chang, W., Chae, S.J. et al. scLENS: data-driven signal detection for unbiased scRNA-seq data analysis. Nat Commun 15, 3575 (2024). https://doi.org/10.1038/s41467-024-47884-3**
