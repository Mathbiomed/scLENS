# scLENS: Data-driven signal detection for unbiased scRNA-seq data analysis

## Table of Contents
0. [System requirements](#system-requirements)
1. [Installation of Julia Language](#installation-of-julia-language)
    - [Windows](#windows)
    - [Linux (Debian)](#linux-debian)
    - [MacOS](#macos)
2. [Download scLENS from GitHub](#download-sclens-from-github)
3. [Install Required Packages](#install-required-packages)
4. [Convert 10x Files](#convert-10x-files-to-a-jld-file)
5. [Run scLENS](#run-sclens)

## System requirements
- scLENS was tested on Windows 11 / MacOS 13.3 / Ubuntu 22.04.

## Installation of Julia Language
**Warning about Dependency Packages:** To ensure optimal performance and compatibility, it's recommended to use the latest versions of dependent packages. Potential compatibility issues may arise if outdated versions are used.

### Windows

1. **Download Julia**: Visit the [Julia Downloads page](https://julialang.org/downloads/).
2. **Install Julia**: Run the downloaded `.exe` file.
3. **Environment Path**: Check the box to add Julia to your system PATH during installation.

### Linux (Debian)

1. **Download Julia**: Use `wget` to download the pre-compiled binary.
    ```bash
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.1-linux-x86_64.tar.gz
    ```
2. **Install Julia**: Extract the downloaded archive.
    ```bash
    tar zxvf julia-1.10.1-linux-x86_64.tar.gz
    ```
3. **Environment Path**: Add Julia to your PATH. (Please change the name of home folder)
    ```bash
    echo 'export PATH="$PATH:$HOME/julia-1.10.1/bin"' >> ~/.bashrc
    source ~/.bashrc
    ```
This will install Julia and make it accessible from the command line.

### MacOS

1. **Download Julia**: Get the `julia-1.10.0-mac64.dmg` file from the Julia website, which will include `Julia-1.10.app`.
2. **Install Julia**: Drag `Julia-1.10.app` into your Applications folder. This is the standard installation process for Mac applications.
- Note: `Julia-1.10.app` is compatible with macOS 10.9 Mavericks and newer versions.
- For macOS 10.6 Snow Leopard and older, or for 32-bit systems, you can build Julia from source. However, these are not fully supported.
3. **Environment Path**: Add Julia to your PATH.
    ```bash
    sudo mkdir -p /usr/local/bin
    sudo rm -f /usr/local/bin/julia
    sudo ln -s /Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
    ```

## Download scLENS from GitHub

1. Navigate to the scLENS GitHub repository.
2. Click on the `Code` button and download the ZIP file or clone the repository using git:
    ```bash
    git clone https://github.com/Mathbiomed/scLENS.git
    ```

## Install Required Packages

1. Navigate to the downloaded `sclens` folder.
2. Open a terminal in that folder and run (Installation typically takes 5 to 10 minutes on a standard laptop.) :
    ```bash
    julia installation.jl
    ```

## Convert 10x Files to a JLD File

To convert 10x files to a JLD file, you can use the `convert_to_jld.jl` script. The folder containing your 10x files should include the following three files:

- `matrix.mtx.gz`
- `barcodes.tsv.gz`
- `features.tsv.gz`

Run the following command to perform the conversion:

```bash
julia convert_to_jld.jl /path/to/10xfiles/folder/ -d data/ -n test.jld2
```

**Note:**
-d specifies the directory where the output JLD file will be saved.
-n specifies the name of the output JLD file.

The file extension must always be .jld2.

**Output Information:**
After running the script convert_to_jld.jl, the resulting .jld2 file includes a DataFrame representing single-cell RNA sequencing (scRNA-seq) count data. The first column in this DataFrame is designated for the cell ID. Each row corresponds to an individual cell, and each column represents a specific gene. You can customize the cell IDs and gene IDs as needed by altering the barcodes.tsv.gz and features.tsv.gz files, respectively.

## Run scLENS

To run scLENS, you can use the following options:

- `--true_label`, `-t`: Specify a CSV file containing true cell-type labels for each cell. The number of labels must match the number of rows in the input matrix.  
  **Type**: String  
  **Default**: None

- `--plot`, `-p`: Enable this option to save three types of plots to the output directory: eigenvalue distribution, UMAP embedding, and signal stability plot.  
  **Action**: store_true

- `--device`: Select the computing device for scLENS. Options are 'gpu' or 'cpu'.  
  **Type**: String  
  **Default**: `gpu`

- `--out_dir`, `-d`: Specify the directory where output files will be saved.  
  **Type**: String  
  **Default**: `tempname()`
  
- `--scaling` : Choose the scaling method for gene-scaling. Options are "mean" for z-score scaling and "median" for median scaling.  
  **Type**: String  
  **Default**: `mean`

- `--out_type`, `-o`: Choose the format for the output files. Options are 'julia' (jld2), 'python' (npz), 'r' (RData), 'anndata' (h5ad), 'seurat' (rds), and 'csv'. Note that selecting 'csv' will only output PCA results.
  **Type**: String  
  **Default**: `julia`
  
  **Additional Details**:
  1. **Seurat Dependency**: To use the saving options for 'seurat', `seurat` should be installed in your R environments.
  2. **R Environment for Seurat**: The `R_HOME` environment variable for RCall.jl should point to the R home directory where `seurat` is installed. Refer to the [RCall.jl installation guide](https://juliainterop.github.io/RCall.jl/stable/installation/) for more details.
  
- `arg1`: Provide the path to the input file, which should be a CSV or JLD2 file. The file must contain a matrix with rows representing cells and columns representing genes.  
  **Required**: true


Example usage:

```bash
julia scLENS.jl data/your_dataset.csv.gz --true_label data/your_labels.csv --out_dir out_dir --device cpu --out_type julia --plot
```
**Note**: The time required for analysis using scLENS varies depending on the size of the dataset.

## Output Details

After successfully running scLENS, a single output file will be generated in the directory specified by the `--out_dir` option. The format of this output file is determined by the `--out_type` option. This file will contain a dictionary with the following key-value pairs:

- **`pca`**: Reduced data after noise filtering based on Random Matrix Theory (RMT).
- **`pca_n1`**: Reduced data after both RMT noise filtering and signal stability tests.
- **`sig_id`**: Identifiers for robust signals.
- **`L`**: All eigenvalues.
- **`L_mp`**: Eigenvalues related to noise.
- **`λ`**: Tracy-Widom (TW) threshold value.
- **`st_mat`**: Signal stability matrix.
- **`m_scores`**: Mean stability vector.
- **`sd_scores`**: Standard deviation stability vector.
- **`signal_evec`**: Eigenvectors corresponding to signals.
- **`signal_ev`**: Eigenvalues corresponding to signals.
- **`cell_id`**: Identifiers for the input cells.
- **`umap`**: 2D UMAP coordinates.
- **`umap_obj`**: UMAP object including a UMAP graph (only included in JLD2 files).

**Note**: 
- Please utilize the reduced data with the key "pca_n1" for your analyses, rather than the data with the key "pca."
- If you wish to save the output as an RData file, make sure to install the R language on your system.

This single output file offers a comprehensive set of variables useful for analyzing and interpreting the results generated by scLENS.
