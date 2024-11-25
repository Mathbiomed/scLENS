module scLENS

using DataFrames
using CUDA
using SparseArrays
using CSV
using StatsBase
using ProgressMeter
using Random:shuffle, shuffle!
using LinearAlgebra
using Distributions:Normal
using NaNStatistics: histcounts, nanmaximum
using UMAP:UMAP_
using Distances:CosineDist
using JLD2:save as jldsave, load as jldload
using CairoMakie
using ColorSchemes
using Muon
using MatrixMarket
using GZip


"""
The `read_file` function reads count matrix files in either CSV or JLD2 format.

## Arguments
- `test_file::String`: The path to the file containing data (CSV or JLD2 format). For CSV files, rows should represent cells and columns should represent genes. For JLD2 files, the file must contain a variable named `"data"`, which is a DataFrame.
- `gid_file=nothing`: (optional) Path to a CSV file containing new gene names. If `gid_file=nothing`, gene names from the original data will not be modified.

## File Formats
- **CSV format**: Rows represent cells, and columns represent genes. The first row must contain gene names or IDs, and the first column must contain cell IDs.
- **JLD2 format**: The file should contain a variable named `"data"` as a DataFrame. The first column in this DataFrame should be named `:cell` and must represent cell IDs.

## Using a Gene Dictionary
To change gene names, you can provide a second argument to the `scLENS.read_file` function as follows:

```julia
ndf = scLENS.read_file("data/Z8eq.csv.gz", gid_file="path/to/gene_id.csv")
```

The `gene_id_file` must be in **CSV format** and should contain two columns:
- `"gene"`: Original gene names
- `"gene_ID"`: Corresponding new gene names

This file should follow the structure of the `gene_dictionary/gene_id.csv` file.

## Example
```julia
# Load the compressed CSV file into a dataframe
ndf = scLENS.read_file("data/Z8eq.csv.gz")

# Alternatively, load with a gene dictionary
ndf = scLENS.read_file("data/Z8eq.csv.gz", gid_file="gene_dictionary/gene_id.csv")
```

"""
function read_file(test_file::String;gid_file=nothing)
    fname_base = splitext(split(test_file,"/")[end])[1]
    println(fname_base)
    ndf = if occursin("csv",test_file)
        ndf = CSV.read(test_file,DataFrame,buffer_in_memory=true)
        if !("cell" in names(ndf))
            println("Warning: The first column should contain cell IDs, and the column name should be 'cell.' However, the 'cell' column was not found. It is recommended to check the file.")
        end
        rename!(ndf,names(ndf)[1] => :cell)
        tmp_gname = change_gname(names(ndf)[2:end],gid_file)
        rename!(ndf,["cell";tmp_gname],makeunique=true)
        return ndf
    elseif occursin("jld2",test_file)
        ndf = jldload(test_file,"data")
        if !("cell" in names(ndf))
            println("Warning: The first column should contain cell IDs, and the column name should be 'cell.' However, the 'cell' column was not found. It is recommended to check the file.")
        end
        tmp_gname = change_gname(names(ndf)[2:end],gid_file)
        rename!(ndf,["cell";tmp_gname],makeunique=true)
        return ndf
    end
end

function change_gname(g_names,inp_file=nothing)
    if isnothing(inp_file)
        return g_names
    else
        gene_iid = CSV.read(inp_file,DataFrame,buffer_in_memory=true)
        g_dict = Dict(gene_iid.gene_ID .=> gene_iid.gene)
        return [s in keys(g_dict) ? g_dict[s] : s for s in g_names]
    end
end

function df2sparr(inp_df::DataFrame;for_df=false)
    if issparse(inp_df[:,2])
        if for_df
            a_=Vector{Int64}([])
            b_=Vector{Int64}([])
            c_=Vector{Float32}([])
            for (i,s) in enumerate(eachcol(inp_df[!,2:end]))
                append!(a_, ones(Int64,lastindex(s.nzind)).*i)
                append!(b_, Int64.(s.nzind))
                append!(c_, Float32.(s.nzval))
            end
            sparse(b_,a_,c_,size(inp_df,1),size(inp_df,2)-1)
        else
            a_=Vector{UInt32}([])
            b_=Vector{UInt32}([])
            c_=Vector{Float32}([])
            for (i,s) in enumerate(eachcol(inp_df[!,2:end]))
                append!(a_, ones(UInt32,lastindex(s.nzind)).*i)
                append!(b_, UInt32.(s.nzind))
                append!(c_, Float32.(s.nzval))
            end
            sparse(b_,a_,c_,size(inp_df,1),size(inp_df,2)-1)
        end
    else
        if for_df
            SparseMatrixCSC{Float32,Int64}(mat_(inp_df))
        else
            SparseMatrixCSC{Float32,UInt32}(mat_(inp_df))
        end
    end
end


mat_(x::DataFrame) = Matrix{Float32}(x[!,2:end])
mat_(x::SubDataFrame) = Matrix{Float32}(x[!,2:end])
sparsity_(A::Matrix) = 1-length(nonzeros(SparseMatrixCSC(A)))/length(A)
sparsity_(A::SparseMatrixCSC) = 1-length(nonzeros(A))/length(A)
sparsity_(A::Vector) = 1-sum(.!iszero.(A))/length(A)
sparsity_(A::SparseVector) = 1-sum(.!iszero.(A))/length(A)
sparsity_(A::DataFrame) = issparse(A[!,2]) ? 1-length(nonzeros(df2sparr(A)))/length(df2sparr(A)) : sparsity_(mat_(A))
"""
`preprocess(tmp_df; min_tp_c=0, min_tp_g=0, max_tp_c=Inf, max_tp_g=Inf,
    min_genes_per_cell=200, max_genes_per_cell=0, min_cells_per_gene=15, mito_percent=5.,
    ribo_percent=0.)`

The `preprocess` function filters and cleans a given count matrix DataFrame to retain high-quality cells and genes based on various criteria.

## Arguments
- `tmp_df`: A DataFrame where each row represents a cell, and each column (except the first) represents a gene. The first column contains cell IDs, and subsequent columns contain gene expression values.
- `min_tp_c`: Minimum total counts per cell. Cells with fewer counts are filtered out.
- `max_tp_c`: Maximum total counts per cell. Cells with counts exceeding this value are filtered out.
- `min_tp_g`: Minimum total counts per gene. Genes with fewer counts are filtered out.
- `max_tp_g`: Maximum total counts per gene. Genes with counts exceeding this value are filtered out.
- `min_genes_per_cell`: Minimum number of genes per cell. Only cells with at least this number of expressed genes are retained.
- `max_genes_per_cell`: Maximum number of genes per cell. Cells with more than this number of expressed genes are filtered out.
- `min_cells_per_gene`: Minimum number of cells per gene. Only genes expressed in at least this number of cells are retained.
- `mito_percent`: Upper threshold for mitochondrial gene expression as a percentage of total cell expression. Cells exceeding this threshold are filtered out.
- `ribo_percent`: Upper threshold for ribosomal gene expression as a percentage of total cell expression. Cells exceeding this threshold are filtered out.

## Example
```julia
# Basic preprocessing with default filtering parameters
filtered_df = preprocess(tmp_df)

# Advanced preprocessing with custom mitochondrial and ribosomal thresholds
filtered_df = preprocess(tmp_df, mito_percent=10, ribo_percent=5)
```

This function is designed to improve data quality by filtering cells and genes that do not meet specified quality criteria, enabling more reliable downstream analyses.
"""
function preprocess(tmp_df;min_tp_c=0,min_tp_g=0,max_tp_c=Inf,max_tp_g=Inf,
    min_genes_per_cell=200,max_genes_per_cell=0,min_cells_per_gene=15,mito_percent=5.,
    ribo_percent=0.)

    cell_name = tmp_df.cell
    gene_name = names(tmp_df)[2:end]
    size_ndf = (size(tmp_df))
    sp_ndf = sparsity_(tmp_df)
    println("Inp_spec")
    println("data size: $size_ndf, sparsity: $sp_ndf")

    X = if sp_ndf < 0.3
        mat_(tmp_df)
    else
        SparseMatrixCSC{Float32,Int64}(df2sparr(tmp_df))
    end

    tmp_df = if issparse(X)
        DataFrame(X,gene_name)
    else
        tmp_df[!,2:end]
    end

    c_nonz(X,dim) = sum(X .!= 0,dims=dim)[:]
    n_cell_counts = c_nonz(X,1)
    n_cell_counts_sum = sum(X,dims=1)[:]
    bidx_1 = n_cell_counts_sum .> min_tp_g
    bidx_2 = n_cell_counts_sum .< max_tp_g
    bidx_3 = n_cell_counts .>= min_cells_per_gene
    fg_idx = bidx_1[:] .& bidx_2[:] .& bidx_3[:]

    n_gene_counts = c_nonz(X,2)
    n_gene_counts_sum = sum(X,dims=2)[:]
    bidx_1 = n_gene_counts_sum .> min_tp_c
    bidx_2 = n_gene_counts_sum .< max_tp_c
    bidx_3 =  n_gene_counts .>= min_genes_per_cell
    bidx_mito = occursin.(r"^(?i)mt-.",gene_name)
    bidx_ribo = occursin.(r"^(?i)RP[SL].",gene_name)
    if mito_percent == 0
        bidx_4 = ones(Bool,size(bidx_1))
    else
        bidx_4 = sum(X[:,bidx_mito],dims=2)[:]./sum(X,dims=2)[:] .< mito_percent/100 #drop mito
    end

    if ribo_percent == 0
        bidx_5 = ones(Bool,size(bidx_1))
    else
        bidx_5 = sum(X[:,bidx_ribo],dims=2)[:]./sum(X,dims=2)[:] .< ribo_percent/100 #drop mito
    end

    if max_genes_per_cell == 0
        bidx_6 = ones(Bool,size(bidx_1))
    else
        bidx_6 = n_gene_counts .< max_genes_per_cell
    end

    fc_idx = bidx_1[:] .& bidx_2[:] .& bidx_3[:] .& bidx_4[:] .& bidx_5[:] .& bidx_6[:]

    if any(fc_idx) && any(fc_idx)
        oo_X = X[fc_idx,fg_idx]
        nn_idx = (sum(oo_X,dims=1) .!= 0)[:]
        oo_X = oo_X[:,nn_idx]
        norm_gene = gene_name[fg_idx][nn_idx]

        s_idx = sortperm(mean(oo_X,dims=1)[:])
        o_df = DataFrame(oo_X[:,s_idx],norm_gene[s_idx])
        insertcols!(o_df,1,:cell => cell_name[fc_idx])

        size_odf = (size(o_df))
        sp_odf = sparsity_(o_df)
        println("After filtering>> data size: $size_odf, sparsity: $sp_odf")
        return o_df
    else
        println("There is no high quality cells and genes")
        return nothing
    end
end


function _random_matrix(X;dims=1)
    if issparse(X)
        if dims==1
            # m,n = size(X)
            nz_row, nz_col, nz_val = findnz(X)
            ldict = countmap(nz_col)
            n_ldict = keys(ldict)
            # println(sort(collect(keys(ldict))))
            row_i = vcat([sample(1:lastindex(X,1),ldict[s],replace=false) for s in n_ldict]...)
            sparse(row_i,nz_col,nz_val)
        elseif dims == 2
            nz_row, nz_col, nz_val = findnz(X)
            ldict = countmap(nz_row)
            n_ldict = keys(ldict)
            col_i = vcat([sample(1:lastindex(X,2),ldict[s],replace=false) for s = 1:n_ldict]...)
            sparse(nz_row,col_i,nz_val)
        end
    else
        mapslices(shuffle!,X,dims=dims)
    end
end

function random_nz(pre_df;rmix=true,mix_p=nothing)
    tmp_X, return_mat = if typeof(pre_df) <: DataFrame
        df2sparr(pre_df), false
    else
        sparse(pre_df), true
    end

    nz_row, nz_col, nz_val = findnz(tmp_X)
    nz_idx = sparse(nz_row,nz_col,true)
    if !isnothing(mix_p)
        tmp_i = findall(nz_idx)[sample(1:sum(nz_idx),Int(ceil(sum(nz_idx)*(1-mix_p))),replace=false)]
        nz_idx[tmp_i] .= false
    end

    tmp_X.nzval .= shuffle(nz_val)
    tmp_X = if rmix
        _random_matrix(tmp_X)
    else
        SparseMatrixCSC{Float32,Int64}(tmp_X)
    end

    if !return_mat
        tmp_df = DataFrame(tmp_X,names(pre_df)[2:end])
        insertcols!(tmp_df,1,:cell => pre_df.cell)
        return tmp_df
    else
        return tmp_X
    end
end

function scaled_gdata(X;dim=1, position_="mean")
    tmp_mean = if position_ == "mean"
        mean(X,dims=dim)
    elseif position_ == "median"
        if issparse(X)
            mapslices(x -> nnz(x) > length(x)/2 ? median(x) : Float32(0.0), X,dims=dim)
        else
            mapslices(median, X,dims=dim)
        end
    elseif position_ == "cent"
        mean(X,dims=dim)
    end

    if position_ == "cent"
        return @. (X - tmp_mean)
    else
        tmp_std = std(X,dims=dim)
        if isnothing(tmp_mean)
            if dim == 1
                return X./tmp_std
            elseif dim == 2
                return tmp_std./X
            end
        elseif issparse(X) && (position_ == "median")
            new_nz = zeros(length(X.nzval))
            for i = 1:lastindex(X,2)
                sub_ii = X.colptr[i]:X.colptr[i+1]-1
                new_nz[sub_ii] = (X.nzval[sub_ii] .- tmp_mean[i]) ./ (tmp_std[i])
                X[:,i].nzind
            end
            new_X = copy(X)
            new_X.nzval .= new_nz

            nz_a = sparse(- tmp_mean ./ tmp_std) .* iszero.(new_X)
            return new_X + nz_a
        else
            return @. (X - tmp_mean) / (tmp_std)
        end
    end
end


function _wishart_matrix(X;device="gpu")
    if device == "gpu"
        gpu_x = CuMatrix{Float32}(X)
        out = CuArray{Float32,2}(undef, size(X,1),size(X,1))
        mul!(out,gpu_x,gpu_x')
        return Matrix(out) ./ size(X,2)
    elseif device == "cpu"
        X = if issparse(X)
            Matrix{eltype(X)}(X)
        else
            X
        end
        out = Array{eltype(X),2}(undef, size(X,1),size(X,1))
        mul!(out,X,X')
        return out ./ size(X,2)
    end
end

function corr_mat(X,Y;device="gpu")
    if device == "gpu"
        gpu_x = cu(X)
        gpu_y = cu(Y)
        out = CuArray{Float32,2}(undef, size(gpu_x,2),size(gpu_y,2))
        mul!(out,gpu_x',gpu_y)
        return Matrix(out)
    elseif device == "cpu"
        return X'*Y
    end
end

function _get_eigen(Y;device="gpu")
    if device == "gpu"
        tmp_L, tmp_V = CUDA.CUSOLVER.syevd!('V','U',cu(Y))
        tmp_L,tmp_V = Array{Float32,1}(tmp_L), Array{Float32,2}(tmp_V)
        if !isnothing(findfirst(isnan.(tmp_L)))
            tmp_L, tmp_V = eigen(convert.(Float64,Y))
        end
        return tmp_L, tmp_V
    elseif device == "cpu"
        tmp_L, tmp_V = eigen(Y)
        return tmp_L, tmp_V
    end
end


function _mp_parameters(L)
    moment_1 = mean(L)
    moment_2 = mean(L.^2)
    gamma = moment_2 / moment_1^2 - 1
    s = moment_1
    sigma = moment_2
    b_plus = s * (1 + sqrt(gamma))^2
    b_minus = s * (1 - sqrt(gamma))^2
    x_peak = s * (1.0-gamma)^2.0 / (1.0+gamma)
    dic = Dict("moment_1" => moment_1,
          "moment_2" => moment_2,
          "gamma" => gamma,
          "b_plus" => b_plus,
          "b_minus" => b_minus,
          "s" => s,
          "peak" => x_peak,
          "sigma" => sigma
          )
end


function _marchenko_pastur(x,y)
    if y["b_minus"]<x<y["b_plus"]
        pdf = sqrt((y["b_plus"] - x) * (x - y["b_minus"])) /
         (2 * y["s"] * pi * y["gamma"] * x)
    else
        pdf = 0
    end
end

function _mp_pdf(x,L)
    _marchenko_pastur.(x,Ref(_mp_parameters(L)))
end

function _mp_calculation(L, Lr, eta=1, eps=1e-6, max_iter=10000)
    converged = false
    iter = 0
    loss_history = []
    mpp_Lr = _mp_parameters(Lr)
    b_plus = mpp_Lr["b_plus"]
    b_minus = mpp_Lr["b_minus"]
    L_updated = L[b_minus .< L .< b_plus]
    new_mpp_L = _mp_parameters(L_updated)
    new_b_plus = new_mpp_L["b_plus"]
    new_b_minus = new_mpp_L["b_minus"]

    while ~converged
        loss = (1 - new_b_plus / b_plus)^2
        push!(loss_history,loss)
        iter += 1
        if loss <= eps
            converged = true
        elseif iter == max_iter
            println("Max interactions exceeded!")
            converged = true
        else
            gradient = new_b_plus - b_plus
            new_b_plus = b_plus + eta * gradient
            L_updated = L[new_b_minus .< L .< new_b_plus]
            b_plus = new_b_plus
            b_minus = new_b_minus
            up_mpp_L = _mp_parameters(L_updated)
            new_b_plus = up_mpp_L["b_plus"]
            new_b_minus = up_mpp_L["b_minus"]
        end
    end
    b_plus = new_b_plus
    b_minus = new_b_minus
    return L[new_b_minus .< L .< new_b_plus], b_plus, b_minus
end

function _tw(L, L_mp)
    gamma = _mp_parameters(L_mp)["gamma"]
    p = length(L) / gamma
    sigma = 1 / p^(2/3) * gamma^(5/6) * (1 + sqrt(gamma))^(4/3)
    lambda_c = mean(L_mp) * (1 + sqrt(gamma))^2 + sigma
    return lambda_c, gamma, p, sigma
end



function mp_check(test_L,p_val = 0.05)
    bin_x = LinRange(minimum(test_L)-1,maximum(test_L)+1,100)
    count_ = histcounts(test_L,bin_x)
    pdf_arr = count_./sum(count_)
    cdf_arr = cumsum(pdf_arr)

    new_binx = (bin_x[2:end] .+ bin_x[1:end-1])./2

    mp_pdf = x ->_mp_pdf(x, test_L)
    c_cdf2 = cumsum(mp_pdf.(new_binx))
    nc_cdf2 = c_cdf2./maximum(c_cdf2)
    
    D_ = maximum(abs.(cdf_arr .- nc_cdf2))
    c_α = sqrt(-1/2*log(p_val))
    m = length(cdf_arr)
    n = length(nc_cdf2)

    return Dict(:ks_static => D_,:pass => D_ <= c_α*sqrt((m+n)/m/n))
end

function get_eigvec(X;device="gpu")
    N, M = size(X)
    if N > M
        Y = _wishart_matrix(X',device=device)
        ev_, V = _get_eigen(Y,device=device)
        L = ev_
        positive_idx = L .> 0
        L = L[positive_idx]
        V = V[:,positive_idx]
        
        nLi = sortperm(L,rev=true)
        nL = L[nLi]
        nVs = V[:,nLi]

        mul_X = nVs.*sqrt.(1 ./nL)'
        new_nVs = try
            mapslices(s -> s/norm(s),Matrix(cu(X)*cu(mul_X)),dims=1)
        catch
            mapslices(s -> s/norm(s),Matrix{Float32}(X*mul_X),dims=1)
        end

        return nL, new_nVs
    else
        Y = _wishart_matrix(X,device=device)
        ev_, V = _get_eigen(Y,device=device)
        L = ev_
        positive_idx = L .> 0
        L = L[positive_idx]
        V = V[:,positive_idx]

        nLi = sortperm(L,rev=true)
        nL = L[nLi]
        nVs = V[:,nLi]
        return nL, nVs
    end
end

function get_sigev(X,Xr;device="gpu")
    n,m = size(X)
    if n > m
        Y = _wishart_matrix(X',device=device)
        ev_, V = _get_eigen(Y,device=device)
        Yr = _wishart_matrix(Xr',device=device)
        evr_, _ = _get_eigen(Yr,device=device)

        L = ev_
        Lr = evr_

        L_mp, _, b_min = _mp_calculation(L, Lr[1:end-1])
        lambda_c, _ = _tw(L,L_mp)
        println("(Using $device) number of signal ev: $(sum(L .> lambda_c))")

        sel_L = L[L .> lambda_c]
        sel_Vs = V[:, L .> lambda_c]

        noiseL = L[b_min .<= L .<= lambda_c]
        noiseV = V[:,b_min .<= L .<= lambda_c]

        nLi = sortperm(sel_L,rev=true)
        nLi2 = sortperm(noiseL,rev=true)

        nL = sel_L[nLi]
        nVs = sel_Vs[:,nLi]

        snL = noiseL[nLi2]
        noiseVs = noiseV[:,nLi2]
        
        mul_X = nVs.*sqrt.(1 ./nL)'
        mul_X2 = noiseVs.*sqrt.(snL)'
        new_nVs = mapslices(s -> s/norm(s),Matrix(cu(X)*cu(mul_X)),dims=1)
        CUDA.reclaim()
        new_noiseV = try
            mapslices(s -> s/norm(s),Matrix(cu(X)*cu(mul_X2)),dims=1)    
        catch
            mapslices(s -> s/norm(s),Matrix(X*mul_X2),dims=1)
        end
        
        return nL, new_nVs, L, L_mp, lambda_c, snL, new_noiseV
        
    else
        Y = _wishart_matrix(X,device=device)
        ev_, V = _get_eigen(Y,device=device)
        Yr = _wishart_matrix(Xr,device=device)
        evr_, _ = _get_eigen(Yr,device=device)
        L = ev_
        Lr = evr_

        L_mp, _, b_min = _mp_calculation(L, Lr[1:end-1])
        lambda_c, _ = _tw(L,L_mp)
        println("(Using $device) number of signal ev: $(sum(L .> lambda_c))")

        sel_L = L[L .> lambda_c]
        sel_Vs = V[:, L .> lambda_c]

        noiseL = L[b_min .<= L .<= lambda_c]
        noiseV = V[:,b_min .<= L .<= lambda_c]

        nLi = sortperm(sel_L,rev=true)
        nLi2 = sortperm(noiseL,rev=true)

        nL = sel_L[nLi]
        nVs = sel_Vs[:,nLi]
        
        return nL, nVs, L, L_mp, lambda_c, noiseL[nLi2], noiseV[:, nLi2]
    end
end


proj_l = x -> issparse(x) ? spdiagm(1 ./sum(x,dims=2)[:]) * x : x ./ sum(x,dims=2)
norm_l = x -> issparse(x) ? spdiagm(mean(sqrt.(sum(x.^2,dims=2)[:])) ./ sqrt.(sum(x.^2,dims=2)[:])) * x : x ./ sqrt.(sum(x.^2,dims=2)) * mean(sqrt.(sum(x.^2,dims=2)))
"""
`sclens(inp_df; device_="gpu", th=70, l_inp=nothing, p_step=0.001, return_scaled=true, n_perturb=20, centering="mean")`

The `sclens` is a function for dimensionality reduction and noise filtering in scRNA-seq data, designed to detect biologically meaningful signals without extensive parameter tuning. 

## Arguments
- `device_`: Specifies the device to be used, either `"cpu"` or `"gpu"`. If `"gpu"` is not available, the function will automatically fall back to `"cpu"`. Note that using `"gpu"` requires an Nvidia graphics card and a compatible driver.
- `th`: The threshold angle (in degrees) used in the signal robustness test. After perturbation, any changes in signal angle greater than this threshold are filtered out. Acceptable values range from 0 to 90. A value of 90 means no filtering, while 0 filters out all signals. Modifying this value is generally not recommended.
- `p_step`: The decrement level for sparsity in the signal robustness test. Increasing `p_step` allows faster computation but reduces the accuracy of the signal robustness test.
- `return_scaled`: If `true`, the output includes the fully preprocessed dataset (including L2-normalization) under the key `:scaled_df`.
- `n_perturb`: Specifies the number of perturbations to perform during the signal robustness test. Increasing this value enhances the accuracy of the test but increases computation time.
- `centering`: Determines whether to center the data on the mean or median during the z-score scaling in log normalization. Only `"mean"` and `"median"` are allowed.
- `l_inp`: Optional input metadata for cells, such as an annotation. This is reflected in the output under the key `:cell_id`.

## Output
The function returns a dictionary containing the following keys:

- `:pca`: A DataFrame containing the PC score matrix after applying Random Matrix Theory filtering, in a cell-by-PC format.
- `:pca_n1`: A DataFrame of the PC score matrix after completing the signal robustness test, also in a cell-by-PC format.
- `:L`: The eigenvalues of the data.
- `:L_mp`: The noise eigenvalues.
- `:λ`: The eigenvalue threshold obtained using Random Matrix Theory (RMT).
- `:robustness_scores`: The robustness scores of each signal obtained after the signal robustness test, represented as a dictionary containing:
    - `:m_scores`: Mean scores of each signal' robustness.
    - `:sd_scores`: Standard deviation scores of each signal' robustness.
- `:signal_ev`: The signal eigenvalues, distinguishing significant signals from noise.
- `:signal_evec`: The signal eigenvector matrix.
- `:cell_id`: Annotations for each cell as provided in the input.
- `:scaled_df`: The fully preprocessed dataset.

## Example
```julia
# Basic sclens run with default parameters
result = sclens(inp_df)

# Advanced run with a custom threshold and CPU as the device
result = sclens(inp_df, device_="cpu", th=45, p_step=0.005)
```
"""
function sclens(inp_df;device_="gpu",th=70,l_inp=nothing,p_step=0.001,return_scaled=true,n_perturb=20, centering="mean")
    pre_scale = x -> log1p.(proj_l(x))
    logn_scale = if centering == "mean"
        x -> issparse(x) ? scaled_gdata(norm_l(scaled_gdata(Matrix{Float32}(x),position_="mean")),position_="cent") : scaled_gdata(norm_l(scaled_gdata(x,position_="mean")),position_="cent")
    elseif centering == "median"
        x -> issparse(x) ? norm_l(scaled_gdata(Matrix{Float32}(x),position_="median")) : norm_l(scaled_gdata(x,position_="median"))
    else
        println("Warning: The specified centering method is not supported in the current algorithm. scLENS will automatically use mean centering.")
        x -> issparse(x) ? scaled_gdata(norm_l(scaled_gdata(Matrix{Float32}(x),position_="mean")),position_="cent") : scaled_gdata(norm_l(scaled_gdata(x,position_="mean")),position_="cent")
    end
    
    println("Extracting matrices")
    X_ = df2sparr(inp_df)
    
    nz_row, nz_col, nz_val = findnz(X_)
    nzero_idx = sparse(nz_row,nz_col,ones(Float32,length(nz_row)))
    z_idx1,z_idx2,_ = findnz(iszero.(nzero_idx))
    N,M = size(X_)
    scaled_X = logn_scale(pre_scale(X_))
    
    X_r = df2sparr(random_nz(inp_df,rmix=true))
    println("Extracting Signals...")
    GC.gc()
    nL, nV, L, L_mp, lambda_c, _ = get_sigev(scaled_X,logn_scale(pre_scale(X_r)),device=device_)
 
    mpC_ = mp_check(L_mp)
    println("Calculating noise baseline...")

    nm = min(N,M)
    model_norm = Normal(0,sqrt(1/nm))
    p_tharr = [maximum(abs.(rand(model_norm,nm))) for _ =1:5000]
    p_th = mean(p_tharr)
    println("spth_: $p_th")
    
    p_ = 0.999
    println("Calculating sparsity level for the perturbation...")
    Vr2 = if N > M 
        get_eigvec(logn_scale(pre_scale(nzero_idx))',device=device_)[end]
    else
        get_eigvec(logn_scale(pre_scale(nzero_idx)),device=device_)[end]
    end
    n_2 = round(Int,lastindex(Vr2,2)/2)
    tank_ = zeros(5,0)
    tank_n = 5
    while true
        sple_idx = sample(UInt32(1):UInt32(lastindex(z_idx1)),Int(round((1-p_)*M*N)),replace=false)
        GC.gc()
        nV_2 = if N > M
            get_eigvec(logn_scale(pre_scale(
                sparse(vcat(nz_row,z_idx1[sple_idx]),vcat(nz_col,z_idx2[sple_idx]),ones(Float32,length(nz_col)+length(sple_idx)),N,M)))',device=device_)[end]
        else
            get_eigvec(logn_scale(pre_scale(
                sparse(vcat(nz_row,z_idx1[sple_idx]),vcat(nz_col,z_idx2[sple_idx]),ones(Float32,length(nz_col)+length(sple_idx)),N,M))),device=device_)[end]
        end
        
        d_arr = try     
            nanmaximum(abs.(corr_mat(Vr2,nV_2[:,end-n_2:end],device=device_)),dims=1)[:]
        catch
            nanmaximum(abs.(corr_mat(Vr2,nV_2[:,end-n_2:end],device="cpu")),dims=1)[:]
        end
        
        tmp_A = sort(d_arr)
        tank_ = hcat(tank_,tmp_A[1:5])

        ppj_ = if size(tank_,2) < tank_n
            tank_[2,:]
        else
            tank_[2,end-(tank_n-1):end]
        end
        
        println(ppj_[end])
        if (sum(ppj_ .< p_th) > (tank_n-1)) | (p_ < 0.9)
            p_ += (tank_n-1)p_step
            break
        end
        p_ -= p_step
    end
    println("Selected perturb sparisty: $p_")
    
    Vr2 = nothing
    nzero_idx = nothing

    nV_set = Matrix[]
    nL_set = Vector[]
    min_s = size(nV,2)
    @showprogress "perturbing..." for _ in 1:n_perturb
        sple_idx = sample(UInt32(1):UInt32(lastindex(z_idx1)),Int(round((1-p_)*M*N)),replace=false)
        GC.gc()
        tmp_X = sparse(vcat(nz_row,z_idx1[sple_idx]),vcat(nz_col,z_idx2[sple_idx]),vcat(nz_val,ones(Float32,lastindex(sple_idx))),N,M)
        tmp_nL,tmp_nV = get_eigvec(logn_scale(pre_scale(tmp_X)),device=device_)
        push!(nV_set, tmp_nV[:,1:min(min_s*2,size(tmp_nV,2))])
        push!(nL_set, tmp_nL[1:min(min_s*2,size(tmp_nV,2))])
    end
 
    if iszero(min_s)
        println("warning: There is no signal")
        results = Dict(:L => L, :L_mp => L_mp,
        :λ => lambda_c, :cell_id => string.(inp_df.cell))
        return results
    else
        th_ = cos(deg2rad(th))
        println("Finding robust signals...")
        a_b = hcat([[s[2] for s in argmax(abs.(nV'*j),dims=2)] for j in nV_set]...)

        sub_nVset = [nV_set[s][:,a_b[:,s]] for s = 1:lastindex(nV_set)]
        a_ = hcat([maximum(abs.(nV'*j),dims=2) for j in sub_nVset]...)
        b_vec = []
        for i = 1:n_perturb, j=i+1:n_perturb
            push!(b_vec,maximum(abs.(sub_nVset[i]'*sub_nVset[j]),dims=2)[:])
        end
        b_ = hcat(b_vec...)
        pvalue_1 = sum(b_ .<= th_,dims=2)[:]./size(b_,2)
        
        m_score = mean(b_,dims=2)[:]
        sd_score = std(b_,dims=2)[:]
        pvals_ = pvalue_1

        sig_id = findall(pvals_ .<= 0.01)
        println("Number of filtered signal: $(size(sig_id,1))")
    
        println("Reconstructing reduced data...")
        Xout0 = nV.*(sqrt.(nL))'
        Xout1 = nV[:,sig_id].*sqrt.(nL[sig_id])'
        df_X0 = DataFrame(Xout0,:auto)
        insertcols!(df_X0,1,:cell => inp_df.cell)
        df_X1 = DataFrame(Xout1,:auto)
        insertcols!(df_X1,1,:cell => inp_df.cell)
    
 
        results = if return_scaled
            scaled_df = DataFrame(scaled_X,names(inp_df)[2:end])
            insertcols!(scaled_df,1,:cell=>inp_df.cell)
            Dict(:pca => df_X0,:pca_n1 => df_X1, :sig_id=>sig_id, :L => L, :L_mp => L_mp,
        :λ => lambda_c, :st_mat => a_, :robustness_scores => Dict(:b_ => b_,:pvalue => pvals_,:m_scores=>m_score, :sd_scores => sd_score), :signal_evec => nV, :signal_ev =>nL,
        :cell_id => l_inp,:ppj => tank_, :scaled_df => scaled_df,
        :ks_static => mpC_[:ks_static], :pass => mpC_[:pass])
        else
            Dict(:pca => df_X0,:pca_n1 => df_X1, :sig_id=>sig_id, :L => L, :L_mp => L_mp,
        :λ => lambda_c, :st_mat => a_, :robustness_scores => Dict(:b_ => b_,:pvalue => pvals_,:m_scores=>m_score, :sd_scores => sd_score), :signal_evec => nV, :signal_ev =>nL,
        :cell_id => l_inp,:ppj => tank_,
        :ks_static => mpC_[:ks_static], :pass => mpC_[:pass])
        end

        return results
    end
end

"""
`apply_umap!(input_dict; k=15, nc=2, md=0.1, metric=CosineDist())`

The `apply_umap!` function applies UMAP (Uniform Manifold Approximation and Projection) to the results from `scLENS`, stored in `input_dict`, and adds the UMAP-transformed coordinates and graph object to `input_dict`.

## Arguments
- `input_dict`: The dictionary output from `scLENS`, containing the processed data to which UMAP will be applied.
- `k`: Number of nearest neighbors considered for UMAP. This parameter influences how local relationships are preserved in the embedding.
- `nc`: The number of output dimensions for UMAP, defining the dimensionality of the transformed space (e.g., 2 or 3 for visualization).
- `md`: Minimum distance between points in the UMAP embedding. Smaller values will allow points to be closer together in the low-dimensional space, preserving more local detail.
- `metric`: The distance metric used to measure cell-to-cell distances in the PCA space. While `CosineDist` is used by default, it is generally recommended not to change this metric for consistent results.

## Output
After executing this function:
- `:umap` key in `input_dict` contains the UMAP-transformed coordinates.
- `:umap_obj` key in `input_dict` contains the UMAP graph object with the underlying connectivity information.

## Example
```julia
# Apply UMAP to the scLENS results with default parameters
apply_umap!(input_dict)

# Customize UMAP parameters, if needed
apply_umap!(input_dict, k=10, nc=3, md=0.2)
```

This function integrates UMAP embeddings into the `scLENS` results, facilitating visualization and further analysis in the reduced-dimensional space.
"""
function apply_umap!(l_dict;k=15,nc=2,md=0.1,metric=CosineDist())
    pca_y = mat_(l_dict[:pca_n1])
    model = if size(pca_y,2) > nc
        UMAP_(pca_y',nc,metric=metric,n_neighbors=k,min_dist=md)
    else
        UMAP_(mat_(l_dict[:pca])[:,1:3]',metric=metric,n_neighbors=k,min_dist=md)
    end

    l_dict[:umap] = Matrix(model.embedding')
    l_dict[:umap_obj]=model
end

function save_anndata(fn,input)
    tmp_adata = if haskey(input,:umap)
        AnnData(X=Matrix{Float32}(input[:scaled_df][!,2:end]),
        obs = DataFrame(:cell => input[:scaled_df].cell),
        var = DataFrame(:gene => names(input[:scaled_df])[2:end]),
        obsm=Dict("X_pca" => Matrix(input[:pca_n1][!,2:end]),"umap_sclens" => input[:umap])
        )
    else
        AnnData(X=Matrix{Float32}(input[:scaled_df][!,2:end]),
        obs = DataFrame(:cell => input[:scaled_df].cell),
        var = DataFrame(:gene => names(input[:scaled_df])[2:end]),
        obsm=Dict("X_pca" => Matrix(input[:pca_n1][!,2:end]))
        )
    end
    writeh5ad(fn,tmp_adata)
end
 

"""
`tenx2jld2(p_dir, out_name="out_jld2/out.jld2", mode="gz")`

The `tenx2jld2` function converts 10x Genomics data from compressed `gz` format to `JLD2` format, facilitating efficient storage and access within Julia.

## Arguments
- `p_dir`: Path to the directory containing the 10x data files. The directory should include the following files:
  - `matrix.mtx.gz`
  - `features.tsv.gz`
  - `barcodes.tsv.gz`
- `out_name`: The name of the output file, including the path where the converted data will be saved. The default location is `out_jld2/out.jld2`.
- `mode`: Specifies the file format of the 10x data. Set to `"gz"` by default for compatibility with `.gz` compressed files.

## Usage Example
```julia
# Convert 10x data from gz format to JLD2 format
scLENS.tenx2jld2("/path/to/10x/data", "output_data.jld2")
```

## Loading the JLD2 Data
Once the data has been converted, you can load it back into Julia as a DataFrame:

```julia
using JLD2

# Load the DataFrame
df = JLD2.load("output_data.jld2", "data")
```

The converted JLD2 file will contain the data under the variable name `"data"`, allowing for easy access and analysis within Julia.
"""
function tenx2jld2(p_dir,out_name="out_jld2/out.jld2",mode="gz")
    if mode=="gz"
        println("loading matrix file..")
        M = try
            tmp_f = joinpath(p_dir,"matrix.mtx.gz")
            f_obj_ = GZip.open(tmp_f)
            tmp_obj = readlines(f_obj_)
            # n = length(tmp_obj)-2
            a_tmp = [parse.(Int,split(s," ")) for s in tmp_obj[3:end]]
            I = [s[1] for s in a_tmp[2:end]]
            J = [s[2] for s in a_tmp[2:end]]
            K = [s[3] for s in a_tmp[2:end]]
            GZip.close(f_obj_)
            sparse(I,J,K,a_tmp[1][1],a_tmp[1][2])
        catch
            mmread(joinpath(p_dir,"matrix.mtx"))
        end
        println("loading cell_id file..")
        cells_ = try
            values(CSV.read(joinpath((p_dir,"barcodes.tsv.gz")),DataFrame,header=false,buffer_in_memory=true)[!,1])
        catch
            values(CSV.read(joinpath((p_dir,"barcodes.tsv")),DataFrame,header=false)[!,1])
        end
        println("loading gene_id file..")
        gene_ = try
            values(CSV.read(joinpath(p_dir,"features.tsv.gz"),DataFrame,header=false,buffer_in_memory=true)[!,2])
        catch
            values(CSV.read(joinpath(p_dir,"features.tsv"),DataFrame,header=false)[!,2])
        end
        println("constructing DataFrame...")
        ndf = DataFrame(M',gene_,makeunique=true)
        insertcols!(ndf,1,:cell => cells_)
        if !isdir(dirname(out_name)) & !isempty(dirname(out_name))
            mkdir(dirname(out_name))
        end
        println("Saving...")
        jldsave(out_name,Dict("data" => ndf);compress=true)
        println("JLD2 file has been successfully saved as: $out_name")
    else
        println("Currently, only 10x gz files are supported.")
    end
end

function plot_embedding(inp, l_inp = nothing)
    xlabel_ = "UMAP 1"; ylabel_ = "UMAP 2";  title_ = ""
    
    CairoMakie.activate!()
    
    inp1 = inp[:umap]
    label = if isnothing(l_inp)
        inp[:cell_id]
    else
        l_inp
    end

    fig = Figure()
    ax = Axis(fig[1, 1], title=title_, xlabel=xlabel_, ylabel=ylabel_, xgridvisible=false, ygridvisible=false)

    if isnothing(label)
        scatter!(ax, inp1[:, 1], inp1[:, 2])
    else
        tmp_df1 = DataFrame(x = inp1[:, 1], y = inp1[:, 2], type = label)
        unique_labels = unique(tmp_df1.type)
        # clist = distinguishable_colors(length(unique_labels))
        clist = get(ColorSchemes.tab20,collect(LinRange(0,1,length(unique_labels))))
        sc_list = []
        for (i,ul) in enumerate(unique_labels)
            indices = findall(tmp_df1.type .== ul)
            push!(sc_list, scatter!(ax, tmp_df1.x[indices], tmp_df1.y[indices], color = clist[i], markersize = 5))
        end
        Legend(fig[1, 2],sc_list,string.(unique_labels))
    end

    return fig
end
 
function plot_stability(l_dict)
    CairoMakie.activate!()

    m_scores = l_dict[:robustness_scores][:m_scores]
    sd_scores = l_dict[:robustness_scores][:sd_scores]
    nPC = 1:length(m_scores)  
    color_map = CairoMakie.colormap("RdBu")

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="nPC", ylabel="Stability", title= "$(length(l_dict[:sig_id])) robust signals were detected")
    scatter!(ax, nPC, m_scores, color = 1 .- m_scores, colormap=color_map, markersize = 10)
    errorbars!(nPC, m_scores, sd_scores,
    sd_scores, color = :grey,whiskerwidth=10)

    return fig
end
 
function plot_mpdist(out_ours; dx = 2000)
    L = out_ours[:L]
    L_mp = out_ours[:L_mp]
    x = LinRange(0, round(maximum(L) + 0.5), dx)
    lmp_max = maximum(L_mp)
    y = _mp_pdf(x, L_mp)  

    CairoMakie.activate!()

    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "Eigenvalue",
        ylabel = "Probability density",
        title = "$(size(out_ours[:pca],2)-1) signals were detected"
    )

    # Histogram for L
    hista = hist!(ax, L, bins = 200, normalization = :pdf, color = :blue)
    # Histogram for L_mp
    histb = hist!(ax, L_mp, bins = 200, normalization = :pdf, color = :gray)
    # Scatter plot for fitted MP distribution pdf
    lin = lines!(ax, x[x .< lmp_max + 0.5], y[x .< lmp_max + 0.5], color = :black, linewidth=2)

    Legend(fig[1, 2],
    [hista, histb, lin],
    ["eigenvalues", "eigenvalues between [a,b]", "fitted MP dist. pdf"])
    return fig
end

end

# #############
