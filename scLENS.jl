using ArgParse
using DataFrames
using CUDA
using SparseArrays
using Glob:glob
using JLD2:save as jldsave, load as jldload
using CodecLz4:LZ4FrameCompressor as lz4
using StatsBase
using Random
using ProgressMeter
using LinearAlgebra
using UMAP
using Distances
using Distributions: Normal
using InlineStrings
using Clustering
using CSV
using NPZ
# using PlotlyBase
# using PlotlyJS
using Colors
using Makie
using CairoMakie
using ColorSchemes
using Suppressor
using NaNStatistics
import Printf: @sprintf
using Muon


sparsity_(A::Matrix) = 1-length(nonzeros(SparseMatrixCSC(A)))/length(A)
sparsity_(A::SparseMatrixCSC) = 1-length(nonzeros(A))/length(A)
sparsity_(A::Vector) = 1-sum(.!iszero.(A))/length(A)
sparsity_(A::SparseVector) = 1-sum(.!iszero.(A))/length(A)
sparsity_(A::DataFrame) = issparse(A[!,2]) ? 1-length(nonzeros(df2sparr(A)))/length(df2sparr(A)) : sparsity_(mat_(A))
# tmp_df = ndf;min_tp_c=0;min_tp_g=0;max_tp_c=Inf;max_tp_g=Inf;min_genes_per_cell=200;max_genes_per_cell=0;min_cells_per_gene=15;mito_percent=0; ribo_percent=0
function preprocess(tmp_df;min_tp_c=0,min_tp_g=0,max_tp_c=Inf,max_tp_g=Inf,
   min_genes_per_cell=200,max_genes_per_cell=0,min_cells_per_gene=15,mito_percent=0,
   ribo_percent=0)

   cell_name = tmp_df.cell
   gene_name = names(tmp_df)[2:end]
   size_ndf = (size(tmp_df))
   sp_ndf = sparsity_(tmp_df)
   println("Inp_spec")
   println("data size: $size_ndf, sparsity: $sp_ndf")

   X = if !issparse(tmp_df[!,2])
      mat_(tmp_df)
   else
      a_=Vector{Int64}([])
      b_=Vector{Int64}([])
      c_=Vector{Float32}([])
      for (i,s) in enumerate(eachcol(tmp_df[!,2:end]))
         append!(a_, ones(Int64,lastindex(s.nzind)).*i)
         append!(b_, s.nzind)
         append!(c_, s.nzval)
      end
      sparse(b_,a_,c_)
      # SparseMatrixCSC{Float32,Int64}(hcat(eachcol(tmp_df[:,2:end])...))
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
      bidx_5 = sum(X[:,bidx_ribo],dims=2)[:]./sum(X,dims=2)[:] .> ribo_percent/100 #drop mito
   end

   if max_genes_per_cell == 0
      bidx_6 = ones(Bool,size(bidx_1))
   else
      bidx_6 = n_gene_counts .< max_genes_per_cell
   end

   fc_idx = bidx_1[:] .& bidx_2[:] .& bidx_3[:] .& bidx_4[:] .& bidx_5[:] .& bidx_6[:]
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
   return o_df, [fc_idx,fg_idx]
end

mat_(x::DataFrame) = Matrix{Float32}(x[!,2:end])
mat_(x::SubDataFrame) = Matrix{Float32}(x[!,2:end])
function df2sparr(inp_df)
   if issparse(inp_df[:,2])
      a_=Vector{UInt32}([])
      b_=Vector{UInt32}([])
      c_=Vector{Float32}([])
      for (i,s) in enumerate(eachcol(inp_df[!,2:end]))
         append!(a_, ones(UInt32,lastindex(s.nzind)).*i)
         append!(b_, UInt32.(s.nzind))
         append!(c_, Float32.(s.nzval))
      end
      sparse(b_,a_,c_,size(inp_df,1),size(inp_df,2)-1)
   else
      sparse(mat_(inp_df))
   end
end

function change_gname(gname_)
   tmp_gname = if occursin(r"^ENSG.",gname_[1])
      [s in keys(dict_es2gn) ? dict_es2gn[s] : s for s in gname_]
   else
      gname_
   end
   return tmp_gname
end

tmp_obj = jldload("gene_info/gene_ids.jld2")
glist_name = tmp_obj["glist_name"]
glist_id = tmp_obj["glist_id"]
g_dict = Dict(glist_id => glist_name)

tmp_dicES = CSV.read("gene_info/EtoS_rat.txt",DataFrame)
replace!(tmp_dicES[!,"Gene name"],missing => "Unknown")
dict_es2gn = Dict(ismissing(j) ? i => i : i => j for (i,j) in zip(tmp_dicES[!,"Gene stable ID"], tmp_dicES[!,"Gene name"]))
dict_es2gn_key = collect(keys(dict_es2gn))

function _random_matrix(X;dims=1)
   if issparse(X)
      if dims==1
         nz_row, nz_col, nz_val = findnz(X)
         ldict = countmap(nz_col)
         n_ldict = length(ldict)
         row_i = vcat([sample(1:lastindex(X,1),ldict[s],replace=false) for s = 1:n_ldict]...)
         sparse(row_i,nz_col,nz_val)
      elseif dims == 2
         nz_row, nz_col, nz_val = findnz(X)
         ldict = countmap(nz_row)
         n_ldict = length(ldict)
         col_i = vcat([sample(1:lastindex(X,2),ldict[s],replace=false) for s = 1:n_ldict]...)
         sparse(nz_row,col_i,nz_val)
      end
   else
      mapslices(shuffle!,X,dims=dims)
   end
end

function random_nz(pre_df;rmix=true,return_mat=false,mix_p=nothing)
   tmp_X = if typeof(pre_df) <: DataFrame
      df2sparr(pre_df)
   else
      return_mat = true
      sparse(pre_df)
   end

   nz_row, nz_col, nz_val = findnz(tmp_X)
   nz_idx = sparse(nz_row,nz_col,ones(Bool,length(nz_row)))
   if !isnothing(mix_p)
      tmp_i = findall(nz_idx)[sample(1:sum(nz_idx),Int(ceil(sum(nz_idx)*(1-mix_p))),replace=false)]
      nz_idx[tmp_i] .= false
   end

   tmp_X.nzval .= shuffle(nz_val)
   tmp_X = if rmix
      _random_matrix(tmp_X)
   else
      tmp_X
   end

   if !return_mat
      tmp_df = DataFrame(convert(SparseMatrixCSC{Float32,Int64},tmp_X),names(pre_df)[2:end])
      insertcols!(tmp_df,1,:cell => pre_df.cell)
      return tmp_df
   else
      return tmp_X
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

proj_l = x -> issparse(x) ? SparseMatrixCSC{Float32,UInt32}(spdiagm(1 ./sum(x,dims=2)[:])) * x : x ./ sum(x,dims=2)
norm_l = x -> issparse(x) ? SparseMatrixCSC{Float32,UInt32}(spdiagm(mean(sqrt.(sum(x.^2,dims=2)[:])) ./ sqrt.(sum(x.^2,dims=2)[:]))) * x : x ./ sqrt.(sum(x.^2,dims=2)) * mean(sqrt.(sum(x.^2,dims=2)))
function scLENS(inp_df;device_="gpu",th=60,l_inp=nothing,p_step=0.001,return_scaled=true,obs_pt = "mean")
   pre_scale = x -> log1p.(proj_l(x))
   logn_scale = if obs_pt=="median"
      x -> issparse(x) ? norm_l(scaled_gdata(Matrix{Float32}(x),position_="median")) : norm_l(scaled_gdata(x,position_="median"))
   elseif obs_pt=="mean"
      x -> issparse(x) ? scaled_gdata(norm_l(scaled_gdata(Matrix{Float32}(x),position_="mean")),position_="cent") : scaled_gdata(norm_l(scaled_gdata(x,position_="mean")),position_="cent")
   else
      println("warning: wrong centering input, continue using mean centering")
      x -> issparse(x) ? scaled_gdata(norm_l(scaled_gdata(Matrix{Float32}(x),position_="mean")),position_="cent") : scaled_gdata(norm_l(scaled_gdata(x,position_="mean")),position_="cent")
   end

   if occursin(r"^ENSG.",names(inp_df)[3])
       tmp_gname = [s in keys(dict_es2gn) ? dict_es2gn[s] : "unknown" for s in names(inp_df)[2:end]]
   else
       tmp_gname = names(inp_df)[2:end]
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
   nL, nV, L, L_mp, lambda_c, _, noiseV = get_sigev(scaled_X,logn_scale(pre_scale(X_r)),device=device_)
   # nz_row_r, nz_col_r, _ = findnz(X_r)

   mpC_ = mp_check(L_mp)
   # mpC_[:plot]

   println("Calculating noise baseline...")

   nm = min(N,M)
   p_th = mean(maximum(abs.(rand(Normal(0,sqrt(1/nm) ),nm,100)),dims=1))
   println("spth_: $p_th")
   
   p_ = 0.999
   mean_cor = Array{Float32,1}([])
   println("Calculating sparsity level for the perturbation...")
   Vr2 = if N > M 
       get_eigvec(logn_scale(pre_scale(nzero_idx))',device=device_)[end]
   else
       get_eigvec(logn_scale(pre_scale(nzero_idx)),device=device_)[end]
   end
   n_2 = round(Int,lastindex(Vr2,2)/2)
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
           nanmaximum(abs.(corr_mat(Vr2[:,:],nV_2[:,end-n_2:end],device=device_)),dims=1)[:]
       catch
           nanmaximum(abs.(corr_mat(Vr2,nV_2[:,end-n_2:end],device="cpu")),dims=1)[:]
       end
       
       tmp_A = sort(d_arr) 
       ppj_ = tmp_A[tmp_A .> 1e-3][1]
       println(ppj_)
       push!(mean_cor,ppj_)
       if (ppj_ < p_th) | (p_ < 0.9)
          break
       end      
       p_ -= p_step
   end
   println("Selected perturb sparisty: $p_")
   
   Vr2 = nothing
   nzero_idx = nothing

   nV_set = []
   nL_set = []
   min_s = size(nV,2)
   @showprogress "perturbing..." for _ in 1:10
       sple_idx = sample(UInt32(1):UInt32(lastindex(z_idx1)),Int(round((1-p_)*M*N)),replace=false)
       GC.gc()
       tmp_X = sparse(vcat(nz_row,z_idx1[sple_idx]),vcat(nz_col,z_idx2[sple_idx]),vcat(nz_val,ones(Float32,lastindex(sple_idx))),N,M)
       tmp_nL,tmp_nV = get_eigvec(logn_scale(pre_scale(tmp_X)),device=device_)
       push!(nV_set, tmp_nV[:,1:min(min_s*3,N)])
       push!(nL_set, tmp_nL[1:min(min_s*3,N)])
   end

   if iszero(min_s)
       println("warning: There is no signal")
       results = Dict(:L => L, :L_mp => L_mp,
       :λ => lambda_c, :cell_id => string.(inp_df.cell))
       return results
   else
       println("Finding robust signals...")
       a_ = hcat([maximum(abs.(nV'*j[:,1:min_s]),dims=2) for j in nV_set]...)
       th_ = cos(deg2rad(th))
       m_score = mean(a_,dims=2)[:]
       sig_id = findall(m_score .> th_)
       sd_score = std(a_,dims=2)[:]
   
       println("Reconstructing reduced data...")
       Xout0 = nV.*(sqrt.(nL))'
       Xout1 = nV[:,sig_id].*sqrt.(nL[sig_id])'
       df_X0 = DataFrame(Xout0,:auto)
       insertcols!(df_X0,1,:cell => inp_df.cell)
       df_X1 = DataFrame(Xout1,:auto)
       insertcols!(df_X1,1,:cell => inp_df.cell)
   

       results = if return_scaled
           Dict(:pca => df_X0,:pca_n1 => df_X1, :sig_id=>sig_id, :L => L, :L_mp => L_mp,
       :λ => lambda_c, :st_mat => a_,:m_scores=>m_score, :sd_scores=>sd_score, :signal_evec => nV, :signal_ev =>nL,
       :cell_id => l_inp,:ppj => mean_cor, :scaled_X => scaled_X,
       :ks_static => mpC_[:ks_static], :pass => mpC_[:pass])
       else
           Dict(:pca => df_X0,:pca_n1 => df_X1, :sig_id=>sig_id, :L => L, :L_mp => L_mp,
       :λ => lambda_c, :st_mat => a_,:m_scores=>m_score, :sd_scores=>sd_score, :signal_evec => nV, :signal_ev =>nL,
       :cell_id => l_inp,:ppj => mean_cor,
       :ks_static => mpC_[:ks_static], :pass => mpC_[:pass])
       end
       
       return results
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

function get_eigvec(X;device="gpu")
   N, M = size(X)
   if N > M
      new_X = sparse(X')
      Y = _wishart_matrix(new_X,device=device)
      ev_, V = _get_eigen(Y,device=device)
      ev_[ev_ .< 0] .= 0
      L = ev_
      nLi = sortperm(L,rev=true)
      nL = L[nLi]
      nVs = V[:,nLi]
      
      mul_X = sqrt.(1 ./nL).*nVs'
      new_nVs = try
         mapslices(s -> s/norm(s),Matrix(cu(mul_X)*cu(new_X))',dims=1)
      catch
         mapslices(s -> s/norm(s),X*mul_X,dims=1)
      end

      return nL, new_nVs
   else
      Y = _wishart_matrix(X,device=device)
      ev_, V = _get_eigen(Y,device=device)
      L = ev_

      nLi = sortperm(L,rev=true)
      nL = L[nLi]
      nVs = V[:,nLi]
      return nL, nVs
   end
end

function get_sigev(X,Xr;device="gpu")
   n,m = size(X)
   if n > m
      new_X = SparseMatrixCSC{Float32}(X')
      new_Xr = SparseMatrixCSC{Float32}(Xr')
      Y = _wishart_matrix(new_X,device=device)
      ev_, V = _get_eigen(Y,device=device)
      Yr = _wishart_matrix(new_Xr,device=device)
      evr_, _ = _get_eigen(Yr,device=device)

      L = ev_
      Lr = evr_

      L_mp, _, b_min = _mp_calculation(L, Lr[1:end-1])
      lambda_c, _ = _tw(L,L_mp)
      if isnan(lambda_c)
         throw("Failed to fit the MP-distribution!!")
      end
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
      
      mul_X = sqrt.(1 ./nL).*nVs'
      mul_X2 = sqrt.(snL).*noiseVs'
      
      new_nVs = mapslices(s -> s/norm(s),Matrix(cu(mul_X)*cu(new_X))',dims=1)
      CUDA.reclaim()
      new_noiseV = try
         mapslices(s -> s/norm(s),Matrix(cu(mul_X2)*cu(new_X))',dims=1)
      catch
         tmp_X = zeros(Float32,size(mul_X2,1),size(new_X,2))
         mul!(tmp_X,mul_X2,(new_X))
         mapslices(s -> s/norm(s),tmp_X',dims=1)
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
      if isnan(lambda_c)
         throw("Failed to fit the MP-distribution!!")
      end
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
   # plot([scatter(x=new_binx,y=cdf_arr),scatter(x=new_binx,y=nc_cdf2)])
   D_ = maximum(abs.(cdf_arr .- nc_cdf2))
   c_α = sqrt(-1/2*log(p_val))
   m = length(cdf_arr)
   n = length(nc_cdf2)
   return Dict(:ks_static => D_,:pass => D_ <= c_α*sqrt((m+n)/m/n))
end

function _wishart_matrix(X;device="gpu")
   if device == "gpu"
      X = if issparse(X)
         Matrix{Float32}(X)
      else
         X
      end
      gpu_x = cu(X)
      out = CuArray{Float32,2}(undef, size(X,1),size(X,1))
      mul!(out,gpu_x,gpu_x')
      return Matrix{Float32}(out) ./ size(X,2)
   elseif device == "cpu"
      X = if issparse(X)
         Matrix(X)
      else
         X
      end
      out = Matrix{Float32}(undef, size(X,1),size(X,1))
      mul!(out,X,X')
      return out ./ size(X,2)
   end
end

function apply_umap!(l_dict,args=:pca_n1;nn=15,nc=2,md=0.1)
   pca_y = mat_(l_dict[args])
   model = if size(pca_y,2) > nc
      if nn > 1
         UMAP_(pca_y',nc,metric=CosineDist(),n_neighbors=nn,min_dist=md)
      else
         make_umg(pca_y,10)
      end
   else
      UMAP_(pca_y',size(pca_y,2)-1,metric=CosineDist(),n_neighbors=nn,min_dist=md)
   end

   l_dict[:umap] = Matrix(model.embedding')
   l_dict[:umap_obj]=model
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
   elseif position_ == "scaling"
       nothing
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

function df_to_jld2_sigle(file_name,inp)
   jldsave(file_name,Dict("data" => inp);compress=lz4())
end

function plot_embedding(inp, l_inp = nothing; title_ = "", xlabel_ = "UMAP 1", ylabel_ = "UMAP 2")
   CairoMakie.activate!()

   inp1 = inp[:umap]
   label = if isnothing(l_inp)
      tmp_l = inp[:pca_n1].cell
      if lastindex(unique(tmp_l),1) == lastindex(tmp_l,1)
         ones(size(tmp_l,1))
      else
         tmp_l
      end
   else
      l_inp
   end

   fig = Figure()
   ax = Axis(fig[1, 1],title=title_,xlabel=xlabel_,ylabel=ylabel_)

   if isnothing(label)
      scatter!(ax, inp1[:, 1], inp1[:, 2])
   else
      tmp_df1 = DataFrame(x = inp1[:, 1], y = inp1[:, 2], type = label)
      unique_labels = unique(tmp_df1.type)
      clist = distinguishable_colors(length(unique_labels))
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

   m_scores = l_dict[:m_scores]
   sd_scores = l_dict[:sd_scores]
   nPC = 1:length(m_scores)  
   color_map = get(colorschemes, :RdBu, reverse(colormap("RdBu")))

   fig = Figure()
   ax = Axis(fig[1, 1], xlabel = "nPC", ylabel = "Stability",title= "Stability Plot")
   scatter!(ax, nPC, m_scores, color = 1 .- m_scores,colormap=color_map, markersize = 10)
   errorbars!(nPC, m_scores, sd_scores,
   sd_scores, color = :grey,whiskerwidth=10)

   ax.xticks = (1:length(m_scores), string.(1:length(m_scores)))
   ax.yticks = 0.1:0.2:1

   return fig
end


function plot_mpdist(out_ours; dx = 2000, bin_size = 0.2, title = "")
   L = out_ours[:L]
   L_mp = out_ours[:L_mp]
   x = LinRange(0, round(maximum(L) + 0.5), dx)
   lmp_max = maximum(L_mp)
   y = _mp_pdf(x, L_mp)  
   yr = _mp_pdf(x, L)    

   CairoMakie.activate!() 

   fig = Figure()
   ax = Axis(fig[1, 1],
      title = title,
      xlabel = "Eigenvalue",
      ylabel = "Probability density"
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


# Scanpy, Seurat saving modules
function save_anndata(fn,pre_df,out_ours)
   adata = if haskey(out_ours,:umap)
      AnnData(X=Matrix(out_ours[:scaled_X]),
      obs_names = string.(pre_df.cell),
      var_names = names(pre_df)[2:end],
      obsm=Dict("X_pca" => mat_(out_ours[:pca_n1]),"X_umap" => out_ours[:umap])
      )
   else
      AnnData(X=Matrix(out_ours[:scaled_X]),
      obs_names = string.(pre_df.cell),
      var_names = names(pre_df)[2:end],
      obsm=Dict("X_pca" => mat_(out_ours[:pca_n1]))
      )
   end
   writeh5ad(fn,adata)
end

global r_flag = true
try
   using RCall
   if RCall.Rhome == ""
      global r_flag = false
  end 
catch
   global r_flag = false
end

if r_flag
   seurat = RCall.rimport("Seurat")
   r_asmatrix = RCall.reval("as.matrix")
   r_asdframe = RCall.reval("as.data.frame")
   r_save = RCall.reval("saveRDS")
   r_rep_ = RCall.reval(" function(pbmc, sclens) {
            pbmc[['sclens']] <- sclens
            return(pbmc)
            }")
   function save_seuratobj(fn,inp_df,out_ours)
      t_df = transpose_df(inp_df,:genes)[!,2:end];
      rdf = seurat.as_sparse(t_df,var"row.names"=names(inp_df)[2:end]);
      seurat_mat = seurat.CollapseSpeciesExpressionMatrix(rdf);
      seurat_obj = seurat.CreateSeuratObject(seurat_mat);
      seurat_obj = seurat.SetAssayData(seurat_obj,layer="scale.data",var"new.data"=out_ours[:scaled_X]',assay="RNA");
      
      row_names = rcopy(RCall.rcall(:colnames, rdf))
      tmp_mat = r_asmatrix(r_asdframe(mat_(out_ours[:pca_n1]),var"row.names"=row_names));
      dr_obj = seurat.CreateDimReducObject(embeddings = tmp_mat , key = "PC_", assay = seurat.DefaultAssay(seurat_obj))
      seurat_obj = r_rep_(seurat_obj,dr_obj)
      r_save(seurat_obj, file=fn)
   end   
else
   function save_seuratobj(fn,inp_df,out_ours)
      println("Warning: Please install R language and Seurat to use this option,\n jld2 file will be saved as outcome")
      fn = replace(fn,".rds" => ".jld2")
      jldsave(fn,Dict("data" => out_ours);compress=lz4())
   end
end

function transpose_df(tmp_df,col1name=:cell)
   M = mat_(tmp_df)
   col_names = string.(tmp_df[!,1])
   row_names = names(tmp_df)[2:end]
   out_df = DataFrame(M',col_names,makeunique=true)
   insertcols!(out_df,1,col1name => row_names)
   out_df
end

function extract_file(test_file)
   fname_base = splitext(split(test_file,"/")[end])[1]
   println(fname_base)
   ndf = if occursin("csv",test_file)
      ndf = CSV.read(test_file,DataFrame,buffer_in_memory=true)
      rename!(ndf,names(ndf)[1] => :cell)
      tmp_gname = change_gname(names(ndf)[2:end])
      rename!(ndf,["cell";tmp_gname],makeunique=true)
      return ndf
   elseif occursin("jld2",test_file)
      ndf = jldload(test_file,"data")
      if !("cell" in names(ndf))
         insertcols!(ndf,1,:cell => 1:size(ndf,1))
      end
      tmp_gname = change_gname(names(ndf)[2:end])
      rename!(ndf,["cell";tmp_gname],makeunique=true)
      return ndf
   end
end

function parse_commandline()
   s = ArgParseSettings()
   @add_arg_table s begin
      "--true_label", "-t"
         help = "true label file: or use1"
         arg_type = String
         default = nothing
      "--plot", "-p"
         help = "an option without argument, i.e. a flag"
         action = :store_true
      "--device"
         help = "device for scLENS"
         arg_type = String
         default = "gpu"
      "--scaling"
         help = "scaling method for gene-scaling"
         arg_type = String
         default = "mean"
      "--out_dir", "-d"
         help = "directory for output"
         arg_type = String
         default = tempname()
      "--out_type", "-o"
         help = "output file types: julia, python, r"
         arg_type = String
         default = "julia"
      "arg1"
         help = "input file: csv, jld2"
         required = true
   end
   return parse_args(s)
end

function main()
   @show parsed_args = parse_commandline()
   println("Parsed args:")
   for (arg,val) in parsed_args
      print("  $arg  =>  ")
      show(val)
      println()
   end

   test_file = parsed_args["arg1"]
   # test_file = "data/Z8eq.csv.gz"
   ndf = extract_file(test_file)

   println("preprocessing...")
   pre_df,f_idx = preprocess(ndf,min_tp_c=0,min_tp_g=0,max_tp_c=Inf,max_tp_g=Inf,
   min_genes_per_cell=100,max_genes_per_cell=0,min_cells_per_gene=15,mito_percent=100,
   ribo_percent=0)

   true_lfile = parsed_args["true_label"]
   # true_lfile = "data/Z8eq_l.csv"
   l_true = if isnothing(true_lfile)
      ones(size(pre_df,1))
   elseif occursin("use1",true_lfile)
      pre_df.cell
   else
      if occursin("jld2",true_lfile)
         l_true = jldload(true_lfile,"data")
         l_true[f_idx[1],end]
      elseif occursin("csv",true_lfile)
         l_true = CSV.read(true_lfile,DataFrame)[!,end]
         l_true[f_idx[1]]
      else
         ones(size(pre_df,1))
      end
   end
   
   println("scLENS...")
   dev_ = parsed_args["device"]
   # dev_ = "gpu"
   scaling_m = parsed_args["scaling"]
   # scaling_m = "mean"
   out_ours = if dev_ == "gpu"
      try
         scLENS(pre_df,device_=dev_,th=60,l_inp=l_true,obs_pt=scaling_m)
      catch
         println("warning: it is unable to use gpu")
         scLENS(pre_df,device_="cpu",th=60,l_inp=l_true,obs_pt=scaling_m)
      end
   else
      scLENS(pre_df,device_="cpu",th=60,l_inp=l_true,obs_pt=scaling_m)
   end
   
   o_filename = joinpath(parsed_args["out_dir"],"out")
   if parsed_args["plot"]
      println("umapping...")
      apply_umap!(out_ours)
      println("saving all plots...")
      p1 = plot_mpdist(out_ours)
      p2 = plot_stability(out_ours)
      p3 = plot_embedding(out_ours,l_true)

      p1_name = o_filename*"-MPdist.png"
      p2_name = o_filename*"-Stability.png"
      p3_name = o_filename*"-UMAP.png"
      save(p1_name,p1)
      save(p2_name,p2)
      save(p3_name,p3)
   end

   println("saving...")
   if occursin("julia", parsed_args["out_type"])
      println("jld2 file is saving at $(parsed_args["out_dir"])")
      jldsave(o_filename*".jld2",Dict("data" => out_ours);compress=lz4())
   elseif occursin("python", parsed_args["out_type"])
      dict_for_py = Dict(i => out_ours[Symbol(i)] for i in string.(collect(keys(out_ours))))
      delete!(dict_for_py,"umap_obj")
      delete!(dict_for_py,"cell_id")
      delete!(dict_for_py,"λ")
      cell_arr = out_ours[:pca].cell
      ctype_arr = out_ours[:cell_id]
      pdataf = DataFrame(:cell_id => cell_arr,:cell_type => ctype_arr)
      CSV.write(o_filename*"_cell_id.csv",pdataf)
      pca_all = mat_(out_ours[:pca])
      pca_sub = mat_(out_ours[:pca_n1])
      dict_for_py["pca_n1"] = pca_sub
      dict_for_py["pca"] = pca_all
      dict_for_py["lambda"] = out_ours[:λ]
      NPZ.npzwrite(o_filename*".npz",dict_for_py)
   elseif occursin("r", parsed_args["out_type"])
      save_seuratobj(o_filename*".rds",pre_df,out_ours)
   elseif occursin("seurat", lowercase(parsed_args["out_type"]))
      save_seuratobj(o_filename*".rds",pre_df,out_ours)
   elseif occursin("anndata", lowercase(parsed_args["out_type"]))
      save_anndata(o_filename*".h5ad",pre_df,out_ours)
   else
      println("Warning: There is no option for what you give")
      jldsave(o_filename*".jld2",Dict("data" => out_ours);compress=lz4())
   end

end
main()
