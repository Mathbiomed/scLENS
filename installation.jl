using Pkg
Pkg.add(Pkg.PackageSpec(;name="CUDA", version="5.2.0"))
Pkg.add(["ArgParse","DataFrames","Pandas","SparseArrays","Glob","JLD2","CodecLz4","StatsBase", "Muon","RCall","Makie","CairoMakie","ColorSchemes",
  "Random", "ProgressMeter", "LinearAlgebra", "UMAP", "Distances", "InlineStrings", "Clustering", "CSV", "NPZ", "Colors", "Printf","Suppressor", "NaNStatistics","MatrixMarket", "GZip","Distributions"])

