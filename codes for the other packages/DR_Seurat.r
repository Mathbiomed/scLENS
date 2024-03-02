# install.packages("Seurat")
library(Seurat)
library(ggplot2)
library(patchwork)
library(parallel)
library(stringr)
read.tcsv = function(file, header=TRUE, sep=",", ...) {
  
  n = max(count.fields(file, sep=sep), na.rm=TRUE)
  x = readLines(file)
  
  .splitvar = function(x, sep, n) {
    var = unlist(strsplit(x, split=sep))
    length(var) = n
    return(var)
  }
  
  x = do.call(cbind, lapply(x, .splitvar, sep=sep, n=n))
  x = apply(x, 1, paste, collapse=sep) 
  out = read.csv(text=x, sep=sep, header=header, ...)
  return(out)
  
}

args_ = commandArgs(trailingOnly=TRUE)
f = args_[1]
# f = '/home/khlab/Downloads/tmp_csv.gz'
base_n = strsplit(basename(f),".csv.gz")[[1]]
raw_d = as.sparse(read.tcsv(file = f, sep = ",",header = TRUE, row.names = 1))

s_time = Sys.time()
cbmc.rna <- raw_d
cbmc.rna <- CollapseSpeciesExpressionMatrix(cbmc.rna)
cbmc <- CreateSeuratObject(counts = cbmc.rna)
cbmc <- NormalizeData(cbmc)
cbmc <- FindVariableFeatures(cbmc)
# cbmc <- FindVariableFeatures(cbmc,nfeatures=dim(cbmc)[1])
cbmc <- ScaleData(cbmc)
cbmc <- RunPCA(cbmc, verbose = FALSE)
# cbmc <- FindNeighbors(cbmc,dims=sel_pc)
# cbmc <- RunUMAP(cbmc, dims = sel_pc)

cbmc <- FindNeighbors(cbmc)
cbmc <- FindClusters(cbmc,random.seed=as.numeric(args_[3]),resolution=as.numeric(args_[2]))
cbmc <- RunUMAP(cbmc,dims=1:10)
# DimPlot(cbmc, reduction = "umap")
e_time = Sys.time()

t_diff = as.numeric(e_time - s_time,unit="secs")
t_df = as.data.frame(t_diff)

outname1 = str_glue(dirname(f),"/scaled_seurat.csv",sep="")
outname2 = str_glue(dirname(f),"/dr_seurat.csv",sep="")
outname3 = str_glue(dirname(f),"/e_time.csv",sep="")
outname4 = str_glue(dirname(f),"/umap_seurat.csv",sep="")
outname5 = str_glue(dirname(f),"/label_seurat.csv",sep="")
write.csv(file=outname1,cbmc@assays$RNA$scale.data)
write.csv(file=outname2,cbmc[["pca"]]@cell.embeddings)
write.csv(file=outname4,cbmc[["umap"]]@cell.embeddings)
write.csv(file=outname5,as.data.frame(cbmc@active.ident)[,1])
write.csv(file=outname3,t_df)


