library(Seurat)
library(PCAtools)
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
base_n = strsplit(basename(f),".csv.gz")[[1]]
raw_d = as.sparse(read.tcsv(file = f, sep = ",",header = TRUE, row.names = 1))

cbmc.rna <- raw_d
cbmc.rna <- CollapseSpeciesExpressionMatrix(cbmc.rna)
cbmc <- CreateSeuratObject(counts = cbmc.rna)
# scanpy option
cbmc <- NormalizeData(cbmc,scale.factor=median(colSums(cbmc.rna)))
cbmc <- FindVariableFeatures(cbmc,selection.method="mvp",loess.span=0.3,mean.cutoff=c(0.0125,3),dispersion.cutoff=c(0.5,Inf))
# scanpy option
cbmc <- ScaleData(cbmc)

scaled_mat = cbmc@assays$RNA@layers$scale.data
p_out = parallelPCA(scaled_mat)
HornPA_out = p_out$original$rotated[,1:p_out$n]

outname2 = str_glue(dirname(f),"/dr_parallelpca.csv",sep="")
write.csv(file=outname1,HornPA_out)
