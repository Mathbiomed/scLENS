# install.packages("Seurat")
library(zinbwave)
library(scRNAseq)
library(stringr)
library(Seurat)
library(BiocParallel)
BiocParallel::register(BiocParallel::SerialParam())
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
f = args_

# tmp_dir = '/home/khlab/Documents/Data/gene_test_data/sim_Tcell_add/'
# files = sort(Sys.glob(file.path(tmp_dir,"preprocessed", "*.csv.gz")))
# f = files[1]
# f = '/home/khlab/Downloads/tmp_csv.gz'
base_n = strsplit(basename(f),".csv.gz")[[1]]
raw_d = as.sparse(read.tcsv(file = f, sep = ",",header = TRUE, row.names = 1))
sce <- SingleCellExperiment(
  assays = list(
    counts = as.matrix(raw_d)
  ), 
  colData = colnames(raw_d)
)

# fluidigm <- ReprocessedFluidigmData(assays = "tophat_counts")
filter <- rowSums(assay(sce)>5)>5
table(filter)
assay(sce) %>% log1p %>% rowVars -> vars
names(vars) <- rownames(sce)
vars <- sort(vars, decreasing = TRUE)
head(vars)

s_time = Sys.time()
# sce_zinb1 <- zinbwave(sce, K = 2, epsilon=1000, BPPARAM=BiocParallel::MulticoreParam(3))
sce_zinb1 <- zinbwave(sce)
W1 <- reducedDim(sce_zinb1)
e_time = Sys.time()

outname2 = str_glue(dirname(f),"/dr_zinbwave.csv",sep="")
write.csv(file=outname2,W1)

t_diff = as.numeric(e_time - s_time,unit="secs")
t_df = as.data.frame(t_diff)
outname3 = str_glue(dirname(f),"/e_time.csv",sep="")
write.csv(file=outname3,t_df)

