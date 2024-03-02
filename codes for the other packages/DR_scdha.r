library(SingleCellExperiment)
library(scDHA)
library(Seurat)
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
f = args_
# f = '/home/khlab/Downloads/tmp_csv.gz'
base_n = strsplit(basename(f),".csv.gz")[[1]]
raw_d = read.tcsv(file = f, sep = ",",header = TRUE, row.names = 1)

data <- t(raw_d);
label <- colnames(raw_d)
s_time00 = Sys.time()
#Log transform the data 
data <- log2(data + 1)
result <- scDHA(data, seed = 1)
e_time00 = Sys.time()
t_diff = as.numeric(e_time00 - s_time00,unit="secs")


outname2 = str_glue(dirname(f),"/dr_scdha.csv",sep="")
write.csv(file=outname2,result$latent)


t_df = as.data.frame(t_diff)
outname3 = str_glue(dirname(f),"/e_time.csv",sep="")
write.csv(file=outname3,t_df)
