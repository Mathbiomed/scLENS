library(monocle3)
# library(dplyr) # imported for some downstream data manipulation
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
base_n = strsplit(basename(f),".csv.gz")[[1]]
raw_d = read.tcsv(file = f, sep = ",",header = TRUE, row.names = 1)

s_time = Sys.time()
counts = as.matrix(raw_d)
cds <- new_cell_data_set(counts)
cds <- preprocess_cds(cds)
cds <- reduce_dimension(cds)
e_time = Sys.time()
t_diff = as.numeric(e_time - s_time,unit="secs")
t_df = as.data.frame(t_diff)
outname3 = str_glue(dirname(f),"/e_time.csv",sep="")
write.csv(file=outname3,t_df)


outname2 = str_glue(dirname(f),"/dr_monocle3.csv",sep="")
write.csv(file=outname2,reducedDim(cds))


