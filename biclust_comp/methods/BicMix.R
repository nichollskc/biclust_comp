library(BicMix)

logfile <- snakemake@log[[1]]
log <- file(logfile, open="wt")
sink(log)
sink(log, type="message")
print(paste("Logging to file", logfile))

print("Parameters from snakemake:")
params <- snakemake@params[[1]]
data_file <- snakemake@input[["Y"]]

print(params)
write(jsonlite::toJSON(c(params, "data_file"=data_file), pretty=T, auto_unbox=T),
      file=snakemake@output[["params"]])

Y <- as.matrix(read.table(file = data_file))
Y_t <- t(Y)

print(dim(Y_t))

K_init <- params$K_init
max_iter <- params$max_iter
a <- params$a
b <- params$b
tol <- params$tol
qnorm <- params$qnorm

seed <- params$seed

out_dir <- dirname(snakemake@output[["X"]])

out = BicMixR(Y_t,
              nf=K_init,
              a=a,
              b=b,
              itr=max_iter,
              out_dir=NULL,
              tol=tol,
              x_method="sparse",
              rsd=seed,
              qnorm=qnorm)

# BicMix factorises Y_t = lam %*% ex + error
#   thus Y = t(ex) %*% t(lam) + error
#       NxG = NxK      KxG
# We want to output an NxK matrix as X and a GxK matrix as B
X <- t(out$ex)
B <- out$lam

print("Dimension of X and non-zero counts per factor")
print(dim(X))
print(apply(X, 2, function(x) sum(x != 0)))
print("Dimension of B and non-zero counts per factor")
print(dim(B))
print(apply(B, 2, function(x) sum(x != 0)))

K <- ncol(B)

X_binary <- (X != 0) * 1
B_binary <- (B != 0) * 1

write.table(X, file = snakemake@output[["X"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(B, file = snakemake@output[["B"]], row.names = FALSE, col.names = FALSE, sep="\t")
writeLines(as.character(K), con = snakemake@output[["K"]])

