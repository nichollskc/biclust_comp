set.seed(123456789)

N <- 10
C <- 5
G <- 100
K <- 10

make_sparse <- function(mat, sparsity=0.8) {
  mat[sample(1:length(mat), sparsity*length(mat))] <- 0
  mat
}

# A represents individuals, B represents genes and Z represents cell types
A <- matrix(rpois(N*K, 10), nrow=N, ncol=K)
A <- make_sparse(A)

B <- matrix(rpois(G*K, 10), nrow=G, ncol=K)
B <- make_sparse(B)

Z <- matrix(rpois(C*K, 10), nrow=C, ncol=K)
Z <- make_sparse(Z)

# X represents individuals x cells
X <- matrix(0, nrow=N*C, ncol=K)
for (k in 1:K) {
  X[, k] <- c(A[, k] %o% Z[, k])
}

Y_tensor_raw <- array(0, dim=c(N, G, C))
for (k in 1:K) {
  Y_tensor_raw <- Y_tensor_raw + A[, k] %o% B[, k] %o% Z[, k]
}
Y_flattened <- apply(Y_tensor_raw, 2, c)

Y <- Y_flattened * 100 + matrix(rpois(G * N * C, lambda=5), nrow=N*C, G)

write.table(Y, file = snakemake@output[["Y"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(X, file = snakemake@output[["X"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(B, file = snakemake@output[["B"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(A, file = snakemake@output[["A"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(Z, file = snakemake@output[["Z"]], row.names = FALSE, col.names = FALSE, sep="\t")
writeLines(as.character(N), con = snakemake@output[["N"]])
writeLines(as.character(K), con = snakemake@output[["K"]])
