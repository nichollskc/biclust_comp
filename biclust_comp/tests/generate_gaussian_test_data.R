set.seed(123456789)

N <- 10
C <- 5
G <- 100
K <- 1

# A represents individuals, B represents genes and Z represents cell types
A_binary <- rep(0:1, each=N/2)
B_binary <- c(rep(0, G))
B_binary[11:20] <- 1
Z_binary <- c(1, 1, 1, 0, 0)

# X represents individuals x cells
X_binary <- c(A_binary %o% Z_binary)

Y_tensor_raw <- A_binary %o% B_binary %o% Z_binary
Y_flattened <- apply(Y_tensor_raw, 2, c)

Y <- Y_flattened * 100 + matrix(rnorm(G * N * C, mean=0, sd=1), nrow=N*C, G)

write.table(Y, file = snakemake@output[["Y"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(X_binary, file = snakemake@output[["X_binary"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(B_binary, file = snakemake@output[["B_binary"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(A_binary, file = snakemake@output[["A_binary"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(Z_binary, file = snakemake@output[["Z_binary"]], row.names = FALSE, col.names = FALSE, sep="\t")
writeLines(as.character(N), con = snakemake@output[["N"]])
writeLines(as.character(K), con = snakemake@output[["K"]])
