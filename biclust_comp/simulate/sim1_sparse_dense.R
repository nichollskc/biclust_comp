# Adapted from gemoran's github repo SSLB-examples on 19/11/19
# https://github.com/gemoran/SSLB-examples/blob/003d5724d/sim_study_2/sim_study_2.R
logfile <- snakemake@log[[1]]
log <- file(logfile, open="wt")
sink(log)
sink(log, type="message")
print(paste("Logging to file", logfile))

# Requires these functions
source("biclust_comp/SSLB_functions.R")

set.seed(123456789)

N <- as.integer(snakemake@wildcards[["N"]]) # number of samples
G <- as.integer(snakemake@wildcards[["G"]]) # number of features
Tissues <- as.integer(snakemake@wildcards[["T"]]) # number of features
K <- as.integer(snakemake@wildcards[["K"]]) # number of biclusters

#-----------------------------------
get_data <- generate_dense_bic(n_f = N, n_l = G, n_bic = K, n_dense = 2, min_f = 5, max_f = 20,
                                 min_l = 10, max_l = 50, overlap_f = 5, overlap_l = 15,
                                 mean_f = 2, sd_f = 1, mean_l = 3, sd_l = 1,
                                 sd_f_dense = 2, sd_l_dense = 2,
                                 sd_f_noise = 0.2, sd_l_noise = 0.2, sd_epsilon = 1)

# Y is the full dataset, with noise
Y <- get_data$data
Y_full <- do.call(rbind, replicate(Tissues, Y, simplify=FALSE))
# X_binary is the binary matrix describing membership of samples to clusters
X_binary <- get_data$factors_bic
X_binary_full <- do.call(rbind, replicate(Tissues, X_binary, simplify=FALSE))
# B_binary is the binary matrix describing membership of genes to clusters
B_binary <- get_data$loadings_bic

X <- get_data$factors
X_full <- do.call(rbind, replicate(Tissues, X, simplify=FALSE))
B <- get_data$loadings

write.table(Y_full, file = snakemake@output[["Y"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(X_full, file = snakemake@output[["X"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(B, file = snakemake@output[["B"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(X_binary_full, file = snakemake@output[["X_binary"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(B_binary, file = snakemake@output[["B_binary"]], row.names = FALSE, col.names = FALSE, sep="\t")
writeLines(as.character(K), con = snakemake@output[["K"]])
writeLines(as.character(N), con = snakemake@output[["N"]])
