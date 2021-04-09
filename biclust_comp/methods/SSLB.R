library(SSLB)

logfile <- snakemake@log[[1]]
log <- file(logfile, open="wt")
sink(log)
sink(log, type="message")
print(paste("Logging to file", logfile))

print("Parameters from snakemake:")
params <- snakemake@params[[1]]
data_file <- snakemake@input[["Y"]]

print(params)

Y <- as.matrix(read.table(file = data_file))
N <- nrow(Y)
G <- ncol(Y)

print(dim(Y))

if (is.null(params$alpha)) {
    print("Changing alpha to 1/N")
    params$alpha <- 1 / N
}

if (is.null(params$a)) {
    print("Changing a to 1/K")
    params$a <- 1 / params$K_init
}

if (is.null(params$b)) {
    print("Changing b to 1/K")
    params$b <- 1 / params$K_init
}

print("Parameters after updating NULL values to default")
print(params)
write(jsonlite::toJSON(c(params, "data_file"=data_file), pretty=T, auto_unbox=T),
      file=snakemake@output[["params"]])

K_init <- params$K_init
lambda0s <- params$lambda0s
lambda0_tildes <- params$lambda0_tildes
lambda1 <- params$lambda1
lambda1_tilde <- params$lambda1_tilde
alpha <- as.double(params$alpha)
a <- as.double(params$a)
b <- as.double(params$b)
d <- as.double(params$d)
IBP <- params$IBP
EPSILON <- params$EPSILON
MAX_ITER <- params$max_iter

set.seed(params$seed)

out <- SSLB(Y,
            K_init = K_init,
            lambda0s = lambda0s,
            lambda0_tildes = lambda0_tildes,
            lambda1 = lambda1,
            lambda1_tilde = lambda1_tilde,
            alpha = alpha,
            a = a,
            b = b,
            d = d,
            IBP = IBP,
            EPSILON = EPSILON,
            MAX_ITER=MAX_ITER)

print("Dimension of X and non-zero counts per factor")
print(dim(out$X))
print(apply(out$X, 2, function(x) sum(x != 0)))
print("Dimension of B and non-zero counts per factor")
print(dim(out$B))
print(apply(out$B, 2, function(x) sum(x != 0)))

X_SSLB <- out$X
B_SSLB <- out$B
K_SSLB <- ncol(B_SSLB)

X_binary <- (X_SSLB != 0) * 1
B_binary <- (B_SSLB != 0) * 1

write.table(X_SSLB, file = snakemake@output[["X"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(B_SSLB, file = snakemake@output[["B"]], row.names = FALSE, col.names = FALSE, sep="\t")
writeLines(as.character(K_SSLB), con = snakemake@output[["K"]])
