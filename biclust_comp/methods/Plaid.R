library(biclust)

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
N <- nrow(Y)
G <- ncol(Y)

seed <- params$seed

max.layers <- params$K_init
col.release <- params$col_release
row.release <- params$row_release
iter.startup <- params$iter_startup
iter.layer <- params$iter_layer

set.seed(seed)

out_Plaid <- biclust(Y, method = BCPlaid(), cluster = "b", fit.model = y ~ m + a + b,
                     background = TRUE, background.layer = NA, background.df = 1, row.release = row.release,
                     col.release = col.release, shuffle = 3, back.fit = 0, max.layers = max.layers, iter.startup = iter.startup,
                     iter.layer = iter.layer, verbose = TRUE)

K_recovered = out_Plaid@Number

X_binary <- out_Plaid@RowxNumber * 1
B_binary <- t(out_Plaid@NumberxCol) * 1

if (K_recovered > 0) {
    zeroes_B <- which(apply(B_binary, 2, function(x) all(x == 0)))
    zeroes_X <- which(apply(X_binary, 2, function(x) all(x == 0)))
    zeroes <- union(zeroes_B, zeroes_X)

    if (length(zeroes) > 0) {
      X_binary <- as.matrix(X_binary[, -zeroes])
      B_binary <- as.matrix(B_binary[, -zeroes])
    }

    K_recovered <- ncol(X_binary)
}

write.table(X_binary, file = snakemake@output[["X"]], row.names = FALSE, col.names = FALSE, sep="\t")
write.table(B_binary, file = snakemake@output[["B"]], row.names = FALSE, col.names = FALSE, sep="\t")
writeLines(as.character(K_recovered), con = snakemake@output[["K"]])
