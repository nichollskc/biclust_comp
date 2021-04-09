library(ggridges)
library(ggplot2)

factor_recovery = read.table("analysis/accuracy/recovery_scores_factors_SPARSITY.csv",
                             sep=",",
                             header=TRUE)
factor_recovery$factor_size_bin_tidy_ordered = factor(factor_recovery$factor_size_bin_tidy,
                                                      levels=c('< 1%', '1% - 10%', '10% - 20%', '20% - 50%', '> 50%'))

plot_ridge_factor_size <- function(df,
                                   x_var,
                                   x_var_tidy,
                                   group_var,
                                   filename) {
    p = ggplot(df,
               aes(x=get(x_var),
                   y=method)) +
          geom_density_ridges() +
          theme_ridges() +
          labs(x=x_var_tidy, y="Method") +
          facet_grid(reformulate(group_var, "."))
    ggsave(filename,
           p,
           width=15,
           height=20,
           units="cm")
}

plot_ridge_factor_size(factor_recovery,
                       "recovery_score",
                       "True bicluster recovery",
                       "factor_size_bin_tidy_ordered",
                       "plots/simulated_accuracy/recovery/binned_recovery_scores_factors_dist_SPARSITY.pdf")

factor_relevance = read.table("analysis/accuracy/relevance_scores_factors_SPARSITY.csv",
                              sep=",",
                              header=TRUE)
factor_relevance$factor_size_bin_tidy_ordered = factor(factor_relevance$factor_size_bin_tidy,
                                                      levels=c('< 1%', '1% - 10%', '10% - 20%', '20% - 50%', '> 50%'))
factor_relevance$factor_size_bin_recovered_tidy_ordered = factor(factor_relevance$factor_size_bin_recovered_tidy,
                                                      levels=c('< 1%', '1% - 10%', '10% - 20%', '20% - 50%', '> 50%'))

plot_ridge_factor_size(factor_relevance,
                       "jaccard_relevance_scores",
                       "Relevance of returned biclusters",
                       "factor_size_bin_tidy_ordered",
                       "plots/simulated_accuracy/recovery/binned_relevance_scores_factors_dist_SPARSITY.pdf")

plot_ridge_factor_size(factor_relevance,
                       "jaccard_relevance_scores",
                       "Relevance of returned biclusters",
                       "factor_size_bin_recovered_tidy_ordered",
                       "plots/simulated_accuracy/recovery/binned_relevance_scores_factors_dist_rec_SPARSITY.pdf")
