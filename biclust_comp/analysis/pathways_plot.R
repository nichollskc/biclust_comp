library(ggplot2)
library(plyr)

pathways = read.table("analysis/IMPC/restricted_results_with_num_unique_factors.csv",
                      sep=",",
                      header=TRUE)
# Tidy labels of preprocessing
pathways$preprocess_tidy = revalue(pathways$preprocess, c("log"="Log",
                                                          "quantnorm"="Gaussian",
                                                          "deseq_sf/raw"="Size factor"))
pathways$tensor_tidy = revalue(pathways$tensor, c("tensor"="Tensor",
                                                  "non-tensor"="Non-tensor"))
print(colnames(pathways))

plot_pathway_enrichment <- function(x_axis_var, tidy_var_name, filename) {
    p = ggplot(pathways,
               aes(colour=preprocess_tidy,
                   shape=tensor_tidy,
                   x=get(x_axis_var),
                   y=factors_pathways_nz_alpha.0.01)) +
       geom_point(size=2) +
       scale_shape_manual(values=c(1,3))+
       theme(legend.position="bottom",
             legend.margin = margin(0.1,0,0,0, unit="cm"),
             legend.box="vertical") +
       labs(x=tidy_var_name,
            y="Proportion of biclusters enriched for a pathway",
            shape="Tensor",
            colour="Preprocessing") +
       scale_color_manual(values=c('#000000','#e41a1c','#377eb8')) +
       facet_wrap('~ method')
    ggsave(filename,
           p,
           width=12,
           height=14,
           units="cm")
}

plot_pathway_enrichment("unique_best_pathways",
                        "Number of distinct biclusters recovered",
                        "plots/IMPC/pathway_enrichment_num_unique_best_pathways.pdf")
plot_pathway_enrichment("unique_factors_1.0",
                        "Number of biclusters recovered",
                        "plots/IMPC/pathway_enrichment_num_factors.pdf")
plot_pathway_enrichment("unique_factors_0.75",
                        "Number of unique biclusters recovered",
                        "plots/IMPC/pathway_enrichment_num_unique_factors_75.pdf")
