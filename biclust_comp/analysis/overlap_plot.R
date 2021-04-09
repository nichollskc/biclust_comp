library(cowplot)
library(dplyr)
library(ggplot2)
library(ggrepel)

overlap_with_scores = read.table("analysis/accuracy/overlap_info.csv",
                                 sep=",",
                                 header=TRUE)
# Treat K as a factor rather than numeric so that we can use a discrete colour scale
overlap_with_scores$K = as.factor(overlap_with_scores$K)

# Restrict to datasets which are broadly similar i.e. same noise distribution and bicluster sizes
restricted_overlap_with_scores = overlap_with_scores %>%
  filter(bicluster_size == "medium", noise == "negbin")

# This dataset contains a row for each method for each dataset
# Extract the information about overlap for each dataset by
#    1) Extracting only columns about the dataset
#    2) Removing duplicate rows
overlap = restricted_overlap_with_scores %>%
  select("short_name", "noise", "bicluster_size", "mean_max_overlap", "mean_mean_overlap", "N", "T", "G", "K") %>%
  unique()

# Custom palette with clear colours, grey for default K=20, blue for lower values
#    and red for higher values
blue_red_palette = c('#1f3696','#6b8df0','#666677','#f3c7b1','#e46e56','#cf453c','#802520')

plot_overlap <- function(overlap_metric, overlap_metric_tidy) {
    # First subplot shows how dataset parameters affect overlap
    # Main contributing parameters are dataset size (number of samples + number of genes)
    # and number of factors (K)
    # Use dataset size as x axis, and K as colour
    overlap_explained = ggplot(overlap,
                               aes(x=K,
                                   y=get(overlap_metric),
                                   label=short_name,
                                   colour=K)) +
      geom_point() +
      labs(x="Number of factors (K)",
           y=overlap_metric_tidy) +
      geom_label_repel(show.legend=FALSE,
                      size=2.5,
                      box.padding=0.2,
                      min.segment.length = 0) +
      scale_colour_manual(values=blue_red_palette) +
      scale_x_discrete(expand=expansion(mult=0.3))

    # Second plot shows how overlap affects performance, for each method individually
    score_vs_overlap = ggplot(restricted_overlap_with_scores,
                              aes(x=get(overlap_metric),
                                  y=metric_mean,
                                  colour=K)) +
      geom_point() +
      geom_smooth(method='lm',
                  colour='black') +
      labs(y="CE",
           x=overlap_metric_tidy) +
      scale_color_manual(values=blue_red_palette) +
      scale_x_continuous(breaks=c(0, 0.25, 0.5, 0.75, 1),
                         labels=c("0", "0.25", "0.5", "0.75", "1")) +
      facet_wrap("method")

    # Use cowplot's plot_grid to join the two plots into one plot
    plots = plot_grid(overlap_explained + theme(legend.position="none"),
                      score_vs_overlap + theme(legend.position="none"),
                      ncol=1,
                      rel_heights = c(0.8, 1),     # More space for score plot
                      labels="AUTO")

    # extract a legend that is laid out horizontally
    legend <- get_legend(
      overlap_explained +
        guides(color = guide_legend(nrow = 1)) +
        theme(legend.position = "bottom")
    )

    # add the legend underneath the row we made earlier. Give it 10%
    # of the height of one plot (via rel_heights).
    p = plot_grid(plots, legend, ncol = 1, rel_heights = c(1, 0.1))

    # Return this entire plot, containing both subplots and the (shared) legend
    p
}

ggsave("plots/simulated_accuracy/restricted_mean_max_overlap.pdf",
       plot_overlap("mean_max_overlap",
                    "Mean maximum overlap"),
       width=15,
       height=20,
       units="cm")
ggsave("plots/simulated_accuracy/restricted_mean_mean_overlap.pdf",
       plot_overlap("mean_mean_overlap",
                    "Mean overlap"),
       width=15,
       height=20,
       units="cm")
