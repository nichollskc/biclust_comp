import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import biclust_comp.utils as utils
import biclust_comp.analysis.enrichment as enrich
import biclust_comp.utils as utils
import biclust_comp.analysis.accuracy_utils as acc_utils
import biclust_comp.analysis.plots as plots

def presnell_summary(run_id, filename, sample_info):
    sample_info = pd.read_csv(sample_info, sep="\t")

    # Tidy variable values
    sample_info.replace({'monocytes': 'Monocytes',
                         'neutrophils': 'Neutrophils',
                         'whole blood': 'Whole blood',
                         'multiple sclerosis': 'MS',
                         'Type 1 Diabetes': 'T1D',
                         'normal': 'Healthy',
                         'sepsis': 'Sepsis',
                         'before IFN-beta treatment': 'Before IFN-β',
                         'after IFN-beta treatment': 'After IFN-β',
                         '    ': np.nan},
                        inplace=True)
    sample_traits = sample_info[['FactorValue..cell.type.',
                                 'FactorValue..disease.',
                                 'FactorValue..treatment.']]
    sample_traits.columns = ['Cell', 'Disease', 'Treatment']
    dummies_df = pd.get_dummies(sample_traits,
                                prefix='', prefix_sep='') # Don't add prefixes to keep names tidy
    dummies_df['Myeloid'] = dummies_df['Monocytes'] + dummies_df['Neutrophils']
    dummies_df['Lymphoid'] = dummies_df['CD4'] + dummies_df['CD8'] + dummies_df['B-Cells']

    # Reorder columns - healthy at end of disease list
    print(dummies_df.columns.tolist())
    dummies_df = dummies_df[['B-Cells',
                             'CD4',
                             'CD8',
                             'Monocytes',
                             'Neutrophils',
                             'Whole blood',
                             'ALS',
                             'MS',
                             'Sepsis',
                             'T1D',
                             'Healthy',
                             'Before IFN-β',
                             'After IFN-β',
                             'Myeloid',
                             'Lymphoid']]

    plot_presnell_summary(run_id, filename, dummies_df)

def plot_presnell_summary(run_id, filename, dummies_df):
    matplotlib.use('agg')

    K, X, B = acc_utils.read_result_binary_best_threshold(f"results/{run_id}")

    f1_scores, intersections, fisher_pvals, odds_ratios = enrich.calculate_trait_enrichment(pd.DataFrame(X),
                                                                                            dummies_df)

    vmin = 0
    vmax = 1

    to_plot = f1_scores.T
    to_plot['Samples'] = vmin
    to_plot['Genes'] = vmin

    annot = intersections.T.copy()
    annot['Samples'] = (X != 0).sum(axis=0)
    annot['Genes'] = (B != 0).sum(axis=0)
    # Don't annotate 0s - emphasises the empty cells
    annot.replace(0, np.nan, inplace=True)

    factor_info = intersections.T.iloc[:, :6]
    for cell_type in factor_info.columns:
        factor_info[f"{cell_type}_nz"] = (factor_info[cell_type] != 0) * 1
        factor_info[f"{cell_type}_count"] = factor_info[cell_type].copy()
    factor_info['num_cell_types'] = (factor_info.filter(regex="nz")).sum(axis=1)
    # Sort by total number of cell types, then by which cell types are present, then by
    # counts within cell types
    sort_columns = (['num_cell_types'] +
                    list(factor_info.filter(regex='nz').columns) +
                    list(factor_info.filter(regex='count').columns))
    sorted_df = factor_info.sort_values(by=sort_columns, ascending=False)
    print(sorted_df)
    sorted_indices = list(sorted_df.index)

    figw = to_plot.shape[1]/2 + 4
    top_padding = 5.5/4
    bottom_padding = 3.5/4
    figh = (K + 1)/4 + top_padding + bottom_padding
    fig, ((ax1)) = plt.subplots(1, 1, figsize=(figw, figh))
    fig.subplots_adjust(left=0.02, bottom=bottom_padding/figh, right=0.95,
                        top=1-(top_padding/figh), wspace=0.05)
    sample_trait_totals = dummies_df.sum(axis=0)

    plots.annotated_heatmap_with_summary_row(ax1,
                                             to_plot.iloc[sorted_indices, :],
                                             annot.iloc[sorted_indices, :],
                                             sample_trait_totals, vmin,
                                             textcolors=["black", "black", "white"],
                                             valfmt='{x:.8g}',
                                             cmap="Blues", vmin=vmin, vmax=vmax,
                                             cbarlabel="F1 score")
    ax1.tick_params(axis="x", labelsize=15)

    for x in [5.5, 10.5, 12.5, 14.5]:
        plt.axvline(x=x, c='black', linewidth=4)
    plt.axhline(y= K - 0.5, c='black', linewidth=4)

    plt.setp(ax1.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")

    for x, s in [(2.5, "Cell type"),
                 (8, "Disease"),
                 (11.5, "MS\nTreatment"),
                 (13.5, "Lineage"),
                 (15.5, "Bicluster\nsize")]:
        plt.text(x=x, y=-9, s=s,
                 fontsize=18,
                 horizontalalignment='center',
                 verticalalignment='top',
                 fontweight='bold')

    utils.save_plot(filename)
