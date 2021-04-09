import logging

import munkres
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import *
import numpy as np

from biclust_comp import logging_utils, utils
from biclust_comp.analysis import accuracy as acc
from biclust_comp.analysis import accuracy_utils as acc_utils


MARKERS = ['s', # Square
           'X', # X
           'o', # Circle
           '.', # Point
           'P', # Plus
           'D', # Diamond
           'v', # Triangle down
           '^'] # Triangle up
DASH_STYLES = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2)]

METHODS = [
  'SSLB',
  'BicMix',
  'BicMix-Q',
  'nsNMF',
  'SNMF',
  'FABIA',
  'Plaid',
  'MultiCluster',
  'SDA',
  'baseline_XB_true',
  'BASELINE'
]
# http://colorbrewer2.org/#type=qualitative&scheme=Set1&n=8
# https://davidmathlogic.com/colorblind/#%23377eb8-%23984ea3-%23f781bf-%234daf4a-%23ffff33-%23e41a1c-%23ff7f00-%23a65628-%23852122-%23000000-%23000000
# https://davidmathlogic.com/colorblind/#%23f781bf-%230F540C-%23ff7f00-%2350BD4C-%23307DBD-%23C50F11-%237F1C8E-%23F7BD0C-%23003296
METHOD_PALETTE = [
    '#C50F11',
    '#F7BD0C',
    '#ff7f00',
    '#3B93DC',
    '#003296',
    '#50BD4C',
    '#0F540C',
    '#7F1C8E',
    '#F781BF',
    '#000000',
    '#000000'
]
METHOD_TO_GROUP_DICT = {
    'SSLB' : 'Adaptive',
    'BicMix' : 'Adaptive',
    'BicMix-Q' : 'Adaptive',
    'nsNMF' : 'NMF',
    'SNMF' : 'NMF',
    'FABIA' : 'Popular',
    'Plaid' : 'Popular',
    'MultiCluster' : 'Tensor',
    'SDA' : 'Tensor',
    'baseline_XB_true' : 'Baseline',
    'BASELINE' : 'Baseline'
}
METHOD_GROUPS = [
  'Adaptive',
  'NMF',
  'Popular',
  'Tensor',
  'Baseline'
]
METHOD_GROUPS_PALETTE = [
    '#C50F11',
    '#3B93DC',
    '#50BD4C',
    '#7F1C8E',
    '#000000'
]
METHOD_GROUPS_PALETTE_DICT = dict(zip(METHOD_GROUPS, METHOD_GROUPS_PALETTE))
METHOD_PALETTE_DICT = dict(zip(METHODS, METHOD_PALETTE))
METHOD_MARKER_DICT = dict(zip(METHODS, MARKERS))
TIDY_NAMES_DICT = {'clust_err':'Clustering Error (CE)',
                   'sparse_clust_err':'Clustering Error (CE) - sparse factors only',
                   'redundancy_mean':'Mean bicluster redundancy (MBR)',
                   'adjusted_redundancy_mean':'Mean bicluster redundancy (MBR)',
                   'recon_error_normalised':'Normalised reconstruction error (NRE)',
                   'recon_error_fold':'Reconstruction error (fold above baseline)',
                   'redundancy_average_max':'Average (maximum) similarity to other factors',
                   'recovered_K':'Number of factors recovered',
                   'traits_tissue_mean_f1_score': 'Mean F1 score (over tissue traits)',
                   'traits_mean_f1_score': 'Mean F1 score (over traits)',
                   'traits_genotype_mean_f1_score': 'Mean F1 score (over genotype traits)',
                   'traits_factors_mean_max_f1_score': 'Mean F1 score (over factors)',
                   'factors_pathways_nz_alpha 0.05': 'Pathway enrichment',
                   'ko_traits_nz_alpha 0.05': 'Enrichment of relevant pathways',
                   'num_genes':'Number of genes in factor',
                   'mean_num_genes':'Number of genes in factor',
                   'num_samples':'Number of samples in factor',
                   'num_total':'Number of elements in factor'}

def gradient_palette_hex(low, high, N):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap',
                                                               [low, high],
                                                               N=N)

    palette = []
    for i in range(N):
        decimal = i/(N-1)
        palette.append(matplotlib.colors.rgb2hex(cmap(decimal)))

    return palette

def setup_method_group_ordered(error_df, method_col):
    error_df[method_col] = error_df[method_col].astype('category')
    error_df['method_group'] = error_df[method_col].map(METHOD_TO_GROUP_DICT).astype('category')

    existing_methods = error_df[method_col].cat.categories
    filtered_methods = [method for method in METHODS if method in existing_methods]
    error_df[method_col] = error_df[method_col].cat.reorder_categories(filtered_methods)

    existing_method_groups = error_df['method_group'].cat.categories
    filtered_method_groups = [group for group in METHOD_GROUPS if group in existing_method_groups]
    error_df['method_group'] = error_df['method_group'].cat.reorder_categories(filtered_method_groups)

def plot_factors_matched(X_a, B_a, X_b, B_b, cmap='RdBu', vmin=-20, vmax=20):
    intersect_mat, union_mat = acc.calc_overlaps((X_a != 0),
                                                 (B_a != 0),
                                                 (X_b != 0),
                                                 (B_b != 0))
    cost_matrix = -1 * intersect_mat
    optimal_pairings = munkres.Munkres().compute(cost_matrix.tolist())

    a_factors_order = [pair[0] for pair in optimal_pairings]
    b_factors_order = [pair[1] for pair in optimal_pairings]

    # settings
    nrows, ncols = 2, 2  # array of sub-plots
    figsize = [10, 12]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # access the axes by ax[row_id][col_id]
    ax[0][0].imshow(X_a[:, a_factors_order],
                    aspect='auto',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax)
    ax[0][1].imshow(B_a[:, a_factors_order],
                    aspect='auto',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax)
    ax[1][0].imshow(X_b[:, b_factors_order],
                    aspect='auto',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax)
    im = ax[1][1].imshow(B_b[:, b_factors_order],
                    aspect='auto',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax)

    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.show()

def multilevel_column_to_single_level(column_names, sep='::'):
    # What we want: if a column has multiple levels we want to concatenate the levels
    #   if a column consists of only the first level (and the other levels are empty), just return this
    # Let's be a little lazy and just join non-empty levels
    # Not what we want for e.g. ('upper', '', 'lower')
    #   but we'll assume this sort of annoying column naming doesn't occur
    non_empty_names = filter(None, column_names)
    return sep.join(non_empty_names)

def lower_quartile(x):
    return x.quantile(0.25)

def upper_quartile(x):
    return x.quantile(0.75)

def mean_metric_grouped(error_df, grouping_vars, metric):
    mean_df = error_df.groupby(grouping_vars).agg({'s': ['mean', 'count', 'size', 'sem', 'std', 'median', lower_quartile, upper_quartile],
                                                   metric: ['min', 'max', 'mean', 'count', 'size', 'sem', 'std', 'median', lower_quartile, upper_quartile]}).reset_index()
    mean_df['metric_mean'] = mean_df[metric]['mean']
    mean_df['metric_sem'] = mean_df[metric]['sem']
    mean_df['metric_std'] = mean_df[metric]['std']
    mean_df['metric_median'] = mean_df[metric]['median']
    mean_df['metric_lq'] = mean_df[metric]['lower_quartile']
    mean_df['metric_uq'] = mean_df[metric]['upper_quartile']
    mean_df['total_expected_runs'] = mean_df['s']['size']
    mean_df['completed_runs'] = mean_df['s']['count']
    mean_df.columns = [multilevel_column_to_single_level(col) for col in mean_df.columns.values]
    return mean_df

def plot_summary_combined(simulated_error_file, impc_error_file, baseline_file, metric, filename):
    matplotlib.use('agg')
    simulated_error = pd.read_csv(simulated_error_file, index_col=None)
    impc_error = pd.read_csv(impc_error_file, index_col=None)
    baseline_df = utils.read_error_df(baseline_file, index_col=None)

    if metric in TIDY_NAMES_DICT:
        y_label = TIDY_NAMES_DICT[metric]
    else:
        y_label = metric

    if metric.startswith('recon'):
        simulated_error = acc_utils.add_baseline_rows(simulated_error, baseline_df)
        simulated_error = simulated_error[(~simulated_error['method'].isin(['BicMix-Q', 'Plaid', 'FABIA', 'MultiCluster']))]

        impc_error = impc_error[(~impc_error['method'].isin(['BicMix-Q', 'Plaid', 'FABIA', 'MultiCluster']))]

    simulated_error['dataset_group'] = 'Simulated'
    logging.info(impc_error['tensor'].value_counts())
    impc_error['dataset_group'] = 'IMPC (' + impc_error['tensor'] + ')'

    combined = pd.concat([simulated_error, impc_error])
    setup_method_group_ordered(combined, 'method')

    p = (ggplot(combined,
                aes(x='method',
                    y=metric,
                    fill='method_group')) +
         geom_boxplot() +
         scale_fill_manual(values=(METHOD_GROUPS_PALETTE_DICT)) +
         labs(y=y_label,
              fill='Method group') +
         facet_wrap('~ dataset_group') +
         theme(axis_text_x=element_text(angle=45, hjust=1),
               axis_title_x=element_blank(),
               subplots_adjust={'bottom': 0.25},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal')
    )
    p.save(filename, dpi=300)
    return p


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    if 'orientation' in cbar_kw and cbar_kw['orientation'] == 'horizontal':
        cbar.ax.set_xlabel(cbarlabel)
    else:
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     thresholds=None, **textkw):
    """
    A function to annotate a heatmap from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    cell_colors = im.get_array()
    logging.info(cell_colors)

    vmin, vmax = im.get_clim()
    im_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Normalize the threshold to the images color range.
    if thresholds is not None:
        thresholds = [im_norm(thr) for thr in thresholds]
    else:
        thresholds = [i/float(len(textcolors)) for i in range(len(textcolors))]

    logging.info(textcolors)
    logging.info(thresholds)

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                cell_value = cell_colors[i, j]
                textcolor = textcolors[0]
                for index, threshold in enumerate(thresholds):
                    if im_norm(cell_value) >= threshold:
                        textcolor = textcolors[index]
                kw.update(color=textcolor)
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

    return texts

def annotated_heatmap_with_summary_row(ax, colors_df, texts_df, summary_row, summary_colour, textcolors, valfmt, **kwargs):
    colors_df_ext = colors_df.copy()
    colors_df_ext.loc['Total', :] = summary_colour

    texts_df_ext = texts_df.copy()
    if summary_row is None:
        texts_df_ext.loc['Total', :] = np.nan
    else:
        texts_df_ext.loc['Total', :] = summary_row

    height = 2
    width = texts_df_ext.shape[1]
    if width == 1:
        shrink = 1
        aspect = 5
    else:
        shrink = height/width
        aspect = 10
    cbar_kw = {'orientation': 'horizontal',
               'shrink' : shrink,
               'aspect': aspect,
               'pad': 0.01}
    kwargs.update({'cbar_kw': cbar_kw})

    im, _ = heatmap(colors_df_ext.values, colors_df_ext.index, colors_df_ext.columns, ax=ax, **kwargs)
    annotate_heatmap(im, data=texts_df_ext.values, valfmt=valfmt, textcolors=textcolors)
    ax.set_aspect(aspect="auto")

    return colors_df_ext, texts_df_ext
