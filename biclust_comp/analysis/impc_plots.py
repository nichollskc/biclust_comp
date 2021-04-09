import logging
from pathlib import Path
import re

import matplotlib
import pandas as pd
from plotnine import *

import biclust_comp.analysis.accuracy_utils as acc_utils
from biclust_comp.analysis import benchmarking
import biclust_comp.analysis.plots as plots

COL_PREFIX_TO_TIDYNAME = {'ko_traits_nz_alpha 0.0': 'Enrichment of relevant pathways',
                          'factors_pathways_nz_alpha 0.0': 'Pathway enrichment',
                          'ko_traits_mean_recall_alpha 0.0': 'Recovery of relevant pathways'}

K_CHOICE_STR_TO_FUNCTION = {
    '50': lambda df: df[df['K_init'] == 50],
    '200': lambda df: df[df['K_init'] == 200],
    'theoretical': acc_utils.impc_restrict_to_best_theoretical_K_init
}


def combine_measure_dfs_IMPC(df_file_list):
    logging.info(f"Using list of dfs: {df_file_list}")
    first_df_file = df_file_list.pop()
    logging.info(f"First is {first_df_file}")
    combined_df = pd.read_csv(first_df_file, sep="\t")

    invalid_dataset_rows = (~ combined_df['dataset'].str.startswith('real/IMPC/'))
    combined_df.loc[invalid_dataset_rows, 'dataset'] = 'real/IMPC/' + combined_df.loc[invalid_dataset_rows, 'dataset']
    if 'processing' not in combined_df.columns:
        combined_df['processing'] = combined_df['postprocessing']
    if 'postprocessing' not in combined_df.columns:
        combined_df['postprocessing'] = combined_df['processing']

    logging.info(f"Dimensions after 1st df added: {combined_df.shape}")
    logging.info(f"Columns are {combined_df.columns}")
    for df_name in df_file_list:
        df = pd.read_csv(df_name, sep="\t")

        invalid_dataset_rows = (~ df['dataset'].str.startswith('real/IMPC/'))
        df.loc[invalid_dataset_rows, 'dataset'] = 'real/IMPC/' + df.loc[invalid_dataset_rows, 'dataset']
        if 'processing' not in df.columns:
            df['processing'] = df['postprocessing']
        if 'postprocessing' not in df.columns:
            df['postprocessing'] = df['processing']
        columns = ['method', 'dataset', 'run_id', 'processing', 'postprocessing']
        columns_present = [col for col in columns if col in combined_df.columns and col in df.columns]
        combined_df = pd.merge(left=combined_df, right=df, how='outer', on=columns_present)

        logging.info(f"Dimensions after df {df_name} added: {combined_df.shape}")
        logging.info(f"Columns are {combined_df.columns}")

    combined_df = acc_utils.merge_columns(combined_df,
                                          ['recovered_K_x', 'recovered_K_y'],
                                          'recovered_K')

    return combined_df


def setup_combined_df_IMPC(df_file_list):
    combined_df = combine_measure_dfs_IMPC(df_file_list)
    # Note - IMPC datasets don't have seeds, so we just duplicate the column
    combined_df['seedless_dataset'] = combined_df['dataset'].copy()

    results_folder = "results"
    logging.info(f"Reading in all parameter information from folder {results_folder}")
    params_files = [str(path) for path in Path(results_folder).rglob('params.json')]
    params_df = acc_utils.construct_params_df(params_files).reset_index()

    logging.info('Adding parameter information')
    combined_df = pd.merge(left=combined_df, right=params_df, how='left', on=['method', 'dataset', 'run_id'])

    logging.info(f"Reading in all benchmarking information from folder {results_folder}")
    benchmark_files = [str(path) for path in Path(results_folder).rglob('benchmark.txt')]
    benchmark_df = benchmarking.construct_benchmark_df(benchmark_files).reset_index()

    logging.info('Adding benchmarking information')
    combined_df = pd.merge(left=combined_df, right=benchmark_df, how='left', on=['method', 'dataset', 'run_id'])

    combined_df = acc_utils.add_info_columns_IMPC(combined_df)
    acc_utils.merge_K_init_columns(combined_df)

    return combined_df


def impc_plot_computation_requirements_against_K(error_df, param_to_plot, img_filename):
    matplotlib.use('agg')
    df_to_plot = plots.mean_metric_grouped(error_df, ['_K_init', 'dataset', 'method'], param_to_plot)

    plots.setup_method_group_ordered(df_to_plot, 'method')

    p = (ggplot(df_to_plot,
                aes('_K_init', 'metric_median', group='dataset')) +
         geom_line(aes(color='method_group'), alpha=0.5) +
         geom_point(size=0.5) +
         geom_errorbar(aes(ymin='metric_lq', ymax='metric_uq')) +
         scale_y_log10(labels=lambda breaks: ["{:.0f}".format(x) for x in breaks]) +
         scale_color_manual(values=plots.METHOD_GROUPS_PALETTE_DICT) +
         labs(x='Initial value of K (K_init)',
              y='Time to run (seconds - log scale)',
              color='Method group') +
         facet_wrap('~method') +
         theme(panel_grid_major_x=element_blank(),
               subplots_adjust={'bottom': 0.2},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal')
    )
    p.save(img_filename, dpi=300)

    return df_to_plot


def impc_plot_two_variables_scatter(error_df, x_param, y_param, img_filename):
    matplotlib.use('agg')

    error_df_mean = error_df.groupby(['method', 'tensor', 'K_init'])[x_param, y_param].mean().reset_index()

    p = (ggplot(error_df_mean,
                aes(x=x_param,
                    y=y_param,
                    colour='method',
                    size='K_init')) +
         scale_colour_manual(values=(plots.METHOD_PALETTE_DICT)) +
         geom_point(alpha=0.5) +
         xlim((0,1)) +
         ylim((0,1)) +
         labs(x=plots.TIDY_NAMES_DICT[x_param],
              y=plots.TIDY_NAMES_DICT[y_param],
              colour='Method') +
         facet_wrap('~tensor')
    )
    p.save(img_filename, dpi=300)

    return error_df_mean


def impc_plot_summary(error_df, img_filename, metric, y_lower=0, y_upper=1):
    matplotlib.use('agg')

    df_to_plot = plots.mean_metric_grouped(error_df, ['method', 'tensor'], metric)

    if metric in plots.TIDY_NAMES_DICT:
        y_label = plots.TIDY_NAMES_DICT[metric]
    else:
        y_label = metric

    if metric.startswith('recon'):
        df_to_plot = df_to_plot[(~df_to_plot['method'].isin(['BicMix-Q', 'Plaid', 'FABIA', 'MultiCluster']))]
        error_df = error_df[(~error_df['method'].isin(['BicMix-Q', 'Plaid', 'FABIA', 'MultiCluster']))]

    plots.setup_method_group_ordered(error_df, 'method')

    p = (ggplot(error_df,
                aes('tensor',
                    metric,
                    fill='method')) +
         geom_boxplot() +
         scale_fill_manual(values=(plots.METHOD_PALETTE_DICT)) +
         ylim(y_lower,
              y_upper) +
         labs(x='Dataset type',
              y=y_label,
              fill='Method')
    )
    p.save(img_filename, dpi=300)
    return df_to_plot


def impc_plot_pathway_enrichment_thresholds(error_df, img_filename, K_choice, column_prefix):
    matplotlib.use('agg')
    error_df_K = K_CHOICE_STR_TO_FUNCTION[K_choice](error_df)

    y_label = COL_PREFIX_TO_TIDYNAME[column_prefix]

    enrichment_cols = [col for col in error_df_K.columns if col.startswith(column_prefix)]
    enrichment_df = error_df_K.groupby(['method', 'tensor']).median()[enrichment_cols]
    df_to_plot = pd.melt(enrichment_df.reset_index(),
                         id_vars=['method', 'tensor'],
                         value_vars=enrichment_cols,
                         var_name='threshold',
                         value_name='prop_factors_enriched')

    df_to_plot['threshold_level'] = df_to_plot['threshold'].map(lambda x: float(x.split(' ')[1]))
    df_to_plot['threshold_level_str'] = df_to_plot['threshold_level'].astype(str)
    df_to_plot['threshold_level_str_cat'] = pd.Categorical(df_to_plot['threshold_level_str'],
                                                           categories=df_to_plot['threshold_level_str'].unique())

    plots.setup_method_group_ordered(df_to_plot, 'method')

    p = (ggplot(df_to_plot,
                aes(x='method', y='prop_factors_enriched', fill='threshold_level_str_cat')) +
         geom_col(position=(position_dodge(0.8)), width=0.8) +
         # Colour using the Blues palette, but the last colour will be basically white, which we don't want to use
         # So generate an extra colour and then discard it (using slicing to remove last element)
         scale_fill_manual(plots.gradient_palette_hex('#105ba4', '#abd0e6', len(enrichment_cols))) +
         ylim(0, 1) +
         labs(x='Method', y=y_label, fill='Enrichment threshold') +
         guides(fill=guide_legend(title_position='left')) +
         theme(axis_text_x=element_text(size=8),
               subplots_adjust={'bottom': 0.2},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical') +
         facet_grid('tensor~')
    )
    p.save(img_filename, dpi=300)
    return df_to_plot


def impc_plot_enrichment_precision_recall(error_df, img_filename):
    matplotlib.use('agg')

    df_to_plot = error_df.groupby(['method', 'tensor'])[['ko_traits_mean_recall_0.05',
                                                         'ko_traits_mean_precision_0.05']].mean()

    p = (ggplot(df_to_plot.reset_index(),
                aes(x='ko_traits_mean_recall_0.05',
                    y='ko_traits_mean_precision_0.05',
                    colour='method',
                    shape='tensor')) +
         geom_point(size=4) +
         scale_colour_manual(values=(plots.METHOD_PALETTE_DICT)) +
         ylim(0, 0.25) +
         xlim(0, 0.25) +
         labs(x='Recall of relevant pathways',
              y='Precision of enriched pathways',
              colour='Method',
              shape='Dataset type')
    )
    p.save(img_filename, dpi=300)
    return df_to_plot


def construct_tidy_gene_selection_name(row):
    tidy_name = row['gene_selection']
    if row['gene_selection'] == 'pooled':
        tidy_name += f"-{int(row['num_genes'])}"
    return tidy_name


def impc_compare_samegenes_datasets(error_df, img_filename, metric, K_choice):
    matplotlib.use('agg')
    error_df = error_df[error_df['run_complete']]
    if K_choice == 'faceted':
        facet_string = 'preprocess + K_init ~ tensor'
        height = 9
        legend_y = 0.15
        custom_theme = theme(strip_text=element_text(lineheight=1.4),
                             subplots_adjust={'bottom': 0.25})

        if matplotlib.checkdep_usetex(True):
            matplotlib.rcParams['text.usetex'] = True
            error_df['K_init'] = r'K\textsubscript{init} = ' + error_df['K_init'].astype(int).astype(str)
        else:
            error_df['K_init'] = r'K_init = ' + error_df['K_init'].astype(int).astype(str)
    else:
        facet_string = 'preprocess ~ tensor'
        height = 6
        legend_y = 0.05
        error_df = K_CHOICE_STR_TO_FUNCTION[K_choice](error_df)

        custom_theme = theme(subplots_adjust={'bottom': 0.22})

    error_df['tensor'] = error_df['tensor'].replace({'tensor' : 'Tensor',
                                                     'non-tensor': 'Non-tensor'})
    error_df['preprocess'] = error_df['preprocess'].replace({'deseq_sf/raw' : 'Size factor',
                                                             'log': 'Log',
                                                             'quantnorm': 'Gaussian'})

    df_to_plot = plots.mean_metric_grouped(error_df, ['dataset', 'method'], metric)

    plots.setup_method_group_ordered(error_df, 'method')

    p = (ggplot(error_df,
                aes(x='method',
                    y=metric,
                    fill='method_group')) +
         geom_boxplot() +
         scale_fill_manual(values=(plots.METHOD_GROUPS_PALETTE_DICT)) +
         labs(x='Method',
              y=plots.TIDY_NAMES_DICT[metric],
              fill='Method group',
              shape='Dataset type') +
         facet_grid(facet_string) +
         theme(axis_text_x=element_text(angle=45, hjust=1),
               axis_title_x=element_blank(),
               legend_position=(0.5, legend_y),
               legend_direction='horizontal') +
         custom_theme
    )
    p.save(img_filename, width=6, height=height, units='in', dpi=300)
    return df_to_plot


def impc_compare_datasets(error_df, img_filename, metric):
    matplotlib.use('agg')

    mean = error_df.groupby(['dataset', 'method'])[metric].mean().unstack().stack(dropna=False)
    standard_deviation = error_df.groupby(['dataset', 'method'])[metric].std().unstack().stack(dropna=False)

    df_to_plot = pd.DataFrame({'mean': mean, 'std': standard_deviation}).reset_index()
    df_to_plot = acc_utils.add_info_columns_IMPC(df_to_plot)
    df_to_plot['gene_selection_full'] = df_to_plot.apply(construct_tidy_gene_selection_name,
                                                         axis=1)
    error_df['gene_selection_full'] = error_df.apply(construct_tidy_gene_selection_name,
                                                     axis=1)

    p = (ggplot(error_df,
                aes(x='gene_selection_full',
                    y=metric,
                    fill='method')) +
         geom_boxplot() +
         scale_fill_manual(values=(plots.METHOD_PALETTE_DICT)) +
         ylim(0, 1) +
         theme(axis_text_x=element_text(angle=45, hjust=1)) +
         labs(x='Method of selecting genes',
              y=plots.TIDY_NAMES_DICT[metric],
              fill='Method',
              shape='Dataset type') +
         facet_grid('tensor ~ preprocess')
    )
    p.save(img_filename, height=5, width=9, units='in', dpi=300)
    return df_to_plot

def impc_largest_tissue_prop(method_dataset_run_ids):
    largest_tissue_prop_dfs = []
    for mdr in method_dataset_run_ids:
        method = mdr.split('/')[0]
        if method == 'BicMix' and 'qnorm_0' not in mdr:
            method = 'BicMix-Q'
        thresh = acc_utils.BEST_THRESHOLD[method]
        try:
            intersections_df = pd.read_csv(f"analysis/IMPC/{mdr}/traits_intersections{thresh}.tsv",
                                           sep="\t",
                                           index_col=0)

            # The first 7 rows hold the intersections between each of the tissues and the factors
            tissue_props = intersections_df.iloc[:7, :] / intersections_df.iloc[:7, :].sum(axis=0)
            largest_tissue_prop = tissue_props.max(axis=0)
            largest_tissue_prop_df = pd.DataFrame(largest_tissue_prop, columns=['largest_tissue'])
            largest_tissue_prop_df['method'] = method
            largest_tissue_prop_df['method_dataset_run_id'] = mdr
            largest_tissue_prop_dfs.append(largest_tissue_prop_df)
        except FileNotFoundError as e:
            logging.warning(f"File not found error:\n{e}")
            continue

    df = pd.concat(largest_tissue_prop_dfs)
    return df

def impc_plot_largest_tissue_prop(tissue_prop_df, method_dataset_run_ids, img_filename):
    matplotlib.use('agg')
    tissue_prop_df = tissue_prop_df[tissue_prop_df['method_dataset_run_id'].isin(method_dataset_run_ids)]
    plots.setup_method_group_ordered(tissue_prop_df, 'method')

    p = (ggplot(tissue_prop_df,
                aes(x='largest_tissue',
                    fill='method')) +
         geom_histogram(bins=20,
                        center=0.5) +
         scale_fill_manual(values=(plots.METHOD_PALETTE_DICT)) +
         xlim(0, 1) +
         labs(x='Proportion of factor taken up by largest tissue',
              y='Number of factors',
              fill='Method')
    )
    p.save(img_filename, dpi=300)
    return tissue_prop_df, p


def impc_plot_K_init_robustness_by_tensor(error_df, filename, processing, expected_runs_file):
    matplotlib.use('agg')
    error_df = acc_utils.restrict_to_expected_runs(error_df, expected_runs_file)
    if processing == '_best':
        df_to_plot = acc_utils.restrict_to_best_threshold(error_df)
    else:
        df_to_plot = error_df[error_df['processing'] == processing]
    df_to_plot.fillna(acc_utils.FAILURE_VALUES, inplace=True)

    p = (ggplot(df_to_plot,
                aes(x='K_init',
                    y='recovered_K',
                    colour='tensor')) +
         geom_point() +
         scale_colour_manual(values={'tensor': 'lightsalmon', 'non-tensor': 'dodgerblue'}) +
         geom_smooth(method='lm', mapping=aes(group='tensor', linetype='tensor')) +
         scale_linetype_discrete(name='tensor',
                                 breaks=['tensor', 'non-tensor'],
                                 labels=['tensor', 'non-tensor']) +
         geom_abline(slope=1, intercept=0, colour='#A70103') +
         facet_wrap('~method') +
         labs(title='Robustness to K_init',
              x='Initial value of K (K_init)',
              y='Final value of K (recovered_K)',
              fill='Method',
              linetype='Dataset')
    )
    p.save(filename, dpi=300)

    return df_to_plot


def impc_plot_K_init_robustness(error_df, filename, processing):
    matplotlib.use('agg')
    if processing == '_best':
        df_to_plot = acc_utils.restrict_to_best_threshold(error_df)
    else:
        df_to_plot = error_df[error_df['processing'] == processing]

    plots.setup_method_group_ordered(df_to_plot, 'method')

    p = (ggplot(df_to_plot,
                aes(x='K_init',
                    y='recovered_K',
                    colour='method')) +
         geom_point(alpha=0.2) +
         scale_colour_manual(values=plots.METHOD_PALETTE_DICT) +
         geom_smooth(method='lm', mapping=aes(group='tensor', linetype='tensor')) +
         geom_abline(slope=1, intercept=0, colour='#A70103') +
         facet_wrap('~method') +
         labs(title='Robustness to K_init',
              x='Initial value of K (K_init)',
              y='Final value of K (recovered_K)',
              colour='Method',
              linetype='Dataset')
    )
    p.save(filename, dpi=300)

    return df_to_plot
