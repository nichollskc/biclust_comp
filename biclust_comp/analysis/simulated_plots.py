import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from plotnine import *

from biclust_comp.analysis import accuracy_utils as acc_utils
from biclust_comp.analysis import accuracy as acc
from biclust_comp.analysis import plots
from biclust_comp import logging_utils
from biclust_comp import utils


def plot_computation_requirements(error_df_file, param_to_plot, full_filename, summary_filename, run_seeds, sim_seeds):
    matplotlib.use('agg')

    benchmark_df = pd.read_csv(error_df_file)

    comp_req_datasets = ['simulated/constant_negbin/size_mixed/K20_N10_G1000_T10',
                        'simulated/constant_negbin/size_mixed/K20_N10_G100_T10',
                        'simulated/constant_negbin/size_mixed/K20_N10_G5000_T10',
                        'simulated/constant_negbin/size_mixed/K20_N300_G10000_T20',
                        'simulated/constant_negbin/size_mixed/K100_N300_G10000_T20']
    comp_req_datasets_short = ['simulated/constant_negbin/size_mixed/K20_N10_G1000_T10',
                               'simulated/constant_negbin/size_mixed/K20_N10_G5000_T10',
                               'simulated/constant_negbin/size_mixed/K20_N300_G10000_T20']
    dataset_labels = [acc_utils.DATASET_NAMES[long_name] for long_name in comp_req_datasets]
    dataset_labels_short = [acc_utils.DATASET_NAMES[long_name] for long_name in comp_req_datasets_short]
    expected_method_dataset_run_ids = [f"{method}/{dataset}/seed_{sim_seed}/run_seed_{run_seed}_K_20"
                                       for method in ['SDA', 'MultiCluster', 'Plaid', 'nsNMF', 'SNMF']
                                       for dataset in comp_req_datasets
                                       for sim_seed in sim_seeds
                                       for run_seed in run_seeds] + \
                                      [f"{method}/{dataset}/seed_{sim_seed}/run_seed_{run_seed}_K_25"
                                       for method in ['BicMix', 'SSLB']
                                       for dataset in comp_req_datasets
                                       for sim_seed in sim_seeds
                                       for run_seed in run_seeds] + \
                                      [f"BicMix/{dataset}/seed_{sim_seed}/run_seed_{run_seed}_K_25_qnorm_0"
                                       for dataset in comp_req_datasets
                                       for sim_seed in sim_seeds
                                       for run_seed in run_seeds] + \
                                      [f"FABIA/{dataset}/seed_{sim_seed}/run_seed_{run_seed}_K_20_spz_1.5"
                                       for dataset in comp_req_datasets
                                       for sim_seed in sim_seeds
                                       for run_seed in run_seeds]
    restricted_benchmark_df = benchmark_df[benchmark_df['method_dataset_run_id'].isin(expected_method_dataset_run_ids)]
    acc_utils.add_descriptive_columns(restricted_benchmark_df)

    summary_df = plots.mean_metric_grouped(restricted_benchmark_df,
                                           ['seedless_dataset', 'method'],
                                           param_to_plot)

    plots.setup_method_group_ordered(restricted_benchmark_df, 'method')

    p = (ggplot(restricted_benchmark_df,
                aes('seedless_dataset', param_to_plot, fill='method')) +
         geom_boxplot() +
         scale_y_log10(labels=lambda breaks: ["{:.0f}".format(x) for x in breaks]) +
         scale_fill_manual(values=plots.METHOD_PALETTE_DICT) +
         scale_x_discrete(limits=comp_req_datasets, labels=dataset_labels) +
         labs(x='Dataset',
              y='Time to run (seconds - log scale)',
              fill='Method') +
         theme(panel_grid_major_x=element_blank(),
               subplots_adjust={'bottom': 0.25},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical')
    )
    p.save(full_filename, dpi=300)

    p = (ggplot(restricted_benchmark_df[restricted_benchmark_df['seedless_dataset'].isin(comp_req_datasets_short)],
            aes('seedless_dataset', param_to_plot, fill='method')) +
         geom_boxplot() +
         scale_y_log10(labels=lambda breaks: ["{:.0f}".format(x) for x in breaks]) +
         scale_fill_manual(values=plots.METHOD_PALETTE_DICT) +
         scale_x_discrete(limits=comp_req_datasets_short, labels=dataset_labels_short) +
         labs(x='Dataset',
              y='Time to run (seconds - log scale)',
              fill='Method') +
         theme(panel_grid_major_x=element_blank(),
               subplots_adjust={'bottom': 0.25},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical')
    )
    p.save(summary_filename, dpi=300)

    return summary_df


def plot_computation_requirements_against_K(error_df, param_to_plot, img_filename):
    matplotlib.use('agg')

    summary_df = plots.mean_metric_grouped(error_df,
                                           ['K_init', 'seedless_dataset', 'method'],
                                           param_to_plot)
    logging.info(summary_df)
    plots.setup_method_group_ordered(summary_df, 'method')

    p = (ggplot(summary_df,
                aes('K_init', 'metric_median', group='seedless_dataset')) +
         geom_line(aes(color='method_group')) +
         geom_errorbar(aes(ymin='metric_lq', ymax='metric_uq')) +
         scale_y_log10(labels=lambda breaks: ["{:.0f}".format(x) for x in breaks]) +
         scale_color_manual(values=plots.METHOD_GROUPS_PALETTE_DICT) +
         labs(x='Initial value of K',
              y='Time to run (seconds - log scale)',
              fill='Method group') +
         facet_wrap('~method') +
         theme(panel_grid_major_x=element_blank(),
               subplots_adjust={'bottom': 0.25},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical')
    )
    p.save(img_filename, dpi=300)

    return summary_df


def plot_thresholds(error_df, filename, filename_lines, param_to_plot, include_failures=True):
    matplotlib.use('agg')
    if include_failures:
        error_df.fillna(acc_utils.FAILURE_VALUES, inplace=True)
    else:
        error_df = error_df[error_df['run_complete']]
    if param_to_plot in plots.TIDY_NAMES_DICT:
        y_label = plots.TIDY_NAMES_DICT[param_to_plot]
    else:
        y_label = param_to_plot

    # Plaid and BicMix-Q can't produce reconstruction error information
    if param_to_plot.startswith('recon'):
        error_df = error_df[~ error_df['method'].isin(['BicMix-Q', 'Plaid', 'FABIA',
                                                                         'MultiCluster'])]

    # Remove entries from K400 dataset, as we didn't calculate threshold values for it
    error_df = error_df[~ error_df['dataset'].str.contains('K400')]

    thresh_order = ['_thresh_0e+0',
                    '_thresh_1e-4',
                    '_thresh_1e-3',
                    '_thresh_1e-2',
                    '_thresh_1e-1',
                    '_thresh_5e-1',
                    '_thresh_1e+0']
    discretise_dict = {0: 0,
                       1e-4: 1,
                       1e-3: 2,
                       1e-2: 3,
                       1e-1: 4,
                       5e-1: 5,
                       1: 6}

    error_df['processing_values'] = error_df.processing.map(dict(zip(thresh_order, discretise_dict.keys())))
    error_df['processing_discrete'] = error_df.processing_values.map(discretise_dict)

    method_processing = plots.mean_metric_grouped(error_df,
                                                  ['method', 'processing_discrete'],
                                                  param_to_plot)
    method_processing_dataset = plots.mean_metric_grouped(error_df,
                                                          ['method', 'processing_discrete', 'seedless_dataset'],
                                                          param_to_plot)

    # Exclude Plaid from the lines, since it only uses one threshold
    logging.info(method_processing_dataset)

    if matplotlib.checkdep_usetex(True):
        matplotlib.rcParams['text.usetex'] = True
        thresh_labels = [0,
                         r'10\textsuperscript{-4}',
                         r'10\textsuperscript{-3}',
                         r'10\textsuperscript{-2}',
                         r'10\textsuperscript{-1}',
                         0.5,
                         1]
    else:
        thresh_labels = [0,
                         '$10^{-4}$',
                         '$10^{-3}$',
                         '$10^{-2}$',
                         0.1,
                         0.5,
                         1]

    # Ensure we don't use '_' (especially important in tex mode)
    method_processing.replace('BicMix_Q', 'BicMix-Q', inplace=True)
    method_processing_dataset.replace('BicMix_Q', 'BicMix-Q', inplace=True)

    plots.setup_method_group_ordered(method_processing, 'method')
    plots.setup_method_group_ordered(method_processing_dataset, 'method')

    p = (ggplot(method_processing,
                aes(x='processing_discrete',
                    y='metric_median',
                    fill='method_group')) +
         geom_col(stat='identity',
                  position=position_dodge(width=0.9),
                  colour='black') +
         scale_fill_manual(values=(plots.METHOD_GROUPS_PALETTE_DICT)) +
         scale_x_continuous(breaks=(list(discretise_dict.values())),
                            labels=thresh_labels) +
         facet_wrap('~method') +
         labs(x='Threshold',
              y=y_label,
              fill='Method group') +
         theme(axis_text=element_text(size=8),
               subplots_adjust={'bottom': 0.2},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical',
               panel_grid_minor_x=(element_blank()))
    )
    p.save(filename, dpi=300)

    p = (ggplot(method_processing,
                aes(x='processing_discrete',
                   y='metric_median',
                   fill='method_group')) +
         geom_line(data=method_processing_dataset,
                   mapping=aes(group='seedless_dataset'),
                   alpha=0.3) +
         geom_col(stat='identity',
                  position=position_dodge(width=0.9),
                  colour='black',
                  width=0.5) +
         geom_point(fill='black',
                    size=0.5) +
         geom_line() +
         scale_fill_manual(values=(plots.METHOD_GROUPS_PALETTE_DICT)) +
         scale_x_continuous(breaks=[x for x in discretise_dict.values()],
                            labels=thresh_labels) +
         facet_wrap('~method') +
         labs(x='Threshold',
              y=y_label,
              fill='Method group') +
         theme(axis_text=element_text(size=8),
               subplots_adjust={'bottom': 0.2},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical',
               panel_grid_minor_x=(element_blank()))
    )
    p.save(filename_lines, dpi=300)
    return method_processing, method_processing_dataset


def plot_K(error_df, filename, filename_lines, filename_lines_point, param_to_plot):
    matplotlib.use('agg')
    if param_to_plot in plots.TIDY_NAMES_DICT:
        y_label = plots.TIDY_NAMES_DICT[param_to_plot]
    else:
        y_label = param_to_plot

    error_df = error_df[(error_df['method'] != 'baseline_XB_true')]
    error_df.K_init = error_df.K_init.astype(int)

    method_K_init = plots.mean_metric_grouped(error_df,
                                              ['method', 'K_init'],
                                              param_to_plot)
    method_K_init_dataset = plots.mean_metric_grouped(error_df,
                                                      ['method', 'K_init', 'seedless_dataset'],
                                                      param_to_plot)

    seedless_dataset_to_K = dict(zip(error_df['seedless_dataset'], error_df['K']))
    assert len(seedless_dataset_to_K) == error_df['K'].nunique(), \
        "Expecting each seedless dataset to have a unique true K value"
    method_K_init_dataset['true_K'] = method_K_init_dataset['seedless_dataset'].map(seedless_dataset_to_K)
    method_K_init_dataset['K_init_is_true_K'] = (method_K_init_dataset['true_K'] == method_K_init_dataset['K_init'])

    plots.setup_method_group_ordered(method_K_init, 'method')
    plots.setup_method_group_ordered(method_K_init_dataset, 'method')

    p = (ggplot(method_K_init,
                aes(x='K_init',
                    y='metric_median',
                    fill='method')) +
         geom_col(stat='identity',
                  position=position_dodge(width=0.9),
                  colour='black') +
         scale_fill_manual(values=(plots.METHOD_PALETTE_DICT)) +
         facet_wrap('~method') +
         labs(x='K init',
              y=y_label,
              fill='Method') +
         theme(panel_grid_minor_x=(element_blank()),
               legend_position='none')
    )
    p.save(filename, dpi=300)

    p = (ggplot(method_K_init,
                aes(x='K_init',
                    y='metric_median',
                    fill='method')) +
         geom_line(method_K_init_dataset,
                   aes(group='seedless_dataset'),
                   alpha=0.5) +
         geom_col(stat='identity',
                  position=position_dodge(width=0.9),
                  colour='black',
                  width=0.5) +
         geom_point(fill='black',
                    size=0.5) +
         geom_line() +
         facet_wrap('~method') +
         labs(x='K init',
              y=y_label,
              colour='True K') +
         theme(panel_grid_minor_x=(element_blank()),
               legend_position='none')
    )
    p.save(filename_lines, dpi=300)

    p = (ggplot(method_K_init_dataset,
                aes(x='K_init',
                    y='metric_median')) +
         geom_line(mapping=aes(group='seedless_dataset',
                               colour='true_K')) +
         geom_point(data=method_K_init_dataset[method_K_init_dataset['K_init_is_true_K']],
                    mapping=aes(x='true_K',
                                colour='true_K'),
                    size=0.7) +
         facet_wrap('~method') +
         labs(x='K init',
              y=y_label,
              colour='True K') +
         theme(axis_text=element_text(size=8),
               panel_grid_minor_x=(element_blank()))
    )
    p.save(filename_lines_point, dpi=300)

    return method_K_init, method_K_init_dataset


def plot_K_init_clust_err(error_df_file, filename, processing):
    matplotlib.use('agg')
    error_df = pd.read_csv(error_df_file, index_col=None)
    error_df = error_df[error_df['run_complete']]

    if processing == '_best':
        summary_df = acc_utils.restrict_to_best_threshold(error_df)
    else:
        summary_df = error_df[error_df['processing'] == processing]

    plots.setup_method_group_ordered(summary_df, 'method')

    p = (ggplot(summary_df,
                aes(x='K_init', y='clust_err', color='method')) +
         geom_point() +
         scale_colour_manual(values=plots.METHOD_PALETTE_DICT) +
         geom_smooth(method='lm') +
         facet_wrap('~method') +
         labs(x='Initial value of K',
              y=plots.TIDY_NAMES_DICT['clust_err'],
              color='Method')
    )
    p.save(filename, dpi=300)

    return summary_df


def plot_K_init_robustness_single_K(summary_df, filename, K):
    matplotlib.use('agg')
    summary_df = summary_df[summary_df['K'] == K]

    medians = summary_df.groupby(['method', 'K_init']).agg({'recovered_K' : 'median'}).reset_index()

    plots.setup_method_group_ordered(summary_df, 'method')
    plots.setup_method_group_ordered(medians, 'method')

    p = (ggplot(summary_df,
                aes(x='K_init',
                    y='recovered_K',
                    color='method_group')) +
         geom_point() +
         scale_colour_manual(values=plots.METHOD_GROUPS_PALETTE_DICT) +
         geom_abline(slope=1, intercept=0, colour='#A70103') +
         geom_line(data=medians,
                   colour='black') +
         facet_wrap('~method') +
         labs(x='Initial value of K',
              color='Method group',
              y='Final value of K') +
         theme(subplots_adjust={'bottom': 0.20},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical')
    )
    p.save(filename, dpi=300)

    return summary_df

def plot_K_init_robustness(summary_df, filename):
    matplotlib.use('agg')

    medians_by_K = summary_df.groupby(['method', 'K_init', 'K']).agg({'recovered_K' : 'median'}).reset_index()
    medians = summary_df.groupby(['method', 'K_init']).agg({'recovered_K' : 'median'}).reset_index()

    plots.setup_method_group_ordered(summary_df, 'method')
    plots.setup_method_group_ordered(medians, 'method')
    plots.setup_method_group_ordered(medians_by_K, 'method')

    p = (ggplot(summary_df,
                aes(x='K_init',
                    y='recovered_K',
                    color='method_group')) +
         geom_point() +
         scale_colour_manual(values=plots.METHOD_GROUPS_PALETTE_DICT) +
         geom_abline(slope=1, intercept=0, colour='#A70103') +
         geom_line(data=medians,
                   colour='black') +
         geom_line(data=medians_by_K,
                   mapping=aes(group='K'),
                   colour='black',
                   alpha=0.3) +
         facet_wrap('~method') +
         labs(x='Initial value of K',
              color='Method group',
              y='Final value of K') +
         theme(subplots_adjust={'bottom': 0.25},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical')
    )
    p.save(filename, dpi=300)

    return summary_df


def plot_K_recovery(error_df, filename, processing, K_init):
    matplotlib.use('agg')
    summary_df = error_df[error_df['K_init'] == K_init]
    logging.info(summary_df.method.value_counts())

    if processing == '_best':
        summary_df = acc_utils.restrict_to_best_threshold(summary_df)
    else:
        summary_df = summary_df[summary_df['processing'] == processing]

    logging.info(summary_df.method.value_counts())
    logging.info(summary_df.processing.value_counts())

    return _plot_K_recovery(summary_df, filename)


def _plot_K_recovery(summary_df, filename):
    matplotlib.use('agg')
    logging.info(summary_df.method.value_counts())
    logging.info(summary_df.processing.value_counts())

    plots.setup_method_group_ordered(summary_df, 'method')

    p = (ggplot(summary_df,
                aes(x='K',
                    y='recovered_K',
                    color='method_group')) +
         geom_point() +
         scale_colour_manual(values=(plots.METHOD_GROUPS_PALETTE_DICT)) +
         geom_smooth(method='lm') +
         geom_abline(slope=1,
                     intercept=0,
                     colour='#509210') +
         facet_wrap('~method') +
         labs(x='True number factors',
              y='Recovered number of factors',
              color='Method group') +
         theme(subplots_adjust={'bottom': 0.25},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical')
    )
    p.save(filename, dpi=300)
    return summary_df

def plot_best_mean_K(filename, expected_runs_file, max_K=100):
    matplotlib.use('agg')
    binary_error = pd.read_csv('analysis/accuracy/combined_accuracy_ext.csv', index_col=None)
    binary_error = acc_utils.restrict_to_expected_runs(binary_error, expected_runs_file)
    binary_error_thr = acc_utils.restrict_to_best_threshold(binary_error)
    best_mean_K_init = acc_utils.restrict_to_best_mean_K_init(binary_error_thr)
    summary_df = best_mean_K_init[(best_mean_K_init['K'] <= max_K)]

    plots.setup_method_group_ordered(summary_df, 'method')

    p = (ggplot(summary_df,
                aes(x='K',
                    y='recovered_K',
                    colour='method')) +
         geom_point(alpha=0.4) +
         geom_smooth(method='lm') +
         scale_colour_manual(values=(plots.METHOD_PALETTE_DICT)) +
         geom_abline(slope=1,
                     intercept=0,
                     colour='#509210') +
         facet_wrap('~method') +
         labs(x='True number factors',
              y='Recovered number of factors',
              color='Method')
    )

    p.save(filename, dpi=300)
    logging.info(summary_df.shape)
    return summary_df


def plot_best_mean_K_init(error_df_file, filename, max_K=100, max_K_init=100):
    matplotlib.use('agg')
    combined_exp = pd.read_csv(error_df_file, index_col=None)
    combined_exp = combined_exp[combined_exp['run_complete']]

    means = combined_exp.groupby(['method', 'seedless_dataset', 'K_init'])['clust_err'].mean()
    best_K_init = pd.DataFrame(means.unstack().idxmax(axis=1)).reset_index()
    best_K_init.columns = ['method', 'seedless_dataset', 'K_init']
    best_K_init['K'] = best_K_init['seedless_dataset'].str.extract(r'/K(\d+)').astype(int)

    plots.setup_method_group_ordered(best_K_init, 'method')

    p = (ggplot(best_K_init[(best_K_init['K'] <= max_K) &
                           (best_K_init['K_init'] <= max_K_init)] ,
                aes(x='K', y='K_init', colour='method')) +
         geom_point(alpha=0.4) +
         scale_colour_manual(values=(plots.METHOD_PALETTE_DICT)) +
         geom_abline(slope=1, intercept=0, colour='black') +
         labs(x='True number factors',
              y='Optimal K_init (maximising CE)',
              color='Method') +
         facet_wrap('~method')
    )
    p.save(filename, dpi=300)
    return p, best_K_init


def plot_summary(combined_exp, baseline_df, filename, metric, y_lower=0, y_upper=1, include_failures=False, K_init_choice='best_mean'):
    matplotlib.use('agg')
    logging.info(combined_exp.shape)
    logging.info(combined_exp[metric].size)
    logging.info(combined_exp[metric].count())
    if include_failures:
        combined_exp[metric].fillna(acc_utils.FAILURE_VALUES[metric], inplace=True)
        logging.info(combined_exp.shape)
        logging.info(combined_exp[metric].size)
        logging.info(combined_exp[metric].count())
    else:
        combined_exp = combined_exp[combined_exp['run_complete']]
        logging.info(combined_exp.shape)
        logging.info(combined_exp[metric].size)
        logging.info(combined_exp[metric].count())

    if K_init_choice == 'best_mean':
        combined_exp_K = acc_utils.restrict_to_best_mean_K_init(combined_exp)
    elif K_init_choice == 'best_theoretical':
        combined_exp_K = acc_utils.restrict_to_best_theoretical_K_init(combined_exp)
    else:
        raise ValueError("Expecting 'K_init_choice' to be one of 'best_mean' or 'best_theoretical'")

    if metric.startswith('recon'):
        combined_exp_K = acc_utils.add_baseline_rows(combined_exp_K, baseline_df)
    summary_df = plots.mean_metric_grouped(combined_exp_K,
                                           ['method'],
                                           metric)

    if metric in plots.TIDY_NAMES_DICT:
        y_label = plots.TIDY_NAMES_DICT[metric]
    else:
        y_label = metric

    if metric.startswith('recon'):
        summary_df = summary_df[(~summary_df['method'].isin(['BicMix-Q', 'Plaid', 'FABIA', 'MultiCluster']))]
        combined_exp_K = combined_exp_K[(~combined_exp_K['method'].isin(['BicMix-Q', 'Plaid', 'FABIA', 'MultiCluster']))]

    plots.setup_method_group_ordered(combined_exp_K, 'method')

    p = (ggplot(combined_exp_K,
                aes(x='method',
                    y=metric,
                    fill='method_group')) +
         geom_boxplot() +
         scale_fill_manual(values=(plots.METHOD_GROUPS_PALETTE_DICT)) +
         labs(x='Method',
              y=y_label,
              fill='Method group') +
         theme(axis_text_x=element_text(angle=45, hjust=1),
               axis_title_x=element_blank(),
               subplots_adjust={'bottom': 0.28},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical')
    )
    p.save(filename, dpi=300)
    return summary_df


def plot_datasets(accuracy_file, baseline_file, filename, metric, datasets, K_init_choice='best_mean', y_lower=0, y_upper=1, include_failures=False):
    dataset_names = [acc_utils.DATASET_NAMES[dataset] for dataset in datasets]
    matplotlib.use('agg')
    combined_exp = utils.read_error_df(accuracy_file, index_col=None)
    baseline_df = utils.read_error_df(baseline_file, index_col=None)
    if include_failures:
        logging.info(combined_exp)
        logging.info(combined_exp[metric])
        combined_exp.fillna(acc_utils.FAILURE_VALUES, inplace=True)
        logging.info(combined_exp)
        logging.info(combined_exp[metric])
    else:
        combined_exp = combined_exp[combined_exp['run_complete']]
        logging.info("Deleted rows with nan in s column")

    if K_init_choice == 'best_mean':
        combined_exp_K = acc_utils.restrict_to_best_mean_K_init(combined_exp)
    elif K_init_choice == 'best_theoretical':
        combined_exp_K = acc_utils.restrict_to_best_theoretical_K_init(combined_exp)
    else:
        raise ValueError("Expecting 'K_init_choice' to be one of 'best_mean' or 'best_theoretical'")

    if metric.startswith('recon'):
        combined_exp_K = acc_utils.add_baseline_rows(combined_exp_K,
                                                     baseline_df)
    logging.info(combined_exp_K.method.value_counts())

    summary_df = plots.mean_metric_grouped(combined_exp_K,
                                           ['seedless_dataset', 'method'],
                                           metric)

    logging.info(summary_df.method.value_counts())

    if metric in plots.TIDY_NAMES_DICT:
        y_label = plots.TIDY_NAMES_DICT[metric]
    else:
        y_label = metric

    if metric.startswith('recon'):
        summary_df = summary_df[(~summary_df['method'].isin(['BicMix-Q', 'Plaid', 'FABIA', 'MultiCluster']))]
        combined_exp_K = combined_exp_K[(~combined_exp_K['method'].isin(['BicMix-Q', 'Plaid', 'FABIA', 'MultiCluster']))]

    combined_exp_K['tidy_seedless_dataset'] = combined_exp_K['seedless_dataset'].map(acc_utils.DATASET_NAMES)
    combined_exp_K['dataset_category'] = pd.Categorical(combined_exp_K['tidy_seedless_dataset'],
                                                        ordered=True,
                                                        categories=dataset_names)

    plots.setup_method_group_ordered(combined_exp_K, 'method')

    p = (ggplot(combined_exp_K,
                aes(x='method',
                    y=metric,
                    fill='method')) +
         geom_boxplot() +
         scale_fill_manual(values=(plots.METHOD_PALETTE_DICT)) +
         labs(x='Method',
              y=y_label,
              fill='Method') +
         theme(panel_grid_major_x=element_blank(),
               figure_size=(1.2*len(datasets) + 0.4, 6),
               subplots_adjust={'bottom': 0.15},
               axis_text_x=element_blank(),
               axis_ticks_major_x=element_blank(),
               axis_title_x=element_blank(),
               strip_text=element_text(margin={'t': 8, 'b': 8}),
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical') +
         facet_grid('~ dataset_category')
    )
    p.save(filename, dpi=300)
    return summary_df, p

def plot_factor_distribution(factor_info_df, datasets, filename, metric, num_bins=20):
    matplotlib.use('agg')
    dataset_names = [acc_utils.DATASET_NAMES[dataset] for dataset in datasets]
    factor_info_df.replace('baseline', 'BASELINE', inplace=True)
    factor_info_df.seedless_dataset.replace((acc_utils.DATASET_NAMES), inplace=True)
    binwidth = factor_info_df[metric].max() / num_bins

    if metric in plots.TIDY_NAMES_DICT:
        x_label = plots.TIDY_NAMES_DICT[metric]
    else:
        x_label = metric

    plots.setup_method_group_ordered(factor_info_df, 'method')

    p = (ggplot(factor_info_df,
                aes(x=metric,
                    fill='method_group')) +
         geom_histogram(aes(y=f"stat(density) * {binwidth}"),
                        binwidth=binwidth) +
         scale_fill_manual(values=(plots.METHOD_GROUPS_PALETTE_DICT)) +
         facet_grid('method ~ seedless_dataset') +
         labs(x=x_label,
              y='Proportion',
              fill='Method group') +
         theme(figure_size=(6, 11),
               subplots_adjust={'bottom': 0.12},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               axis_text=element_text(size=6))
    )
    p.save(filename, dpi=300)
    return p


def plot_embedded_factors(distance_matrix, coords, labels, factor_idx, true_indices, colorblind=False):
    matplotlib.use('agg')
    coords_df = pd.DataFrame(coords, columns=['x', 'y'])
    coords_df['method'] = labels
    coords_df['factor_index'] = factor_idx
    coords_df['nearest_True'] = distance_matrix[:, true_indices].argmin(axis=1)
    coords_df['nearest_True_x'] = coords_df['nearest_True'].map(lambda i: coords[(i, 0)])
    coords_df['nearest_True_y'] = coords_df['nearest_True'].map(lambda i: coords[(i, 1)])

    palette = plots.METHOD_PALETTE_DICT
    palette['True'] = '#000000'

    if colorblind:
        extra_point_aes = aes(shape='method')
    else:
        extra_point_aes = aes()

    p = (ggplot(coords_df,
                aes('x',
                    'y',
                    colour='method')) +
         geom_point(alpha=0.3,
                    size=10,
                    data=(coords_df[(coords_df['method'] == 'True')]),
                    show_legend=False) +
         scale_colour_manual(values=palette) +
         geom_segment(aes(x='x',
                          y='y',
                          xend='nearest_True_x',
                          yend='nearest_True_y'),
                      show_legend=False) +
         geom_point(extra_point_aes,
                    alpha=0.7,
                    size=2,
                    data=(coords_df[(coords_df['method'] != 'True')]))
    )
    return coords_df, p


def plot_embedded_factors_against_true(method_dataset_run_ids):
    extracted_datasets = set(['/'.join(mdr_id.split('/')[1:-1]) for mdr_id in method_dataset_run_ids])
    assert len(extracted_datasets) == 1
    dataset = extracted_datasets.pop()

    X_bin_true = np.loadtxt(f"data/{dataset}/X_binary.txt")
    B_bin_true = np.loadtxt(f"data/{dataset}/B_binary.txt")
    K_true = X_bin_true.shape[1]

    labels = ['True' for x in range(K_true)]
    factor_indices = [x for x in range(K_true)]
    true_indices = [x for x in range(K_true)]

    X_bins = [X_bin_true]
    B_bins = [B_bin_true]
    for method_dataset_run_id in method_dataset_run_ids:
        method = method_dataset_run_id.split('/')[0]
        if method == 'BicMix':
            if not method_dataset_run_id.endswith('qnorm_0'):
                method = 'BicMix-Q'
        threshold = utils.threshold_str_to_float(acc_utils.BEST_THRESHOLD[method])
        K, X_bin, B_bin = utils.read_result_threshold_binary(f"results/{method_dataset_run_id}",
                                                             threshold)
        labels += [method for x in range(K)]
        factor_indices += [x for x in range(K)]
        X_bins.append(X_bin)
        B_bins.append(B_bin)

    X_bin_all = np.hstack(X_bins)
    B_bin_all = np.hstack(B_bins)
    intersect_matrix, union_matrix = acc.calc_overlaps(X_bin_all, B_bin_all, X_bin_all, B_bin_all)
    jaccard_matrix = intersect_matrix / union_matrix
    distance_matrix = 1 - jaccard_matrix
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=6)
    coords = mds.fit(distance_matrix).embedding_
    return plot_embedded_factors(distance_matrix, coords, labels, factor_indices, true_indices)

def restrict_error_df(error_df, ids_file, include_failures):
    logging.info(f"Starting with shape {error_df.shape}")
    error_df['run_complete'] = (error_df['recovered_K'] > 0)
    error_df_exp = acc_utils.restrict_to_expected_runs(error_df,
                                                       ids_file)
    logging.info(f"After restricting to expected runs {error_df_exp.shape}")
    error_df_exp_failures = acc_utils.add_na_rows_expected_runs(error_df_exp,
                                                                ids_file)
    logging.info(f"After adding empty rows for expected rows that failed {error_df_exp_failures.shape}")
    error_df_exp_failures.fillna({'run_complete': False}, inplace=True)
    error_df_exp_failures_thr = acc_utils.restrict_to_best_threshold(error_df_exp_failures)
    logging.info(f"Restricting to best threshold {error_df_exp_failures_thr.shape}")
    error_df_exp_failures_thr_K = acc_utils.restrict_to_best_theoretical_K_init(error_df_exp_failures_thr)
    logging.info(f"Restricting to best K {error_df_exp_failures_thr_K.shape}")

    if not include_failures:
        df = error_df_exp_failures_thr_K[error_df_exp_failures_thr_K['run_complete']]
        logging.info(f"Removing failure rows {df.shape}")
    else:
        df = error_df_exp_failures_thr_K
    return df

def factor_num_total_to_size_bin(num_total, max_total=100000):
    if num_total > max_total/2:
        size_bin_lower = 0.5
    elif num_total > max_total/20:
        size_bin_lower = 0.2
    elif num_total > max_total/10:
        size_bin_lower = 0.1
    elif num_total > max_total/100:
        size_bin_lower = 0.01
    else:
        size_bin_lower = 0
    return size_bin_lower


def count_factors_per_bin(factor_recovery_scores):
    df = factor_recovery_scores
    df['factor_size_bin'] = df['num_total'].astype(int).map(factor_num_total_to_size_bin)

    logging.info(df[['dataset', 'factor_index']].drop_duplicates().shape)
    deduplicated = df[['dataset', 'num_total', 'factor_size_bin', 'factor_index']].drop_duplicates()
    logging.info(deduplicated.shape)
    return deduplicated['factor_size_bin'].value_counts()

def factor_size_lower_to_name(size_bin_lower):
    float_size_bin = float(size_bin_lower)
    factor_size_bin_names = {
        0.5 : '> 50%',
        0.2 : '20% - 50%',
        0.1 : '10% - 20%',
        0.01 : '1% - 10%',
        0.0 : '< 1%',
    }
    size_name = factor_size_bin_names[float_size_bin]
    return size_name

def plot_factor_recovery_by_size(factor_recovery_scores_df_file, expected_ids_file, filename, filename_factors):
    matplotlib.use('agg')
    factor_recovery_scores = prepare_factor_recovery(factor_recovery_scores_df_file,
                                                     expected_ids_file)

    mean_recovery_by_size_raw = factor_recovery_scores.groupby(['method', 'num_total'])['recovery_score'].mean()
    mean_recovery_by_size = mean_recovery_by_size_raw.reset_index()
    mean_recovery_by_size['factor_size_bin'] = mean_recovery_by_size['num_total'].astype(int).map(factor_num_total_to_size_bin)

    counts_per_bin = count_factors_per_bin(factor_recovery_scores)

    def factor_size_lower_to_name_with_count(size_bin_lower):
        float_size_bin = float(size_bin_lower)
        size_name = factor_size_lower_to_name(size_bin_lower)
        full_name = f'{size_name} ({counts_per_bin[float_size_bin]} factors)'
        logging.info(full_name)
        return full_name

    mean_recovery_by_size['factor_size_bin_tidy'] = mean_recovery_by_size['factor_size_bin'].map(factor_size_lower_to_name)

    plots.setup_method_group_ordered(mean_recovery_by_size, 'method')
    plots.setup_method_group_ordered(factor_recovery_scores, 'method')

    p = (ggplot(mean_recovery_by_size, aes(x='method', y='recovery_score', fill='method_group')) +
         geom_boxplot() +
         scale_fill_manual(values=plots.METHOD_GROUPS_PALETTE_DICT) +
         labs(x='Method',
              y='Mean recovery of factor',
              fill='Method group') +
         theme(axis_text_x=element_text(angle=45, hjust=1),
               axis_title_x=element_blank(),
               subplots_adjust={'bottom': 0.25},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal') +
         facet_wrap('~factor_size_bin',
                    labeller=factor_size_lower_to_name_with_count))
    p.save(filename, dpi=300)

    p = (ggplot(factor_recovery_scores, aes(x='method', y='recovery_score', fill='method_group')) +
         geom_jitter(alpha=0.1, colour='black') +
         scale_fill_manual(values=plots.METHOD_GROUPS_PALETTE_DICT) +
         labs(x='Method',
              y='Recovery of factor',
              fill='Method group') +
         theme(axis_text_x=element_text(angle=45, hjust=1),
               axis_title_x=element_blank(),
               subplots_adjust={'bottom': 0.25},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal') +
         facet_wrap('~factor_size_bin',
                    labeller=factor_size_lower_to_name_with_count))
    p.save(filename_factors, dpi=300)

    return mean_recovery_by_size, factor_recovery_scores

def prepare_factor_recovery(factor_recovery_scores_df_file, expected_ids_file):
    factor_recovery_scores = utils.read_error_df(factor_recovery_scores_df_file)
    factor_recovery_scores = restrict_error_df(factor_recovery_scores, expected_ids_file, include_failures=False)

    factor_recovery_scores['recovery_score'] = factor_recovery_scores['jaccard_recovery_scores'].astype(float)
    factor_recovery_scores['factor_size_bin'] = factor_recovery_scores['num_total'].astype(int).map(factor_num_total_to_size_bin)

    factor_recovery_scores['factor_size_bin_tidy'] = factor_recovery_scores['factor_size_bin'].map(factor_size_lower_to_name)
    return factor_recovery_scores

def find_recovered_factor_sizes(factor_recovery):
    factors_dicts = []
    total_factors = 0
    for run_id in factor_recovery['method_dataset_run_id'].unique():
        K, X, B = acc_utils.read_result_binary_best_threshold(f"results/{run_id}")
        total_factors += K
        for k in range(K):
            num_samples = X[:, k].sum()
            num_genes = B[:, k].sum()
            num_total = num_samples * num_genes

            factors_dict = {'recovered_index': k,
                               'method_dataset_run_id': run_id,
                               'num_genes_recovered': num_genes,
                               'num_samples_recovered': num_samples,
                               'num_total_recovered': num_total}
            factors_dicts.append(factors_dict)
    factors_df = pd.DataFrame(factors_dicts)
    return factors_df

def prepare_factor_recovery_relevance(error_df_file, factor_recovery_scores_df_file, expected_ids_file):
    error_df = pd.read_csv(error_df_file, index_col=None)
    error_df = error_df[error_df['run_complete']]

    factor_recovery = prepare_factor_recovery(factor_recovery_scores_df_file,
                                              expected_ids_file)
    factors_df = find_recovered_factor_sizes(factor_recovery)

    exploded_relevance = acc_utils.explode_on_matched_columns(error_df,
                                                              ['jaccard_relevance_scores',
                                                               'jaccard_relevance_idx'],
                                                              [])
    exploded_relevance['recovered_index'] = exploded_relevance.groupby(['method_dataset_run_id']).cumcount()
    relevance_factor_sizes = exploded_relevance.merge(factors_df,
                                                      how='left',
                                                      left_on=['recovered_index', 'method_dataset_run_id'],
                                                      right_on=['recovered_index', 'method_dataset_run_id'])

    factor_info = factor_recovery[['num_total', 'num_genes', 'num_samples', 'factor_size_bin_tidy',
                                   'factor_index', 'method_dataset_run_id']]
    relevance_both_sizes = relevance_factor_sizes.merge(factor_info,
                                                        how='left',
                                                        left_on=['jaccard_relevance_idx', 'method_dataset_run_id'],
                                                        right_on=['factor_index', 'method_dataset_run_id'])

    relevance_both_sizes['factor_size_bin_recovered'] = relevance_both_sizes['num_total_recovered'].astype(int).map(factor_num_total_to_size_bin)
    relevance_both_sizes['factor_size_bin_recovered_tidy'] = relevance_both_sizes['factor_size_bin_recovered'].map(factor_size_lower_to_name)

    return factor_recovery, relevance_both_sizes

def similarity_matrix_to_df(similarity):
    values = similarity.values
    assert list(similarity.columns) == list(similarity.index), \
        "Expected square matrix. Instead got {similarity.shape}."

    # Set all except upper-triangular elements to invalid value: -2
    indices = np.tril_indices(values.shape[0])
    values[indices] = -2

    copied = pd.DataFrame(data=values,
                          columns=similarity.columns,
                          index=similarity.index)

    # Unstack so we have a row for each pair
    df = copied.unstack().reset_index()
    df.columns = ['folder_A', 'folder_B', 'score']
    # Discard rows with the score -2, since these are diagonal or duplicate entries
    df.drop(df.index[df['score'] == -2], inplace=True)

    # Prepare information df about folders
    folders = similarity.index
    folder_dicts = {folder: acc_utils.read_information_from_mdr_id(folder)
                    for folder in folders}
    folder_info = pd.DataFrame(folder_dicts).T
    df = pd.merge(df, folder_info, how='left', left_on='folder_A', right_index=True)
    df = pd.merge(df, folder_info, how='left', left_on='folder_B', right_index=True, suffixes=('_A', '_B'))
    return df

def construct_similarity_matrix(similarity_matrix_files):
    dfs = []
    for sim_file in similarity_matrix_files:
        logging.info(sim_file)
        sim = pd.read_csv(sim_file, index_col=0)
        dfs.append(similarity_matrix_to_df(sim))

    df = pd.concat(dfs)

    df['same_method'] = (df['method_A'] == df['method_B'])
    df['same_K_init'] = (df['K_init_A'] == df['K_init_B'])
    df['same_folder'] = (df['folder_A'] == df['folder_B'])

    df.drop(df.index[df['score'] == -1], inplace=True)

    return df

def plot_similarity(similarity_file, filename, filename_K, filename_BicMix):
    matplotlib.use('agg')
    similarity_df = pd.read_csv(similarity_file)
    plots.setup_method_group_ordered(similarity_df, 'method_A')

    logging.info(similarity_df)

    p = (ggplot(similarity_df[similarity_df['same_method']],
                aes(x='method_A',
                    y='score',
                    fill='method_group')) +
         scale_fill_manual(values=plots.METHOD_GROUPS_PALETTE_DICT) +
         geom_boxplot() +
         labs(x='Method',
              fill='Method group',
              y='Similarity between pairs of runs (CE)') +
         theme(axis_text_x=element_text(angle=45, hjust=1),
               subplots_adjust={'bottom': 0.28},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical')
    )
    p.save(filename, dpi=300)

    p = (ggplot(similarity_df[(similarity_df['same_method']) & (similarity_df['same_K_init'])],
                aes(x='method_A',
                    y='score',
                    fill='method_group')) +
         scale_fill_manual(values=plots.METHOD_GROUPS_PALETTE_DICT) +
         geom_boxplot() +
         labs(x='Method',
              fill='Method group',
              y='Similarity between pairs of runs (CE)') +
         theme(axis_text_x=element_text(angle=45, hjust=1),
               subplots_adjust={'bottom': 0.28},
               legend_position=(0.5, 0.05),
               legend_direction='horizontal',
               legend_box='vertical')
    )
    p.save(filename_K, dpi=300)

    p = (ggplot(similarity_df[(similarity_df['method_A'] == 'BicMix') & (similarity_df['method_B'] == 'BicMix-Q')],
                aes(x='same_K_init',
                    y='score')) +
         geom_boxplot() +
         scale_x_discrete(labels=['Different K', 'Same K']) +
         labs(x='K_init values',
              y='Similarity between pairs of runs (CE)') +
         theme(axis_text_x=element_text(angle=45, hjust=1),
               legend_position='none'))
    p.save(filename_BicMix, dpi=300)

    return similarity_df, p
