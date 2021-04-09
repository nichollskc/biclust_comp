import logging
import re

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

import biclust_comp.analysis.accuracy as acc
import biclust_comp.analysis.accuracy_utils as acc_utils
import biclust_comp.analysis.plots as plots
import biclust_comp.utils as utils

def calculate_average_max_overlap(X, B):
    intersect_matrix, _union_matrix = acc.calc_overlaps(X, B, X, B)
    factor_areas = intersect_matrix.diagonal().copy()
    np.fill_diagonal(intersect_matrix, 0)
    max_overlaps = intersect_matrix.max(axis=1)
    max_overlap_proportions = max_overlaps / factor_areas

    mean_overlaps = intersect_matrix.mean(axis=1)
    mean_overlap_proportions = mean_overlaps / factor_areas

    return max_overlap_proportions.mean(), np.median(max_overlap_proportions), mean_overlap_proportions.mean()

def calculate_overlap_all_datasets():
    overlap_info_dicts = []
    for seedless_dataset, short_name in acc_utils.DATASET_NAMES.items():
        if 'sweep' not in short_name:
            mean_maxes = []
            median_maxes = []
            mean_means = []
            for seed in ['1234', '1313', '1719']:
                dataset = f"{seedless_dataset}/seed_{seed}"
                _K, X, B = utils.read_result_threshold_binary(f"data/{dataset}", 0)
                mean_max_overlap, median_max_overlap, mean_mean_overlap = calculate_average_max_overlap(X, B)
                mean_maxes.append(mean_max_overlap)
                median_maxes.append(median_max_overlap)
                mean_means.append(mean_mean_overlap)

            logging.info(f"Overlap scores for {seedless_dataset} are:")
            logging.info(f"Mean max: {mean_maxes}")
            logging.info(f"Median max: {median_maxes}")
            logging.info(f"Mean mean: {mean_means}")

            overlap_info = {
                'dataset': dataset,
                'seedless_dataset': seedless_dataset,
                'short_name': short_name,
                'mean_max_overlap': np.mean(mean_maxes),
                'median_max_overlap': np.mean(median_maxes),
                'mean_mean_overlap': np.mean(mean_means),
            }
            overlap_info_dicts.append(overlap_info)

    overlap_info_df = pd.DataFrame(overlap_info_dicts)
    logging.info(overlap_info_df)
    return overlap_info_df

def short_name_to_size(short_name):
    dataset_size = 'normal'
    N = 20
    G = 1000
    T = 10
    NTmatch = re.match(r'N(\d+)-T(\d+)', short_name)
    Gmatch = re.match(r'G(\d+)', short_name)
    if 'large' in short_name:
        dataset_size = 'large'
        N = 300
        T = 20
        G = 10000
    elif NTmatch:
        N = int(NTmatch[1])
        T = int(NTmatch[2])
        num_samples = N * T
        if num_samples > 100:
            dataset_size = 'large'
        elif num_samples == 100:
            dataset_size = 'normal'
        else:
            dataset_size = 'small'
    elif Gmatch:
        G = int(Gmatch[1])
        if G > 1000:
            dataset_size = 'large'
        else:
            dataset_size = 'small'
    dataset_info = {'short_name': short_name,
                    'dataset_size': dataset_size,
                    'N': N,
                    'T': T,
                    'G': G,
                    'NtimesT': N*T,
                    'NtimesTtimesG': N*T*G,
                    'NtimesTaddG': N*T + G}
    return dataset_info

def calculate_overlap_all_datasets_with_scores(combined_exp_file):
    overlap_info_df = calculate_overlap_all_datasets()

    combined_exp = pd.read_csv(combined_exp_file)
    combined_exp = combined_exp[combined_exp['run_complete']]
    combined_exp_K = acc_utils.restrict_to_best_theoretical_K_init(combined_exp)
    combined_exp_K['tidy_seedless_dataset'] = combined_exp_K['seedless_dataset'].map(acc_utils.DATASET_NAMES)

    summary_df = plots.mean_metric_grouped(combined_exp_K,
                                           ['seedless_dataset', 'method'],
                                           'clust_err')
    logging.info(summary_df)
    overlap_info_with_scores_df = summary_df.merge(overlap_info_df,
                                                   on='seedless_dataset')
    logging.info(overlap_info_with_scores_df)
    acc_utils.add_descriptive_columns(overlap_info_with_scores_df)
    logging.info(overlap_info_with_scores_df)

    overlap_info_with_scores_df['K'] = overlap_info_with_scores_df['K'].astype(int).astype('category')

    size_category = CategoricalDtype(categories=['square_sparse', 'sparse', 'medium', 'square_dense', 'dense'], ordered=True)
    overlap_info_with_scores_df['bicluster_size'] = overlap_info_with_scores_df['bicluster_size'].astype(size_category)

    dataset_size_info = pd.DataFrame(list(overlap_info_with_scores_df['short_name'].map(short_name_to_size)))
    print(dataset_size_info)
    overlap_info_with_scores_df = overlap_info_with_scores_df.merge(dataset_size_info.drop_duplicates(), on='short_name')
    print(list(overlap_info_with_scores_df.columns))

    overlap_info_with_scores_df['K'] = overlap_info_with_scores_df['K'].astype(int).astype('category')
    overlap_info_with_scores_df['N'] = overlap_info_with_scores_df['N'].astype(int).astype('category')
    overlap_info_with_scores_df['T'] = overlap_info_with_scores_df['T'].astype(int).astype('category')
    overlap_info_with_scores_df['G'] = overlap_info_with_scores_df['G'].astype(int).astype('category')
    overlap_info_with_scores_df['NtimesT'] = overlap_info_with_scores_df['NtimesT'].astype(int).astype('category')
    overlap_info_with_scores_df['NtimesTtimesG'] = overlap_info_with_scores_df['NtimesTtimesG'].astype(int).astype('category')
    overlap_info_with_scores_df['NtimesTaddG'] = overlap_info_with_scores_df['NtimesTaddG'].astype(int).astype('category')
    dat_size_category = CategoricalDtype(categories=['small', 'normal', 'large'], ordered=True)
    overlap_info_with_scores_df['dataset_size'] = overlap_info_with_scores_df['dataset_size'].astype(dat_size_category)

    return overlap_info_with_scores_df
