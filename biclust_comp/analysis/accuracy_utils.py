import json
import logging
import os
from pathlib import Path
import re
import yaml

import numpy as np
import pandas as pd

from biclust_comp.analysis import benchmarking
import biclust_comp.analysis.accuracy as acc
from biclust_comp import logging_utils, utils

DATASET_ORDER = [
    "base",
    "N50-T2",
    "N10-T20",
    "N100-T10",
    "N500-T10",
    "G100",
    "G5000",
    "large-K20",
    "Negbin-medium",
    "Negbin-high",
    "Gaussian",
    "Gaussian-medium",
    "Gaussian-high",
    "noiseless",
    "sparse",
    "dense",
    "sparse-square",
    "dense-square",
    "K5",
    "K10",
    "K50",
    "K70",
    "large-K100",
    "large-K400"
]

DATASET_NAMES = {'simulated/constant_negbin/size_mixed/K20_N10_G1000_T10':'base',
                 'simulated/constant_negbin/size_mixed/K20_N50_G1000_T2':'N50-T2',
                 'simulated/constant_negbin/size_mixed/K20_N10_G1000_T20':'N10-T20',
                 'simulated/constant_negbin/size_mixed/K20_N100_G1000_T10':'N100-T10',
                 'simulated/constant_negbin/size_mixed/K20_N500_G1000_T10':'N500-T10',
                 'simulated/constant_negbin/size_mixed/K20_N10_G100_T10':'G100',
                 'simulated/constant_negbin/size_mixed/K20_N10_G5000_T10':'G5000',
                 'simulated/constant_negbin/size_mixed/K20_N300_G10000_T20':'large-K20',
                 'simulated/constant_gaussian/size_mixed/K20_N10_G1000_T10':'Gaussian',
                 'simulated/constant/size_mixed/K20_N10_G1000_T10':'noiseless',
                 'simulated/constant_negbin_1e-1/size_mixed/K20_N10_G1000_T10':'Negbin\nmedium',
                 'simulated/constant_negbin_1e-2/size_mixed/K20_N10_G1000_T10':'Negbin\nhigh',
                 'simulated/constant_gaussian_100/size_mixed/K20_N10_G1000_T10':'Gaussian\nmedium',
                 'simulated/constant_gaussian_300/size_mixed/K20_N10_G1000_T10':'Gaussian\nhigh',
                 'simulated/constant_negbin/size_mixed_small/K20_N10_G1000_T10':'sparse',
                 'simulated/constant_negbin/size_mixed_large/K20_N10_G1000_T10':'dense',
                 'simulated/constant_negbin/square_size_mixed_small/K20_N10_G1000_T10':'sparse-square',
                 'simulated/constant_negbin/square_size_mixed_large/K20_N10_G1000_T10':'dense-square',
                 'simulated/constant/size_mixed_large/K5_N10_G1000_T10': 'large-K5',
                 'simulated/constant/size_mixed_large/K1_N10_G1000_T10': 'large-K1',
                 'simulated/constant/size_mixed_large/K2_N10_G1000_T10': 'large-K2',
                 'simulated/moran_gaussian/moran_spare_dense/K5_N10_G1000_T10': 'moran-K5',
                 'simulated/moran_gaussian/moran_spare_dense/K10_N30_G1000_T10': 'moran-K10',
                 'simulated/moran_gaussian/moran_spare_dense/K15_N30_G1000_T10': 'moran-K15',
                 'simulated/moran_gaussian/moran_spare_dense/K15_N300_G1000_T1': 'moran-K15-T1',
                 'simulated/shift_scale_0/size_mixed/K20_N10_G1000_T10': 'constant\nsamples',
                 'simulated/shift_scale_1/size_mixed/K20_N10_G1000_T10': 'shift',
                 'simulated/shift_scale_0_1/size_mixed/K20_N10_G1000_T10': 'scale-1',
                 'simulated/shift_scale_1_1/size_mixed/K20_N10_G1000_T10': 'shift-scale-1',
                 'simulated/shift_scale_0_5e-1/size_mixed/K20_N10_G1000_T10': 'scale',
                 'simulated/shift_scale_1_5e-1/size_mixed/K20_N10_G1000_T10': 'shift-scale',
                 'simulated/constant_negbin/size_mixed/K5_N10_G1000_T10':'K5',
                 'simulated/constant_negbin/size_mixed/K10_N10_G1000_T10':'K10',
                 'simulated/constant_negbin/size_mixed/K50_N10_G1000_T10':'K50',
                 'simulated/constant_negbin/size_mixed/K70_N10_G1000_T10':'K70',
                 'simulated/constant_negbin/size_mixed/K100_N300_G10000_T20':'large-K100',
                 'simulated/constant_negbin/size_mixed/K400_N300_G10000_T20':'large-K400',
                 'simulated/constant/size_mixed_large/K10_N10_G1000_T10':'sweep-Gaussian',
                 'simulated/constant_gaussian/size_mixed_small/K50_N10_G1000_T10':'sweep-noiseless'}

IMPC_DATASET_ORDER = [
    'Size factor',
    'Log',
    'Gaussian',
    'Size factor (Tensor)',
    'Log (Tensor)',
    'Gaussian (Tensor)'
]

IMPC_DATASET_NAMES = {
    'real/IMPC/deseq_sf/raw/small_pathways': 'Size factor',
    'real/IMPC/log/small_pathways': 'Log',
    'real/IMPC/quantnorm/small_pathways': 'Gaussian',
    'real/IMPC/tensor/deseq_sf/raw/small_pathways': 'Size factor (Tensor)',
    'real/IMPC/tensor/log/small_pathways': 'Log (Tensor)',
    'real/IMPC/tensor/quantnorm/small_pathways': 'Gaussian (Tensor)',
}

# Based on threshold_clust_err.png with K datasets - trying to keep it simple as possible
#   Threshold 1e-2, with the exception of Plaid which gets threshold 0 (no other option)
BEST_THRESHOLD = {'Plaid': '_thresh_0e+0',
                  'SDA': '_thresh_1e-2',
                  'nsNMF': '_thresh_1e-2',
                  'BicMix': '_thresh_1e-2',
                  'BicMix-Q': '_thresh_1e-2',
                  'BicMix_Q': '_thresh_1e-2',
                  'SSLB': '_thresh_1e-2',
                  'FABIA': '_thresh_1e-2',
                  'SNMF': '_thresh_1e-2',
                  'MultiCluster': '_thresh_1e-2'}

FAILURE_VALUES = {'clust_err': 0,
                  'tensor': 'non-tensor',
                  'traits_tissue_mean_f1_score': 0,
                  'traits_genotype_mean_f1_score': 0,
                  'traits_mean_f1_score': 0,
                  'traits_factors_mean_max_f1_score': 0,
                  'factors_pathways_nz_alpha 1': 0,
                  'factors_pathways_nz_alpha 0.1': 0,
                  'factors_pathways_nz_alpha 0.05': 0,
                  'factors_pathways_nz_alpha 0.01': 0,
                  'factors_pathways_nz_alpha 0.001': 0,
                  'factors_pathways_nz_alpha 0.0001': 0,
                  'factors_pathways_nz_alpha 1e-05': 0,
                  'ko_traits_nz_alpha 1': 0,
                  'ko_traits_nz_alpha 0.1': 0,
                  'ko_traits_nz_alpha 0.05': 0,
                  'ko_traits_nz_alpha 0.01': 0,
                  'ko_traits_nz_alpha 0.001': 0,
                  'ko_traits_nz_alpha 0.0001': 0,
                  'ko_traits_nz_alpha 1e-05': 0,
                  'recon_error_normalised': 1,
                  'recovered_K': 0,
                  'redundancy_average_max': 0,
                  'redundancy_max': 0,
                  'redundancy_mean': 0,
                  'adjusted_redundancy_mean': 0}

EXPECTED_SIMULATED_RUNIDS="analysis/accuracy/expected_method_dataset_run_ids.txt"
EXPECTED_IMPC_RUNIDS="analysis/IMPC/expected_method_dataset_run_ids.txt"
EXPECTED_IMPC_RUNIDS_ALL="analysis/IMPC/expected_method_dataset_run_ids_all.txt"

def read_result_binary_best_threshold(folder):
    method = folder.split('/')[1]
    threshold_str = BEST_THRESHOLD[method]
    threshold = utils.threshold_str_to_float(threshold_str)
    return utils.read_result_threshold_binary(folder, threshold)

def calculate_adjusted_mean_redundancy_IMPC(df):
    if 'recovered_K' not in df:
        assert df['recovered_K_y'].astype(int).equals(df['recovered_K_x'].astype(int)), \
                "Expected column 'recovered_K' in the dataframe, but if not then expected " \
                "columns recovered_K_x and recovered_K_y, which should be equal.\n" \
               f"{df['recovered_K_x'][:10]}\n{df['recovered_K_y'][:10]}"
        df['recovered_K'] = df['recovered_K_x']
    return calculate_adjusted_mean_redundancy(df)


def calculate_adjusted_mean_redundancy(df):
    # In construction of accuracy dataframes we calculated mean_redundancy as
    #   the mean of the Jaccard matrix, with diagonal entries set to 0.
    # We actually want the mean of *off-diagonal* entries.
    # Example showing the difference: 2 factors returned, identical.
    #   Jaccard matrix with diagonal 0 is [[0,1], [1,0]], which has mean 1/2
    #   off-diagonal mean is 1
    # Let S be the sum of the off-diagonal entries. We have mean_redundancy = S/K**2
    #   and want adjusted_mean_redundancy = S / (K**2 - K)
    #   So we should multiply the scores by (K**2 -K)/K**2, or equivalently (K-1)/K
    scale_factors = (df['recovered_K'] - 1)/(df['recovered_K'])
    adjusted = df['redundancy_mean'] * scale_factors
    return adjusted


def extract_run_info_IMPC(run_id):
    match = re.match(r'^run_seed_(\d+)_K_(\d+)(_qnorm_0)?$',
                     run_id)
    return match.groups()


def extract_dataset_info_IMPC(dataset_name):
    tensor = "(tensor|liver)?/?"
    preprocess = "(deseq/raw|deseq/log|deseq_sf/raw|quantnorm|scaled|raw|log)"
    gene_selection = "(pooled_cv|pooled_log|small_pathways|pooled)"
    num_genes = "/?(5000|10000)?"
    pattern = re.compile(f"real/IMPC/{tensor}{preprocess}/{gene_selection}{num_genes}$")
    match = re.match(pattern,
                     dataset_name)
    if match is None:
        logging.error(f"{dataset_name} doesn't match expected form for IMPC dataset")
    return match.groups()


def add_info_columns_IMPC(df):
    extracted_info = df['dataset'].apply(extract_dataset_info_IMPC)
    df['tensor'], df['preprocess'], df['gene_selection'], df['num_genes'] = zip(*extracted_info)
    print(df['tensor'].value_counts())
    df = df.fillna({'postprocessing': '_',
                    'tensor': 'non-tensor'})

    if 'run_id' in df.columns:
        df['_seed'], df['_K_init'], df['qnorm'] = zip(*df['run_id'].apply(extract_run_info_IMPC))
        df['_K_init'] = df['_K_init'].astype(int)

        df['_method'] = df['method'].copy()
        df.loc[(df['method'] == 'BicMix') & (df['qnorm'].isna()), 'method'] = 'BicMix-Q'

        df['method_dataset_run_id'] = df['_method'] + '/' + \
                                      df['dataset'] + '/' + \
                                      df['run_id']

    return df


def restrict_to_expected_runs_list(df, expected_runs_list):
    return df[df.method_dataset_run_id.isin(expected_runs_list)]


def restrict_to_expected_runs(df, expected_runs_file):
    if expected_runs_file is None:
        restricted = df
    else:
        with open(expected_runs_file, 'r') as (f):
            method_dataset_run_ids = [line.strip() for line in f.readlines()]
        restricted = restrict_to_expected_runs_list(df, method_dataset_run_ids)

    return restricted

def add_baseline_rows(df, baseline_df):
    """Add any rows from baseline_df whose dataset matches one of the datasets in the df,
    and whose method is 'baseline_XB_true'."""
    datasets = df.dataset.unique()
    restricted_baseline_df = baseline_df[baseline_df['dataset'].isin(datasets)]

    df_w_baseline = pd.concat([df,
                               restricted_baseline_df[restricted_baseline_df['method'] == 'baseline_XB_true']])
    df_w_baseline.method.replace({'baseline_XB_true': 'BASELINE'}, inplace=True)
    return df_w_baseline

def impc_pick_theoretical_best_K_init(row):
    theoretical_best_K_init = 50
    return theoretical_best_K_init == row['K_init']


def impc_restrict_to_best_theoretical_K_init(df):
    df_theoretical_best_K_init = df[df.apply(impc_pick_theoretical_best_K_init, axis=1)]
    return df_theoretical_best_K_init


def pick_theoretical_best_K_init(row):
    theoretical_best_K_init = row['K']
    if row['method'] in ('BicMix', 'BicMix-Q', 'SSLB'):
        if row['K'] == 20:
            theoretical_best_K_init = 25
        else:
            theoretical_best_K_init = row['K'] + 10
    return theoretical_best_K_init == row['K_init']


def pick_mean_best_K_init(row, best_K_init_dict):
    best_mean_K_init = best_K_init_dict[(row['method'], row['seedless_dataset'])]
    return best_mean_K_init == row['K_init']


def restrict_to_best_mean_K_init(df, param_to_optimise='clust_err'):
    means = df.groupby(['method', 'seedless_dataset', 'K_init'])[param_to_optimise].mean()
    best_K_init = pd.DataFrame(means.unstack().idxmax(axis=1)).reset_index()
    best_K_init.columns = ['method', 'seedless_dataset', 'K_init']
    print(best_K_init)
    best_K_init_dict = {(row['method'], row['seedless_dataset']) : row['K_init']
                        for row in best_K_init.to_dict('records')}
    df_mean_best_K_init = df[df.apply(lambda row: pick_mean_best_K_init(row, best_K_init_dict),
                                      axis=1)]
    return df_mean_best_K_init


def restrict_to_best_theoretical_K_init(df):
    df_theoretical_best_K_init = df[df.apply(pick_theoretical_best_K_init, axis=1)]
    return df_theoretical_best_K_init


def restrict_to_best_threshold(df):
    return df[df['processing'] == df['method'].map(BEST_THRESHOLD)]


def add_na_rows_expected_runs(df, expected_runs_file, processing=None, plaid_processing='_thresh_0e+0'):
    logging.info(f"Full list of columns in df is {list(df.columns)}")

    if expected_runs_file is None:
        combined = df
    else:
        with open(expected_runs_file, 'r') as f:
            method_dataset_run_ids = [line.strip() for line in f.readlines()]

        expected_runs = set(method_dataset_run_ids)
        logging.info(f"Expected {len(expected_runs)} runs, first: {logging_utils.get_example_element(expected_runs)}")
        actual_runs = set(df.method_dataset_run_id.unique())
        logging.info(f"Actually found {len(actual_runs)} runs, first: {logging_utils.get_example_element(actual_runs)}")
        failed_runs = expected_runs.difference(actual_runs)
        logging.info(f"Missing {len(failed_runs)} runs, first: {logging_utils.get_example_element(failed_runs)}")
        all_runs = '\n'.join(sorted(failed_runs))
        logging.debug(f"Missing runs:\n{all_runs}")

        if len(failed_runs) > 0:
            failed_runs_dicts = [read_information_from_mdr_id(failed_run) for failed_run in failed_runs]
            failed_runs_df = pd.DataFrame(failed_runs_dicts)
            logging.info(failed_runs_df)
            logging.info(f"Full list of columns in failed_runs_df is {list(failed_runs_df.columns)}")

            if processing is None:
                processing = list(df.processing.unique())
            logging.info(f"Using list of processing values: {processing}")
            assert plaid_processing in processing

            copies = []
            for processing_str in processing:
                copy = failed_runs_df.copy()
                copy['processing'] = processing_str
                copies.append(copy)
            failed_runs_processed = pd.concat(copies)
            logging.info(failed_runs_processed)
            logging.info(f"Full list of columns in failed_runs_processed is {list(failed_runs_processed.columns)}")

            failed_runs_processed = failed_runs_processed[(failed_runs_processed.method != 'Plaid') |
                                                          (failed_runs_processed.processing == plaid_processing)]

            combined = pd.concat([df, failed_runs_processed])
        else:
            combined = df

    return combined


def read_information_from_mdr_id(method_dataset_run_id):
    run_info = {}

    # If we have 'results/' at start, remove it
    if method_dataset_run_id.startswith('results/'):
        method_dataset_run_id = method_dataset_run_id[len('results/'):]

    split_mdr_id = method_dataset_run_id.split("/")
    run_info['method_dataset_run_id'] = method_dataset_run_id
    run_info['method'] = split_mdr_id[0]
    run_info['dataset'] = "/".join(split_mdr_id[1:-1])
    if 'seed' in run_info['dataset']:
        run_info['seedless_dataset'] = "/".join(split_mdr_id[1:-2])
    else:
        run_info['seedless_dataset'] = run_info['dataset']

    if 'simulated' in run_info['dataset']:
        match = re.match(r'simulated/(\w*)/(\w*)/K(\d+)_.*/seed_(\d+)',
                         run_info['dataset'])
        if match is not None:
            run_info['noise'] = match[1]
            run_info['K'] = int(match[3])
            run_info['sim_seed'] = match[4]
    elif 'IMPC' in run_info['dataset']:
        matches = extract_dataset_info_IMPC(run_info['dataset'])
        run_info['tensor'] = matches[0]
        run_info['preprocess'] = matches[1]
        run_info['gene_selection'] = matches[2]
        run_info['num_genes'] = matches[3]

    run_info['run_id'] = split_mdr_id[-1]
    match = re.match(r'run_seed_\d+_K_(\d+)', run_info['run_id'])
    run_info['K_init'] = int(match[1])

    run_info['_method'] = run_info['method']
    if run_info['method'] == 'BicMix':
        if 'qnorm_0' in run_info['run_id']:
            run_info['method'] = 'BicMix'
        else:
            run_info['method'] = 'BicMix-Q'
    return run_info


def construct_params_df(params_files):
    """Given a list of params.json files, read in each and label with the method, dataset
    and run_id indicated by the filename. Return as a dataframe, with each row corresponding
    to one params.json file."""
    params_dicts = []
    methods = []
    datasets = []
    run_ids = []

    for params_file in params_files:
        # Read in parameters from JSON file
        with open(params_file, 'r') as f:
            params = json.load(f)
        params_dicts.append(params)

        # Deduce the method, dataset and run_id from the filename
        match = re.match(r'[/\w]*results/(\w+)/([/\-\w]+)/(run_.+)/params.json$', params_file)
        methods.append(match.groups()[0])
        datasets.append(match.groups()[1])
        run_ids.append(match.groups()[2])

    # Concatenate all parameters into a dataframe
    #   and add method, dataset and run_id columns
    params_df = pd.DataFrame(params_dicts)
    params_df['method'] = methods
    params_df['dataset'] = datasets
    params_df['run_id'] = run_ids

    # Use method, dataset and run_id to index the rows
    params_df = params_df.set_index(['method', 'dataset', 'run_id'])

    return params_df


def include_all_rows(df):
    return [True for row in df.iterrows()]


def merge_columns(error_df, cols_to_merge, merged_name):
    df = error_df[cols_to_merge]

    for row in df.T.columns:
        # Check that there is at most one non-NA value for each row (allowing duplicates)
        all_values = df.T[row]
        unique_values = df.T[row].dropna().unique()

        num_unique_values = len(unique_values)
        if num_unique_values == 0:
            error_df.loc[row, merged_name] = np.nan
        elif num_unique_values == 1:
                # Assign this value to the new column
            error_df.loc[row, merged_name] = unique_values[0]
        else:
            logging.error(f"Expected at most one unique value across the columns {cols_to_merge}," \
                              f" found {num_unique_values}:\n{all_values}")
            assert False

    # Drop the old columns
    error_df.drop(cols_to_merge, axis=1)
    return error_df

def merge_K_init_columns(error_df,
                         K_init_names=['rank', 'K_init', 'num_comps', 'n_clusters'],
                         include_rows=include_all_rows):
    # Check that each row has no more than one K value selected (failed runs will have none)
    used_K_names = set(K_init_names).intersection(error_df.columns)
    num_K_names_used = error_df.loc[include_rows, used_K_names].notna().sum(axis=1)
    assert (num_K_names_used != 1).sum() == 0,\
        f"Each row should have value in one column corresponding to K: min {num_K_names_used.min()}, max " \
        f"{num_K_names_used.max()}.\n{error_df.loc[(num_K_names_used != 1), used_K_names]}"

    # Now that we've checked there's only one non-zero value for K,
    #   we can sum the possible columns to give the overall K value
    error_df['K_init'] = error_df[used_K_names].sum(axis=1)
    return error_df


def add_descriptive_columns(error_df):
    """
    Add columns such as 'seedless_dataset' and 'noise' to a dataframe, which
    are useful for plotting. Dataframe must have 'dataset' column with entries
    of the form 'simulated/<noise_type>/<shape>/K<K>*/seed_<sim_seed>'

    Dataframe will be updated in place.
    """
    # Extract information from dataset name - each (?P<name>) defines a
    #   named capture group
    dataset_regex = re.compile('(?P<seedless>simulated/(?P<noise>[\w-]*)/(?P<shape>\w*)/K(?P<K>\d+)_.*)/seed_(?P<seed>\d+)')
    matches = [re.match(dataset_regex, dataset) for dataset in error_df['dataset']]

    error_df.loc[:, 'seedless_dataset'] = [match['seedless'] for match in matches]
    noise_to_shortnoise_map = {'constant_negbin':   'negbin',
                               'constant_negbin_1e-1': 'negbin_medium',
                               'constant_negbin_1e-2': 'negbin_high',
                               'constant_gaussian_100': 'gaussian_medium',
                               'constant_gaussian_300': 'gaussian_high',
                               'constant_gaussian': 'gaussian',
                               'moran_gaussian': 'moran_gaussian',
                               'shift_scale_0': 'constant-genes',
                               'shift_scale_1': 'shift',
                               'shift_scale_0_1': 'scale-1',
                               'shift_scale_1_1': 'shift-scale-1',
                               'shift_scale_0_5e-1': 'scale',
                               'shift_scale_1_5e-1': 'shift-scale',
                               'constant':          'no_noise'}
    error_df.loc[:, 'noise'] = [noise_to_shortnoise_map[match['noise']]
                                for match in matches]
    error_df.loc[:, 'K'] = [match['K'] for match in matches]
    error_df.loc[:, 'sim_seed'] = [match['seed'] for match in matches]

    shape_to_size_map = {'size_mixed_large': 'dense',
                         'size_mixed_small': 'sparse',
                         'square_size_mixed_large': 'square_dense',
                         'square_size_mixed_small': 'square_sparse',
                         'size_mixed': 'medium'}
    error_df.loc[:, 'bicluster_size'] = [shape_to_size_map[match['shape']]
                                         for match in matches]

    # Combine other columns to give useful variables for plotting
    error_df.loc[:, 'short_seedless_dataset'] = error_df['noise'] + "_K" + error_df['K']
    error_df.loc[:, 'short_dataset'] = error_df['noise'] + "_K" + error_df['K'] + "_" + error_df['sim_seed']


def calculate_fold_diff_baseline(error_df):
    """Given a column 'recon_error', add the column 'recon_error_fold' which
    divides 'recon_error' by the recon_error achieved by the baseline. Each
    dataset must contain a row with method='baseline_XB_true' that corresponds
    to the frobenius norm of (Y_true - X_true * B_true.T)"""
    baseline = error_df[(error_df['method'] == 'baseline_XB_true')].groupby('dataset').mean()['recon_error']
    error_df.loc[:, 'baseline'] = error_df['dataset'].map(baseline)
    error_df.loc[:, 'recon_error_fold'] = error_df['recon_error'] / error_df['baseline']
    logging.info(f"Added column 'recon_error_fold' to dataframe in place")


def explode_df_on_column(df, column_to_expand):
    """Given the name of a column where each entry corresponds to a list,
    return a dataframe where the list of each row is transformed into len(list)
    rows, each of which contains one entry of the list and the remaining columns
    are identical.

    Lists from different rows need not have the same number of elements."""
    # Expand the dataframe by duplicating each row i len(list_i) times
    exploded = pd.DataFrame({
      col:np.repeat(df[col].values, df[column_to_expand].str.len())
      for col in df.columns.drop(column_to_expand)}
    )

    # Add the column with list entries back in
    exploded_with_col = exploded.assign(**{column_to_expand:np.concatenate(df[column_to_expand].values)})[df.columns]

    return exploded_with_col


def explode_on_stringlist_column(df, stringlist_column):
    """Given the name of a column where each entry is a string encoding a list,
    return a dataframe where the list of each row is transformed into len(list)
    rows, each of which contains one entry of the list and the remaining columns
    are identical.

    Lists from different rows need not have the same number of elements."""
    copied_df = df.copy()
    copied_df = copied_df.dropna(subset=[stringlist_column])
    copied_df[stringlist_column] = copied_df[stringlist_column].map(yaml.safe_load)
    exploded = explode_df_on_column(copied_df, stringlist_column)
    return exploded


def replace_nan_with_empty_list(column_to_replace, list_column_to_match, df):
    """Given the name of a column that either contains strings that encode lists
    (e.g. "[1, 2, 3]") or nan values, replace the strings encoding lists with
    the lists themselves, and nan values with a list of np.nan values, where
    the length of the list in each row should match the length of the list
    in the list_column_to_match column.

    Return the column values."""
    column = df.apply(
        lambda row:
            yaml.safe_load(row[column_to_replace]) if isinstance(row[column_to_replace], str)
            else [np.nan for x in row[list_column_to_match]],
        axis=1)
    return column


def explode_on_matched_columns(df, safe_columns, other_columns):
    """Given the name of multiple columns where each entry is a string encoding
    a list, and where for each row the lists in all columns are the same length,
    return a dataframe where the each row is transformed into len(list)
    rows, each of which contains one entry of the various lists and the
    remaining columns are identical.

    The columns are split into 'safe_columns', which must always contain strings
    that encode lists and 'other_columns' which can sometimes be np.nan. If
    a column from other_columns has a np.nan entry in some row, it will be
    replaced with a list of np.nan values, with the list the same length
    as the lists in safe_columns for that row.

    Lists from different rows need not have the same number of elements."""
    stringlist_columns = safe_columns + other_columns

    copied_df = df.copy()
    # Only keep rows where at least one of the stringlist columns is present
    copied_df = copied_df.dropna(subset=stringlist_columns, how='all')

    # Map the safe columns from strings (strings encoding lists) to lists
    for stringlist_column in safe_columns:
        copied_df[stringlist_column] = copied_df[stringlist_column].map(yaml.safe_load)

    for column in other_columns:
        # Replace any nan values with an empty list, matching the list lengths
        #     from one of the safe columns
        copied_df[column] = replace_nan_with_empty_list(column,
                                                        safe_columns[0],
                                                        copied_df)

    exploded = pd.DataFrame({
      col:np.repeat(copied_df[col].values, copied_df[stringlist_columns[0]].str.len())
      for col in copied_df.columns.drop(stringlist_columns)}
    )
    exploded_with_col = exploded.assign(**{column_to_expand:np.concatenate(copied_df[column_to_expand].values)
                                           for column_to_expand in stringlist_columns})[df.columns]

    return exploded_with_col


def add_factor_info_to_df(exploded_rec_df, factor_info_dfs):
    """Each row of exploded_rec_df should correspond to the best recovery score
    for some true factor. Rows from the same method/dataset/run_id should be
    together and e.g. the 1st row from a method/dataset/run_id should be
    the Jaccard index of the best match between a recovered factor and the 1st
    *true* factor.

    This function will add columns describing the true factors, e.g. their
    size and frobenius norm."""
    dfs = []

    # Separate into one dataframe for each method/dataset/run_id
    try:
        grouped = exploded_rec_df.groupby(['method_dataset_run_id', 'processing'],
                                          as_index=False)
    except KeyError as e:
        grouped = exploded_rec_df.groupby(['method', 'dataset', 'processing'],
                                          as_index=False)

    for name, df in list(grouped):
        dataset = df.dataset.iloc[0]
        assert df.dataset.nunique() == 1, \
                f"Expecting only one dataset in this grouped df: {name}"

        # Extract the factor_information we will add
        this_factor_info = factor_info_dfs[dataset]
        assert this_factor_info.shape[0] == df.shape[0], \
               f"Unexpected shape of factor_info {this_factor_info.shape} for dataset {dataset}, " \
               f"expected shape {df.shape[0]}"
        df.reset_index(drop=True, inplace=True)
        dfs.append(pd.concat([df, this_factor_info], axis=1, join='outer'))

    return pd.concat(dfs)


def improve_method_info(error_df):
    error_df['_method'] = error_df['method'].copy()
    if 'qnorm' in error_df.columns:
        error_df.loc[(error_df['method'] == 'BicMix') &
                     (error_df['qnorm'] == 1), 'method'] = 'BicMix-Q'
    else:
        error_df.loc[(error_df['method'] == 'BicMix') &
                     (~error_df['run_id'].str.contains('qnorm_0')), 'method'] = 'BicMix-Q'
    error_df['method_dataset_run_id'] = error_df['_method'] + '/' + \
                                        error_df['dataset'] + '/' + \
                                        error_df['run_id']
    return error_df


def setup_error_dfs(binary_thresholds_error_file, exact_thresholds_error_file, folder=''):
    logging.info('Reading in binary error')
    binary_error = pd.read_csv(binary_thresholds_error_file, index_col=None)
    binary_error.fillna({'processing': '_thresh_0e+0'}, inplace=True)

    logging.info('Reading in exact error')
    exact_error = pd.read_csv(exact_thresholds_error_file, index_col=None)
    exact_error.fillna({'processing': '_thresh_0e+0'}, inplace=True)

    logging.info('Adding extra columns to error dataframes')
    add_descriptive_columns(binary_error)
    add_descriptive_columns(exact_error)
    calculate_fold_diff_baseline(exact_error)

    logging.info('Merging binary error and exact error into one dataframe')
    combined_df = pd.merge(left=binary_error, right=exact_error, how='outer')

    results_folder = os.path.join(folder, 'results')
    logging.info(f"Reading in all parameter information from folder {results_folder}")
    params_files = [str(path) for path in Path(results_folder).rglob('params.json')]
    params_df = construct_params_df(params_files).reset_index()

    logging.info('Adding parameter information')
    binary_error = pd.merge(left=binary_error, right=params_df, how='left')
    merge_K_init_columns(binary_error)

    exact_error = pd.merge(left=exact_error, right=params_df, how='left')
    combined_df = pd.merge(left=combined_df, right=params_df, how='left')
    is_non_baseline_row = exact_error.method != 'baseline_XB_true'
    merge_K_init_columns(exact_error, include_rows=is_non_baseline_row)
    is_non_baseline_row = combined_df.method != 'baseline_XB_true'
    merge_K_init_columns(combined_df, include_rows=is_non_baseline_row)

    logging.info(f"Reading in all benchmarking information from folder {results_folder}")
    benchmark_files = [str(path) for path in Path(results_folder).rglob('benchmark.txt')]
    benchmark_df = benchmarking.construct_benchmark_df(benchmark_files).reset_index()

    logging.info('Adding benchmarking information')
    binary_error = pd.merge(left=binary_error, right=benchmark_df, how='left')
    exact_error = pd.merge(left=exact_error, right=benchmark_df, how='left')
    combined_df = pd.merge(left=combined_df, right=benchmark_df, how='left')
    binary_error = improve_method_info(binary_error)
    binary_error = binary_error[(~binary_error.duplicated(subset=['method_dataset_run_id', 'processing'], keep=False)) |
                                (~binary_error.redundancy_average_max.isna())]
    combined_df = improve_method_info(combined_df)
    combined_df = combined_df[~combined_df.duplicated(subset=['method_dataset_run_id', 'processing'], keep='last')]
    exact_error = improve_method_info(exact_error)
    exact_error = exact_error[~exact_error.duplicated(subset=['method_dataset_run_id', 'processing'], keep='first')]
    params_df = improve_method_info(params_df)
    benchmark_df = improve_method_info(benchmark_df)
    out_dict = {'binary_error':        binary_error,
                'exact_error':         exact_error,
                'combined_error':      combined_df,
                'benchmark_df':        benchmark_df,
                'params_df':           params_df}
    return out_dict


def construct_factor_recovery_df(error_df_file, datasets):
    combined_df_ext = pd.read_csv(error_df_file, index_col=None)
    logging.info('Producing dataframe with individual factor recovery scores')
    exploded_rec = explode_on_matched_columns(combined_df_ext, [
                                                 'jaccard_recovery_scores',
                                                 'jaccard_recovery_idx'], [])

    logging.info('Reading in factor_info dataframes')
    factor_info_dfs = {}
    for dataset in datasets:
        factor_info_df = pd.read_csv(f"analysis/accuracy/data/{dataset}/factor_info.csv",
                                     index_col=0)
        factor_info_df['factor_index'] = factor_info_df.index
        factor_info_dfs[dataset] = factor_info_df

    logging.info('Adding factor_info to dataframe of individual recovery scores')
    exploded_rec_w_info = add_factor_info_to_df(exploded_rec, factor_info_dfs)

    return exploded_rec_w_info


def setup_error_dfs_with_factor_info(datasets, rec_file, rel_file, rec_factor_info_file, folder=''):
    combined_df_ext = pd.read_csv('analysis/accuracy/combined_accuracy_ext.csv', index_col=None)
    logging.info('Producing dataframe with individual factor recovery scores')
    try:
        exploded_rec = pd.read_csv(rec_file, index_col=None)
    except FileNotFoundError:
        exploded_rec = explode_on_matched_columns(combined_df_ext, [
         'jaccard_recovery_scores',
         'jaccard_recovery_idx'], [])
        exploded_rec.to_csv(rec_file, index=False)

    logging.info('Producing dataframe with individual factor relevance scores')
    try:
        exploded_rel = pd.read_csv(rel_file, index_col=None)
    except FileNotFoundError:
        exploded_rel = explode_on_matched_columns(combined_df_ext, [
         'jaccard_relevance_scores',
         'jaccard_relevance_idx'], [])
        exploded_rel.to_csv(rel_file, index=False)

    logging.info('Reading in factor_info dataframes')
    factor_info_dfs = {dataset: pd.read_csv(f"{folder}analysis/accuracy/data/{dataset}/factor_info.csv",
                                            index_col=0)
                       for dataset in datasets}

    logging.info('Adding factor_info to dataframe of individual recovery scores')
    exploded_rec_w_info = add_factor_info_to_df(exploded_rec, factor_info_dfs)
    exploded_rec_w_info.to_csv(rec_factor_info_file, index=False)

    out_dict = {'factor_info_dfs':  factor_info_dfs,
                'exploded_rec':     exploded_rec_w_info,
                'exploded_rel':     exploded_rel}
    return out_dict


def construct_factor_info_df(datasets, error_df):
    error_df = error_df[error_df['seedless_dataset'].isin(datasets)]
    print(error_df.shape)
    error_df['processing_value'] = error_df.processing.str.extract('_thresh_(\\de[+-]\\d)').astype(float)
    factor_info_run_dfs = []
    for index, row in error_df.iterrows():
        factor_info_run_df = acc.construct_factor_info_threshold(f"results/{row['method_dataset_run_id']}", row['processing_value'])
        factor_info_run_df['method'] = row['method']
        factor_info_run_df['method_dataset_run_id'] = row['method_dataset_run_id']
        factor_info_run_df['dataset'] = row['dataset']
        factor_info_run_df['seedless_dataset'] = row['seedless_dataset']
        factor_info_run_df['processing'] = row['processing']
        factor_info_run_dfs.append(factor_info_run_df)

    run_factor_info = pd.concat(factor_info_run_dfs)
    dataset_factor_info_dfs = {dataset:pd.read_csv(f"analysis/accuracy/data/{dataset}/factor_info.csv", index_col=0) for dataset in error_df.dataset}
    for dataset, df in dataset_factor_info_dfs.items():
        df['dataset'] = dataset

    dataset_factor_info = pd.concat(dataset_factor_info_dfs.values())
    add_descriptive_columns(dataset_factor_info)
    dataset_factor_info['method'] = 'baseline'
    return pd.concat([run_factor_info, dataset_factor_info])


def recover_binary_at_threshold(run_folder, threshold=None):
    method = run_folder.split('/')[1]
    if threshold is None:
        threshold = utils.threshold_str_to_float(BEST_THRESHOLD[method])
    elif not isinstance(threshold, float):
        raise AssertionError
    K, X_bin, B_bin = utils.read_result_threshold_binary(run_folder, threshold)

    return K, X_bin, B_bin

