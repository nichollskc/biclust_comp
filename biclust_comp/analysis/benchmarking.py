import json
import os
from pathlib import Path
import re

import pandas as pd

from biclust_comp.analysis import plots
from biclust_comp.analysis import accuracy_utils as acc_utils

# List of the headers given in a snakemake benchmark.txt file
BENCHMARK_MEASURES = ['s', 'h:m:s', 'max_rss', 'max_vms', 'max_uss', 'max_pss',
                      'io_in', 'io_out', 'mean_load']

def get_benchmark_filename(method_dataset_runid):
    return f"results/{method_dataset_runid}/benchmark.txt"


def get_params_filename(method_dataset_runid):
    return f"results/{method_dataset_runid}/params.json"


def find_all_benchmark_params_pairs(folder):
    benchmark_files = []
    params_files = []

    # Loop through all benchmark.txt files
    for benchmark_path in Path(folder).rglob('benchmark.txt'):

        # See if there is a corresponding params file
        params_path = f"{os.path.dirname(benchmark_path)}/params.json"

        # If so, we have a pair of benchmark.txt and params.json files
        #   so add both to their lists
        if os.path.isfile(params_path):
            benchmark_files.append(str(benchmark_path))
            params_files.append(str(params_path))

    return benchmark_files, params_files


def find_existing_benchmark_files(method_dataset_runids):
    """Given a list of method_dataset_runids, restrict to those that have both a correpsonding
    benchmark.txt file and a params.json file. Return the list of all benchmark.txt files
    and all params.json files."""
    # Construct a list of tuples (benchmark_file, params_file)
    #   one for each method_dataset_runid
    file_pairs = [(get_benchmark_filename(method_dataset_runid),
                   get_params_filename(method_dataset_runid))
                   for method_dataset_runid in method_dataset_runids]

    # Restrict the list to pairs where both files exist
    existing_files = [pair for pair in file_pairs
                      if (os.path.isfile(pair[0]) and os.path.isfile(pair[1]))]

    # Split the list into two lists
    #   one of benchmark_files and one of params_files
    benchmark_files = [pair[0] for pair in existing_files]
    params_files = [pair[1] for pair in existing_files]
    return benchmark_files, params_files


def construct_benchmark_df(benchmark_files):
    """Given a list of benchmark.txt files, read in each and label it with the
    method, dataset and run_id indicated by the filename. Return this as a dataframe,
    with each row corresponding to one benchmark.txt file.

    This assumes that each benchmark.txt file contains only one line (plus the header)."""
    bench_df_list = []
    methods = []
    datasets = []
    run_ids = []

    for filename in benchmark_files:
        # Read in the benchmark data
        df = pd.read_csv(filename, index_col=None, header=0, sep='\t', na_values=['-'])
        bench_df_list.append(df)

        # Deduce the method, dataset and run_id from the filename
        match = re.match(r'[/\w]*results/(\w+)/([-/\w]+)/(run_.+)/benchmark.txt$', filename)
        if match is not None:
            methods.append(match.groups()[0])
            datasets.append(match.groups()[1])
            run_ids.append(match.groups()[2])
        else:
            methods.append("")
            datasets.append("")
            run_ids.append("")

    # Concatenate all benchmark results into a dataframe
    #   and add method, dataset and run_id columns
    bench_df = pd.concat(bench_df_list, axis=0, ignore_index=True)
    bench_df['method'] = methods
    bench_df['dataset'] = datasets
    bench_df['run_id'] = run_ids

    # Use method, dataset and run_id to index the rows
    bench_df = bench_df.set_index(['method', 'dataset', 'run_id'])

    return bench_df


def construct_combined_df(method_dataset_runids, merge_K=True, deduce_TNG=True):
    """Construct a dataframe consisting of benchmark data and parameter information for
    all runs. method_dataset_runids should be a list of strings, each of the form
    '{method}/{dataset}/run_{runid}'. We will search for the benchmark.txt and params.json
    files at
    'results/{method_dataset_runid}/benchmark.txt' and
    'results/{method_dataset_runid}/params.json'

    If merge_K is True, all columns that correspond to K (number of factors) are merged into
    a single column called K.

    If deduce_TNG is True, columns T, N and G are added, with the values extracted from the
    dataset name (assuming GTEx/GTEx_T\d+_N\d+_G\d+ form)."""

    benchmark_files, params_files = find_existing_benchmark_files(method_dataset_runids)
    combined = construct_combined_df_raw(benchmark_files,
                                         params_files,
                                         merge_K=merge_K,
                                         deduce_TNG=deduce_TNG)
    return combined


def construct_combined_df_all(folder, merge_K=True, deduce_TNG=True):
    benchmark_files, params_files = find_all_benchmark_params_pairs(folder)
    combined = construct_combined_df_raw(benchmark_files,
                                         params_files,
                                         merge_K=merge_K,
                                         deduce_TNG=deduce_TNG)
    return combined


def construct_combined_df_raw(benchmark_files, params_files, merge_K=True, deduce_TNG=True):

    bench_df = construct_benchmark_df(benchmark_files)
    params_df = acc_utils.construct_params_df(params_files)

    combined = bench_df.join(params_df)

    if merge_K:
        # Check that each row has no more than one K value selected (failed runs will have none)
        K_names = ['rank', 'K_init', 'num_comps']
        used_K_names = set(K_names).intersection(combined.columns)
        assert (combined[used_K_names].notna().sum(axis=1) > 1).sum() == 0,\
            f"Each row should have value in one column corresponding to K {combined[used_K_names]}"

        # Now that we've checked there's only one non-zero value for K,
        #   we can sum the possible columns to give the overall K value
        combined['K'] = combined[set(used_K_names).intersection(combined.columns)].sum(axis=1)
        combined.drop(columns=used_K_names)

    if deduce_TNG:
        # Deduce T, N and G values for each dataset, using the regex structure
        #   GTEx/GTEx_T\d+_N\d+_G\d+
        dataset_names = combined.index.get_level_values('dataset')

        combined['T'] = dataset_names.str.extract(r'GTEx/GTEx_T(\d+)_N\d+_G\d+').values
        combined['N'] = dataset_names.str.extract(r'GTEx/GTEx_T\d+_N(\d+)_G\d+').values
        combined['G'] = dataset_names.str.extract(r'GTEx/GTEx_T\d+_N\d+_G(\d+)').values

    return combined

