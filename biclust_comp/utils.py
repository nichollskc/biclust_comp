import errno
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import statsmodels.stats.multitest as multitest

from biclust_comp import logging_utils


def threshold_float_to_str(threshold_float):
    threshold_str = f"_thresh_{np.format_float_scientific(threshold_float, trim='-', exp_digits=1)}"
    return threshold_str


def threshold_str_to_float(threshold_str):
    threshold_float = float(re.match(r'_thresh_(\de[-+]\d)', threshold_str)[1])
    return threshold_float


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_list_from_file(*filename_parts, strip_quotes=False):
    full_name = os.path.join(*filename_parts)
    if not os.path.exists(full_name):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), full_name)
    with open(full_name, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    if strip_quotes:
        lines = [l.strip('"\'') for l in lines]

    return lines


def read_int_from_file(*filename_parts):
    full_name = os.path.join(*filename_parts)
    if not os.path.exists(full_name):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), full_name)
    with open(full_name, 'r') as f:
        contents = f.read()
    return int(contents.strip())


def read_np(*filename_parts):
    full_name = os.path.join(*filename_parts)
    if not os.path.exists(full_name):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), full_name)
    return np.loadtxt(full_name, ndmin=2)


def save_np(filename, obj):
    np.savetxt(filename, obj, delimiter='\t')


def read_df(folder, filename):
    full_name = os.path.join(folder, filename)
    return read_matrix_tsv(full_name)


def read_matrix_tsv(filename):
    return pd.read_csv(filename, delim_whitespace=True, index_col=None, header=None)


def combine_inds_tissues(A, Z):
    """
    Given a matrices corresponding to individuals (A) and to tissues (Z), return
    a matrix where rows now correspond to samples i.e. all tissues from all
    individuals. They will be arranged with all individuals from tissue 1, then
    all from tissue 2 etc.

    Args:
        A: (N, K) np.ndarray with each row corresponding to one individual
        Z: (T, K) np.ndarray with each row corresponding to one tissue

    Returns:
        Matrix with same dtype as A, with each column having entries
        (A_1k * Z_1k, A_2k * Z_1k, ..., A_Nk * Z_1k, A_1k * Z_2k, ..., A_Nk * Z_Tk)
    """
    N, K = A.shape
    T, K_ = Z.shape
    assert K == K_, f"A and Z not compatible shapes to combine: A has shape"\
        " {A.shape}, Z has shape {Z.shape} - need same number of columns"

    X = np.zeros((N*T, K), dtype=A.dtype)
    for k in range(K):
        logging.debug(f"Factor {k}, Z then A")
        logging.debug(Z[:10,k])
        logging.debug(A[:10,k])
        X[:,k] = np.kron(Z[:,k], A[:,k])
    return X


def remove_empty_factors(X_raw, B_raw, threshold=0):
    assert type(X_raw) == np.ndarray
    assert type(B_raw) == np.ndarray
    assert X_raw.shape[1] == B_raw.shape[1]

    factors_zero_X = np.all(abs(X_raw) <= threshold, axis=0)
    factors_zero_B = np.all(abs(B_raw) <= threshold, axis=0)

    # If a factor has all of X[:,k] equal to 0, or all of B[:,k] equal to 0,
    #   this is an empty factor and we should exclude it
    factors_zero = (factors_zero_X) | (factors_zero_B)

    X_nonempty = X_raw[:, ~ factors_zero]
    B_nonempty = B_raw[:, ~ factors_zero]

    return X_nonempty, B_nonempty


def binarise_matrix(matrix):
    return (matrix != 0).astype(int)


def remove_below_threshold(matrix, threshold):
    copied = matrix.copy()
    copied[abs(matrix) < threshold] = 0
    return copied


def remove_below_threshold_norm_1(matrix, threshold):
    copied = matrix.copy()
    norms = np.linalg.norm(copied, axis=0)
    copied[abs(matrix / norms) < threshold] = 0
    return copied


def scale_factors_same_norm(X, B):
    X_norms = np.linalg.norm(X, axis=0)
    B_norms = np.linalg.norm(B, axis=0)
    logging.debug(X_norms)
    logging.debug(B_norms)
    combined_norms = X_norms * B_norms
    X_scaled = X * np.sqrt(combined_norms) / X_norms
    B_scaled = B * np.sqrt(combined_norms) / B_norms
    return X_scaled, B_scaled


def scale_factors_by_X(X, B, desired_X_norm=1):
    X_norms = np.linalg.norm(X, axis=0)
    print(X_norms)
    X_scaled = X / X_norms * desired_X_norm
    B_scaled = B * X_norms / desired_X_norm
    return X_scaled, B_scaled


def save_plot(filename):
    # Create directory if it doesn't alreayd exist
    directory = Path(filename).parent
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(filename,
                bbox_inches='tight',
                dpi=300)
    logging.info(f"Saved figure to {filename}")


def transform_dict_to_count_df(item_dict):
    """Given a dictionary, where each element of the dictionary is a list,
    return the data frame with columns all the elements that occur in any of
    the lists (union of the lists) and rows the keys of the dictionary. Each
    entry is the number of times that the item occurs in the list given by key.

    Input:
    {'A': ['1', '2'], 'B': ['2', '3', '5']}
    Output:
        1   2   3   5
    A   1   1   0   0
    B   0   1   1   1
    """
    # Assemble df where each column corresponds to one list and there are
    #   max(len(list)) rows
    df = pd.concat([pd.Series(v, name=k).astype(str)
                    for k, v in item_dict.items()],
                   axis=1)

    # Transform so that we have the desired form
    return pd.get_dummies(df.stack()).sum(level=1)


def correct_multiple_testing(pval_df, alpha=0.0001):
    """Given a data frame filled with p-values, perform the FDR Benjamini-Yekutieli
    multiple testing adjustment to the p-values. Constrain the FDR to alpha

    Returns:

    reject
    dataFrame of bools indicating which null hypotheses should be rejected

    corrected_pvals
    dataFrame of p-values corrected for multiple testing (unaffected by alpha)
    """
    flattened = pval_df.values.flatten()
    reject_flat, corrected_flat, _, _ = multitest.multipletests(flattened, alpha=alpha, method='fdr_by')

    reject = pd.DataFrame(reject_flat.reshape(pval_df.shape))
    reject.index = pval_df.index
    reject.columns = pval_df.columns

    corrected = pd.DataFrame(corrected_flat.reshape(pval_df.shape), dtype ='float64')
    corrected.index = pval_df.index
    corrected.columns = pval_df.columns

    return reject, corrected


def scaled_to_thresholded(X_scaled, B_scaled, threshold):
    X_thr_raw = remove_below_threshold_norm_1(X_scaled, threshold)
    B_thr_raw = remove_below_threshold_norm_1(B_scaled, threshold)
    X_thr, B_thr = remove_empty_factors(X_thr_raw, B_thr_raw)

    K = X_thr.shape[1]
    return K, X_thr, B_thr


def scaled_to_thresholded_binary(X_scaled, B_scaled, threshold):
    K, X_thr, B_thr = scaled_to_thresholded(X_scaled, B_scaled, threshold)
    X_thr_bin = binarise_matrix(X_thr)
    B_thr_bin = binarise_matrix(B_thr)
    return K, X_thr_bin, B_thr_bin


def read_result_threshold_binary(run_folder, threshold):
    K, X_thr, B_thr = read_result_threshold(run_folder, threshold)
    X_bin = binarise_matrix(X_thr)
    B_bin = binarise_matrix(B_thr)
    return K, X_bin, B_bin


def read_result_scaled(run_folder):
    read_fail = False
    try:
        X_raw = read_np(run_folder, 'X.txt')
        B_raw = read_np(run_folder, 'B.txt')
    except ValueError:
        # Plaid just makes an empty file when K=0,
        #   so we have to look out for being unable to read the matrix
        logging.info(f"Value error detected when reading files from {run_folder}")
        read_fail = True

    if not read_fail:
        X, B = remove_empty_factors(X_raw, B_raw)

        if run_folder.split('/')[1] == 'Plaid':
            X_scaled_raw, B_scaled_raw = X, B
            np.testing.assert_array_equal(binarise_matrix(X_scaled_raw),
                                          X,
                                          "Expecting Plaid to have binary matrix as X.txt")
        else:
            X_scaled_raw, B_scaled_raw = scale_factors_same_norm(X, B)

        X_scaled, B_scaled = remove_empty_factors(X_scaled_raw, B_scaled_raw)
        K = X_scaled.shape[1]
    else:
        K = 0
        X_scaled = np.zeros((0,0))
        B_scaled = np.zeros((0,0))
    return K, X_scaled, B_scaled


def read_result_threshold(run_folder, threshold):
    if run_folder.split('/')[1] == 'Plaid' and threshold != 0:
        K_thr = 0
        X_thr = np.zeros((0,0))
        B_thr = np.zeros((0,0))
    else:
        K_scaled, X_scaled, B_scaled = read_result_scaled(run_folder)
        K_thr, X_thr, B_thr = scaled_to_thresholded(X_scaled, B_scaled, threshold)

    return K_thr, X_thr, B_thr


def save_reconstructed_X(A_file, Z_file, X_outfile):
    A = read_np(A_file)
    Z = read_np(Z_file)

    X = combine_inds_tissues(A=A, Z=Z)
    save_np(X_outfile, X)

def save_json(dictionary, jsonfile):
    with open(jsonfile, 'w') as f:
        json.dump(dictionary, f, indent=2)

def load_json(jsonfile):
    with open(jsonfile, 'r') as f:
        dictionary = json.load(f)
    return dictionary

def extract_dataset_from_mdr(mdr):
    split_mdr = mdr.split('/')
    dataset = "/".join(split_mdr[1:-1])
    return dataset


def read_error_df(maybe_df_file, *args, **kwargs):
    """Accept either a dataframe or a filename"""
    if isinstance(maybe_df_file, pd.DataFrame):
        df = maybe_df_file
    else:
        assert isinstance(maybe_df_file, str), \
            f"Expecting either dataframe or filename, instead got {type(maybe_df_file)}"
        df = pd.read_csv(maybe_df_file, *args, **kwargs)
    return df
