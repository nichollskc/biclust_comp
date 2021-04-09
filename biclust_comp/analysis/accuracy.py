import glob
import logging
import os
import re

import munkres
import numpy as np
import pandas as pd

from biclust_comp import logging_utils, utils

RECON_METHODS=['BicMix', 'nsNMF', 'SDA', 'SNMF', 'SSLB']

def calculate_union_size_combined(X_bin_a, B_bin_a, X_bin_b, B_bin_b):
    """
    Calculate the size of the union counting overlapping, as defined by Eqn 3
    in Horta & Campello 2014.

    Args:
        X_bin_a: (N, K_a) binary np.ndarray where X_bin_a[i, k] indicates
            whether row i is included in bicluster k of biclustering A
        B_bin_a: (G, K_a) binary np.ndarray where B_bin_a[j, k] indicates
            whether column j is included in bicluster k of biclustering A
        X_bin_b: (N, K_b) binary np.ndarray where X_bin_b[i, k] indicates
            whether row i is included in bicluster k of biclustering B
        B_bin_b: (G, K_b) binary np.ndarray where B_bin_b[j, k] indicates
            whether column j is included in bicluster k of biclustering B

    Returns:
       float giving size of the union counting overlaps
    """
    # Count number of times each element (i, j) is contained in a bicluster
    #   from biclustering a - bic_counts_a[i, j] = number of biclusters
    #   containing element (i, j)
    bic_counts_a = np.matmul(X_bin_a, B_bin_a.T)
    # Similarly count for each (i, j) with biclustering b
    bic_counts_b = np.matmul(X_bin_b, B_bin_b.T)

    # The combined union size is then the sum over all (i, j) of the maximum
    #   of bic_counts_a[i, j] and bic_counts_b[i, j]
    max_bic_counts = np.maximum(bic_counts_a, bic_counts_b)
    return max_bic_counts.sum()


def count_intersection_1D_arrays(array_a, array_b):
    """
    Count number of indices i such that array_a[i] and array_b[i] both non-zero

    Args:
        array_a: one-dimensional np.ndarray
        array_b: one-dimensional np.ndarray

    Returns:
        int giving size of intersection of the two arrays
    """
    non_zero_a = np.where(array_a == 1)[0]
    non_zero_b = np.where(array_b == 1)[0]

    non_zero_both = np.intersect1d(non_zero_a, non_zero_b)
    return len(non_zero_both)


def calc_overlaps(X_bin_a, B_bin_a, X_bin_b, B_bin_b):
    """
    Calculate the overlaps between the biclusters given by (X_bin_a, B_bin_a)
    and by (X_bin_b, B_bin_b). In particular, consider for each pair (k1, k2)
    in {1, ..., K_a} x {1, ..., K_b} the biclusters
    (X_bin_a[:, k1], B_bin_a[:, k1]) and
    (X_bin_b[:, k2], B_bin_b[:, k2]) and compute
        (1) the size of the intersection of the two biclusters
        (2) the size of the union of the two biclusters

    Args:
        X_bin_a: (N, K_a) binary np.ndarray where X_bin_a[i, k] indicates
            whether row i is included in bicluster k of biclustering A
        B_bin_a: (G, K_a) binary np.ndarray where B_bin_a[j, k] indicates
            whether column j is included in bicluster k of biclustering A
        X_bin_b: (N, K_b) binary np.ndarray where X_bin_b[i, k] indicates
            whether row i is included in bicluster k of biclustering B
        B_bin_b: (G, K_b) binary np.ndarray where B_bin_b[j, k] indicates
            whether column j is included in bicluster k of biclustering B

    Returns:
        intersect_matrix: np.ndarray where entry (k1, k2) gives the number of
            entries of the (N, G) matrix that are contained in both
            bicluster k1 of biclustering A and bicluster k2 of biclustering B
        union_matrix: np.ndarray where entry (k1, k2) gives the number of
            entries of the (N, G) matrix that are contained in either
            bicluster k1 of biclustering A or bicluster k2 of biclustering B

    """
    # Count number of biclusters in each biclustering
    K_a = X_bin_a.shape[1]
    K_b = X_bin_b.shape[1]

    # Set up empty matrices
    intersect_matrix = np.zeros((K_a, K_b))
    union_matrix = np.zeros((K_a, K_b))

    # Cycle through each pair of biclusters k_a, k_b
    for k_a in range(K_a):
        X_a_k = X_bin_a[:, k_a]
        B_a_k = B_bin_a[:, k_a]
        for k_b in range(K_b):
            X_b_k = X_bin_b[:, k_b]
            B_b_k = B_bin_b[:, k_b]

            # Count number of rows that overlap between the biclusters
            rows_overlapping = count_intersection_1D_arrays(X_a_k, X_b_k)
            # Count number of columns that overlap
            cols_overlapping = count_intersection_1D_arrays(B_a_k, B_b_k)

            # Total number of elements (cells) that overlap is the product
            cells_overlapping = rows_overlapping * cols_overlapping
            intersect_matrix[k_a, k_b] = cells_overlapping

            # Union of two sets is generally |X u Y| = |X| + |Y| - |X n Y|
            size_a = X_a_k.sum() * B_a_k.sum()
            size_b = X_b_k.sum() * B_b_k.sum()
            union_matrix[k_a, k_b] = size_a + size_b - cells_overlapping

    return intersect_matrix, union_matrix


def calc_clust_error(intersect_matrix, union_size):
    # The Hungarian algorithm minimises the sum of costs. We really have a
    #   'profit' matrix, where each entry gives the profit/reward for pairing
    #   two biclusters together. Thus multiply by -1 to give a cost matrix.
    cost_matrix = -1 * intersect_matrix
    optimal_pairings = munkres.Munkres().compute(cost_matrix.tolist())
    logging.debug(f"Optimal pairings are {optimal_pairings}")

    # Calculate the sum of the intersection sizes of the paired biclusters
    dmax = 0
    for pairing in optimal_pairings:
        score = intersect_matrix[pairing[0], pairing[1]]
        logging.debug(f"For pairing {pairing}, the intersection has score {score}")
        dmax += score

    logging.debug(f"dmax is {dmax}")

    return dmax / union_size


def calc_jaccard_rec_rel(intersect_matrix, union_matrix):
    jaccard_matrix = np.divide(intersect_matrix, union_matrix)

    jaccard_dict = {}

    jaccard_dict['jaccard_relevance_idx'] = list(np.argmax(jaccard_matrix, axis=0))
    jaccard_dict['jaccard_recovery_idx'] = list(np.argmax(jaccard_matrix, axis=1))

    jaccard_dict['jaccard_relevance_scores'] = list(np.max(jaccard_matrix, axis=0))
    jaccard_dict['jaccard_recovery_scores'] = list(np.max(jaccard_matrix, axis=1))

    return jaccard_dict


def calc_clust_error_full(X_bin_a, B_bin_a, X_bin_b, B_bin_b):
    """
    Calculate the (bi)-clustering error, given by S_ce in Horta & Campello 2014
    and defined by Eqn 3 of Patrikainen & Meila 2006.

    First calculate the intersection size between each pair of biclusters, then
    use the Hungarian algorithm to find an optimal 1-1 pairing so that the
    sum of the intersections of the paired biclusters is maximised. This sum
    is then divided by the size of the union (counting overlaps) to normalise it.

    Args:
        X_bin_a: (N, K_a) binary np.ndarray where X_bin_a[i, k] indicates
            whether row i is included in bicluster k of biclustering A
        B_bin_a: (G, K_a) binary np.ndarray where B_bin_a[j, k] indicates
            whether column j is included in bicluster k of biclustering A
        X_bin_b: (N, K_b) binary np.ndarray where X_bin_b[i, k] indicates
            whether row i is included in bicluster k of biclustering B
        B_bin_b: (G, K_b) binary np.ndarray where B_bin_b[j, k] indicates
            whether column j is included in bicluster k of biclustering B

    Returns:
        float giving clustering_error

    """
    # Calculate the intersections between biclusters
    intersect_matrix, union_matrix = calc_overlaps(X_bin_a,
                                                   B_bin_a,
                                                   X_bin_b,
                                                   B_bin_b)

    # Normalise by the size of the combined union
    union_size = calculate_union_size_combined(X_bin_a,
                                               B_bin_a,
                                               X_bin_b,
                                               B_bin_b)

    return calc_clust_error(intersect_matrix, union_size)


def identify_sparse_biclusters(X_bin, B_bin):
    """Return a list of column indices of sparse factors, and another of the
    dense factors.

    A sparse factor is one with no more than 40% of the genes and no more than
    40% of the samples. All other factors are dense."""
    max_X_prop = 0.4
    max_B_prop = 0.4
    X_is_sparse = X_bin.mean(axis=0) < max_X_prop
    B_is_sparse = B_bin.mean(axis=0) < max_B_prop

    is_sparse = X_is_sparse * B_is_sparse

    sparse_indices = np.where(is_sparse)[0]
    dense_indices = np.where(is_sparse != True)[0]
    return sparse_indices, dense_indices


def calculate_restricted_clust_error(X_bin_a, B_bin_a, X_bin_b, B_bin_b, ind_a, ind_b):

    if len(ind_a) == 0:
        # Use -1 as a dummy 'na' value to make sure we don't include this in the score
        CE = -1
    else:
        if len(ind_b) == 0:
            CE = 0
        else:
            CE = calc_clust_error_full(X_bin_a[:, ind_a],
                                       B_bin_a[:, ind_a],
                                       X_bin_b[:, ind_b],
                                       B_bin_b[:, ind_b])

    return CE

def calc_clust_error_sparse_dense(X_bin_a, B_bin_a, X_bin_b, B_bin_b):
    """Calculate the (bi)-clustering error, split into CE for sparse clusters and
    for dense clusters. A is treated as truth.

    I.e. first split each set of biclusters into sparse and dense, then calculate
    CE for the sparse biclusters from A against the sparse biclusters from B, then
    calculate CE for the dense biclusters from A against the dense biclusters from B.

    If A contains no sparse biclusters, the (sparse CE) score is -1.
    Else, if A contains sparse biclusters but B does not, then the (sparse CE) score is 0.
    Similarly for dense biclusters.
    """
    ind_sparse_a, ind_dense_a = identify_sparse_biclusters(X_bin_a, B_bin_a)
    ind_sparse_b, ind_dense_b = identify_sparse_biclusters(X_bin_b, B_bin_b)

    print(ind_sparse_a)
    print(X_bin_a[:, ind_sparse_a].shape)
    print(B_bin_a[:, ind_sparse_a].shape)
    print(X_bin_a[:, ind_sparse_a])

    sparse_CE = calculate_restricted_clust_error(X_bin_a,
                                                 B_bin_a,
                                                 X_bin_b,
                                                 B_bin_b,
                                                 ind_sparse_a,
                                                 ind_sparse_b)

    dense_CE = calculate_restricted_clust_error(X_bin_a,
                                                B_bin_a,
                                                X_bin_b,
                                                B_bin_b,
                                                ind_dense_a,
                                                ind_dense_b)

    return sparse_CE, dense_CE


def calc_recon_error_normalised(X, B, Y_true):
    Y_recon = np.matmul(X, B.T)
    recon_error = np.linalg.norm(Y_true - Y_recon, ord='fro')
    normaliser = np.linalg.norm(Y_recon, ord='fro') + np.linalg.norm(Y_true, ord='fro')
    return recon_error / normaliser


def calc_recon_error(X, B, Y_true):
    Y_recon = np.matmul(X, B.T)
    return np.linalg.norm(Y_true - Y_recon, ord='fro')


def calc_factor_differences(X_a, B_a, X_b, B_b):
    _N, K_a = X_a.shape
    _N, K_b = X_b.shape

    differences = np.zeros((K_a, K_b))
    normalised_diffs = np.zeros((K_a, K_b))

    for k_a in range(K_a):
        factor_a = np.outer(X_a[:, k_a], B_a[:, k_a])
        norm_a = np.linalg.norm(factor_a, ord='fro')

        for k_b in range(K_b):
            factor_b = np.outer(X_b[:, k_b], B_b[:, k_b])
            norm_b = np.linalg.norm(factor_b, ord='fro')

            diff = np.linalg.norm(factor_a - factor_b, ord='fro')
            differences[k_a, k_b] = diff
            normalised_diffs[k_a, k_b] = diff / (norm_a + norm_b)

    return differences, normalised_diffs


def calc_factor_differences_rec_rel(X_a, B_a, X_b, B_b):
    diff_matrix, diff_norm_matrix = calc_factor_differences(X_a, B_a, X_b, B_b)

    diff_dict = {}

    diff_dict['factor_diff_relevance_idx'] = list(np.argmax(diff_matrix, axis=0))
    diff_dict['factor_diff_recovery_idx'] = list(np.argmax(diff_matrix, axis=1))

    diff_dict['factor_diff_relevance_scores'] = list(np.max(diff_matrix, axis=0))
    diff_dict['factor_diff_recovery_scores'] = list(np.max(diff_matrix, axis=1))

    diff_dict['factor_diff_norm_relevance_idx'] = list(np.argmax(diff_norm_matrix, axis=0))
    diff_dict['factor_diff_norm_recovery_idx'] = list(np.argmax(diff_norm_matrix, axis=1))

    diff_dict['factor_diff_norm_relevance_scores'] = list(np.max(diff_norm_matrix, axis=0))
    diff_dict['factor_diff_norm_recovery_scores'] = list(np.max(diff_norm_matrix, axis=1))

    return diff_dict


def find_keys_from_exact_filename(X_file, method):
    try:
        match = re.match('.*/(run_[\w_\.-]*)/X(.*).txt', X_file)
        run_id = match[1]
        extra = match[2]
    except TypeError as e:
        logging.error(f"Tried to find run_id in filename {X_file} but failed")
        run_id = ""
        extra = ""

    extended_method = f"{method}{extra}"
    error_df_keys = {'method_ext': extended_method,
                     'method': method,
                     'processing': extra,
                     'run_id': run_id}
    return error_df_keys


def find_keys_from_binary_filename(X_binary_file, method):
    try:
        match = re.match('.*/(run_[\w_\.-]*)/X(.*)_binary.txt', X_binary_file)
        run_id = match[1]
        extra = match[2]
    except TypeError as e:
        logging.error(f"Tried to find run_id in filename {X_binary_file} but failed")
        run_id = ""
        extra = ""

    extended_method = f"{method}{extra}"
    error_df_keys = {'method_ext': extended_method,
                     'method': method,
                     'processing': extra,
                     'run_id': run_id}
    return error_df_keys


def construct_error_dict_exact_threshold(run_folder, X_scaled, B_scaled, X_true, B_true, Y_true, error_dict, threshold, do_full=False):
    K, X_hat, B_hat = utils.scaled_to_thresholded(X_scaled, B_scaled, threshold)

    error_dict['recovered_K'] = K

    if K == 0:
        logging.warning(f"Skipping run {run_folder} threshold {threshold} as K is 0")
    else:
        error = calc_recon_error(X_hat, B_hat, Y_true)
        error_dict['recon_error'] = error
        error_normalised = calc_recon_error_normalised(X_hat, B_hat, Y_true)
        error_dict['recon_error_normalised'] = error_normalised

        if do_full:
            diff_dict = calc_factor_differences_rec_rel(X_true, B_true, X_hat, B_hat)
            error_dict.update(diff_dict)

    return error_dict


def construct_error_dict_exact_threshold_real(run_folder, X_scaled, B_scaled, Y_true, error_dict, threshold, do_full=False):
    K, X_hat, B_hat = utils.scaled_to_thresholded(X_scaled, B_scaled, threshold)

    error_dict['recovered_K'] = K

    if K == 0:
        logging.warning(f"Skipping run {run_folder} threshold {threshold} as K is 0")
    else:
        method = run_folder.split("/")[1]
        if method in RECON_METHODS:
            error = calc_recon_error(X_hat, B_hat, Y_true)
            error_dict['recon_error'] = error
            error_normalised = calc_recon_error_normalised(X_hat, B_hat, Y_true)
            error_dict['recon_error_normalised'] = error_normalised

        X_bin = utils.binarise_matrix(X_hat)
        B_bin = utils.binarise_matrix(B_hat)
        self_intersect, self_union = calc_overlaps(X_bin, B_bin, X_bin, B_bin)
        self_jaccard = (self_intersect / self_union)
        np.fill_diagonal(self_jaccard, 0)
        error_dict['redundancy_mean'] = self_jaccard.mean()
        error_dict['redundancy_average_max'] = self_jaccard.max(axis=0).mean()
        error_dict['redundancy_max'] = self_jaccard.max()

    return error_dict


def compare_exact_thresholds_real(dataset, methods, thresholds=[0, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1]):
    dataset_folder = os.path.join("data", dataset)
    Y_true = utils.read_np(dataset_folder, "Y.txt")

    error_df = pd.DataFrame.from_dict({})

    for method in methods:
        results_folder = os.path.join("results", method, dataset)
        logging.info(f"Looking for results from method {method} "\
                     f"within folder {results_folder}")
        paths = glob.glob(os.path.join(results_folder, "run_*/X.txt"))

        logging.info(f"Found results {paths}")
        for path in paths:
            logging.info(f"Looking at folder: {path}")
            error_df_keys = find_keys_from_exact_filename(path, method)

            run_folder = os.path.dirname(path)
            K, X_scaled, B_scaled = utils.read_result_scaled(run_folder)
            for threshold in thresholds:
                # For Plaid, ignore any thresholds that aren't 0
                if method == 'Plaid' and threshold != 0:
                    continue

                threshold_str = f"_thresh_{np.format_float_scientific(threshold, trim='-', exp_digits=1)}"
                error_df_keys['processing'] = threshold_str
                error_df_keys['method_ext'] = f"{method}{threshold_str}"
                try:
                    error_dict = construct_error_dict_exact_threshold_real(run_folder,
                                                                           X_scaled,
                                                                           B_scaled,
                                                                           Y_true,
                                                                           error_df_keys.copy(),
                                                                           threshold)
                    error_df = error_df.append(error_dict, ignore_index=True)
                except FileNotFoundError as e:
                    logging.error(f"Failed to find required files for thresholding in folder: {path}")
                    raise e

    logging.info(f"Full table:\n{error_df}")
    error_df['dataset'] = dataset
    return error_df


def construct_error_dict_binary_threshold(run_folder, X_scaled, B_scaled, X_true, B_true, error_dict, threshold, do_full=False):
    K, X_hat, B_hat = utils.scaled_to_thresholded_binary(X_scaled, B_scaled, threshold)
    logging.debug(threshold)
    logging.debug(X_hat[:10,:10])

    error_dict['recovered_K'] = K
    if K == 0:
        logging.warning(f"Skipping run {run_folder} threshold {threshold} as K is 0")
        clust_error = 0
    else:
        intersect_matrix, union_matrix = calc_overlaps(X_true, B_true, X_hat, B_hat)
        union_size = calculate_union_size_combined(X_true, B_true, X_hat, B_hat)

        clust_error = calc_clust_error(intersect_matrix, union_size)
        sparse_CE, dense_CE = calc_clust_error_sparse_dense(X_true, B_true, X_hat, B_hat)
        error_dict['sparse_clust_err'] = sparse_CE
        error_dict['dense_clust_err'] = dense_CE

        if do_full:
            jaccard_dict = calc_jaccard_rec_rel(intersect_matrix, union_matrix)
            error_dict.update(jaccard_dict)

            self_intersect, self_union = calc_overlaps(X_hat, B_hat, X_hat, B_hat)
            self_jaccard = (self_intersect / self_union)
            np.fill_diagonal(self_jaccard, 0)
            error_dict['redundancy_mean'] = self_jaccard.mean()
            error_dict['redundancy_average_max'] = self_jaccard.max(axis=0).mean()
            error_dict['redundancy_max'] = self_jaccard.max()
    error_dict['clust_err'] = clust_error
    return error_dict


def compare_exact_thresholds(dataset, methods, thresholds=[0, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1], do_full=False):
    dataset_folder = os.path.join("data", dataset)
    Y_true = utils.read_np(dataset_folder, "Y.txt")

    X_true = utils.read_np(dataset_folder, "X.txt")
    B_true = utils.read_np(dataset_folder, "B.txt")
    baseline = calc_recon_error(X_true, B_true, Y_true)
    baseline_norm = calc_recon_error_normalised(X_true, B_true, Y_true)
    logging.info(f"Baseline error is {baseline}, normalised version is {baseline_norm}")

    error_df = pd.DataFrame.from_dict({'method': ['baseline_XB_true'],
                                       'run_id': [1],
                                       'recon_error': [baseline],
                                       'recon_error_normalised': [baseline_norm]})

    for method in methods:
        if method != 'Plaid':
            results_folder = os.path.join("results", method, dataset)
            logging.info(f"Looking for results from method {method} "\
                         f"within folder {results_folder}")
            paths = glob.glob(os.path.join(results_folder, "run_*/X.txt"))

            logging.info(f"Found results {paths}")
            for path in paths:
                logging.info(f"Looking at folder: {path}")
                error_df_keys = find_keys_from_exact_filename(path, method)

                run_folder = os.path.dirname(path)
                K, X_scaled, B_scaled = utils.read_result_scaled(run_folder)
                for threshold in thresholds:
                    threshold_str = f"_thresh_{np.format_float_scientific(threshold, trim='-', exp_digits=1)}"
                    error_df_keys['processing'] = threshold_str
                    error_df_keys['method_ext'] = f"{method}{threshold_str}"
                    try:
                        error_dict = construct_error_dict_exact_threshold(run_folder,
                                                                          X_scaled,
                                                                          B_scaled,
                                                                          X_true,
                                                                          B_true,
                                                                          Y_true,
                                                                          error_df_keys.copy(),
                                                                          threshold,
                                                                          do_full=do_full)
                        error_df = error_df.append(error_dict, ignore_index=True)
                    except FileNotFoundError as e:
                        logging.error(f"Failed to find required files for thresholding in folder: {path}")
                        raise e

    logging.info(f"Full table:\n{error_df}")
    error_df['dataset'] = dataset

    return error_df


def compare_binary_thresholds(dataset, methods, thresholds=[0, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1], do_full=True):
    dataset_folder = os.path.join("data", dataset)

    X_true = utils.read_np(dataset_folder, "X_binary.txt")
    B_true = utils.read_np(dataset_folder, "B_binary.txt")

    error_df = pd.DataFrame.from_dict({'method': [],
                                       'run_id': []})

    for method in methods:
        # Plaid can't do thresholding, so just use 0
        if method == 'Plaid':
            valid_thresholds = [0]
        else:
            valid_thresholds = thresholds

        results_folder = os.path.join("results", method, dataset)
        logging.info(f"Looking for results from method {method} "\
                     f"within folder {results_folder}")
        paths = glob.glob(os.path.join(results_folder, "run_*/X.txt"))
        logging.info(f"Found results {paths}")

        for path in paths:
            logging.info(f"Looking at folder: {path}")
            error_df_keys = find_keys_from_exact_filename(path, method)
            run_folder = os.path.dirname(path)
            K, X_scaled, B_scaled = utils.read_result_scaled(run_folder)
            logging.debug(path)
            logging.debug(X_scaled[:10,:10])

            for threshold in valid_thresholds:
                threshold_str = f"_thresh_{np.format_float_scientific(threshold, trim='-', exp_digits=1)}"
                error_df_keys['processing'] = threshold_str
                error_df_keys['method_ext'] = f"{method}{threshold_str}"
                try:
                    error_dict = construct_error_dict_binary_threshold(run_folder,
                                                                       X_scaled,
                                                                       B_scaled,
                                                                       X_true,
                                                                       B_true,
                                                                       error_df_keys.copy(),
                                                                       threshold,
                                                                       do_full=do_full)
                    error_df = error_df.append(error_dict, ignore_index=True)
                except FileNotFoundError as e:
                    logging.error(f"Failed to find required files for thresholding in folder: {path}")
                    raise e

    logging.info(f"Full table:\n{error_df}")
    error_df['dataset'] = dataset

    return error_df


def construct_factor_info_dict(folder, processing):
    B_bin = utils.read_np(folder, f"B{processing}_binary.txt")
    X_bin = utils.read_np(folder, f"X{processing}_binary.txt")

    found_AZ = True
    # Not all methods return A, Z (only tensor methods do)
    #   so be prepared for FileNotFound
    try:
        A_bin = utils.read_np(folder, f"A{processing}_binary.txt")
        Z_bin = utils.read_np(folder, f"Z{processing}_binary.txt")
    except FileNotFoundError as e:
        logging.info(f"File not found error when looking for A, Z binary: {e}")
        found_AZ = False

    found_XB_raw = True
    # Some methods (e.g. Plaid) don't return non-binary ('raw') versions of
    #   X, B so be prepared for FileNotFound
    try:
        X = utils.read_np(folder, f"X{processing}.txt")
        B = utils.read_np(folder, f"B{processing}.txt")
    except FileNotFoundError as e:
        logging.info(f"File not found error when looking for X, B raw: {e}")
        found_XB_raw = False

    _N, K = X_bin.shape

    intersect_matrix, union_matrix = calc_overlaps(X_bin, B_bin, X_bin, B_bin)
    jaccard_matrix = (intersect_matrix / union_matrix)
    np.fill_diagonal(jaccard_matrix, 0)

    factors_dict = {}
    for k in range(K):
        num_samples = X_bin[:, k].sum()
        num_genes = B_bin[:, k].sum()
        num_total = num_samples * num_genes

        num_individuals = num_samples
        num_tissues = 1
        factor_norm = np.nan

        if found_AZ:
            num_individuals = A_bin[:, k].sum()
            num_tissues = Z_bin[:, k].sum()

        if found_XB_raw:
            factor = np.outer(X[:, k], B[:, k])
            factor_norm = np.linalg.norm(factor, ord='fro')

        redundancy_mean = jaccard_matrix[:, k].mean()
        redundancy_max = jaccard_matrix[:, k].max()

        factors_dict[k] = {'num_individuals': num_individuals,
                           'num_genes': num_genes,
                           'num_samples': num_samples,
                           'num_tissues': num_tissues,
                           'num_total': num_total,
                           'redundancy_mean': redundancy_mean,
                           'redundancy_max': redundancy_max,
                           'factor_norm': factor_norm}
    return factors_dict


def collect_factor_information(folder, processing=""):
    K = utils.read_int_from_file(os.path.join(folder, f"K{processing}.txt"))
    if K == 0:
        factors_dict = {}
    else:
        factors_dict = construct_factor_info_dict(folder, processing)

    factors_df = pd.DataFrame.from_dict(factors_dict, orient='index')
    return factors_df


def construct_factor_info_threshold(run_folder, threshold):
    K, X_bin, B_bin = utils.read_result_threshold_binary(run_folder, threshold)

    factors_dict = {}
    if K == 0:
        logging.warning(f"Skipping run {run_folder} threshold {threshold} as K is 0")
    else:
        for k in range(K):
            num_samples = X_bin[:, k].sum()
            num_genes = B_bin[:, k].sum()
            num_total = num_samples * num_genes

            factors_dict[k] = {'num_genes': num_genes,
                               'num_samples': num_samples,
                               'num_total': num_total}

    factors_df = pd.DataFrame.from_dict(factors_dict, orient='index')
    return factors_df


def calculate_similarity_for_dataset(dataset, method_run_id_patterns, threshold=None):
    similarities = {}
    for name, (method, run_id_pattern) in method_run_id_patterns.items():
        if threshold is None:
            if method == 'Plaid':
                method_threshold = 0
            else:
                method_threshold = 1e-2
        else:
            method_threshold = threshold

        print(name, method, run_id_pattern, method_threshold)

        similarity = calculate_similarity_for_dataset_method(dataset,
                                                             method,
                                                             run_id_pattern,
                                                             method_threshold)
        print(similarity)
        similarities[name] = {'similarity' : similarity,
                              'method': method,
                              'run_id_pattern': run_id_pattern}

    return pd.DataFrame(similarities).T


def calculate_similarity_for_dataset_method(dataset, method, run_id_pattern, threshold):
    X_files = glob.glob(f"results/{method}/{dataset}/*/X.txt")
    print(len(X_files))
    run_folders = [X_file[:-5] for X_file in X_files
                   if re.match(run_id_pattern, X_file)]
    print(len(run_folders))

    similarity_matrix = calculate_similarity_between_all_runs(run_folders, threshold)
    print(similarity_matrix)
    np.fill_diagonal(similarity_matrix, 0)

    return similarity_matrix.mean()


def calculate_similarity_between_all_runs(run_folders, threshold):
    num_runs = len(run_folders)
    similarity_matrix = np.zeros((num_runs, num_runs))

    for i in range(num_runs):
        for j in range(i, num_runs):
            run_folder_a = run_folders[i]
            run_folder_b = run_folders[j]
            clust_err = calculate_similarity_between_runs(run_folder_a,
                                                          run_folder_b,
                                                          threshold)
            similarity_matrix[i, j] = clust_err
            similarity_matrix[j, i] = clust_err

    return similarity_matrix

def calculate_similarity_between_runs(run_folder_a, run_folder_b, threshold):
    _K_a, X_thr_a, B_thr_a = utils.read_result_threshold_binary(run_folder_a, threshold)
    _K_b, X_thr_b, B_thr_b = utils.read_result_threshold_binary(run_folder_b, threshold)

    if _K_a == 0 or _K_b == 0:
        clust_err = 0
    else:
        clust_err = calc_clust_error_full(X_thr_a, B_thr_a, X_thr_b, B_thr_b)

    return clust_err
