import logging
import sys

import pandas as pd

import biclust_comp.analysis.accuracy_utils as acc_utils
import biclust_comp.analysis.accuracy as acc
import biclust_comp.utils as utils

def calculate_similarity_between_runs_file(run_ids_file):
    run_ids = utils.read_list_from_file(run_ids_file)
    folders = [f"results/{run_id}" for run_id in run_ids]
    return calculate_similarity_between_runs(folders)

def calculate_similarity_between_runs(folders):
    # For each pair of run folders, we will calculate similarity between the two runs
    # Output a similarity matrix
    logging.info(f"Comparing {len(folders)} runs, first is {folders[0]}, last is {folders[-1]}")

    # Initialise matrix
    similarity = pd.DataFrame(-1, columns=folders, index=folders)

    for folder_A in folders:
        logging.info(f"Working on folder {folder_A}")
        try:
            K_A, X_A, B_A = acc_utils.read_result_binary_best_threshold(folder_A)
        except:
            logging.error(f"Unexpected error encountered reading in '{folder_A}': {sys.exc_info()[0]}")
            continue

        for folder_B in folders:
            logging.info(f"Comparing to folder {folder_B}")
            # If we haven't already calculated for this pair, do it now
            if similarity.loc[folder_A, folder_B] == -1:
                try:
                    K_B, X_B, B_B = acc_utils.read_result_binary_best_threshold(folder_B)
                except:
                    logging.error(f"Unexpected error encountered reading in '{folder_B}': {sys.exc_info()[0]}")
                    continue

                try:
                    clust_err = acc.calc_clust_error_full(X_A, B_A, X_B, B_B)
                    similarity.loc[folder_A, folder_B] = clust_err
                    similarity.loc[folder_B, folder_A] = clust_err
                except:
                    logging.error(f"Unexpected error encountered for folders '{folder_A}', '{folder_B}': {sys.exc_info()[0]}")

    return similarity
