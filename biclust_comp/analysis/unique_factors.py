import numpy as np
import pandas as pd

import biclust_comp.analysis.accuracy as acc
import biclust_comp.analysis.accuracy_utils as acc_utils
import biclust_comp.analysis.enrichment as enrich

def get_results_with_num_unique(error_df_file):
    error_df = pd.read_csv(error_df_file)
    unique_factors_df = get_unique_factors_df(error_df)
    unique_pathways_df = enrich.get_number_unique_pathways(error_df_file)

    results_with_fac = error_df.merge(unique_factors_df, how='left', on='method_dataset_run_id')
    results_with_fac_path = results_with_fac.merge(unique_pathways_df, on='method_dataset_run_id', how='left')
    return results_with_fac_path

def get_unique_factors_df(error_df):
    unique_factors_dicts = []
    thresholds = np.arange(0.2*100, 1.05*100, 0.05*100) / 100
    for mdr in error_df[error_df['run_complete']]['method_dataset_run_id'].unique():
        unique_factors, jaccard = get_unique_biclusters_thresholds(mdr, thresholds)
        unique_factors_dicts.append(unique_factors)
    unique_factors_df = pd.DataFrame(unique_factors_dicts)
    return unique_factors_df

def get_unique_biclusters_thresholds(method_dataset_run_id, sim_thresholds):
    K, X, B = acc_utils.read_result_binary_best_threshold(f"results/{method_dataset_run_id}")
    intersect, union = acc.calc_overlaps(X, B, X, B)
    jaccard = intersect/union

    similarity_matrix = jaccard.copy()
    similarity_matrix_reduced = similarity_matrix
    np.fill_diagonal(similarity_matrix, 0)
    indices = np.array(range(K))

    unique_factors = {"method_dataset_run_id": method_dataset_run_id}
    for sim_threshold in sorted(list(sim_thresholds), reverse=True):
        indices = reduce_to_sim_threshold(similarity_matrix, indices, sim_threshold)
        unique_factors[f"unique_factors_{sim_threshold}"] = len(indices)

    return unique_factors, jaccard

def reduce_to_sim_threshold(similarity_matrix, current_indices, sim_threshold):
    indices = current_indices.copy()

    similarity_matrix_reduced = similarity_matrix[np.ix_(indices, indices)]
    while similarity_matrix_reduced.max() > sim_threshold:
        max_similarities = similarity_matrix_reduced.max(axis=1)
        is_max = np.where(max_similarities == similarity_matrix_reduced.max())[0]
        print("Maximum", is_max)
        keep = is_max[0]
        discard = is_max[1:]

        print(discard)
        indices = np.delete(indices, discard)
        similarity_matrix_reduced = similarity_matrix[np.ix_(indices, indices)]
        print(indices)

    return indices
