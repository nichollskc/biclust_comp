import logging
from pathlib import Path
import re

import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.stats.multitest as multitest
import sklearn.metrics

from intermine.webservice import Service

import biclust_comp.utils as utils

def plot_sample_enrichment_impc(X_file, max_factors=None, max_traits=None):
    sample_info = read_sample_info_IMPC("data/real/IMPC/sample_info.txt")

    X = utils.read_matrix_tsv(X_file)

    trait_dummies = pd.get_dummies(sample_info[['tissue', 'genotype']])
    return plot_enrichment(trait_dummies, X, max_factors, max_traits)


def plot_pathway_enrichment(B_file, gene_ensembl_ids_file,
                            full_pathways_file="analysis/IMPC/full_pathways.tsv",
                            max_factors=None, max_pathways=None):
    with open(gene_ensembl_ids_file) as f:
        gene_ensembl_ids = [line.strip() for line in f.readlines()]

    B = pd.read_csv(B_file, sep="\t")
    full_pathways_df = pd.read_csv(full_pathways_file, sep="\t")

    pathways_df = construct_pathways_df(gene_ensembl_ids, full_pathways_df)

    return plot_enrichment(pathways_df, B, max_factors, max_pathways)


def construct_ko_pathways_df():
    sample_info = read_sample_info_IMPC("data/real/IMPC/sample_info.txt")

    service = Service("http://www.mousemine.org/mousemine/service")
    knocked_out_genes = []
    for genotype in sample_info.genotype.unique():
        match = re.match(r"(.*) knockout", genotype)
        if match:
            knocked_out_genes.append(match[1])

    ko_genes_pathways = {}
    pathway_names_dict = {}

    for knocked_out_gene in knocked_out_genes:
        query = service.new_query("ProteinCodingGene")
        query.add_view("pathways.identifier", "pathways.name", "symbol")
        query.add_constraint("symbol", "=", knocked_out_gene)
        pathways = [f"{row['pathways.name']}_-_{row['pathways.identifier']}" for row in query.rows()]
        ko_genes_pathways[knocked_out_gene] = pathways

        for row in query.rows():
            pathway_names_dict[row["pathways.identifier"]] = row["pathways.name"]

    ko_genes_pathways_df = utils.transform_dict_to_count_df(ko_genes_pathways)
    return ko_genes_pathways_df, pathway_names_dict


def construct_full_pathways_df(pathways):
    service = Service("http://www.mousemine.org/mousemine/service")

    pathways_dict = {}
    for pathway in pathways:
        query = service.new_query("Pathway")
        query.add_view(
            "genes.primaryIdentifier", "genes.symbol", "genes.name",
            "genes.sequenceOntologyTerm.name", "genes.chromosome.primaryIdentifier"
        )
        query.add_constraint("identifier", "=", pathway)
        pathways_dict[pathway] = [row["genes.primaryIdentifier"]
                                  for row in query.rows()]

    pathways_df = utils.transform_dict_to_count_df(pathways_dict).T
    return pathways_df


def construct_pathways_df(gene_ensembl_ids, full_pathways_df,
                          ensembl_to_mgi_file="analysis/mart_export.txt"):
    ensembl_to_mgi = pd.read_csv(ensembl_to_mgi_file,
                                 sep="\t",
                                 index_col=0)
    pathways_df = pd.DataFrame(index=gene_ensembl_ids,
                               columns=full_pathways_df.columns,
                               dtype=int,
                               data=0)

    for ensembl_id in gene_ensembl_ids:
        unversioned_id = ensembl_id.split('.')[0]
        try:
            mgi_id = ensembl_to_mgi.loc[unversioned_id, 'MGI ID']
            if isinstance(mgi_id, str) and mgi_id.startswith('MGI'):
                pass
            else:
                raise KeyError
        except KeyError as e:
            print(f"Unable to translate ID {ensembl_id}")
        try:
            pathways_df.loc[ensembl_id, :] = full_pathways_df.loc[mgi_id, :]
        except KeyError as e:
            print(f"MGI ID not found in pathways matrix {mgi_id}")

    return pathways_df


def plot_enrichment(trait_df, factor_df, max_factors, max_traits):
    f1_scores, intersections, _fisher_pvals = calculate_trait_enrichment(factor_df, trait_df)
    if max_factors:
        num_factors = min(factor_df.shape[1], max_factors)
    else:
        num_factors = factor_df.shape[1]

    if max_traits:
        num_traits = min(trait_df.shape[1], max_traits)
    else:
        num_traits = trait_df.shape[1]

    # Sort the columns and rows by maximum f1 score, so that the factors with
    #     best enrichment will be left-most in the chart, and traits with best
    #     enrichment will be highest in the chart
    ordered_columns = sorted(list(f1_scores.columns),
                             key=lambda k: f1_scores.iloc[:, k].max(),
                             reverse=True)
    ordered_rows = sorted(list(f1_scores.index),
                          key=lambda row: f1_scores.loc[row, :].max(),
                          reverse=True)

    intersections.loc['total', :] = (factor_df != 0).sum()
    f1_scores.loc['total', :] = 0
    ordered_rows.insert(0, 'total')

    ordered_intersections = intersections.loc[ordered_rows, ordered_columns]
    ordered_intersections.insert(0, 'total', trait_df.sum())
    ordered_f1_scores = f1_scores.loc[ordered_rows, ordered_columns]
    ordered_f1_scores.insert(0, 'total', 0)

    fig, ax = plt.subplots(figsize=(num_factors * 0.7 + 3,
                                    num_traits * 0.7))

    # Colour each square by the F1 score
    plt.imshow(ordered_f1_scores.iloc[:num_traits + 1, :num_factors + 1],
               aspect='auto',
               cmap='Blues')

    # Sort out axis labels
    ax.set_yticks(np.arange(num_traits + 1))
    ax.set_xticks(np.arange(num_factors + 1))
    ax.set_yticklabels(ordered_f1_scores.index)
    ax.set_xticklabels(ordered_f1_scores.columns)

    # Add text that notes the number of samples in intersection of trait and factor
    threshold_black = 0.5
    for j in range(num_factors + 1):
        for i in range(num_traits + 1):
            value = ordered_intersections.iloc[i, j]
            opacity = ordered_f1_scores.iloc[i, j]
            if opacity < threshold_black and value != 0:
                color="black"
            else:
                color="white"
            text = ax.text(j, i, value,
                   ha="center", va="center", color=color)

    plt.axvline(x=0.5, color='black')
    plt.axhline(y=0.5, color='black')
    plt.colorbar()
    fig.tight_layout()
    plt.show()
    return ordered_f1_scores, ordered_intersections


def calculate_trait_enrichment(factor_df, trait_df):
    f1_scores = pd.DataFrame(index=trait_df.columns,
                            columns=factor_df.columns,
                            dtype=float)
    fisher_pvals = pd.DataFrame(index=trait_df.columns,
                            columns=factor_df.columns,
                            dtype=float)
    odds_ratios = pd.DataFrame(index=trait_df.columns,
                            columns=factor_df.columns,
                            dtype=float)
    intersections = pd.DataFrame(index=trait_df.columns,
                                 columns=factor_df.columns,
                                 dtype=int)

    for trait_name, trait_column in trait_df.items():
        for factor_index, factor_column in factor_df.items():
            total_from_trait = trait_column.sum()
            total_population = len(trait_column)
            factor_size = (factor_column != 0).sum()
            trait_non_zero = np.where(trait_column)[0]
            intersection_size = ((factor_column.iloc[trait_non_zero]) != 0).sum()
            trait_size = trait_column.sum()

            intersections.loc[trait_name, factor_index] = intersection_size
            f1_scores.loc[trait_name, factor_index] = sklearn.metrics.f1_score(trait_column,
                                                                               factor_column != 0)
            # sf is the 'survival' function i.e. 1 - cdf
            # So we are finding the probability that the intersection size is at least
            #   equal to the intersection size we have observed, under the assumption that this
            #   has Hypergeometric distribution with M=total_population, n=trait_size and N=factor_size
            #   where M is 'total number of objects in the bin', N is 'number of objects we pick'
            #   n is 'total number of objects which are successes' and
            #   m is 'number of objects we pick which are successes'
            fisher_pvals.loc[trait_name, factor_index] = ss.hypergeom.sf(intersection_size - 1,
                                                                         total_population,
                                                                         trait_size,
                                                                         factor_size)

            odds_in_factor = intersection_size / (factor_size - intersection_size)
            notfactor_nottrait = total_population - trait_size - factor_size + intersection_size
            odds_out_of_factor = (trait_size - intersection_size) / notfactor_nottrait
            odds_ratios.loc[trait_name, factor_index] = odds_in_factor / odds_out_of_factor

    _reject, corrected_fisher_pvals = utils.correct_multiple_testing(fisher_pvals)

    return f1_scores, intersections, corrected_fisher_pvals, odds_ratios


def summarise_enrichment(sort_measure_name, measures_dict, factor_df, trait_df):
    trait_enrichment_dicts = []
    sort_measure_df = measures_dict[sort_measure_name]

    for trait in sort_measure_df.index:
        best_factor = sort_measure_df.loc[trait, :].argmax()
        trait_enrichment_dict = {'trait': trait,
                                 'best factor (by F1 score)': best_factor,
                                 'factor size': (factor_df.loc[:, best_factor] != 0).sum(),
                                 'trait size': (trait_df.loc[:, trait] != 0).sum()}
        for measure, measure_df in measures_dict.items():
            trait_enrichment_dict[measure] = measure_df.loc[trait, best_factor]
        trait_enrichment_dicts.append(trait_enrichment_dict)

    return pd.DataFrame(trait_enrichment_dicts)


def read_sample_info_IMPC(filename, read_ID=False):
    sample_info = pd.read_csv(filename, sep="\t")
    sample_info['genotype'] = sample_info['Factor Value[genotype]']
    sample_info['tissue'] = sample_info['Factor Value[organism part]']
    if read_ID:
        sample_info['ID'] = sample_info['Comment[ENA_SAMPLE]']
    return sample_info


def summarise_pathways_summary_IMPC(folder, postprocessing='*'):
    logging.info(f"Looking in folder {folder} for files of the form 'pathways_summary{postprocessing}.tsv'")
    files = [str(filename) for filename in Path(folder).rglob(f"pathways_summary{postprocessing}.tsv")]
    logging.info(f"Found {len(files)} files")
    file_pattern = re.compile(r'analysis/IMPC/(\w+)/real/IMPC/([\w/]+)/(run_.+)/pathways_summary(.*).tsv')
    run_info_dicts = []

    for file in files:
        logging.info(f"Processing file {file}")
        match = re.match(file_pattern, str(file))
        if match:
            run_info = {'method': match[1],
                        'dataset': match[2],
                        'run_id': match[3],
                        'postprocessing': match[4]}
            try:
                pathways = pd.read_csv(str(file), sep="\t", header=0)
                # Mean (over factors) of log10 of the smallest p-value
                run_info['factors_pathways_mean_min_pval'] = np.log10(pathways['min_pval']).mean()
                for alpha_col in pathways.columns[pathways.columns.str.startswith('alpha')]:
                    # For each threshold, the mean (over factors) number of pathways significant at that threshold and
                    #   the proportion of factors that had at least one pathway significant at that threshold
                    run_info[f"factors_pathways_mean_{alpha_col}"] = pathways[alpha_col].mean()
                    run_info[f"factors_pathways_nz_{alpha_col}"] = (pathways[alpha_col] != 0).mean()
            except pd.errors.EmptyDataError as e:
                logging.warning(f"Empty file: {file}")
            except KeyError as e:
                logging.warning(f"Required columns not found: {file}")

            run_info_dicts.append(run_info)
        else:
            logging.warning(f"Failed to decode file name: {file}")

    return pd.DataFrame(run_info_dicts)


def summarise_traits_summary_IMPC(folder, postprocessing='*'):
    logging.info(f"Looking in folder {folder} for files of the form 'traits_summary{postprocessing}.tsv'")
    files = [str(filename) for filename in Path(folder).rglob(f"traits_summary{postprocessing}.tsv")]
    logging.info(f"Found {len(files)} files")
    file_pattern = re.compile(r'analysis/IMPC/(\w+)/real/IMPC/([\w/]+)/(run_.+)/traits_summary(.*).tsv')
    run_info_dicts = []

    for file in files:
        logging.info(f"Processing file {file}")
        match = re.match(file_pattern, str(file))
        if match:
            run_info = {'method': match[1],
                        'dataset': match[2],
                        'run_id': match[3],
                        'postprocessing': match[4]}
            try:
                traits = pd.read_csv(str(file), sep="\t", header=0)
                tissue_rows = traits['trait'].str.startswith('tissue')
                genotype_rows = traits['trait'].str.startswith('genotype')
                # Mean (over traits) of f1 score from best factor, mean (over traits) of log of Fisher exact p-value
                #   (again from best factor), min p-value (min over traits, of p-value from best factor), max (over traits)
                #   of f1 score from best factor
                run_info['traits_mean_f1_score'] = traits.loc[:, 'F1 score'].mean()
                run_info['traits_mean_log10_pval'] = np.log10(traits.loc[:, 'Fisher\'s exact test']).mean()
                run_info['traits_min_pval'] = traits.loc[:, 'Fisher\'s exact test'].min()
                run_info['traits_max_f1_score'] = traits.loc[:, 'F1 score'].max()
                # Same as above, but only for 'genotype traits'
                run_info['traits_genotype_mean_f1_score'] = traits.loc[genotype_rows, 'F1 score'].mean()
                run_info['traits_genotype_mean_log10_pval'] = np.log10(traits.loc[genotype_rows, 'Fisher\'s exact test']).mean()
                run_info['traits_genotype_min_pval'] = traits.loc[genotype_rows, 'Fisher\'s exact test'].min()
                run_info['traits_genotype_max_f1_score'] = traits.loc[genotype_rows, 'F1 score'].max()
                # Same as above, but only for 'tissue traits'
                run_info['traits_tissue_mean_f1_score'] = traits.loc[tissue_rows, 'F1 score'].mean()
                run_info['traits_tissue_mean_log10_pval'] = np.log10(traits.loc[tissue_rows, 'Fisher\'s exact test']).mean()
                run_info['traits_tissue_min_pval'] = traits.loc[tissue_rows, 'Fisher\'s exact test'].min()
                run_info['traits_tissue_max_f1_score'] = traits.loc[tissue_rows, 'F1 score'].max()

                # Proportion of traits which have a factor significant for them, with threshold 0.01 and 0.05 resp.
                run_info['traits_sig_traits 0.01'] = (traits.loc[:, 'Fisher\'s exact test'] < 0.01).sum() / len(traits)
                run_info['traits_sig_traits 0.05'] = (traits.loc[:, 'Fisher\'s exact test'] < 0.05).sum() / len(traits)
            except pd.errors.EmptyDataError as e:
                logging.warning(f"Empty file: {file}")
            except KeyError as e:
                logging.warning(f"Required columns not found: {file}")

            run_info_dicts.append(run_info)
        else:
            logging.warning(f"Failed to decode file name: {file}")

    return pd.DataFrame(run_info_dicts)


def summarise_traits_fisherpvals_IMPC(folder, postprocessing='*'):
    logging.info(f"Looking in folder {folder} for files of the form 'traits_fisherpvals{postprocessing}.tsv'")
    files = [str(filename) for filename in Path(folder).rglob(f"traits_fisherpvals{postprocessing}.tsv")]
    logging.info(f"Found {len(files)} files")
    file_pattern = re.compile(r'analysis/IMPC/(\w+)/real/IMPC/([\w/]+)/(run_.+)/traits_fisherpvals(.*).tsv')
    run_info_dicts = []

    for file in files:
        logging.info(f"Processing file {file}")
        match = re.match(file_pattern, str(file))
        if match:
            run_info = {'method': match[1],
                        'dataset': match[2],
                        'run_id': match[3],
                        'postprocessing': match[4]}
            try:
                traits_pvals = pd.read_csv(str(file), header=0, index_col=0, sep="\t")
                min_pvals_per_factor = traits_pvals.min(axis=0)
                # For each threshold, the proportion of factors that are enriched for at least one trait
                for threshold in [1, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]:
                    run_info[f"traits_factors_alpha {threshold}"] = (min_pvals_per_factor < threshold).mean()
            except pd.errors.EmptyDataError as e:
                logging.warning(f"Empty file: {file}")
            except KeyError as e:
                logging.warning(f"Required columns not found: {file}")

            run_info_dicts.append(run_info)
        else:
            logging.warning(f"Failed to decode file name: {file}")

    return pd.DataFrame(run_info_dicts)


def summarise_traits_f1scores_IMPC(folder, postprocessing='*'):
    logging.info(f"Looking in folder {folder} for files of the form 'traits_f1scores{postprocessing}.tsv'")
    files = [str(filename) for filename in Path(folder).rglob(f"traits_f1scores{postprocessing}.tsv")]
    logging.info(f"Found {len(files)} files")
    file_pattern = re.compile(r'analysis/IMPC/(\w+)/real/IMPC/([\w/]+)/(run_.+)/traits_f1scores(.*).tsv')
    run_info_dicts = []

    for file in files:
        logging.info(f"Processing file {file}")
        match = re.match(file_pattern, str(file))
        if match:
            run_info = {'method': match[1],
                        'dataset': match[2],
                        'run_id': match[3],
                        'postprocessing': match[4]}
            try:
                traits_f1scores = pd.read_csv(str(file), header=0, index_col=0, sep="\t")
                # Mean (over factors) of the best F1 score that factor attains (across all traits)
                run_info['traits_factors_mean_max_f1_score'] = traits_f1scores.max(axis=0).mean()
            except pd.errors.EmptyDataError as e:
                logging.warning(f"Empty file: {file}")
            except KeyError as e:
                logging.warning(f"Required columns not found: {file}")

            run_info_dicts.append(run_info)
        else:
            logging.warning(f"Failed to decode file name: {file}")

    return pd.DataFrame(run_info_dicts)


def summarise_ko_enrichment_summary_IMPC(folder, postprocessing='*'):
    logging.info(f"Looking in folder {folder} for files of the form 'ko_enrichment_summary{postprocessing}.tsv'")
    files = [str(filename) for filename in Path(folder).rglob(f"ko_enrichment_summary{postprocessing}.tsv")]
    logging.info(f"Found {len(files)} files")
    file_pattern = re.compile(r'analysis/IMPC/(\w+)/real/IMPC/([\w/]+)/(run_.+)/ko_enrichment_summary(.*).tsv')
    run_info_dicts = []

    for file in files:
        logging.info(f"Processing file {file}")
        match = re.match(file_pattern, str(file))
        if match:
            run_info = {'method': match[1],
                        'dataset': match[2],
                        'run_id': match[3],
                        'postprocessing': match[4]}
            try:
                ko_enrichment = pd.read_csv(str(file), sep="\t", header=0)
                # Mean (over traits - only knockout genes) of the best F1 score obtained by any factor on that trait,
                #   also minimum pvalue
                run_info['ko_traits_mean_f1_score'] = ko_enrichment['f1_score (trait)'].mean()
                run_info['ko_traits_mean_min_pval'] = np.log10(ko_enrichment['min_pval']).mean()
                # For the threshold 0.05, the mean of precision and recall, considering the set of pathways
                #   significantly enriched at that threshold as the set of predictions, and the set
                #   of pathways that contained the gene knocked out as successes
                run_info['ko_traits_mean_precision_0.05'] = (ko_enrichment['alpha 0.05'] / ko_enrichment['all_pathways alpha 0.05']).mean()
                run_info['ko_traits_mean_recall_0.05'] = (ko_enrichment['alpha 0.05'] / ko_enrichment['pathways']).mean()
                for alpha_col in ko_enrichment.columns[ko_enrichment.columns.str.startswith('alpha')]:
                    # Mean recall, as above but for different thresholds
                    run_info[f"ko_traits_mean_recall_{alpha_col}"] = (ko_enrichment[alpha_col] / ko_enrichment['pathways']).mean()
                    # Proportion of traits (only ko genotype traits) that had at least one relevant pathway
                    #   (i.e. one containing this knocked out gene) significant at this threshold
                    run_info[f"ko_traits_nz_{alpha_col}"] = (ko_enrichment[alpha_col] != 0).mean()
            except pd.errors.EmptyDataError as e:
                logging.warning(f"Empty file: {file}")
            except KeyError as e:
                logging.warning(f"Required columns not found: {file}")

            run_info_dicts.append(run_info)
        else:
            logging.warning(f"Failed to decode file name: {file}")

    return pd.DataFrame(run_info_dicts)


def summarise_factor_info_IMPC(folder, postprocessing='*'):
    logging.info(f"Looking in folder {folder} for files of the form 'factor_info{postprocessing}.tsv'")
    files = [str(filename) for filename in Path(folder).rglob(f"factor_info{postprocessing}.tsv")]
    logging.info(f"Found {len(files)} files")
    files = Path(folder).rglob(f"factor_info{postprocessing}.tsv")
    file_pattern = re.compile(r'analysis/IMPC/(\w+)/real/IMPC/([\w/]+)/(run_.+)/factor_info(.*).tsv')
    run_info_dicts = []

    for file in files:
        logging.info(f"Processing file {file}")
        match = re.match(file_pattern, str(file))
        if match:
            run_info = {'method': match[1],
                        'dataset': match[2],
                        'run_id': match[3],
                        'postprocessing': match[4]}
            try:
                factor_info = pd.read_csv(str(file), sep="\t", index_col=0, header=0)
                # Number of factors, mean number of genes and samples in factor,
                #   mean of genes*samples (over factors), which I'm calling number of cells
                run_info['recovered_K'] = factor_info.shape[0]
                run_info['mean_num_genes'] = factor_info['num_genes'].mean()
                run_info['mean_num_samples'] = factor_info['num_samples'].mean()
                run_info['mean_num_cells'] = (factor_info['num_samples'] * factor_info['num_genes']).mean()
                # Mean (over factors) of the maximum (over other factors) Jaccard similarity
                run_info['mean_redundancy_max'] = factor_info['redundancy_max'].mean()
                # Mean (over factors) of the mean (over other factors) Jaccard similarity
                run_info['mean_redundancy_mean'] = factor_info['redundancy_mean'].mean()
            except pd.errors.EmptyDataError as e:
                logging.warning(f"Empty file: {file}")
            except KeyError as e:
                logging.warning(f"Required columns not found: {file}")

            run_info_dicts.append(run_info)
        else:
            logging.warning(f"Failed to decode file name: {file}")

    return pd.DataFrame(run_info_dicts)

def get_number_unique_pathways_mdr(method_dataset_run_id, enrich_thresholds=[0.001, 0.01, 0.05]):
    if 'Plaid' in method_dataset_run_id:
        thresh = "0e+0"
    else:
        thresh = "1e-2"
    pathway_pvals = pd.read_csv(f"analysis/IMPC/{method_dataset_run_id}/pathways_fisherpvals_thresh_{thresh}.tsv",
                            sep='\t',
                            index_col=0)
    main_pathways = (pathway_pvals.values.argmin(axis=0))

    results = {'method_dataset_run_id': method_dataset_run_id}
    results["unique_best_pathways"] = len(set(main_pathways))
    for threshold in enrich_thresholds:
        results[f"pathways_{threshold}"] = sum(pathway_pvals.min(axis=1) < threshold)

    return results

def get_number_unique_pathways(error_df_file):
    error_df = pd.read_csv(error_df_file)

    pathways_dicts = []
    for mdr in error_df[error_df['run_complete']]['method_dataset_run_id'].unique():
        try:
            results = get_number_unique_pathways_mdr(mdr)
            pathways_dicts.append(results)
        except FileNotFoundError:
            logging.warn(f"Skipping mdr {mdr}")
            continue

    return pd.DataFrame(pathways_dicts)
