import logging
import numpy as np
import pandas as pd
import re

from intermine.webservice import Service

from biclust_comp import utils


def construct_single_tissue_dataset(sample_info_file, counts_df_file, tissue):
    sample_info = pd.read_csv(sample_info_file, sep="\t")
    sample_info['genotype'] = sample_info['Factor Value[genotype]']
    sample_info['tissue'] = sample_info['Factor Value[organism part]']
    sample_info['ID'] = sample_info['Comment[ENA_SAMPLE]']

    counts_df = pd.read_csv(counts_df_file, sep="\t", index_col=0)

    # Restrict to tissue specified
    samples_from_tissue = sample_info[sample_info['tissue'] == tissue]
    num_individuals = len(samples_from_tissue)

    reduced_counts_df = counts_df.loc[samples_from_tissue.ID, :]

    return reduced_counts_df, num_individuals, samples_from_tissue


def construct_tensor(sample_info_file, counts_df_file):
    sample_info = pd.read_csv(sample_info_file, sep="\t")
    sample_info['genotype'] = sample_info['Factor Value[genotype]']
    sample_info['tissue'] = sample_info['Factor Value[organism part]']
    sample_info['ID'] = sample_info['Comment[ENA_SAMPLE]']

    counts_df = pd.read_csv(counts_df_file, sep="\t", index_col=0)

    # Restrict to tissues from this list, and to genotypes that have at least
    #   one sample from each of these tissues
    big_tissues = ['liver', 'lung', 'cardiac ventricle']
    samples_big_tissues = sample_info[sample_info['tissue'].isin(big_tissues)]
    genotype_has_all_3_tissues = (samples_big_tissues.groupby('genotype')['tissue'].nunique() == 3)
    genotypes_with_3_tissues = genotype_has_all_3_tissues[genotype_has_all_3_tissues].index
    samples_3_tissues = sample_info[sample_info.genotype.isin(genotypes_with_3_tissues) &
                                    (sample_info.tissue.isin(big_tissues))]

    # Take average of replicates from each genotype, tissue pair
    #   Order is important since tensor methods want samples from one tissue
    #   to be in a single block
    groups = samples_3_tissues.groupby(['tissue', 'genotype'])

    names = []
    counts = []
    tissues = []
    genotypes = []
    for name, grouped in groups:
        names.append(str(name))
        tissues.append(name[0])
        genotypes.append(name[1])
        ids = grouped['ID']
        logging.info(name)
        logging.info(ids)
        counts.append(np.mean(counts_df.loc[ids, :]))

    mean_counts_df = pd.DataFrame(counts, index=names)
    num_individuals = len(genotypes_with_3_tissues)

    sample_info = pd.DataFrame({'Factor Value[organism part]' : tissues,
                                'Factor Value[genotype]': genotypes})

    return mean_counts_df, num_individuals, sample_info


def calculate_pooled_variances(counts_df, group_by_pool):
    # To calculate pooled variance we need to take the weighted average of the sample variances
    #    where each sample variance is weighted by (n_i - 1)
    denominator = 0
    numerators = pd.DataFrame(index=counts_df.columns)

    for pool_name, pool_df in group_by_pool:
        logging.info(f"Calculating variance for group {pool_name}, with {len(pool_df)} samples")
        # np.var uses the denomiator (n - ddof), so to get sample variance use ddof=1
        numerators[pool_name] = np.var(counts_df.loc[pool_df.index, :], axis=0, ddof=1) * (len(pool_df) - 1)
        denominator += (len(pool_df) - 1)

    return numerators.sum(axis=1)/denominator


def sort_genes_by_variance(counts_array, variances, max_genes):
    # Indices, ordered by the sort_array (largest to smallest)
    sorted_indices = sorted(range(len(variances)),
                            key=lambda k: variances[k],
                            reverse=True)

    indices = sorted_indices[:max_genes]
    return counts_array[:, indices], indices

def restrict_to_knocked_out_genes(ensembl_gene_ids,
                                  ensembl_to_mgi_file,
                                  knocked_out_genes):
    ensembl_to_mgi = pd.read_csv(ensembl_to_mgi_file,
                                 sep="\t",
                                 index_col=0)

    # Build up list of knocked out gene IDs using MGI ID format
    knocked_out_genes_mgi = []
    service = Service("http://www.mousemine.org/mousemine/service")
    for knocked_out_gene in knocked_out_genes:
        query = service.new_query("ProteinCodingGene")
        query.add_view("symbol")
        query.add_constraint("symbol", "=", knocked_out_gene)

        # only care about first result, there should only be one!
        results = query.rows(size=1)
        try:
            result = next(results)
            knocked_out_genes_mgi.append(result['primaryIdentifier'])
        except StopIteration:
            print(f"Couldn't find gene {knocked_out_gene} in mousemine")

    gene_ids = []
    gene_indices = []

    # Check each gene ID one by one to see if it matches any MGI ID
    for idx, ensembl_id in enumerate(ensembl_gene_ids):
        unversioned_id = ensembl_id.split('.')[0]
        try:
            mgi_id = ensembl_to_mgi.loc[unversioned_id, 'MGI ID']
            if isinstance(mgi_id, str) and mgi_id.startswith('MGI'):
                pass
            else:
                raise KeyError
        except KeyError as e:
            print(f"Unable to translate ID {ensembl_id}")
            continue
        if mgi_id in knocked_out_genes_mgi:
            gene_ids.append(ensembl_id)
            gene_indices.append(idx)

    return gene_ids, gene_indices

def select_genes_from_knockout_pathways(ensembl_gene_ids,
                                        ensembl_to_mgi_file,
                                        genes_in_knockout_pathways_file,
                                        max_pathway_size):
    genes_in_knockout_pathways_df = pd.read_csv(genes_in_knockout_pathways_file,
                                                sep='\t',
                                                index_col=0)
    ensembl_to_mgi = pd.read_csv(ensembl_to_mgi_file,
                                 sep="\t",
                                 index_col=0)

    genes_in_small_knockout_pathways_ids = []
    genes_in_small_knockout_pathways_indices = []
    small_pathways = (genes_in_knockout_pathways_df.sum(axis=0) <= max_pathway_size)
    is_in_small_pathway = (genes_in_knockout_pathways_df.loc[:, small_pathways].sum(axis=1) > 0)

    for idx, ensembl_id in enumerate(ensembl_gene_ids):
        unversioned_id = ensembl_id.split('.')[0]
        try:
            mgi_id = ensembl_to_mgi.loc[unversioned_id, 'MGI ID']
            if isinstance(mgi_id, str) and mgi_id.startswith('MGI'):
                pass
            else:
                raise KeyError
        except KeyError as e:
            logging.info(f"Unable to translate ID {ensembl_id}")
            continue
        if mgi_id in genes_in_knockout_pathways_df.index and is_in_small_pathway[mgi_id]:
            genes_in_small_knockout_pathways_ids.append(ensembl_id)
            genes_in_small_knockout_pathways_indices.append(idx)

    return genes_in_small_knockout_pathways_indices, genes_in_small_knockout_pathways_ids

def find_genes_in_knockout_pathways(sample_info_file):
    sample_info = pd.read_csv(sample_info_file, sep="\t")
    sample_info['genotype'] = sample_info['Factor Value[genotype]']

    knocked_out_genes = []
    for genotype in sample_info.genotype.unique():
        match = re.match(r"(.*) knockout", genotype)
        if match:
            knocked_out_genes.append(match[1])

    service = Service("http://www.mousemine.org/mousemine/service")

    # Construct dataframe where columns are the knocked-out genes and each row is a pathway
    # Also construct a way to translate between pathway ID and name
    ko_genes_pathways = {}
    pathway_names_dict = {}
    for knocked_out_gene in knocked_out_genes:
        query = service.new_query("ProteinCodingGene")
        query.add_view("pathways.identifier", "pathways.name", "symbol")
        query.add_constraint("symbol", "=", knocked_out_gene)

        pathways = []
        for row in query.rows():
            pathway_names_dict[row["pathways.identifier"]] = row["pathways.name"]
            pathways.append(row["pathways.identifier"])
        ko_genes_pathways[knocked_out_gene] = pathways

    ko_genes_pathways_df = utils.transform_dict_to_count_df(ko_genes_pathways)

    # Construct dataframe where columns are pathways, rows are genes
    pathways_dict = {}
    for pathway in ko_genes_pathways_df.columns:
        pathway_query = service.new_query("Pathway")
        pathway_query.add_view(
            "genes.primaryIdentifier", "genes.symbol", "genes.name",
            "genes.sequenceOntologyTerm.name", "genes.chromosome.primaryIdentifier"
        )
        pathway_query.add_constraint("identifier", "=", pathway)
        pathways_dict[pathway] = [row["genes.primaryIdentifier"]
                                  for row in pathway_query.rows()]

    pathways_df = utils.transform_dict_to_count_df(pathways_dict).T
    return pathways_df, pathway_names_dict
