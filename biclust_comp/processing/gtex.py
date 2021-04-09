import re
import string

import pandas as pd
from sklearn.impute import KNNImputer


def get_longid(donor, cell, GTEx_ID):
    """Constructs a long ID {donor}_{cell}_{GTEx_ID}, ensuring that it uses
    only sensible characters."""
    assert '_' not in donor,\
        f"Donor should not contain the delimiter '_': {donor}"
    assert '_' not in donor, \
        f"Cell should not contain the delimiter '_': {cell}"
    assert '_' not in donor, \
        f"GTEx_ID should not contain the delimiter '_': {GTEx_ID}"
    full = f"{donor}_{cell}_{GTEx_ID}"

    valid_chars = f"-_.(){string.ascii_letters}{string.digits}"
    sanitised = ''.join(c for c in full if c in valid_chars)
    return sanitised


def extract_donor_id(sample_id):
    """Extract the donor ID from the GTEx sample ID.
    Returns 'GTEX-[donor ID]'

    GTEx ID should be of the form:
    GTEX-[donor ID]-[tissue site ID]-SM-[aliquot ID]

    In fact, there is one sample ID which starts K-562 instead of GTEX-[donor ID]
    This function will return an error if the sample ID is not of the expected form.
    """
    match = re.match("^(\w+-\w+)[\w-]+$", sample_id)
    return match[1]


def choose_samples_genes(outfile, gtex_tpm_file, gtex_samp_atts, n_tissues=10, max_genes=1000):
    print(f"Reading in sample names from {gtex_tpm_file}")
    with open(gtex_tpm_file, 'r') as f:
        f.readline()
        f.readline()
        sample_names = f.readline().split('\t')

    # Read in the information about samples
    print(f"Reading sample attributes from {gtex_samp_atts}")
    samples_df = pd.read_csv(gtex_samp_atts,
                             sep="\t",
                             index_col=0)
    # Drop any samples that aren't actually in the dataset
    samples_df = samples_df[samples_df.index.isin(sample_names)]

    # Construct a column for the donor ID
    samples_df['DONOR_ID'] = [extract_donor_id(sample_id)
                              for sample_id in samples_df.index]
    samples_df['LONGID'] = [get_longid(samples_df['DONOR_ID'][i],
                                       samples_df['SMTSD'][i],
                                       samples_df.index[i]) for i in range(samples_df.shape[0])]

    # Pick the 10 cell types which have the largest number of samples
    sorted_cell_types = samples_df.groupby('SMTSD').size().sort_values(ascending=False)
    cell_types = sorted_cell_types[:n_tissues].keys()
    print(f"Selected {n_tissues} cell types: {cell_types}")

    # Find the individuals who have samples from all 10 cell types
    # First make a matrix where rows are donors, columns are cell types, with entries
    #   the number of samples with that cell type and that donor
    donor_cell_matrix = samples_df.groupby(['DONOR_ID', 'SMTSD']).size().unstack()
    # Restrict to the cell types we are interested in
    donor_cell_matrix = donor_cell_matrix[cell_types]
    # Drop any rows with NaN (i.e. donors that are missing one of the cell types)
    #   and see which donors remain
    min_tissues = n_tissues - int(n_tissues / 3)
    donors = donor_cell_matrix.dropna(thresh=min_tissues).index
    print(f"Restricting to donors with at least {min_tissues} cell types "
          f"from this list of {n_tissues} cell types leaves {len(donors)} donors")

    # Restrict to the samples that belong to the cell types and donors we have chosen
    reduced_samples_df = samples_df[(samples_df['DONOR_ID'].isin(donors)) &
                                    (samples_df['SMTSD'].isin(cell_types))]
    # Remove duplicates so exactly one sample with each (cell type, donor) pair
    reduced_samples_df = reduced_samples_df.drop_duplicates(['SMTSD', 'DONOR_ID'])
    print(f"Total samples: {reduced_samples_df.shape[0]}")

    sample_indices = [sample_names.index(sample)
                      for sample in reduced_samples_df.index]

    tpm = pd.read_csv(gtex_tpm_file,
                      sep="\t",
                      skiprows=2,       # Skip the 2 header lines in GTEx file
                      usecols=[0] + sample_indices,
                      nrows=max_genes,
                      index_col=[0])
    print(f"Read in GTEx data for these samples for {max_genes} genes from {gtex_tpm_file}")

    reduced_tpm = tpm[(tpm == 0).sum(axis=1) < tpm.shape[1] / 10]
    print(f"{reduced_tpm.shape[0]} genes were non-zero in at least 10% of samples")

    map_donor_cell_to_longid = {(row.DONOR_ID, row.SMTSD): row.LONGID
                                for row in
                                reduced_samples_df[['SMTSD', 'LONGID', 'DONOR_ID']].itertuples()}
    reduced_tpm.columns = pd.MultiIndex.from_frame(
        reduced_samples_df.loc[reduced_tpm.columns, :][['DONOR_ID', 'SMTSD']])
    print(f"Added empty columns for missing tissues. "
          f"Total size is now {reduced_tpm.shape} (genes, samples).")

    # existing_pairs = list(reduced_samples_df[['SMTSD', 'DONOR_ID']].itertuples(index=False,
    #                                                                            name=None))
    # extra_column_names = [donor + "_" + cell_type + "_" for donor in donors for cell_type in
    #                       cell_types if (cell_type, donor) not in existing_pairs]
    # reduced_tpm = reduced_tpm.reindex(reduced_tpm.columns.tolist() + extra_column_names, axis=1)

    # We need to rearrange to have donors as rows and features as columns
    # The features should be all genes from all cell types, so we can use genes from non-missing
    # cell types to infer the missing cell types
    reduced_tpm = reduced_tpm.sort_index(axis=1)
    stacked = reduced_tpm.stack().transpose()

    print(f"Imputing zero values (including missing tissues)")
    imputer = KNNImputer(n_neighbors=10,
                         metric='nan_euclidean',
                         weights="uniform")

    imputed_stacked_df = pd.DataFrame(imputer.fit_transform(stacked),
                                      columns=stacked.columns,
                                      index=stacked.index)
    imputed_df = imputed_stacked_df.transpose().unstack()
    print(f"Imputation finished.")

    # For each sample (column), take the list of column names
    #   (corresponding to 'DONOR_ID' and 'SMTSD')
    #   See if there was a GTEx sample with this ID and cell type
    #   by searching in the dictionary map_donor_cell_to_longid we set up before
    #   If so, we use the longid constructed then, which will contain the GTEx_ID
    #   If not, there is no GTEx_ID so use the function get_longid with blank GTEx_ID
    #   to construct the longid
    imputed_df.columns = [map_donor_cell_to_longid.get(colnames,
                                                       get_longid(colnames[0],
                                                                  colnames[1],
                                                                  ""))
                          for colnames in imputed_df.columns]

    imputed_df.to_csv(outfile, sep="\t")


def process_gtex_subset(infile, outfile, gene_info, sample_info, N, G):
    """Filename should contain '_T<n_tissues>_' e.g.
    'data/GTEx_large_T10_tpm.txt'"""
    print(str(infile))
    print(type(infile))
    match = re.match(r'.*_T(\d+)_.*', infile)
    n_tissues = int(match.groups()[0])

    tpm_T = pd.read_csv(infile,
                        sep="\t",
                        usecols=list(range((N * n_tissues) + 1)),
                        nrows=G,
                        index_col=0)

    split_column_names = [col.split("_") for col in tpm_T.columns]
    tpm_T.columns = pd.MultiIndex.from_arrays(list(map(list, zip(*split_column_names))))

    tpm_T_sorted = tpm_T.sort_index(axis=1, level=[1, 0])

    # Save sample information to file
    sample_info_df = tpm_T_sorted.columns.to_frame(index=False,
                                                   name=['Donor', 'Tissue', 'SampleID'])
    sample_info_df.to_csv(sample_info,
                          sep="\t")
    # Save gene information to file
    tpm_T_sorted.index.to_frame(index=False).to_csv(gene_info)

    tpm_T_sorted.columns = ["_".join(col) for col in tpm_T_sorted.columns]

    tpm = tpm_T_sorted.transpose()

    tpm.to_csv(outfile,
               sep="\t",
               header=False,
               index=False)
