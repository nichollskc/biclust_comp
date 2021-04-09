import logging
import numpy as np
import pandas as pd
import random

from biclust_comp import logging_utils, utils

random.seed(42)

def sample_negbin(size, negbin_n=1, negbin_p=0.3):
    """
    Sample a negative binomial distribution with the given parameters, using
    the size given. Essentially just a wrapper around
    np.random.negative_binomial, with some debugging possible.

    Args:
        size: to be passed to np.random.negative_binomial (e.g. int or tuple)
        negbin_n: number of successes for negative binomial
        negbin_p: probability of success for negative binomial

    Returns:
        np.ndarray of given size
    """
    # Calculate theoretical mean and probability of drawing 0 with these parameters
    mean = (negbin_n * (1 - negbin_p))/negbin_p
    zero_prob = negbin_p ** negbin_n
    logging.debug(f"Mean of negbin is {mean}, probability of zero is {zero_prob}")

    return np.random.negative_binomial(p=negbin_p, n=negbin_n, size=size)


def sample_negbin_mean(size, mean, negbin_p=0.3):
    # Calculate the required negbin_n parameter to achieve this mean with this p
    negbin_n = mean * negbin_p/(1 - negbin_p)

    # Sample values from negbin with the calculated parameters
    values = np.random.negative_binomial(p=negbin_p, n=negbin_n, size=size)
    logging.info(f"Values from negbin are {values.flatten()[:10]}")
    return values


def generate_binary_biclusters_mixed(N, G, K, bic_props, square=False):
    """
    Generate K biclusters for a matrix of size (N, G). For each bicluster,
    choose a random value from bic_props to give proportion of genes to add to
    cluster (e.g. if 0.01 is chosen then 0.01*G genes will be added to bicluster)
    and similarly choose a random value from bic_props to give proportion of
    samples to add to cluster.

    A different proportion can be chosen for genes and for samples, so the
    biclusters will often *not* be square in shape. If 'square'=True, then
    the same proportion will be used for both dimensions.

    This function generates only binary biclusters i.e. corresponding to
    bicluster *membership* rather than generating values for these biclusters.

    Args:
        N: Number of samples
        G: Number of genes
        K: Number of factors/biclusters
        bic_props: list of floats in the range (0, 1] giving proportions of
            genes and of samples to use e.g. [0.1, 0.2, 0.5, 1]

    Returns:
        A: (N, K) np.ndarray giving membership of individuals to biclusters
        B: (G, K) np.ndarray giving membership of genes to biclusters
    """
    logging.info(f"Generating {K} biclusters for matrix of size {(N, G)}, "
                 f"with proportion of genes/samples included taken from list "
                 f"{bic_props}")
    A = np.zeros((N, K), dtype=int)
    B = np.zeros((G, K), dtype=int)

    for k in range(K):
        prop_genes = np.random.choice(bic_props)
        if square:
            prop_inds = prop_genes
        else:
            prop_inds = np.random.choice(bic_props)
        n_genes = int(np.ceil(prop_genes * G))
        n_inds = int(np.ceil(prop_inds * N))

        logging.debug(f"Bicluster {k} will have {n_genes} genes and {n_inds} "
                      f"individuals.")

        first_gene = np.random.randint(low=0, high=(G - n_genes + 1))
        first_ind = np.random.randint(low=0, high=(N - n_inds + 1))

        gene_idx = np.arange(first_gene, first_gene + n_genes)
        B[gene_idx, k] = 1

        ind_idx = np.arange(first_ind, first_ind + n_inds)
        A[ind_idx, k] = 1

    return A, B


def sample_tissue_members_uniform(T, K):
    """
    Generate a size (T, K) matrix where each row corresponds to a tissue,
    and each column corresponds to a factor. Each column will contain
    num_tissues ~ U[1, 2, ..., T] 1s and the rest are 0s. The 1s will be in
    a contiguous block e.g. (0, 1, 1, 0) and not e.g. (1, 0, 1, 0).

    Args:
        T: Number of tissues
        K: Number of factors to generate

    Returns:
       np.ndarray of size (T, K) with dtype int
    """
    Z = np.zeros((T, K), dtype=int)

    for k in range(K):
        n_tissues = np.random.randint(low=1, high=T)
        first_tissue = np.random.randint(low=0, high=(T - n_tissues + 1))

        idx = np.arange(first_tissue, first_tissue + n_tissues)
        logging.debug(f"Tissue indices chosen for bicluster {k}: {idx}")
        Z[idx, k] = 1
    return Z


def generate_biclusters(A_bin, B_bin, Z_bin,
                        sample_bic_values_fn, sample_bg_noise_fn,
                        gamma_shape=2, gamma_scale=600, negbin_p=0.3):
    """
    Given matrices defining bicluster membership of individuals (A_bin),
    genes (B_bin) and tissues (Z_bin), simulate values using
    sample_bic_values_fn and add background noise using
    sample_bg_noise_fn.

    Args:
        A_bin: bicluster membership of individuals - shape (N, K)
        B_bin: bicluster membership of genes - shape (G, K)
        Z_bin: bicluster membership of tissues - shape (T, K)
        sample_bic_values_fn:
        sample_bg_noise_fn:

    Returns:
       out_dict containing keys Y, Y_raw (Y before background noise added),
       A, A_bin, X, X_bin, Z, Z_bin, B, B_bin. The _bin version
       is binary, giving bicluster membership status.
    """
    logging.info(f"Constructing matrix X (individuals x tissues, factors)")
    X_bin = utils.combine_inds_tissues(A_bin, Z_bin)

    NxT, K = X_bin.shape
    G, K_ = B_bin.shape
    assert K == K_, f"A and B not compatible shapes to combine: A has shape"\
        " {A.shape}, B has shape {B.shape} - need same number of columns"

    mat_size = (NxT, G)
    Y = np.zeros(mat_size)
    B = np.zeros(B_bin.shape)

    means = np.random.gamma(gamma_shape, gamma_scale, size=K)
    logging.info(f"Calculated means for each bicluster: {means}")

    for k in range(K):
        samp_idx = np.where(X_bin[:,k])[0]
        gene_idx = np.where(B_bin[:,k])[0]
        values = sample_bic_values_fn((len(samp_idx), len(gene_idx)),
                                      means[k])
        logging.debug(f"Some generated values for bicluster {k}: {values[:10,:10]}")

        Y[np.ix_(samp_idx, gene_idx)] += values
        B[gene_idx, k] = means[k]

    out_dict = {}
    # Write out matrices - noting that since for each bicluster the mean is the
    #   same across the whole block we only need to have B as non-binary
    #   in order to be able to recover the mean of each bicluster
    out_dict['Y_raw'] = Y.copy()
    out_dict['B'] = B
    out_dict['B_binary'] = B_bin
    out_dict['X'] = X_bin
    out_dict['X_binary'] = X_bin
    out_dict['A'] = A_bin
    out_dict['A_binary'] = A_bin
    out_dict['Z'] = Z_bin
    out_dict['Z_binary'] = Z_bin

    logging.info(f"Adding background noise to matrix")
    background = sample_bg_noise_fn(mat_size)
    Y += background

    out_dict['Y'] = Y
    return out_dict


def simulate_mixed(output_file_dict, N, G, T, K, bic_props,
                   sample_bic_values_fn, sample_bg_noise_fn,
                   seed=42, square=False):
    """
    Simulate a matrix of the given dimensions (NxT, G) with K biclusters and
    write the resulting matrix (and the 'true' X and B matrices that will
    generate it) to files given by output_file_dict. The biclusters will have
    constant mean, with noise added by sample_bic_values_fn and sample_bg_noise_fn.
    The bicluster sizes will be guided by bic_props. Number of tissues for each
    bicluster will be chosen uniformly from [0,T].

    The rows of Y will be grouped by tissue, with individuals in the same order
    in each tissue.

    Args:
        output_file_dict: dictionary with keys Y, Y_raw, N, X_binary, B_binary,
            A_binary, Z_binary, X, B, A, Z and K
            and with values the corresponding filenames
            Files don't need to exist, but the directory should already exist
        N: Number of individuals for matrix
        G: Number of genes for matrix
        T: Number of tissues for matrix
        K: Number of biclusters to add
        bic_props: List of proportions to use to generate biclusters. Each
            bicluster will have number of genes chosen as one of the proportions
            in this list, and independently the number of individuals also chosen
            this way.
        seed: seed to use for random number generators
    """
    logging.info(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)

    logging.info(f"Filenames provided for output: {output_file_dict}")
    A, B = generate_binary_biclusters_mixed(N, G, K, bic_props, square=square)
    Z = sample_tissue_members_uniform(T, K)
    out_dict = generate_biclusters(A, B, Z,
                                   sample_bic_values_fn,
                                   sample_bg_noise_fn)
    logging.info(f"Generated biclusters, output obtained: {out_dict.keys()}")

    # Write values
    with open(output_file_dict['N'], 'w') as f:
        f.write(str(N))

    with open(output_file_dict['K'], 'w') as f:
        f.write(str(K))

    for key, array in out_dict.items():
        pd.DataFrame(array).to_csv(output_file_dict[key],
                                   sep='\t',
                                   header=None,
                                   index=None)
    logging.info(f"Done.")


def simulate_nonoise_constant_mixed(output_file_dict, N, G, T, K, bic_props,
                                    seed=42, square=False):
    sample_bic_values_fn = lambda size, mean: np.ones(size) * mean
    sample_bg_noise_fn = lambda size: np.zeros(size)

    simulate_mixed(output_file_dict, N, G, T, K, bic_props,
                   sample_bic_values_fn,
                   sample_bg_noise_fn,
                   seed)


def simulate_negbin_constant_mixed(output_file_dict, N, G, T, K, bic_props,
                                   seed=42, negbin_p=0.3, square=False):
    sample_bic_values_fn = lambda size, mean: sample_negbin_mean(size, mean, negbin_p)
    sample_bg_noise_fn = sample_negbin

    simulate_mixed(output_file_dict, N, G, T, K, bic_props,
                   sample_bic_values_fn,
                   sample_bg_noise_fn,
                   seed,
                   square=square)


def simulate_gaussian_constant_mixed(output_file_dict, N, G, T, K, bic_props,
                                     seed=42, gaussian_sigma=20, square=False):
    sample_bic_values_fn = lambda size, mean: abs(np.random.normal(loc=mean, scale=gaussian_sigma, size=size))
    sample_bg_noise_fn = lambda size: sample_negbin_mean(size, 0.5)

    simulate_mixed(output_file_dict, N, G, T, K, bic_props,
                   sample_bic_values_fn,
                   sample_bg_noise_fn,
                   seed)


def simulate_biclusters_mixed(output_file_dict, N, G, T, K, bic_props,
                              seed=42, noise="_negbin", noise_param="_0.3", square=False):
    if noise == "_negbin":
        negbin_p = 0.3
        if noise_param != "":
            negbin_p = float(noise_param[1:])
        simulate_negbin_constant_mixed(output_file_dict, N, G, T, K, bic_props, seed, negbin_p, square)
    elif noise == "_gaussian":
        sigma = 20
        if noise_param != "":
            sigma = float(noise_param[1:])
        simulate_gaussian_constant_mixed(output_file_dict, N, G, T, K, bic_props, seed, sigma, square)
    else:
        assert noise == "", "Only valid noise types are '_negbin', '_gaussian' and '' i.e. no noise"
        simulate_nonoise_constant_mixed(output_file_dict, N, G, T, K, bic_props, seed, square)

