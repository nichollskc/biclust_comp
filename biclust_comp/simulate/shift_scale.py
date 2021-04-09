import numpy as np

import biclust_comp.simulate.simple as sim

def sample_bic_values_shift_scale(size, base, shift, scale):
    N, G = size
    gene_shifts = np.random.exponential(shift, size=(G))
    if scale is not None:
        gene_scales = np.random.exponential(scale, size=(G)) + 1/2
        print(gene_scales.mean())
        print(gene_scales)
    else:
        gene_scales = np.ones(G)
    base_sample_values = np.random.exponential(base, size=(N))

    bicluster_values = np.zeros((N, G))
    for i in range(N):
        for j in range(G):
            bicluster_values[i][j] = (base_sample_values[i] * gene_scales[j] +
                                      gene_shifts[j])
    return bicluster_values

def simulate_shift_scale_mixed(output_file_dict, N, G, T, K, bic_props,
                               shift, scale,
                               seed=42, square=False):
    sample_bic_values_fn = lambda size, mean: sample_bic_values_shift_scale(size, 1, shift, scale)
    sample_bg_noise_fn = lambda size: np.zeros(size)

    sim.simulate_mixed(output_file_dict, N, G, T, K, bic_props,
                       sample_bic_values_fn,
                       sample_bg_noise_fn,
                       seed)

def _simulate_shift_scale_mixed_snakemake(output_file_dict, N, G, T, K, bic_props,
                                          shift_param, scale_param,
                                          seed=42, square=False):
    if scale_param == "":
        scale = None
    else:
        # Scale param from snakemake will have a '_' at the start, so only
        #   convert the string from first character onwards to a float
        scale = float(scale_param[1:])
    shift = float(shift_param)

    simulate_shift_scale_mixed(output_file_dict, N, G, T, K, bic_props,
                               shift, scale,
                               seed=42, square=False)
