import string

import numpy as np
import pandas as pd
import scipy.stats as ss

import biclust_comp.analysis.accuracy_utils as acc_utils
import biclust_comp.analysis.plots as plots

def format_fraction_failing(row):
    expected_runs = row['size']
    successful_runs = row['count']

    prefix = ""
    suffix = ""

    #if expected_runs != successful_runs:
    #    prefix += '\textit{'
    #    suffix += '}'
    if successful_runs == 0 and expected_runs > 0:
        prefix += '\textbf{'
        suffix += '}'

    return f"{prefix}{expected_runs - successful_runs} / {expected_runs}{suffix}"

def add_midrules_latex(latex, indices, line_to_add='\midrule'):
    lines = latex.splitlines()
    # We have to insert lines from largest index to smallest so we don't mess up indexing
    for index in sorted(indices, reverse=True):
        lines.insert(index, line_to_add)
    return '\n'.join(lines)

def output_latex_table_failure_counts(input_df,
                                      output_txt,
                                      name=None):
    error_df = pd.read_csv(input_df, index_col=None)
    failures = table_failure_counts(error_df,
                                    name=name)
    latex = failures.to_latex(escape=False)

    if name == 'Simulated':
        latex = add_midrules_latex(latex, [29], r'\bottomrule')
        latex = add_midrules_latex(latex, [6, 13, 19, 23])
    elif name == 'IMPC':
        latex = add_midrules_latex(latex, [11], r'\bottomrule')
        latex = add_midrules_latex(latex, [8])

    with open(output_txt, "w") as f:
        f.write(latex)


def count_failures_by_metric(error_df, metric):
    size_and_count = error_df.groupby(['method', 'seedless_dataset'])[metric].agg(['size', 'count']).unstack()
    size_and_count.fillna(0, inplace=True)
    size_and_count = size_and_count.astype(int)

    return size_and_count

def table_failure_counts(error_df, name=None):
    if name == 'Simulated':
        metric = 'clust_err'
    elif name == 'IMPC':
        metric = 'ko_traits_mean_f1_score'
    else:
        metric = 's'

    print(metric)
    size_and_count = count_failures_by_metric(error_df, metric)
    full_table = size_and_count.stack().agg(format_fraction_failing, axis=1).unstack().T

    full_table = full_table.reindex(columns=plots.METHODS[:9])

    if name == 'Simulated':
        full_table.index = full_table.index.map(acc_utils.DATASET_NAMES)
        full_table.index = full_table.index.str.replace("\n", "-")
        full_table = full_table.reindex(acc_utils.DATASET_ORDER)
    elif name == 'IMPC':
        full_table.index = full_table.index.map(acc_utils.IMPC_DATASET_NAMES)
        full_table = full_table.reindex(acc_utils.IMPC_DATASET_ORDER)

    method_totals = size_and_count.stack().groupby(['method']).sum().apply(format_fraction_failing, axis=1)
    method_totals.name = '\textbf{Total}'
    print(method_totals)
    print(count_failures_by_metric(error_df, 's').stack().groupby(['method']).sum().apply(format_fraction_failing, axis=1))
    full_table = full_table.append(method_totals)

    return full_table

def add_section_headers_latex(latex, section_headers):
    lines = latex.splitlines()
    # We have to insert lines from largest index to smallest so we don't mess up indexing
    #   Sorting a list of tuples will automatically sort by the first element of the tuple first
    sorted_headers = sorted(section_headers, reverse=True)
    for index, header in section_headers:
        lines.insert(index, r" & \textbf{" + f"{header}" + r"} & & & & & & & & & \\")
        lines.insert(index, '\midrule')
    return '\n'.join(lines)

def format_float_sig_figures(x):
    if np.isnan(x):
        return 'N/A'
    elif x > 10:
        return f'{float(f"{x:.1f}"):g}'
    return f'{float(f"{x:.3g}"):g}'

def format_results_row(values, normalised):
    """Format a row of results, with values to print and a matching array giving normalised values, so
    that bigger values in normalised indicate better scores. The best score (and any within 1%) gets
    the \bestscore label and any score within 10% gets the \runnerup label.

    All scores are formatted using format_float_sig_figures"""

    df = pd.DataFrame({'value': values,
                       'norm': normalised})
    max_norm = df['norm'].max()
    std_norm = df['norm'].std()
    median_norm = df['norm'].median()
    lq_norm = df['norm'].quantile(q=0.25)

    def format_value_by_distance_to_max(row):
        prefix = ""
        suffix = ""

        norm = row['norm']
        value = row['value']

        normalised_difference = abs(norm - max_norm)

        if abs(norm - max_norm) < abs(norm - lq_norm) / 10:
            prefix = r"\bestscore{"
            suffix = "}"
        elif abs(norm - max_norm) < abs(norm - lq_norm):
            prefix = r"\runnerup{"
            suffix = "}"

        to_sig_figs = format_float_sig_figures(value)
        formatted = f"{prefix}{to_sig_figs}{suffix}"
        return formatted

    strings = df.apply(format_value_by_distance_to_max, axis=1)
    return strings

def output_results_table(combined_error_sim_ws, combined_error_IMPC_ws, combined_error_sim_K_ws, output_txt):
    measures_df = summarise_results(combined_error_sim_ws, combined_error_IMPC_ws, combined_error_sim_K_ws)
    measures_dict = {}

    # Convert to strings, highlighting best scores
    for row_name, values in measures_df.iterrows():
        normalised = values
        if 'Robustness' in row_name:
            normalised = - abs(values)
        elif 'Time' in row_name:
            normalised = - np.log(values)
        elif 'NRE' in row_name:
            normalised = - values

        formatted = format_results_row(values, normalised)

        # N/A values in this row correspond to special cases - same K recovered for every value of K init
        #   so correlation coefficient can't be calculated
        if 'Recovery of $K' in row_name:
            formatted = formatted.replace('N/A', '*')
        # N/A values in this row occur when a method fails to complete any runs (Plaid)
        elif 'large-K400' in row_name:
            formatted = formatted.replace('N/A', '*')

        measures_dict[row_name] = formatted

    print(measures_dict)
    table = pd.DataFrame.from_dict(measures_dict, orient='index')

    # Add column with letter to label each row for easy reference from the caption
    table.index = table.index.rename('Measures')
    table['Letter'] = [f"({letter})" for letter in string.ascii_uppercase[:table.shape[0]]]
    table = table.set_index('Letter', append=True).reorder_levels(['Letter', 'Measures'])

    # Reorder columns to be in the standard order
    table = table.reindex(columns=plots.METHODS[:9])

    latex = table.to_latex(escape=False)
    latex = add_section_headers_latex(latex,
                                      [(5, 'Simulated datasets'),
                                       (9, 'Ease of use'),
                                       (17, 'Tensor IMPC datasets'),
                                       (23, 'Non-tensor IMPC datasets'),
                                       (29, 'Time')])

    with open(output_txt, "w") as f:
        f.write(latex)

    return measures_df

def safe_pearson_r(x, y):
    na_rows = np.logical_or(np.isnan(x), np.isnan(y))
    return ss.pearsonr(x[~na_rows], y[~na_rows])[0]

def summarise_results(combined_error_sim_ws, combined_error_IMPC_ws, combined_error_sim_K_ws):
    comb = pd.read_csv(f"{combined_error_sim_ws}analysis/accuracy/all_results_expected.csv")
    comb_thr = pd.read_csv(f"{combined_error_sim_ws}analysis/accuracy/thresholded_results.csv")
    comb_thr_theoretK = pd.read_csv(f"{combined_error_sim_ws}analysis/accuracy/restricted_results.csv")

    comb_K = pd.read_csv(f"{combined_error_sim_K_ws}analysis/accuracy/all_results_expected_K_SWEEP.csv")
    comb_K_thr = pd.read_csv(f"{combined_error_sim_K_ws}analysis/accuracy/thresholded_results_K_SWEEP.csv")
    comb_K_thr_theoretK = pd.read_csv(f"{combined_error_sim_K_ws}analysis/accuracy/restricted_results_K_SWEEP.csv")

    comb_IMPC = pd.read_csv(f"{combined_error_IMPC_ws}analysis/IMPC/all_results_expected.csv")
    comb_IMPC_thr = pd.read_csv(f"{combined_error_IMPC_ws}analysis/IMPC/thresholded_results.csv")
    comb_IMPC_thr_theoretK = pd.read_csv(f"{combined_error_IMPC_ws}analysis/IMPC/restricted_results.csv")

    similarity = pd.read_csv(f"{combined_error_IMPC_ws}analysis/IMPC/similarity_methods.csv")
    similarity_median = similarity[similarity['same_method'] & similarity['same_K_init']].groupby(['method_A'])['score'].median()

    all_measures_dicts = {}
    for method in comb.method.unique():
        print(method)
        if method == 'baseline_XB_true':
            continue
        measures = {}

        df = comb_thr_theoretK[comb_thr_theoretK['method'] == method]
        measures['Biclustering accuracy (CE)'] = df.clust_err.mean()
        measures['Reconstruction error (NRE)'] = df.recon_error_normalised.mean()
        if method in ['Plaid', 'FABIA', 'MultiCluster', 'BicMix-Q']:
            measures['Reconstruction error (NRE)'] = np.nan

        df = comb_thr[comb_thr['method'] == method]
        measures['Redundancy (MBR - thresholded)'] = df['redundancy_mean'].mean()

        df = comb[(comb['method'] == method) &
                  (comb['processing'] == '_thresh_0e+0')]
        measures['Redundancy (MBR raw)'] = df['redundancy_mean'].mean()

        df = comb_K_thr[comb_K_thr['method'] == method]
        measures['Robustness to $K_\text{init}$ (CE)'] = safe_pearson_r(df['K_init'], df['clust_err'])

        df = comb_K_thr[comb_K_thr['method'] == method]
        measures['Robustness to $K_\text{init}$ (Recovered K)'] = safe_pearson_r(df['K_init'], df['recovered_K'])

        df = comb_K_thr[(comb_K_thr['method'] == method) &
                        (comb_K_thr['K_init'] == 100)]
        measures['Recovery of $K_\text{true}$'] = safe_pearson_r(df['recovered_K'], df['K'])

        measures['Similarity between runs'] = similarity_median[method]

        df = comb_IMPC_thr_theoretK[(comb_IMPC_thr_theoretK['method'] == method) &
                                    (comb_IMPC_thr_theoretK['tensor'] == 'tensor')]
        measures['Tissue clustering (tensor)'] = df['traits_tissue_mean_f1_score'].mean()
        measures['Genotype clustering (tensor)'] = df['traits_genotype_mean_f1_score'].mean()
        measures['Gene clustering (tensor)'] = df['factors_pathways_nz_alpha 0.05'].mean()
        measures['Relevant pathway clustering (tensor)'] = df['ko_traits_nz_alpha 0.05'].mean()

        df = comb_IMPC_thr_theoretK[(comb_IMPC_thr_theoretK['method'] == method) &
                                    (comb_IMPC_thr_theoretK['tensor'] == 'non-tensor')]
        measures['Tissue clustering (non-tensor)'] = df['traits_tissue_mean_f1_score'].mean()
        measures['Genotype clustering (non-tensor)'] = df['traits_genotype_mean_f1_score'].mean()
        measures['Gene clustering (non-tensor)'] = df['factors_pathways_nz_alpha 0.05'].mean()
        measures['Relevant pathway clustering (non-tensor)'] = df['ko_traits_nz_alpha 0.05'].mean()

        df = comb_K_thr[(comb_K_thr['K_init'] == 20) &
                        (comb_K_thr['method'] == method)]
        measures['Time ($K_\text{init}=20$)'] = df.s.mean()

        df = comb_K_thr[(comb_K_thr['K_init'] == 100) &
                        (comb_K_thr['method'] == method)]
        measures['Time ($K_\text{init}=100$)'] = df.s.mean()

        if method in ['BicMix', 'BicMix-Q', 'SSLB']:
            K_init_value = 25
        else:
            K_init_value = 20
        df = comb_thr[(comb_thr['seedless_dataset'] == 'simulated/constant_negbin/size_mixed/K400_N300_G10000_T20') &
                      (comb_thr['method'] == method) &
                      (comb_thr['K_init'] == K_init_value)]
        measures['Time (large-K400 dataset)'] = df.s.mean()

        df = comb_IMPC_thr[(comb_IMPC_thr['tensor'] == 'tensor') &
                           (comb_IMPC_thr['K_init'] == 200) &
                           (comb_IMPC_thr['preprocess'] == 'log') &
                           (comb_IMPC_thr['method'] == method)]
        measures['Time (tensor IMPC datasets)'] = df.s.mean()

        df = comb_IMPC_thr[(comb_IMPC_thr['tensor'] == 'non-tensor') &
                           (comb_IMPC_thr['K_init'] == 200) &
                           (comb_IMPC_thr['preprocess'] == 'log') &
                           (comb_IMPC_thr['method'] == method)]
        measures['Time (non-tensor IMPC datasets)'] = df.s.mean()

        all_measures_dicts[method] = measures

    measures_df = pd.DataFrame(all_measures_dicts)
    return measures_df

