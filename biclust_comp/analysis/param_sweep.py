import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from biclust_comp.analysis import accuracy_utils as acc_utils
from biclust_comp.analysis import plots
from biclust_comp import logging_utils
from biclust_comp import utils

PARAMS_SCALES_DICT = {('FABIA', 'eps')     : 'log',
                      ('SDA', 'step_size') : 'log',
                      ('SDA', 'conv_crit') : 'symlog',
                      ('SNMF', 'beta')     : 'log',
                      ('SSLB', 'alpha')    : 'log',
                      ('SSLB', 'a')        : 'log',
                      ('SSLB', 'b')        : 'log'}
TIDY_ERROR_NAME = {'clust_err': 'CE',
                   'recon_error': 'RE',
                   'recon_error_normalised': 'NRE'}


def plot_all_param_sweeps(snakemake_config_log):
    """
    Make all the plots showing parameter sweep for the methods. Use the
    snakemake config log file to fetch default values, parameters changed in the
    parameter sweep and datasets used.
    Args:
        snakemake_config_log: Name of a file containing the config dump from
                                snakemake
    """
    # Fetch the dataframes containing accuracy information for each run
    logging.info(f"Fetching accuracy results")
    combined_error = pd.read_csv('analysis/accuracy/all_results_expected_PARAM_SWEEP.csv')

    # Collect config used by snakemake - contains e.g. default values
    logging.info(f"Reading in config from file {snakemake_config_log}")
    with open(snakemake_config_log, 'r') as f:
        config_dict = json.load(f)

    logging.info(combined_error['method'].value_counts())
    combined_error['method'].replace({'BicMix-Q': 'BicMix'}, inplace=True)
    logging.info(combined_error['method'].value_counts())

    combined_error = combined_error[combined_error['run_complete']]

    # For any parameters that might have scientific notation
    #     ensure they are converted to floats
    logging.info(f"Updating parameter values")
    combined_error['conv_crit'] = combined_error['conv_crit'].astype('float')
    combined_error['step_size'] = combined_error['step_size'].astype('float')
    combined_error['alpha'] = combined_error['alpha'].astype('float')
    combined_error['a'] = combined_error['a'].astype('float')
    combined_error['b'] = combined_error['b'].astype('float')
    combined_error['beta'] = combined_error['beta'].astype('float')
    combined_error['eps'] = combined_error['eps'].astype('float')

    # Set default values that weren't specified in snakemake config
    combined_error.loc[(combined_error['conv_crit'].isna()) &
                       (combined_error['method'] == 'SDA'), 'conv_crit'] = 0
    combined_error.loc[(combined_error['num_dense'].isna()) &
                       (combined_error['method'] == 'SDA'), 'num_dense'] = 0
    combined_error.loc[(combined_error['step_size'].isna()) &
                       (combined_error['method'] == 'SDA'), 'step_size'] = 0.0001
    combined_error.loc[(combined_error['rescale_l'].isna()) &
                       (combined_error['method'] == 'FABIA'), 'rescale_l'] = False

    # Plaid doesn't produce exact matrices, so only needs clustering error
    plot_param_sweep('Plaid', combined_error, config_dict, "clust_err")
    utils.save_plot("plots/param_sweep/Plaid_CE.png")

    # Restrict to after scaling and thresholding
    combined_error_thr = combined_error[combined_error.processing == '_thresh_1e-4']

    # SSLB has its own functions - defaults are based on dataset, so more
    #   complicated than for the other methods
    plot_param_sweep_SSLB(combined_error, config_dict, "recon_error_normalised")
    utils.save_plot("plots/param_sweep/SSLB_NRE.png")
    plot_param_sweep_SSLB(combined_error, config_dict, "clust_err")
    utils.save_plot("plots/param_sweep/SSLB_CE.png")

    # FABIA - exclude rescale_l from plot as it often failed
    #   exclude thresZ as we don't use it - instead we manually extract members
    plot_param_sweep('FABIA',
                     combined_error_thr,
                     config_dict,
                     "recon_error_normalised",
                     swept_params=['alpha', 'spz', 'spl', 'eps'])
    utils.save_plot(f"plots/param_sweep/FABIA_NRE.png")
    plot_param_sweep('FABIA',
                     combined_error_thr,
                     config_dict,
                     "clust_err",
                     swept_params=['alpha', 'spz', 'spl', 'eps'])
    utils.save_plot(f"plots/param_sweep/FABIA_CE.png")

    # All other methods
    # Exclude MultiCluster as it doesn't have any parameters to tune
    for method in ['BicMix', 'nsNMF', 'SDA', 'SNMF']:
        plot_param_sweep(method,
                         combined_error_thr,
                         config_dict,
                         "recon_error_normalised")
        utils.save_plot(f"plots/param_sweep/{method}_NRE.png")

        plot_param_sweep(method,
                         combined_error_thr,
                         config_dict,
                         "clust_err")
        utils.save_plot(f"plots/param_sweep/{method}_CE.png")

    # BicMix with qnorm=0
    plot_param_sweep('BicMix',
                     combined_error_thr,
                     config_dict,
                     "recon_error_normalised",
                     constraints_update={'qnorm': 0})
    utils.save_plot(f"plots/param_sweep/BicMix_qnorm_0_NRE.png")
    plot_param_sweep('BicMix',
                     combined_error_thr,
                     config_dict,
                     "clust_err",
                     constraints_update={'qnorm': 0})
    utils.save_plot(f"plots/param_sweep/BicMix_qnorm_0_CE.png")

    # FABIA with spz=1.5
    plot_param_sweep('FABIA',
                     combined_error_thr,
                     config_dict,
                     "recon_error_normalised",
                     constraints_update={'spz': 1.5},
                     swept_params=['alpha', 'spl', 'eps'])
    utils.save_plot(f"plots/param_sweep/FABIA_spz_1.5_NRE.png")

    plot_param_sweep('FABIA',
                     combined_error_thr,
                     config_dict,
                     "clust_err",
                     constraints_update={'spz': 1.5},
                     swept_params=['alpha', 'spl', 'eps'])
    utils.save_plot(f"plots/param_sweep/FABIA_spz_1.5_CE.png")

    plot_time_against_param('SDA', combined_error_thr, config_dict, "conv_crit")
    utils.save_plot(f"plots/param_sweep/SDA_conv_crit_s.png")

    plot_time_against_param('SDA', combined_error_thr, config_dict, "step_size")
    utils.save_plot(f"plots/param_sweep/SDA_step_size_s.png")


def plot_param_sweep(method, error_df, config_dict, error_to_plot, ylim=(0,1),
                     constraints_update={}, swept_params=None):
    method_to_K_config_list = {'SSLB':     'K_SWEEP_OVERESTIMATE',
                               'BicMix': 'K_SWEEP_OVERESTIMATE'}
    K_list_name = method_to_K_config_list.get(method, 'K_SWEEP')
    K_list = config_dict['PARAM_SWEEP'][K_list_name]

    datasets = config_dict['SIMULATED']['param_sweep_datasets']

    if swept_params is None:
        swept_params = config_dict['PARAM_SWEEP'][method].keys()

    nrows, ncols = 1, len(swept_params) + 1
    figsize = [4 * ncols + 2, 6]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)

    plot_param_sweep_single(method, error_df, config_dict, error_to_plot,
                            datasets, None, 'K_init', ax[0], ylim=ylim,
                            constraints_update=constraints_update)

    for ind, param in enumerate(swept_params):
        if ind != 0:
            legend=False
        else:
            legend='brief'

        # Look up scale for x-axis in dictionary, using 'linear' as default
        xscale = PARAMS_SCALES_DICT.get((method, param),
                                        'linear')
        plot_param_sweep_single(method, error_df, config_dict, error_to_plot,
                                datasets, K_list, param, ax[ind + 1],
                                legend=legend, xscale=xscale, ylim=ylim,
                                constraints_update=constraints_update)


def plot_param_sweep_single(method, error_df, config_dict, error_to_plot,
                            datasets, K_list, param, ax, ylim=(0,1),
                            xscale='linear', legend='brief', constraints_update={}):
    logging.info(f"Plotting parameter sweep of parameter {param} for method {method}")

    expected_runs = len(config_dict['SIMULATED']['run_seeds']) * len(datasets)
    if K_list:
        expected_runs *= len(K_list)
    logging.info(f"expecting {expected_runs} runs for each value of parameter")

    restricted = construct_param_sweep_df(method, error_df, config_dict,
                                          datasets, K_list, param,
                                          constraints_update)
    logging.info(restricted[param].value_counts())

    # Exclude runs that were purely using default parameters
    #   to get even numbers for each parameter we only use the full run
    restricted = restricted[restricted.run_id.str.match(r'run_seed_\d+_')]

    ax.set(ylim=ylim, xscale=xscale)
    if param == 'conv_crit':
        ax.set_xscale('symlog', linthreshx=1e-7)

    if param != 'K_init':
        style = 'K_init'
    else:
        style = None

    logging.info(f"These counts should all be {expected_runs}:\n{restricted[param].value_counts()}")
    sns.lineplot(x=param, y=error_to_plot,
                 hue='short_seedless_dataset', style=style,
                 hue_order=['gaussian_K50', 'negbin_K20', 'no_noise_K10'],
                 data=restricted, ax=ax, legend=legend)
    sns.scatterplot(x=param, y=error_to_plot,
                    hue='short_seedless_dataset', style=style,
                    hue_order=['gaussian_K50', 'negbin_K20', 'no_noise_K10'],
                    data=restricted , ax=ax, legend=False)
    values = list(restricted[param].unique())

    # Set the y-axis label to a tidier version of the error_name if we have
    #   defined a tidier version, else leave it as it is
    ax.set_ylabel(TIDY_ERROR_NAME.get(error_to_plot,
                                      error_to_plot))

    if param in ['K_init', 'qnorm', 'rescale_l', 'num_dense', 'IBP']:
        values = [int(val) for val in values]

    ax.set_xticks(values)
    ax.set_xticklabels(values)
    format_param_sweep_xaxis(param, values, ax, config_dict['PARAMETERS'][method])

def format_param_sweep_xaxis(param, values, ax, default_params):
    labels = ['{:.0e}'.format(float(label.get_text()))
              for label in ax.get_xticklabels()]
    logging.info(labels)

    if param in ['conv_crit', 'step_size', 'eps']:
        ax.set_xticklabels(labels)

    # Emphasise the default parameter label
    #   not for K_init as defaults have less meaning for this parameter
    if param != 'K_init':
        default_index = labels.index('{:.0e}'.format(float(default_params[param])))
        ax.get_xticklabels()[default_index].set_color('red')
        ax.get_xticklabels()[default_index].set_weight("bold")


def construct_param_sweep_df(method, error_df, config_dict, datasets, K_list,
                             param, constraints_update={}):
    default_params = config_dict['PARAMETERS'][method]

    constraints = {key: val for key, val in default_params.items()
                   if key not in ['seed', 'set_seed', 'K_init', 'seed_method',
                                  'n_clusters', 'rank', 'num_comps',
                                  'random_state', 'lambda0s', 'lambda0_tildes',
                                  param]}
    for key, val in constraints_update.items():
        if key != param:
            constraints[key] = val
    logging.info(constraints)

    mask_default = pd.DataFrame([error_df[key] == float(val)
                                 for key, val in constraints.items()]).T.all(axis=1)
    matching_default = error_df[mask_default]
    logging.debug(matching_default.shape)

    restricted = matching_default[(matching_default['seedless_dataset'].isin(datasets)) &
                                  (matching_default['method'] == method)]
    restricted.loc[:, param] = restricted[param].astype('float')

    if param != 'K_init':
        restricted = restricted[restricted['K_init'].isin(K_list)]

    return restricted


# SSLB needs a separate function as it has some parameters where the default
# depends on the dataset
def plot_param_sweep_SSLB(error_df, config_dict, error_to_plot, ylim=(0,1)):
    method = 'SSLB'

    datasets = config_dict['SIMULATED']['param_sweep_datasets']
    K_list = config_dict['PARAM_SWEEP']['K_SWEEP_OVERESTIMATE']

    logging.info(f"Plotting parameter sweep for method {method}")
    expected_runs = len(K_list) * len(config_dict['SIMULATED']['run_seeds']) * len(datasets)
    logging.info(f"expecting {expected_runs} runs for each value of each parameter")

    swept_params = [p for p in config_dict['PARAM_SWEEP'][method].keys()]
    default_params = config_dict['PARAMETERS'][method]

    nrows, ncols = 1, len(swept_params) + 1
    figsize = [5 * ncols + 2, 10]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)

    logging.info("K_init")
    # Default parameters - hard-coded since N=10, T=10 in all datasets
    masks = {'IBP'     : (error_df['IBP'] == 1),
             'd'       : (error_df['d'] == 0),
             'alpha'   : (error_df['alpha'] == 1 / 100),
             'a'       : np.isclose(error_df['a'],
                                    1 / error_df['K_init'],
                                    rtol = 1e-03),
             'b'       : np.isclose(error_df['b'],
                                    1 / error_df['K_init'],
                                    rtol = 1e-03)}
    mask_default_df = pd.DataFrame([mask for mask in masks.values()])
    mask_default = mask_default_df.all(axis=0)
    matching_default = error_df[mask_default]
    restricted = matching_default[(matching_default['seedless_dataset'].isin(datasets)) &
                                  (matching_default['method'] == method)]
    restricted = restricted[restricted.run_id.str.match(r'run_seed_\d+_')]
    logging.info(restricted['K_init'].value_counts())
    logging.debug(restricted[['short_dataset', 'run_id', 'processing', 'K_init']])
    sns.lineplot(x='K_init', y=error_to_plot,
                 hue='short_seedless_dataset',
                 hue_order=['gaussian_K50', 'negbin_K20', 'no_noise_K10'],
                data=restricted, ax=ax[0], legend=False)
    sns.scatterplot(x='K_init', y=error_to_plot,
                    hue='short_seedless_dataset',
                    hue_order=['gaussian_K50', 'negbin_K20', 'no_noise_K10'],
                data=restricted , ax=ax[0], legend='brief')
    ax[0].set(ylim=ylim)
    values = [int(K) for K in restricted['K_init'].unique()]
    ax[0].set_xticks(values)
    ax[0].set_xticklabels(values)

    # Set the y-axis label to a tidier version of the error_name if we have
    #   defined a tidier version, else leave it as it is
    ax[0].set_ylabel(TIDY_ERROR_NAME.get(error_to_plot,
                                         error_to_plot))


    for ind, param in enumerate(swept_params):

        logging.info(f"Parameter: {param}")
        good_keys = [key for key in masks.keys() if key != param]
        mask_default_df = pd.DataFrame([masks[key] for key in good_keys])
        mask_default = mask_default_df.all(axis=0)
        matching_default = error_df[mask_default]
        restricted = matching_default[(matching_default['K_init'].isin(K_list)) &
                                      (matching_default['seedless_dataset'].isin(datasets)) &
                                      (matching_default['method'] == method) &
                                      (matching_default['processing'] == '_thresh_1e-4')]
        restricted.loc[:, param] = restricted[param].astype('float')

        logging.info(restricted[param].value_counts())

        # Exclude runs that were purely using default parameters
        #   to get even numbers for each parameter we only use the full run
        restricted = restricted[restricted.run_id.str.match(r'run_seed_\d+_')]

        logging.info(f"These counts should all be {expected_runs}:\n{restricted[param].value_counts()}")
        logging.debug(restricted[['short_dataset', 'run_id', 'processing', 'K_init', param]])

        if ind == 0:
            legend_type = 'brief'
        else:
            legend_type = False

        # Look up scale for x-axis in dictionary, using 'linear' as default
        xscale = PARAMS_SCALES_DICT.get((method, param),
                                        'linear')

        sns.lineplot(x=param, y=error_to_plot,
                     hue='short_seedless_dataset',
                     hue_order=['gaussian_K50', 'negbin_K20', 'no_noise_K10'],
                     style='K_init',
                    data=restricted, ax=ax[ind + 1], legend=legend_type)
        sns.scatterplot(x=param, y=error_to_plot,
                        hue='short_seedless_dataset', style='K_init',
                        hue_order=['gaussian_K50', 'negbin_K20', 'no_noise_K10'],
                    data=restricted , ax=ax[ind + 1], legend=False)
        ax[ind + 1].set(ylim=ylim, xscale=xscale)

        values = list(restricted[param].unique())
        ax[ind + 1].set_xticks(values)
        ax[ind + 1].set_xticklabels(values)

        # Set the y-axis label to a tidier version of the error_name if we have
        #   defined a tidier version, else leave it as it is
        ax[ind + 1].set_ylabel(TIDY_ERROR_NAME.get(error_to_plot,
                                                   error_to_plot))

        if param in ['alpha', 'a', 'b']:
            # Use scientific notation for all labels, and rotate so they fit
            ax[ind + 1].set_xticklabels(['{:.0e}'.format(val) for val in values])
            ax[ind + 1].set_xticklabels(ax[ind + 1].get_xticklabels(),
                                        rotation=45,
                                        ha='right')


def plot_time_against_param(method, error_df, config_dict, param):
    restricted = construct_param_sweep_df('SDA',
                                          error_df,
                                          config_dict,
                                          config_dict['SIMULATED']['param_sweep_datasets'],
                                          [20, 50],
                                          param)
    logging.info(restricted[param].value_counts())

    # Exclude runs that were purely using default parameters
    #   to get even numbers for each parameter we only use the full run
    restricted = restricted[restricted.run_id.str.match(r'run_seed_\d+_')]

    logging.info(f"These counts should all be equal:\n{restricted[param].value_counts()}")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=restricted, x=param, y="s", ax=ax)
    ax.set_ylabel("runtime (seconds)")
    values = list(restricted[param].unique())
    format_param_sweep_xaxis(param, values, ax, config_dict['PARAMETERS'][method])
    return restricted


if __name__ == "__main__":
    matplotlib.rcParams['axes.labelsize'] = 'xx-large'
    logging_utils.setup_logging(3)
    snakemake_config_log = "analysis/accuracy/snakemake_config_param_sweep.log"
    plot_all_param_sweeps(snakemake_config_log)

