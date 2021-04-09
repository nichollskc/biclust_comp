import copy
import re

METHOD_TO_K_NAME = {'SDA':      'num_comps',
                    'nsNMF':    'rank',
                    'SNMF':     'rank',
                    'FABIA':    'n_clusters'}
METHOD_TO_SEED_DICT_FN = {'SDA': (lambda seed: {'set_seed': f"{seed} {seed * 2}"}),
                          'MultiCluster': (lambda seed: {}),
                          'FABIA': (lambda seed: {'random_state': seed})}
METHOD_TO_K_CONFIG_LIST = {'SSLB':     'OVERESTIMATE',
                           'BicMix':   'OVERESTIMATE'}
K_FUNCTION_DICT = {'add_10': (lambda K: K + 10),
                   'identity': (lambda K: K)}

def construct_planned_runs_dict(config, dataset_groups=None):
    simulate_config = copy.deepcopy(config['SIMULATED'])
    del simulate_config['dataset_groups']

    params_dict = {}
    run_ids_dict = {}
    all_datasets = []

    if dataset_groups is None:
        dataset_groups = list(config['SIMULATED']['dataset_groups'].keys())

    for group_name, group in config['SIMULATED']['dataset_groups'].items():
        if group_name not in dataset_groups:
            continue

        group_config = copy.deepcopy(simulate_config)
        group_config.update(group)

        run_ids_dict[group_name] = []
        for dataset in group['datasets']:
            # (1) For sim_seeds versions of each simulated dataset, run each method
            #   run_seeds times.
            for sim_seed in simulate_config['sim_seeds']:
                full_dataset = f"{dataset}/seed_{sim_seed}"
                all_datasets.append(full_dataset)

                for method in config['METHODS']:
                    run_ids, params = construct_planned_runs_dict_method_dataset(method,
                                                                                 full_dataset,
                                                                                 group_config,
                                                                                 config['PARAMETERS'])
                    params_dict.update(params)
                    run_ids_dict[group_name].extend(run_ids)

    return all_datasets, run_ids_dict, params_dict

def construct_planned_runs_dict_method_dataset(method, full_dataset, simulate_config, default_parameters):
    params_dict = {}

    # Find the name for K
    K_name = METHOD_TO_K_NAME.get(method, 'K_init')

    for seed in simulate_config['run_seeds']:
        # Set up the dictionary describing the seed
        seed_dict_fn = METHOD_TO_SEED_DICT_FN.get(method,                       # key to use
                                                  lambda seed: {'seed': seed})  # default value
        seed_dict = seed_dict_fn(seed)

        K_list_name = METHOD_TO_K_CONFIG_LIST.get(method, 'NORMAL')
        K_list = simulate_config['K_init'][K_list_name].copy()

        # Find out the true K value
        match = re.match(r'simulated/.*/.*/K(\d+)_N\d+_G\d+', full_dataset)
        assert match, f"Dataset name {full_dataset} not of expected form 'simulated/.*/.*/K(\d+)_N\d+_G\d+'"
        true_K = int(match[1])

        K_list = [K_init if isinstance(K_init, int) else K_FUNCTION_DICT[K_init](true_K) for K_init in K_list]

        for K in K_list:
            full_name = f"{method}/{full_dataset}/run_seed_{seed}_K_{K}"
            run_params_dict = {}
            run_params_dict[K_name] = K
            run_params_dict.update(seed_dict)

            if method == 'FABIA':
                run_params_dict.update({'spz': 1.5})
                # Note we are updating the full_name after having searched for parameters, but there shouldn't be
                # any settings in the config file for the specific run_id - only for method and dataset
                full_name += "_spz_1.5"

            params_dict[full_name] = run_params_dict

            # Run BicMix twice - once with qnorm=0, once with qnorm=1
            if method == 'BicMix':
                qnorm0_params_dict = copy.deepcopy(run_params_dict)
                full_name = f"{method}/{full_dataset}/run_seed_{seed}_K_{K}"
                qnorm0_params_dict.update({'qnorm': 0})
                full_name += "_qnorm_0"
                params_dict[full_name] = qnorm0_params_dict

    return params_dict.keys(), params_dict

def construct_planned_runs_dict_param_sweep_method_dataset(method, full_dataset, config):
    param_sweep_config_dict = {}
    param_sweep_run_seeds = config['PARAM_SWEEP']['run_seeds']
    K_name = METHOD_TO_K_NAME.get(method, 'K_init')

    for seed in param_sweep_run_seeds:
        # Set up the dictionary describing the seed
        seed_dict_fn = METHOD_TO_SEED_DICT_FN.get(method,                       # key to use
                                                  lambda seed: {'seed': seed})  # default value
        seed_dict = seed_dict_fn(seed)

        # (2A) Vary K
        for K in config['PARAM_SWEEP']['K']:
            full_name = f"{method}/{full_dataset}/run_seed_{seed}_K_{K}"
            cfg_dict = {}
            cfg_dict[K_name] = K
            cfg_dict.update(seed_dict)
            param_sweep_config_dict[full_name] = cfg_dict

        # (2B) For each value of K, vary the other parameters
        K_list_name = METHOD_TO_K_CONFIG_LIST.get(method, 'NORMAL')
        for K in config['PARAM_SWEEP']['K_SWEEP'][K_list_name]:

            for param_name, param_val_list in config['PARAM_SWEEP'][method].items():
                for param_value in param_val_list:
                    param_str = f"K_{K}_{param_name}_{param_value}"
                    assert '_' not in str(param_value)
                    # We will later assume that the run id doesn't contain
                    #    a '/' character
                    assert '/' not in param_str
                    full_name = f"{method}/{full_dataset}/run_seed_{seed}_{param_str}"

                    cfg_dict = {}
                    cfg_dict[K_name] = K
                    cfg_dict[param_name] = param_value
                    cfg_dict.update(seed_dict)
                    param_sweep_config_dict[full_name] = cfg_dict

        # (2C) Do extra runs for FABIA spz/eps and BicMix qnorm
        if method == 'BicMix':
            for K in config['PARAM_SWEEP']['K_SWEEP'][K_list_name]:
                for param_name in ['a', 'b']:
                    param_val_list = config['PARAM_SWEEP'][method][param_name]
                    for param_value in param_val_list:
                        param_str = f"K_{K}_{param_name}_{param_value}_qnorm_0"
                        assert '_' not in str(param_value)
                        # We will later assume that the run id doesn't contain
                        #    a '/' character
                        assert '/' not in param_str
                        full_name = f"{method}/{full_dataset}/run_seed_{seed}_{param_str}"

                        cfg_dict = {}
                        cfg_dict['qnorm'] = 0
                        cfg_dict[K_name] = K
                        cfg_dict.update(seed_dict)
                        cfg_dict[param_name] = param_value
                        param_sweep_config_dict[full_name] = cfg_dict

        if method == 'FABIA':
            for K in config['PARAM_SWEEP']['K_SWEEP'][K_list_name]:
                params_to_change = ['eps', 'spl', 'alpha']
                for param in params_to_change:
                    param_val_list = config['PARAM_SWEEP'][method][param]
                    for param_value in param_val_list:
                        param_str = f"K_{K}_{param}_{param_value}_spz_1.5"
                        assert '_' not in str(param_value)
                        # We will later assume that the run id doesn't contain
                        #    a '/' character
                        assert '/' not in param_str
                        full_name = f"{method}/{full_dataset}/run_seed_{seed}_{param_str}"

                        cfg_dict = {}
                        cfg_dict['spz'] = 1.5
                        cfg_dict[K_name] = K
                        cfg_dict.update(seed_dict)
                        cfg_dict[param] = param_value
                        param_sweep_config_dict[full_name] = cfg_dict

    return param_sweep_config_dict

def construct_planned_runs_dict_param_sweep(config):
    param_sweep_config_dict = {}
    sim_seed = config['PARAM_SWEEP']['sim_seed']

    for dataset in config['PARAM_SWEEP']['datasets']:
        full_dataset = f"{dataset}/seed_{sim_seed}"
        for method in config['METHODS']:
            params_dict = construct_planned_runs_dict_param_sweep_method_dataset(method, full_dataset, config)
            param_sweep_config_dict.update(params_dict)

    return param_sweep_config_dict

def combine_parameter_dicts(dict_1, dict_2):
    combined = dict_1.copy()
    for key, value in dict_2.items():
        if key in combined:
            assert combined[key] == value, \
                "Config from param sweep dictionary should match normal simulated runs " \
               f"Instead we found values {combined[key]} and {value} respectively " \
               f"for the key {key}"
        else:
            combined[key] = value
    return combined
