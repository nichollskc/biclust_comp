localrules: all

import itertools
import json
import logging
import time
import os

from biclust_comp import utils
from biclust_comp import logging_utils
import biclust_comp.simulate.plan as plan

logging_utils.setup_logging(3)

def get_scripts_absolute_path():
    working_dir = os.getcwd()
    if 'BICLUST_COMP_SCRIPTS_RELATIVE' in config:
        relative_dir = config['BICLUST_COMP_SCRIPTS_RELATIVE']
    else:
        relative_dir = ""

    return os.path.join(working_dir, relative_dir)

config['BICLUST_COMP_SCRIPTS'] = get_scripts_absolute_path()

configfile: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/snakemake_default_config.yml"

# Setup
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/utils.rules"

# Preprocess data
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/preprocess.rules"

# Generating data
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/simulate.rules"

# Running methods
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/run_biclustering.rules"
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/simulated_runs.rules"
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/impc_runs.rules"
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/presnell_runs.rules"

# Analysis
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/simulated_analysis.rules"
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/simulated_plots.rules"
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/impc_analysis.rules"
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/impc_plots.rules"
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/presnell_plots.rules"

# Testing
include: f"{config['BICLUST_COMP_SCRIPTS']}/biclust_comp/workflows/testing.rules"

wildcard_constraints:
    N="\d+",
    T="\d+",
    G="\d+",
    K="\d+",
    dataset_group="|".join(['PARAM_SWEEP'] + list(config['SIMULATED']['dataset_groups'].keys()))

timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')
logfile = f"logs/snakemake_config/{timestamp}"
utils.ensure_dir(logfile)

with open(logfile, 'w') as f:
    json.dump(config, f, indent=4)

rule simulated_plots:
    input:
        rules.main_accuracy_plots.input,
        rules.comp_K_accuracy_plots.input,

rule all_plots:
    input:
        rules.impc_plots.input,
        rules.simulated_plots.input,
        rules.combined_plots.input,
        rules.presnell_summaries.input,
