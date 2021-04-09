import json
import logging
import random

import nimfa
import numpy as np

from biclust_comp import logging_utils
from biclust_comp import utils

# Set up logging with 'INFO' level
logging_utils.setup_logging(3, logfile=snakemake.log[0])

Y = np.loadtxt(snakemake.input.Y, delimiter='\t')

params = snakemake.params[0]
params.update({'data_file': snakemake.input.Y})

logging.info(f"Parameters from snakemake: {params}")
with open(snakemake.output.params, 'w') as f:
    json.dump(params, f, indent=2)

random.seed(params['seed'])

snmf = nimfa.Snmf(Y,
                  seed=params['seed_method'],   # Seeding method
                  rank=params['rank'],
                  max_iter=params['max_iter'],  # Maximum number of iterations
                  version='r',                  # Sparsity in right matrix (i.e. genes)
                  beta=float(params['beta']),          # Higher beta increases sparsity
                  i_conv=params['i_conv'],      # Number of iterations that clusters
                                                # must be constant to deem convergence
                  w_min_change=params['w_min_change'])   # Convergence criterion - num row clusters
                                                         # that can change
snmf_fit = snmf()

# Decomposes matrix Y to Y = basis * coef = W * H = t(X) * B
X_raw = snmf_fit.basis()
B_raw = snmf_fit.coef().transpose()

X_nonempty, B_nonempty = utils.remove_empty_factors(np.array(X_raw),
                                                    np.array(B_raw))

X_binary = utils.binarise_matrix(X_nonempty)
B_binary = utils.binarise_matrix(B_nonempty)

K = X_nonempty.shape[1]

np.savetxt(snakemake.output.X, X_nonempty, delimiter='\t')
np.savetxt(snakemake.output.B, B_nonempty, delimiter='\t')

with open(snakemake.output.K, 'w') as f:
    f.write(str(K))
