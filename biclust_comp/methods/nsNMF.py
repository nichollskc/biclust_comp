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

rank = params['rank']
theta = params['theta']
max_iter = params['max_iter']

seed = params['seed']

random.seed(seed)

nsnmf = nimfa.Nsnmf(Y, max_iter=max_iter, rank=rank, theta=theta)
nsnmf_fit = nsnmf()

# Decomposes matrix Y to Y = basis * coef = W * H = t(X) * B
X_raw = nsnmf_fit.basis()
B_raw = nsnmf_fit.coef().transpose()

X_nonempty, B_nonempty = utils.remove_empty_factors(np.array(X_raw),
                                                    np.array(B_raw))

X_binary = utils.binarise_matrix(X_nonempty)
B_binary = utils.binarise_matrix(B_nonempty)

K = X_nonempty.shape[1]

np.savetxt(snakemake.output.X, X_nonempty, delimiter='\t')
np.savetxt(snakemake.output.B, B_nonempty, delimiter='\t')

with open(snakemake.output.K, 'w') as f:
    f.write(str(K))
