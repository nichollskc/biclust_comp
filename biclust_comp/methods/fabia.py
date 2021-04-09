import json
import logging
import random

import fabia
import numpy as np

from biclust_comp import logging_utils
from biclust_comp import utils

# Set up logging with 'INFO' level
logging_utils.setup_logging(3, logfile=snakemake.log[0])

Y = np.loadtxt(snakemake.input.Y, delimiter='\t')

params = snakemake.params[0]
params.update({'data_file': snakemake.input.Y})
if 'eps' in params:
    params['eps'] = float(params['eps'])
if 'rescale_l' in params:
    params['rescale_l'] = bool(params['rescale_l'])

logging.info(f"Parameters from snakemake: {params}")
with open(snakemake.output.params, 'w') as f:
    json.dump(params, f, indent=2)

params.pop('data_file')
fabia_obj = fabia.FabiaBiclustering(**params)
fabia_obj.fit(Y)

logging.info(f"FABIA model has been fit, will now extract X, B")

# Factorises Y = Z*L where Z is (N, K) and L is (K, G)
# We want Y = X*B^T where X is (N, K) and B is (G, K)
X_raw = fabia_obj.Z_
B_raw = fabia_obj.L_.T

X_nonempty, B_nonempty = utils.remove_empty_factors(X_raw, B_raw)

X_binary = utils.binarise_matrix(X_nonempty)
B_binary = utils.binarise_matrix(B_nonempty)

K = X_nonempty.shape[1]

logging.info(f"FABIA found {K} factors")

np.savetxt(snakemake.output.X, X_nonempty, delimiter='\t')
np.savetxt(snakemake.output.B, B_nonempty, delimiter='\t')

with open(snakemake.output.K, 'w') as f:
    f.write(str(K))
