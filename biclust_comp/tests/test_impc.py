import os
import unittest

import numpy as np
import pandas as pd

import biclust_comp.processing.impc as impc

class TestPooledVariance(unittest.TestCase):

    def test_pooled_variance(self):
        group1 = [10, 12, 13, 15, 9]
        group2 = [2, 3, 1]
        group3 = [1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 2, 3]
        # Calculated using https://www.easycalculation.com/statistics/pooled-standard-deviation.php
        expected_pooled_variance = (1.326650165133974)**2

        counts = np.concatenate([group1, group2, group3])
        alpha = 62
        beta = 13.1312

        ids = [f"ID{i}" for i in range(len(counts))]
        counts_df = pd.DataFrame({'variable1': counts,
                                  'variable2': alpha * counts,
                                  'variable3': beta * counts,
                                  'variable4': -1 * counts},
                                  index=ids)
        group_labels = np.repeat(['group1', 'group2', 'group3'],
                                 [len(group1), len(group2), len(group3)])
        sample_info = pd.DataFrame({'ID': ids, 'group': group_labels})

        pooled_variance = impc.calculate_pooled_variances(counts_df,
                                                          sample_info.set_index('ID').groupby('group'))

        self.assertTrue(np.isclose(pooled_variance[0], expected_pooled_variance))
        self.assertTrue(np.isclose(pooled_variance[1], expected_pooled_variance * (alpha ** 2)))
        self.assertTrue(np.isclose(pooled_variance[2], expected_pooled_variance * (beta ** 2)))
        self.assertTrue(np.isclose(pooled_variance[3], expected_pooled_variance))

        # Shuffling the order of samples in both counts_df and sample_info df shouldn't affect calculations
        pooled_variance = impc.calculate_pooled_variances(counts_df.sample(frac=1),
                                                          sample_info.sample(frac=1).set_index('ID').groupby('group'))

        self.assertTrue(np.isclose(pooled_variance[0], expected_pooled_variance))
        self.assertTrue(np.isclose(pooled_variance[1], expected_pooled_variance * (alpha ** 2)))
        self.assertTrue(np.isclose(pooled_variance[2], expected_pooled_variance * (beta ** 2)))
        self.assertTrue(np.isclose(pooled_variance[3], expected_pooled_variance))

if __name__ == '__main__':
    unittest.main()
