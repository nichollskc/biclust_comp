import os
import unittest

import numpy as np
import pandas as pd

from biclust_comp.analysis import accuracy as acc

class TestAccuracy(unittest.TestCase):
    expected_int_mat = np.array([[1, 1, 0, 0],
                                 [4, 8, 0, 5],
                                 [1, 2, 0, 2],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 5]])
    expected_union_mat = np.array([[7, 11, 4, 9],
                                   [42, 42, 42, 42],
                                   [13, 16, 10, 13],
                                   [5, 9, 1, 5],
                                   [6, 10, 2, 6],
                                   [9, 13, 5, 5]])

    def __init__(self, *args, **kwargs):
        super(TestAccuracy, self).__init__(*args, **kwargs)
        self._setup_example()

    def _setup_example(self):
        N = 6
        G = 7
        K1 = 6
        K2 = 4

        X1 = np.zeros((N, K1))
        B1 = np.zeros((G, K1))
        X1[:, 0] = [1, 1, 0, 0, 0, 0]
        B1[:, 0] = [1, 1, 0, 0, 0, 0, 0]
        X1[:, 1] = [1, 1, 1, 1, 1, 1]
        B1[:, 1] = [1, 1, 1, 1, 1, 1, 1]
        X1[:, 2] = [0, 0, 1, 1, 0, 0]
        B1[:, 2] = [0, 0, 1, 1, 1, 1, 1]
        X1[:, 3] = [0, 0, 0, 0, 0, 1]
        B1[:, 3] = [0, 0, 0, 0, 0, 0, 1]
        X1[:, 4] = [1, 1, 0, 0, 0, 0]
        B1[:, 4] = [0, 0, 0, 0, 0, 0, 1]
        X1[:, 5] = [0, 1, 1, 1, 1, 1]
        B1[:, 5] = [0, 0, 0, 0, 0, 0, 1]

        X2 = np.zeros((N, K2))
        B2 = np.zeros((G, K2))
        X2[:, 0] = [0, 1, 1, 0, 0, 0]
        B2[:, 0] = [0, 1, 1, 0, 0, 0, 0]
        X2[:, 1] = [0, 1, 1, 1, 1, 0]
        B2[:, 1] = [0, 1, 1, 0, 0, 0, 0]
        X2[:, 2] = [0, 0, 0, 0, 0, 0]
        B2[:, 2] = [0, 0, 0, 0, 0, 0, 0]
        X2[:, 3] = [0, 1, 1, 1, 1, 1]
        B2[:, 3] = [0, 0, 0, 0, 0, 0, 1]

        self.X1 = X1
        self.B1 = B1
        self.X2 = X2
        self.B2 = B2

    def test_stable(self):
        np.random.seed(123)

        for i in range(10):
            nrow = np.random.randint(low=2, high= 1000)
            ncol = np.random.randint(low=2, high= 1000)

            X = np.random.binomial(n=1, p=0.2, size=(nrow, 10))
            B = np.random.binomial(n=1, p=0.2, size=(ncol, 10))
            Y_true = np.matmul(X, B.T)

            X_b = np.random.binomial(n=1, p=0.4, size=(nrow, 20))
            B_b = np.random.binomial(n=1, p=0.4, size=(ncol, 20))

            int_mat, union_mat = acc.calc_overlaps(X, B, X_b, B_b)
            union_size = acc.calculate_union_size_combined(X, B, X_b, B_b)

            acc.calc_jaccard_rec_rel(int_mat, union_mat)
            acc.calc_clust_error(int_mat, union_size)
            acc.calc_recon_error(X_b, B_b, Y_true)

    def test_identity(self):
        clust_err = acc.calc_clust_error_full(self.X1, self.B1, self.X1, self.B1)
        self.assertEqual(clust_err, 1)

        clust_err = acc.calc_clust_error_full(self.X2, self.B2, self.X2, self.B2)
        self.assertEqual(clust_err, 1)

    def test_overlaps(self):
        int_mat, union_mat = acc.calc_overlaps(self.X1, self.B1, self.X2, self.B2)
        np.testing.assert_array_equal(self.expected_int_mat, int_mat)
        np.testing.assert_array_equal(self.expected_union_mat, union_mat)

    def test_jaccard_rec_rel(self):
        expected_jac = np.array([[0.14285714,  0.09090909, 0., 0.        ],
                                 [0.0952381 ,  0.19047619, 0., 0.11904762],
                                 [0.07692308,  0.125     , 0., 0.15384615],
                                 [0.        ,  0.        , 0., 0.2       ],
                                 [0.        ,  0.        , 0., 0.16666667],
                                 [0.        ,  0.        , 0., 1.        ]])

        expected_jac_rel_idx = [0, 1, 0, 5]
        expected_jac_rel_vals = [0.14285714, 0.19047619, 0, 1]
        expected_jac_rec_idx = [0, 1, 3, 3, 3, 3]
        expected_jac_rec_vals = [0.14285714, 0.19047619, 0.15384615, 0.2, 0.16666667, 1]

        jac_dict = acc.calc_jaccard_rec_rel(self.expected_int_mat,
                                            self.expected_union_mat)
        np.testing.assert_almost_equal(jac_dict['jaccard_relevance_idx'],
                                      expected_jac_rel_idx)
        np.testing.assert_almost_equal(jac_dict['jaccard_relevance_scores'],
                                      expected_jac_rel_vals)
        np.testing.assert_almost_equal(jac_dict['jaccard_recovery_idx'],
                                      expected_jac_rec_idx)
        np.testing.assert_almost_equal(jac_dict['jaccard_recovery_scores'],
                                      expected_jac_rec_vals)

    def test_union_size(self):
        expected_max_count = np.array([[2, 2, 1, 1, 1, 1, 2],
                                       [2, 2, 2, 1, 1, 1, 3],
                                       [1, 2, 2, 2, 2, 2, 3],
                                       [1, 1, 2, 2, 2, 2, 3],
                                       [1, 1, 1, 1, 1, 1, 2],
                                       [1, 1, 1, 1, 1, 1, 3]])
        expected_union_size = expected_max_count.sum()

        union_size = acc.calculate_union_size_combined(self.X1, self.B1,
                                                       self.X2, self.B2)
        self.assertEqual(union_size, expected_union_size)

if __name__ == '__main__':
    unittest.main()
