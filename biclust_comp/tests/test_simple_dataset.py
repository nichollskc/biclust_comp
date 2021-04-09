import os
import unittest

import numpy as np

from nose.plugins.attrib import attr

from biclust_comp import utils


class BiclusterComparer:

    def _check_K(self, results_dir):
        K = utils.read_int_from_file(results_dir, "K.txt")

        self.assertEqual(K, self.true_K)

    def _check_binary_shape(self, results_dir):
        K, X_binary, B_binary = utils.read_result_threshold_binary(results_dir, 0)

        self.assertEqual(B_binary.shape, (self.G, K))
        self.assertEqual(X_binary.shape, (self.NtimesC, K))

    def _check_shape(self, results_dir):
        K, X, B = utils.read_result_threshold(results_dir, 0)

        self.assertEqual(B.shape, (self.G, K))
        self.assertEqual(X.shape, (self.NtimesC, K))

    def _check_tensor_shape(self, results_dir):
        K = utils.read_int_from_file(results_dir, "K.txt")

        A = utils.read_np(results_dir, "A.txt")
        B = utils.read_np(results_dir, "B.txt")
        Z = utils.read_np(results_dir, "Z.txt")

        self.assertEqual(A.shape, (self.N, K))
        self.assertEqual(B.shape, (self.G, K))
        self.assertEqual(Z.shape, (self.C, K))

    def _check_approximate_output(self, results_dir, threshold=0):
        K, X, B = utils.read_result_threshold(results_dir, threshold)

        self.assertEqual(X.shape, self.true_X_shape)
        self.assertEqual(B.shape, self.true_B_shape)

        X_scaled = abs(X) / np.std(X)
        B_scaled = abs(B) / np.std(B)

        X_error = np.linalg.norm(self.true_X_binary_scaled - X_scaled)
        B_error = np.linalg.norm(self.true_B_binary_scaled - B_scaled)

        self.assertTrue(X_error < 1, f"Error is {X_error}")
        self.assertTrue(B_error < 1, f"Error is {B_error}")

    def _check_binary_output(self, results_dir):
        K, X_binary, B_binary = utils.read_result_threshold_binary(results_dir, 0)

        self.assertEqual(B_binary.shape, self.true_B_shape)
        np.testing.assert_array_equal(self.true_B_binary, B_binary)

        self.assertEqual(X_binary.shape, self.true_X_shape)
        np.testing.assert_array_equal(self.true_X_binary, X_binary)

    def _check_approximate_output_tensor(self, results_dir):
        A = utils.read_np(results_dir, "A.txt")
        B = utils.read_np(results_dir, "B.txt")
        Z = utils.read_np(results_dir, "Z.txt")

        self.assertEqual(A.shape, self.true_A_shape)
        self.assertEqual(B.shape, self.true_B_shape)
        self.assertEqual(Z.shape, self.true_Z_shape)

        A_scaled = abs(A) / np.std(A)
        B_scaled = abs(B) / np.std(B)
        Z_scaled = abs(Z) / np.std(Z)

        A_error = np.linalg.norm(self.true_A_binary_scaled - A_scaled)
        B_error = np.linalg.norm(self.true_B_binary_scaled - B_scaled)
        Z_error = np.linalg.norm(self.true_Z_binary_scaled - Z_scaled)

        self.assertTrue(A_error < 0.5, f"Error is {A_error}")
        self.assertTrue(B_error < 0.5, f"Error is {B_error}")
        self.assertTrue(Z_error < 0.5, f"Error is {Z_error}")

    def _check_binary_output_tensor(self, results_dir):
        K, X_binary, B_binary = utils.read_result_threshold_binary(results_dir, 0)

        self.assertEqual(B_binary.shape, (100, 1))
        np.testing.assert_array_equal(self.true_B_binary, B_binary)

        self.assertEqual(X_binary.shape, (50, 1))
        np.testing.assert_array_equal(self.true_X_binary, X_binary)


class TestGaussian(unittest.TestCase, BiclusterComparer):
    true_K = 1

    true_X_binary = utils.read_np("data/tests/simple_gaussian", "X_binary.txt")
    true_B_binary = utils.read_np("data/tests/simple_gaussian", "B_binary.txt")
    true_A_binary = utils.read_np("data/tests/simple_gaussian", "A_binary.txt")
    true_Z_binary = utils.read_np("data/tests/simple_gaussian", "Z_binary.txt")

    true_X_binary_scaled = true_X_binary / np.std(true_X_binary)
    true_B_binary_scaled = true_B_binary / np.std(true_B_binary)
    true_A_binary_scaled = true_A_binary / np.std(true_A_binary)
    true_Z_binary_scaled = true_Z_binary / np.std(true_Z_binary)

    true_X_shape = true_X_binary.shape
    true_B_shape = true_B_binary.shape
    true_A_shape = true_A_binary.shape
    true_Z_shape = true_Z_binary.shape

    @attr('plaid')
    def test_plaid_gaussian_output(self):
        results_dir = "results/Plaid/tests/simple_gaussian/run_1"

        self._check_K(results_dir)
        self._check_binary_output(results_dir)

    @attr('sslb')
    def test_sslb_gaussian_output(self):
        results_dir = "results/SSLB/tests/simple_gaussian/run_1"

        self._check_K(results_dir)
        self._check_binary_output(results_dir)

    @attr('fabia')
    def test_fabia_gaussian_output(self):
        results_dir = "results/FABIA/tests/simple_gaussian/run_1"

        self._check_K(results_dir)
        self._check_approximate_output(results_dir)

    @attr('bicmix')
    def test_bicmix_gaussian_output(self):
        results_dir = "results/BicMix/tests/simple_gaussian/run_1"

        self._check_K(results_dir)
        self._check_approximate_output(results_dir)

    @attr('matlab')
    def test_multicluster_gaussian_output(self):
        results_dir = "results/MultiCluster/tests/simple_gaussian/run_1"

        self._check_K(results_dir)
        self._check_approximate_output_tensor(results_dir)

    @attr('sda')
    def test_sda_gaussian_output(self):
        results_dir = "results/SDA/tests/simple_gaussian/run_1"

        self._check_K(results_dir)
        self._check_approximate_output_tensor(results_dir)


class TestNegBin(unittest.TestCase, BiclusterComparer):
    true_K = 1

    true_X_binary = utils.read_np("data/tests/simple_negbin", "X_binary.txt")
    true_B_binary = utils.read_np("data/tests/simple_negbin", "B_binary.txt")
    true_A_binary = utils.read_np("data/tests/simple_negbin", "A_binary.txt")
    true_Z_binary = utils.read_np("data/tests/simple_negbin", "Z_binary.txt")

    true_X_binary_scaled = true_X_binary / np.std(true_X_binary)
    true_B_binary_scaled = true_B_binary / np.std(true_B_binary)
    true_A_binary_scaled = true_A_binary / np.std(true_A_binary)
    true_Z_binary_scaled = true_Z_binary / np.std(true_Z_binary)

    true_X_shape = true_X_binary.shape
    true_B_shape = true_B_binary.shape
    true_A_shape = true_A_binary.shape
    true_Z_shape = true_Z_binary.shape

    @attr('plaid')
    def test_plaid_negbin_output(self):
        results_dir = "results/Plaid/tests/simple_negbin/run_1"

        self._check_K(results_dir)
        self._check_binary_output(results_dir)

    @attr('fabia')
    def test_fabia_negbin_output(self):
        results_dir = "results/FABIA/tests/simple_negbin/run_1"

        self._check_K(results_dir)
        self._check_approximate_output(results_dir, threshold=0.7)

    @attr('nsnmf')
    def test_nsnmf_negbin_output(self):
        results_dir = "results/nsNMF/tests/simple_negbin/run_1"

        self._check_K(results_dir)
        self._check_approximate_output(results_dir)

    @attr('snmf')
    def test_snmf_negbin_output(self):
        results_dir = "results/SNMF/tests/simple_negbin/run_1"

        self._check_K(results_dir)
        self._check_approximate_output(results_dir)

    @attr('bicmix')
    def test_bicmix_negbin_output(self):
        results_dir = "results/BicMix/tests/simple_negbin/run_1"

        self._check_K(results_dir)
        self._check_approximate_output(results_dir)

    @attr('matlab')
    def test_multicluster_negbin_output(self):
        results_dir = "results/MultiCluster/tests/simple_negbin/run_1"

        self._check_K(results_dir)
        self._check_approximate_output_tensor(results_dir)

    @attr('sda')
    def test_sda_negbin_output(self):
        results_dir = "results/SDA/tests/simple_negbin/run_1"

        self._check_K(results_dir)
        self._check_approximate_output_tensor(results_dir)


class TestMultifactor(unittest.TestCase, BiclusterComparer):
    Y = utils.read_np("data/tests/multifactor", "Y.txt")

    N = utils.read_int_from_file("data/tests/multifactor", "N.txt")

    NtimesC, G = Y.shape
    C = int(NtimesC / N)

    @attr('plaid')
    def test_plaid_multifactor_output(self):
        results_dir = "results/Plaid/tests/multifactor/run_1"
        self._check_binary_shape(results_dir)

    @attr('fabia')
    def test_fabia_multifactor_output(self):
        results_dir = "results/FABIA/tests/multifactor/run_1"
        self._check_shape(results_dir)
        self._check_binary_shape(results_dir)

    @attr('nsnmf')
    def test_nsnmf_multifactor_output(self):
        results_dir = "results/nsNMF/tests/multifactor/run_1"
        self._check_shape(results_dir)
        self._check_binary_shape(results_dir)

    @attr('snmf')
    def test_snmf_multifactor_output(self):
        results_dir = "results/SNMF/tests/multifactor/run_1"
        self._check_shape(results_dir)
        self._check_binary_shape(results_dir)

    @attr('bicmix')
    def test_bicmix_multifactor_output(self):
        results_dir = "results/BicMix/tests/multifactor/run_1"
        self._check_shape(results_dir)
        self._check_binary_shape(results_dir)

    @attr('matlab')
    def test_multicluster_multifactor_output(self):
        results_dir = "results/MultiCluster/tests/multifactor/run_1"
        self._check_tensor_shape(results_dir)

    @attr('sda')
    def test_sda_multifactor_output(self):
        results_dir = "results/SDA/tests/multifactor/run_1"
        self._check_tensor_shape(results_dir)

    @attr('sslb')
    def test_sslb_multifactor_output(self):
        results_dir = "results/SSLB/tests/multifactor/run_1"
        self._check_shape(results_dir)
        self._check_binary_shape(results_dir)


if __name__ == '__main__':
    unittest.main()
