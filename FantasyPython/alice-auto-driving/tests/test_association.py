import unittest

import numpy as np

from alice_auto_driving.association import euclidean_distance, mahalanobis_distance


class AssociationTest(unittest.TestCase):
    def test_euclidean_distance_matrix(self):
        result = euclidean_distance([[0, 0]], [[3, 4], [0, 2]])
        np.testing.assert_allclose(result, [[5, 2]])

    def test_mahalanobis_accounts_for_covariance(self):
        result = mahalanobis_distance([[2, 1]], [[0, 0]], [[4, 0], [0, 1]])
        np.testing.assert_allclose(result, [[np.sqrt(2)]])

    def test_rejects_invalid_covariance(self):
        with self.assertRaises(ValueError):
            mahalanobis_distance([[1, 2]], [[0, 0]], [[1]])


if __name__ == "__main__":
    unittest.main()
