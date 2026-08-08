import unittest

import numpy as np

from alice_auto_driving.ransac import generate_data, least_squares_fit, ransac_fit


class RansacTest(unittest.TestCase):
    def test_least_squares_recovers_exact_curve(self):
        x = np.linspace(-2, 2, 20)
        y = 0.5 * x**3 - 2 * x**2 + x + 3
        np.testing.assert_allclose(least_squares_fit(x, y), [0.5, -2, 1, 3], atol=1e-10)

    def test_seeded_ransac_finds_most_inliers(self):
        x, y, _ = generate_data(seed=8, outlier_ratio=0.1)
        theta, inliers = ransac_fit(x, y, max_iters=200, threshold=1.5, seed=8)
        self.assertIsNotNone(theta)
        self.assertGreaterEqual(int(inliers.sum()), 75)

    def test_rejects_too_few_samples(self):
        with self.assertRaises(ValueError):
            ransac_fit([1, 2], [1, 2])


if __name__ == "__main__":
    unittest.main()
