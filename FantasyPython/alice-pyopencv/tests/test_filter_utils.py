import unittest

import numpy as np

from alice_pyopencv.filter_utils import add_pepper_salt_noise


class PepperSaltNoiseTest(unittest.TestCase):
    def test_does_not_mutate_input(self):
        image = np.full((4, 4), 127, dtype=np.uint8)
        result = add_pepper_salt_noise(image, 8, seed=7)
        np.testing.assert_array_equal(image, np.full((4, 4), 127, dtype=np.uint8))
        self.assertTrue(np.isin(result, [0, 127, 255]).all())

    def test_rejects_negative_count(self):
        with self.assertRaises(ValueError):
            add_pepper_salt_noise(np.zeros((2, 2), dtype=np.uint8), -1)


if __name__ == "__main__":
    unittest.main()
