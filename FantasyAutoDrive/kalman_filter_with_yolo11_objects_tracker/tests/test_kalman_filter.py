import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai.tracker.kalman_filter import KalmanFilter


def make_constant_velocity_filter():
    kalman = KalmanFilter(nb_dynamics=4, nb_measurements=2)
    kalman.transition_matrix = np.array(
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float32
    )
    kalman.measurement_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    kalman.process_noise_cov = np.eye(4, dtype=np.float32) * 1e-3
    kalman.measurement_noise_cov = np.eye(2, dtype=np.float32) * 1e-2
    kalman.post_state = np.array([[10], [2], [20], [-1]], dtype=np.float32)
    return kalman


class KalmanFilterTest(unittest.TestCase):
    def test_prediction_is_deterministic(self):
        first = make_constant_velocity_filter()
        second = make_constant_velocity_filter()

        first_state, _ = first.predict_step()
        second_state, _ = second.predict_step()

        np.testing.assert_array_equal(first_state, second_state)
        np.testing.assert_array_equal(
            first_state, np.array([[12], [2], [19], [-1]], dtype=np.float32)
        )

    def test_correction_remains_finite_and_reduces_position_error(self):
        kalman = make_constant_velocity_filter()
        predicted, _ = kalman.predict_step()
        measurement = np.array([[11], [20]], dtype=np.float32)
        before = np.linalg.norm(measurement - kalman.measurement_matrix @ predicted)

        corrected = kalman.correct_step(measurement)
        after = np.linalg.norm(measurement - kalman.measurement_matrix @ corrected)

        self.assertTrue(np.isfinite(corrected).all())
        self.assertLess(after, before)
        np.testing.assert_allclose(
            kalman.post_err_cov, kalman.post_err_cov.T, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
