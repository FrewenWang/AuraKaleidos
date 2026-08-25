import numpy as np


class KalmanFilter:
    def __init__(
        self, nb_dynamics: int, nb_measurements: int, nb_controls: int = 0
    ) -> None:
        """
        Initialize the Kalman filter with the given system dimensions.
        Args:
            nb_dynamics (int): Number of dynamic states (state vector size).
            nb_measurements (int): Number of measurement states (measurement vector size).
            nb_controls (int): Number of control states (control vector size), optional.
        """
        self.nb_dynamics = nb_dynamics
        self.nb_measurements = nb_measurements
        self.nb_controls = nb_controls

        if nb_dynamics <= 0 or nb_measurements <= 0:
            raise ValueError(
                "Number of dynamics and measurements must be positive integers."
            )
        if nb_controls < 0:
            raise ValueError("Number of controls cannot be negative.")

        self.__initialize_filter_variables__()

    def predict_step(self, control_vector=None) -> None:
        """
        Perform the prediction step of the Kalman filter.
        Args:
            control_vector (numpy.ndarray): The control input vector (if applicable).
        Returns:
            tuple: The predicted state and predicted error covariance matrix.
        """
        if self.nb_controls and control_vector is not None:
            assert control_vector.shape == (self.nb_controls, 1), (
                f"Control vector shape {control_vector.shape} does not match expected shape ({self.nb_controls}, 1)"
            )
            self.control_vector = control_vector

        if not self.nb_controls:
            # Case of no control parameters provided
            self.predicted_state = self.transition_matrix @ self.post_state

            assert self.predicted_state.shape == (self.nb_dynamics, 1), (
                f"Predicted state: {self.predicted_state.shape}"
            )
        else:
            # Case: control parameters provided
            self.predicted_state = (
                self.transition_matrix @ self.post_state
                + self.control_matrix @ self.control_vector
            )

        self.predicted_err_cov = (
            self.transition_matrix @ self.post_err_cov @ self.transition_matrix.T
            + self.process_noise_cov
        )

        return self.predicted_state, self.predicted_err_cov

    def correct_step(self, new_measurement: np.ndarray) -> np.ndarray:
        """
        Perform the correction step of the Kalman filter.

        Args:
            new_measurement (numpy.ndarray): The new measurement vector.

        Returns:
            numpy.ndarray: The updated post-state estimate.
        """

        assert new_measurement.shape == (self.nb_measurements, 1), (
            f"New measurement shape {new_measurement.shape} does not match expected shape ({self.nb_measurements}, 1)"
        )

        # The error between the prediction and the true measurement
        residual = new_measurement - self.measurement_matrix @ self.predicted_state
        # Residual covariance
        self.residual_cov = (
            self.measurement_matrix @ self.predicted_err_cov @ self.measurement_matrix.T
            + self.measurement_noise_cov
        )

        assert self.residual_cov.shape == (
            self.nb_measurements,
            self.nb_measurements,
        ), f"S shape: {self.residual_cov.shape}"
        # Asserts that the residual_cov is invertible
        if (
            np.linalg.cond(self.residual_cov)
            > 1 / np.finfo(self.residual_cov.dtype).eps
        ):
            raise np.linalg.LinAlgError(
                "Residual covariance matrix is non-invertible or ill-conditioned."
            )

        # Kalman gain
        innovation_cross_cov = self.predicted_err_cov @ self.measurement_matrix.T
        self.gain = np.linalg.solve(self.residual_cov, innovation_cross_cov.T).T
        # Estimate state update
        self.post_state = self.predicted_state + self.gain @ residual
        # Estimate error covariance update
        identity = np.eye(self.nb_dynamics, dtype=self.predicted_err_cov.dtype)
        correction = identity - self.gain @ self.measurement_matrix
        # Joseph form is more numerically stable and preserves positive semidefiniteness.
        self.post_err_cov = (
            correction @ self.predicted_err_cov @ correction.T
            + self.gain @ self.measurement_noise_cov @ self.gain.T
        )

        return self.post_state

    def __initialize_filter_variables__(self):
        """
        Initialize all matrices and variables required for the Kalman filter.
        """

        self.transition_matrix = np.eye(self.nb_dynamics, dtype=np.float32)
        self.measurement_matrix = np.zeros(
            (self.nb_measurements, self.nb_dynamics), dtype=np.float32
        )

        if self.nb_controls:
            self.control_matrix = np.zeros(
                (self.nb_dynamics, self.nb_controls), dtype=np.float32
            )
            self.control_vector = np.zeros((self.nb_controls, 1), dtype=np.float32)
        else:
            self.control_matrix = None
            self.control_vector = None

        self.process_noise_cov = np.eye(self.nb_dynamics, dtype=np.float32)
        self.measurement_noise_cov = np.eye(self.nb_measurements, dtype=np.float32)

        self.predicted_err_cov = np.eye(self.nb_dynamics, dtype=np.float32)
        self.post_err_cov = np.eye(self.nb_dynamics, dtype=np.float32)

        self.predicted_state = np.zeros((self.nb_dynamics, 1), dtype=np.float32)
        self.post_state = np.zeros((self.nb_dynamics, 1), dtype=np.float32)
        self.gain = np.zeros((self.nb_dynamics, self.nb_measurements), dtype=np.float32)
        self.residual_cov = np.eye(self.nb_measurements, dtype=np.float32)
