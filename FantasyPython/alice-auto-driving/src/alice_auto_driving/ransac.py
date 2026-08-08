"""Cubic least-squares and RANSAC fitting helpers."""

import numpy as np


def _design_matrix(x):
    x = np.asarray(x, dtype=float)
    return np.column_stack([x**3, x**2, x, np.ones_like(x)])


def least_squares_fit(x, y):
    return np.linalg.lstsq(_design_matrix(x), np.asarray(y, dtype=float), rcond=None)[0]


def ransac_fit(x, y, max_iters=100, threshold=0.5, min_samples=4, seed=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size < min_samples:
        raise ValueError("not enough samples")

    random = np.random.default_rng(seed)
    best_inliers = None
    for _ in range(max_iters):
        indices = random.choice(x.size, min_samples, replace=False)
        theta = least_squares_fit(x[indices], y[indices])
        prediction = _design_matrix(x) @ theta
        inliers = np.abs(prediction - y) < threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < min_samples:
        return None, None
    return least_squares_fit(x[best_inliers], y[best_inliers]), best_inliers


def generate_data(seed=123, outlier_ratio=0.1):
    random = np.random.default_rng(seed)
    x = np.linspace(-3, 5, 100)
    y_true = 0.5 * x**3 - 2 * x**2 + x + 3
    noise = random.normal(0, 0.5, size=x.shape)
    outliers = random.random(x.size) < outlier_ratio
    y = y_true + noise + outliers * random.normal(0, 10, size=x.shape)
    return x, y, y_true
