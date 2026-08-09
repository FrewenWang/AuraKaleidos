"""Image-filter helper functions."""

import numpy as np


def add_pepper_salt_noise(image: np.ndarray, count: int = 10_000, *, seed: int | None = None) -> np.ndarray:
    """Return a copy of *image* with reproducible salt-and-pepper noise.

    Coordinates include image borders and never mutate the caller's array.
    """
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    if count < 0:
        raise ValueError("count must be non-negative")

    result = image.copy()
    height, width = image.shape[:2]
    if height == 0 or width == 0 or count == 0:
        return result

    rng = np.random.default_rng(seed)
    rows = rng.integers(0, height, size=count)
    columns = rng.integers(0, width, size=count)
    values = rng.choice(np.array([0, 255], dtype=result.dtype), size=count)
    result[rows, columns] = values[:, None] if result.ndim > 2 else values
    return result


def add_peppersalt_noise(image: np.ndarray, n: int = 10_000) -> np.ndarray:
    """Backward-compatible alias for historical examples."""
    return add_pepper_salt_noise(image, n)
