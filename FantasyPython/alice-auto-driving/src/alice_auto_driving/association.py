"""Distance metrics used by object-association examples."""

import numpy as np


def euclidean_distance(observations, references):
    observations = np.asarray(observations, dtype=float)
    references = np.asarray(references, dtype=float)
    delta = observations[:, np.newaxis, :] - references[np.newaxis, :, :]
    return np.linalg.norm(delta, axis=2)


def mahalanobis_distance(observations, references, covariance):
    observations = np.asarray(observations, dtype=float)
    references = np.asarray(references, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (observations.shape[1], observations.shape[1]):
        raise ValueError("covariance shape must match the point dimension")
    covariance_inverse = np.linalg.inv(covariance)
    delta = observations[:, np.newaxis, :] - references[np.newaxis, :, :]
    weighted = np.einsum("...i,ij->...j", delta, covariance_inverse)
    return np.sqrt(np.einsum("...i,...i->...", weighted, delta))
