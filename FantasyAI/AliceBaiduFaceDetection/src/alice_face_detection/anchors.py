"""Dataset-driven YOLO anchor fitting."""

from __future__ import annotations

import numpy as np


def size_iou(box_sizes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    intersection = np.minimum(box_sizes[:, None], anchors[None]).prod(axis=2)
    union = (
        box_sizes.prod(axis=1)[:, None]
        + anchors.prod(axis=1)[None]
        - intersection
    )
    return intersection / np.maximum(union, 1e-9)


def anchor_quality(
    box_sizes: np.ndarray, anchors: np.ndarray
) -> dict[str, float]:
    best = size_iou(box_sizes, anchors).max(axis=1)
    return {
        "mean_best_iou": float(best.mean()),
        "recall_at_0.5": float((best >= 0.5).mean()),
        "recall_at_0.75": float((best >= 0.75).mean()),
    }


def fit_anchors(
    box_sizes: np.ndarray,
    clusters: int = 9,
    seed: int = 0,
    iterations: int = 100,
) -> np.ndarray:
    """Fit YOLO anchors with IoU-distance k-means and median centers."""
    boxes = np.asarray(box_sizes, dtype=np.float32).reshape(-1, 2)
    boxes = boxes[np.isfinite(boxes).all(axis=1) & (boxes > 0).all(axis=1)]
    if len(boxes) < clusters:
        raise ValueError(
            f"Need at least {clusters} valid boxes, got {len(boxes)}"
        )
    rng = np.random.default_rng(seed)
    centers = [boxes[int(rng.integers(len(boxes)))]]
    while len(centers) < clusters:
        distance = 1.0 - size_iou(boxes, np.asarray(centers)).max(axis=1)
        probability = distance**2
        if probability.sum() == 0:
            centers.append(boxes[int(rng.integers(len(boxes)))])
        else:
            centers.append(
                boxes[
                    int(
                        rng.choice(
                            len(boxes), p=probability / probability.sum()
                        )
                    )
                ]
            )
    anchors = np.asarray(centers, dtype=np.float32)

    previous = None
    for _ in range(iterations):
        assignment = np.argmax(size_iou(boxes, anchors), axis=1)
        if previous is not None and np.array_equal(assignment, previous):
            break
        previous = assignment.copy()
        for cluster in range(clusters):
            members = boxes[assignment == cluster]
            if len(members):
                anchors[cluster] = np.median(members, axis=0)
            else:
                anchors[cluster] = boxes[
                    np.argmin(size_iou(boxes, anchors).max(axis=1))
                ]
    return anchors[np.argsort(anchors.prod(axis=1))]
