#!/usr/bin/env python3
"""Fit YOLO anchors to a JSONL training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _bootstrap import add_source_root
from PIL import Image

add_source_root()

from alice_face_detection.anchors import (  # noqa: E402
    anchor_quality,
    fit_anchors,
)
from alice_face_detection.repro import (  # noqa: E402
    load_config,
    resize_transform,
    resolve_from_project,
    sanitize_boxes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/wider_cpu.yaml")
    )
    parser.add_argument("--clusters", type=int, default=9)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wider_cpu/anchors.json")
    )
    return parser.parse_args()


def collect_box_sizes(
    manifest: Path, image_size: list[int], resize_mode: str
) -> np.ndarray:
    height, width = map(int, image_size)
    sizes = []
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = manifest.parent / record["image"]
            with Image.open(image_path) as image:
                original_width, original_height = image.size
            boxes = sanitize_boxes(
                record.get("boxes", []), original_width, original_height
            )
            transform = resize_transform(
                original_width, original_height, width, height, resize_mode
            )
            transformed = transform.apply_boxes(boxes)
            if len(transformed):
                sizes.append(transformed[:, 2:] - transformed[:, :2])
    if not sizes:
        raise ValueError(f"No valid boxes found in {manifest}")
    return np.concatenate(sizes).astype(np.float32)


def main() -> None:
    args = parse_args()
    config = load_config(resolve_from_project(args.config))
    data = config["data"]
    manifest = resolve_from_project(data["root"]) / "train.jsonl"
    sizes = collect_box_sizes(
        manifest, data["image_size"], data.get("resize_mode", "stretch")
    )
    fitted = fit_anchors(sizes, args.clusters, int(config["seed"]))
    fitted_integer = np.maximum(1, np.rint(fitted)).astype(int)
    current = np.asarray(config["model"]["anchors"], dtype=np.float32)
    result = {
        "boxes": len(sizes),
        "image_size": data["image_size"],
        "resize_mode": data.get("resize_mode", "stretch"),
        "current": {
            "anchors": current.astype(int).tolist(),
            **anchor_quality(sizes, current),
        },
        "fitted": {
            "anchors": fitted_integer.tolist(),
            **anchor_quality(sizes, fitted_integer),
        },
    }
    output = resolve_from_project(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
