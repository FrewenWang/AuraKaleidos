#!/usr/bin/env python3
"""Create a deterministic face-detection dataset from public face crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from _bootstrap import add_source_root

add_source_root()

from alice_face_detection.repro import (  # noqa: E402
    load_config,
    resolve_from_project,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/repro_cpu.yaml")
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated dataset",
    )
    return parser.parse_args()


def procedural_faces(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    faces = []
    for _ in range(count):
        image = np.full((64, 64), rng.uniform(0.18, 0.38), dtype=np.float32)
        skin = float(rng.uniform(0.52, 0.88))
        cv2.ellipse(image, (32, 33), (22, 27), 0, 0, 360, skin, -1)
        eye_y = int(rng.integers(25, 31))
        eye_dx = int(rng.integers(8, 12))
        for x in (32 - eye_dx, 32 + eye_dx):
            cv2.circle(image, (x, eye_y), 3, float(rng.uniform(0.05, 0.2)), -1)
        cv2.ellipse(
            image, (32, 44), (8, 3), 0, 0, 180, float(rng.uniform(0.1, 0.35)), 2
        )
        faces.append(np.clip(image, 0, 1))
    return np.asarray(faces, dtype=np.float32)


def load_faces(
    data_root: Path, source: str, seed: int
) -> tuple[np.ndarray, str]:
    if source in {"auto", "olivetti"}:
        try:
            from sklearn.datasets import fetch_olivetti_faces

            dataset = fetch_olivetti_faces(
                data_home=str(data_root / "source"),
                shuffle=False,
                download_if_missing=True,
            )
            return dataset.images.astype(
                np.float32
            ), "Olivetti faces via scikit-learn"
        except Exception as error:
            if source == "olivetti":
                raise
            print(
                f"Olivetti download failed ({error}); using procedural fallback."
            )
    return procedural_faces(400, seed), "procedural fallback"


def background(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    small = rng.uniform(
        0.08, 0.75, size=(max(2, height // 16), max(2, width // 16))
    ).astype(np.float32)
    result = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    result += rng.normal(0, 0.035, size=result.shape).astype(np.float32)
    return np.clip(result, 0, 1)


def overlaps(candidate: list[int], boxes: list[list[int]]) -> bool:
    x1, y1, x2, y2 = candidate
    for bx1, by1, bx2, by2 in boxes:
        intersection = max(0, min(x2, bx2) - max(x1, bx1)) * max(
            0, min(y2, by2) - max(y1, by1)
        )
        smaller = min((x2 - x1) * (y2 - y1), (bx2 - bx1) * (by2 - by1))
        if intersection / max(smaller, 1) > 0.15:
            return True
    return False


def make_scene(
    rng: np.random.Generator,
    faces: np.ndarray,
    height: int,
    width: int,
    min_faces: int,
    max_faces: int,
) -> tuple[np.ndarray, list[list[int]]]:
    canvas = background(rng, height, width)
    boxes: list[list[int]] = []
    for _ in range(int(rng.integers(min_faces, max_faces + 1))):
        face = faces[int(rng.integers(0, len(faces)))]
        face_height = int(
            rng.integers(max(20, height // 4), max(22, height // 2))
        )
        face_width = int(face_height * rng.uniform(0.78, 1.02))
        resized = cv2.resize(
            face, (face_width, face_height), interpolation=cv2.INTER_CUBIC
        )
        if rng.random() < 0.5:
            resized = resized[:, ::-1]
        resized = np.clip(
            (resized - 0.5) * rng.uniform(0.75, 1.25) + rng.uniform(0.35, 0.65),
            0,
            1,
        )

        placed = False
        for _attempt in range(30):
            x1 = int(rng.integers(1, max(2, width - face_width - 1)))
            y1 = int(rng.integers(1, max(2, height - face_height - 1)))
            candidate = [x1, y1, x1 + face_width, y1 + face_height]
            if not overlaps(candidate, boxes):
                placed = True
                break
        if not placed:
            continue

        mask = np.zeros((face_height, face_width), dtype=np.float32)
        cv2.ellipse(
            mask,
            (face_width // 2, face_height // 2),
            (face_width // 2, face_height // 2),
            0,
            0,
            360,
            1.0,
            -1,
        )
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        region = canvas[y1 : y1 + face_height, x1 : x1 + face_width]
        canvas[y1 : y1 + face_height, x1 : x1 + face_width] = (
            region * (1 - mask) + resized * mask
        )
        boxes.append(candidate)
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8), boxes


def write_split(
    output_root: Path,
    split: str,
    faces: np.ndarray,
    count: int,
    seed: int,
    image_size: list[int],
    min_faces: int,
    max_faces: int,
) -> None:
    rng = np.random.default_rng(seed)
    image_dir = output_root / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_root / f"{split}.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for index in range(count):
            image, boxes = make_scene(
                rng, faces, image_size[0], image_size[1], min_faces, max_faces
            )
            relative = Path("images") / split / f"{index:05d}.png"
            if not cv2.imwrite(str(output_root / relative), image):
                raise RuntimeError(f"Unable to write {output_root / relative}")
            stream.write(
                json.dumps({"image": relative.as_posix(), "boxes": boxes})
                + "\n"
            )


def main() -> None:
    args = parse_args()
    config_path = resolve_from_project(args.config)
    config = load_config(config_path)
    data = config["data"]
    output_root = resolve_from_project(data["root"])
    if output_root.exists() and not args.force:
        raise SystemExit(
            f"Dataset already exists: {output_root}. Pass --force to regenerate it."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    all_faces, source_name = load_faces(
        output_root, data.get("source", "auto"), int(config["seed"])
    )
    # Olivetti is ordered as 40 people x 10 images. Keep identities disjoint.
    split = int(len(all_faces) * 0.8)
    train_faces, val_faces = all_faces[:split], all_faces[split:]
    write_split(
        output_root,
        "train",
        train_faces,
        int(data["train_samples"]),
        int(config["seed"]),
        data["image_size"],
        int(data["min_faces"]),
        int(data["max_faces"]),
    )
    write_split(
        output_root,
        "val",
        val_faces,
        int(data["val_samples"]),
        int(config["seed"]) + 1,
        data["image_size"],
        int(data["min_faces"]),
        int(data["max_faces"]),
    )
    summary = {
        "source": source_name,
        "seed": int(config["seed"]),
        "image_size": data["image_size"],
        "train_samples": int(data["train_samples"]),
        "val_samples": int(data["val_samples"]),
        "note": "Synthetic scenes for pipeline validation; use WIDER FACE or private production data for deployment quality.",
    }
    (output_root / "dataset_info.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
