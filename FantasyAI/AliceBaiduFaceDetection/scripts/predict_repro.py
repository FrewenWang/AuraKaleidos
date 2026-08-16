#!/usr/bin/env python3
"""Run the reproducibly trained model and save an annotated image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import paddle
from _bootstrap import add_source_root

add_source_root()

from alice_face_detection.repro import (  # noqa: E402
    decode_outputs,
    load_config,
    load_model,
    preprocess_image_with_transform,
    raw_model_outputs,
    resolve_from_project,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/repro_cpu.yaml")
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/repro_cpu/best.pdparams")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/repro_cpu/prediction.png")
    )
    parser.add_argument(
        "--confidence", type=float, help="Override the confidence threshold"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_from_project(args.config))
    height, width = map(int, config["data"]["image_size"])
    image_path = resolve_from_project(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Unable to read image: {image_path}")
    processed, transform = preprocess_image_with_transform(
        image, width, height, config["data"].get("resize_mode", "stretch")
    )
    tensor = paddle.to_tensor(processed[None])
    model = load_model(resolve_from_project(args.weights), config)
    with paddle.no_grad():
        result = decode_outputs(
            raw_model_outputs(model, tensor),
            width,
            height,
            args.confidence
            if args.confidence is not None
            else float(config["evaluation"]["confidence_threshold"]),
            float(config["evaluation"]["nms_iou_threshold"]),
            config["model"].get("anchors"),
            config["model"].get("anchor_masks"),
        )[0]
    restored_boxes = transform.restore_boxes(result["boxes"])
    valid = (restored_boxes[:, 2] > restored_boxes[:, 0]) & (
        restored_boxes[:, 3] > restored_boxes[:, 1]
    )
    restored_boxes = restored_boxes[valid]
    restored_scores = result["scores"][valid]
    for box, score in zip(restored_boxes, restored_scores, strict=True):
        x1, y1, x2, y2 = box
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            image,
            f"face {score:.2f}",
            (p1[0], max(14, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )
    output = resolve_from_project(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Unable to write {output}")
    print(f"detections={len(restored_boxes)} output={output}")


if __name__ == "__main__":
    main()
