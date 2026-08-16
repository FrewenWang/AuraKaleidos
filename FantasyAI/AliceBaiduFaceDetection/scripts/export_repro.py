#!/usr/bin/env python3
"""Export raw YOLO-head outputs to a portable Paddle inference model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import paddle
from _bootstrap import add_source_root

add_source_root()

from alice_face_detection.repro import (  # noqa: E402
    load_config,
    load_model,
    preprocess_image,
    resolve_from_project,
)


class RawDetector(paddle.nn.Layer):
    """Export-friendly detector without Python-side decode or NMS."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, images):
        features = self.detector.backbone(images)
        features = self.detector.neck(features)
        outputs = self.detector.yolo_head(features)
        return outputs[0], outputs[1], outputs[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/repro_cpu.yaml")
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/repro_cpu/best.pdparams")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/repro_cpu/export/model")
    )
    parser.add_argument(
        "--verify-image",
        type=Path,
        help="Optional image used for numerical verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_from_project(args.config))
    height, width = map(int, config["data"]["image_size"])
    detector = load_model(resolve_from_project(args.weights), config)
    model = RawDetector(detector)
    model.eval()
    input_spec = [
        paddle.static.InputSpec(
            shape=[None, 3, height, width], dtype="float32", name="images"
        )
    ]
    static_model = paddle.jit.to_static(
        model, input_spec=input_spec, full_graph=True
    )
    output_prefix = resolve_from_project(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paddle.jit.save(static_model, str(output_prefix))

    if args.verify_image:
        image_path = resolve_from_project(args.verify_image)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"Unable to read verification image: {image_path}")
        inputs = paddle.to_tensor(
            preprocess_image(
                image,
                width,
                height,
                config["data"].get("resize_mode", "stretch"),
            )[None]
        )
    else:
        rng = np.random.default_rng(int(config["seed"]))
        inputs = paddle.to_tensor(
            rng.normal(size=(1, 3, height, width)).astype(np.float32)
        )

    loaded = paddle.jit.load(str(output_prefix))
    loaded.eval()
    with paddle.no_grad():
        dynamic_outputs = model(inputs)
        static_outputs = loaded(inputs)
    errors = [
        float(np.max(np.abs(dynamic.numpy() - static.numpy())))
        for dynamic, static in zip(dynamic_outputs, static_outputs, strict=True)
    ]
    if max(errors) > 1e-5:
        raise RuntimeError(f"Static export mismatch: max_abs_errors={errors}")
    print(f"Paddle inference model: {output_prefix}")
    print(f"output_shapes={[list(output.shape) for output in static_outputs]}")
    print(f"max_abs_errors={errors}")


if __name__ == "__main__":
    main()
