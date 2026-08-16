#!/usr/bin/env python3
"""Compare an exported ONNX model with the Paddle dynamic model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import paddle
from _bootstrap import add_source_root

add_source_root()

from export_repro import RawDetector

from alice_face_detection.repro import (  # noqa: E402
    load_config,
    load_model,
    preprocess_image,
    resolve_from_project,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/repro_cpu.yaml")
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/repro_cpu/best.pdparams")
    )
    parser.add_argument(
        "--onnx", type=Path, default=Path("outputs/repro_cpu/export/model.onnx")
    )
    parser.add_argument(
        "--verify-image",
        type=Path,
        help="Optional image used for numerical verification",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def verification_input(args: argparse.Namespace, config: dict) -> np.ndarray:
    height, width = map(int, config["data"]["image_size"])
    if args.verify_image:
        image_path = resolve_from_project(args.verify_image)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"Unable to read verification image: {image_path}")
        return preprocess_image(
            image, width, height, config["data"].get("resize_mode", "stretch")
        )[None]
    rng = np.random.default_rng(int(config["seed"]))
    return rng.normal(size=(1, 3, height, width)).astype(np.float32)


def main() -> None:
    args = parse_args()
    config = load_config(resolve_from_project(args.config))
    inputs = verification_input(args, config)

    detector = load_model(resolve_from_project(args.weights), config)
    model = RawDetector(detector)
    model.eval()
    with paddle.no_grad():
        paddle_outputs = [
            output.numpy() for output in model(paddle.to_tensor(inputs))
        ]

    session = ort.InferenceSession(
        str(resolve_from_project(args.onnx)), providers=["CPUExecutionProvider"]
    )
    onnx_outputs = session.run(None, {session.get_inputs()[0].name: inputs})
    if [output.shape for output in paddle_outputs] != [
        output.shape for output in onnx_outputs
    ]:
        raise RuntimeError(
            "ONNX output shapes differ from Paddle: "
            f"paddle={[x.shape for x in paddle_outputs]}, onnx={[x.shape for x in onnx_outputs]}"
        )
    errors = [
        float(np.max(np.abs(paddle_output - onnx_output)))
        for paddle_output, onnx_output in zip(
            paddle_outputs, onnx_outputs, strict=True
        )
    ]
    if max(errors) > args.atol:
        raise RuntimeError(
            f"ONNX export mismatch: max_abs_errors={errors}, atol={args.atol}"
        )
    print(f"ONNX model: {resolve_from_project(args.onnx)}")
    print(f"output_shapes={[list(output.shape) for output in onnx_outputs]}")
    print(f"max_abs_errors={errors}")


if __name__ == "__main__":
    main()
