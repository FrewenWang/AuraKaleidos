#!/usr/bin/env python3
"""调用本地 YOLOv3 Flask 检测服务。"""

import argparse
import os
from pathlib import Path

import requests

PYTORCH_REST_API_URL = os.getenv(
    "PYTORCH_REST_API_URL", "http://127.0.0.1:5000/predict"
)
DEFAULT_IMAGE = (
    Path(__file__).resolve().parent
    / ".."
    / "deploy-pytorch-model-master"
    / "dog.jpg"
).resolve()


def predict_result(image_path):
    """上传图片并打印检测结果。"""
    with Path(image_path).open("rb") as image_file:
        response = requests.post(
            PYTORCH_REST_API_URL,
            files={"image": image_file},
            timeout=60,
        )
    response.raise_for_status()
    result = response.json()
    if not result.get("success"):
        raise RuntimeError(result.get("error", "检测失败"))

    for detection in result["predictions"]:
        print(
            f"  {detection['class']}: "
            f"x={detection['x']:.1f} y={detection['y']:.1f} "
            f"w={detection['w']:.1f} h={detection['h']:.1f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"测试图片路径（默认: {DEFAULT_IMAGE}）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.file.is_file():
        raise SystemExit(f"错误: 图片不存在 — {args.file}")
    print(f"检测图片: {args.file}")
    predict_result(args.file)


if __name__ == "__main__":
    main()
