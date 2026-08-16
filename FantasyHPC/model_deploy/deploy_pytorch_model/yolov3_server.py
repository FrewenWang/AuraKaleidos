#!/usr/bin/env python3
"""提供 YOLOv3 检测和 ResNet50 分类的 Flask 服务。"""

import argparse
import ast
import io
import os
import sys
from pathlib import Path

import cv2
import flask
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

PROJECT_DIR = (
    Path(__file__).resolve().parent / ".." / "PyTorch-YOLO-V3"
).resolve()
sys.path.insert(0, str(PROJECT_DIR))

from models import Darknet  # noqa: E402
from utils.utils import load_classes, non_max_suppression  # noqa: E402

app = flask.Flask(__name__)
use_cuda = torch.cuda.is_available()
TENSOR_TYPE = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

detect_model = None
classify_model = None
classes = None


def load_classify_model():
    """从 torchvision 加载预训练 ResNet50 分类模型。"""
    global classify_model

    try:
        from torchvision.models import ResNet50_Weights

        classify_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    except ImportError:  # 兼容旧版 torchvision
        classify_model = resnet50(pretrained=True)
    classify_model.eval()
    if use_cuda:
        classify_model.cuda()
    print("ResNet50 分类模型加载完成")


def load_detect_model(config_path=None, weights_path=None, class_path=None):
    """从本地配置和权重加载 YOLOv3 检测模型。"""
    global classes, detect_model

    config_path = Path(
        config_path
        or os.getenv("YOLO_CONFIG", PROJECT_DIR / "config" / "yolov3.cfg")
    )
    weights_path = Path(
        weights_path
        or os.getenv("YOLO_WEIGHTS", PROJECT_DIR / "weights" / "yolov3.weights")
    )
    class_path = Path(
        class_path
        or os.getenv("YOLO_CLASSES", PROJECT_DIR / "data" / "coco.names")
    )

    missing_paths = [
        str(path)
        for path in (config_path, weights_path, class_path)
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "缺少 YOLOv3 运行文件：" + ", ".join(missing_paths)
        )

    classes = load_classes(str(class_path))
    detect_model = Darknet(str(config_path))
    detect_model.load_darknet_weights(str(weights_path))
    if use_cuda:
        detect_model.cuda()
    detect_model.eval()
    print(f"YOLOv3 检测模型加载完成，类别数: {len(classes)}")


def numpy_to_tensor(image_array, image_size):
    """将 RGB NumPy 图像转换为 YOLOv3 输入张量。"""
    height, width, _ = image_array.shape
    dimension_difference = abs(height - width)
    padding_start = dimension_difference // 2
    padding_end = dimension_difference - padding_start
    padding = (
        ((padding_start, padding_end), (0, 0), (0, 0))
        if height <= width
        else ((0, 0), (padding_start, padding_end), (0, 0))
    )
    input_image = (
        np.pad(image_array, padding, "constant", constant_values=127.5) / 255.0
    )
    input_image = cv2.resize(input_image, (image_size, image_size))
    input_image = np.transpose(input_image, (2, 0, 1))
    return torch.from_numpy(input_image).float()


def yolo_detection(image_array, image_size=416):
    """使用 YOLOv3 检测图像中的目标。"""
    image_array = np.asarray(image_array)
    image_tensor = numpy_to_tensor(image_array, image_size)
    image_tensor = image_tensor.type(TENSOR_TYPE).unsqueeze(0)

    with torch.no_grad():
        detections = detect_model(image_tensor)
        detections = non_max_suppression(detections)

    padding_x = max(image_array.shape[0] - image_array.shape[1], 0) * (
        image_size / max(image_array.shape)
    )
    padding_y = max(image_array.shape[1] - image_array.shape[0], 0) * (
        image_size / max(image_array.shape)
    )
    unpadded_height = image_size - padding_y
    unpadded_width = image_size - padding_x

    results = []
    if detections and detections[0] is not None:
        for (
            x1,
            y1,
            x2,
            y2,
            _confidence,
            _class_confidence,
            class_id,
        ) in detections[0]:
            box_height = (y2 - y1) / unpadded_height * image_array.shape[0]
            box_width = (x2 - x1) / unpadded_width * image_array.shape[1]
            y1 = (y1 - padding_y // 2) / unpadded_height * image_array.shape[0]
            x1 = (x1 - padding_x // 2) / unpadded_width * image_array.shape[1]
            results.append(
                {
                    "class": classes[int(class_id)],
                    "x": x1.item(),
                    "y": y1.item(),
                    "h": box_height.item(),
                    "w": box_width.item(),
                }
            )
    return results


IMAGENET_CLASS_PATH = Path(__file__).with_name("imagenet_class.txt")
if IMAGENET_CLASS_PATH.exists():
    with IMAGENET_CLASS_PATH.open(encoding="utf-8") as class_file:
        IMAGENET_CLASSES = ast.literal_eval(class_file.read())
else:
    IMAGENET_CLASSES = {}

CLASSIFY_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def classify_image(image_array):
    """使用 ResNet50 返回图像的 Top-5 分类结果。"""
    image = Image.fromarray(np.asarray(image_array)).convert("RGB")
    image_tensor = CLASSIFY_TRANSFORM(image).unsqueeze(0)
    if use_cuda:
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        outputs = classify_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probabilities, top_indices = torch.topk(probabilities, 5)
    results = []
    for probability, class_index in zip(
        top_probabilities, top_indices, strict=True
    ):
        index = class_index.item()
        results.append(
            {
                "class": IMAGENET_CLASSES.get(index, f"class_{index}"),
                "probability": round(probability.item(), 4),
            }
        )
    return results


def read_request_image():
    """读取并校验 Flask 请求中的图片。"""
    image_file = flask.request.files.get("image")
    if not image_file:
        raise ValueError("请求中缺少 image 文件")
    try:
        return Image.open(io.BytesIO(image_file.read())).convert("RGB")
    except OSError as error:
        raise ValueError(f"无法读取图片：{error}") from error


@app.route("/detect", methods=["POST"])
def detect():
    """YOLOv3 目标检测接口。"""
    if detect_model is None:
        return flask.jsonify(success=False, error="检测模型未加载"), 503
    try:
        predictions = yolo_detection(read_request_image())
    except ValueError as error:
        return flask.jsonify(success=False, error=str(error)), 400
    return flask.jsonify(success=True, predictions=predictions)


@app.route("/classify", methods=["POST"])
def classify():
    """ResNet50 图像分类接口。"""
    if classify_model is None:
        return flask.jsonify(success=False, error="分类模型未加载"), 503
    try:
        predictions = classify_image(np.asarray(read_request_image()))
    except ValueError as error:
        return flask.jsonify(success=False, error=str(error)), 400
    return flask.jsonify(success=True, predictions=predictions)


@app.route("/predict", methods=["POST"])
def predict():
    """兼容旧接口，默认使用 YOLOv3 检测。"""
    return detect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="detect",
        choices=("detect", "classify", "all"),
        help="启动时加载的模型，默认只加载 YOLOv3",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.models in ("detect", "all"):
        load_detect_model()
    if args.models in ("classify", "all"):
        load_classify_model()
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
