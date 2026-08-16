"""Shared, side-effect-free utilities for the reproducible training pipeline."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import paddle
import yaml

from .PPYoloMobileNetV3 import DEFAULT_ANCHOR_MASKS, DEFAULT_ANCHORS, PPYoloTiny

PROJECT_ROOT = Path(__file__).resolve().parents[2]


ANCHORS = np.asarray(DEFAULT_ANCHORS, dtype=np.float32)
ANCHOR_MASKS = DEFAULT_ANCHOR_MASKS
DOWNSAMPLE_RATIOS = [32, 16, 8]
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class ResizeTransform:
    """Geometry needed to map boxes between an original image and model input."""

    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale_x: float
    scale_y: float
    pad_x: float = 0.0
    pad_y: float = 0.0

    def apply_boxes(self, boxes: np.ndarray) -> np.ndarray:
        result = boxes.astype(np.float32, copy=True).reshape(-1, 4)
        result[:, [0, 2]] = result[:, [0, 2]] * self.scale_x + self.pad_x
        result[:, [1, 3]] = result[:, [1, 3]] * self.scale_y + self.pad_y
        return sanitize_boxes(result, self.input_width, self.input_height)

    def restore_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Restore boxes without changing their order or count."""
        result = boxes.astype(np.float32, copy=True).reshape(-1, 4)
        result[:, [0, 2]] = (result[:, [0, 2]] - self.pad_x) / self.scale_x
        result[:, [1, 3]] = (result[:, [1, 3]] - self.pad_y) / self.scale_y
        result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, self.original_width)
        result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, self.original_height)
        return result


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def model_kwargs(config: dict) -> dict:
    """Validate and return the model geometry shared by training and decode."""
    model = config["model"]
    anchors = model.get("anchors", DEFAULT_ANCHORS)
    anchor_masks = model.get("anchor_masks", DEFAULT_ANCHOR_MASKS)
    num_classes = int(model.get("num_classes", 1))
    if num_classes != 1:
        raise ValueError(
            "The reproducible face pipeline currently supports exactly one class"
        )
    if len(anchor_masks) != len(DOWNSAMPLE_RATIOS):
        raise ValueError(
            f"Expected {len(DOWNSAMPLE_RATIOS)} anchor masks, got {len(anchor_masks)}"
        )
    if any(len(mask) != len(anchor_masks[0]) for mask in anchor_masks):
        raise ValueError(
            "Every detection level must use the same number of anchors"
        )
    anchor_array = np.asarray(anchors, dtype=np.float32)
    if (
        anchor_array.ndim != 2
        or anchor_array.shape[1] != 2
        or np.any(anchor_array <= 0)
    ):
        raise ValueError("anchors must be positive [width, height] pairs")
    if any(
        index < 0 or index >= len(anchors)
        for mask in anchor_masks
        for index in mask
    ):
        raise ValueError(
            "anchor_masks contains an index outside the anchors list"
        )
    return {
        "model": model["variant"],
        "num_classes": num_classes,
        "anchors": anchor_array.tolist(),
        "anchor_masks": anchor_masks,
    }


def resolve_from_project(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def sanitize_boxes(
    boxes: np.ndarray | list, width: int, height: int
) -> np.ndarray:
    """Return finite, clipped, positive-area ``xyxy`` boxes with shape ``[N, 4]``."""
    result = np.asarray(boxes, dtype=np.float32)
    if result.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if result.size % 4:
        raise ValueError(
            f"Bounding boxes must contain groups of four coordinates, got {result.shape}"
        )
    result = result.reshape(-1, 4)
    result = result[np.isfinite(result).all(axis=1)]
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, width)
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, height)
    return result[(result[:, 2] > result[:, 0]) & (result[:, 3] > result[:, 1])]


def horizontal_flip(
    image: np.ndarray, boxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flip an image and ``xyxy`` boxes using the original image width."""
    width = image.shape[1]
    flipped_boxes = boxes.astype(np.float32, copy=True).reshape(-1, 4)
    if len(flipped_boxes):
        old_x1 = flipped_boxes[:, 0].copy()
        flipped_boxes[:, 0] = width - flipped_boxes[:, 2]
        flipped_boxes[:, 2] = width - old_x1
    return np.ascontiguousarray(image[:, ::-1]), flipped_boxes


def resize_image(
    image: np.ndarray,
    width: int,
    height: int,
    mode: str = "stretch",
) -> tuple[np.ndarray, ResizeTransform]:
    """Resize an image and return the exact forward/inverse box transform."""
    original_height, original_width = image.shape[:2]
    transform = resize_transform(
        original_width, original_height, width, height, mode
    )
    if mode == "stretch":
        return cv2.resize(
            image, (width, height), interpolation=cv2.INTER_AREA
        ), transform

    resized_width = int(round(original_width * transform.scale_x))
    resized_height = int(round(original_height * transform.scale_y))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    left, top = int(transform.pad_x), int(transform.pad_y)
    border = 114 if image.ndim == 2 else (114,) * image.shape[2]
    output = cv2.copyMakeBorder(
        resized,
        top,
        height - resized_height - top,
        left,
        width - resized_width - left,
        cv2.BORDER_CONSTANT,
        value=border,
    )
    return output, transform


def resize_transform(
    original_width: int,
    original_height: int,
    width: int,
    height: int,
    mode: str = "stretch",
) -> ResizeTransform:
    """Build resize geometry without decoding or resizing an image."""
    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            f"Invalid image size: {original_width}x{original_height}"
        )
    if mode == "stretch":
        return ResizeTransform(
            original_width,
            original_height,
            width,
            height,
            width / original_width,
            height / original_height,
        )
    if mode != "letterbox":
        raise ValueError(
            f"Unsupported resize mode: {mode!r}; use 'stretch' or 'letterbox'"
        )

    scale = min(width / original_width, height / original_height)
    resized_width = max(1, min(width, int(round(original_width * scale))))
    resized_height = max(1, min(height, int(round(original_height * scale))))
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    return ResizeTransform(
        original_width,
        original_height,
        width,
        height,
        resized_width / original_width,
        resized_height / original_height,
        float(left),
        float(top),
    )


def xyxy_to_normalized_xywh(
    boxes: np.ndarray, width: int, height: int
) -> np.ndarray:
    result = np.zeros_like(boxes, dtype=np.float32)
    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5 / width
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5 / height
    result[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
    result[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
    return np.clip(result, 0.0, 1.0)


def _size_iou(box_wh: np.ndarray, anchor_wh: np.ndarray) -> float:
    intersection = np.minimum(box_wh, anchor_wh).prod()
    union = box_wh.prod() + anchor_wh.prod() - intersection
    return float(intersection / max(union, 1e-9))


def build_yolo_targets(
    boxes_xywh: np.ndarray,
    width: int,
    height: int,
    anchors: np.ndarray | list = ANCHORS,
    anchor_masks: list[list[int]] = ANCHOR_MASKS,
) -> list[np.ndarray]:
    """Build targets expected by the historical YOLOv3Loss implementation."""
    anchors = np.asarray(anchors, dtype=np.float32)
    targets = []
    for mask, downsample in zip(anchor_masks, DOWNSAMPLE_RATIOS, strict=True):
        targets.append(
            np.zeros(
                (len(mask), 7, height // downsample, width // downsample),
                dtype=np.float32,
            )
        )

    for gx, gy, gw, gh in boxes_xywh:
        if gw <= 0 or gh <= 0:
            continue
        pixel_wh = np.asarray([gw * width, gh * height], dtype=np.float32)
        best_anchor = max(
            range(len(anchors)), key=lambda i: _size_iou(pixel_wh, anchors[i])
        )
        for level, (mask, downsample) in enumerate(
            zip(anchor_masks, DOWNSAMPLE_RATIOS, strict=True)
        ):
            if best_anchor not in mask:
                continue
            grid_w, grid_h = width // downsample, height // downsample
            gi = min(int(gx * grid_w), grid_w - 1)
            gj = min(int(gy * grid_h), grid_h - 1)
            anchor_slot = mask.index(best_anchor)
            target = targets[level]
            target[anchor_slot, 0, gj, gi] = gx * grid_w - gi
            target[anchor_slot, 1, gj, gi] = gy * grid_h - gj
            target[anchor_slot, 2, gj, gi] = math.log(
                max(pixel_wh[0] / anchors[best_anchor, 0], 1e-9)
            )
            target[anchor_slot, 3, gj, gi] = math.log(
                max(pixel_wh[1] / anchors[best_anchor, 1], 1e-9)
            )
            target[anchor_slot, 4, gj, gi] = 2.0 - gw * gh
            target[anchor_slot, 5, gj, gi] = 1.0
            target[anchor_slot, 6, gj, gi] = 1.0
    return targets


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    chw = image.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (chw - MEAN) / STD


def preprocess_image(
    image: np.ndarray,
    width: int,
    height: int,
    resize_mode: str = "stretch",
) -> np.ndarray:
    resized, _ = resize_image(image, width, height, resize_mode)
    return normalize_image(resized)


def preprocess_image_with_transform(
    image: np.ndarray,
    width: int,
    height: int,
    resize_mode: str = "stretch",
) -> tuple[np.ndarray, ResizeTransform]:
    resized, transform = resize_image(image, width, height, resize_mode)
    return normalize_image(resized), transform


class ManifestDataset(paddle.io.Dataset):
    def __init__(
        self,
        manifest: Path,
        image_size: Iterable[int],
        augment: bool = False,
        resize_mode: str = "stretch",
        anchors: np.ndarray | list = ANCHORS,
        anchor_masks: list[list[int]] = ANCHOR_MASKS,
        max_boxes: int = 10,
        limit: int | None = None,
        sample_seed: int = 0,
    ):
        self.root = manifest.parent
        with manifest.open("r", encoding="utf-8") as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        if limit is not None and int(limit) < len(self.records):
            rng = np.random.default_rng(sample_seed)
            selected = np.sort(
                rng.choice(len(self.records), size=int(limit), replace=False)
            )
            self.records = [self.records[int(index)] for index in selected]
        self.height, self.width = map(int, image_size)
        self.augment = augment
        self.resize_mode = resize_mode
        self.anchors = np.asarray(
            ANCHORS if anchors is None else anchors, dtype=np.float32
        )
        self.anchor_masks = (
            ANCHOR_MASKS if anchor_masks is None else anchor_masks
        )
        self.max_boxes = int(max_boxes)
        if self.max_boxes <= 0:
            raise ValueError("max_boxes must be positive")

    def __len__(self) -> int:
        return len(self.records)

    def prepare_sample(
        self,
        index: int,
        augment: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        record = self.records[index]
        image = cv2.imread(
            str(self.root / record["image"]), cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            raise FileNotFoundError(self.root / record["image"])
        boxes = sanitize_boxes(
            record.get("boxes", []), image.shape[1], image.shape[0]
        )

        use_augmentation = self.augment if augment is None else augment
        if use_augmentation and random.random() < 0.5:
            image, boxes = horizontal_flip(image, boxes)
        resized, transform = resize_image(
            image, self.width, self.height, self.resize_mode
        )
        return normalize_image(resized), transform.apply_boxes(boxes)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        image, boxes = self.prepare_sample(index)
        boxes_xywh = xyxy_to_normalized_xywh(boxes, self.width, self.height)
        selected_boxes = boxes_xywh[: self.max_boxes]
        padded_boxes = np.zeros((self.max_boxes, 4), dtype=np.float32)
        padded_boxes[: len(selected_boxes)] = selected_boxes
        targets = build_yolo_targets(
            selected_boxes,
            self.width,
            self.height,
            self.anchors,
            self.anchor_masks,
        )
        return {
            "image": image,
            "gt_bbox": padded_boxes,
            "target0": targets[0],
            "target1": targets[1],
            "target2": targets[2],
        }


def raw_model_outputs(
    model: PPYoloTiny, images: paddle.Tensor
) -> list[paddle.Tensor]:
    features = model.backbone(images)
    features = model.neck(features)
    return model.yolo_head(features)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        ious = box_iou(boxes[current : current + 1], boxes[remaining])[0]
        order = remaining[ious <= threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_outputs(
    outputs: list[paddle.Tensor],
    width: int,
    height: int,
    confidence_threshold: float,
    nms_iou_threshold: float,
    anchors: np.ndarray | list = ANCHORS,
    anchor_masks: list[list[int]] = ANCHOR_MASKS,
) -> list[dict[str, np.ndarray]]:
    anchors = np.asarray(
        ANCHORS if anchors is None else anchors, dtype=np.float32
    )
    anchor_masks = ANCHOR_MASKS if anchor_masks is None else anchor_masks
    batch_size = outputs[0].shape[0]
    decoded: list[dict[str, np.ndarray]] = []
    arrays = [output.numpy() for output in outputs]
    for batch_index in range(batch_size):
        all_boxes, all_scores = [], []
        for level, output in enumerate(arrays):
            mask = anchor_masks[level]
            prediction = output[batch_index].reshape(
                len(mask), 6, output.shape[2], output.shape[3]
            )
            grid_h, grid_w = output.shape[2], output.shape[3]
            grid_x, grid_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            for anchor_slot, anchor_index in enumerate(mask):
                item = prediction[anchor_slot]
                center_x = (1.05 * _sigmoid(item[0]) - 0.025 + grid_x) / grid_w
                center_y = (1.05 * _sigmoid(item[1]) - 0.025 + grid_y) / grid_h
                box_w = (
                    np.exp(np.clip(item[2], -10, 10))
                    * anchors[anchor_index, 0]
                    / width
                )
                box_h = (
                    np.exp(np.clip(item[3], -10, 10))
                    * anchors[anchor_index, 1]
                    / height
                )
                score = _sigmoid(item[4]) * _sigmoid(item[5])
                selected = score >= confidence_threshold
                if not selected.any():
                    continue
                boxes = np.stack(
                    [
                        (center_x - box_w / 2) * width,
                        (center_y - box_h / 2) * height,
                        (center_x + box_w / 2) * width,
                        (center_y + box_h / 2) * height,
                    ],
                    axis=-1,
                )[selected]
                all_boxes.append(boxes)
                all_scores.append(score[selected])
        if all_boxes:
            boxes = np.clip(
                np.concatenate(all_boxes),
                [0, 0, 0, 0],
                [width - 1, height - 1, width - 1, height - 1],
            )
            scores = np.concatenate(all_scores)
            keep = _nms(boxes, scores, nms_iou_threshold)[:100]
            boxes, scores = boxes[keep], scores[keep]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        decoded.append({"boxes": boxes, "scores": scores})
    return decoded


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    first_area = np.clip(first[:, 2:] - first[:, :2], 0, None).prod(axis=1)
    second_area = np.clip(second[:, 2:] - second[:, :2], 0, None).prod(axis=1)
    return intersection / np.maximum(
        first_area[:, None] + second_area[None, :] - intersection, 1e-9
    )


def load_model(weights: Path, config: dict | str) -> PPYoloTiny:
    parameters = (
        model_kwargs(config) if isinstance(config, dict) else {"model": config}
    )
    model = PPYoloTiny(**parameters)
    model.set_state_dict(paddle.load(str(weights)))
    model.eval()
    return model
