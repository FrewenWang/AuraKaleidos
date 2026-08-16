#!/usr/bin/env python3
"""Train the historical PPYoloTiny face detector with a portable pipeline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import paddle
from _bootstrap import add_source_root

add_source_root()

from alice_face_detection.logger import setup_logger  # noqa: E402
from alice_face_detection.repro import (  # noqa: E402
    ManifestDataset,
    PPYoloTiny,
    box_iou,
    decode_outputs,
    load_config,
    model_kwargs,
    raw_model_outputs,
    resolve_from_project,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/repro_cpu.yaml")
    )
    parser.add_argument(
        "--epochs", type=int, help="Override the configured epoch count"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from .pdstate or load legacy .pdparams weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the configured output directory",
    )
    parser.add_argument(
        "--log-dir", type=Path, help="Override the configured log directory"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Override max training batches per epoch"
    )
    return parser.parse_args()


def make_loader(
    dataset: ManifestDataset, batch_size: int, shuffle: bool, workers: int
) -> paddle.io.DataLoader:
    return paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        return_list=True,
        use_shared_memory=False,
    )


def evaluate_loss(model: PPYoloTiny, loader: paddle.io.DataLoader) -> float:
    model.eval()
    losses = []
    with paddle.no_grad():
        for batch in loader:
            outputs = model(
                batch["image"],
                {key: value for key, value in batch.items() if key != "image"},
            )
            losses.append(float(outputs["loss"]))
    return float(np.mean(losses))


def collect_detections(
    model: PPYoloTiny,
    manifest: Path,
    config: dict,
) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
    data_config = config["data"]
    eval_config = config["evaluation"]
    geometry = model_kwargs(config)
    height, width = map(int, data_config["image_size"])
    dataset = ManifestDataset(
        manifest,
        data_config["image_size"],
        augment=False,
        resize_mode=data_config.get("resize_mode", "stretch"),
        anchors=geometry["anchors"],
        anchor_masks=geometry["anchor_masks"],
        max_boxes=int(data_config.get("max_boxes", 10)),
        limit=data_config.get("val_limit"),
        sample_seed=int(config["seed"]) + 1,
    )
    model.eval()
    predictions = []
    truths = []
    with paddle.no_grad():
        for index in range(len(dataset)):
            image, truth = dataset.prepare_sample(index, augment=False)
            images = paddle.to_tensor(image[None])
            prediction = decode_outputs(
                raw_model_outputs(model, images),
                width,
                height,
                float(eval_config.get("ap_confidence_floor", 0.001)),
                float(eval_config["nms_iou_threshold"]),
                geometry["anchors"],
                geometry["anchor_masks"],
            )[0]
            predictions.append(prediction)
            truths.append(truth)
    return predictions, truths


def threshold_metrics(
    predictions: list[dict[str, np.ndarray]],
    truths: list[np.ndarray],
    confidence_threshold: float,
    match_iou_threshold: float,
) -> dict[str, float]:
    true_positive = false_positive = false_negative = 0
    for prediction, truth in zip(predictions, truths, strict=True):
        selected = prediction["scores"] >= confidence_threshold
        boxes = prediction["boxes"][selected]
        scores = prediction["scores"][selected]
        order = scores.argsort()[::-1]
        matched: set[int] = set()
        for box in boxes[order]:
            ious = box_iou(box[None], truth)[0]
            best = int(np.argmax(ious)) if len(ious) else -1
            if (
                best >= 0
                and ious[best] >= match_iou_threshold
                and best not in matched
            ):
                true_positive += 1
                matched.add(best)
            else:
                false_positive += 1
        false_negative += len(truth) - len(matched)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "precision_at_0.5": precision,
        "recall_at_0.5": recall,
        "f1_at_0.5": 2 * precision * recall / max(precision + recall, 1e-9),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def average_precision(
    predictions: list[dict[str, np.ndarray]],
    truths: list[np.ndarray],
    iou_threshold: float,
) -> float:
    ranked = []
    for image_id, prediction in enumerate(predictions):
        ranked.extend(
            (float(score), image_id, box)
            for score, box in zip(
                prediction["scores"], prediction["boxes"], strict=True
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    matched = [set() for _ in truths]
    true_positives, false_positives = [], []
    for _score, image_id, box in ranked:
        truth = truths[image_id]
        ious = box_iou(box[None], truth)[0]
        best = int(np.argmax(ious)) if len(ious) else -1
        is_match = (
            best >= 0
            and ious[best] >= iou_threshold
            and best not in matched[image_id]
        )
        true_positives.append(1.0 if is_match else 0.0)
        false_positives.append(0.0 if is_match else 1.0)
        if is_match:
            matched[image_id].add(best)

    if not ranked or sum(len(truth) for truth in truths) == 0:
        return 0.0
    tp = np.cumsum(true_positives)
    fp = np.cumsum(false_positives)
    recall = tp / sum(len(truth) for truth in truths)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(
        np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1])
    )


def summarize_detections(
    predictions: list[dict[str, np.ndarray]],
    truths: list[np.ndarray],
    evaluation: dict,
    confidence_threshold: float | None = None,
) -> dict[str, float]:
    metrics = threshold_metrics(
        predictions,
        truths,
        float(
            evaluation["confidence_threshold"]
            if confidence_threshold is None
            else confidence_threshold
        ),
        float(evaluation["match_iou_threshold"]),
    )
    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    ap_values = [
        average_precision(predictions, truths, float(iou))
        for iou in iou_thresholds
    ]
    metrics["ap50"] = ap_values[0]
    metrics["map50_95"] = float(np.mean(ap_values))
    return metrics


def detection_metrics(
    model: PPYoloTiny, manifest: Path, config: dict
) -> dict[str, float]:
    predictions, truths = collect_detections(model, manifest, config)
    return summarize_detections(predictions, truths, config["evaluation"])


def checkpoint_state(
    model: PPYoloTiny,
    optimizer: paddle.optimizer.Optimizer,
    epoch: int,
    best_loss: float,
    history: list[dict],
) -> dict:
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "history": history,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        # Paddle 2.6 exposes GeneratorState objects that cannot be pickled.
        # Persist the generator seed; Python and NumPy states remain exact.
        "paddle_random_seed": paddle.get_rng_state()[0].current_seed(),
    }


def restore_checkpoint(
    path: Path,
    model: PPYoloTiny,
    optimizer: paddle.optimizer.Optimizer,
) -> tuple[int, float, list[dict], bool]:
    state = paddle.load(str(path))
    if not isinstance(state, dict) or "model" not in state:
        model.set_state_dict(state)
        return 0, float("inf"), [], False

    model.set_state_dict(state["model"])
    optimizer.set_state_dict(state["optimizer"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    paddle.seed(int(state["paddle_random_seed"]))
    return (
        int(state["epoch"]),
        float(state["best_loss"]),
        list(state.get("history", [])),
        True,
    )


def main() -> None:
    args = parse_args()
    config_path = resolve_from_project(args.config)
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    training = config["training"]
    data = config["data"]
    paddle.set_device(str(training.get("device", "cpu")))
    log_dir = resolve_from_project(
        args.log_dir or training.get("log_dir", "logs")
    )
    logger = setup_logger(
        "alice_face_detection.train", str(log_dir / "train.log")
    )

    data_root = resolve_from_project(data["root"])
    train_manifest, val_manifest = (
        data_root / "train.jsonl",
        data_root / "val.jsonl",
    )
    if not train_manifest.exists() or not val_manifest.exists():
        raise SystemExit(
            "Data is missing. Run scripts/prepare_dataset.py first."
        )
    output_dir = resolve_from_project(args.output_dir or training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    resize_mode = data.get("resize_mode", "stretch")
    geometry = model_kwargs(config)
    train_dataset = ManifestDataset(
        train_manifest,
        data["image_size"],
        augment=True,
        resize_mode=resize_mode,
        anchors=geometry["anchors"],
        anchor_masks=geometry["anchor_masks"],
        max_boxes=int(data.get("max_boxes", 10)),
        limit=data.get("train_limit"),
        sample_seed=int(config["seed"]),
    )
    val_dataset = ManifestDataset(
        val_manifest,
        data["image_size"],
        augment=False,
        resize_mode=resize_mode,
        anchors=geometry["anchors"],
        anchor_masks=geometry["anchor_masks"],
        max_boxes=int(data.get("max_boxes", 10)),
        limit=data.get("val_limit"),
        sample_seed=int(config["seed"]) + 1,
    )
    train_loader = make_loader(
        train_dataset,
        int(training["batch_size"]),
        True,
        int(training["num_workers"]),
    )
    val_loader = make_loader(
        val_dataset,
        int(training["batch_size"]),
        False,
        int(training["num_workers"]),
    )

    model = PPYoloTiny(**geometry)
    optimizer = paddle.optimizer.Adam(
        learning_rate=float(training["learning_rate"]),
        parameters=model.parameters(),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = args.epochs or int(training["epochs"])
    start_epoch, best_loss, history = 0, float("inf"), []
    if args.resume:
        resume_path = resolve_from_project(args.resume)
        start_epoch, best_loss, history, full_resume = restore_checkpoint(
            resume_path, model, optimizer
        )
        mode = "full checkpoint" if full_resume else "weights only"
        logger.info(
            "Resumed %s from %s at epoch %d", mode, resume_path, start_epoch
        )
        if not (output_dir / "best.pdparams").exists():
            paddle.save(model.state_dict(), str(output_dir / "best.pdparams"))

    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        # Make each epoch's sampling and augmentation deterministic so a
        # resumed run produces the same future batches as an uninterrupted run.
        epoch_seed = int(config["seed"]) + epoch
        seed_everything(epoch_seed)
        started = time.time()
        model.train()
        losses = []
        max_steps = (
            args.max_steps
            if args.max_steps is not None
            else training.get("max_steps_per_epoch")
        )
        for step, batch in enumerate(train_loader, start=1):
            targets = {
                key: value for key, value in batch.items() if key != "image"
            }
            outputs = model(batch["image"], targets)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses.append(float(loss))
            if max_steps is not None and step >= int(max_steps):
                break
        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(model, val_loader)
        finite_val_loss = val_loss if np.isfinite(val_loss) else None
        row = {
            "epoch": epoch,
            "seed": epoch_seed,
            "steps": len(losses),
            "train_loss": train_loss,
            "val_loss": finite_val_loss,
            "seconds": time.time() - started,
        }
        history.append(row)
        logger.info(json.dumps(row))
        paddle.save(model.state_dict(), str(output_dir / "last.pdparams"))
        is_best = finite_val_loss is not None and val_loss < best_loss
        if is_best:
            best_loss = val_loss
            paddle.save(model.state_dict(), str(output_dir / "best.pdparams"))
        state = checkpoint_state(model, optimizer, epoch, best_loss, history)
        paddle.save(state, str(output_dir / "last.pdstate"))
        if is_best:
            paddle.save(state, str(output_dir / "best.pdstate"))

    model.set_state_dict(paddle.load(str(output_dir / "best.pdparams")))
    metrics = detection_metrics(model, val_manifest, config)
    result = {
        "model": "PPYoloTiny/MobileNetV3",
        "paddle_version": paddle.__version__,
        "start_epoch": start_epoch,
        "run_epochs": epochs,
        "final_epoch": start_epoch + epochs,
        "best_val_loss": best_loss,
        "history": history,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    logger.info("Metrics:\n%s", json.dumps(metrics, indent=2))
    logger.info("Best weights: %s", output_dir / "best.pdparams")


if __name__ == "__main__":
    main()
