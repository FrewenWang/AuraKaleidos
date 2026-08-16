#!/usr/bin/env python3
"""Evaluate a trained checkpoint at one or more confidence thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_source_root

add_source_root()

from train_repro import collect_detections, summarize_detections

from alice_face_detection.repro import (  # noqa: E402
    load_config,
    load_model,
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
    parser.add_argument("--thresholds", default="0.05,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_from_project(args.config))
    model = load_model(resolve_from_project(args.weights), config)
    manifest = resolve_from_project(config["data"]["root"]) / "val.jsonl"
    predictions, truths = collect_detections(model, manifest, config)
    results = {
        str(threshold): summarize_detections(
            predictions, truths, config["evaluation"], threshold
        )
        for threshold in (float(value) for value in args.thresholds.split(","))
    }
    text = json.dumps(results, indent=2)
    if args.output:
        output = resolve_from_project(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
