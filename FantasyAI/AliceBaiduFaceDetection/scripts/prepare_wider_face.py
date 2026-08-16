#!/usr/bin/env python3
"""Convert an extracted WIDER FACE dataset to reproducible JSONL manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_source_root

add_source_root()

from alice_face_detection.repro import resolve_from_project  # noqa: E402
from alice_face_detection.wider import convert_wider_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/wider"))
    parser.add_argument(
        "--splits", default="train,val", help="Comma-separated train/val splits"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Defaults to <dataset-root>/manifests"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_from_project(args.dataset_root)
    output_dir = (
        resolve_from_project(args.output_dir)
        if args.output_dir
        else root / "manifests"
    )
    summaries = [
        convert_wider_split(
            root, split.strip(), output_dir / f"{split.strip()}.jsonl"
        )
        for split in args.splits.split(",")
        if split.strip()
    ]
    (output_dir / "dataset_info.json").write_text(
        json.dumps({"dataset": "WIDER FACE", "splits": summaries}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
