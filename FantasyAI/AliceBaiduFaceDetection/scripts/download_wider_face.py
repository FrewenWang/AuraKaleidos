#!/usr/bin/env python3
"""Download, verify, and extract the official WIDER FACE train/val data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_source_root

add_source_root()

from alice_face_detection.wider import (  # noqa: E402
    WIDER_ARCHIVES,
    download_archive,
    safe_extract_zip,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/wider"))
    parser.add_argument(
        "--archives",
        default="train,val,annotations",
        help="Comma-separated choices: train,val,annotations",
    )
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser()
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[1] / root).resolve()
    download_dir = root / "downloads"
    selected = [
        name.strip() for name in args.archives.split(",") if name.strip()
    ]
    unknown = sorted(set(selected) - set(WIDER_ARCHIVES))
    if unknown:
        raise SystemExit(f"Unknown archives: {', '.join(unknown)}")

    results = []
    for name in selected:
        metadata = WIDER_ARCHIVES[name]
        archive = download_dir / metadata["filename"]
        downloaded = download_archive(
            metadata["url"], archive, metadata["sha256"]
        )
        expected = root / metadata["expected"]
        extracted = False
        if not args.download_only and not expected.exists():
            safe_extract_zip(archive, root)
            extracted = True
        if not args.download_only and not expected.exists():
            raise FileNotFoundError(
                f"Archive did not create expected path: {expected}"
            )
        results.append(
            {
                "archive": name,
                "path": str(archive),
                "sha256": sha256_file(archive),
                "downloaded": downloaded,
                "extracted": extracted,
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
