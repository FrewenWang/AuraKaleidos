"""WIDER FACE annotation parsing and manifest conversion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path

WIDER_ARCHIVES = {
    "train": {
        "filename": "WIDER_train.zip",
        "url": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip",
        "sha256": "e23b76129c825cafae8be944f65310b2e1ba1c76885afe732f179c41e5ed6d59",
        "expected": "WIDER_train/images",
    },
    "val": {
        "filename": "WIDER_val.zip",
        "url": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip",
        "sha256": "f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a",
        "expected": "WIDER_val/images",
    },
    "annotations": {
        "filename": "wider_face_split.zip",
        "url": "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip",
        "sha256": "c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c",
        "expected": "wider_face_split/wider_face_train_bbx_gt.txt",
    },
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, destination: Path, expected_sha256: str) -> bool:
    """Download and atomically install an archive after checksum validation.

    Returns ``False`` when a valid existing archive can be reused.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return False

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "AliceFaceDetection/0.1"}
    )
    with (
        urllib.request.urlopen(request) as response,
        partial.open("wb") as output,
    ):
        shutil.copyfileobj(response, output, length=1024 * 1024)
    actual = sha256_file(partial)
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {destination.name}: expected {expected_sha256}, got {actual}"
        )
    partial.replace(destination)
    return True


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Validate a ZIP and extract it without accepting links or traversal paths."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        bad_member = bundle.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt ZIP member {bad_member!r} in {archive}")
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(
                    f"Unsafe ZIP path {member.filename!r} in {archive}"
                )
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(
                    f"ZIP links are not allowed: {member.filename!r}"
                )
        bundle.extractall(destination)


def parse_wider_annotations(path: Path) -> list[dict]:
    """Parse a WIDER FACE ``*_bbx_gt.txt`` file without loading images."""
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    index = 0
    while index < len(lines):
        image = lines[index].strip()
        index += 1
        if not image:
            continue
        if index >= len(lines):
            raise ValueError(f"Missing box count after {image!r} in {path}")
        try:
            box_count = int(lines[index].strip())
        except ValueError as error:
            raise ValueError(
                f"Invalid box count for {image!r} in {path}"
            ) from error
        index += 1
        boxes = []
        ignored = 0
        # Official WIDER files keep one all-zero placeholder row when the
        # declared count is zero. Consume it without turning it into a face.
        annotation_rows = max(box_count, 1)
        if index + annotation_rows > len(lines):
            raise ValueError(
                f"Expected {annotation_rows} box rows for {image!r} in {path}"
            )
        for row_index in range(annotation_rows):
            values = [int(value) for value in lines[index].split()]
            index += 1
            if len(values) < 4:
                raise ValueError(f"Invalid WIDER box for {image!r} in {path}")
            if box_count == 0:
                if row_index == 0 and any(values):
                    raise ValueError(
                        f"Expected a zero-box placeholder for {image!r} in {path}"
                    )
                continue
            x, y, width, height = values[:4]
            invalid = len(values) > 7 and values[7] != 0
            if invalid or width <= 0 or height <= 0:
                ignored += 1
                continue
            boxes.append([x, y, x + width, y + height])
        records.append(
            {"image": image, "boxes": boxes, "ignored_boxes": ignored}
        )
    return records


def convert_wider_split(dataset_root: Path, split: str, output: Path) -> dict:
    """Convert one official WIDER split to the project's JSON Lines format."""
    if split not in {"train", "val"}:
        raise ValueError("WIDER conversion supports the train and val splits")
    annotation = (
        dataset_root / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    )
    image_root = dataset_root / f"WIDER_{split}" / "images"
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    records = parse_wider_annotations(annotation)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_count = face_count = ignored_count = 0
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            image_path = image_root / record["image"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            try:
                manifest_image = Path(
                    os.path.relpath(image_path, output.parent)
                ).as_posix()
            except ValueError:
                # Windows cannot form a relative path across drive letters.
                manifest_image = image_path.resolve().as_posix()
            stream.write(
                json.dumps({"image": manifest_image, "boxes": record["boxes"]})
                + "\n"
            )
            image_count += 1
            face_count += len(record["boxes"])
            ignored_count += record["ignored_boxes"]
    return {
        "split": split,
        "images": image_count,
        "faces": face_count,
        "ignored_boxes": ignored_count,
        "manifest": str(output.resolve()),
    }
