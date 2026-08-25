"""Command-line entry point for the object-tracking demo."""

import argparse
from importlib.resources import files
from pathlib import Path


def load_class_names(filename: str | Path) -> dict[int, str]:
    """Load ``id: label`` pairs from a UTF-8 text file."""
    classes: dict[int, str] = {}
    for line_number, line in enumerate(
        Path(filename).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            class_id, class_name = line.split(":", maxsplit=1)
            classes[int(class_id)] = class_name.strip()
        except ValueError as error:
            raise ValueError(
                f"Invalid class entry on line {line_number}: {line!r}"
            ) from error
    return classes


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without importing optional AI dependencies."""
    parser = argparse.ArgumentParser(
        description="YOLO and Kalman-filter object tracking demo"
    )
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument(
        "--video-source", default="0", help="Video path or 0 for webcam"
    )
    parser.add_argument(
        "--show-classes", action="store_true", help="Display supported class IDs"
    )
    parser.add_argument("--target-class", action="append", type=int, default=[])
    parser.add_argument("--estimate-acceleration", action="store_true")
    parser.add_argument(
        "--association-metric",
        choices=("euclidean", "mahalanobis"),
        default="euclidean",
    )
    return parser


def run_tracking() -> None:
    """Track selected object classes with YOLO and a Kalman filter."""
    args = build_parser().parse_args()
    from app.app import ObjectTrackerApp

    class_file = files("app").joinpath("assets/classes.txt")
    classes = load_class_names(Path(str(class_file)))
    if args.show_classes:
        for class_id, class_name in classes.items():
            print(f"{class_id}: {class_name}")
        return

    selected_ids = args.target_class or [0]
    if args.mode == "single" and len(selected_ids) > 1:
        build_parser().error("single mode accepts only one --target-class")
    unknown_ids = [class_id for class_id in selected_ids if class_id not in classes]
    if unknown_ids:
        build_parser().error(f"unknown class IDs: {unknown_ids}")

    source: int | str = 0 if args.video_source == "0" else args.video_source
    if isinstance(source, str) and not Path(source).is_file():
        build_parser().error(f"video file does not exist: {source}")

    tracker = ObjectTrackerApp(
        mode=args.mode,
        target_classes=[classes[class_id] for class_id in selected_ids],
        estimate_acceleration=args.estimate_acceleration,
        association_metric=args.association_metric,
    )
    tracker.process_video(video_source=source)
