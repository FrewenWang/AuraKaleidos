import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from alice_face_detection.anchors import anchor_quality, fit_anchors, size_iou
from alice_face_detection.wider import (
    parse_wider_annotations,
    safe_extract_zip,
    sha256_file,
)


class AnchorUtilityTests(unittest.TestCase):
    def test_size_iou_and_quality(self):
        boxes = np.asarray([[10, 10], [20, 10]], dtype=np.float32)
        anchors = np.asarray([[10, 10], [40, 20]], dtype=np.float32)

        np.testing.assert_allclose(size_iou(boxes, anchors)[0], [1.0, 0.125])
        self.assertEqual(anchor_quality(boxes, anchors)["recall_at_0.5"], 1.0)

    def test_anchor_fitting_is_deterministic_and_sorted(self):
        sizes = np.asarray(
            [[10, 12], [11, 13], [12, 14], [40, 45], [42, 44], [44, 48]],
            dtype=np.float32,
        )

        first = fit_anchors(sizes, clusters=2, seed=7)
        second = fit_anchors(sizes, clusters=2, seed=7)

        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(first[0].prod(), first[1].prod())

    def test_anchor_fitting_rejects_too_few_valid_boxes(self):
        with self.assertRaisesRegex(ValueError, "at least 2 valid boxes"):
            fit_anchors([[10, 10], [0, 5], [np.nan, 1]], clusters=2)


class WiderUtilityTests(unittest.TestCase):
    def test_parse_annotations_handles_faces_ignored_boxes_and_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations = Path(directory) / "wider.txt"
            annotations.write_text(
                "event/face.jpg\n2\n10 20 30 40 0 0 0 0\n"
                "1 2 3 4 0 0 0 1\n"
                "event/negative.jpg\n0\n0 0 0 0 0 0 0 0\n",
                encoding="utf-8",
            )

            records = parse_wider_annotations(annotations)

        self.assertEqual(records[0]["boxes"], [[10, 20, 40, 60]])
        self.assertEqual(records[0]["ignored_boxes"], 1)
        self.assertEqual(records[1]["boxes"], [])

    def test_parse_annotations_rejects_truncated_input(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations = Path(directory) / "wider.txt"
            annotations.write_text("event/face.jpg\n1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Expected 1 box rows"):
                parse_wider_annotations(annotations)

    def test_checksum_and_safe_zip_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.bin"
            payload.write_bytes(b"abc")
            archive = root / "sample.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("images/sample.jpg", b"image")

            self.assertEqual(
                sha256_file(payload),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            safe_extract_zip(archive, root / "output")
            self.assertEqual(
                (root / "output/images/sample.jpg").read_bytes(), b"image"
            )

    def test_safe_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", b"unsafe")

            with self.assertRaisesRegex(ValueError, "Unsafe ZIP path"):
                safe_extract_zip(archive, root / "output")


if __name__ == "__main__":
    unittest.main()
