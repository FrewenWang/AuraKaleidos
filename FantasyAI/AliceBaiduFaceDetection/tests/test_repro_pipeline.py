import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SCRIPTS))

from train_repro import average_precision, checkpoint_state, restore_checkpoint

from alice_face_detection.anchors import anchor_quality, fit_anchors
from alice_face_detection.repro import (
    ManifestDataset,
    PPYoloTiny,
    box_iou,
    build_yolo_targets,
    horizontal_flip,
    model_kwargs,
    resize_image,
    sanitize_boxes,
    xyxy_to_normalized_xywh,
)
from alice_face_detection.wider import (
    convert_wider_split,
    parse_wider_annotations,
    safe_extract_zip,
    sha256_file,
)


class ReproPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Historical layers use explicit Paddle parameter names, so the model
        # must be shared within one process instead of instantiated per test.
        cls.model = PPYoloTiny("mbv3")

    def test_target_contains_one_positive_anchor(self):
        boxes = xyxy_to_normalized_xywh(
            np.asarray([[30, 20, 70, 70]], dtype=np.float32), 160, 96
        )
        targets = build_yolo_targets(boxes, 160, 96)
        positives = sum(int((target[:, 5] > 0).sum()) for target in targets)
        self.assertEqual(positives, 1)

    def test_model_geometry_validation(self):
        config = {
            "model": {
                "variant": "mbv3",
                "num_classes": 1,
                "anchors": [[10, 10], [20, 20], [30, 30]],
                "anchor_masks": [[2], [1], [0]],
            }
        }
        self.assertEqual(model_kwargs(config)["anchor_masks"], [[2], [1], [0]])
        config["model"]["anchor_masks"] = [[3], [1], [0]]
        with self.assertRaises(ValueError):
            model_kwargs(config)

    def test_anchor_fitting_is_deterministic(self):
        sizes = np.asarray(
            [[10, 12], [11, 13], [12, 14], [40, 45], [42, 44], [44, 48]],
            dtype=np.float32,
        )
        first = fit_anchors(sizes, clusters=2, seed=7)
        second = fit_anchors(sizes, clusters=2, seed=7)
        np.testing.assert_allclose(first, second)
        self.assertGreater(anchor_quality(sizes, first)["recall_at_0.5"], 0.9)

    def test_box_iou(self):
        first = np.asarray([[0, 0, 10, 10]], dtype=np.float32)
        second = np.asarray(
            [[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32
        )
        np.testing.assert_allclose(box_iou(first, second), [[1.0, 0.0]])

    def test_sanitize_boxes_supports_empty_and_invalid_annotations(self):
        self.assertEqual(sanitize_boxes([], 100, 80).shape, (0, 4))
        boxes = sanitize_boxes(
            [[-5, 10, 50, 90], [20, 20, 10, 30], [0, 0, np.nan, 5]],
            100,
            80,
        )
        np.testing.assert_allclose(boxes, [[0, 10, 50, 80]])

    def test_horizontal_flip_uses_original_width(self):
        image = np.zeros((50, 200), dtype=np.uint8)
        _, boxes = horizontal_flip(
            image, np.asarray([[20, 10, 100, 40]], dtype=np.float32)
        )
        np.testing.assert_allclose(boxes, [[100, 10, 180, 40]])

    def test_letterbox_box_mapping_is_reversible(self):
        image = np.zeros((100, 200), dtype=np.uint8)
        resized, transform = resize_image(image, 160, 160, "letterbox")
        boxes = np.asarray([[20, 10, 100, 50]], dtype=np.float32)
        self.assertEqual(resized.shape, (160, 160))
        np.testing.assert_allclose(
            transform.apply_boxes(boxes), [[16, 48, 80, 80]]
        )
        np.testing.assert_allclose(
            transform.restore_boxes(transform.apply_boxes(boxes)), boxes
        )
        padding_boxes = np.asarray(
            [[0, 0, 10, 20], [20, 50, 40, 70]], dtype=np.float32
        )
        self.assertEqual(transform.restore_boxes(padding_boxes).shape, (2, 4))

    def test_manifest_resizes_real_image_boxes_and_accepts_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(
                str(root / "face.png"), np.zeros((100, 200), dtype=np.uint8)
            )
            cv2.imwrite(
                str(root / "negative.png"), np.zeros((80, 120), dtype=np.uint8)
            )
            records = [
                {"image": "face.png", "boxes": [[20, 10, 100, 50]]},
                {"image": "negative.png", "boxes": []},
            ]
            manifest = root / "train.jsonl"
            manifest.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            dataset = ManifestDataset(
                manifest,
                [96, 160],
                augment=False,
                max_boxes=16,
                limit=1,
                sample_seed=3,
            )
            self.assertEqual(len(dataset), 1)
            # Re-open without a limit to exercise the negative sample.
            dataset = ManifestDataset(
                manifest, [96, 160], augment=False, max_boxes=16
            )
            _, face_boxes = dataset.prepare_sample(0)
            negative = dataset[1]
        np.testing.assert_allclose(face_boxes, [[16, 9.6, 80, 48]], rtol=1e-6)
        self.assertEqual(negative["gt_bbox"].shape, (16, 4))
        self.assertEqual(
            int(
                sum((negative[f"target{i}"][:, 5] > 0).sum() for i in range(3))
            ),
            0,
        )

    def test_wider_annotation_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "WIDER_train" / "images" / "0--Parade" / "sample.jpg"
            image.parent.mkdir(parents=True)
            cv2.imwrite(str(image), np.zeros((80, 120), dtype=np.uint8))
            split_dir = root / "wider_face_split"
            split_dir.mkdir()
            annotation = split_dir / "wider_face_train_bbx_gt.txt"
            annotation.write_text(
                "0--Parade/sample.jpg\n2\n10 20 30 40 0 0 0 0 0 0\n"
                "1 2 3 4 0 0 0 1 0 0\n",
                encoding="utf-8",
            )
            parsed = parse_wider_annotations(annotation)
            output = root / "manifests" / "train.jsonl"
            summary = convert_wider_split(root, "train", output)
            manifest_record = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(parsed[0]["boxes"], [[10, 20, 40, 60]])
        self.assertEqual(summary["faces"], 1)
        self.assertEqual(summary["ignored_boxes"], 1)
        self.assertEqual(manifest_record["boxes"], [[10, 20, 40, 60]])

    def test_wider_zero_box_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "annotations.txt"
            annotation.write_text(
                "event/negative.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n"
                "event/face.jpg\n1\n1 2 3 4 0 0 0 0 0 0\n",
                encoding="utf-8",
            )
            records = parse_wider_annotations(annotation)
        self.assertEqual(records[0]["boxes"], [])
        self.assertEqual(records[1]["boxes"], [[1, 2, 4, 6]])

    def test_wider_archive_checksum_and_safe_extraction(self):
        import zipfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("WIDER_train/images/sample.jpg", b"image")
            self.assertEqual(len(sha256_file(archive)), 64)
            safe_extract_zip(archive, root / "output")
            self.assertEqual(
                (root / "output/WIDER_train/images/sample.jpg").read_bytes(),
                b"image",
            )

            unsafe = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe, "w") as bundle:
                bundle.writestr("../escape.txt", b"unsafe")
            with self.assertRaises(ValueError):
                safe_extract_zip(unsafe, root / "unsafe-output")

    def test_average_precision_for_perfect_predictions(self):
        truths = [np.asarray([[0, 0, 10, 10]], dtype=np.float32)]
        predictions = [
            {
                "boxes": truths[0].copy(),
                "scores": np.asarray([0.9], dtype=np.float32),
            }
        ]
        self.assertAlmostEqual(average_precision(predictions, truths, 0.5), 1.0)

    def test_model_forward_backward(self):
        boxes = np.asarray([[0.5, 0.5, 0.25, 0.35]], dtype=np.float32)
        targets = build_yolo_targets(boxes, 160, 96)
        target_dict = {
            "gt_bbox": paddle.to_tensor(boxes[None]),
            **{
                f"target{index}": paddle.to_tensor(target[None])
                for index, target in enumerate(targets)
            },
        }
        self.model.train()
        output = self.model(paddle.randn([1, 3, 96, 160]), target_dict)
        self.assertTrue(np.isfinite(float(output["loss"])))
        output["loss"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in self.model.parameters()
            )
        )
        self.model.clear_gradients()

    def test_native_inference_supports_batches(self):
        self.model.eval()
        with paddle.no_grad():
            boxes, counts = self.model(paddle.randn([2, 3, 96, 160]))
        self.assertEqual(list(counts.shape), [2])
        self.assertEqual(boxes.shape[1], 6)
        self.assertEqual(int(counts.sum()), boxes.shape[0])

    def test_checkpoint_round_trip(self):
        optimizer = paddle.optimizer.Adam(
            0.001, parameters=self.model.parameters()
        )
        original = self.model.parameters()[0].numpy().copy()
        state = checkpoint_state(self.model, optimizer, 3, 12.5, [{"epoch": 3}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pdstate"
            paddle.save(state, str(path))
            self.model.parameters()[0].set_value(
                self.model.parameters()[0] + 1.0
            )
            epoch, best_loss, history, full = restore_checkpoint(
                path, self.model, optimizer
            )
        np.testing.assert_allclose(self.model.parameters()[0].numpy(), original)
        self.assertEqual(epoch, 3)
        self.assertEqual(best_loss, 12.5)
        self.assertEqual(history, [{"epoch": 3}])
        self.assertTrue(full)


if __name__ == "__main__":
    unittest.main()
