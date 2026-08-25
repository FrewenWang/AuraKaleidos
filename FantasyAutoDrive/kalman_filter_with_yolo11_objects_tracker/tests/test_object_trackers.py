import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai.object_trackers import KalmanTracker, TrackedObject
from app.app import ObjectTrackerApp


def blank_frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


class TrackedObjectTest(unittest.TestCase):
    def test_acceleration_model_uses_six_state_variables(self):
        state = np.asarray([[10], [0], [20], [0], [0], [0]], dtype=np.float32)
        tracked = TrackedObject(0, state=state, acceleration=True)

        self.assertEqual(tracked.state.shape, (6, 1))
        self.assertEqual(tracked.predict().shape, (6, 1))

    def test_object_is_deleted_at_configured_lost_frame_limit(self):
        tracked = TrackedObject(0)

        for _ in range(tracked.max_lost_frames - 1):
            tracked.mark_lost()
        self.assertFalse(tracked.is_deleted())

        tracked.mark_lost()
        self.assertTrue(tracked.is_deleted())


class KalmanTrackerTest(unittest.TestCase):
    def test_single_tracker_keeps_identity_for_matching_detection(self):
        tracker = KalmanTracker(mode="single")
        frame = blank_frame()

        _, initial = tracker.update([["person", 30, 40, 10, 20]], frame)
        _, updated = tracker.update([["person", 31, 40, 10, 20]], frame)

        self.assertEqual(set(initial), {0})
        self.assertEqual(set(updated), {0})
        self.assertEqual(len(tracker.tracked_objects), 1)
        self.assertEqual(tracker.tracked_objects[0].lost_frames, 0)

    def test_multi_tracker_adds_an_unmatched_detection(self):
        tracker = KalmanTracker(mode="multi")
        frame = blank_frame()
        tracker.update([["person", 10, 10, 8, 8]], frame)

        _, estimates = tracker.update([["car", 100, 100, 8, 8]], frame)

        self.assertEqual(set(estimates), {0, 1})
        self.assertEqual(len(tracker.tracked_objects), 2)
        self.assertEqual(tracker.tracked_objects[0].lost_frames, 1)

    def test_tracker_removes_object_after_consecutive_misses(self):
        tracker = KalmanTracker(mode="single")
        frame = blank_frame()
        tracker.update([["person", 30, 40, 10, 20]], frame)

        for _ in range(tracker.tracked_objects[0].max_lost_frames):
            tracker.update([], frame)

        self.assertEqual(tracker.tracked_objects, [])


class HistoryCacheTest(unittest.TestCase):
    def test_cache_round_trip_restores_integer_object_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "history.json"
            app = ObjectTrackerApp(detector=object(), cache_file=cache_file)
            app.update_cache({2: ([1.5, -0.5], [10, 20])})
            app.save_cache()

            restored = ObjectTrackerApp(detector=object(), cache_file=cache_file)

        self.assertEqual(set(restored.history_cache), {2})
        self.assertEqual(list(restored.history_cache[2]["velocity"]), [[1.5, -0.5]])

    def test_invalid_cache_is_treated_as_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "history.json"
            cache_file.write_text('{"bad": {}}', encoding="utf-8")

            app = ObjectTrackerApp(detector=object(), cache_file=cache_file)

        self.assertEqual(app.history_cache, {})


if __name__ == "__main__":
    unittest.main()
