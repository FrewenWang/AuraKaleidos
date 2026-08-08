from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from data_comparator import SimpleDataComparator


class SimpleDataComparatorTest(unittest.TestCase):
    def _write_pair(self, directory, first, second, dtype=np.uint8):
        first_path = Path(directory) / "first.bin"
        second_path = Path(directory) / "second.bin"
        np.asarray(first, dtype=dtype).tofile(first_path)
        np.asarray(second, dtype=dtype).tofile(second_path)
        return first_path, second_path

    def test_reports_changed_values_and_percentage(self):
        with TemporaryDirectory() as directory:
            paths = self._write_pair(directory, [0, 10, 20, 30], [0, 12, 20, 25])
            result = SimpleDataComparator(*paths, np.uint8, (2, 2)).find_differences()
        self.assertEqual(result["diff_count"], 2)
        self.assertEqual(result["diff_percent"], 50)
        self.assertEqual(result["diff_list"][0]["difference"], 2)
        self.assertEqual(result["diff_list"][1]["difference"], 5)

    def test_honors_tolerance(self):
        with TemporaryDirectory() as directory:
            paths = self._write_pair(directory, [1.0, 2.0], [1.01, 2.2], np.float32)
            result = SimpleDataComparator(*paths, np.float32, (1, 2)).find_differences(0.05)
        self.assertEqual(result["diff_count"], 1)

    def test_rejects_data_that_does_not_match_shape(self):
        with TemporaryDirectory() as directory:
            paths = self._write_pair(directory, [1, 2], [1, 2])
            with self.assertRaises(ValueError):
                SimpleDataComparator(*paths, np.uint8, (2, 2))


if __name__ == "__main__":
    unittest.main()
