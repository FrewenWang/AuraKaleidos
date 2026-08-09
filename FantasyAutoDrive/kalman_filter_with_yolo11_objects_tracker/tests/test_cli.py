import tempfile
import unittest
from pathlib import Path

from app.cli import load_class_names


class ClassFileTest(unittest.TestCase):
    def test_loads_class_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "classes.txt"
            fixture.write_text("0: person\n2: car\n", encoding="utf-8")
            self.assertEqual(load_class_names(fixture), {0: "person", 2: "car"})

    def test_rejects_malformed_line(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "classes.txt"
            fixture.write_text("invalid\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_class_names(fixture)


if __name__ == "__main__":
    unittest.main()
