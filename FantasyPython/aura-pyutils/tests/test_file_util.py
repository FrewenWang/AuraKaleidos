import unittest

from aura_pyutils.file_util import is_end_with


class FileUtilTest(unittest.TestCase):
    def test_extension_match_is_case_insensitive(self):
        self.assertTrue(is_end_with("photo.JPEG", "jpeg"))

    def test_non_matching_extension(self):
        self.assertFalse(is_end_with("archive.tar.gz", "zip"))

    def test_empty_extension_matches_any_filename(self):
        self.assertTrue(is_end_with("README", ""))


if __name__ == "__main__":
    unittest.main()
