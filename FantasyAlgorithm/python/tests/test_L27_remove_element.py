import unittest

from L27_remove_element import Solution


class RemoveElementTest(unittest.TestCase):
    def test_removes_matching_values(self):
        values = [0, 1, 2, 2, 3, 0, 4, 2]
        length = Solution().removeElement(values, 2)
        self.assertEqual(length, 5)
        self.assertEqual(values, [0, 1, 3, 0, 4])

    def test_handles_empty_and_all_matching_lists(self):
        empty = []
        self.assertEqual(Solution().removeElement(empty, 1), 0)
        self.assertEqual(empty, [])

        matching = [3, 3]
        self.assertEqual(Solution().removeElement(matching, 3), 0)
        self.assertEqual(matching, [])


if __name__ == "__main__":
    unittest.main()
