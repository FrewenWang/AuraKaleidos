import unittest

from L80_remove_duplicates_from_sorted_array_ii import Solution


class RemoveDuplicatesTest(unittest.TestCase):
    def test_keeps_at_most_two_copies(self):
        values = [0, 0, 1, 1, 1, 1, 2, 3, 3]
        length = Solution().removeDuplicates(values)
        self.assertEqual(length, 7)
        self.assertEqual(values, [0, 0, 1, 1, 2, 3, 3])

    def test_handles_short_lists(self):
        values = [1]
        self.assertEqual(Solution().removeDuplicates(values), 1)
        self.assertEqual(values, [1])


if __name__ == "__main__":
    unittest.main()
