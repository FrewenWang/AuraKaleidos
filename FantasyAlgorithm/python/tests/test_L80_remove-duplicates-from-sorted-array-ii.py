import unittest


class TestMySolution(unittest.TestCase):

    def test_case_1(self):
        solution = Solution() 
        self.assertEqual(MyFunction(1, 2), 3)

    def test_case_2(self):
        self.assertTrue(MyFunction(0, 0))

if __name__ == '__main__':
    unittest.main()