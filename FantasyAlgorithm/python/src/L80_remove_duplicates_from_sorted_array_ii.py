#!/usr/bin/python3

from typing import List

"""
代码参考：
https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/?envType=study-plan-v2&envId=top-interview-150
"""


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        """Keep at most two copies of each value in a sorted list."""
        write_index = 0
        for value in nums:
            if write_index < 2 or value != nums[write_index - 2]:
                nums[write_index] = value
                write_index += 1
        del nums[write_index:]
        return write_index
