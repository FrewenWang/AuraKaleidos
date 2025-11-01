#!/usr/bin/python3

from typing import List

"""
代码参考：
https://leetcode.cn/problems/merge-sorted-array/solutions/666608/he-bing-liang-ge-you-xu-shu-zu-by-leetco-rrb0/?envType=study-plan-v2&envId=top-interview-150
"""


class Solution:
    def merge_sort(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # nums1 和 nums2：这两个都是列表（或数组）。nums1 是一个已经存在的列表，而 nums2 是另一个列表，你想要将其内容赋值给 nums1 的一部分。
        # 切片 m:：m: 是一个切片操作，表示从索引 m 开始一直到列表的末尾。例如，如果 m 等于 3，那么 nums1[m:] 就指的是 nums1 从索引 3 开始到末尾的所有元素。
        # 赋值操作：nums1[m:] = nums2 这部分代码的意思是将 nums2 的内容替换 nums1 从索引 m 开始的部分。这种操作会把 nums2 的所有元素放到 nums1 从第 m 个位置开始，直到 nums2 的元素全部放完。
        nums1[m:] = nums2
        nums1.sort()

    def merge_2points(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        sorted = []
        p1, p2 = 0, 0
        while p1 < m and p2 < n:
            if nums1[p1] < nums2[p2]:
                sorted.append(nums1[p1])
                p1 += 1
            else:
                sorted.append(nums2[p2])
                p2 += 1

        if p1 == m:
            sorted.extend(nums2[p2:n])
        elif p2 == n:
            sorted.extend(nums1[p1:m])
        # nums1[:]：这是一个切片操作，它表示整个 nums1 列表。使用 [:] 可以访问列表中的所有元素。
        # 赋值 =：这个赋值操作将右侧的内容（即 nums2）复制到左侧指定的位置。
        nums1[:] = sorted

    def merge_2point_reverse(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        p1, p2 = m - 1, n - 1
        tail = m + n - 1
        while p1 >= 0 or p2 >= 0:
            if p1 == -1:  # 说明nums1的数据先遍历完毕。那就继续在tail处添加nums2
                nums1[tail] = nums2[p2]
                p2 -= 1
            elif p2 == -1: # 说明nums2的数据先遍历完毕。那就继续在tail处添加nums1
                nums1[tail] = nums1[p1]
                p1 -= 1
            elif nums1[p1] > nums2[p2]:
                # 只有当num1大约num2然后放到对应问题
                nums1[tail] = nums1[p1]
                p1 -= 1
            else:
                # 否则直接存放num2就可以了。
                nums1[tail] = nums2[p2]
                p2 -= 1
            tail -= 1


if __name__ == "__main__":
    nums1 = [1, 2, 3, 0, 0, 0]
    m = 3
    nums2 = [2, 5, 6]
    n = 3
    solution = Solution()
    solution.merge_sort(nums1, m, nums2, n)
    print("=======================merge_sort===========================")
    print(nums1)
