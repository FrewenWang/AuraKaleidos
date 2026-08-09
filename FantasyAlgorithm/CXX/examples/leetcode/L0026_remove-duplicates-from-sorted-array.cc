//
// Created by frewen on 25-2-10.
//
#include <stdio.h>
#include <vector>

using namespace std;

class Solution {
public:
  //方法一：双指针
  //这道题目的要求是：对给定的有序数组 nums 删除重复元素，
  //在删除重复元素之后，每个元素只出现一次，并返回新的长度，上述操作必须通过原地修改数组的方法，使用 O(1) 的空间复杂度完成。

  int removeDuplicates(vector<int>& nums) {
      int left = 0, right = 0;
      while (right < nums.size()) {
         if (nums[right] != nums[left]) {
            left++;
            nums[left] = nums[right];
         } else {
           right++;
         }
      }
      return left+1;
  }
};


int main() {
    vector<int> nums = {1,1,2};
    Solution s;
    s.removeDuplicates(nums);

}