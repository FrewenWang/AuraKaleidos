//
// Created by frewen on 25-2-12.
//
#include <vector>
#include <cassert>
using namespace std;

//35. 搜索插入位置
//https://leetcode.cn/problems/search-insert-position/description/?envType=study-plan-v2&envId=top-interview-150
//给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
//
//请必须使用时间复杂度为 O(log n) 的算法。
//
//
//
//示例 1:
//
//输入: nums = [1,3,5,6], target = 5
//输出: 2
//示例 2:
//
//输入: nums = [1,3,5,6], target = 2
//输出: 1
//示例 3:
//
//输入: nums = [1,3,5,6], target = 7
//输出: 4
//
//
//提示:
//
//1 <= nums.length <= 104
//-104 <= nums[i] <= 104
//nums 为 无重复元素 的 升序 排列数组
//-104 <= target <= 104

class Solution {
public:
  /**
    * @brief 二分查找
    * 首先，定义左右指针，初始值分别为0和数组长度减一。
    * 然后开始进行while循环（一般需要进行遍历，且在遍历过程中需要修改指针的位置的都基本最好使用while循环，例如双指针做法、滑动窗口、二分查找）
    * 我们先进行就求解中间位置的mid，然后进行判断，
    * 如果nums[mid] == target，则返回mid，这个很好解释，指直接就找到了对应的元素嘛。
    * 如果nums[mid] < target，说明待查找的数据在mid的右边，所以，我们只需要将left指针移动到mid + 1，继续进行二分查找。
    * 如果nums[mid] > target，说明待查找的数据在mid的左边，所以，我们只需要将right指针移动到mid - 1，继续进行二分查找。
    * 直到while循环最后，也就是left == right，此时，left和right都指向了最后一个待查找的元素
    * 此时，我们判断nums[left] < target，如果小于，说明待查找的数据在left的右边，所以，我们只需要将left指针移动到left + 1，继续进行二分查找。
    */
  int searchInsert(vector<int>& nums, int target) {
      int left = 0, right = nums.size() - 1;
      while(left < right){
        int mid = left + (right - left) / 2;
        if(nums[mid] == target) {
          return mid;
        } else if(nums[mid] < target) {
          left = mid + 1;
        } else {
          right = mid-1;
        }
      }
      return nums[left] < target ? left + 1 : left;
  }
};


void testSearchInsert() {
  Solution sol;

  // Test case 1: Target exists in the array
  vector<int> nums1 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums1, 5) == 2);

  // Test case 2: Target does not exist in the array
  vector<int> nums2 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums2, 2) == 1);

  // Test case 3: Target is less than all elements
  vector<int> nums3 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums3, 0) == 0);

  // Test case 4: Target is greater than all elements
  vector<int> nums4 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums4, 7) == 4);

  // Test case 5: Empty array
  vector<int> nums5 = {};
  assert(sol.searchInsert(nums5, 1) == 0);

  // Test case 6: Target is equal to the last element
  vector<int> nums6 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums6, 6) == 3);

  // Test case 7: Target is equal to the first element
  vector<int> nums7 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums7, 1) == 0);

  // Test case 8: Target is between two elements
  vector<int> nums8 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums8, 4) == 2);

  // Test case 9: Single element array, target is less than the element
  vector<int> nums9 = {1};
  assert(sol.searchInsert(nums9, 0) == 0);

  // Test case 10: Single element array, target is equal to the element
  vector<int> nums10 = {1};
  assert(sol.searchInsert(nums10, 1) == 0);

  // Test case 11: Single element array, target is greater than the element
  vector<int> nums11 = {1};
  assert(sol.searchInsert(nums11, 2) == 1);

  // Test case 12: Target is equal to the second last element
  vector<int> nums12 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums12, 5) == 2);

  // Test case 13: Target is equal to the second element
  vector<int> nums13 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums13, 3) == 1);

  // Test case 14: Target is equal to the third element
  vector<int> nums14 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums14, 5) == 2);

  // Test case 15: Target is equal to the fourth element
  vector<int> nums15 = {1, 3, 5, 6};
  assert(sol.searchInsert(nums15, 6) == 3);

  // Test case 16: Target is equal to the fifth element
  vector<int> nums16 = {1, 3, 5, 6, 8};
  assert(sol.searchInsert(nums16, 8) == 4);

  // Test case 17: Target is equal to the sixth element
  vector<int> nums17 = {1, 3, 5, 6, 8, 10};
  assert(sol.searchInsert(nums17, 10) == 5);

  // Test case 18: Target is equal to the seventh element
  vector<int> nums18 = {1, 3, 5, 6, 8, 10, 12};
  assert(sol.searchInsert(nums18, 12) == 6);
}

int main() {
  testSearchInsert();
  return 0;
}