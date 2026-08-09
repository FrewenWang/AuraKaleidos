//
// Created by Frewen.Wang on 25-3-13.
//
#include<vector>
#include<string>

using namespace std;

class Solution {
public:
  int search(vector<int>& nums, int target) {
    int len = nums.size();
    int l = 0, r = len-1;
    while(l <= r) {
      /// 二分查找最核心的思想就是计算中间处的数据（也就是mid = (r-l) / 2 + l）和要要查找的数据进行比较
      int mid = (r-l) / 2 + l;

      if(nums[mid] > target) {
        r = mid - 1;
      } else if(nums[mid] < target) {
        l = mid + 1;
      } else {
        return mid;
      }
    }
    return -1;
  }
};