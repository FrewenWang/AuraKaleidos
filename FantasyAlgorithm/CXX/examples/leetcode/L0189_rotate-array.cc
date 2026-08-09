//
// Created by frewen on 25-2-17.
//
//189. 轮转数组
//题目链接：https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2&envId=top-interview-150
//给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
//
//
//
//示例 1:
//
//输入: nums = [1,2,3,4,5,6,7], k = 3
//输出: [5,6,7,1,2,3,4]
//解释:
//向右轮转 1 步: [7,1,2,3,4,5,6]
//向右轮转 2 步: [6,7,1,2,3,4,5]
//向右轮转 3 步: [5,6,7,1,2,3,4]
//示例 2:
//
//输入：nums = [-1,-100,3,99], k = 2
//输出：[3,99,-1,-100]
//解释:
//向右轮转 1 步: [99,-1,-100,3]
//向右轮转 2 步: [3,99,-1,-100]
//
//
//提示：
//
//1 <= nums.length <= 105
//-231 <= nums[i] <= 231 - 1
//0 <= k <= 105
//
//
//进阶：
//
//尽可能想出更多的解决方案，至少有 三种 不同的方法可以解决这个问题。
//你可以使用空间复杂度为 O(1) 的 原地 算法解决这个问题吗？

#include<vector>

using namespace std;

class Solution {
public:
    // 这个方法。就是将原有的数组分成两部分。重新申请一块内存，线盛放后面那部分的数据，再放前半部分的数据
    // 这里面有个问题需要注意：K的值有可能大于数组的长度，导致直接使用的k的时候，出现数组越界
    // 这个时候注意：使用int result_k = k%length;进行防止索引越界
    // 还有一点：这个方法不是很好。需要重新申请一块内存来存储。空间复杂度是O(n) 。我们么可以进行优化。
    //  void rotate(vector<int>& nums,int k) {
    //      if(nums.size() == 1 || nums.size() == 0) {
    //        return;
    //      }
    //      vector<int> result;
    //      int length = nums.size();
    //      result.reserve(length);
    //      int result_k = k%length;
    //      for(int i=result_k;i>0;i--) {
    //        result.emplace_back(nums[length-i]);
    //      }
    //      for(int i=0;i<length-result_k;i++) {
    //        result.emplace_back(nums[i]);
    //      }
    //      nums = result;
    //  }

      void rotate(vector<int>& nums,int k) {
          if(nums.size() == 1 || nums.size() == 0) {
            return;
          }
          vector<int> result;
          int length = nums.size();
          result.reserve(length);
          k = k%length;
          for(int i=length-k;i<length;i++) {
            for(int j = i-1; j>0;j--) {
               int temp = nums[j+1];

            }

          }
          for(int i=0;i<length-k;i++) {
            result.emplace_back(nums[i]);
          }
          nums = result;
      }


};