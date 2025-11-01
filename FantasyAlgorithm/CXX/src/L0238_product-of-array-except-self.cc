//
// Created by frewen on 25-2-19.
//
//238. 除自身以外数组的乘积
//
//给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
//
//题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。
//
//请不要使用除法，且在 O(n) 时间复杂度内完成此题。
//
//
//
//示例 1:
//
//输入: nums = [1,2,3,4]
//输出: [24,12,8,6]
//示例 2:
//
//输入: nums = [-1,1,0,-3,3]
//输出: [0,0,9,0,0]
//
//
//提示：
//
//2 <= nums.length <= 105
//-30 <= nums[i] <= 30
//输入 保证 数组 answer[i] 在  32 位 整数范围内
//
//
//进阶：你可以在 O(1) 的额外空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组 不被视为 额外空间。）

#include<vector>

using namespace std;

class Solution {
public:
//  复杂度分析
//
//时间复杂度：O(N)，其中N指的是数组nums的大小。预处理L和R数组以及最后的遍历计算都是O(N)的时间复杂度。
//空间复杂度：O(N)，其中N指的是数组nums的大小。使用了L和R数组去构造答案，L和R数组的长度为数组nums的大小。
//  vector<int> productExceptSelf(vector<int>& nums){
//    int length = nums.size();
//    vector<int> result(length); // 数组大小是length
//    vector<int> L(length),R(length);
//
//    L[0] = 1;
//    for(int i=1;i<length;i++){
//      L[i] = nums[i-1] * L[i-1];
//    }
//
//    R[length-1] = 1;
//    for(int i=length-1-1;i>=0;i--){
//      R[i] = nums[i+1] * R[i+1];
//    }
//
//    for(int k=0;k<length;k++) {
//      result[k] = L[k] * R[k];
//    }
//    return result;
//  }

    vector<int> productExceptSelf(vector<int>& nums){
      int length = nums.size();
      vector<int> result(length);

      result[0] = 1;
      for(int i=1;i<length;i++){
        result[i] = nums[i-1] * result[i-1];
      }

      int R = 1;
      for(int i=length-1;i>=0;i--){
        // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
        result[i] = result[i] * R;
        // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
        R *= nums[i];
      }
      return result;
    }
};