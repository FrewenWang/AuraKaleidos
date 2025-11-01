//
// Created by Frewen.Wang on 25-2-27.
//

#include<vector>

using namespace std;


class Solution {
public:
  /**
   * 这道题目的核心思想：就是既然在能到最后
   * 
   */
  bool canJump(vector<int>& nums) {
    bool canJump = false;
    int len = nums.size();
    int reach = 0;

    for(int i=0;i<len;i++) {
      if(i > reach) {
        return false;
      }
      reach = std::max(i+nums[i],reach);
    }
    return true;
  }
};