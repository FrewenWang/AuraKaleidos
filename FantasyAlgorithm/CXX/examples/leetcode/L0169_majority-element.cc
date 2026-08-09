//
// Created by frewen on 25-2-20.
//

#include<vector>
#include<unordered_map>
#include<algorithm>

using namespace std;

class Solution {
public:
  /**
   * 这个解决方法的核心算法是： 通过一个 unordered_map 来存储每个元素出现的次数。
   * 遍历所有的元素，记录每个元素出现的次数。
   * 当记录到某个元素出现的次数大于n/2 的时候，则返回这个结果。
   * 算法比较简单。
   **/
    //  int majorityElement(vector<int> &nums) {
    //    int anw_item = 0;
    //    int len = nums.size();
    //    unordered_map<int,int> counts;
    //    for(int item:nums){
    //      ++counts[item];
    //      if(counts[item] > len/2 ) {
    //        anw_item = item;
    //        return anw_item;
    //      }
    //    }
    //    return anw_item;
    //  }

    // 这个是我自己
    //  int majorityElement(vector<int> &nums) {
    //    std::sort(nums.begin(),nums.end());
    //    int len = nums.size();
    //    int target = 0;
    //    int target_count = 0;
    //    for(int item:nums){
    //      if(item != target) {
    //        target_count = 1;
    //        target = item;
    //      } else {
    //        ++target_count;
    //      }
    //      if(target_count > len/2) {
    //        return target_count;
    //      }
    //    }
    //    return target_count;
    //  }
};