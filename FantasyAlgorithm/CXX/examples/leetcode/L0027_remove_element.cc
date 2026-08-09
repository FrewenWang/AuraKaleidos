//
// Created by Frewen.Wang on 25-2-10.
//
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
  static int removeElement(vector<int> &nums, int val) {
    int n = nums.size();
    int left = 0;
    // 双指针算法：遍历右侧指针，
    for (int right = 0; right < n; right++) {
      // 如果右侧的指针的指向的这个数据，不等于Val，也就是说不是我们要进行移除的数据
      // 那么我们就需要保留这个数据
      if (nums[right] != val) {
        nums[left] = nums[right];
        left++;
      }
    }
    return left;
  }

  /**
   * 复杂度分析
   * 时间复杂度：O(n)，其中 n 为序列的长度。我们只需要遍历该序列至多一次。
   * 空间复杂度：O(1)。我们只需要常数的空间保存若干变量。
   * @param nums
   * @param val
   * @return
   */
  static int removeElement2(vector<int> &nums, int val) {
    int left = 0;
    int right = nums.size();
    while (left < right) {
      if (nums[left] == val) {
        nums[left] = nums[right - 1];
        right--;
      } else {
        left++;
      }
    }
    return left;
  }
};

int main() {
  cout << " ====================case1====================================" << endl;
  vector<int> nums = {4, 5};
  int val = 4;
  Solution s;
  s.removeElement(nums, val);
  for (int &num: nums) {
    cout << num << " ";
  }
  cout << endl;

  cout << " ====================case2====================================" << endl;
  nums = {1, 2, 3, 4, 5};
  val = 1;
  s.removeElement(nums, val);
  for (int &num: nums) {
    cout << num << " ";
  }
  cout << endl;
}
