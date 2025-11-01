//
// Created by frewen on 25-2-11.
//
#include <string>
#include <iostream>
#include <unordered_set>

using namespace std;

class Solution {
public:
  int lengthOfLongestSubstring(string s) {
    int n = s.length();
    int left = 0, right = 0;
    int ans = 0;
    // 无序的的集合的性能更高
    unordered_set<char> set;
    for (; right < n; ++right) {
      char ch = s[right];
      // 我们判断数据里面是否还存在对应的字符
      // 如果还存在对应的字符，则移除对应的字符，同时left指针向右移动
      while (set.count(ch)) {
        // 判断左指针处的字符
        char c = s[left];
        // 在集合中移除左指针处的字符
        set.erase(c);
        // 将左指针向右右移动
        left++;
      }
      // 将有指针处的字符插入到set
      set.insert(ch);
      // 计算这个长度，和新的长度
      ans = std::max(ans, right - left + 1);
    }
    return ans;
  }
};

int main() {
  string s = "abcabcbb";
  Solution so;

  int result = so.lengthOfLongestSubstring(s);
  cout << result << endl;


  s = "bbbbb";
  result = so.lengthOfLongestSubstring(s);
  cout << result << endl;

  s = "pwwkew";
  result = so.lengthOfLongestSubstring(s);
  cout << result << endl;
}
