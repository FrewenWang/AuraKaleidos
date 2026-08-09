//
// Created by frewen on 25-2-10.
//
//392. 判断子序列
//题目链接：https://leetcode.cn/problems/is-subsequence/description/?envType=study-plan-v2&envId=top-interview-150
//给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
//
//字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
//
//进阶：
//
//如果有大量输入的 S，称作 S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？
//
//致谢：
//
//特别感谢 @pbrother 添加此问题并且创建所有测试用例。

#include <string>

using namespace std;

class Solution {
public:
  bool isSubsequence(string s, string t) {
    int n = s.length();
    int m = t.length();
    int i = 0;
    int j = 0;

    while (i < n && j < m) {
      if (s[i] == t[j]) {
        i++;
      }
      j++;
    }
    if (i == n && j < m ) {
      return true;
    }
    return false;
  }
};

int main() {
  string s = "abc";
  string t = "ahbgdc";
  Solution so;
  so.isSubsequence(s, t);
}