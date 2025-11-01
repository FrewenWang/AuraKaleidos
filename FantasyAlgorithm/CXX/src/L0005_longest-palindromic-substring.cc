//
// Created by frewen on 25-2-28.
//
#include<string>

using namespace std;

class Solution{
  public:
    bool isPalind(const string &s,int l,int r) {
      while(l < r) {
        if(s[l] != s[r]) {
          return false;
        }
        l++;
        r--;
      }
      return true;
    }

     string longestPalindrome(string s) {
       int len = s.length();
       int begin = 0;
       int maxLen = 1;
       for(int i=0; i<len -1; i++) {
          for(int j=i+1; j<len; j++) {
            // 上下循环进行错开，然后依次遍历所有的子串，判断这个子串的是不是回文字符串
            // 如果是回文字符串，则我们就记录他的长度
            // 并且他的长度如果比之前的长度更长。我们就更新begin 和 maxLen;
            /// 注意：这个地方不要过早的计算substr的拷贝。而是将所有的
            if(j -i+1 >maxLen  && isPalind(s,i,j)) {
              begin = i;
              maxLen = j-i+1;
            }
          }
       }
       return s.substr(begin,maxLen);
     }
};


void testCases(){
  string s = "cbbd";
  Solution so;
  so.longestPalindrome(s);
}

int main() {
  testCases();
}

