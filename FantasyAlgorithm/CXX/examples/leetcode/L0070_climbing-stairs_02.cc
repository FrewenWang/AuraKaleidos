//
// Created by Frewen.Wang on 25-3-5.
//
#include<vector>

using namespace std;

class Solution {
public:
  int climbStairs(int n) {
     int f_1 = 0, f_2 = 0,ans = 1;
     for(int i = 0;i <= n;i++) {
       f_1 = f_2;
       f_2 = ans;
       ans = f_1+f_2;
     }
     return ans;
  }

};