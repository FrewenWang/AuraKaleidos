//
// Created by Frewen.Wang on 25-3-5.
//
#include<vector>

using namespace std;

class Solution {
public:
  int climbStairs(int n) {
    vector<int> memo(n+1,0);
    return climbStairsMemo(n,memo);

  }

  int climbStairsMemo(int n, vector<int>& memo) {
    if(memo[n] > 0 ) {
      return memo[n];
    }
    if(n == 1) {
      memo[n] = 1;
    } else if(n == 2) {
      memo[n] = 2;
    } else {
      memo[n] = climbStairsMemo(n-1,memo) + climbStairsMemo(n-2,memo);
    }
    return memo[n];
  }
};