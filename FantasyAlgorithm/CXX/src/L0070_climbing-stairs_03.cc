//
// Created by Frewen.Wang on 25-3-5.
//
#include<vector>

using namespace std;

/**
* 这个我们就可以思考一下，假设我们不用数组来进行存储所有的结果
* 那么其实我们只需要一直保存后面三个结果 f(n)、f(n-1)、f(n-2)
*/
class Solution {
public:
  int climbStairs(int n) {
    // 我们首先定义一个数组，来存储n阶的台阶的需要的步数
    vector<int> ansList(n+1,0);
    for(int i=1;i<=n;i++) {
      if(i == 1) {
        ansList[i] = 1;
      } else if( i==2) {
        ansList[i] = 2;
      } else {
        ansList[i] = ansList[i-2] + ansList[i-1];
      }
    }
    return ansList[n];
  }

};