//
// Created by frewen on 25-2-24.
//
#include<vector>
#include<algorithm>
#include<climits>

using namespace std;



class Solution {
public:
  /**
   * 这个买卖彩票的时机：
   * 其实思想逻辑很简单。就是进行一次遍历每天的卖彩票的价格，如果当前价格减去之前的最低价
   * 差值越大越说明，我们今日卖彩票是最合适的
   * 所以，我们依次遍历每天的价格。
   * 当前计算出来当前的价格减去历史最低价。得出来的值和best进行比较。
   * 谁大，就说明盈利空间还有进一步扩大的情况
   * 注意： 计算完成之后，我们还要是记录一下这个最低价（拿历史最低价和当前价格比对，谁小就记录谁。）
   */
  int maxProfit(vector<int> &prices) {
      int min_price = INT_MAX;
      int best = 0;
      for(int price:prices){
         best = max(best,price-min_price);
         min_price = min(min_price,price);
      }
      return best;
  }
};