#include "alice_algorithm/stock_profit.h"

namespace alice::algorithm {

int max_profit(const std::vector<int>& prices) {
    int result = 0;
    for (std::size_t index = 1; index < prices.size(); ++index) {
        if (prices[index] > prices[index - 1]) {
            result += prices[index] - prices[index - 1];
        }
    }
    return result;
}

}  // namespace alice::algorithm
