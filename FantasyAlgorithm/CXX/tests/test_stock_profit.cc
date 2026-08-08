#include "alice_algorithm/stock_profit.h"

#include <cassert>
#include <vector>

int main() {
    using alice::algorithm::max_profit;
    assert(max_profit({7, 1, 5, 3, 6, 4}) == 7);
    assert(max_profit({1, 2, 3, 4, 5}) == 4);
    assert(max_profit({7, 6, 4, 3, 1}) == 0);
    assert(max_profit({}) == 0);
    return 0;
}
