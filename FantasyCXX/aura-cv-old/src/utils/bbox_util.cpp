//
// Created by frewen on 22-9-27.
//

#include "aura/cv/utils/BBoxUtil.h"
#include <cmath>
#include <algorithm>

namespace aura::aura_cv {

float BBoxUtil::iou(const VRect &r1, const VRect &r2) {
    auto x01 = r1.left;
    auto y01 = r1.top;
    auto x02 = r1.right;
    auto y02 = r1.bottom;

    auto x11 = r2.left;
    auto y11 = r2.top;
    auto x12 = r2.right;
    auto y12 = r2.bottom;

    auto dist_center_x = std::fabs((x01 + x02) / 2.f - (x11 + x12) / 2.f);
    auto dist_center_y = std::fabs((y01 + y02) / 2.f - (y11 + y12) / 2.f);
    auto dist_sum_x = (std::fabs(x01 - x02) + std::fabs(x11 - x12)) / 2.f;
    auto dist_sum_y = (std::fabs(y01 - y02) + std::fabs(y11 - y12)) / 2.f;
    if (dist_center_x > dist_sum_x || dist_center_y > dist_sum_y) {
        return 0.f;
    }
    auto cols = std::min(x02, x12) - std::max(x01, x11);
    auto rows = std::min(y02, y12) - std::max(y01, y11);
    // 计算重合区域的面积
    auto intersection = cols * rows;
    auto area1 = (x02 - x01) * (y02 - y01);
    auto area2 = (x12 - x11) * (y12 - y11);
    // 重合区域面积/(区域一 + 区域二 - 重合区域)
    auto coincide = intersection / (area1 + area2 - intersection);
    return coincide;
}

} // namespace aura::aura_cv
