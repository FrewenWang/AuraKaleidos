//
// Created by frewen on 22-9-27.
//
#pragma once
#include "structs.h"

namespace aura::aura_cv {

class bbox_util {
public:
    /**
     * 计算两个检测框的重合度（iou = Intersection Over Union）
     * @param r1
     * @param r2
     * @return
     */
    static float iou(const VRect &r1, const VRect &r2);
};

} // namespace aura::aura_cv