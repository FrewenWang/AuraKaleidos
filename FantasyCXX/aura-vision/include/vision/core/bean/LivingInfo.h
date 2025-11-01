//
// Created by wangzhijiang on 2022/4/26.
//
#ifndef VISION_LIVING_INFO_H
#define VISION_LIVING_INFO_H

#include "vision/core/common/VStructs.h"

namespace aura::vision {

enum LivingCategory : short {
    F_CATEGORY_NONE = -1,
    F_CATEGORY_CAT = 0,
    F_CATEGORY_DOG = 1,
    F_CATEGORY_BABY = 2,
    F_CATEGORY_TYPE_NUM
};

/**
 * 猫狗婴儿活体信息
 */
class LivingInfo {
public:
    int id;                 /// the detected living index
    short livingType;       /// -1-none 0-cat, 1-dog, 2-baby  3-matchstick men(Retain)
    short livingTypeSingle; /// livingCategory single frame-1-none 0-cat,1-dog,2-baby,3-matchstick men(ONLY TEST)
    float rectConfidence;   /// living rect confidence
    VPoint rectCenter;      /// living rect center point
    VPoint rectLT;          /// the left-top point of living rect
    VPoint rectRB;          /// the right-bottom point of living rect
    /**
     * 判断是否检测到活体生物
     * @return
     */
    bool hasLiving() { return id > 0; }
    void copy(const LivingInfo &info);
    void clearAll();
    void clear();

    LivingInfo() noexcept;
    LivingInfo(const LivingInfo &) noexcept;
    LivingInfo(LivingInfo &&) noexcept;
    LivingInfo &operator=(const LivingInfo &) noexcept;
    LivingInfo &operator=(LivingInfo &&) noexcept;
    ~LivingInfo() = default;
    void toString(std::stringstream &ss) const;
};

} // namespace vision
#endif // VISION_LIVING_INFO_H
