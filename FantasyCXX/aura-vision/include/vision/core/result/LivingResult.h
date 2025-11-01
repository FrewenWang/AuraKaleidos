//
// Created by Li,Wendong on 2019-01-17.
//

#ifndef VISION_LIVING_RESULT_H
#define VISION_LIVING_RESULT_H

#include "AbsVisionResult.h"
#include "vision/config/runtime_config/RtConfig.h"
#include "vision/core/bean/LivingInfo.h"
#include "vision/core/common/VConstants.h"
#include <sstream>

namespace aura::vision {

/**
 * @brief Detection result of living request
 */
class LivingResult : public AbsVisionResult {
public:
    explicit LivingResult(RtConfig *cfg);

    /**
     * 析构函数
     */
    ~LivingResult() override;

    /**
     * @brief Get the tag of the result
     * @return
     */
    short tag() const override;

    /**
     * @brief Clear the frame only
     */
    void clear() override;

    /**
     * @brief Clear all data
     */
    void clearAll() override;

    /**
     * @brief resize gestureInfos count
     * @param count
     */
    void resize(int count);

    void toString(std::stringstream &ss) const;

public:
    int livingCount;             /// max detected gesture count
    LivingInfo **livingInfos;       /// detected living info

private:
    bool useInternalMem;
    static const short TAG = ABILITY_LIVING;
};

} // namespace vision

#endif //VISION_LIVING_RESULT_H
