//
// Created by Li,Wendong on 2019-01-17.
//

#ifndef VISION_GESTURE_RESULT_H
#define VISION_GESTURE_RESULT_H

#include "AbsVisionResult.h"
#include "vision/config/runtime_config/RtConfig.h"
#include "vision/core/bean/GestureInfo.h"
#include <sstream>

namespace aura::vision {

/**
 * @brief Detection result of gesture request
 */
class GestureResult : public AbsVisionResult {
public:
    explicit GestureResult(RtConfig* cfg);
    ~GestureResult() override;

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
     * @brief Check whether a gesture is detected
     * @return detected -- true; no gesture -- false;
     */
    bool noGesture() const;

    /**
     * @brief resize gestureInfos count
     * @param count
     */
    void resize(int count);

    /**
     * 获取 GestureResult 的检测到人脸数量
     * @return
     */
    short gestureCount();

    void toString(std::stringstream &ss) const;

public:
    int gestureMaxCount;             /// max detected gesture count
    GestureInfo**gestureInfos;       /// detected gesture info

private:
    bool useInternalMem;
    static const short TAG = ABILITY_GESTURE;
};

} // namespace vision

#endif //VISION_GESTURE_RESULT_H
