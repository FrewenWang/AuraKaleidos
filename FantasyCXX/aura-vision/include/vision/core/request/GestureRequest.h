//
// Created by Li,Wendong on 2019-01-17.
//

#ifndef VISION_GESTURE_REQUEST_H
#define VISION_GESTURE_REQUEST_H

#include "AbsVisionRequest.h"
#include "vision/core/common/VConstants.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

class RtConfig;

/**
 * @brief Gesture detection request
 */
class GestureRequest : public AbsVisionRequest {
public:
    explicit GestureRequest(RtConfig* cfg);
    ~GestureRequest() override;

    /**
     * @brief clear the frame only
     */
    void clear() override;

    /**
     * @brief clear all data
     */
    void clearAll() override;

    /**
     * @brief get the tag of the request
     * @return
     */
    short tag() const override;

public:
    /**
     *
     */
    int gestureCount; /// max gesture detection number

private:
    static const short TAG = ABILITY_GESTURE;
};

} // namespace vision

#endif //VISION_GESTURE_REQUEST_H
