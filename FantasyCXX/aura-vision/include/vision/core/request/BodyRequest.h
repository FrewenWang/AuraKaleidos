#ifndef VISION_HUMAN_POSE_REQUEST_H
#define VISION_HUMAN_POSE_REQUEST_H

#include "AbsVisionRequest.h"
#include "vision/core/common/VConstants.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

class RtConfig;

/**
 * @brief Human Pose detection request
 */
class BodyRequest : public AbsVisionRequest {
public:
    explicit BodyRequest(RtConfig* cfg);
    ~BodyRequest() override;

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
     * 最大的肢体检测的数量
     */
    int bodyCount;

private:
    static const short TAG = ABILITY_BODY;
};

} // namespace vision

#endif //VISION_HUMAN_POSE_REQUEST_H
