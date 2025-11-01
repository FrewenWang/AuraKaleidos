#ifndef VISION_FACE_REQUEST_H
#define VISION_FACE_REQUEST_H

#include "AbsVisionRequest.h"
#include "vision/core/common/VConstants.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

class RtConfig;

/**
 * @brief Face detection request
 */
class FaceRequest : public AbsVisionRequest {
public:
    explicit FaceRequest(RtConfig* cfg);
    ~FaceRequest() override;

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
    int faceCount;            /// max detect face number
    VPoint* landmark2d106;   /// face landmark

private:
    static const short TAG = ABILITY_FACE;
};

} // namespace vision

#endif //VISION_FACE_REQUEST_H
