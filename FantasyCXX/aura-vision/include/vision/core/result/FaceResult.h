//
// Created by Li,Wendong on 2019-01-17.
//

#ifndef VISION_FACE_RESULT_H
#define VISION_FACE_RESULT_H

#include "AbsVisionResult.h"
#include "vision/config/runtime_config/RtConfig.h"
#include "vision/core/bean/FaceInfo.h"
#include <sstream>

namespace aura::vision {

/**
 * @brief Detection result of face request
 */
class FaceResult : public AbsVisionResult {
public:
    explicit FaceResult(RtConfig *cfg);

    ~FaceResult();

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
     * @brief Copy from another detection result
     * @param src
     */
    void copy(AbsVisionResult *src) override;

    /**
     * @brief Whether the face is occluded by mask or something else
     * @return
     */

    // 过时方法注销
//    [[deprecated("deprecated method")]]
//    bool faceOccluded() const;

    /**
     * @brief Whether the face is spoof
     * @return
     */
    // 过时方法注销
//    [[deprecated("deprecated method")]]
//    bool faceLive() const;

    /**
     * @brief Check whether a face is detected
     * @return detected -- true; no face detected -- false;
     */
    bool noFace() const;

    /**
     * Check whether a face is detected
     * @see noFace
     * @return
     */
    bool hasFace() const;

    /**
     * @brief resize faceInfos count
     * @param count
     */
    void resize(int count);

    /**
     * 获取 FaceResult 的人脸数量
     * @return
     */
    short faceCount();

    void toString(std::stringstream &ss) const;

public:
    int faceMaxCount;              /// max face count of vision-ability support
    FaceInfo **faceInfos;          /// detected face info

private:
    bool useInternalMem;
    static const short TAG = ABILITY_FACE;
};

} // namespace vision

#endif //VISION_FACE_RESULT_H
