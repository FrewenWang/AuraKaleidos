//
// Created by Li,Wendong on 2019-01-01.
//

#pragma once

#include "AbsVisionResult.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/bean/GestureInfo.h"
#include "vision/core/bean/FrameInfo.h"
#include "vision/core/bean/BodyInfo.h"
#include "vision/core/bean/LivingInfo.h"
#include "FaceResult.h"
#include "GestureResult.h"
#include "BodyResult.h"
#include "LivingResult.h"
#include "vision/util/ObjectPool.hpp"
#include <string>

namespace aura::vision {

/**
 * @brief Detection results of all the requests
 */
class VisionResult : public AbsVisionResult, public ObjectPool<VisionResult> {
public:
    explicit VisionResult(RtConfig* cfg);
    ~VisionResult() override;

    /**
     * @brief Clear the frame only
     */
    void clear() override;

    /**
     * @brief Clear all data
     */
    void clearAll() override;

    /**
     * @brief Get the tag of the request
     * @return
     */
    short tag() const override;

    /**
     * @brief Get the face result
     * @return pointer to the face result
     */
    FaceResult *getFaceResult() const;

    /**
     * @brief setFaceResult
     * @param result
     */
    void setFaceResult(FaceResult* result);

    /**
     * @brief Get the gesture result
     * @return pointer to the gesture result
     */
    GestureResult *getGestureResult() const;

    /**
     * @brief setGestureResult
     * @param result
     */
    void setGestureResult(GestureResult* result);

    /**
     * @brief Get the frame info
     * @return pointer to the frame info
     */
    FrameInfo *getFrameInfo() const;

    /**
     * @brief setFrameInfo
     * @param info
     */
    void setFrameInfo(FrameInfo* info);

    /**
     * @brief Get the living result
     * @return pointer to the living result
     */
    LivingResult *getLivingResult() const;

    /**
     * @brief setLivingInfo
     * @param result
     */
    void setLivingResult(LivingResult *result);

    /**
     * @brief Get the body result
     * @return pointer to body result
     */
    BodyResult *getBodyResult() const;

    /**
     * @brief setBodyResult
     * @param result
     */
    void setBodyResult(BodyResult* result);

    /**
     * @brief Whether there is a face in the image
     * @return
     */
    bool hasFace() const;

    /**
     * @brief detect face count of the vision result
     * @return
     */
    short faceCount() const;

    /**
     * @brief Whether the face is occluded by mask or something else
     * @return
     */
    bool isFaceOccluded() const;

    /**
     * @brief Whether the face is spoof
     * @return
     */
    bool isFaceLive() const;

    /**
     * @brief Whether there is a hand gesture int the image
     * @return
     */
    bool hasGesture() const;

    /**
     * @brief Whether there is a body in the image
     * @return
     */
    bool hasBody() const;

    /**
     * @brief Get the pointer to the perfUtil
     * @return pointer to perfUtil
     */
    PerfUtil *getPerfUtil() const;

    void toString(std::stringstream &ss);

private:
    /**
     * 资源Source
     */
    int mSource = Source::SOURCE_UNKNOWN;
    FaceResult *mFaceResult = nullptr;
    GestureResult *mGestureResult = nullptr;
    LivingResult *mLivingResult = nullptr; // 猫狗宠物婴儿活体检测结果
    BodyResult *mBodyResult = nullptr;
	FrameInfo *mFrameInfo = nullptr;
	PerfUtil * mPerfUtil = nullptr;

    bool mUseInternalMem = false;

    static const short TAG = ABILITY_ALL;
};

} // namespace vision
