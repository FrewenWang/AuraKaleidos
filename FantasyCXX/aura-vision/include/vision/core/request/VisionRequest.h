//
// Created by Li,Wendong on 2018/12/27.
//
#pragma once

#include "AbsVisionRequest.h"
#include "vision/core/common/VFrame.h"
#include "FaceRequest.h"
#include "GestureRequest.h"
#include "BodyRequest.h"
#include "vision/util/ObjectPool.hpp"
#include "vision/core/bean/VehicleInfo.h"
#include "vacv/cv.h"

namespace aura::vision {

class RtConfig;

/**
 * @brief Request fed to VisionService as the input of detection
 */
class VisionRequest : public AbsVisionRequest, public ObjectPool<VisionRequest> {
public:
    explicit VisionRequest(RtConfig *cfg);

    ~VisionRequest() override;

    /**
     * @brief verify the request, e.g. frame data and size
     * @return whether the request is valid
     */
    bool verify() override;

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

    /**
     * @brief if set, only the set ability will be detected
     * @param id
     */
    void set_specific_ability(AbilityId id);

    /**
     * @brief get the specific ability
     * @return 0 will be returned if not set
     */
    AbilityId specific_ability() const;

    /**
     * @brief whether the request is for detecting single ability
     * @return
     */
    bool specific_detection() const;

    // todo@wangyan:
    [[deprecated]]
    int get_single_detect_type();

    // todo@wangyan:
    [[deprecated]]
    void set_single_detect_type(int type);

    /**
     * @brief get face request
     * @return pointer to face request
     */
    FaceRequest *getFaceRequest() const;

    /**
     * @brief set face request
     * @param request
     */
    void setFaceRequest(FaceRequest *request);

    /**
     * @brief get gesture request
     * @return pointer to gesture request
     */
    GestureRequest *getGestureRequest() const;

    /**
     * @brief set gesture request
     * @param request
     */
    void setGestureRequest(GestureRequest *request);

    /**
     * @brief get human_pose request
     * @return pointer to human_pose request
     */
    BodyRequest *getBodyRequest() const;

    /**
     * @brief set human pose request
     * @param request
     */
    void setBodyRequest(BodyRequest *request);

    /**
     * @brief make VFrameInfo from the input request
     */
//    void makeFrame();

    /**
     * @brief get the VFrameInfo
     * @return reference to the VFrameInfo
     */
//    VFrameInfo& getFrame();

    void setFrame(unsigned char *f) override;

    unsigned char *getFrame() override;

    bool hasFrame() override;

    bool hasGray();

    bool hasBgr();

    bool hasRgb();

    void convertFrameToGray();

    void convertFrameToBGR();

    void convertFrameToRGB();

    int getSource();

    bool checkIsIrOrRgb();

public:
    VTensor frameTensor;
    VTensor gray; // grey image
    VTensor bgr;  // rgb image, format is B-G-R
    VTensor rgb;

    // 方差
    float mVariance;
    vision::VTensor mVarianceImage;
    va_cv::VSize mVarianceSize;

    // 暂时设置为公开的，仅供服务层获取参数使用
    RtConfig *mRtConfig = nullptr; // VisionService运行时配置
    /**  车辆信息 */
    std::shared_ptr<VehicleInfo> vehicleInfo;

private:
    FaceRequest *mFaceRequest = nullptr;
    GestureRequest *mGestureRequest = nullptr;
    BodyRequest *mBodyRequest = nullptr;
    int mSingleDetectType;
    AbilityId mSpecificAbility;

    FrameConvertFormat grayCovertFormat;
    FrameConvertFormat bgrCovertFormat;
    FrameConvertFormat rgbCovertFormat;

    unsigned char *y = nullptr;
    unsigned char *u = nullptr;
    unsigned char *v = nullptr;
    unsigned char *uv = nullptr;

    bool isConvertGray = false;
    bool isConvertBGR = false;
    bool isConvertRGB = false;
    bool mIsConvertCamLightType = false;

    static const short TAG = ABILITY_ALL;
    static const short MANAGER_ID = ABILITY_ALL;
};

} // namespace aura::vision
