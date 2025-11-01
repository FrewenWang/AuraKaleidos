
#include "vision/core/request/VisionRequest.h"
#include "vision/util/VaAllocator.h"
#include "vacv/cv.h"
#include "vacv/resize.h"
#include "vision/config/runtime_config/RtConfig.h"
#include "util/DebugUtil.h"
#include "vision/core/bean/VehicleInfo.h"
#include "util/TensorConverter.h"

namespace aura::vision {

static const char *VISION_TAG = "VisionRequest";
// 暂时设置threshold==10
static const int threshold = 10;

VisionRequest::VisionRequest(RtConfig *cfg) : AbsVisionRequest() {
    _mgr_id = MANAGER_ID;
    mFaceRequest = new FaceRequest(cfg);
    mGestureRequest = new GestureRequest(cfg);
    mBodyRequest = new BodyRequest(cfg);
    mSingleDetectType = 0;
    mSpecificAbility = ABILITY_UNKNOWN;
    // 实例化车辆请求信息
    vehicleInfo = std::make_shared<VehicleInfo>();
    if (cfg) {
		mRtConfig = cfg;
		width = static_cast<short>(mRtConfig->frameWidth);
		height = static_cast<short>(mRtConfig->frameHeight);
		format = static_cast<FrameFormat>(mRtConfig->frameFormat);
		switch (format) {
            // FrameFormat::YUV_422_UYVY 是BGR 图片转化为 UYVY 格式。
            // FrameFormat::UYVY_BUFFER 是直接读取 UYVY 原始数据文件
            case FrameFormat::YUV_422_UYVY:
            case FrameFormat::UYVY_BUFFER:{
				frameTensor = VTensor(width, height, 2, nullptr, INT8);
				grayCovertFormat = FrameConvertFormat::COLOR_YUV2GRAY_UYVY;
#if defined(BUILD_QNX) and defined (BUILD_FASTCV)
				// 高通 fastcv 不支持 COLOR_YUV2BGR_UYVY，暂用 COLOR_YUV2RGB2BGR 两部操作代替
				bgrCovertFormat = FrameConvertFormat::COLOR_YUV2RGB2BGR;
#else
                bgrCovertFormat = FrameConvertFormat::COLOR_YUV2BGR_UYVY; // Linux 版本暂时用
#endif
                rgbCovertFormat = FrameConvertFormat::COLOR_YUV2RGB_UYVY;
				break ;
			}
			case FrameFormat::YUV_420_NV21: {
				frameTensor = VTensor(width, height * 3 / 2, nullptr, INT8);
				grayCovertFormat = FrameConvertFormat::COLOR_YUV2GRAY_NV21;
				bgrCovertFormat = FrameConvertFormat::COLOR_YUV2BGR_NV21;
				rgbCovertFormat = FrameConvertFormat::COLOR_YUV2RGB_NV21;
				break;
			}
            case FrameFormat::BGR: {
                frameTensor = VTensor(width, height, 3, nullptr, INT8);
                grayCovertFormat = FrameConvertFormat::COLOR_BGR2GRAY;
                bgrCovertFormat = FrameConvertFormat::COPY_BUFFER;
                rgbCovertFormat = FrameConvertFormat::COLOR_BGR2RGB;
                break;
            }
            // 考虑暂时未用到的格式类型 YUV_420_NV12  |  RGB
            case FrameFormat::YUV_420_NV12:
            case FrameFormat::RGB: {
                break;
            }
			case FrameFormat::UNKNOWN: {
				throw std::runtime_error("Fail to create VisionRequest: Unknown FrameFormat");
			}
		}
    }
    mVarianceSize = va_cv::VSize(width / 20, height / 20);
}

VisionRequest::~VisionRequest() {
    delete mFaceRequest;
    delete mGestureRequest;
    delete mBodyRequest;

    mFaceRequest = nullptr;
    mGestureRequest = nullptr;
    mBodyRequest = nullptr;
}

bool VisionRequest::verify() {
    if (frame == nullptr) {
        VLOGE(VISION_TAG, "vision request verify failed frame null");
        return false;
    }
    if (mRtConfig) {
        if (width <= 0) {
			width = static_cast<short>(mRtConfig->frameWidth);
        }
        if (height <= 0) {
			height = static_cast<short>(mRtConfig->frameHeight);
        }
    } else {
        VLOGE(VISION_TAG, "vision request verify failed runtime config null");
        return false;
    }
    if (width <= 0 || height <= 0) {
        VLOGE(VISION_TAG, "vision request verify failed height: %d, width: %d", width, height);
        return false;
    }
    mFaceRequest->frame = frame;
    mFaceRequest->width = width;
    mFaceRequest->height = height;
    mGestureRequest->frame = frame;
    mGestureRequest->width = width;
    mGestureRequest->height = height;
    mBodyRequest->frame = frame;
    mBodyRequest->width = width;
    mBodyRequest->height = height;
    return true;
}

FaceRequest *VisionRequest::getFaceRequest() const {
    return mFaceRequest;
}

void VisionRequest::setFaceRequest(FaceRequest* request) {
    if (mFaceRequest == request) {
        return;
    }
    delete mFaceRequest;
    mFaceRequest = request;
}

GestureRequest *VisionRequest::getGestureRequest() const {
    return mGestureRequest;
}

void VisionRequest::setGestureRequest(GestureRequest *request) {
    if (mGestureRequest == request) {
        return;
    }
    delete mGestureRequest;
    mGestureRequest = request;
}

BodyRequest *VisionRequest::getBodyRequest() const {
    return mBodyRequest;
}

void VisionRequest::setBodyRequest(BodyRequest *request) {
    if (mBodyRequest == request) {
        return;
    }
    delete mBodyRequest;
    mBodyRequest = request;
}

short VisionRequest::tag() const {
    return TAG;
}

void VisionRequest::clear() {
    AbsVisionRequest::clear();
    mFaceRequest->clear();
    mGestureRequest->clear();
    mBodyRequest->clear();
	mSingleDetectType = 0;
	mSpecificAbility = ABILITY_UNKNOWN;

	isConvertGray = false;
	isConvertBGR = false;
	isConvertRGB = false;
    mIsConvertCamLightType = false;
	// Clear 中未清除 gray / bgr / rgb，是为了避免内存重复申请，但同时也会持久保存当前帧的图像
}

void VisionRequest::clearAll() {
	AbsVisionRequest::clearAll();
	mFaceRequest->clearAll();
	mGestureRequest->clearAll();
	mBodyRequest->clearAll();
	mSingleDetectType = 0;

    isConvertGray = false;
    isConvertBGR = false;
    isConvertRGB = false;
    mIsConvertCamLightType = false;
	gray.release();
	bgr.release();
	rgb.release();

#if defined(BUILD_QNX) and defined (BUILD_FASTCV)
	VaAllocator::deallocateInFcv(y);
	VaAllocator::deallocateInFcv(u);
	VaAllocator::deallocateInFcv(v);
	VaAllocator::deallocateInFcv(uv);
#endif

}

int VisionRequest::get_single_detect_type() {
    return mSingleDetectType;
}

void VisionRequest::set_single_detect_type(int type) {
	mSingleDetectType = type;
}

void VisionRequest::set_specific_ability(AbilityId id) {
	mSpecificAbility = id;
}

AbilityId VisionRequest::specific_ability() const {
    return mSpecificAbility;
}

bool VisionRequest::specific_detection() const {
    return mSpecificAbility != ABILITY_UNKNOWN;
}

void VisionRequest::setFrame(unsigned char *f) {
    frame = f;
	frameTensor.data = frame;
	isConvertGray = false;
	isConvertBGR = false;
	isConvertRGB = false;
    mIsConvertCamLightType = false;
}

unsigned char* VisionRequest::getFrame() {
    return frame;
}

bool VisionRequest::hasFrame() {
    return frame != nullptr;
}

//void VisionRequest::makeFrame() {
//	mFrameInfo = VFrameInfo(width, height, format, frame);
//	mFrameInfo.gray = gray;
//	mFrameInfo.bgr = bgr;
//	mFrameInfo.rgb = rgb;
//}

//VFrameInfo& VisionRequest::getFrame() {
//    return mFrameInfo;
//}

bool VisionRequest::hasGray() {
    return !gray.empty();
}

bool VisionRequest::hasBgr() {
	return !bgr.empty();
}

bool VisionRequest::hasRgb() {
    return !rgb.empty();
}

void VisionRequest::convertFrameToGray() {
    if (isConvertGray) {
        return;
    }

	if (gray.empty()) {
#if defined(BUILD_QNX) and defined (BUILD_FASTCV)
		gray.create(width, height, 1, INT8, NHWC, FCV_ALLOC);
		uv = VaAllocator::allocateInFcv(width * height * 2, 16); // 用于格式转换临时变量
		gray.uv = uv;
#else
		gray.create(width, height, 1, INT8, NHWC);
#endif
	}
    if (grayCovertFormat == FrameConvertFormat::COPY_BUFFER) {
        gray = frameTensor;
    } else {
    	va_cv::cvt_color(frameTensor, gray, grayCovertFormat);
    }
    // DBG_PRINT_ARRAY((char *) gray.data, 100, "request_gray");
    isConvertGray = true;
}

void VisionRequest::convertFrameToBGR() {
	if (isConvertBGR) {
		return;
	}

	if (bgr.empty()) {
#if defined(BUILD_QNX) and defined (BUILD_FASTCV)
		bgr.create(width, height, 3, INT8, NHWC, FCV_ALLOC);
#else
		bgr.create(width, height, 3, INT8, NHWC);
#endif
	}
    if (bgrCovertFormat == FrameConvertFormat::COPY_BUFFER) {
        bgr = frameTensor;
    } else {
        va_cv::cvt_color(frameTensor, bgr, bgrCovertFormat);
    }
    isConvertBGR = true;
}

void VisionRequest::convertFrameToRGB() {
	if (isConvertRGB) {
		return;
	}
	if (rgb.empty()) {
#if defined(BUILD_QNX) and defined (BUILD_FASTCV)
		rgb.create(width, height, 3, INT8, NHWC, FCV_ALLOC);
#else
		rgb.create(width, height, 3, INT8, NHWC);
#endif
	}
    if (rgbCovertFormat == FrameConvertFormat::COPY_BUFFER) {
        rgb = frameTensor;
    } else {
        va_cv::cvt_color(frameTensor, rgb, rgbCovertFormat);
    }
    isConvertRGB = true;
}

int VisionRequest::getSource() {
    return mRtConfig->sourceId;
}

bool VisionRequest::checkIsIrOrRgb() {
    if(mIsConvertCamLightType == true || mRtConfig->sourceId != Source::SOURCE_2) {
        return true;
    }

    if (mRtConfig->sourceId == Source::SOURCE_2 &&
         mRtConfig->cameraLightSwapMode == CameraLightSwapMode::CAMERA_SWAP_MODE_MANUAL) {
        mIsConvertCamLightType = true;
        return false;
    }

    mIsConvertCamLightType = true;

    this->convertFrameToBGR();

    mVarianceImage.release();
    va_cv::Resize::resize(this->bgr, mVarianceImage, mVarianceSize, 0, 0, cv::INTER_AREA);
    if (mVarianceImage.empty()) {
        return false;
    }
    mVarianceImage = mVarianceImage.changeDType(FP32);
    bool varianceStatus = va_cv::variance(this->mVarianceImage, mVariance);

    if(varianceStatus == false){
        return false;
    }
    CameraLightType cameraLightType;
    if((int)mVariance <= threshold) {
        cameraLightType = CameraLightType::CAMERA_LIGHT_TYPE_IR;
    } else {
        cameraLightType = CameraLightType::CAMERA_LIGHT_TYPE_RGB;
    }

    VLOGI(VISION_TAG, "cameraLightType current mode:%d; old mode:%d; mVariance:%f",
          (int)cameraLightType, (int)mRtConfig->cameraLightType, mVariance);
    if ((CameraLightType)mRtConfig->cameraLightType != cameraLightType) {
        mRtConfig->cameraLightType = cameraLightType;
    }

    return true;
}

} // namespace aura::vision
