
#include "vision/core/result/VisionResult.h"
#include <sstream>

using namespace std;

namespace aura::vision {

VisionResult::VisionResult(RtConfig *cfg) : AbsVisionResult() {
    mSource = cfg->sourceId;
    mFaceResult = new FaceResult(cfg);
    mGestureResult = new GestureResult(cfg);
    mLivingResult = new LivingResult(cfg);
    mFrameInfo = new FrameInfo();
    mBodyResult = new BodyResult(cfg);
    mUseInternalMem = V_F_TO_BOOL(cfg->useInternalMem);

#ifdef ENABLE_PERF
    mPerfUtil = new PerfUtil(mSource);
#endif
}

VisionResult::~VisionResult() {
    mSource = 0;
    delete mFaceResult;
    delete mGestureResult;
    delete mLivingResult;
    delete mBodyResult;
    delete mFrameInfo;
#ifdef ENABLE_PERF
    delete mPerfUtil;
#endif

	mFaceResult = nullptr;
	mGestureResult = nullptr;
    mLivingResult = nullptr;
	mBodyResult = nullptr;
	mFrameInfo = nullptr;
	mPerfUtil = nullptr;
}

FaceResult *VisionResult::getFaceResult() const {
    return mFaceResult;
}

void VisionResult::setFaceResult(FaceResult *result) {
    if (mFaceResult == result) {
        return;
    }
    if (mUseInternalMem) {
        delete mFaceResult;
    }
	mFaceResult = result;
}

GestureResult *VisionResult::getGestureResult() const {
    return mGestureResult;
}

void VisionResult::setGestureResult(GestureResult *result) {
    if (mGestureResult == result) {
        return;
    }
    if (mUseInternalMem) {
        delete mGestureResult;
    }
	mGestureResult = result;
}

FrameInfo *VisionResult::getFrameInfo() const {
    return mFrameInfo;
}

void VisionResult::setFrameInfo(FrameInfo *info) {
    if (mFrameInfo == info) {
        return;
    }
    if (mUseInternalMem) {
        delete mFrameInfo;
    }
	mFrameInfo = info;
}

LivingResult *VisionResult::getLivingResult() const {
    return mLivingResult;
}

void VisionResult::setLivingResult(LivingResult *result) {
    if (mLivingResult == result) {
        return;
    }
    if (mUseInternalMem) {
        delete mLivingResult;
    }
	mLivingResult = result;
}


BodyResult *VisionResult::getBodyResult() const {
    return mBodyResult;
}

void VisionResult::setBodyResult(BodyResult *result) {
    if (mBodyResult == result) {
        return;
    }
    if (mUseInternalMem) {
        delete mBodyResult;
    }
	mBodyResult = result;
}

void VisionResult::clear() {
    mSource = 0;
    AbsVisionResult::clear();
    mFaceResult->clear();
    mGestureResult->clear();
    mBodyResult->clear();
    mLivingResult->clear();
    mFrameInfo->clear();
#ifdef ENABLE_PERF
    if (mPerfUtil != nullptr) {
        mPerfUtil->clear();
    }
    PerfUtil::global()->clear();
#endif
}

void VisionResult::clearAll() {
    mSource = 0;
    AbsVisionResult::clearAll();
    mFaceResult->clearAll();
    mGestureResult->clearAll();
    mBodyResult->clearAll();
    mLivingResult->clearAll();
    mFrameInfo->clearAll();
#ifdef ENABLE_PERF
    if (mPerfUtil != nullptr) {
        mPerfUtil->clear();
    }
    PerfUtil::global()->clear();
#endif
}

short VisionResult::tag() const {
    return TAG;
}

bool VisionResult::hasFace() const {
    return !mFaceResult->noFace();
}

short VisionResult::faceCount() const {
    return mFaceResult->faceCount();
}

bool VisionResult::isFaceOccluded() const {
    // mFaceResult->faceOccluded() 过时方法已注销
    // return mFaceResult->faceOccluded();
    return false;
}

bool VisionResult::isFaceLive() const {
    // mFaceResult->faceLive() 过时方法已注销
    // return mFaceResult->faceLive();
    return false;
}

bool VisionResult::hasGesture() const {
    return !mGestureResult->noGesture();
}

bool VisionResult::hasBody() const {
    return !mBodyResult->noBody();
}

PerfUtil *VisionResult::getPerfUtil() const {
    return mPerfUtil;
}

void VisionResult::toString(std::stringstream &ss) {
    mFaceResult->toString(ss);
    mGestureResult->toString(ss);
    mBodyResult->toString(ss);
    mLivingResult->toString(ss);
    mFrameInfo->toString(ss);
}

} // namespace aura::vision
