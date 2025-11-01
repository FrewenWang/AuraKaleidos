
#include "vision/core/request/AbsVisionRequest.h"

namespace aura::vision {

AbsVisionRequest::AbsVisionRequest()
    : AbsVisionRequest(nullptr) {
}

AbsVisionRequest::AbsVisionRequest(unsigned char *f)
    : AbsVisionRequest(WIDTH_UNKNOWN, HEIGHT_UNKNOWN, f) {
}

AbsVisionRequest::AbsVisionRequest(short w, short h, unsigned char *f)
    : AbsVisionRequest(w, h, f, ABILITY_UNKNOWN) {
}

AbsVisionRequest::AbsVisionRequest(short w, short h, unsigned char *f, short i)
    : width(w),
	  height(h),
	  frame(f),
	  _mgr_id(i) {
}

AbsVisionRequest::~AbsVisionRequest() {
	frame = nullptr;
}

bool AbsVisionRequest::verify() {
    if (frame == nullptr) {
        return false;
    }
    // 目前的逻辑中暂时无法获取图像数据帧的款多
    // if (width <= 0) {
	// 	width = static_cast<short>(1280);
    // }
    // if (height <= 0) {
	// 	height = static_cast<short>(720);
    // }
    return true;
}

short AbsVisionRequest::tag() const {
    return ABILITY_UNKNOWN;
}

void AbsVisionRequest::clear() {
	frame = nullptr;
}

void AbsVisionRequest::clearAll() {
    clear();
}

void AbsVisionRequest::setFrame(unsigned char *f) {
    frame = f;
}

unsigned char * AbsVisionRequest::getFrame() {
    return frame;
}

bool AbsVisionRequest::hasFrame() {
    return frame != nullptr;
}

} // namespace aura::vision