
#include "vision/core/bean/FrameInfo.h"

using namespace std;

namespace aura::vision {

FrameInfo::FrameInfo() noexcept {
    clearAll();
}

FrameInfo::FrameInfo(const FrameInfo &info) noexcept {
    copy(info);
}

FrameInfo::FrameInfo(FrameInfo &&info) noexcept {
    copy(info);
}

FrameInfo &FrameInfo::operator=(FrameInfo &&info) noexcept {
    copy(info);
    return *this;
}

FrameInfo &FrameInfo::operator=(const FrameInfo &info) noexcept {
    if (&info != this) {
        copy(info);
    }
    return *this;
}

void FrameInfo::copy(const FrameInfo &info) {
    tag = info.tag;
    width = info.width;
    height = info.height;
    frame = info.frame;
    timestamp = info.timestamp;
    brightness = info.brightness;
    state_frame_lightness = info.state_frame_lightness;
    state_spoof = info.state_spoof;
    state_frame_occlusion = info.state_frame_occlusion;
}

void FrameInfo::clearAll() {
    clear();
    tag = "";
}

void FrameInfo::clear() {
    width = 0;
    height = 0;
    frame = nullptr;
    timestamp = 0;
    brightness = 0;
    state_frame_lightness = 0;
    state_spoof = 0;
    state_frame_occlusion = 0;
}

void FrameInfo::toString(std::stringstream &ss) const {
    ss << "\n[FrameInfo] ========================================\n";
    ss << "tag : " << tag << endl;
    ss << "width : " << width << endl;
    ss << "height : " << height << endl;
//    ss << "timestamp : " << timestamp << endl;
    ss << "brightness : " << brightness << endl;
    ss << "state_frame_lightness : " << state_frame_lightness << endl;
    ss << "state_spoof : " << state_spoof << endl;
    ss << "state_frame_occlusion : " << state_frame_occlusion << endl;
}

}
