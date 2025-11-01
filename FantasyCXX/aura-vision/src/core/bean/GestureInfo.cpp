
#include "vision/core/bean/GestureInfo.h"
#include <sstream>

using namespace std;

namespace aura::vision {

GestureInfo::GestureInfo() noexcept
    : id(0), rectConfidence(0.f), rectType(RectType::G_RECT_TYPE_UNKNOWN), landmarkConfidence(0.f), staticTypeSingle(0),
      typeConfidence(0.f), staticType(0), dynamicType(0), dynamicTypeSingle(0) {
    clear_all();
}

GestureInfo::GestureInfo(const GestureInfo& info) noexcept
    : GestureInfo() {
    copy(info);
}

GestureInfo::GestureInfo(GestureInfo&& info) noexcept
    : GestureInfo() {
    copy(info);
}

GestureInfo& GestureInfo::operator= (const GestureInfo& info) noexcept {
    if (&info != this) {
        copy(info);
    }
    return *this;
}

GestureInfo& GestureInfo::operator= (GestureInfo&& info) noexcept {
    copy(info);
    return *this;
}

void GestureInfo::clear_all() {
    id = 0; // id 不能放在 clear() 中，会导致每帧检测到的数据无法复用到下一帧
    rectLT.clear();
    rectRB.clear();
    rectConfidence = 0;
    staticTypeSingle = 0;
    landmarkConfidence = 0;
    typeConfidence = 0;
    memset(landmark21, 0, 21 * sizeof(VPoint));
    clear();
}

void GestureInfo::clear() {
    rectType = G_RECT_TYPE_UNKNOWN;
    staticTypeSingle = GESTURE_NO_DETECT;
    staticType = GESTURE_NO_DETECT;
    dynamicType = GESTURE_DYNAMIC_NO_DETECT;
    dynamicTypeSingle = GESTURE_DYNAMIC_NO_DETECT;
    statePlayPhoneSingle = G_PLAY_PHONE_STATUS_NONE;
    statePlayPhone = G_PLAY_PHONE_STATUS_NONE;
    playPhoneVState.clear();
}

void GestureInfo::copy(const GestureInfo &info) {
    id = info.id;
    rectLT.copy(info.rectLT);
    rectRB.copy(info.rectRB);
    rectConfidence = info.rectConfidence;
    landmarkConfidence = info.landmarkConfidence;
    staticTypeSingle = info.staticTypeSingle;
    typeConfidence = info.typeConfidence;
    staticType = info.staticType;
    dynamicType = info.dynamicType;
    dynamicTypeSingle = info.dynamicTypeSingle;
    rectType = info.rectType;
    for (int i = 0; i < GESTURE_LM_21_COUNT; ++i) {
        landmark21[i].copy(info.landmark21[i]);
    }
}

float GestureInfo::rectWidth() {
    return rectRB.x - rectLT.x;
}

float GestureInfo::rectHeight() {
    return rectRB.y - rectLT.y;
}

void GestureInfo::toString(std::stringstream &ss) const {
    ss << "[GestureInfo] ------------------------------" << endl;
    ss << "id : " << id << endl;
    ss << "staticTypeSingle : " << staticTypeSingle << endl;
    ss << "rectConfidence : " << rectConfidence << endl;
    ss << "rectLT : "; rectLT.toString(ss); ss << endl;
    ss << "rectRB : "; rectRB.toString(ss); ss << endl;
    ss << "landmarkConfidence : " << landmarkConfidence << endl;
    ss << "landmark21[0] : "; landmark21[0].toString(ss); ss << endl;
    ss << "typeConfidence : " << typeConfidence << endl;
    ss << "staticType : " << staticType << endl;
    ss << "dynamicType : " << dynamicType << endl;
    ss << "statePlayPhoneSingle : " << statePlayPhoneSingle << endl;
    ss << "statePlayPhone : " << statePlayPhone << endl;
    ss << "playPhoneVState : "; playPhoneVState.toString(ss); ss << endl;
    ss << "dynamicTypeSingle : " << dynamicTypeSingle << endl;
}

} // namespace aura::vision
