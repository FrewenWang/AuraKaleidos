
#include "vision/core/bean/BodyInfo.h"
#include "vision/core/common/VConstants.h"
#include <sstream>

using namespace std;

namespace aura::vision {

BodyInfo::BodyInfo() noexcept
        : id(0) { clearAll();
}

BodyInfo::BodyInfo(const BodyInfo &info) noexcept
        : BodyInfo() {
    copy(info);
}

BodyInfo::BodyInfo(BodyInfo &&info) noexcept
        : BodyInfo() {
    copy(info);
}

BodyInfo &BodyInfo::operator=(const BodyInfo &info) noexcept {
    if (&info != this) {
        copy(info);
    }
    return *this;
}

BodyInfo &BodyInfo::operator=(BodyInfo &&info) noexcept {
    copy(info);
    return *this;
}

void BodyInfo::copy(const BodyInfo &info) {
    id = info.id;

    headShoulderRectCenter.copy(info.headShoulderRectCenter);
    headShoulderRectLT.copy(info.headShoulderRectLT);
    headShoulderRectRB.copy(info.headShoulderRectRB);
    for (int i = 0; i < BODY_LM_2D_12_COUNT; ++i) {
        bodyLandmark2D12[i].copy(info.bodyLandmark2D12[i]);
    }

    // memcpy(_landmark_2d, info._landmark_2d, LM_2D_7_COUNT * sizeof(VPoint));
}

void BodyInfo::clearAll() {
    clear();
}

void BodyInfo::clear() {
    id = 0;
    headShoulderRectCenter.clear();
    headShoulderRectLT.clear();
    headShoulderRectRB.clear();
    bodyLandmarkConfidence = 0.f;
    rectConfidence = 0.f;
    memset(bodyLandmark2D12, 0, BODY_LM_2D_12_COUNT * sizeof(VPoint));
    
    /** 标记当前肢体框是否有匹配上的人脸框，默认为false。每帧都进行还原 */
    hasMatchedFace = false;
    /// 进行推理计算肢体框的标记当前body所赋值的Index。默认为INT_MAX(因为默认无人脸排在最后。索引matchIndex的设置比较大的值)
    matchIndex = INT_MAX;
}

void BodyInfo::toString(std::stringstream &ss) const {
    ss << "[BodyInfo] ------------------------------" << endl;
    ss << "id : " << id << endl;
    ss << "rectConfidence : " << rectConfidence << endl;
    ss << "headShoulderRectCenter : "; headShoulderRectCenter.toString(ss); ss << endl;
    ss << "headShoulderRectLT : "; headShoulderRectLT.toString(ss); ss << endl;
    ss << "headShoulderRectRB : "; headShoulderRectRB.toString(ss); ss << endl;
}

} // namespace aura::vision
