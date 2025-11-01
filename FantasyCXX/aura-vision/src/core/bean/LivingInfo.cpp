
#include "vision/core/bean/LivingInfo.h"

using namespace std;

namespace aura::vision {

LivingInfo::LivingInfo() noexcept {
    clearAll();
}

LivingInfo::LivingInfo(const LivingInfo &info) noexcept {
    copy(info);
}

LivingInfo::LivingInfo(LivingInfo &&info) noexcept {
    copy(info);
}

LivingInfo &LivingInfo::operator=(LivingInfo &&info) noexcept {
    copy(info);
    return *this;
}

LivingInfo &LivingInfo::operator=(const LivingInfo &info) noexcept {
    if (&info != this) {
        copy(info);
    }
    return *this;
}

void LivingInfo::copy(const LivingInfo &info) {
    id = info.id;
    rectConfidence = info.rectConfidence;
    rectCenter.copy(info.rectCenter);
    rectLT.copy(info.rectLT);
    rectRB.copy(info.rectLT);

    livingType = info.livingType;
    livingTypeSingle = info.livingTypeSingle;
}

void LivingInfo::clearAll() {
    clear();
    id = 0;
    rectCenter.clear();
    rectLT.clear();
    rectRB.clear();
}

void LivingInfo::clear() {
    rectConfidence = 0;
    livingType = F_CATEGORY_NONE;
    livingTypeSingle = F_CATEGORY_NONE;
}

void LivingInfo::toString(std::stringstream &ss) const {
    ss << "[LivingInfo] ------------------------------" << endl;
    ss << "id : " << id << endl;
    ss << "livingType : " << livingType << endl;
    ss << "livingTypeSingle : " << livingTypeSingle << endl;
    ss << "rectConfidence : " << rectConfidence << endl;
    ss << "rectCenter : "; rectCenter.toString(ss); ss << endl;
    ss << "rectLT : "; rectLT.toString(ss); ss << endl;
    ss << "rectRB : "; rectRB.toString(ss); ss << endl;
}

}
