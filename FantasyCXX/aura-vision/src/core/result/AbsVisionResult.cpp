

#include "vision/core/result/AbsVisionResult.h"
#include "vision/core/common/VConstants.h"

namespace aura::vision {

std::unordered_map<short, std::list<AbsVisionResult * >> AbsVisionResult::resultsPool;

AbsVisionResult::AbsVisionResult() {
    errorCode = SUCCESS;
    recyclable = false;
    clearAllData = false;
}

// AbsVisionResult::~AbsVisionResult() = default;

short AbsVisionResult::tag() const {
    return ABILITY_UNKNOWN;
}

void AbsVisionResult::clear() {
    abilityExecMarkers.clear();
    errorCode = V_TO_SHORT(Error::OK);
}

void AbsVisionResult::clearAll() {
    clear();
}

bool AbsVisionResult::isAbilityExec(short ability) const {
    const auto& it = abilityExecMarkers.find(ability);
    if (it != abilityExecMarkers.end()) {
        return it->second;
    }
    return false;
}

void AbsVisionResult::setAbilityExec(short ability) { abilityExecMarkers[ability] = true;
}

void AbsVisionResult::copy(AbsVisionResult *src) {

}

} // namespace aura::vision