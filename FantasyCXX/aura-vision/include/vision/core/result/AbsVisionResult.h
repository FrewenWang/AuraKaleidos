//
// Created by Li,Wendong on 2018/12/27.
//

#ifndef VISION_ABS_VISION_RESULT_H
#define VISION_ABS_VISION_RESULT_H

#include <list>
#include <unordered_map>

#include "vision/core/common/VMacro.h"
#include "vision/util/PerfUtil.h"

namespace aura::vision {

/**
 * @brief Base class of detection result
 */
class AbsVisionResult {
public:
    AbsVisionResult();

    virtual ~AbsVisionResult() = default;

    /**
     * @brief Clone the data from the input object
     * @param src
     */
    virtual void copy(AbsVisionResult *src);

    /**
     * @brief Get the tag of the request
     * @return
     */
    virtual short tag() const;

    /**
     * @brief Clear the frame only
     */
    virtual void clear();

    /**
     * @brief Clear all data
     */
    virtual void clearAll();

    /**
     * @brief Get whether the ability has been executed
     * @param ability
     * @return
     */
    virtual bool isAbilityExec(short ability) const;

    /**
     * @brief Set the executed status
     * @param ability
     */
    virtual void setAbilityExec(short ability);

public:
    /// @brief 执行模型检测的错误码，具体错误码定义参见：VConstant::
    short errorCode;
    bool recyclable;
    bool clearAllData;
    /**
     * @brief ability executed status markers
     * these markers would be clear after detected of each frame
     */
    std::unordered_map<short, bool> abilityExecMarkers;

private:
    static std::unordered_map<short, std::list<AbsVisionResult *>> resultsPool;

    const static short SUCCESS = 0;
    const static short FAILURE = 1;
};

} // namespace vision

#endif //VISION_VISION_RESULT_H
