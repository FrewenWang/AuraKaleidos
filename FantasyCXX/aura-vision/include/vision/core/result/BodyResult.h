#ifndef VISION_BODY_RESULT_H
#define VISION_BODY_RESULT_H

#include "AbsVisionResult.h"
#include "vision/config/runtime_config/RtConfig.h"
#include "vision/core/bean/BodyInfo.h"
#include "vision/core/common/VConstants.h"
#include <sstream>

namespace aura::vision {

/**
 * @brief Detection result of body info request
 */
class BodyResult : public AbsVisionResult {
public:
    explicit BodyResult(RtConfig *cfg);

    ~BodyResult() override;

    /**
     * @brief Copy from another VisionResult instance
     * @param src
     */
    void copy(AbsVisionResult *src) override;

    /**
     * @brief Get the tag of the result
     * @return
     */
    short tag() const override;

    /**
     * @brief Clear the frame only
     */
    void clear() override;

    /**
     * @brief Clear all data
     */
    void clearAll() override;

    /**
     * @brief Check whether a body is detected
     * @return
     */
    bool noBody() const;

    /**
     * @brief Check whether a body is detected
     * @return
     */
    bool hasBody() const;
    /**
     * @brief resize BodyInfos count
     * @param count
     */
    void resize(int count);

    /**
     * 获取 BodyResult 的检测到的BodyInfo数量
     * @return
     */
    short bodyCount() const;

    void toString(std::stringstream &ss) const;

public:
    int bodyMaxCount = 0;            /// max body count of vision-ability support
    BodyInfo **pBodyInfos;      /// detected human pose info

private:
    bool _use_internal_mem;
    static const short TAG = ABILITY_BODY;
};

} // namespace vision

#endif //VISION_BODY_RESULT_H
