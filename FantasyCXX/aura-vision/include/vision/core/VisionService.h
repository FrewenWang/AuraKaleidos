#ifndef VISION_VISION_SERVICE_H
#define VISION_VISION_SERVICE_H

#include <cstdio>
#include <memory>
#include <unordered_map>

#include "vision/core/common/VMacro.h"
#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"

namespace aura::vision {

/**
 * @brief vision ability SDK interfaces
 */
class VA_PUBLIC VisionService {
public:
    /**
     * @brief init the sdk
     */
    int init();

    /**
     * @brief deinit the sdk
     */
    int deinit();

    /**
     * @brief Detect the input request, the detection info will be written into result
     * @param request detection request, which contains image or video frame
     * @param result detection results, which contains face and gesture info
     */
    void detect(VisionRequest *request, VisionResult *result);

    /**
     * @brief Set the ability switch,
     * this is used for enable or disable detection abilities
     * @param ability key of the ability
     * @param enable whether to enable the ability
     */
    void set_switch(short ability, bool enable);

    /**
     * @brief Set the ability switches,
     * this is used for enable or disable detection abilities
     * @param switches the switch map(key-enabled pairs)
     */
    void set_switches(const std::unordered_map<short, bool> &switches);

    /**
     * @brief Set the ability switches,
     * this is used for enable or disable detection abilities
     * @param keys the ability keys, these abilities share the enable setting
     * @param enable whether to enable the ability
     */
    void set_switches(const std::vector<short> &keys, bool enable);

    /**
     * @brief Get the ability switch
     * @param ability the ability key
     * @return the ability switch value
     */
    bool get_switch(short ability) const;

    /**
     * @brief Get the ability switch map
     * @return the switch map(key-enabled pairs)
     */
    const std::unordered_map<short, bool> &get_switches() const;

    /**
     * Set the config parameters
     * @param key parameter key
     * @param value parameter value
     */
    void set_config(int key, float value);

    /**
     * @brief get config by key
     * @param key parameter key
     * @return parameter value
     */
    float get_config(int key) const;

    /**
     * 向对应能力发送 command 指令
     * @param abilityId AbilityId
     * @param cmd  command指令
     * @return
     */
    bool setAbilityCmd(int abilityId, int cmd);

    std::shared_ptr<RtConfig> getRtConfig() const;

    /**
     * @brief make a VisionRequest object from object pool
     */
    VisionRequest *make_request();

    /**
     * @brief make a VisionResult object from object pool
     */
    VisionResult *make_result();

    /**
     * @brief recycle a VisionRequest object to the object pool
     */
    void recycle_request(VisionRequest *req);

    /**
     * @brief recycle a VisionResult object to the object pool
     */
    void recycle_result(VisionResult *res);

    /**
     * 清除能力对应的manager中策略数据
     *
     * @param abilityId 能力id
     */
    void clearManagerStrategy(int abilityId) const;

    /**
     * @brief clear the trigger count
     * @param ability ability key
     * @return whether the trigger count is cleared
     */
    bool clean_ability_trigger_accumulative(short ability);


public:
    explicit VisionService(int sourceId) noexcept;

    VisionService(const VisionService &service) = delete;

    VisionService(VisionService &&service) = default;

    VisionService &operator=(const VisionService &service) = delete;

    VisionService &operator=(VisionService &&service) = default;

    ~VisionService();

private:
    class Impl;

    std::unique_ptr<Impl> impl;
};

} // namespace vision
#endif // VISION_VISION_ABILITY_SERVICE_H
