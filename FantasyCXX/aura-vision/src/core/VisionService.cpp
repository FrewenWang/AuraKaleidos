
#include "vision/core/VisionService.h"

#include "scheduler/DagScheduler.h"
#include "scheduler/NaiveScheduler.h"
#include "vision/util/ThreadPool.h"
#include "util/FpsUtil.h"
#include "util/SchedUtil.h"
#include "vision/core/VisionContext.h"
#include "vision/core/version.h"
#include "vision/manager/VisionManagerRegistry.h"
#include "vision/util/log.h"
#include "util/CsvFile.hpp"
#include <algorithm>
#include <memory>
#if ENABLE_PERF
#include "util/CsvFile.hpp"
#endif

#include <iostream>
#include <fstream>

namespace aura::vision {

static const char *TAG = "VisionService";

#if ENABLE_PERF
static WriteCsvFile sWriteCsvFile;
#endif

/**
 * implementation class of VisionService
 */
class VisionService::Impl {
public:
    Impl(int sourceId);

    ~Impl();

    int init();

    int deinit();

    void detect(VisionRequest *request, VisionResult *result);

    /**
     * 设置开关变量
     * @param ability  开关ID
     * @param enable    开关
     */
    void setSwitch(short ability, bool enable);
    /**
     * 批量设置能力开关
     * @param switches
     */
    void setSwitches(const std::unordered_map<short, bool> &switches);
    /**
     * 批量设置能力开关
     * @param keys
     * @param enable
     */
    void setSwitches(const std::vector<short> &keys, bool enable);
    /**
     * 获取对应能力的开关
     * @param ability
     * @return
     */
    bool getSwitch(short ability) const;
    /**
     * 获取能力开关列表
     * @return
     */
    const std::unordered_map<short, bool> &getSwitches() const;
    /**
     * 设置配置参数
     * @param key
     * @param value
     */
    void setConfig(int key, float value);

    /**
     * 获取配置参数的值
     * @param key
     * @return
     */
    float getConfig(int key) const;

    /**
     * 发送给对应能力的消息指令。
     * 注意:目前能力层逻辑都是单线程逻辑，故发送消息指令需要和检测线程一致
     * @param ability 设置对应能力
     * @param cmd     对应消息指令
     * @return
     */
    bool setAbilityCmd(int ability, int cmd) const;

    /**
     * 获取RuntimeConfig
     * @return
     */
    std::shared_ptr<RtConfig> getRtConfig() const;
    /**
     * 清除对应能力的累计的行为触发次数
     * @param ability
     * @return
     */
    bool cleanAbilityTriggerAccumulative(short ability);

    VisionRequest *makeRequest() const;

    VisionResult *makeResult() const;

    void recycleRequest(VisionRequest *req) const;

    void recycleResult(VisionResult *res) const;

    /**
     * 清除对应Manager的Strategy的滑窗结果
     * @param abilityId
     */
    void clearManagerStrategy(int abilityId) const;

private:
    std::shared_ptr<AbsScheduler> scheduler;
    std::shared_ptr<RtConfig> mRtConfig;
    std::shared_ptr<FpsUtil> fpsUtil;
    int mSourceId = Source::SOURCE_UNKNOWN;
};

VisionService::Impl::Impl(int sourceId) {
    mSourceId = sourceId;
    mRtConfig.reset(VisionContext::getRtConfig(mSourceId));
    auto manager_registry = std::make_shared<VisionManagerRegistry>();
    mRtConfig->inject_manager_registry(manager_registry);
    mRtConfig->init();
    // 实例化帧率计算器。设置平滑帧率的策略：低通滤波
    fpsUtil = FpsUtil::instance(mSourceId);
    fpsUtil->setSmoothStrategy(SmoothType::LowPassFilter);

    SchedUtil::set_affinity(mRtConfig);
}

VisionService::Impl::~Impl() {
    if (mRtConfig != nullptr) {
        mRtConfig->getManagerRegistry().reset();
        mRtConfig->deinit();
        mRtConfig.reset();
    }
    if (fpsUtil != nullptr) {
        fpsUtil.reset();
    }
}

int VisionService::Impl::init() {
    if (V_TO_INT(mRtConfig->scheduleMethod) == SchedulerMethod::NAIVE) {
        scheduler = make_scheduler<SCHED_NAIVE>();
        VLOGD(TAG, "Use Naive Scheduler");
    } else {
        scheduler = make_scheduler<SCHED_DAG>();
        auto count = V_TO_INT(mRtConfig->scheduleDagThreadCount);
        if (count != -1) {
            ThreadPool::mInitThreadsSize = count;
        }
        VLOGD(TAG, "Use DAG Scheduler");
    }
    if (scheduler == nullptr) {
        VLOGE(TAG, "scheduler is nullptr, init failed!!!");
        V_RET(Error::INIT_FAILURE);
    }
    scheduler->injectManagerRegistry(mRtConfig->getManagerRegistry());
    scheduler->registerManagers();
    scheduler->initManagers(mRtConfig.get());

    // pthread_t tid;
    // int ret = pthread_create(&tid, NULL, exiter, NULL);
    //  VLOGE(TAG, "====== Because this is a DEMO version, Vision SDK will exit(0) after 1800 seconds! ======");

    V_RET(Error::OK);
}

int VisionService::Impl::deinit() {
    if (scheduler != nullptr) {
        scheduler->deinit();
        scheduler.reset();
    }
    return 0;
}

void VisionService::Impl::detect(VisionRequest *request, VisionResult *result) {
    if (request == nullptr || !request->verify()) {
        result->errorCode = V_TO_SHORT(Error::INVALID_PARAM);
        VLOGE(TAG, "request verify error, detection stopped for request:%p", request);
        return;
    }
    fpsUtil->update();
    VLOGD(TAG, "===========startDetect source=[%d],timestamp=[%ld],curFps=[%f]==========", mSourceId,
          request->timestamp, fpsUtil->getSmoothedFps());
    // 赋值VisionResult的结果数据
    result->getFrameInfo()->frame = request->frame;
    result->getFrameInfo()->timestamp = request->timestamp;
#if ENABLE_PERF
    sWriteCsvFile.write(mSourceId, request->timestamp, fpsUtil->getSmoothedFps());
    PerfUtil::global()->setLogging(true);
    PERF_AUTO(result->getPerfUtil(), PerfUtil::TAG_TOTAL);
#endif
    scheduler->run(request, result);
#if ENABLE_PERF
    PerfUtil::global()->setLogging(false);
#endif
}

bool VisionService::Impl::cleanAbilityTriggerAccumulative(short ability) {
    auto manager = scheduler->getManagerRegistry()->getManager(ability);
    V_CHECK_NULL_RET(manager, false);
    manager->clearTriggerAccumulative();
    return true;
}

void VisionService::Impl::setSwitch(short ability, bool enable) {
    mRtConfig->set_switch(ability, enable);
}

void VisionService::Impl::setSwitches(const std::unordered_map<short, bool> &switches) {
    mRtConfig->set_switches(switches);
}

void VisionService::Impl::setSwitches(const std::vector<short> &keys, bool enable) {
    std::unordered_map<short, bool> map_switches;
    for (auto &key : keys) {
        map_switches[key] = enable;
    }
    setSwitches(map_switches);
}

bool VisionService::Impl::getSwitch(short ability) const {
    return mRtConfig->get_switch(ability);
}

const std::unordered_map<short, bool> &VisionService::Impl::getSwitches() const {
    return mRtConfig->get_switches();
}

void VisionService::Impl::setConfig(int key, float value) {
    mRtConfig->set_config(key, value);

    if (key == ParamKey::THREAD_AFFINITY_POLICY) {
        SchedUtil::set_affinity(mRtConfig);
    }
}

float VisionService::Impl::getConfig(int key) const {
    return mRtConfig->get_config(key);
}

std::shared_ptr<RtConfig> VisionService::Impl::getRtConfig() const {
    return mRtConfig;
}

VisionRequest *VisionService::Impl::makeRequest() const {
    return VisionRequest::obtain(mRtConfig.get());
}

VisionResult *VisionService::Impl::makeResult() const {
    return VisionResult::obtain(mRtConfig.get());
}

void VisionService::Impl::recycleRequest(VisionRequest *req) const {
    VisionRequest::recycle(req);
}

void VisionService::Impl::recycleResult(VisionResult *res) const {
    VisionResult::recycle(res);
}

void VisionService::Impl::clearManagerStrategy(int abilityId) const {
    auto manager = scheduler->getManagerRegistry()->getManager(abilityId);
    manager->clear();
}

bool VisionService::Impl::setAbilityCmd(int abilityId, int cmd) const {
    // 获取对应能力的Manager
    auto manager = scheduler->getManagerRegistry()->getManager(abilityId);
    if (manager == nullptr) {
        return false;
    }
    return manager->onAbilityCmd(cmd);
}

VisionService::VisionService(int sourceId) noexcept {
    impl = std::unique_ptr<VisionService::Impl>(new VisionService::Impl(sourceId));
}

VisionService::~VisionService() = default;

int VisionService::init() {
    return impl->init();
}

int VisionService::deinit() {
    auto res = impl->deinit();
    impl = nullptr;
    return res;
}

void VisionService::detect(VisionRequest *request, VisionResult *result) {
    impl->detect(request, result);
}

bool VisionService::get_switch(short ability) const {
    return impl->getSwitch(ability);
}

const std::unordered_map<short, bool> &VisionService::get_switches() const {
    return impl->getSwitches();
}

void VisionService::set_switch(short ability, bool enable) {
    impl->setSwitch(ability, enable);
}

void VisionService::set_switches(const std::unordered_map<short, bool> &switches) {
    impl->setSwitches(switches);
}

void VisionService::set_switches(const std::vector<short> &keys, bool enable) {
    impl->setSwitches(keys, enable);
}

bool VisionService::clean_ability_trigger_accumulative(short ability) {
    return impl->cleanAbilityTriggerAccumulative(ability);
}

void VisionService::set_config(int key, float value) {
    impl->setConfig(key, value);
}

float VisionService::get_config(int key) const {
    return impl->getConfig(key);
}

std::shared_ptr<RtConfig> VisionService::getRtConfig() const {
    return impl->getRtConfig();
}

VisionRequest *VisionService::make_request() {
    return impl->makeRequest();
}

VisionResult *VisionService::make_result() {
    return impl->makeResult();
}

void VisionService::recycle_request(VisionRequest *req) {
    impl->recycleRequest(req);
}

void VisionService::recycle_result(VisionResult *res) {
    impl->recycleResult(res);
}

void VisionService::clearManagerStrategy(int abilityId) const {
    impl->clearManagerStrategy(abilityId);
}

bool VisionService::setAbilityCmd(int abilityId, int cmd) {
    return impl->setAbilityCmd(abilityId, cmd);
}

} //namespace aura::vision
