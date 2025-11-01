//
// Created by Frewen.Wang on 25-9-17.
//


// AuraUtils库里面的context模块，需要依赖AuraUtils模块的核心库模块
#include "aura/utils/core.h"
#include "context_config.hpp"

namespace aura::utils
{
/**
 * @brief 设置 CPU 亲和性的 enum 常量
 * @brief Enumeration representing CPU affinity options.
 */
enum class CpuAffinity
{
    ALL = 0, /*!< The initiate threads with no CPU affinity */
    BIG, /*!< The all big cores are used */
    LITTLE, /*!< The all small cores are used */
};

/**
 * @brief 设置OpenCL的运行的性能登记
 * @brief Enumeration representing OpenCL performance levels.
 */
enum class CLPerfLevel
{
    PERF_DEFAULT = 0, /*!< Default performance level. */
    PERF_LOW, /*!< Low performance level. */
    PERF_NORMAL, /*!< Normal performance level. */
    PERF_HIGH, /*!< High performance level. */
};

class AURA_EXPORTS Context
{

public:
    Context(const ContextConfig &config);
    ~Context();

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};


} // namespace aura::utils
