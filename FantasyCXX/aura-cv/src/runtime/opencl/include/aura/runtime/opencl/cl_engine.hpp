#ifndef AURA_RUNTIME_OPENCL_CL_ENGINE_HPP__
#define AURA_RUNTIME_OPENCL_CL_ENGINE_HPP__

#include "aura/runtime/opencl/cl_runtime.hpp"

#include <string>
#include <memory>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup cl OpenCL
 * @}
 */

namespace aura
{

/**
 * @addtogroup cl
 * @{
 */

/**
 * @brief The CLEngine class providing initialization, configuration and access to the OpenCL runtime.
 *
 * The CLEngine class facilitates the setup and management of OpenCL, allowing
 * users to configure the OpenCL environment, access the runtime, and perform
 * computations on supported hardware devices.
 */
class AURA_EXPORTS CLEngine
{
public:
    /**
     * @brief Constructs a CLEngine object.
     *
     * @param ctx The pointer to the Context object.
     * @param enable_cl Flag indicating if OpenCL is enabled.
     * @param cache_path Path to the cache directory.
     * @param cache_prefix Prefix for cache files.
     * @param cl_precompiled_type Type of precompiled OpenCL sources.
     * @param precompiled_sources Sources for precompiled OpenCL.
     * @param external_version Version of external components.
     * @param cl_perf_level Performance level of OpenCL.
     * @param cl_priority_level Priority level of OpenCL.
     */
    CLEngine(Context *ctx, MI_BOOL enable_cl, 
             const std::string &cache_path,
             const std::string &cache_prefix,
             CLPrecompiledType cl_precompiled_type,
             const std::string &precompiled_sources,
             const std::string &external_version,
             CLPerfLevel cl_perf_level,
             CLPriorityLevel cl_priority_level);

    /**
     * @brief Destructor for the CLEngine object.
     */
    ~CLEngine();

    AURA_DISABLE_COPY_AND_ASSIGN(CLEngine); /*!< Macro to disable copy and assignment operations. */

    /**
     * @brief Get the CLRuntime object.
     *
     * @return Shared pointer to the CLRuntime object.
     */
    std::shared_ptr<CLRuntime> GetCLRuntime();

    /**
     * @brief  Get the CLRuntime object (const version).
     *
     * @return Shared pointer to CLRuntime object.
     */
    std::shared_ptr<CLRuntime> GetCLRuntime() const;

private:
    std::shared_ptr<CLRuntime> m_cl_runtime; /*!< Pointer to the OpenCL runtime. */
    CLEngineConfig m_cl_config; /*!< Configuration for the OpenCL engine. */
};

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_OPENCL_CL_ENGINE_HPP__
