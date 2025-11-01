#ifndef AURA_RUNTIME_CONTEXT_HOST_CONTEXT_HPP__
#define AURA_RUNTIME_CONTEXT_HOST_CONTEXT_HPP__

#include "aura/runtime/context/host/config.hpp"

#include <memory>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup context Context
 *      @{
 *          @defgroup context_host Context Host
 *      @}
 * @}
*/

namespace aura
{

/**
 * @addtogroup context_host
 * @{
*/
class Logger;
class MemPool;
class WorkerPool;

#if defined(AURA_BUILD_ANDROID)
class Systrace;
#endif // AURA_BUILD_ANDROID

#if defined(AURA_ENABLE_OPENCL)
class CLEngine;
#endif // AURA_ENABLE_OPENCL

#if defined(AURA_ENABLE_HEXAGON)
class HexagonEngine;
#endif // AURA_ENABLE_HEXAGON

#if defined(AURA_ENABLE_XTENSA)
class XtensaEngine;
#endif // AURA_ENABLE_XTENSA

#if defined(AURA_ENABLE_NN)
class NNEngine;
#endif // AURA_ENABLE_NN

/**
 * @brief Context class for AURA host runtime.
 *
 * The context class manages the lifecycle of various modules, such as memory pool, thread pool, logging system,
 * OpenCL runtime, Hexagon runtime, NN runtime, and XTENSA runtime. Objects of this class should be created at the beginning of the project and destroyed at the end.
 */
class AURA_EXPORTS Context
{
public:
    /**
     * @brief Constructor for the Context class.
     *
     * @param config Configuration parameters for the AURA context.
     */
    Context(const Config &config);

    /**
     * @brief Destructor for the Context class.
     */
    ~Context();

    /**
     * @brief Checks if platform is supported, Currently only detected on the android platform
     *
     * check whether cpu hardware supports atomics, very low android platform will crash when you don't check
     * 
     * @return True if platform is supported; otherwise, false.
     */
    static MI_BOOL IsPlatformSupported();

    /**
     * @brief Initialize the AURA context.
     *
     * Create the objects of memory pool, thread pool, logging system, OpenCL runtime, Hexagon runtime, XTENSA runtime, and NN runtime, and initialize them based on the configuration parameters.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Initialize();

    /**
     * @brief Get the version of the AURA library.
     *
     * @return The version string.
     */
    std::string GetVersion() const;

    /**
     * @brief Get the logger associated with the AURA context.
     *
     * @return Pointer to the Logger object.
     */
    Logger* GetLogger() const;

    /**
     * @brief Get the memory pool associated with the AURA context.
     *
     * @return Pointer to the MemPool object.
     */
    MemPool* GetMemPool() const;

    /**
     * @brief Get the worker pool associated with the AURA context.
     *
     * @return Pointer to the WorkerPool object.
     */
    WorkerPool* GetWorkerPool() const;

#if defined(AURA_BUILD_ANDROID)
    /**
     * @brief Get the systrace object associated with the AURA context.
     *
     * @return Pointer to the Systrace object.
     */
    Systrace* GetSystrace() const;
#endif // AURA_BUILD_ANDROID

#if defined(AURA_ENABLE_OPENCL)
    /**
     * @brief Get the OpenCL engine associated with the AURA context.
     *
     * @return Pointer to the CLEngine object.
     */
    CLEngine* GetCLEngine() const;
#endif // AURA_ENABLE_OPENCL

#if defined(AURA_ENABLE_HEXAGON)
    /**
     * @brief Get the Hexagon engine associated with the AURA context.
     *
     * @return Pointer to the HexagonEngine object.
     */
    HexagonEngine* GetHexagonEngine() const;
#endif // AURA_ENABLE_HEXAGON

#if defined(AURA_ENABLE_NN)
    /**
     * @brief Get the NN engine associated with the AURA context.
     *
     * @return Pointer to the NNEngine object.
     */
    NNEngine* GetNNEngine() const;
#endif // AURA_ENABLE_NN

#if defined(AURA_ENABLE_XTENSA)
    /**
     * @brief Get the Xtensa engine associated with the AURA context.
     *
     * @return Pointer to the XtensaEngine object.
     */
    XtensaEngine* GetXtensaEngine() const;
#endif // AURA_ENABLE_XTENSA

    /**
     * @brief Disable copy and assignment constructor for the Context class.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(Context);

private:
    class Impl;                     /*!< Private implementation class for the Context class. */
    std::shared_ptr<Impl> m_impl;   /*!< Pointer to the implementation class. */
};

/**
 * @}
*/
} // namespace aura

#endif // AURA_RUNTIME_CONTEXT_HOST_CONTEXT_HPP__