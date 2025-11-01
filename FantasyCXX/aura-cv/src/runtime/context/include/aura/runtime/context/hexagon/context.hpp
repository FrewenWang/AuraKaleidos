#ifndef AURA_RUNTIME_CONTEXT_HEXAGON_CONTEXT_HPP__
#define AURA_RUNTIME_CONTEXT_HEXAGON_CONTEXT_HPP__

#include "aura/runtime/core.h"

/**
 * @cond AURA_BUILD_HEXAGON
*/

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup context Context
 *      @{
 *          @defgroup context_hexagon Context Hexagon
 *      @}
 * @}
*/

namespace aura
{

/**
 * @addtogroup context_hexagon
 * @{
*/

class Logger;
class MemPool;
class WorkerPool;
class Buffer;

#if defined(AURA_ENABLE_NN)
class NNEngine;
#endif // AURA_ENABLE_NN

/**
 * @brief Context class for AURA DSP runtime.
 *
 * The context class manages the lifecycle of various modules, such as memory pool, thread pool, logging system, and share buffer.
 * The object of this class is created when the DSP handle is opened and destoryed when the DSP handle is closed.
 */
class AURA_EXPORTS Context
{
public:
    /**
     * @brief Default constructor for the Context class.
     */
    Context();

    /**
     * @brief Destructor for the Context class.
     */
    ~Context();

    /**
     * @brief Initialize the AURA DSP context with specified logging parameters.
     *
     * @param output Output destination for logging.
     * @param level Logging level.
     * @param file Log file path if output is LogOutput::FILE, otherwise ignored.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Initialize(LogOutput output, LogLevel level, const std::string &file);

    /**
     * @brief Get the version of the AURA DSP library.
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

 #if defined(AURA_ENABLE_NN)
    /**
     * @brief Get the NN engine associated with the AURA context.
     *
     * @return Pointer to the NNEngine object.
     */
    NNEngine* GetNNEngine() const;
#endif // AURA_ENABLE_NN

    /**
     * @brief Add buffer to AURA context as shared buffer.
     *
     * @return Pointer to the Buffer object.
     */
    Status AddShareBuffer(const std::string &name, const Buffer &buffer);

    /**
     * @brief Get the shared buffer associated with the AURA context.
     *
     * @return Buffer
     */
    Buffer GetShareBuffer(const std::string &name);

    /**
     * @brief Remove buffer from AURA context.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status RemoveShareBuffer(const std::string &name);

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

/**
 * @endcond
*/
#endif // AURA_RUNTIME_CONTEXT_HEXAGON_CONTEXT_HPP__