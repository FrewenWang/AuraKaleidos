#ifndef AURA_RUNTIME_OPENCL_CL_RUNTIME_HPP__
#define AURA_RUNTIME_OPENCL_CL_RUNTIME_HPP__

#include "aura/runtime/memory.h"

#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wmissing-field-initializers"
#  pragma clang diagnostic ignored "-Wunused-parameter"
#  pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__) || defined(__GNUG__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif // __clang__

#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include "CL/opencl.hpp"

#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#  pragma GCC diagnostic pop
#endif // __clang__


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
 * @brief Enum class representing different types of OpenCL memory.
 */
enum class CLMemType
{
    INVALID    = 0,   /*!< Invalid memory type */
    BUFFER,           /*!< OpenCL buffer memory type */
    IAURA2D,          /*!< OpenCL 2D iaura memory type */
    IAURA3D           /*!< OpenCL 3D iaura memory type */
};

/**
 * @brief Enum class representing different synchronization methods for OpenCL memory.
 */
enum class CLMemSyncMethod
{
    INVALID    = 0,   /*!< Invalid synchronization method */
    AUTO,             /*!< Automatic synchronization method */
    ENQUEUE,          /*!< Enqueue synchronization method */
    FLUSH             /*!< Flush synchronization method */
};

/**
 * @brief Enum class representing different types of GPUs.
 */
enum class GpuType
{
    INVALID    = 0,   /*!< Invalid GPU type */
    ADRENO     = 100, /*!< Adreno GPU type */
    MALI       = 200, /*!< Mali GPU type */
    MOBILE     = 300  /*!< Mobile GPU type */
};

AURA_INLINE std::string GpuTypeToString(GpuType type)
{
    std::string gpu_type_str = "invalid";
    switch (type)
    {
        case GpuType::ADRENO:
        {
            gpu_type_str = "adreno";
            break;
        }
        case GpuType::MALI:
        {
            gpu_type_str = "mali";
            break;
        }
        case GpuType::MOBILE:
        {
            gpu_type_str = "mobile";
            break;
        }
        default:
        {
            break;
        }
    }
    return gpu_type_str;
}

AURA_INLINE std::string CLMemTypeToString(CLMemType cl_type)
{
    std::string cl_mem_type_str = "INVALID";

    switch (cl_type)
    {
        case CLMemType::BUFFER:
        {
            cl_mem_type_str = "BUFFER";
            break;
        }
        case CLMemType::IAURA2D:
        {
            cl_mem_type_str = "IAURA2D";
            break;
        }
        case CLMemType::IAURA3D:
        {
            cl_mem_type_str = "IAURA3D";
            break;
        }
        default:
        {
            break;
        }
    }

    return cl_mem_type_str;
}

AURA_INLINE std::string CLMemSyncMethodToString(CLMemSyncMethod cl_sync_method)
{
    std::string cl_mem_sync_method_str = "INVALID";

    switch (cl_sync_method)
    {
        case CLMemSyncMethod::AUTO:
        {
            cl_mem_sync_method_str = "AUTO";
            break;
        }
        case CLMemSyncMethod::ENQUEUE:
        {
            cl_mem_sync_method_str = "ENQUEUE";
            break;
        }
        case CLMemSyncMethod::FLUSH:
        {
            cl_mem_sync_method_str = "FLUSH";
            break;
        }
        default:
        {
            break;
        }
    }

    return cl_mem_sync_method_str;
}

/**
 * @brief The GpuInfo struct represents GPU information.
 */
struct AURA_EXPORTS GpuInfo
{
    /**
     * @brief Constructor for GpuInfo.
     *
     * @param type The type of GPU.
     */
    GpuInfo(GpuType type) : m_type(type)
    {
        m_name = GpuTypeToString(type);
    }

    GpuType m_type;         /*!< GPU type */
    std::string m_name;     /*!< GPU name */
};

/**
 * @brief The CLProgramString class manages OpenCL program strings.
 */
class AURA_EXPORTS CLProgramString
{
public:
    /**
     * @brief Constructor for CLProgramString.
     *
     * @param name The name of the OpenCL program.
     * @param source The program string to be registered.
     * @param incs The header files required by the program string.
     */
    CLProgramString(const std::string &name, const MI_CHAR *source, const std::vector<std::string> &incs);

    /**
     * @brief Registers the OpenCL program string.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Register();
};

/**
 * @brief The CLEngineConfig class holds configurations for the CLEngine.
 *
 * This class encapsulates various configurations required for initializing and
 * setting up the OpenCL engine (CLEngine). It manages settings such as cache paths,
 * performance levels, precompiled source types, and external version references.
 */
class AURA_EXPORTS CLEngineConfig
{
public:
    /**
     * @brief Constructor for CLEngineConfig.
     *
     * @param enable_cl Flag indicating if OpenCL is enabled.
     * @param cache_path Path to the directory for OpenCL kernel cache storage.
     * @param cache_prefix Prefix used for naming cache files.
     * @param cl_precompiled_type Type of precompiled OpenCL sources.
     * @param precompiled_sources Paths to precompiled OpenCL source files.
     * @param external_version Version of external components (optional).
     * @param cl_perf_level Performance level for OpenCL execution (default: high).
     * @param cl_priority_level Priority level for OpenCL tasks (default: low).
     */
    CLEngineConfig(MI_BOOL enable_cl,
                   const std::string &cache_path,
                   const std::string &cache_prefix,
                   CLPrecompiledType cl_precompiled_type  = CLPrecompiledType::INVALID,
                   const std::string &precompiled_sources = std::string(),
                   const std::string &external_version    = std::string(),
                   CLPerfLevel cl_perf_level              = CLPerfLevel::PERF_HIGH,
                   CLPriorityLevel cl_priority_level      = CLPriorityLevel::PRIORITY_LOW)
                   : m_enable_cl(enable_cl),
                     m_cl_perf_level(cl_perf_level),
                     m_cl_priority_level(cl_priority_level),
                     m_cache_path(cache_path),
                     m_cache_prefix(cache_prefix),
                     m_cl_precompiled_type(cl_precompiled_type),
                     m_precompiled_sources(precompiled_sources),
                     m_external_version(external_version)
    {}

    MI_BOOL           m_enable_cl;            /*!< Flag indicating if OpenCL is enabled. */
    CLPerfLevel       m_cl_perf_level;        /*!< Performance level of OpenCL execution. */
    CLPriorityLevel   m_cl_priority_level;    /*!< Priority level for OpenCL tasks. */
    std::string       m_cache_path;           /*!< Directory path for OpenCL kernel cache storage. */
    std::string       m_cache_prefix;         /*!< Prefix used for naming OpenCL kernel cache files. */
    CLPrecompiledType m_cl_precompiled_type;  /*!< Type of precompiled OpenCL sources. */
    std::string       m_precompiled_sources;  /*!< Paths to precompiled OpenCL source files. */
    std::string       m_external_version;     /*!< Version of external components. */
};

/**
 * @brief The CLRuntime class serves as an abstract interface for managing OpenCL runtime operations.
 *
 * This abstract class defines a set of virtual functions responsible for initializing, configuring,
 * and managing various OpenCL runtime functionalities. Concrete implementations of this interface
 * handle platform-specific OpenCL runtime behavior.
 */
class AURA_EXPORTS CLRuntime
{
public:

    /**
     * @brief Initializes the OpenCL runtime environment.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Initialize() = 0;

    /**
     * @brief Get the OpenCL program associated with the given program name, source, and build options.
     *
     * This function is responsible for obtaining a shared pointer to an OpenCL program specified by its name.
     * Optionally, it allows you to pass source code and build options to create or retrieve the program.
     *
     * @param program_name Name of the OpenCL program.
     * @param source  OpenCL program source code.
     * @param build_options Build options for the program..
     *
     * @return A shared pointer to the requested OpenCL program.
     */
    virtual std::shared_ptr<cl::Program> GetCLProgram(const std::string &program_name,
                                                      const std::string &source = std::string(),
                                                      const std::string &build_options = std::string()) const = 0;

    /**
     * @brief Retrieves the OpenCL platform.
     * 
     * @return Shared pointer to OpenCL platform.
     */
    virtual std::shared_ptr<cl::Platform> GetPlatform() = 0;

    /**
     * @brief Retrieves the OpenCL device.
     * 
     * @return Shared pointer to OpenCL device.
     */
    virtual std::shared_ptr<cl::Device> GetDevice() = 0;

    /**
     * @brief Retrieves the OpenCL context.
     * 
     * @return Shared pointer to OpenCL context.
     */
    virtual std::shared_ptr<cl::Context> GetContext() = 0;

    /**
     * @brief Retrieves the OpenCL command queue.
     * 
     * @return Shared pointer to OpenCL command queue.
     */
    virtual std::shared_ptr<cl::CommandQueue> GetCommandQueue() = 0;

    /**
     * @brief Checks if the OpenCL runtime is valid and functional.
     *
     * @return True if the OpenCL runtime is valid; otherwise, false.
     */
    virtual MI_BOOL IsValid() const                             = 0;

    /**
     * @brief Checks if non-uniform workgroups are supported.
     *
     * @return True if non-uniform workgroups are supported; otherwise, false.
     */
    virtual MI_BOOL IsNonUniformWorkgroupsSupported() const     = 0;

    /**
     * @brief Gets address alignment size based on platform type
     *
     * @return The address alignment size.
     */
    virtual MI_S32 GetCLAddrAlignSize() const = 0;

    /**
     * @brief Checks if memory sharing is supported in the OpenCL runtime.
     *
     * @return True if memory sharing is supported; otherwise, false.
     */
    virtual MI_BOOL IsMemShareSupported() const = 0;

    /**
     * @brief Gets the length alignment size for OpenCL memory objects.
     * 
     * @return The length alignment size.
     */
    virtual MI_S32 GetCLLengthAlignSize() const = 0;

    /**
     * @brief Gets the slice alignment size for a specific OpenCL iaura format and dimensions.
     *
     * @param cl_fmt The OpenCL iaura format.
     * @param width The width of the iaura.
     * @param height The height of the iaura.
     *
     * @return The slice alignment size.
     */
    virtual MI_S32 GetCLSliceAlignSize(const cl_iaura_format &cl_fmt, size_t width, size_t height) const = 0;

    /**
     * @brief Creates a new OpenCL buffer.
     *
     * @param cl_flags The OpenCL memory flags.
     * @param size The size of the buffer to be created (in bytes).
     *
     * @return A pointer to the created OpenCL buffer. Return nullptr if failed.
     */
    virtual cl::Buffer* CreateCLBuffer(cl_mem_flags cl_flags, size_t size) = 0;

    /**
     * @brief Creates a new 2D OpenCL iaura.
     *
     * @param cl_flags The OpenCL memory flags.
     * @param cl_fmt The OpenCL iaura format.
     * @param width The width of the iaura to be created.
     * @param height The height of the iaura to be created.
     *
     * @return A pointer to the created 2D OpenCL iaura. Return nullptr if failed.
     */
    virtual cl::Iaura2D* CreateCLIaura2D(cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width, size_t height) = 0;

    /**
     * @brief Creates a new 3D OpenCL iaura.
     *
     * @param cl_flags The OpenCL memory flags.
     * @param cl_fmt The OpenCL iaura format.
     * @param width The width of the iaura to be created.
     * @param height The height of the iaura to be created.
     * @param depth The depth of the iaura to be created.
     *
     * @return A pointer to the created 3D OpenCL iaura. Return nullptr if failed.
     */
    virtual cl::Iaura3D* CreateCLIaura3D(cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width, size_t height, size_t depth) = 0;

    /**
     * @brief Initializes an OpenCL buffer from a Buffer object.
     *
     * @param buffer The Buffer object to initialize the OpenCL buffer from.
     * @param cl_flags The OpenCL memory flags.
     * @param cl_sync_method The synchronization method for the memory operation.
     *
     * @return A pointer to the initialized OpenCL buffer. Return nullptr if failed.
     */
    virtual cl::Buffer*  InitCLBuffer(const Buffer &buffer, cl_mem_flags cl_flags, CLMemSyncMethod &cl_sync_method) = 0;

    /**
     * @brief Initializes an OpenCL 2D iaura from a Buffer object.
     *
     * @param buffer The Buffer object to initialize the OpenCL 2D iaura from.
     * @param cl_flags The OpenCL memory flags.
     * @param cl_fmt The OpenCL iaura format.
     * @param width The width of the iaura to initialize.
     * @param height The height of the iaura to initialize.
     * @param pitch The pitch of the iaura to initialize.
     * @param cl_sync_method The synchronization method for the memory operation.
     *
     * @return A pointer to the initialized 2D OpenCL iaura. Return nullptr if failed.
     */
    virtual cl::Iaura2D* InitCLIaura2D(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt,
                                       size_t width, size_t height, size_t pitch, CLMemSyncMethod &cl_sync_method) = 0;

    /**
     * @brief Initializes an OpenCL 3D iaura from a Buffer object.
     *
     * @param buffer The Buffer object to initialize the OpenCL 3D iaura from.
     * @param cl_flags The OpenCL memory flags.
     * @param cl_fmt The OpenCL iaura format.
     * @param width The width of the iaura to initialize.
     * @param height The height of the iaura to initialize.
     * @param depth The depth of the iaura to initialize.
     * @param row_pitch The row pitch of the iaura to initialize.
     * @param slice_pitch The slice pitch of the iaura to initialize.
     * @param cl_sync_method The synchronization method for the memory operation.
     *
     * @return A pointer to the initialized 3D OpenCL iaura. Return nullptr if failed.
     */
    virtual cl::Iaura3D* InitCLIaura3D(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt,
                                       size_t width, size_t height, size_t depth, size_t row_pitch, 
                                       size_t slice_pitch, CLMemSyncMethod &cl_sync_method) = 0;

    /**
     * @brief Creates a precompiled OpenCL program.
     *
     * @param file_path The path to the precompiled file.
     * @param prefix The prefix for the program.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status CreatePrecompiledCLProgram(const std::string &file_path, const std::string &prefix) = 0;

    /**
     * @brief Retrieves the string representation of the maximum constant size for an OpenCL program.
     *
     * @param n The constant size.
     *
     * @return The string representation of the maximum constant size.
     */
    virtual std::string GetCLMaxConstantSizeString(MI_S32 n) = 0;

    /**
     * @brief Retrieves the default local size for an OpenCL kernel.
     *
     * @param max_group_size The maximum group size.
     * @param global_size The global size.
     *
     * @return The default local size.
     */
    virtual cl::NDRange GetCLDefaultLocalSize(MI_U32 max_group_size, cl::NDRange &global_size) = 0;

    /**
     * @brief Retrieves information about the GPU.
     *
     * @return Information about the GPU.
     */
    virtual GpuInfo GetGpuInfo() const = 0;

    /**
     * @brief Deletes an OpenCL memory object.
     *
     * @param ptr Pointer to the memory object.
     */
    virtual AURA_VOID DeleteCLMem(AURA_VOID **ptr) = 0;

    /**
     * @brief Virtual destructor for CLRuntime.
     *
     * The destructor must be implemented in derived classes.
     */
    virtual ~CLRuntime() = 0;
};

/**
 * @brief Retrieves a string representation of an OpenCL error code.
 *
 * @param error The OpenCL error code.
 *
 * @return A string containing information about the error code.
 */
AURA_EXPORTS std::string GetCLErrorInfo(cl_int error);

/**
 * @brief Retrieves profiling information for a specific OpenCL kernel.
 *
 * This function retrieves profiling information for a specified OpenCL kernel,
 * including execution time, queued time, submitted time, and start-end time stamps.
 *
 * @param kernel_name The name of the OpenCL kernel.
 * @param cl_event    Reference to the OpenCL event associated with the kernel execution.
 *
 * @return A string containing profiling information for the specified kernel.
 */
AURA_EXPORTS std::string GetCLProfilingInfo(const std::string &kernel_name, cl::Event &cl_event);

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_OPENCL_CL_RUNTIME_HPP__
