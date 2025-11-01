#ifndef AURA_RUNTIME_OPENCL_CL_BUILD_OPTIONS_HPP__
#define AURA_RUNTIME_OPENCL_CL_BUILD_OPTIONS_HPP__

#include "aura/runtime/mat.h"

#include <vector>
#include <string>

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
 * @brief The CLBuildOptions class manages OpenCL build options for a specific Context.
 *
 * This class adds specific build options for OpenCL kernels, such as boundary processing modes,
 * intermediate data processing types, and macro constants etc.
 */
class AURA_EXPORTS CLBuildOptions
{
public:
    /**
     * @brief Constructs a CLBuildOptions object.
     *
     * @param ctx The pointer to the Context object.
     * @param param Vector stores build options related to the element type.
     */
    CLBuildOptions(Context *ctx, const std::vector<std::string> &param = std::vector<std::string>())
                   : m_ctx(ctx), m_tbl(param)
    {}

    /**
     * @brief Adds a build option with a key and value to the existing options.
     *
     * @param key The key is set as the name of the option
     * @param value The value associated with the key.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status AddOption(const std::string &key, const std::string &value = std::string());

    /**
     * @brief Generates a string representation of build options based on a specified type.
     *
     *  @param type Build options for adding element types.
     *
     * @return std::string The string representation of the build options.
     */
    std::string ToString(ElemType type = ElemType::INVALID);

private:
    Context *m_ctx;                     /*!< The pointer to the Context object. */
    std::vector<std::string> m_tbl;     /*!< Vector stores build options related to the element type. */
    std::string m_options;              /*!< String containing consolidated build options. */
};

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_OPENCL_CL_BUILD_OPTIONS_HPP__