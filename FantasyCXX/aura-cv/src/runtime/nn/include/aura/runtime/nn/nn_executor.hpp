#ifndef AURA_RUNTIME_NN_NN_EXECUTOR_HPP__
#define AURA_RUNTIME_NN_NN_EXECUTOR_HPP__

#include "aura/runtime/core.h"
#include "aura/runtime/mat.h"
#include "aura/tools/any_params.h"

#include <unordered_map>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup nn NN
 * @}
 */

namespace aura
{
/**
 * @addtogroup nn
 * @{
 */

/**
 * @brief Alias for a map of string names to Mat pointers.
 */
using MatMap = std::unordered_map<std::string, Mat*>;

/**
 * @brief Structure representing tensor description.
 *
 * This structure holds elemment type and sizes of tensor.
 */
struct AURA_EXPORTS TensorDesc
{
    TensorDesc() : elem_type(ElemType::INVALID), scale(0.f), zero_point(0), graph_id(0)
    {}

    ElemType elem_type;        /*!< Element type of tensor. 张量的元素类型 */
    std::vector<MI_S32> sizes; /*!< Sizes of tensor. 张量的大小*/
    MI_F32 scale;              /*!< quantization parameters scale value. */
    MI_S32 zero_point;         /*!< quantization parameters zero value. */
    MI_S32 graph_id;           /*!< graph id. */
};

/**
 * @brief Alias for a map of string names to TensorDesc object.
 */
using TensorDescMap = std::unordered_map<std::string, TensorDesc>;

/**
 * @brief Overloaded output stream operator for TensorDescMap type.
 *
 * This operator allows printing TensorDescMap type to an output stream.
 *
 * @param os The output stream.
 * @param type The TensorDescMap type to be printed.
 *
 * @return Color TensorDescMap out stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, const TensorDescMap &desc_map)
{
    for (auto &desc : desc_map)
    {
        std::string size_info;
        for (size_t i = 0; i < desc.second.sizes.size() - 1; i++)
        {
            size_info += std::to_string(desc.second.sizes[i]) + "x";
        }
        size_info += std::to_string(desc.second.sizes.back());

        os << std::endl;
        os << "=================[" << desc.first << "]=================" << std::endl;
        os << "graph_id             : " << std::to_string(desc.second.graph_id) << std::endl;
        os << "elem_type            : " << ElemTypesToString(desc.second.elem_type) << std::endl;
        os << "size                 : " << size_info << std::endl;
        os << "scale                : " << std::to_string(desc.second.scale) << std::endl;
        os << "zero_point           : " << std::to_string(desc.second.zero_point) << std::endl;
    }

    return os;
}

/**
 * @brief Overloaded output stream operator for std::vector<TensorDescMap> type.
 *
 * This operator allows printing std::vector<TensorDescMap> type to an output stream.
 *
 * @param os The output stream.
 * @param type The std::vector<TensorDescMap> type to be printed.
 *
 * @return Color std::vector<TensorDescMap> out stream.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, const std::vector<TensorDescMap> &desc_maps)
{
    for (size_t i = 0; i < desc_maps.size(); i++)
    {
        os << desc_maps[i];
    }

    return os;
}

/**
 * @brief Convert TensorDescMap to string representation.
 *
 * This function converts a TensorDescMap type to its string representation.
 *
 * @param type The TensorDescMap type.
 *
 * @return The string representation of the TensorDescMap.
 */
AURA_INLINE std::string TensorDescMapToString(const TensorDescMap &desc_map)
{
    std::ostringstream ss;
    ss << desc_map ;
    return ss.str();
}

/**
 * @brief Convert std::vector<TensorDescMap> to string representation.
 *
 * This function converts a std::vector<TensorDescMap> type to its string representation.
 *
 * @param type The std::vector<TensorDescMap> type.
 *
 * @return The string representation of the std::vector<TensorDescMap>.
 */
AURA_INLINE std::string TensorDescMapToString(const std::vector<TensorDescMap> &desc_maps)
{
    std::ostringstream ss;
    ss << desc_maps;
    return ss.str();
}

/**
 * @brief Abstract base class for neural network executors.
 *
 * This class defines the interface for neural network executors,
 * providing methods for initialization, forward propagation, and retrieving version information.
 */
class AURA_EXPORTS NNExecutor
{
public:
    /**
     * @brief Constructor for NNExecutor.
     *
     * @param ctx Pointer to the execution context.
     */
    NNExecutor(Context *ctx): m_ctx(ctx)
    {}

    /**
     * @brief Virtual destructor for NNExecutor.
     */
    virtual ~NNExecutor()
    {
        m_ctx = MI_NULL;
    }

    /**
     * @brief Disable copy and assignment constructor operations.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(NNExecutor);

    /**
     * @brief Initialize the NNExecutor.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Initialize() = 0;

    /**
     * @brief  Execute backend-specific or task-specific function.
     *
     * @param name Function name.
     * @param params Function args.
     *  Function name-params pairs:
     *     -----------------------------------------------------------------------------------------------------------------------------------------------------
     *    |       name          |                 params format                      |                     explain                                              |
     *    |---------------------|----------------------------------------------------|--------------------------------------------------------------------------|
     *    | update_perf         | params["perf_level"]         : const std::string&  | common config, update performance level, use before Forward()            |
     *    |                     |                                                    | supported platform: qcom(qnn), mtk(xp)                                   |
     *    |                     | params["qnn_hmx_perf_level"] : const std::string&  | specialized config, update qnn hmx perf level                            |
     *    |                     |                                                    | supported platform: qcom(qnn)                                            |
     *    |---------------------|----------------------------------------------------|--------------------------------------------------------------------------|
     *    | register_mem        | params["input_matmap" ]      : const MatMap&       | common config, register input & output buffer, use before Forward()      |
     *    |                     | params["output_matmap"]      : const MatMap&       | supported platform: qcom(qnn), xiaomi(xnn)                               |
     *    |                     | params["graph_id"]           : MI_S32 (default 0)  |                                                                          |
     *    |---------------------|----------------------------------------------------|--------------------------------------------------------------------------|
     *    | deregister_mem      | params["input_matmap" ]      : const MatMap&       | common config, deregister input & output buffer, use after Forward()     |
     *    |                     | params["output_matmap"]      : const MatMap&       | supported platform: qcom(qnn), xiaomi(xnn)                               |
     *    |                     | params["graph_id"]           : MI_S32 (default 0)  |                                                                          |
     *    |---------------------|----------------------------------------------------|--------------------------------------------------------------------------|
     *    | qnn_update_lora     | params["qnn_section_path"]   : const std::string&  | specialized config, update lora binary sections, use before Forward()    |
     *    |                     | params["qnn_decrypt_key"]    : const std::string&  | supported platform: qnn                                                  |
     *    |                     | params["qnn_graph_id"]       : MI_S32              |                                                                          |
     *     -----------------------------------------------------------------------------------------------------------------------------------------------------|
     *
     *  Usage:
     *      MatMap input;
     *      MatMap output;
     *      AnyParams params;
     *      params["input_matmap"]  = std::ref(input);
     *      params["output_matmap"] = std::ref(output);
     *      nn_executor->Update("register_mem", params);
     */
    virtual Status Update(const std::string &name, AnyParams &params) = 0;

    /**
     * @brief Perform forward propagation of the neural network.
     *
     * @param input Map containing input tensors.
     * @param output Map to store output tensors.
     * @param graph_id Id of the graph to be executed, currently only used in llm and
     *                 generally not need to be set unless there are multiple graphs.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Forward(const MatMap &input, MatMap &output, MI_S32 graph_id = 0) = 0;

    /**
     * @brief Get TensorDesc of input tensors for all graphs.
     *
     * @return TensorDesc for each input of each graph.
     */
    virtual std::vector<TensorDescMap> GetInputs() = 0;

    /**
     * @brief Get TensorDesc of output tensors for all graphs.
     *
     * @return TensorDesc for each output of each graph.
     */
    virtual std::vector<TensorDescMap> GetOutputs() = 0;

    /**
     * @brief Get the version of the NNExecutor.
     *
     * @return String representing the version information.
     */
    virtual std::string GetVersion() = 0;

protected:
    Context *m_ctx;  /*!< Pointer to the execution context. */
};

/**
 * @}
 */
} // namespace aura

#endif // AURA_RUNTIME_NN_NN_EXECUTOR_HPP__