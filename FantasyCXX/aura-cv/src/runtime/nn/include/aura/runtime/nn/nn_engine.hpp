#ifndef AURA_RUNTIME_NN_NN_ENGINE_HPP__
#define AURA_RUNTIME_NN_NN_ENGINE_HPP__

#include "aura/runtime/nn/nn_executor.hpp"

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
 * @brief Alias for a map of string names to std::unordered_map<std::string, std::string>.
 */
using NNConfig = std::unordered_map<std::string, std::string>;

/**
 * @brief Class representing a neural network engine.
 *
 * This class provides functionality for creating neural network executors.
 */
class AURA_EXPORTS NNEngine
{
public:
    /**
     * @brief Constructor for NNEngine.
     *
     * @param ctx Pointer to the execution context.
     * @param enable_nn Flag indicating whether neural network support is enabled.
     */
    NNEngine(Context *ctx, MI_BOOL enable_nn);

    /**
     * @brief Destructor for NNEngine.
     */
    ~NNEngine();

    /**
     * @brief Disable copy and assignment constructor operations.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(NNEngine);

    /**
     * @brief Create a neural network executor from minn file.
     * 从 minn 文件创建神经网络执行程序。
     * @param minn_file Path to the minn file.
     * @param decrypt_key Decryption key for the minn file.
     * @param config Configuration for the executor
     * Configuration options:
     *     ----------------------------------------------------------------------------------------------------------------
     *    |       key         |            value           |               explain                                         |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | perf_level        | PERF_HIGH (defalut)        | comm config, performance levels                               |
     *    |                   | PERF_NORMAL                | supported   platform: qcom(snpe/qnn), mtk(np), xiaomi(xnn)    |
     *    |                   | PERF_LOW                   | unsupported platform: mnn                                     |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | profiling_path    | user defined path string   | comm config, path for dump profiling                          |
     *    |                   | (default empty)            | supported   platform: qcom(snpe/qnn), xiaomi(xnn)             |
     *    |                   |                            | unsupported platform: mtk(np), mnn                            |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | log_level         | LOG_ERROR (default)        | comm config, log levels                                       |
     *    |                   | LOG_INFO                   | supported   platform: qcom(snpe/qnn), xiaomi(xnn)             |
     *    |                   | LOG_DEBUG                  | unsupported platform: mtk(np), mnn                            |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | backend           | CPU                        | comm config, backend type                                     |
     *    |                   | GPU                        | supported platform: qcom(snpe),  CPU/GPU/NPU (default NPU)    |
     *    |                   | NPU                        |                     qcom(qnn),   NPU         (default NPU)    |
     *    |                   |                            |                     mtk(np),     NPU         (default NPU)    |
     *    |                   |                            |                     xiaomi(xnn), NPU         (default NPU)    |
     *    |                   |                            |                     mnn,         CPU/GPU/NPU (default CPU)    |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | profiling_level   | PROFILING_OFF (default)    | comm config, profiling levels                                 |
     *    |                   | PROFILING_BASIC            | supported platform: qcom(snpe/qnn)                            |
     *    |                   | PROFILING_DETAILED         | unsupported platform: mtk(np), xiaomi(xnn), mnn               |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | htp_mem_step_size | 0~32 (default 0)           | qcom htp config, unit in MB                                   |
     *    |                   |                            | supported platform: qcom(snpe/qnn)                            |
     *    |                   |                            | unsupported platform: mtk(np), xiaomi(xnn), mnn               |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | htp_async_call    | TRUE (default)             | specialized config, whether qnn/snpe forward use async thread |
     *    |                   |                            |                     call                                      |
     *    |                   | FALSE                      | supported platform: qcom(qnn/snpe)                            |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | snpe_unsigned_pd  | TRUE (default)             | specialized config, whether unsigned PD is enabled            |
     *    |                   | FALSE                      | supported platform: qcom(snpe)                                |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | qnn_budget        | 1,2,...,32,... (default 0) | configure the read memory budget of the model in megabytes    |
     *    |                   |                            | if budget is 0 or budget size is greater than model size,     |
     *    |                   |                            | budget would be set to disabled                               |
     *    |                   |                            | supported platform: qcom(qnn)                                 |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | qnn_graph_ids     | 1;2;3 (default empty)      | specialized config, specify which graph to init               |
     *    |                   |                            | supported platform: qcom(qnn)                                 |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | qnn_udo_path      | user defined path string   | specialized config, Path to the User Defined Operation file   |
     *    |                   | (default empty)            | supported platform: qcom(qnn)                                 |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | qnn_hmx_perf_level| PERF_HIGH (default)        | specialized config, set qnn hmx performance levels            |
     *    |                   | PERF_NORMAL                | supported platform: qcom(qnn)                                 |
     *    |                   | PERF_LOW                   |                                                               |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | mnn_dump_layers   | TRUE                       | specialized config, whether dump all layers output            |
     *    |                   | FALSE (default)            | supported platform: mnn                                       |
     *    |                   |                            |                                                               |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | mnn_precision     | PRECISION_HIGH             | specialized config, Precision for the MNN executor            |
     *    |                   | PRECISION_NORMAL (default) | supported platform: mnn                                       |
     *    |                   | PRECISION_LOW              |                                                               |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | mnn_memory        | MEMORY_HIGH                | specialized config, Memory configuration for the MNN executor |
     *    |                   | MEMORY_NORMAL (default)    | supported platform: mnn                                       |
     *    |                   | MEMORY_LOW                 |                                                               |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | mnn_tuning        | GPU_TUNING_NONE (default)  | specialized config, Tuning options for the MNN executor       |
     *    |                   | GPU_TUNING_HEAVY           | supported platform: mnn                                       |
     *    |                   | GPU_TUNING_WIDE            |                                                               |
     *    |-------------------|----------------------------|---------------------------------------------------------------|
     *    | mnn_clmem         | GPU_MEMORY_NONE (default)  | specialized config, OpenCL memory type for the MNN executor   |
     *    |                   | GPU_MEMORY_BUFFER          | supported platform: mnn                                       |
     *    |                   | GPU_MEMORY_IAURA           |                                                               |
     *     ----------------------------------------------------------------------------------------------------------------
     * Configuration usage
     *     NNConfig config;
     *     config["perf_level"] = "PERF_NORMAL";
     * @return Shared pointer to the created NNExecutor, or MI_NULL if the creation failed.
     */
    std::shared_ptr<NNExecutor> CreateNNExecutor(const std::string &minn_file,
                                                 const std::string &decrypt_key,
                                                 const NNConfig &config = NNConfig());

    /**
     * @brief Create a neural network executor from minn data.
     *
     * @param minn_data minn data pointer. 传入模型数据的内存指针
     * @param minn_size minn data size.     数据大小
     * @param decrypt_key Decryption key for the minn file.  解密密钥
     * @param config Configuration for the executor.       执行器配置
     * Configuration options: same as above
     * Configuration usage: same as above
     * @return Shared pointer to the created NNExecutor, or MI_NULL if the creation failed.
     */
    std::shared_ptr<NNExecutor> CreateNNExecutor(const MI_U8 *minn_data,
                                                 MI_S64 minn_size,
                                                 const std::string &decrypt_key,
                                                 const NNConfig &config = NNConfig());

    /**
     * @brief Create a neural network executor from minb file.
     *
     * @param minb_file Path to the minb file.
     * @param minn_name minn name in minb.
     * @param decrypt_key Decryption key for the target minn in minb file.
     * @param config Configuration for the executor
     * @return Shared pointer to the created NNExecutor, or MI_NULL if the creation failed.
     */
    std::shared_ptr<NNExecutor> CreateNNExecutor(const std::string &minb_file,
                                                 const std::string &minn_name,
                                                 const std::string &decrypt_key,
                                                 const NNConfig &config = NNConfig());

private:
    Context *m_ctx;         /*!< Pointer to the execution context. */
    MI_BOOL m_enable_nn;    /*!< Flag indicating whether neural network support is enabled. */
};

/**
 * @}
 */
} // namespace aura

#endif // AURA_RUNTIME_NN_NN_ENGINE_HPP__