#ifndef AURA_NN_RUN_HPP__
#define AURA_NN_RUN_HPP__

#include "aura/runtime/nn.h"
#include "aura/runtime/logger.h"
#include "aura/tools/unit_test/mat_factory.hpp"

namespace aura
{

typedef DT_VOID(*MemProfilerTrace)(const DT_CHAR *tag);

struct InputDataInfo
{
    std::string tensor_name;
    std::string file_name;
};

struct CommandParam
{
    CommandParam() : show_time(DT_FALSE),
                     get_minb_info(DT_FALSE),
                     get_tensor_info(DT_FALSE),
                     register_mem(DT_FALSE),
                     use_random_inputs(DT_FALSE),
                     qnn_graph_ids("0"),
                     loop_type(0),
                     loop_num(0),
                     inference_interval(0),
                     loop_num_inner(0),
                     loop_num_outer(0),
                     memory_tag_info(DT_FALSE),
                     func_memProfiler_trace_start(NULL),
                     func_memProfiler_trace_end(NULL)
    {}

    DT_BOOL show_time;
    DT_BOOL get_minb_info;
    DT_BOOL get_tensor_info;
    DT_BOOL register_mem;
    DT_BOOL use_random_inputs;
    std::string platform;
    std::string model_path;
    std::string model_container_path;
    std::string password;
    std::string input_list;
    std::string profiling_path;
    std::string dump_path;
    std::string output_path;
    std::string custom_config;
    std::string perf_level;
    std::string profiling_level;
    std::string log_level;
    std::string htp_mem_step_size;
    std::string snpe_unsigned_pd;
    std::string qnn_graph_ids;
    std::string qnn_udo_path;
    std::string mnn_dump_layers;
    std::string mnn_tuning;
    std::string mnn_clmem;
    std::string minn_name;
    std::string src_type;
    std::string dst_type;
    DT_S32 loop_type;
    DT_S32 loop_num;
    DT_S32 inference_interval;
    std::string backend;
    std::string mnn_precision;
    std::string mnn_memory;
    DT_S32 loop_num_inner;
    DT_S32 loop_num_outer;
    DT_S32 memory_tag_info;
    MemProfilerTrace func_memProfiler_trace_start;
    MemProfilerTrace func_memProfiler_trace_end;
};

class AuraNnRun
{
public:
    AuraNnRun() : m_error_str_len(0)
    {
        Config config;

        // set m_config param
        config.SetNNConf(DT_TRUE);

        // Create context for AuraNnRun
        m_ctx = std::make_shared<Context>(config);
    }

    ~AuraNnRun()
    {
        std::string error_str = m_ctx->GetLogger()->GetErrorString();
        if (error_str.size() > m_error_str_len)
        {
             AURA_LOGE(m_ctx, AURA_TAG, "aura-nn-run failed, %s\n", error_str.c_str());
        }
    }

    Status Initialize()
    {
        if (DT_NULL == m_ctx)
        {
            std::cout << "Context create failed." << std::endl;
            return Status::ERROR;
        }

        if (m_ctx->Initialize() != Status::OK)
        {
            std::cout << "Context Initialize failed." << std::endl;
            return Status::ERROR;
        }

        std::string error_str = m_ctx->GetLogger()->GetErrorString();
        m_error_str_len = error_str.size();

        AURA_LOGD(m_ctx.get(), AURA_TAG, "VersionInfo: %s\n", m_ctx->GetVersion().c_str());

        return Status::OK;
    }

    Status ParseCommandLine(DT_S32 argc, DT_CHAR *argv[]);
    Status Run();

private:
    Status NetRun(Context *ctx, const CommandParam &param);

private:
    std::shared_ptr<Context> m_ctx;
    CommandParam             m_commond_param;
    size_t                   m_error_str_len;
};

} // namespace aura
#endif // AURA_NN_RUN_HPP__