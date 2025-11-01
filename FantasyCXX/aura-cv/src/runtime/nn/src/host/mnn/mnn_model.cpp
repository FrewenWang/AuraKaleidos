#include "mnn/mnn_model.hpp"
#include "nn_deserialize.hpp"

namespace aura
{

MnnModel::MnnModel(Context *ctx, const ModelInfo &model_info) : NNModel(ctx, model_info)
{
    do
    {
        if (!m_is_valid)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "create NNModel failed");
            break;
        }

        m_is_valid = MI_FALSE;

        struct
        {
            struct
            {
                MI_U8 precision = 0;
                MI_U8 memory    = 0;
                MI_U8 tuning    = 0;
                MI_U8 cl_mem    = 0;
            } gpu_config;

            MI_U8 model_type    = 0;
        } mnn_info;

        if (std::string::npos == m_framework_version.find("mnn"))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "check framework failed");
            break;
        }

        if ((m_backend_type != "cpu") && (m_backend_type != "gpu"))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "check backend failed");
            break;
        }

        if (NNDeserialize(m_ctx, model_info.minn_buffer, m_data_offset, mnn_info) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
            break;
        }

        if (1 == mnn_info.gpu_config.precision)
        {
            m_precision = MnnPrecision::PRECISION_NORMAL;
        }
        else if (2 == mnn_info.gpu_config.precision)
        {
            m_precision = MnnPrecision::PRECISION_HIGH;
        }
        else if (3 == mnn_info.gpu_config.precision)
        {
            m_precision = MnnPrecision::PRECISION_LOW;
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported precision");
            break;
        }

        if (1 == mnn_info.gpu_config.memory)
        {
            m_memory = MnnMemory::MEMORY_NORMAL;
        }
        else if (2 == mnn_info.gpu_config.memory)
        {
            m_memory = MnnMemory::MEMORY_HIGH;
        }
        else if (3 == mnn_info.gpu_config.memory)
        {
            m_memory = MnnMemory::MEMORY_LOW;
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported memory");
            break;
        }

        if (1 == mnn_info.gpu_config.tuning)
        {
            m_tuning = MnnTuning::GPU_TUNING_NONE;
        }
        else if (2 == mnn_info.gpu_config.tuning)
        {
            m_tuning = MnnTuning::GPU_TUNING_HEAVY;
        }
        else if (3 == mnn_info.gpu_config.tuning)
        {
            m_tuning = MnnTuning::GPU_TUNING_WIDE;
        }
        else if (4 == mnn_info.gpu_config.tuning)
        {
            m_tuning = MnnTuning::GPU_TUNING_NORMAL;
        }
        else if (5 == mnn_info.gpu_config.tuning)
        {
            m_tuning = MnnTuning::GPU_TUNING_FAST;
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported tuning");
            break;
        }

        if (1 == mnn_info.gpu_config.cl_mem)
        {
            m_cl_mem = MnnCLMem::GPU_MEMORY_NONE;
        }
        else if (2 == mnn_info.gpu_config.cl_mem)
        {
            m_cl_mem = MnnCLMem::GPU_MEMORY_BUFFER;
        }
        else if (3 == mnn_info.gpu_config.cl_mem)
        {
            m_cl_mem = MnnCLMem::GPU_MEMORY_IAURA;
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported cl_mem");
            break;
        }

        if (mnn_info.model_type != 2)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported model type");
            break;
        }

        m_is_valid = MI_TRUE;
    } while (0);
}

MnnPrecision MnnModel::GetPrecision() const
{
    return m_precision;
}

MnnMemory MnnModel::GetMemory() const
{
    return m_memory;
}

MnnTuning MnnModel::GetTuning() const
{
    return m_tuning;
}

MnnCLMem MnnModel::GetCLMem() const
{
    return m_cl_mem;
}

} // namespace aura