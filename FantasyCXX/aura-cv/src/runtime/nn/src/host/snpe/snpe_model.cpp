#include "snpe/snpe_model.hpp"
#include "nn_deserialize.hpp"

namespace aura
{

SnpeModel::SnpeModel(Context *ctx, const ModelInfo &model_info) : NNModel(ctx, model_info)
{
    do
    {
        if (!m_is_valid)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "create NNModel failed");
            break;
        }

        m_is_valid = MI_FALSE;

        if (std::string::npos == m_framework_version.find("snpe"))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "check framework failed");
            break;
        }

        if (NNDeserialize(m_ctx, model_info.minn_buffer, m_data_offset, 0x55, m_output_layer_names) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
            break;
        }

        m_is_valid = MI_TRUE;
    } while (0);
}

const std::vector<std::string>& SnpeModel::GetOutputLayerNames() const
{
    return m_output_layer_names;
}

} // namespace aura