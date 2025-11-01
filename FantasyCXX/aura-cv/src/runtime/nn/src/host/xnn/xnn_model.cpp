#include "xnn/xnn_model.hpp"
#include "nn_deserialize.hpp"

namespace aura
{

XnnModel::XnnModel(Context *ctx, const ModelInfo &model_info) : NNModel(ctx, model_info)
{
    do
    {
        if (!m_is_valid)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "create NNModel failed");
            break;
        }

        m_is_valid = MI_FALSE;

        if (std::string::npos == m_framework_version.find("xnn"))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "check framework failed");
            break;
        }

        if (m_backend_type != "npu")
        {
            AURA_ADD_ERROR_STRING(m_ctx, "check backend failed");
            break;
        }

        m_is_valid = MI_TRUE;
    } while (0);
}

} // namespace auras