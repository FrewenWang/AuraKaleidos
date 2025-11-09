#include "qnn/qnn_model.hpp"

namespace aura
{

QnnModel::QnnModel(Context *ctx, const ModelInfo &model_info) : NNModel(ctx, model_info)
{
    do
    {
        if (!m_is_valid)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "create NNModel failed");
            break;
        }

        m_is_valid = DT_FALSE;

        if (std::string::npos == m_framework_version.find("qnn"))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "check framework failed");
            break;
        }

        if ((m_backend_type != "gpu") && (m_backend_type != "npu"))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "check backend failed");
            break;
        }

        m_is_valid = DT_TRUE;
    } while (0);
}

} // namespace aura