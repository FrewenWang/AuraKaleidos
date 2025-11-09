#include "np/np_model.hpp"
#include "nn_deserialize.hpp"

namespace aura
{

static Status NNDeserialize(Context *ctx, const Buffer &minn_buffer, DT_S64 &offset, std::vector<DT_S32> &sizes)
{
    if (!minn_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "minn_buffer is invalid");
        return Status::ERROR;
    }

    DT_S32 size = 0;

    if (NNDeserialize(ctx, minn_buffer, offset, size) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NNDeserialize anallis size falied");
        return Status::ERROR;
    }

    if (0 == size)
    {
        AURA_ADD_ERROR_STRING(ctx, "size is zero, size must greate zero");
        return Status::ERROR;
    }

    if (offset + size > minn_buffer.m_size)
    {
        std::string info = "minn_buffer overflow, curr minn buffer data pos is " + std::to_string(offset) + ", again read " + \
                            std::to_string(size) + "bytes, will excess minn_buffer size " + std::to_string(minn_buffer.m_size);
        AURA_ADD_ERROR_STRING(ctx, info.c_str() );
        return Status::ERROR;
    }

    sizes.resize(size);
    memcpy(sizes.data(), static_cast<DT_CHAR*>(minn_buffer.m_data) + offset, size * sizeof(DT_S32));
    offset += size * sizeof(DT_S32);

    return Status::OK;
}

static Status NNDeserialize(Context *ctx, const Buffer &minn_buffer, DT_S64 &offset, NpModel::TensorAttr &tensor_attr)
{
    if (!minn_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "minn_buffer is invalid");
        return Status::ERROR;
    }

    if (NNDeserialize(ctx, minn_buffer, offset, tensor_attr.zero_point) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NNDeserialize anallis tensor_attr zero_point failed");
        return Status::ERROR;
    }

    if (NNDeserialize(ctx, minn_buffer, offset, tensor_attr.scale) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NNDeserialize anallis tensor_attr scale failed");
        return Status::ERROR;
    }

    if (NNDeserialize(ctx, minn_buffer, offset, tensor_attr.elem_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NNDeserialize anallis tensor_attr elem_type failed");
        return Status::ERROR;
    }

    if (NNDeserialize(ctx, minn_buffer, offset, tensor_attr.sizes) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NNDeserialize anallis tensor_attr sizes failed");
        return Status::ERROR;
    }

    return Status::OK;
}

NpModel::NpModel(Context *ctx, const ModelInfo &model_info) : NNModel(ctx, model_info)
{
    do
    {
        if (!m_is_valid)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "create NNModel failed");
            break;
        }

        m_is_valid = DT_FALSE;

        if (std::string::npos == m_framework_version.find("np"))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "check framework failed");
            break;
        }

        TensorAttr tensor_attr;
        std::vector<std::string> input_attr_name;
        std::vector<std::string> output_attr_name;

        if (NNDeserialize(m_ctx, model_info.minn_buffer, m_data_offset, 0x55, input_attr_name) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
            break;
        }

        for (size_t i = 0; i < input_attr_name.size(); i++)
        {
            if (NNDeserialize(m_ctx, model_info.minn_buffer, m_data_offset, tensor_attr) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
                break;
            }

            m_input_attrs.emplace_back(input_attr_name[i], tensor_attr);
        }

        if (NNDeserialize(m_ctx, model_info.minn_buffer, m_data_offset, 0x55, output_attr_name) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
            break;
        }

        for (size_t i = 0; i < output_attr_name.size(); i++)
        {
            if (NNDeserialize(m_ctx, model_info.minn_buffer, m_data_offset, tensor_attr) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
                break;
            }

            m_output_attrs.emplace_back(output_attr_name[i], tensor_attr);
        }

        m_is_valid = DT_TRUE;
    } while (0);
}

const std::vector<std::pair<std::string, NpModel::TensorAttr>>& NpModel::GetInputAttrs() const
{
    return m_input_attrs;
}

const std::vector<std::pair<std::string, NpModel::TensorAttr>>& NpModel::GetOutputAttrs() const
{
    return m_output_attrs;
}

} // namespace aura