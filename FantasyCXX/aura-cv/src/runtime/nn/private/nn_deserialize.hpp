
#ifndef AURA_RUNTIME_NN_NN_DESERIALIZE_HPP__
#define AURA_RUNTIME_NN_NN_DESERIALIZE_HPP__

#include "aura/runtime/nn/nn_utils.hpp"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"

#include <unordered_map>

namespace aura
{

template <typename Tp>
AURA_INLINE Status NNDeserialize(Context *ctx, const Buffer &minn_buffer, MI_S64 &offset, Tp &val)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    if (!minn_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "minn_buffer is invalid");
        return Status::ERROR;
    }

    if (offset + static_cast<MI_S32>(sizeof(Tp)) > minn_buffer.m_size)
    {
        std::string info = "minn_buffer overflow, curr minn buffer data pos is " + std::to_string(offset) + ", again read " + \
                            std::to_string(sizeof(Tp)) + "bytes, will excess minn_buffer size " + std::to_string(minn_buffer.m_size);
        AURA_ADD_ERROR_STRING(ctx, info.c_str() );
        return Status::ERROR;
    }

#if defined(AURA_BUILD_HEXAGON)
    AuraMemCopy(&val, (static_cast<MI_U8*>(minn_buffer.m_data) + offset), sizeof(Tp));
#else
    memcpy(&val, (static_cast<MI_U8*>(minn_buffer.m_data) + offset), sizeof(Tp));
#endif // AURA_BUILD_HEXAGON
    offset += sizeof(Tp);

    return Status::OK;
}

AURA_INLINE Status NNDeserialize(Context *ctx, const Buffer &minn_buffer, MI_S64 &offset, std::string &str)
{
    MI_U32 size = 0;

    if (NNDeserialize(ctx, minn_buffer, offset, size) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
        return Status::ERROR;
    }

    if (0 == size)
    {
        AURA_ADD_ERROR_STRING(ctx, "empty string");
        return Status::ERROR;
    }

    if (offset + size > minn_buffer.m_size)
    {
        std::string info = "minn_buffer overflow, curr minn buffer data pos is " + std::to_string(offset) + ", again read " + \
                            std::to_string(size) + "bytes, will excess minn_buffer size " + std::to_string(minn_buffer.m_size);
        AURA_ADD_ERROR_STRING(ctx, info.c_str() );
        return Status::ERROR;
    }

    str.resize(size);
    memcpy(&str[0], static_cast<MI_U8*>(minn_buffer.m_data) + offset, size);
    offset += size;

    return Status::OK;
}

AURA_INLINE Status NNDeserialize(Context *ctx, const Buffer &minn_buffer, MI_S64 &offset, MI_CHAR key, std::vector<std::string> &vec)
{
    std::string str;

    if (NNDeserialize(ctx, minn_buffer, offset, str) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
        return Status::ERROR;
    }

    if (str.empty())
    {
        AURA_ADD_ERROR_STRING(ctx, "empty string");
        return Status::ERROR;
    }

    for (size_t i = 0; i < str.size(); i++)
    {
        str[i] ^= key;
    }

    vec = NNSplit(str, ',');

    return Status::OK;
}

}

#endif // AURA_RUNTIME_NN_NN_DESERIALIZE_HPP__