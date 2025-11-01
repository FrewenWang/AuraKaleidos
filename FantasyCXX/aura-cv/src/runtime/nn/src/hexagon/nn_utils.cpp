#include "aura/runtime/nn/nn_utils.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status KeyGenerate(Context *ctx, const std::string &main_key, std::string &gen_key)
{
    if (main_key.empty())
    {
        AURA_ADD_ERROR_STRING(ctx, "key is empty");
        return Status::ERROR;
    }

    gen_key = main_key;
    gen_key.resize(16, main_key.at(main_key.size() - 1));

    return Status::OK;
}

static Status SymmetricKeyAlgo(Context *ctx, const Buffer &src, Buffer &dst, const std::string &key, MI_U64 size)
{
    if (src.m_size != dst.m_size)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst's size should be same");
        return Status::ERROR;
    }

    if (size > static_cast<MI_U64>(dst.m_size))
    {
        AURA_ADD_ERROR_STRING(ctx, "size can not be larger than dst's size");
        return Status::ERROR;
    }

    if (key.size() == 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "key size should not be 0");
        return Status::ERROR;
    }

    std::string gen_key;
    MI_U64 size_align128 = size & (-128);
    MI_U64 size_align16  = size & (-16);
    MI_U8 u8_key_value[128];

    if (KeyGenerate(ctx, key, gen_key) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "KeyGenerate failed");
        return Status::ERROR;
    }

    const MI_U8 *src_data = reinterpret_cast<MI_U8*>(src.m_data);
    MI_U8 *dst_data = reinterpret_cast<MI_U8*>(dst.m_data);
    MI_U64 i = 0;

    for (i = 0; i < 8; i++)
    {
        memcpy(u8_key_value + i * gen_key.size(), gen_key.data(), gen_key.size());
    }

    HVX_Vector v_src, v_dst, v_key;
    i = 0;

    vload(u8_key_value, v_key);
    for (; i < size_align128; i += 128)
    {
        vload(src_data + i, v_src);
        v_dst = Q6_V_vxor_VV(v_src, v_key);
        vstore(dst_data + i, v_dst);
    }

    for (; i < size_align16; i += 16)
    {
        for (MI_S32 k = 0; k < 16; k++)
        {
            dst_data[i + k] = src_data[i + k] ^ (MI_U8)gen_key[k];
        }
    }

    if (src_data != dst_data)
    {
        for (; i < static_cast<MI_U64>(dst.m_size); i++)
        {
            dst_data[i] = src_data[i];
        }
    }

    return Status::OK;
}

Status NNEncrypt(Context *ctx, const Buffer &src, Buffer &dst, const std::string &key, MI_U64 size)
{
    if (SymmetricKeyAlgo(ctx, src, dst, key, size) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "SymmetricKeyAlgo fail");
        return Status::ERROR;
    }

    return Status::OK;
}

Status NNDecrypt(Context *ctx, const Buffer &src, Buffer &dst, const std::string &key, MI_U64 size)
{
    if (SymmetricKeyAlgo(ctx, src, dst, key, size) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "SymmetricKeyAlgo fail");
        return Status::ERROR;
    }

    return Status::OK;
}

Status NNQuantize(Context *ctx, const Mat &src, Mat &dst, MI_S32 zero_point, MI_F32 scale)
{
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(zero_point);
    AURA_UNUSED(scale);

    AURA_ADD_ERROR_STRING(ctx, "dst side not suppose NNQuantize");
    return Status::ERROR;
}

Status NNDeQuantize(Context *ctx, const Mat &src, Mat &dst, MI_S32 zero_point, MI_F32 scale)
{
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(zero_point);
    AURA_UNUSED(scale);

    AURA_ADD_ERROR_STRING(ctx, "dst side not suppose NNDeQuantize");
    return Status::ERROR;
}

std::vector<std::string> NNSplit(const std::string &src, MI_CHAR separator)
{
    std::vector<std::string> result;
    std::string value;
    std::istringstream tokenized_string_stream(src);

    while (getline(tokenized_string_stream, value, separator))
    {
        result.push_back(value);
    }

    return result;
}


} // namespace aura