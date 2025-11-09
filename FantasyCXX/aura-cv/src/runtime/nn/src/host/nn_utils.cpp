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

static Status SymmetricKeyAlgo(Context *ctx, const Buffer &src, Buffer &dst, const std::string &key, DT_U64 size)
{
    if (src.m_size != dst.m_size)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst's size should be same");
        return Status::ERROR;
    }

    if (size > static_cast<DT_U64>(dst.m_size))
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
    DT_U64 size_align16  = size & (-16);

    if (KeyGenerate(ctx, key, gen_key) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "KeyGenerate failed");
        return Status::ERROR;
    }

    const DT_U8 *src_data = reinterpret_cast<DT_U8*>(src.m_data);
    DT_U8 *dst_data = reinterpret_cast<DT_U8*>(dst.m_data);
    DT_U64 i = 0;

#if defined(AURA_ENABLE_NEON)
    uint8x16_t vqu8_src[2];
    uint8x16_t vqu8_dst[2];
    uint8x16_t vqu8_realkey = neon::vload1q((DT_U8 *)(gen_key.data()));
    DT_U64 size_align32  = size & (-32);

    for (; i < size_align32; i += 32)
    {
        vqu8_src[0] = neon::vload1q(src_data + i    );
        vqu8_src[1] = neon::vload1q(src_data + i + 16);

        vqu8_dst[0] = neon::veor(vqu8_src[0], vqu8_realkey);
        vqu8_dst[1] = neon::veor(vqu8_src[1], vqu8_realkey);

        neon::vstore(dst_data + i     , vqu8_dst[0]);
        neon::vstore(dst_data + i + 16, vqu8_dst[1]);
    }
#endif

    for (; i < size_align16; i += 16)
    {
        for (DT_S32 k = 0; k < 16; k++)
        {
            dst_data[i + k] = src_data[i + k] ^ (DT_U8)gen_key[k];
        }
    }

    if (src_data != dst_data)
    {
        for (; i < static_cast<DT_U64>(dst.m_size); i++)
        {
            dst_data[i] = src_data[i];
        }
    }

    return Status::OK;
}

Status NNEncrypt(Context *ctx, const Buffer &src, Buffer &dst, const std::string &key, DT_U64 size)
{
    if (SymmetricKeyAlgo(ctx, src, dst, key, size) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "SymmetricKeyAlgo fail");
        return Status::ERROR;
    }

    return Status::OK;
}

Status NNDecrypt(Context *ctx, const Buffer &src, Buffer &dst, const std::string &key, DT_U64 size)
{
    if (SymmetricKeyAlgo(ctx, src, dst, key, size) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "SymmetricKeyAlgo fail");
        return Status::ERROR;
    }

    return Status::OK;
}

Status NNQuantize(Context *ctx, const Mat &src, Mat &dst, DT_S32 zero_point, DT_F32 scale)
{
    if (!src.IsSizesEqual(dst))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst's sizes should be same");
        return Status::ERROR;
    }

    if (src.GetElemType() != ElemType::F32)
    {
        AURA_ADD_ERROR_STRING(ctx, "src elemtype error");
        return Status::ERROR;
    }

    DT_S32 width  = dst.GetSizes().m_width * dst.GetSizes().m_channel;
    DT_S32 height = dst.GetSizes().m_height;
    const DT_F32 *src_row = DT_NULL;

#if defined(AURA_ENABLE_NEON)
    float32x4_t vqf32_scale;
    neon::vdup(vqf32_scale, 1.f / scale);
#endif

    if (dst.GetElemType() == ElemType::U8)
    {
        DT_U8 *dst_row = DT_NULL;
        DT_U8 zp = static_cast<DT_U8>(zero_point);

#if defined(AURA_ENABLE_NEON)
        DT_S32 width_align16 = width & (-16);
        float32x4_t vqf32_src[4];
        int32x4_t vqs32_tmp[4];
        int16x8_t vqs16_zp;
        neon::vdup(vqs16_zp, (DT_S16)zp);
#endif
        for (DT_S32 y = 0; y < height; y++)
        {
            src_row = src.Ptr<DT_F32>(y);
            dst_row = dst.Ptr<DT_U8>(y);
            DT_S32 x = 0;

#if defined(AURA_ENABLE_NEON)
            for (; x < width_align16; x += 16)
            {
                vqf32_src[0] = neon::vload1q(src_row + x    );
                vqf32_src[1] = neon::vload1q(src_row + x + 4);
                vqf32_src[2] = neon::vload1q(src_row + x + 8);
                vqf32_src[3] = neon::vload1q(src_row + x + 12);

                vqf32_src[0] = neon::vmul(vqf32_src[0], vqf32_scale);
                vqf32_src[1] = neon::vmul(vqf32_src[1], vqf32_scale);
                vqf32_src[2] = neon::vmul(vqf32_src[2], vqf32_scale);
                vqf32_src[3] = neon::vmul(vqf32_src[3], vqf32_scale);

                vqs32_tmp[0] = neon::vcvtn<DT_S32>(vqf32_src[0]);
                vqs32_tmp[1] = neon::vcvtn<DT_S32>(vqf32_src[1]);
                vqs32_tmp[2] = neon::vcvtn<DT_S32>(vqf32_src[2]);
                vqs32_tmp[3] = neon::vcvtn<DT_S32>(vqf32_src[3]);

                int16x8_t vqs16_l = neon::vcombine(neon::vqmovn(vqs32_tmp[0]), neon::vqmovn(vqs32_tmp[1]));
                int16x8_t vqs16_h = neon::vcombine(neon::vqmovn(vqs32_tmp[2]), neon::vqmovn(vqs32_tmp[3]));
                vqs16_l = neon::vadd(vqs16_l, vqs16_zp);
                vqs16_h = neon::vadd(vqs16_h, vqs16_zp);

                uint8x16_t vqu8_dst = neon::vcombine(neon::vqmovun(vqs16_l), neon::vqmovun(vqs16_h));

                neon::vstore(dst_row + x, vqu8_dst);
            }
#endif
            for (; x < width; x++)
            {
                dst_row[x] = SaturateCast<DT_U8>(Round(src_row[x] / scale + zp));
            }
        }
    }
    else if (dst.GetElemType() == ElemType::S8)
    {
        DT_S8 *dst_row = DT_NULL;
        DT_S8 zp       = static_cast<DT_S8>(zero_point);
#if defined(AURA_ENABLE_NEON)
        DT_S32 width_align16 = width & (-16);
        float32x4_t vqf32_src[4];
        int32x4_t vqs32_tmp[4];
        int16x8_t vqs16_zp;
        neon::vdup(vqs16_zp, (DT_S16)zp);
#endif
        for (DT_S32 y = 0; y < height; y++)
        {
            src_row = src.Ptr<DT_F32>(y);
            dst_row = dst.Ptr<DT_S8>(y);
            DT_S32 x = 0;

#if defined(AURA_ENABLE_NEON)
            for (; x < width_align16; x += 16)
            {
                vqf32_src[0] = neon::vload1q(src_row + x    );
                vqf32_src[1] = neon::vload1q(src_row + x + 4);
                vqf32_src[2] = neon::vload1q(src_row + x + 8);
                vqf32_src[3] = neon::vload1q(src_row + x + 12);

                vqf32_src[0] = neon::vmul(vqf32_src[0], vqf32_scale);
                vqf32_src[1] = neon::vmul(vqf32_src[1], vqf32_scale);
                vqf32_src[2] = neon::vmul(vqf32_src[2], vqf32_scale);
                vqf32_src[3] = neon::vmul(vqf32_src[3], vqf32_scale);

                vqs32_tmp[0] = neon::vcvtn<DT_S32>(vqf32_src[0]);
                vqs32_tmp[1] = neon::vcvtn<DT_S32>(vqf32_src[1]);
                vqs32_tmp[2] = neon::vcvtn<DT_S32>(vqf32_src[2]);
                vqs32_tmp[3] = neon::vcvtn<DT_S32>(vqf32_src[3]);

                int16x8_t vqs16_l = neon::vcombine(neon::vqmovn(vqs32_tmp[0]), neon::vqmovn(vqs32_tmp[1]));
                int16x8_t vqs16_h = neon::vcombine(neon::vqmovn(vqs32_tmp[2]), neon::vqmovn(vqs32_tmp[3]));
                vqs16_l = neon::vadd(vqs16_l, vqs16_zp);
                vqs16_h = neon::vadd(vqs16_h, vqs16_zp);

                int8x16_t vqs8_dst = neon::vcombine(neon::vqmovn(vqs16_l), neon::vqmovn(vqs16_h));

                neon::vstore(dst_row + x, vqs8_dst);
            }
#endif
            for (; x < width; x++)
            {
                dst_row[x] = SaturateCast<DT_S8>(Round(src_row[x] / scale + zp));
            }
        }
    }
    else if (dst.GetElemType() == ElemType::U16)
    {
        DT_U16 *dst_row = DT_NULL;
        DT_U16 zp       = static_cast<DT_U16>(zero_point);

#if defined(AURA_ENABLE_NEON)
        DT_S32 width_align8  = width & (-8);
        float32x4_t vqf32_src[2];
        int32x4_t vqs32_zp;
        neon::vdup(vqs32_zp, (DT_S32)zp);
#endif

        for (DT_S32 y = 0; y < height; y++)
        {
            src_row = src.Ptr<DT_F32>(y);
            dst_row = dst.Ptr<DT_U16>(y);
            DT_S32 x = 0;

#if defined(AURA_ENABLE_NEON)
            for (; x < width_align8; x += 8)
            {
                vqf32_src[0] = neon::vload1q(src_row + x    );
                vqf32_src[1] = neon::vload1q(src_row + x + 4);

                vqf32_src[0] = neon::vmul(vqf32_src[0], vqf32_scale);
                vqf32_src[1] = neon::vmul(vqf32_src[1], vqf32_scale);

                int32x4_t vqs32_l = neon::vcvtn<DT_S32>(vqf32_src[0]);
                int32x4_t vqs32_h = neon::vcvtn<DT_S32>(vqf32_src[1]);

                vqs32_l = neon::vqadd(vqs32_l, vqs32_zp);
                vqs32_h = neon::vqadd(vqs32_h, vqs32_zp);

                uint16x8_t vqu16_dst = neon::vcombine(neon::vqmovun(vqs32_l), neon::vqmovun(vqs32_h));

                neon::vstore(dst_row + x, vqu16_dst);
            }
#endif
            for (; x < width; x++)
            {
                dst_row[x] = SaturateCast<DT_U16>(Round(src_row[x] / scale + zp));
            }
        }
    }
    else if (dst.GetElemType() == ElemType::S16)
    {
        DT_S16 *dst_row = DT_NULL;
        DT_S16 zp       = static_cast<DT_S16>(zero_point);

#if defined(AURA_ENABLE_NEON)
        DT_S32 width_align8  = width & (-8);
        float32x4_t vqf32_src[2];
        int32x4_t vqs32_zp;
        neon::vdup(vqs32_zp, (DT_S32)zp);
#endif

        for (DT_S32 y = 0; y < height; y++)
        {
            src_row = src.Ptr<DT_F32>(y);
            dst_row = dst.Ptr<DT_S16>(y);
            DT_S32 x = 0;

#if defined(AURA_ENABLE_NEON)
            for (; x < width_align8; x += 8)
            {
                vqf32_src[0] = neon::vload1q(src_row + x    );
                vqf32_src[1] = neon::vload1q(src_row + x + 4);

                vqf32_src[0] = neon::vmul(vqf32_src[0], vqf32_scale);
                vqf32_src[1] = neon::vmul(vqf32_src[1], vqf32_scale);

                int32x4_t vqs32_l = neon::vcvtn<DT_S32>(vqf32_src[0]);
                int32x4_t vqs32_h = neon::vcvtn<DT_S32>(vqf32_src[1]);

                vqs32_l = neon::vqadd(vqs32_l, vqs32_zp);
                vqs32_h = neon::vqadd(vqs32_h, vqs32_zp);

                int16x8_t vqs16_dst = neon::vcombine(neon::vqmovn(vqs32_l), neon::vqmovn(vqs32_h));

                neon::vstore(dst_row + x, vqs16_dst);
            }
#endif
            for (; x < width; x++)
            {
                dst_row[x] = SaturateCast<DT_S16>(Round(src_row[x] / scale + zp));
            }
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "can not support this dst type quantation");
        return Status::ERROR;
    }

    return Status::OK;
}

Status NNDeQuantize(Context *ctx, const Mat &src, Mat &dst, DT_S32 zero_point, DT_F32 scale)
{
    if (!src.IsSizesEqual(dst))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst's sizes should be same");
        return Status::ERROR;
    }

    if (dst.GetElemType() != ElemType::F32)
    {
        AURA_ADD_ERROR_STRING(ctx, "dst elemtype error");
        return Status::ERROR;
    }

    DT_S32 width  = dst.GetSizes().m_width * dst.GetSizes().m_channel;
    DT_S32 height = dst.GetSizes().m_height;
    DT_F32 *dst_row = DT_NULL;

    if (src.GetElemType() == ElemType::U8)
    {
        const DT_U8 *src_row = DT_NULL;
        DT_U8 zp = static_cast<DT_U8>(zero_point);

#if defined(AURA_ENABLE_NEON)
        DT_S32 width_align16 = width & (-16);
        uint8x16_t vqu8_src;
        float32x4_t vqf32_src[4];

        uint8x16_t vqu8_zp;
        neon::vdup(vqu8_zp, zp);

        float32x4_t vqf32_scale;
        neon::vdup(vqf32_scale, scale);
#endif

        for (DT_S32 y = 0; y < height; y++)
        {
            src_row = src.Ptr<DT_U8>(y);
            dst_row = dst.Ptr<DT_F32>(y);
            DT_S32 x = 0;

#if defined(AURA_ENABLE_NEON)
            for (; x < width_align16; x += 16)
            {
                vqu8_src = neon::vload1q(src_row + x);

                uint16x8_t vqu16_src_l = neon::vmovl(neon::vgetlow(vqu8_src));
                uint16x8_t vqu16_src_h = neon::vmovl(neon::vgethigh(vqu8_src));
                int16x8_t vqs16_src_l = neon::vreinterpret(vqu16_src_l);
                int16x8_t vqs16_src_h = neon::vreinterpret(vqu16_src_h);

                uint16x8_t vqu16_zp_l = neon::vmovl(neon::vgetlow(vqu8_zp));
                uint16x8_t vqu16_zp_h = neon::vmovl(neon::vgethigh(vqu8_zp));
                int16x8_t vqs16_zp_l = neon::vreinterpret(vqu16_zp_l);
                int16x8_t vqs16_zp_h = neon::vreinterpret(vqu16_zp_h);

                vqs16_src_l = neon::vsub(vqs16_src_l, vqs16_zp_l);
                vqs16_src_h = neon::vsub(vqs16_src_h, vqs16_zp_h);

                vqf32_src[0] = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqs16_src_l)));
                vqf32_src[1] = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqs16_src_l)));
                vqf32_src[2] = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqs16_src_h)));
                vqf32_src[3] = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqs16_src_h)));

                vqf32_src[0] = neon::vmul(vqf32_src[0], vqf32_scale);
                vqf32_src[1] = neon::vmul(vqf32_src[1], vqf32_scale);
                vqf32_src[2] = neon::vmul(vqf32_src[2], vqf32_scale);
                vqf32_src[3] = neon::vmul(vqf32_src[3], vqf32_scale);

                neon::vstore(dst_row + x    , vqf32_src[0]);
                neon::vstore(dst_row + x + 4, vqf32_src[1]);
                neon::vstore(dst_row + x + 8, vqf32_src[2]);
                neon::vstore(dst_row + x + 12, vqf32_src[3]);
            }
#endif
            for (; x < width; x++)
            {
                dst_row[x] = static_cast<DT_F32>((src_row[x] - zp) * scale);
            }
        }
    }
    else if (src.GetElemType() == ElemType::S8)
    {
        const DT_S8 *src_row = DT_NULL;
        DT_S8 zp = static_cast<DT_S8>(zero_point);

#if defined(AURA_ENABLE_NEON)
        DT_S32 width_align16 = width & (-16);
        int8x16_t vqs8_src;
        float32x4_t vqf32_src[4];

        int8x16_t vqs8_zp;
        neon::vdup(vqs8_zp, zp);

        float32x4_t vqf32_scale;
        neon::vdup(vqf32_scale, scale);
#endif

        for (DT_S32 y = 0; y < height; y++)
        {
            src_row = src.Ptr<DT_S8>(y);
            dst_row = dst.Ptr<DT_F32>(y);
            DT_S32 x = 0;

#if defined(AURA_ENABLE_NEON)
            for (; x < width_align16; x += 16)
            {
                vqs8_src = neon::vload1q(src_row + x);

                int16x8_t vqs16_src_l = neon::vmovl(neon::vgetlow(vqs8_src));
                int16x8_t vqs16_src_h = neon::vmovl(neon::vgethigh(vqs8_src));

                int16x8_t vqs16_zp_l = neon::vmovl(neon::vgetlow(vqs8_zp));
                int16x8_t vqs16_zp_h = neon::vmovl(neon::vgethigh(vqs8_zp));

                vqs16_src_l = neon::vsub(vqs16_src_l, vqs16_zp_l);
                vqs16_src_h = neon::vsub(vqs16_src_h, vqs16_zp_h);

                vqf32_src[0] = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqs16_src_l)));
                vqf32_src[1] = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqs16_src_l)));
                vqf32_src[2] = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vqs16_src_h)));
                vqf32_src[3] = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vqs16_src_h)));

                vqf32_src[0] = neon::vmul(vqf32_src[0], vqf32_scale);
                vqf32_src[1] = neon::vmul(vqf32_src[1], vqf32_scale);
                vqf32_src[2] = neon::vmul(vqf32_src[2], vqf32_scale);
                vqf32_src[3] = neon::vmul(vqf32_src[3], vqf32_scale);

                neon::vstore(dst_row + x    , vqf32_src[0]);
                neon::vstore(dst_row + x + 4, vqf32_src[1]);
                neon::vstore(dst_row + x + 8, vqf32_src[2]);
                neon::vstore(dst_row + x + 12, vqf32_src[3]);
            }
#endif
            for (; x < width; x++)
            {
                dst_row[x] = static_cast<DT_F32>((src_row[x] - zp) * scale);
            }
        }
    }
    else if (src.GetElemType() == ElemType::U16)
    {
        const DT_U16 *src_row = DT_NULL;
        DT_U16 zp = zero_point;

#if defined(AURA_ENABLE_NEON)
        DT_S32 width_align8 = width & (-8);
        uint16x8_t vqu16_src;
        float32x4_t vqf32_src[2];
        int32x4_t vqs32_src[2];

        int32x4_t vqs32_zp;
        neon::vdup(vqs32_zp, zp);

        float32x4_t vqf32_scale;
        neon::vdup(vqf32_scale, scale);
#endif

        for (DT_S32 y = 0; y < height; y++)
        {
            src_row = src.Ptr<DT_U16>(y);
            dst_row = dst.Ptr<DT_F32>(y);
            DT_S32 x = 0;

#if defined(AURA_ENABLE_NEON)
            for (; x < width_align8; x += 8)
            {
                vqu16_src = neon::vload1q(src_row + x);

                vqs32_src[0] = neon::vreinterpret(neon::vmovl(neon::vgetlow(vqu16_src)));
                vqs32_src[1] = neon::vreinterpret(neon::vmovl(neon::vgethigh(vqu16_src)));
                vqs32_src[0] = neon::vsub(vqs32_src[0], vqs32_zp);
                vqs32_src[1] = neon::vsub(vqs32_src[1], vqs32_zp);

                vqf32_src[0] = neon::vcvt<DT_F32>(vqs32_src[0]);
                vqf32_src[1] = neon::vcvt<DT_F32>(vqs32_src[1]);

                vqf32_src[0] = neon::vmul(vqf32_src[0], vqf32_scale);
                vqf32_src[1] = neon::vmul(vqf32_src[1], vqf32_scale);

                neon::vstore(dst_row + x    , vqf32_src[0]);
                neon::vstore(dst_row + x + 4, vqf32_src[1]);
            }
#endif
            for (; x < width; x++)
            {
                dst_row[x] = static_cast<DT_F32>((src_row[x] - zp) * scale);
            }
        }
    }
    else if (src.GetElemType() == ElemType::S16)
    {
        const DT_S16 *src_row = DT_NULL;
        DT_S16 zp = zero_point;

#if defined(AURA_ENABLE_NEON)
        DT_S32 width_align8 = width & (-8);
        int16x8_t vqs16_src;
        float32x4_t vqf32_src[2];
        int32x4_t vqs32_src[2];

        int32x4_t vqs32_zp;
        neon::vdup(vqs32_zp, zp);

        float32x4_t vqf32_scale;
        neon::vdup(vqf32_scale, scale);
#endif

        for (DT_S32 y = 0; y < height; y++)
        {
            src_row = src.Ptr<DT_S16>(y);
            dst_row = dst.Ptr<DT_F32>(y);
            DT_S32 x = 0;

#if defined(AURA_ENABLE_NEON)
            for (; x < width_align8; x += 8)
            {
                vqs16_src = neon::vload1q(src_row + x);

                vqs32_src[0] = neon::vmovl(neon::vgetlow(vqs16_src));
                vqs32_src[1] = neon::vmovl(neon::vgethigh(vqs16_src));
                vqs32_src[0] = neon::vsub(vqs32_src[0], vqs32_zp);
                vqs32_src[1] = neon::vsub(vqs32_src[1], vqs32_zp);

                vqf32_src[0] = neon::vcvt<DT_F32>(vqs32_src[0]);
                vqf32_src[1] = neon::vcvt<DT_F32>(vqs32_src[1]);

                vqf32_src[0] = neon::vmul(vqf32_src[0], vqf32_scale);
                vqf32_src[1] = neon::vmul(vqf32_src[1], vqf32_scale);

                neon::vstore(dst_row + x    , vqf32_src[0]);
                neon::vstore(dst_row + x + 4, vqf32_src[1]);
            }
#endif
            for (; x < width; x++)
            {
                dst_row[x] = static_cast<DT_F32>((src_row[x] - zp) * scale);
            }
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "can not support this dst type quantation");
        return Status::ERROR;
    }

    return Status::OK;
}

std::vector<std::string> NNSplit(const std::string &src, DT_CHAR separator)
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