#include "xtensa/gaussian_vdsp_impl.hpp"

namespace aura
{
namespace xtensa
{

template <typename Kt, typename std::enable_if<(std::is_same<Kt, DT_S8>::value) || (std::is_same<Kt, DT_U8>::value)>::type* = DT_NULL>
AURA_ALWAYS_INLINE xb_int32pr GetKernel(Kt *kernel)
{
    return IVP_EXTRPRN_2X32((DT_U32)(0 << 24) + (DT_U32)(kernel[2] << 16) + (DT_U32)(kernel[1] << 8) + (DT_U32)(kernel[0] & 0xff), 0);
}

AURA_ALWAYS_INLINE DT_VOID Gaussian3x3HCore(xb_vec2Nx8U &vqu8_src_x0, xb_vec2Nx8U &vqu8_src_x1, xb_int32pr vs32_kernel,
                                            xb_vecNx16U &vqu16_sum_lo, xb_vecNx16U &vqu16_sum_hi)
{
    xb_vec2Nx32w wqs32_sum = IVP_MULUU4T2N8XR8(vqu8_src_x1, vqu8_src_x0, vs32_kernel);
    vqu16_sum_lo = IVP_MOVNX16U_FROMNX16(IVP_CVT16U2NX32WL(wqs32_sum));
    vqu16_sum_hi = IVP_MOVNX16U_FROMNX16(IVP_CVT16U2NX32WH(wqs32_sum));
}

AURA_ALWAYS_INLINE xb_vec2Nx8U Gaussian3x3VCore(xb_vecNx16U &vqu16_sum_p_lo, xb_vecNx16U &vqu16_sum_p_hi,
                                                xb_vecNx16U &vqu16_sum_c_lo, xb_vecNx16U &vqu16_sum_c_hi,
                                                xb_vecNx16U &vqu16_sum_n_lo, xb_vecNx16U &vqu16_sum_n_hi,
                                                DT_U8 *kernel)
{
    xb_vecNx64w wqs64_sum_lo = IVP_MULUUNX16U((DT_U16)kernel[0], vqu16_sum_p_lo);
    xb_vecNx64w wqs64_sum_hi = IVP_MULUUNX16U((DT_U16)kernel[0], vqu16_sum_p_hi);

    IVP_MULUUANX16U(wqs64_sum_lo, (DT_U16)kernel[1], vqu16_sum_c_lo);
    IVP_MULUUANX16U(wqs64_sum_hi, (DT_U16)kernel[1], vqu16_sum_c_hi);

    IVP_MULUUANX16U(wqs64_sum_lo, (DT_U16)kernel[2], vqu16_sum_n_lo);
    IVP_MULUUANX16U(wqs64_sum_hi, (DT_U16)kernel[2], vqu16_sum_n_hi);

    xb_vecNx16 vqs16_dst_lo = IVP_MINNX16(IVP_MAXNX16(IVP_PACKVRNX64W(wqs64_sum_lo, 16), 0), 255);
    xb_vecNx16 vqs16_dst_hi = IVP_MINNX16(IVP_MAXNX16(IVP_PACKVRNX64W(wqs64_sum_hi, 16), 0), 255);

    return IVP_SEL2NX8UI(IVP_MOV2NX8U_FROMNX16(vqs16_dst_hi), IVP_MOV2NX8U_FROMNX16(vqs16_dst_lo), IVP_SELI_8B_DEINTERLEAVE_1_EVEN);
}

template <typename Tp, typename Kt, DT_S32 C>
DT_S32 Gaussian3x3VdspImpl(const xvTile *src, xvTile *dst, Kt *kernel)
{
    using VqType  = typename xtensa::QVector<Tp>::VType;
    using MVqType = typename xtensa::MQVector<Tp, C>::MVType;

    using VKt        = typename xtensa::GaussianTraits<Tp>::KernelType;
    using SumType    = typename xtensa::GaussianTraits<Tp>::SumType;
    using MVqSumType = typename xtensa::MQVector<SumType, C>::MVType;

    constexpr DT_S32 ELEM_COUNTS = static_cast<DT_S32>(sizeof(MVqType) / C / sizeof(Tp));
    constexpr DT_S32 ELEM_BYTES  = ELEM_COUNTS * C * sizeof(Tp);
    valign va_store = IVP_ZALIGN();

    DT_S32 src_x0_len = src->pitch * sizeof(Tp);
    DT_S32 src_x1_len = src_x0_len - ELEM_BYTES;
    DT_S32 dst_len = dst->pitch * sizeof(Tp);

    Tp *src_row = (Tp*)(src->pData) - src->pitch - C;
    Tp *dst_row = (Tp*)(dst->pData);

    VqType *__restrict vq_src, *__restrict vq_dst;
    MVqType mvq_src[2], mvq_dst;
    MVqSumType mvq_sum_p[2], mvq_sum_c[2], mvq_sum_n[2];

    VKt kdata = GetKernel<Tp>(kernel);

    for (DT_S32 x = 0; x < dst->width; x += ELEM_COUNTS)
    {
        // row 0
        {
            vq_src = (VqType*)(src_row);
            vload(vq_src, mvq_src[0], mvq_src[1], src_x0_len, src_x1_len);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Gaussian3x3HCore(mvq_src[0].val[ch], mvq_src[1].val[ch], kdata, mvq_sum_p[0].val[ch], mvq_sum_p[1].val[ch]);
            }
        }

        // row 1
        {
            vq_src = (VqType*)(src_row + src->pitch);
            vload(vq_src, mvq_src[0], mvq_src[1], src_x0_len, src_x1_len);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Gaussian3x3HCore(mvq_src[0].val[ch], mvq_src[1].val[ch], kdata, mvq_sum_c[0].val[ch], mvq_sum_c[1].val[ch]);
            }
        }

        // main(2 ~ height)
        for (DT_S32 y = 0; y < dst->height; y++)
        {
            vq_src = (VqType*)(src_row + src->pitch * (y + 2));
            vload(vq_src, mvq_src[0], mvq_src[1], src_x0_len, src_x1_len);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Gaussian3x3HCore(mvq_src[0].val[ch], mvq_src[1].val[ch], kdata, mvq_sum_n[0].val[ch], mvq_sum_n[1].val[ch]);
                mvq_dst.val[ch] = Gaussian3x3VCore(mvq_sum_p[0].val[ch], mvq_sum_p[1].val[ch],
                                                   mvq_sum_c[0].val[ch], mvq_sum_c[1].val[ch],
                                                   mvq_sum_n[0].val[ch], mvq_sum_n[1].val[ch],
                                                   kernel);
            }

            vq_dst = (VqType*)(dst_row + dst->pitch * y);
            vstore(vq_dst, va_store, mvq_dst, dst_len);
            vflush(va_store, vq_dst);

            for (DT_S32 ch = 0; ch < C; ++ch)
            {
                mvq_sum_p[0].val[ch] = mvq_sum_c[0].val[ch];
                mvq_sum_p[1].val[ch] = mvq_sum_c[1].val[ch];
                mvq_sum_c[0].val[ch] = mvq_sum_n[0].val[ch];
                mvq_sum_c[1].val[ch] = mvq_sum_n[1].val[ch];
            }
        }

        src_row += ELEM_COUNTS * C;
        dst_row += ELEM_COUNTS * C;

        src_x0_len -= ELEM_BYTES;
        src_x1_len -= ELEM_BYTES;
        dst_len -= ELEM_BYTES;
    }

    return AURA_XTENSA_OK;
}

template <typename Tp, typename Kt = Tp>
DT_S32 Gaussian3x3VdspHelper(const xvTile *src, xvTile *dst, Kt *kernel, DT_S32 channel)
{
    DT_S32 ret = AURA_XTENSA_ERROR;

    switch (channel)
    {
        case 1:
        {
            ret = Gaussian3x3VdspImpl<Tp, Kt, 1>(src, dst, kernel);
            break;
        }

        default:
        {
            AURA_XTENSA_LOG("Unsupported channel");
            return AURA_XTENSA_ERROR;
        }
    }

    if (ret != AURA_XTENSA_OK)
    {
        AURA_XTENSA_LOG("Gaussian3x3VdspImpl failed!");
    }

    return ret;
}

DT_S32 Gaussian3x3Vdsp(const xvTile *src, xvTile *dst, DT_VOID *kernel, ElemType elem_type, DT_S32 channel)
{
    DT_S32 ret = AURA_XTENSA_ERROR;

    switch (elem_type)
    {
        case ElemType::U8:
        {
            DT_U8 *kdata = reinterpret_cast<DT_U8*>(kernel);
            ret = Gaussian3x3VdspHelper<DT_U8>(src, dst, kdata, channel);
            break;
        }

        default:
        {
            AURA_XTENSA_LOG("Unsupported elem_type");
            return AURA_XTENSA_ERROR;
        }
    }

    if (ret != AURA_XTENSA_OK)
    {
        AURA_XTENSA_LOG("Gaussian3x3VdspHelper failed!");
        return AURA_XTENSA_ERROR;
    }

    return AURA_XTENSA_OK;
}

} // namespace xtensa
} // namespace aura
