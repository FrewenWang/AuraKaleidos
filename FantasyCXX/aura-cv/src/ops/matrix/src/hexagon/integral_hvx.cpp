#include "aura/ops/matrix/integral.hpp"
#include "integral_impl.hpp"
#include "matrix_comm.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp, typename std::enable_if<(4 == sizeof(Tp))>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID IntegralPostProcess(Tp *addr, const HVX_VectorX1 &mvd32_in)
{
    HVX_Vector *addr_align = (HVX_Vector*)addr;
    *(addr_align) = mvd32_in.val[0];
}

template <typename Tp, typename std::enable_if<(4 == sizeof(Tp))>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID IntegralPostProcess(Tp *addr, const HVX_VectorX2 &mvd32_in)
{
    HVX_Vector *addr_align = (HVX_Vector*)addr;
    HVX_VectorPair wd32_uv_uv = Q6_W_vshuff_VVR(mvd32_in.val[1], mvd32_in.val[0], -4);
    *(addr_align)     = Q6_V_lo_W(wd32_uv_uv);
    *(addr_align + 1) = Q6_V_hi_W(wd32_uv_uv);
}

template <typename Tp, typename std::enable_if<(4 == sizeof(Tp))>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID IntegralPostProcess(Tp *addr, const HVX_VectorX4 &mvd32_in)
{
    HVX_Vector *addr_align = (HVX_Vector*)addr;
    HVX_VectorPair wd32_uv0, wd32_uv1, wd32_uv2, wd32_uv3;

    wd32_uv0 = Q6_W_vshuff_VVR(mvd32_in.val[1], mvd32_in.val[0], -4);
    wd32_uv1 = Q6_W_vshuff_VVR(mvd32_in.val[3], mvd32_in.val[2], -4);
    wd32_uv2 = Q6_W_vshuff_VVR(Q6_V_lo_W(wd32_uv1), Q6_V_lo_W(wd32_uv0), -8);
    wd32_uv3 = Q6_W_vshuff_VVR(Q6_V_hi_W(wd32_uv1), Q6_V_hi_W(wd32_uv0), -8);

    *(addr_align)     = Q6_V_lo_W(wd32_uv2);
    *(addr_align + 1) = Q6_V_hi_W(wd32_uv2);
    *(addr_align + 2) = Q6_V_lo_W(wd32_uv3);
    *(addr_align + 3) = Q6_V_hi_W(wd32_uv3);
}

// using St = MI_U8
template <typename St, typename std::enable_if<std::is_same<St, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID IntegralHCore(HVX_Vector &vu8_src, HVX_Vector &vu32_dst0, HVX_Vector &vu32_dst1, HVX_Vector &vu32_dst2,
                                         HVX_Vector &vu32_dst3, HVX_Vector &vd32_rdelta)
{
    MI_U32 c1c1c1c1 = 0x01010101;
    HVX_Vector vu8_zeros = Q6_V_vzero();
    HVX_Vector vu8_mask = Q6_V_vsplat_R(0x00FF00FF);

    HVX_Vector vu16_sum = Q6_Vh_vdmpy_VubRb(vu8_src, c1c1c1c1);

    HVX_Vector vu16_sum2 = Q6_V_vlalign_VVI(vu16_sum, vu8_zeros, 2);
    vu16_sum = Q6_Vh_vadd_VhVh(vu16_sum, vu16_sum2);

    HVX_Vector vu16_sum4 = Q6_V_vlalign_VVI(vu16_sum, vu8_zeros, 4);
    vu16_sum = Q6_Vh_vadd_VhVh(vu16_sum, vu16_sum4);

    HVX_Vector vu16_sum8 = Q6_V_vlalign_VVR(vu16_sum, vu8_zeros, 8);
    vu16_sum = Q6_Vh_vadd_VhVh(vu16_sum, vu16_sum8);

    HVX_Vector vu16_sum16 = Q6_V_vlalign_VVR(vu16_sum, vu8_zeros, 16);
    vu16_sum = Q6_Vh_vadd_VhVh(vu16_sum, vu16_sum16);

    HVX_Vector vu16_sum32 = Q6_V_vlalign_VVR(vu16_sum, vu8_zeros, 32);
    vu16_sum = Q6_Vh_vadd_VhVh(vu16_sum, vu16_sum32);

    HVX_Vector vu16_sum64 = Q6_V_vlalign_VVR(vu16_sum, vu8_zeros, 64);

    HVX_Vector vu16_odd = Q6_Vh_vadd_VhVh(vu16_sum, vu16_sum64);
    HVX_Vector vu16_even = Q6_V_vand_VV(vu8_src, vu8_mask);

    HVX_Vector vu16_odd2 = Q6_V_vlalign_VVR(vu16_odd, vu8_zeros, 2);
    HVX_Vector vu16_even2 = Q6_Vh_vadd_VhVh(vu16_odd2, vu16_even);

    HVX_VectorPair vu16_pair = Q6_W_vshuff_VVR(vu16_odd, vu16_even2, -2);
    HVX_Vector vu32_rep_last = Q6_V_vrdelta_VV(vu32_dst3, vd32_rdelta);

    HVX_VectorPair vu32_dst01 = Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(vu16_pair));
    HVX_VectorPair vu32_dst23 = Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(vu16_pair));

    vu32_dst0 = Q6_Vw_vadd_VwVw(vu32_rep_last, Q6_V_lo_W(vu32_dst01));
    vu32_dst1 = Q6_Vw_vadd_VwVw(vu32_rep_last, Q6_V_hi_W(vu32_dst01));
    vu32_dst2 = Q6_Vw_vadd_VwVw(vu32_rep_last, Q6_V_lo_W(vu32_dst23));
    vu32_dst3 = Q6_Vw_vadd_VwVw(vu32_rep_last, Q6_V_hi_W(vu32_dst23));
}

// using St = MI_S8
template <typename St, typename std::enable_if<std::is_same<St, MI_S8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID IntegralHCore(HVX_Vector &vs8_src, HVX_Vector &vs32_dst0, HVX_Vector &vs32_dst1, HVX_Vector &vs32_dst2,
                                         HVX_Vector &vs32_dst3, HVX_Vector &vd32_rdelta)
{
    HVX_Vector vu8_zeros = Q6_V_vzero();

    HVX_VectorPair vs16_exp = Q6_Wh_vsxt_Vb(vs8_src);
    HVX_Vector vs16_sum = Q6_Vh_vadd_VhVh(Q6_V_lo_W(vs16_exp), Q6_V_hi_W(vs16_exp));

    HVX_Vector vs16_sum2 = Q6_V_vlalign_VVI(vs16_sum, vu8_zeros, 2);
    vs16_sum = Q6_Vh_vadd_VhVh(vs16_sum, vs16_sum2);

    HVX_Vector vs16_sum4 = Q6_V_vlalign_VVI(vs16_sum, vu8_zeros, 4);
    vs16_sum = Q6_Vh_vadd_VhVh(vs16_sum, vs16_sum4);

    HVX_Vector vs16_sum8 = Q6_V_vlalign_VVR(vs16_sum, vu8_zeros, 8);
    vs16_sum = Q6_Vh_vadd_VhVh(vs16_sum, vs16_sum8);

    HVX_Vector vs16_sum16 = Q6_V_vlalign_VVR(vs16_sum, vu8_zeros, 16);
    vs16_sum = Q6_Vh_vadd_VhVh(vs16_sum, vs16_sum16);

    HVX_Vector vs16_sum32 = Q6_V_vlalign_VVR(vs16_sum, vu8_zeros, 32);
    vs16_sum = Q6_Vh_vadd_VhVh(vs16_sum, vs16_sum32);

    HVX_Vector vs16_sum64 = Q6_V_vlalign_VVR(vs16_sum, vu8_zeros, 64);

    HVX_Vector vs16_odd = Q6_Vh_vadd_VhVh(vs16_sum, vs16_sum64);
    HVX_Vector vs16_even = Q6_V_lo_W(vs16_exp);

    HVX_Vector vs16_odd2 = Q6_V_vlalign_VVR(vs16_odd, vu8_zeros, 2);
    HVX_Vector vs16_even2 = Q6_Vh_vadd_VhVh(vs16_odd2, vs16_even);

    HVX_VectorPair vs16_pair = Q6_W_vshuff_VVR(vs16_odd, vs16_even2, -2);
    HVX_Vector vs32_rep_last = Q6_V_vrdelta_VV(vs32_dst3, vd32_rdelta);

    HVX_VectorPair vs32_dst01 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(vs16_pair));
    HVX_VectorPair vs32_dst23 = Q6_Ww_vunpack_Vh(Q6_V_hi_W(vs16_pair));

    vs32_dst0 = Q6_Vw_vadd_VwVw(vs32_rep_last, Q6_V_lo_W(vs32_dst01));
    vs32_dst1 = Q6_Vw_vadd_VwVw(vs32_rep_last, Q6_V_hi_W(vs32_dst01));
    vs32_dst2 = Q6_Vw_vadd_VwVw(vs32_rep_last, Q6_V_lo_W(vs32_dst23));
    vs32_dst3 = Q6_Vw_vadd_VwVw(vs32_rep_last, Q6_V_hi_W(vs32_dst23));
}

//if dst address and stride is AURA_HVLEN align, using vmem instead of vmemu
template <typename St, MI_S32 C, MI_S32 ALIGN, typename std::enable_if<1 == ALIGN>::type* = MI_NULL>
AURA_NO_INLINE AURA_VOID IntegralRowH(const St *restrict src_c, MI_S32 *restrict dst_c, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    MI_S32 width_align128 = width & (-AURA_HVLEN);
    MI_S32 rest = width - width_align128;

    HVX_Vector vd32_rdelta = *(HVX_Vector *)(vrdelta_replicate_last_d32);
    MVType mvu8_src, mvd32_dst0, mvd32_dst1, mvd32_dst2, mvd32_dst3;

    #pragma unroll(C)
    for (MI_S32 ch = 0; ch < C; ch++)
    {
        mvd32_dst3.val[ch] = Q6_V_vzero();
    }

    for (MI_S32 x = 0; x < width_align128; x += AURA_HVLEN)
    {
        vload(src_c + x * C, mvu8_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            IntegralHCore<St>(mvu8_src.val[ch], mvd32_dst0.val[ch], mvd32_dst1.val[ch], mvd32_dst2.val[ch], mvd32_dst3.val[ch], vd32_rdelta);
        }

        IntegralPostProcess(dst_c + (x * C), mvd32_dst0);
        IntegralPostProcess(dst_c + (x * C) + (AURA_HVLEN * 1 * C) / sizeof(MI_S32), mvd32_dst1);
        IntegralPostProcess(dst_c + (x * C) + (AURA_HVLEN * 2 * C) / sizeof(MI_S32), mvd32_dst2);
        IntegralPostProcess(dst_c + (x * C) + (AURA_HVLEN * 3 * C) / sizeof(MI_S32), mvd32_dst3);
    }

    if (rest > 0)
    {
        MI_S32 shift_cnt = AURA_HVLEN - rest;
        MI_S32 shift_cnt4 = AURA_HVLEN - (shift_cnt & 31) * sizeof(MI_U32);
        if (rest <= 32)
        {
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvd32_dst3.val[ch] = Q6_V_vror_VR(mvd32_dst0.val[ch], shift_cnt4);
            }
        }
        else if (rest <= 64)
        {
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvd32_dst3.val[ch] = Q6_V_vror_VR(mvd32_dst1.val[ch], shift_cnt4);
            }
        }
        else if (rest <= 96)
        {
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvd32_dst3.val[ch] = Q6_V_vror_VR(mvd32_dst2.val[ch], shift_cnt4);
            }
        }
        else
        {
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvd32_dst3.val[ch] = Q6_V_vror_VR(mvd32_dst3.val[ch], shift_cnt4);
            }
        }

        vload(src_c + (width - AURA_HVLEN) * C, mvu8_src);
        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            IntegralHCore<St>(mvu8_src.val[ch], mvd32_dst0.val[ch], mvd32_dst1.val[ch], mvd32_dst2.val[ch], mvd32_dst3.val[ch], vd32_rdelta);
        }

        vstore(dst_c + (width - AURA_HVLEN) * C, mvd32_dst0);
        vstore(dst_c + (width - AURA_HVLEN) * C + (AURA_HVLEN * 1 * C) / sizeof(MI_S32), mvd32_dst1);
        vstore(dst_c + (width - AURA_HVLEN) * C + (AURA_HVLEN * 2 * C) / sizeof(MI_S32), mvd32_dst2);
        vstore(dst_c + (width - AURA_HVLEN) * C + (AURA_HVLEN * 3 * C) / sizeof(MI_S32), mvd32_dst3);
    }
}

template <typename St, MI_S32 C, MI_S32 ALIGN, typename std::enable_if<0 == ALIGN>::type* = MI_NULL>
AURA_NO_INLINE AURA_VOID IntegralRowH(const St *restrict src_c, MI_S32 *restrict dst_c, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    MI_S32 width_align128 = width & (-AURA_HVLEN);
    MI_S32 rest = width - width_align128;

    HVX_Vector vd32_rdelta = *(HVX_Vector *)(vrdelta_replicate_last_d32);
    MVType mvu8_src, mvd32_dst0, mvd32_dst1, mvd32_dst2, mvd32_dst3;

    #pragma unroll(C)
    for (MI_S32 ch = 0; ch < C; ch++)
    {
        mvd32_dst3.val[ch] = Q6_V_vzero();
    }

    for (MI_S32 x = 0; x < width_align128; x += AURA_HVLEN)
    {
        vload(src_c + x * C, mvu8_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            IntegralHCore<St>(mvu8_src.val[ch], mvd32_dst0.val[ch], mvd32_dst1.val[ch], mvd32_dst2.val[ch], mvd32_dst3.val[ch], vd32_rdelta);

        }

        vstore(dst_c + x * C, mvd32_dst0);
        vstore(dst_c + x * C + (AURA_HVLEN * 1 * C) / sizeof(MI_S32), mvd32_dst1);
        vstore(dst_c + x * C + (AURA_HVLEN * 2 * C) / sizeof(MI_S32), mvd32_dst2);
        vstore(dst_c + x * C + (AURA_HVLEN * 3 * C) / sizeof(MI_S32), mvd32_dst3);
    }

    if (rest > 0)
    {
        MI_S32 shift_cnt = AURA_HVLEN - rest;
        MI_S32 shift_cnt4 = AURA_HVLEN - (shift_cnt & 31) * sizeof(MI_U32);
        if (rest <= 32)
        {
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvd32_dst3.val[ch] = Q6_V_vror_VR(mvd32_dst0.val[ch], shift_cnt4);
            }
        }
        else if (rest <= 64)
        {
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvd32_dst3.val[ch] = Q6_V_vror_VR(mvd32_dst1.val[ch], shift_cnt4);
            }
        }
        else if (rest <= 96)
        {
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvd32_dst3.val[ch] = Q6_V_vror_VR(mvd32_dst2.val[ch], shift_cnt4);
            }
        }
        else
        {
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvd32_dst3.val[ch] = Q6_V_vror_VR(mvd32_dst3.val[ch], shift_cnt4);
            }
        }

        vload(src_c + (width - AURA_HVLEN) * C, mvu8_src);
        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            IntegralHCore<St>(mvu8_src.val[ch], mvd32_dst0.val[ch], mvd32_dst1.val[ch], mvd32_dst2.val[ch], mvd32_dst3.val[ch], vd32_rdelta);
        }

        vstore(dst_c + (width - AURA_HVLEN) * C, mvd32_dst0);
        vstore(dst_c + (width - AURA_HVLEN) * C + (AURA_HVLEN * 1 * C) / sizeof(MI_S32), mvd32_dst1);
        vstore(dst_c + (width - AURA_HVLEN) * C + (AURA_HVLEN * 2 * C) / sizeof(MI_S32), mvd32_dst2);
        vstore(dst_c + (width - AURA_HVLEN) * C + (AURA_HVLEN * 3 * C) / sizeof(MI_S32), mvd32_dst3);
    }
}

template <MI_S32 C, MI_S32 ALIGN, typename std::enable_if<(1 == ALIGN)>::type* = MI_NULL>
static MI_S32 IntegralRow4V(MI_U8 *dst_prev, MI_U8 *dst, MI_S32 ostride, MI_S32 width)
{
    width = width * sizeof(MI_S32) * C;
    ostride >>= 7;

    HVX_Vector *dst_p0 = (HVX_Vector*)dst_prev;
    HVX_Vector vs32_p0, vs32_c, vs32_n0, vs32_n1, vs32_n2;

    for (MI_S32 x = 0; x < width; x += AURA_HVLEN)
    {
        vs32_p0 = *(dst_p0++);

        HVX_Vector *ptr_dst_cur = (HVX_Vector*)(dst + x);

        vs32_c = *(ptr_dst_cur);
        vs32_c = Q6_Vw_vadd_VwVw(vs32_c, vs32_p0);
        *(ptr_dst_cur) = vs32_c;

        vs32_n0 = *(ptr_dst_cur + ostride);
        vs32_n0 = Q6_Vw_vadd_VwVw(vs32_n0, vs32_c);
        *(ptr_dst_cur + ostride) = vs32_n0;

        vs32_n1 = *(ptr_dst_cur + ostride * 2);
        vs32_n1 = Q6_Vw_vadd_VwVw(vs32_n1, vs32_n0);
        *(ptr_dst_cur + ostride * 2) = vs32_n1;

        vs32_n2 = *(ptr_dst_cur + ostride * 3);
        vs32_n2 = Q6_Vw_vadd_VwVw(vs32_n2, vs32_n1);
        *(ptr_dst_cur + ostride * 3) = vs32_n2;
    }

    return 0;
}

template <MI_S32 C, MI_S32 ALIGN, typename std::enable_if<(0 == ALIGN)>::type* = MI_NULL>
static MI_S32 IntegralRow4V(MI_U8 *dst_prev, MI_U8 *dst, MI_S32 ostride, MI_S32 width)
{
    width = width * sizeof(MI_S32) * C;
    MI_S32 width_align128 = width & (-AURA_HVLEN);
    HVX_VectorPred q = Q6_Q_vsetq_R(AURA_HVLEN - width + width_align128);

    MI_U8 *dst_p0 = dst_prev;
    HVX_Vector vs32_p0, vs32_c, vs32_n0, vs32_n1, vs32_n2;

    for (MI_S32 x = 0; x < width_align128; x += AURA_HVLEN)
    {
        vload(dst_p0 + x, vs32_p0);
        MI_U8 *ptr_dst_cur = dst + x;

        vload(ptr_dst_cur, vs32_c);
        vs32_c = Q6_Vw_vadd_VwVw(vs32_c, vs32_p0);
        vstore(ptr_dst_cur, vs32_c);

        vload(ptr_dst_cur + ostride, vs32_n0);
        vs32_n0 = Q6_Vw_vadd_VwVw(vs32_n0, vs32_c);
        vstore(ptr_dst_cur + ostride, vs32_n0);

        vload(ptr_dst_cur + ostride * 2, vs32_n1);
        vs32_n1 = Q6_Vw_vadd_VwVw(vs32_n1, vs32_n0);
        vstore(ptr_dst_cur + ostride * 2, vs32_n1);

        vload(ptr_dst_cur + ostride * 3, vs32_n2);
        vs32_n2 = Q6_Vw_vadd_VwVw(vs32_n2, vs32_n1);
        vstore(ptr_dst_cur + ostride * 3, vs32_n2);
    }

    if (width != width_align128)
    {
        vload(dst_p0 + width - AURA_HVLEN, vs32_p0);
        MI_U8 *ptr_dst_cur = dst + width - AURA_HVLEN;

        vload(ptr_dst_cur, vs32_c);
        vs32_p0 = Q6_V_vmux_QVV(q, Q6_V_vzero(), vs32_p0);
        vs32_c = Q6_Vw_vadd_VwVw(vs32_c, vs32_p0);
        vstore(ptr_dst_cur, vs32_c);

        vload(ptr_dst_cur + ostride, vs32_n0);
        vs32_c = Q6_V_vmux_QVV(q, Q6_V_vzero(), vs32_c);
        vs32_n0 = Q6_Vw_vadd_VwVw(vs32_n0, vs32_c);
        vstore(ptr_dst_cur + ostride, vs32_n0);

        vload(ptr_dst_cur + ostride * 2, vs32_n1);
        vs32_n0 = Q6_V_vmux_QVV(q, Q6_V_vzero(), vs32_n0);
        vs32_n1 = Q6_Vw_vadd_VwVw(vs32_n1, vs32_n0);
        vstore(ptr_dst_cur + ostride * 2, vs32_n1);

        vload(ptr_dst_cur + ostride * 3, vs32_n2);
        vs32_n1 = Q6_V_vmux_QVV(q, Q6_V_vzero(), vs32_n1);
        vs32_n2 = Q6_Vw_vadd_VwVw(vs32_n2, vs32_n1);
        vstore(ptr_dst_cur + ostride * 3, vs32_n2);
    }

    return 0;
}

template <MI_S32 C, MI_S32 ALIGN, typename std::enable_if<(1 == ALIGN)>::type* = MI_NULL>
static MI_S32 IntegralRow1V(MI_U8 *dst_prev, MI_U8 *dst, MI_S32 ostride, MI_S32 width, MI_S32 height)
{
    width = width * sizeof(MI_S32) * C;
    ostride >>= 7;

    HVX_Vector *dst_p0 = (HVX_Vector*)dst_prev;
    HVX_Vector vs32_p0, vs32_c;

    for (MI_S32 x = 0; x < width; x += AURA_HVLEN)
    {
        vs32_p0 = *(dst_p0++);
        HVX_Vector *ptr_dst_cur = (HVX_Vector*)(dst + x);

        for (MI_S32 y = 0; y < height; y++)
        {
            vs32_c = *(ptr_dst_cur);

            vs32_p0 = Q6_Vw_vadd_VwVw(vs32_c, vs32_p0);

            *(ptr_dst_cur) = vs32_p0;
            ptr_dst_cur += ostride;
        }
    }

    return 0;
}

template <MI_S32 C, MI_S32 ALIGN, typename std::enable_if<(0 == ALIGN)>::type* = MI_NULL>
static MI_S32 IntegralRow1V(MI_U8 *dst_prev, MI_U8 *dst, MI_S32 ostride, MI_S32 width, MI_S32 height)
{
    width = width * sizeof(MI_S32) * C;
    MI_S32 width_align128 = width & (-AURA_HVLEN);
    HVX_VectorPred q = Q6_Q_vsetq_R(AURA_HVLEN - width + width_align128);

    MI_U8 *dst_p0 = dst_prev;
    HVX_Vector vs32_p0, vs32_c;

    for (MI_S32 x = 0; x < width_align128; x += AURA_HVLEN)
    {
        vload(dst_p0 + x, vs32_p0);

        MI_U8 *ptr_dst_cur = dst + x;

        for (MI_S32 y = 0; y < height; y++)
        {
            vload(ptr_dst_cur, vs32_c);

            vs32_p0 = Q6_Vw_vadd_VwVw(vs32_c, vs32_p0);

            vstore(ptr_dst_cur, vs32_p0);
            ptr_dst_cur += ostride;
        }
    }

    if (width != width_align128)
    {
        vload(dst_prev + width - AURA_HVLEN, vs32_p0);
        MI_U8 *ptr_dst_cur = dst + width - AURA_HVLEN;

        for (MI_S32 y = 0; y < height; y++)
        {
            vload(ptr_dst_cur, vs32_c);
            vs32_p0 = Q6_V_vmux_QVV(q, Q6_V_vzero(), vs32_p0);
            vs32_p0 = Q6_Vw_vadd_VwVw(vs32_c, vs32_p0);

            vstore(ptr_dst_cur, vs32_p0);
            ptr_dst_cur += ostride;
        }
    }

    return 0;
}

template <typename St, typename Dt, MI_S32 C, MI_S32 ALIGN>
static Status IntegralHvxImpl(const Mat &src, Mat &dst)
{
    static_assert(std::is_same<Dt, MI_S32>::value || std::is_same<Dt, MI_U32>::value,
                  "IntegralHvxImpl only support MI_S32 and MI_U32");

    MI_S32 width   = src.GetSizes().m_width;
    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;
    MI_S32 ostride = dst.GetStrides().m_width;
    MI_S32 height_align4 = (height - 1) & (-4);

    const St *src_c = src.Ptr<St>(0);
    MI_S32 *dst_c   = dst.Ptr<MI_S32>(0); // set dst pointer as MI_S32*, because addition logic of MI_S32 and MI_U32 is same

    MI_U64 L2fetch_param1 = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 1, 0); //pre_fetch 1 row
    MI_U64 L2fetch_param4 = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 4, 0); //pre_fetch 4 rows

    L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<St>(0)), L2fetch_param1);
    L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<St>(1)), L2fetch_param4);

    IntegralRowH<St, C, ALIGN>(src_c, dst_c, width);

    MI_S32 y = 1;
    for (; y <= height_align4; y += 4)
    {
        if (y + 4 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<St>(y + 4)), L2fetch_param4);
        }

        MI_S32 *dst_p1 = dst.Ptr<MI_S32>(y - 1);
        dst_c = dst.Ptr<MI_S32>(y);
        MI_S32 *dst_n1 = dst.Ptr<MI_S32>(y + 1);
        MI_S32 *dst_n2 = dst.Ptr<MI_S32>(y + 2);
        MI_S32 *dst_n3 = dst.Ptr<MI_S32>(y + 3);

        src_c = src.Ptr<St>(y);
        const St *src_n1 = src.Ptr<St>(y + 1);
        const St *src_n2 = src.Ptr<St>(y + 2);
        const St *src_n3 = src.Ptr<St>(y + 3);
        IntegralRowH<St, C, ALIGN>(src_c, dst_c, width);
        IntegralRowH<St, C, ALIGN>(src_n1, dst_n1, width);
        IntegralRowH<St, C, ALIGN>(src_n2, dst_n2, width);
        IntegralRowH<St, C, ALIGN>(src_n3, dst_n3, width);

        IntegralRow4V<C, ALIGN>(reinterpret_cast<MI_U8*>(&dst_p1[0]), reinterpret_cast<MI_U8*>(&dst_c[0]), ostride, width);
    }

    MI_S32 rest = height - y;
    if (rest > 0)
    {
        for (MI_S32 i = 0; i < rest; i++)
        {
            src_c = src.Ptr<St>(y + i);
            dst_c = dst.Ptr<MI_S32>(y + i);
            IntegralRowH<St, C, ALIGN>(src_c, dst_c, width);
        }
        MI_S32 *dst_p1 = dst.Ptr<MI_S32>(y - 1);
        dst_c = dst.Ptr<MI_S32>(y);
        IntegralRow1V<C, ALIGN>(reinterpret_cast<MI_U8*>(&dst_p1[0]), reinterpret_cast<MI_U8*>(&dst_c[0]), ostride, width, rest);
    }

    return Status::OK;
}

template<typename St, typename Dt, MI_S32 C>
static Status IntegralHvxHelper(const Mat &src, Mat &dst)
{
    Status ret     = Status::ERROR;
    MI_S32 ostride = dst.GetStrides().m_width;
    MI_S32 *dst_c  = dst.Ptr<MI_S32>(0);
    MI_BOOL align  = ((0 == (ostride & (AURA_HVLEN - 1))) && (0 == (reinterpret_cast<MI_U32>(dst_c) & (AURA_HVLEN - 1))));

    if (align)
    {
        ret = IntegralHvxImpl<St, Dt, C, 1>(src, dst);
    }
    else
    {
        ret = IntegralHvxImpl<St, Dt, C, 0>(src, dst);
    }

    return ret;
}

template<typename St, typename Dt>
static Status IntegralHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret     = Status::ERROR;
    MI_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = IntegralHvxHelper<St, Dt, 1>(src, dst);
            break;
        }

        case 2:
        {
            ret = IntegralHvxHelper<St, Dt, 2>(src, dst);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status IntegralHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType()))
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U32):
        {
            ret = IntegralHvxHelper<MI_U8, MI_U32>(ctx, src, dst);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S32):
        {
            ret = IntegralHvxHelper<MI_S8, MI_S32>(ctx, src, dst);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

IntegralHvx::IntegralHvx(Context *ctx, const OpTarget &target) : IntegralImpl(ctx, target)
{}

Status IntegralHvx::SetArgs(const Array *src, Array *dst, Array *dst_sq)
{
    if (IntegralImpl::SetArgs(src, dst, dst_sq) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "IntegralImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    // dst must be non-null and mat type
    if (MI_NULL == dst)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst is nullptr");
        return Status::ERROR;
    }

    if (dst && (!dst->IsValid() || dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst is invalid or not mat type");
        return Status::ERROR;
    }

    // dst_sq must be null or invalid
    if (dst_sq && dst_sq->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "only support Normal mode, not Sequare mode, dst_sq must be nullptr or invalid");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 2)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2");
        return Status::ERROR;
    }

    return Status::OK;
}

Status IntegralHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    Status ret = IntegralHvxHelper(m_ctx, *src, *dst);

    AURA_RETURN(m_ctx, ret);
}

std::string IntegralHvx::ToString() const
{
    return IntegralImpl::ToString();
}

Status IntegralRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    Mat dst_sq;

    IntegralInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, dst_sq);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    Mat *dst0 = dst.IsValid()    ? &dst    : MI_NULL;
    Mat *dst1 = dst_sq.IsValid() ? &dst_sq : MI_NULL;

    Integral integral(ctx, OpTarget::Hvx());

    return OpCall(ctx, integral, &src, dst0, dst1);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_MATRIX_PACKAGE_NAME, AURA_OPS_MATRIX_INTEGRAL_OP_NAME, IntegralRpc);

} // namespace aura