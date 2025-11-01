#include "make_border_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/core.h"

namespace aura
{

template <typename Tp>
struct BoxFilterTraits
{
    using RowSumType    = typename Promote<Tp>::Type;
    using KernelSumType = typename std::conditional<sizeof(Tp) == 1, typename Promote<RowSumType>::Type,
                                                                     typename Promote<Tp>::Type>::type;
};

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKAddTwoRowSumCore(HVX_Vector &vu8_src0, HVX_Vector &vu8_src1, HVX_Vector &vu16_dst0, HVX_Vector &vu16_dst1)
{
    HVX_VectorPair wu16_src0 = Q6_Wuh_vunpack_Vub(vu8_src0);
    HVX_VectorPair wu16_src1 = Q6_Wuh_vunpack_Vub(vu8_src1);

    vu16_dst0 = Q6_Vuh_vadd_VuhVuh_sat(vu16_dst0, Q6_Vuh_vadd_VuhVuh_sat(Q6_V_lo_W(wu16_src0), Q6_V_lo_W(wu16_src1)));
    vu16_dst1 = Q6_Vuh_vadd_VuhVuh_sat(vu16_dst1, Q6_Vuh_vadd_VuhVuh_sat(Q6_V_hi_W(wu16_src0), Q6_V_hi_W(wu16_src1)));
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKAddTwoRowSumCore(HVX_Vector &vu16_src0, HVX_Vector &vu16_src1, HVX_Vector &vu32_dst0, HVX_Vector &vu32_dst1)
{
    HVX_VectorPair wu32_src0 = Q6_Wuw_vunpack_Vuh(vu16_src0);
    HVX_VectorPair wu32_src1 = Q6_Wuw_vunpack_Vuh(vu16_src1);

    vu32_dst0 = Q6_Vuw_vadd_VuwVuw_sat(vu32_dst0, Q6_Vuw_vadd_VuwVuw_sat(Q6_V_lo_W(wu32_src0), Q6_V_lo_W(wu32_src1)));
    vu32_dst1 = Q6_Vuw_vadd_VuwVuw_sat(vu32_dst1, Q6_Vuw_vadd_VuwVuw_sat(Q6_V_hi_W(wu32_src0), Q6_V_hi_W(wu32_src1)));
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKAddTwoRowSumCore(HVX_Vector &vs16_src0, HVX_Vector &vs16_src1, HVX_Vector &vs32_dst0, HVX_Vector &vs32_dst1)
{
    HVX_VectorPair ws32_src0 = Q6_Ww_vunpack_Vh(vs16_src0);
    HVX_VectorPair ws32_src1 = Q6_Ww_vunpack_Vh(vs16_src1);

    vs32_dst0 = Q6_Vw_vadd_VwVw(vs32_dst0, Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_src0), Q6_V_lo_W(ws32_src1)));
    vs32_dst1 = Q6_Vw_vadd_VwVw(vs32_dst1, Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_src0), Q6_V_hi_W(ws32_src1)));
}

template<typename Tp, typename RowSumType, MI_S32 C>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKAddTwoRowSum(Tp *src0, Tp *src1, RowSumType *dst, MI_S32 width)
{
    constexpr MI_S32 SRC_ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);
    constexpr MI_S32 DST_ELEM_COUNTS = AURA_HVLEN / sizeof(RowSumType);

    const MI_S32 width_align     = width & (-SRC_ELEM_COUNTS);
    const MI_S32 rest            = width - width_align;

    HVX_Vector v_src0, v_src1;
    HVX_Vector v_dst0, v_dst1;

    MI_S32 x = 0;
    for (; x < width_align - SRC_ELEM_COUNTS; x += SRC_ELEM_COUNTS)
    {
        vload(src0 + x, v_src0);
        vload(src1 + x, v_src1);

        vload(dst + x,                   v_dst0);
        vload(dst + x + DST_ELEM_COUNTS, v_dst1);

        BoxFilterKxKAddTwoRowSumCore<Tp>(v_src0, v_src1, v_dst0, v_dst1);

        vstore(dst + x,                   v_dst0);
        vstore(dst + x + DST_ELEM_COUNTS, v_dst1);
    }

    if (x < width_align)
    {
        vload(src0 + x, v_src0);
        vload(src1 + x, v_src1);

        vload(dst + x,                   v_dst0);
        vload(dst + x + DST_ELEM_COUNTS, v_dst1);

        BoxFilterKxKAddTwoRowSumCore<Tp>(v_src0, v_src1, v_dst0, v_dst1);
    }

    HVX_Vector v_border_sum0, v_border_sum1;
    if (rest)
    {
        vload(src0 + width - SRC_ELEM_COUNTS, v_src0);
        vload(src1 + width - SRC_ELEM_COUNTS, v_src1);

        vload(dst + width - DST_ELEM_COUNTS * 2, v_border_sum0);
        vload(dst + width - DST_ELEM_COUNTS,     v_border_sum1);

        BoxFilterKxKAddTwoRowSumCore<Tp>(v_src0, v_src1, v_border_sum0, v_border_sum1);
    }

    if (x < width_align)
    {
        vstore(dst + x,                   v_dst0);
        vstore(dst + x + DST_ELEM_COUNTS, v_dst1);
    }

    if (rest)
    {
        vstore(dst + width - DST_ELEM_COUNTS * 2, v_border_sum0);
        vstore(dst + width - DST_ELEM_COUNTS,     v_border_sum1);
    }
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKSubAddRowSumCore(HVX_Vector &vu8_src_sub, HVX_Vector &vu8_src_add, HVX_Vector &vu16_dst0, HVX_Vector &vu16_dst1)
{
    HVX_VectorPair wu16_src_sub = Q6_Wuh_vunpack_Vub(vu8_src_sub);
    HVX_VectorPair wu16_src_add = Q6_Wuh_vunpack_Vub(vu8_src_add);

    vu16_dst0 = Q6_Vuh_vsub_VuhVuh_sat(vu16_dst0, Q6_V_lo_W(wu16_src_sub));
    vu16_dst1 = Q6_Vuh_vsub_VuhVuh_sat(vu16_dst1, Q6_V_hi_W(wu16_src_sub));
    vu16_dst0 = Q6_Vuh_vadd_VuhVuh_sat(vu16_dst0, Q6_V_lo_W(wu16_src_add));
    vu16_dst1 = Q6_Vuh_vadd_VuhVuh_sat(vu16_dst1, Q6_V_hi_W(wu16_src_add));
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value || std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKSubAddRowSumCore(HVX_Vector &vd16_src_sub, HVX_Vector &vd16_src_add, HVX_Vector &vd32_dst0, HVX_Vector &vd32_dst1)
{
    HVX_VectorPair wd32_src_sub = Q6_Ww_vunpack_Vh(vd16_src_sub);
    HVX_VectorPair wd32_src_add = Q6_Ww_vunpack_Vh(vd16_src_add);

    vd32_dst0 = Q6_Vw_vsub_VwVw(vd32_dst0, Q6_V_lo_W(wd32_src_sub));
    vd32_dst1 = Q6_Vw_vsub_VwVw(vd32_dst1, Q6_V_hi_W(wd32_src_sub));
    vd32_dst0 = Q6_Vw_vadd_VwVw(vd32_dst0, Q6_V_lo_W(wd32_src_add));
    vd32_dst1 = Q6_Vw_vadd_VwVw(vd32_dst1, Q6_V_hi_W(wd32_src_add));
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKSubAddRowSumCoreRest64(HVX_Vector &vu8_src_sub, HVX_Vector &vu8_src_add, HVX_Vector &vu16_dst, const MI_S32 shift_value)
{
    HVX_VectorPair wu16_src_sub = Q6_Wuh_vunpack_Vub(vu8_src_sub);
    HVX_VectorPair wu16_src_add = Q6_Wuh_vunpack_Vub(vu8_src_add);

    HVX_Vector v_src_offset = Q6_V_vror_VR(Q6_V_hi_W(wu16_src_sub), -shift_value);
    v_src_offset            = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vu16_dst                = Q6_Vuh_vsub_VuhVuh_sat(vu16_dst, v_src_offset);

    v_src_offset = Q6_V_vror_VR(Q6_V_hi_W(wu16_src_add), -shift_value);
    v_src_offset = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vu16_dst     = Q6_Vuh_vadd_VuhVuh_sat(vu16_dst, v_src_offset);
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKSubAddRowSumCoreRest64(HVX_Vector &vu16_src_sub, HVX_Vector &vu16_src_add, HVX_Vector &vu32_dst, const MI_S32 shift_value)
{
    HVX_VectorPair wu32_src_sub = Q6_Wuw_vunpack_Vuh(vu16_src_sub);
    HVX_VectorPair wu32_src_add = Q6_Wuw_vunpack_Vuh(vu16_src_add);

    HVX_Vector v_src_offset = Q6_V_vror_VR(Q6_V_hi_W(wu32_src_sub), -shift_value);
    v_src_offset            = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vu32_dst                = Q6_Vuw_vsub_VuwVuw_sat(vu32_dst, v_src_offset);

    v_src_offset = Q6_V_vror_VR(Q6_V_hi_W(wu32_src_add), -shift_value);
    v_src_offset = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vu32_dst     = Q6_Vuw_vadd_VuwVuw_sat(vu32_dst, v_src_offset);
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKSubAddRowSumCoreRest64(HVX_Vector &vs16_src_sub, HVX_Vector &vs16_src_add, HVX_Vector &vs32_dst, const MI_S32 shift_value)
{
    HVX_VectorPair ws32_src_sub = Q6_Ww_vunpack_Vh(vs16_src_sub);
    HVX_VectorPair ws32_src_add = Q6_Ww_vunpack_Vh(vs16_src_add);

    HVX_Vector v_src_offset = Q6_V_vror_VR(Q6_V_hi_W(ws32_src_sub), -shift_value);
    v_src_offset            = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vs32_dst                = Q6_Vw_vsub_VwVw(vs32_dst, v_src_offset);

    v_src_offset = Q6_V_vror_VR(Q6_V_hi_W(ws32_src_add), -shift_value);
    v_src_offset = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vs32_dst     = Q6_Vw_vadd_VwVw(vs32_dst, v_src_offset);
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKSubAddRowSumCoreRest128(HVX_Vector &vu8_src_sub, HVX_Vector &vu8_src_add, HVX_Vector &vu16_dst0, HVX_Vector &vu16_dst1, const MI_S32 shift_value)
{
    HVX_VectorPair wu16_src_sub = Q6_Wuh_vunpack_Vub(vu8_src_sub);
    HVX_Vector v_src_offset     = Q6_V_vror_VR(Q6_V_lo_W(wu16_src_sub), -shift_value);
    v_src_offset                = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vu16_dst0                   = Q6_Vuh_vsub_VuhVuh_sat(vu16_dst0, v_src_offset);
    vu16_dst1                   = Q6_Vuh_vsub_VuhVuh_sat(vu16_dst1, Q6_V_hi_W(wu16_src_sub));

    HVX_VectorPair wu16_src_add = Q6_Wuh_vunpack_Vub(vu8_src_add);
    v_src_offset                = Q6_V_vror_VR(Q6_V_lo_W(wu16_src_add), -shift_value);
    v_src_offset                = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vu16_dst0                   = Q6_Vuh_vadd_VuhVuh_sat(vu16_dst0, v_src_offset);
    vu16_dst1                   = Q6_Vuh_vadd_VuhVuh_sat(Q6_V_hi_W(wu16_src_add), vu16_dst1);
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKSubAddRowSumCoreRest128(HVX_Vector &vu16_src_sub, HVX_Vector &vu16_src_add, HVX_Vector &vu32_dst0, HVX_Vector &vu32_dst1, const MI_S32 shift_value)
{
    HVX_VectorPair wu32_src_sub = Q6_Wuw_vunpack_Vuh(vu16_src_sub);
    HVX_Vector v_src_offset     = Q6_V_vror_VR(Q6_V_lo_W(wu32_src_sub), -shift_value);
    v_src_offset                = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vu32_dst0                   = Q6_Vuw_vsub_VuwVuw_sat(vu32_dst0, v_src_offset);
    vu32_dst1                   = Q6_Vuw_vsub_VuwVuw_sat(vu32_dst1, Q6_V_hi_W(wu32_src_sub));

    HVX_VectorPair wu32_src_add = Q6_Wuw_vunpack_Vuh(vu16_src_add);
    v_src_offset                = Q6_V_vror_VR(Q6_V_lo_W(wu32_src_add), -shift_value);
    v_src_offset                = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vu32_dst0                   = Q6_Vuw_vadd_VuwVuw_sat(vu32_dst0, v_src_offset);
    vu32_dst1                   = Q6_Vuw_vadd_VuwVuw_sat(Q6_V_hi_W(wu32_src_add), vu32_dst1);
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKSubAddRowSumCoreRest128(HVX_Vector &vs16_src_sub, HVX_Vector &vs16_src_add, HVX_Vector &vs32_dst0, HVX_Vector &vs32_dst1, const MI_S32 shift_value)
{
    HVX_VectorPair ws32_src_sub = Q6_Ww_vunpack_Vh(vs16_src_sub);
    HVX_Vector v_src_offset     = Q6_V_vror_VR(Q6_V_lo_W(ws32_src_sub), -shift_value);
    v_src_offset                = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vs32_dst0                   = Q6_Vw_vsub_VwVw(vs32_dst0, v_src_offset);
    vs32_dst1                   = Q6_Vw_vsub_VwVw(vs32_dst1, Q6_V_hi_W(ws32_src_sub));

    HVX_VectorPair ws32_src_add = Q6_Ww_vunpack_Vh(vs16_src_add);
    v_src_offset                = Q6_V_vror_VR(Q6_V_lo_W(ws32_src_add), -shift_value);
    v_src_offset                = Q6_V_valign_VVR(v_src_offset, Q6_V_vzero(), shift_value);
    vs32_dst0                   = Q6_Vw_vadd_VwVw(vs32_dst0, v_src_offset);
    vs32_dst1                   = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_src_add), vs32_dst1);
}

template<typename Tp, typename RowSumType, MI_S32 C>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKSubAddRowSum(Tp *src_sub, Tp *src_add, RowSumType *dst, MI_S32 width)
{
    constexpr MI_S32 SRC_ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);
    constexpr MI_S32 DST_ELEM_COUNTS = AURA_HVLEN / sizeof(RowSumType);

    const MI_S32 width_align = width & (-SRC_ELEM_COUNTS);
    const MI_S32 rest        = width - width_align;
    const MI_S32 back_offset = width - SRC_ELEM_COUNTS;
    const MI_S32 shift_value = (rest % 64)  * sizeof(RowSumType);

    HVX_Vector v_src_sub, v_src_add;
    HVX_Vector v_dst0, v_dst1;

    for (MI_S32 x = 0; x < width_align; x += SRC_ELEM_COUNTS)
    {
        vload(src_sub + x, v_src_sub);
        vload(src_add + x, v_src_add);
        vload(dst + x, v_dst0);
        vload(dst + x + DST_ELEM_COUNTS, v_dst1);

        BoxFilterKxKSubAddRowSumCore<Tp>(v_src_sub, v_src_add, v_dst0, v_dst1);

        vstore(dst + x, v_dst0);
        vstore(dst + x + DST_ELEM_COUNTS, v_dst1);
    }

    if (rest > 0)
    {
        if (rest < SRC_ELEM_COUNTS / 2)
        {
            vload(src_sub + back_offset, v_src_sub);
            vload(src_add + back_offset, v_src_add);
            vload(dst + back_offset + DST_ELEM_COUNTS, v_dst1);

            BoxFilterKxKSubAddRowSumCoreRest64<Tp>(v_src_sub, v_src_add, v_dst1, shift_value);

            vstore(dst + back_offset + DST_ELEM_COUNTS, v_dst1);
        }
        else if (rest < SRC_ELEM_COUNTS)
        {
            vload(src_sub + back_offset, v_src_sub);
            vload(src_add + back_offset, v_src_add);
            vload(dst + back_offset, v_dst0);
            vload(dst + back_offset + DST_ELEM_COUNTS, v_dst1);

            BoxFilterKxKSubAddRowSumCoreRest128<Tp>(v_src_sub, v_src_add, v_dst0, v_dst1, shift_value);

            vstore(dst + back_offset, v_dst0);
            vstore(dst + back_offset + DST_ELEM_COUNTS, v_dst1);
        }
    }
}

AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKRowIntegralU16Core(HVX_Vector &vu16_src, HVX_Vector &vu32_dst0, HVX_Vector &vu32_dst1, HVX_Vector &vd32_rdelta)
{
    HVX_Vector vu32_mask     = Q6_V_vsplat_R(0x0000FFFF);
    HVX_Vector vu32_even     = Q6_V_vand_VV(vu16_src, vu32_mask);

    HVX_Vector vu32_sum      = Q6_Vw_vdmpy_VhRb(vu16_src, 0x01010101);
    HVX_Vector vu32_sum4     = Q6_V_vlalign_VVR(vu32_sum,  Q6_V_vzero(), 4);
    vu32_sum                 = Q6_Vuw_vadd_VuwVuw_sat(vu32_sum, vu32_sum4);
    HVX_Vector vu32_sum8     = Q6_V_vlalign_VVR(vu32_sum, Q6_V_vzero(), 8);
    vu32_sum                 = Q6_Vuw_vadd_VuwVuw_sat(vu32_sum, vu32_sum8);
    HVX_Vector vu32_sum16    = Q6_V_vlalign_VVR(vu32_sum, Q6_V_vzero(), 16);
    vu32_sum                 = Q6_Vuw_vadd_VuwVuw_sat(vu32_sum, vu32_sum16);
    HVX_Vector vu32_sum32    = Q6_V_vlalign_VVR(vu32_sum, Q6_V_vzero(), 32);
    vu32_sum                 = Q6_Vuw_vadd_VuwVuw_sat(vu32_sum, vu32_sum32);
    HVX_Vector vu32_sum64    = Q6_V_vlalign_VVR(vu32_sum, Q6_V_vzero(), 64);

    HVX_Vector vu32_sum_odd  = Q6_Vuw_vadd_VuwVuw_sat(vu32_sum, vu32_sum64);
    HVX_Vector vu32_sum_odd2 = Q6_V_vlalign_VVR(vu32_sum_odd, Q6_V_vzero(), 4);
    HVX_Vector vu32_sum_even = Q6_Vuw_vadd_VuwVuw_sat(vu32_sum_odd2, vu32_even);

    HVX_VectorPair vu32_pair = Q6_W_vshuff_VVR(vu32_sum_odd, vu32_sum_even, -4);
    HVX_Vector vu32_rep_last = Q6_V_vrdelta_VV(vu32_dst1, vd32_rdelta);

    vu32_dst0 = Q6_Vuw_vadd_VuwVuw_sat(vu32_rep_last, Q6_V_lo_W(vu32_pair));
    vu32_dst1 = Q6_Vuw_vadd_VuwVuw_sat(vu32_rep_last, Q6_V_hi_W(vu32_pair));
}

template <typename RowSumType, typename std::enable_if<std::is_same<RowSumType, MI_S32>::value || std::is_same<RowSumType, MI_U32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilterKxKRowIntegralD32Core(HVX_Vector &vs32_src, HVX_Vector &vs32_dst0, HVX_Vector &vd32_rdelta)
{
    HVX_Vector vs32_sum4  = Q6_V_vlalign_VVR(vs32_src, Q6_V_vzero(), 4);
    HVX_Vector vs32_sum   = Q6_Vw_vadd_VwVw(vs32_src, vs32_sum4);
    HVX_Vector vs32_sum8  = Q6_V_vlalign_VVR(vs32_sum, Q6_V_vzero(), 8);
    vs32_sum              = Q6_Vw_vadd_VwVw(vs32_sum, vs32_sum8);
    HVX_Vector vs32_sum16 = Q6_V_vlalign_VVR(vs32_sum, Q6_V_vzero(), 16);
    vs32_sum              = Q6_Vw_vadd_VwVw(vs32_sum, vs32_sum16);
    HVX_Vector vs32_sum32 = Q6_V_vlalign_VVR(vs32_sum, Q6_V_vzero(), 32);
    vs32_sum              = Q6_Vw_vadd_VwVw(vs32_sum, vs32_sum32);
    HVX_Vector vs32_sum64 = Q6_V_vlalign_VVR(vs32_sum, Q6_V_vzero(), 64);
    vs32_sum              = Q6_Vw_vadd_VwVw(vs32_sum, vs32_sum64);

    HVX_Vector vs32_rep_last = Q6_V_vrdelta_VV(vs32_dst0, vd32_rdelta);
    vs32_dst0 = Q6_Vw_vadd_VwVw(vs32_rep_last, vs32_sum);
}

template <typename RowSumType, MI_S32 C, typename std::enable_if<std::is_same<RowSumType, MI_U32>::value || std::is_same<RowSumType, MI_S32>::value>::type* = MI_NULL>
AURA_NO_INLINE AURA_VOID BoxFilterKxKRowIntegral(const RowSumType *src, RowSumType *dst, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;
    constexpr MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(RowSumType);

    MI_S32 width_align = width & (-ELEM_COUNTS);
    MI_S32 rest        = width - width_align;

    HVX_Vector vd32_rdelta = *(HVX_Vector *)(vrdelta_replicate_last_d32);
    MVType mvu32_src, mvu32_dst;

    #pragma unroll(C)
    for (MI_S32 ch = 0; ch < C; ch++)
    {
        mvu32_dst.val[ch] = Q6_V_vzero();
    }

    for (MI_S32 x = 0; x < width_align; x += ELEM_COUNTS)
    {
        vload(src + x * C, mvu32_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            BoxFilterKxKRowIntegralD32Core<RowSumType>(mvu32_src.val[ch], mvu32_dst.val[ch], vd32_rdelta);
        }

        vstore(dst + (x * C), mvu32_dst);
    }

    if (rest > 0)
    {
        MI_S32 shift_cnt  = AURA_HVLEN - rest;
        MI_S32 shift_cnt4 = AURA_HVLEN - (shift_cnt & 31) * sizeof(MI_U32);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvu32_dst.val[ch] = Q6_V_vror_VR(mvu32_dst.val[ch], shift_cnt4);
        }

        vload(src + (width - ELEM_COUNTS) * C, mvu32_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            BoxFilterKxKRowIntegralD32Core<RowSumType>(mvu32_src.val[ch], mvu32_dst.val[ch], vd32_rdelta);
        }

        vstore(dst + (width - ELEM_COUNTS) * C, mvu32_dst);
    }
}

template <typename RowSumType, MI_S32 C, typename std::enable_if<std::is_same<RowSumType, MI_U16>::value>::type* = MI_NULL>
AURA_NO_INLINE AURA_VOID BoxFilterKxKRowIntegral(const RowSumType *src, MI_U32 *dst, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    MI_S32 width_align = width & (-AURA_HALF_HVLEN);
    MI_S32 rest = width - width_align;

    HVX_Vector vd32_rdelta = *(HVX_Vector *)(vrdelta_replicate_last_d32);
    MVType mvu16_src, mvd32_dst0, mvd32_dst1;

    #pragma unroll(C)
    for (MI_S32 ch = 0; ch < C; ch++)
    {
        mvd32_dst1.val[ch] = Q6_V_vzero();
    }

    for (MI_S32 x = 0; x < width_align; x += AURA_HALF_HVLEN)
    {
        vload(src + x * C, mvu16_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            BoxFilterKxKRowIntegralU16Core(mvu16_src.val[ch], mvd32_dst0.val[ch], mvd32_dst1.val[ch], vd32_rdelta);
        }

        vstore(dst + (x * C), mvd32_dst0);
        vstore(dst + (x * C) + (AURA_HALF_HVLEN * C) / 2, mvd32_dst1);
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
                mvd32_dst1.val[ch] = Q6_V_vror_VR(mvd32_dst0.val[ch], shift_cnt4);
            }
        }
        else if (rest <= 64)
        {
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mvd32_dst1.val[ch] = Q6_V_vror_VR(mvd32_dst1.val[ch], shift_cnt4);
            }
        }

        vload(src + (width - AURA_HALF_HVLEN) * C, mvu16_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            BoxFilterKxKRowIntegralU16Core(mvu16_src.val[ch], mvd32_dst0.val[ch], mvd32_dst1.val[ch], vd32_rdelta);
        }

        vstore(dst + (width - AURA_HALF_HVLEN) * C, mvd32_dst0);
        vstore(dst + (width - AURA_HALF_HVLEN) * C + (AURA_HVLEN * C) / sizeof(MI_S32), mvd32_dst1);
    }
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_VOID BoxFilterKxKRowCore(HVX_Vector &vu32_src0, HVX_Vector &vu32_src0_r, HVX_Vector &vu32_src1, HVX_Vector &vu32_src1_r,
                            HVX_Vector &vu32_src2, HVX_Vector &vu32_src2_r, HVX_Vector &vu32_src3, HVX_Vector &vu32_src3_r,
                            HVX_Vector &vu32_replicate, HVX_Vector &vu32_result, const MI_S32 shift_left,
                            HvxVdivnHelper<typename BoxFilterTraits<Tp>::KernelSumType> &vidvn)
{
    HVX_Vector vu32_left;
    HVX_Vector vu32_result0, vu32_result1, vu32_result2, vu32_result3;

    vu32_left    = Q6_V_vlalign_VVR(vu32_src0, vu32_replicate, shift_left);
    vu32_result0 = Q6_Vuw_vsub_VuwVuw_sat(vu32_src0_r, vu32_left);
    vu32_result0 = vidvn(vu32_result0);

    vu32_left    = Q6_V_vlalign_VVR(vu32_src1, vu32_src0, shift_left);
    vu32_result1 = Q6_Vuw_vsub_VuwVuw_sat(vu32_src1_r, vu32_left);
    vu32_result1 = vidvn(vu32_result1);

    vu32_left    = Q6_V_vlalign_VVR(vu32_src2, vu32_src1, shift_left);
    vu32_result2 = Q6_Vuw_vsub_VuwVuw_sat(vu32_src2_r, vu32_left);
    vu32_result2 = vidvn(vu32_result2);

    vu32_left    = Q6_V_vlalign_VVR(vu32_src3, vu32_src2, shift_left);
    vu32_result3 = Q6_Vuw_vsub_VuwVuw_sat(vu32_src3_r, vu32_left);
    vu32_result3 = vidvn(vu32_result3);

    vu32_result = Q6_Vub_vpack_VhVh_sat(Q6_Vuh_vpack_VwVw_sat(vu32_result3, vu32_result2),
                                        Q6_Vuh_vpack_VwVw_sat(vu32_result1, vu32_result0));
}

template <typename Tp, typename KernelSumType, MI_S32 C, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_VOID BoxFilterKxKRow(const KernelSumType *src, Tp *dst, MI_S32 width, const MI_S32 ksize, HvxVdivnHelper<KernelSumType> &vidvn)
{
    using MVType                 = typename MVHvxVector<C>::Type;
    constexpr MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);
    const MI_S32 back_offset     = width - ELEM_COUNTS;
    const MI_S32 k_offset        = ksize - 1;
    const MI_S32 shift_left      = sizeof(KernelSumType);

    MVType mvu32_replicate;
    MVType mvu32_integral0, mvu32_integral1, mvu32_integral2, mvu32_integral3, mvu8_result;
    MVType mvu32_integral0_r, mvu32_integral1_r, mvu32_integral2_r, mvu32_integral3_r;

    for (MI_S32 ch = 0; ch < C; ch++)
    {
        mvu32_replicate.val[ch] = Q6_V_vzero();
    }

    for (MI_S32 x = 0; x <= back_offset; x += ELEM_COUNTS)
    {
        vload(src + x * C,                   mvu32_integral0);
        vload(src + (x + k_offset) * C,      mvu32_integral0_r);
        vload(src + (x + 32) * C,            mvu32_integral1);
        vload(src + (x + 32 + k_offset) * C, mvu32_integral1_r);
        vload(src + (x + 64) * C,            mvu32_integral2);
        vload(src + (x + 64 + k_offset) * C, mvu32_integral2_r);
        vload(src + (x + 96) * C,            mvu32_integral3);
        vload(src + (x + 96 + k_offset) * C, mvu32_integral3_r);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            BoxFilterKxKRowCore<Tp>(mvu32_integral0.val[ch], mvu32_integral0_r.val[ch], mvu32_integral1.val[ch], mvu32_integral1_r.val[ch],
                                    mvu32_integral2.val[ch], mvu32_integral2_r.val[ch], mvu32_integral3.val[ch], mvu32_integral3_r.val[ch],
                                    mvu32_replicate.val[ch], mvu8_result.val[ch], shift_left, vidvn);
        }

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvu32_replicate.val[ch] = mvu32_integral3.val[ch];
        }

        vstore(dst + x * C, mvu8_result);
    }

    //remain
    {
        vload(src + back_offset * C,                   mvu32_integral0);
        vload(src + (back_offset + k_offset) * C,      mvu32_integral0_r);
        vload(src + (back_offset + 32) * C,            mvu32_integral1);
        vload(src + (back_offset + 32 + k_offset) * C, mvu32_integral1_r);
        vload(src + (back_offset + 64) * C,            mvu32_integral2);
        vload(src + (back_offset + 64 + k_offset) * C, mvu32_integral2_r);
        vload(src + (back_offset + 96) * C,            mvu32_integral3);
        vload(src + (back_offset + 96 + k_offset) * C, mvu32_integral3_r);
        vload(src + (back_offset - 32) * C,            mvu32_replicate);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            BoxFilterKxKRowCore<Tp>(mvu32_integral0.val[ch], mvu32_integral0_r.val[ch], mvu32_integral1.val[ch], mvu32_integral1_r.val[ch],
                                    mvu32_integral2.val[ch], mvu32_integral2_r.val[ch], mvu32_integral3.val[ch], mvu32_integral3_r.val[ch],
                                    mvu32_replicate.val[ch], mvu8_result.val[ch], shift_left, vidvn);
        }

        vstore(dst + back_offset * C, mvu8_result);
    }
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_VOID BoxFilterKxKRowCore(HVX_Vector &vu32_src0, HVX_Vector &vu32_src0_r, HVX_Vector &vu32_src1, HVX_Vector &vu32_src1_r,
                            HVX_Vector &vu32_replicate, HVX_Vector &vu16_result, const MI_S32 shift_left,
                            HvxVdivnHelper<typename BoxFilterTraits<Tp>::KernelSumType> &vidvn)
{
    HVX_Vector vu32_left;
    HVX_Vector vu32_result0, vu32_result1;

    vu32_left    = Q6_V_vlalign_VVR(vu32_src0, vu32_replicate, shift_left);
    vu32_result0 = Q6_Vuw_vsub_VuwVuw_sat(vu32_src0_r, vu32_left);
    vu32_result0 = vidvn(vu32_result0);

    vu32_left    = Q6_V_vlalign_VVR(vu32_src1, vu32_src0, shift_left);
    vu32_result1 = Q6_Vuw_vsub_VuwVuw_sat(vu32_src1_r, vu32_left);
    vu32_result1 = vidvn(vu32_result1);

    vu16_result = Q6_Vuh_vpack_VwVw_sat(vu32_result1, vu32_result0);
}

template<typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_VOID BoxFilterKxKRowCore(HVX_Vector &vs32_src0, HVX_Vector &vs32_src0_r, HVX_Vector &vs32_src1, HVX_Vector &vs32_src1_r,
                            HVX_Vector &vs32_replicate, HVX_Vector &vs16_result, const MI_S32 shift_left,
                            HvxVdivnHelper<typename BoxFilterTraits<Tp>::KernelSumType> &vidvn)
{
    HVX_Vector vs32_left;
    HVX_Vector vs32_result0, vs32_result1;

    vs32_left    = Q6_V_vlalign_VVR(vs32_src0, vs32_replicate, shift_left);
    vs32_result0 = Q6_Vw_vsub_VwVw(vs32_src0_r, vs32_left);
    vs32_result0 = vidvn(vs32_result0);

    vs32_left    = Q6_V_vlalign_VVR(vs32_src1, vs32_src0, shift_left);
    vs32_result1 = Q6_Vw_vsub_VwVw(vs32_src1_r, vs32_left);
    vs32_result1 = vidvn(vs32_result1);

    vs16_result = Q6_Vh_vpack_VwVw_sat(vs32_result1, vs32_result0);
}

template <typename D16, typename D32, MI_S32 C, typename std::enable_if<std::is_same<D16, MI_U16>::value ||
          std::is_same<D16, MI_S16>::value>::type* = MI_NULL>
AURA_VOID BoxFilterKxKRow(const D32 *src, D16 *dst, MI_S32 width, const MI_S32 ksize, HvxVdivnHelper<D32> &vidvn)
{
    using MVType                 = typename MVHvxVector<C>::Type;
    constexpr MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(D16);
    const MI_S32 back_offset     = width - ELEM_COUNTS;
    const MI_S32 k_offset        = ksize - 1;
    const MI_S32 shift_left      = sizeof(D32);

    MVType mvd32_replicate;
    MVType mvd32_integral0, mvd32_integral1, mvd16_result;
    MVType mvd32_integral0_r, mvd32_integral1_r;

    for (MI_S32 ch = 0; ch < C; ch++)
    {
        mvd32_replicate.val[ch] = Q6_V_vzero();
    }

    for (MI_S32 x = 0; x <= back_offset; x += ELEM_COUNTS)
    {
        vload(src + x * C,                   mvd32_integral0);
        vload(src + (x + k_offset) * C,      mvd32_integral0_r);
        vload(src + (x + 32) * C,            mvd32_integral1);
        vload(src + (x + 32 + k_offset) * C, mvd32_integral1_r);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            BoxFilterKxKRowCore<D16>(mvd32_integral0.val[ch], mvd32_integral0_r.val[ch], mvd32_integral1.val[ch],
                                     mvd32_integral1_r.val[ch], mvd32_replicate.val[ch], mvd16_result.val[ch], shift_left, vidvn);
        }

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvd32_replicate.val[ch] = mvd32_integral1.val[ch];
        }

        vstore(dst + x * C, mvd16_result);
    }

    //remain
    {
        vload(src + back_offset * C,                   mvd32_integral0);
        vload(src + (back_offset + k_offset) * C,      mvd32_integral0_r);
        vload(src + (back_offset + 32) * C,            mvd32_integral1);
        vload(src + (back_offset + 32 + k_offset) * C, mvd32_integral1_r);
        vload(src + (back_offset - 32) * C,            mvd32_replicate);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            BoxFilterKxKRowCore<D16>(mvd32_integral0.val[ch], mvd32_integral0_r.val[ch], mvd32_integral1.val[ch],
                                     mvd32_integral1_r.val[ch], mvd32_replicate.val[ch], mvd16_result.val[ch], shift_left, vidvn);
        }

        vstore(dst + back_offset * C, mvd16_result);
    }
}

template <typename Tp, BorderType BORDER_TYPE, MI_S32 C, typename KernelSumType = typename BoxFilterTraits<Tp>::KernelSumType>
static Status BoxFilterKxKHvxImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &src_row_buffer, ThreadBuffer &sum_row_buffer,
                                  ThreadBuffer &integral_row_buffer, const Scalar &border_value, HvxVdivnHelper<KernelSumType> &vdivn,
                                  const MI_S32 ksize, MI_S32 start_row, MI_S32 end_row)
{
    using RowSumType = typename BoxFilterTraits<Tp>::RowSumType;

    const MI_S32 ksh           = ksize >> 1;
    const MI_S32 iwidth        = src.GetSizes().m_width;
    const MI_S32 iheight       = src.GetSizes().m_height;
    const MI_S32 istride       = src.GetStrides().m_width;
    const MI_S32 width_padding = (iwidth + 2 * ksh) * C;

    Tp *sum_row_data               = src_row_buffer.GetThreadData<Tp>();
    RowSumType *sum_vert           = sum_row_buffer.GetThreadData<RowSumType>();
    KernelSumType *integral_buffer = integral_row_buffer.GetThreadData<KernelSumType>();
    if ((MI_NULL == sum_row_data) || (MI_NULL == sum_vert) || (MI_NULL == integral_buffer))
    {
        AURA_ADD_ERROR_STRING(ctx, "malloc failed");
        return Status::ERROR;
    }

    std::vector<Tp *> src_row_idx(ksize);
    for (MI_S32 i = 0; i < ksize + 1; i++)
    {
        src_row_idx[i] = sum_row_data + i * width_padding;
    }

    memset(sum_vert,        0, sum_row_buffer.GetBufferSize());
    memset(sum_row_data,    0, src_row_buffer.GetBufferSize());
    memset(integral_buffer, 0, integral_row_buffer.GetBufferSize());

    MI_U64 L2fetch_param1 = L2PfParam(istride, iwidth * C * ElemTypeSize(src.GetElemType()), 1, 0);

    for (MI_S32 k = 0; k < ksize - 1; k += 2)
    {
        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src, start_row + k - ksh, iwidth, ksize, src_row_idx[k + 1], border_value);
        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src, start_row + (k + 1) - ksh, iwidth, ksize, src_row_idx[k + 2], border_value);
        BoxFilterKxKAddTwoRowSum<Tp, RowSumType, C>(src_row_idx[k + 1], src_row_idx[k + 2], sum_vert, width_padding);
    }

    MI_S32 idx_head = 0;
    MI_S32 idx_tail = ksize;
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y + ksh + 1 < iheight)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + ksh + 1)), L2fetch_param1);
        }

        Tp *dst_row = dst.Ptr<Tp>(y);

        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src, y + ksh, iwidth, ksize, src_row_idx[idx_tail], border_value);
        BoxFilterKxKSubAddRowSum<Tp, RowSumType, C>(src_row_idx[idx_head], src_row_idx[idx_tail], sum_vert, width_padding);

        idx_head = (idx_head + 1) % (ksize + 1);
        idx_tail = (idx_tail + 1) % (ksize + 1);

        BoxFilterKxKRowIntegral<RowSumType, C>(sum_vert, integral_buffer, iwidth + 2 * ksh);

        BoxFilterKxKRow<Tp, KernelSumType, C>(integral_buffer, dst_row, iwidth, ksize, vdivn);
    }

    return Status::OK;
}

template<typename Tp, BorderType BORDER_TYPE>
static Status BoxFilterKxKHvxHelper(Context *ctx, const Mat &src, Mat &dst, const MI_S32 ksize, const Scalar &border_value)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 channel = src.GetSizes().m_channel;

    using RowSumType    = typename BoxFilterTraits<Tp>::RowSumType;
    using KernelSumType = typename BoxFilterTraits<Tp>::KernelSumType;
    MI_S32 ksh = ksize >> 1;

    MI_S32 src_row_buffer_size      = AURA_ALIGN(((iwidth + ksh * 2) * channel), AURA_HVLEN) * (ksize + 1) * ElemTypeSize(src.GetElemType());
    MI_S32 sum_row_buffer_size      = AURA_ALIGN(((iwidth + ksh * 2) * channel), AURA_HVLEN) * sizeof(RowSumType);
    MI_S32 integral_row_buffer_size = AURA_ALIGN(((iwidth + ksh * 2) * channel), AURA_HVLEN) * sizeof(KernelSumType);

    ThreadBuffer src_row_buffer(ctx, src_row_buffer_size);
    ThreadBuffer sum_row_buffer(ctx, sum_row_buffer_size);
    ThreadBuffer integral_row_buffer(ctx, integral_row_buffer_size);

    HvxVdivnHelper<KernelSumType> vdivn(ksize * ksize);

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((MI_S32)0, height, BoxFilterKxKHvxImpl<Tp, BORDER_TYPE, 1>, ctx, std::cref(src), std::ref(dst),
                                  std::ref(src_row_buffer), std::ref(sum_row_buffer), std::ref(integral_row_buffer), std::cref(border_value), std::ref(vdivn), ksize);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((MI_S32)0, height, BoxFilterKxKHvxImpl<Tp, BORDER_TYPE, 2>, ctx, std::cref(src), std::ref(dst),
                                  std::ref(src_row_buffer), std::ref(sum_row_buffer), std::ref(integral_row_buffer), std::cref(border_value), std::ref(vdivn), ksize);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((MI_S32)0, height, BoxFilterKxKHvxImpl<Tp, BORDER_TYPE, 3>, ctx, std::cref(src), std::ref(dst),
                                  std::ref(src_row_buffer), std::ref(sum_row_buffer), std::ref(integral_row_buffer), std::cref(border_value), std::ref(vdivn), ksize);
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

template<typename Tp>
static Status BoxFilterKxKHvxHelper(Context *ctx, const Mat &src, Mat &dst, const MI_S32 ksize, const BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            ret = BoxFilterKxKHvxHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, ksize, border_value);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = BoxFilterKxKHvxHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, ksize, border_value);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = BoxFilterKxKHvxHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, ksize, border_value);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported border_type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status BoxFilterKxKHvx(Context *ctx, const Mat &src, Mat &dst, const MI_S32 ksize, const BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = BoxFilterKxKHvxHelper<MI_U8>(ctx, src, dst, ksize, border_type, border_value);
            break;
        }

        case ElemType::U16:
        {
            ret = BoxFilterKxKHvxHelper<MI_U16>(ctx, src, dst, ksize, border_type, border_value);
            break;
        }

        case ElemType::S16:
        {
            ret = BoxFilterKxKHvxHelper<MI_S16>(ctx, src, dst, ksize, border_type, border_value);
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

} // namespace aura