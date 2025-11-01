#include "boxfilter_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/thread_buffer.h"

namespace aura
{

// using Tp = MI_U8
template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11VCore(const HVX_Vector &vu8_src_p0, const HVX_Vector &vu8_src_n0, HVX_VectorPair &ws16_result)
{
    ws16_result = Q6_Wh_vsub_VubVub(vu8_src_n0, vu8_src_p0);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11VCore(const HVX_Vector &vu8_src_p0, const HVX_Vector &vu8_src_n0,
                                               const HVX_Vector &vu8_src_p1, const HVX_Vector &vu8_src_n1,
                                               HVX_VectorPair &ws16_diff0,  HVX_VectorPair &ws16_diff1,
                                               HVX_Vector &vs16_com0, HVX_Vector &vs16_sum1)
{
    ws16_diff0 = Q6_Wh_vsub_VubVub(vu8_src_n0, vu8_src_p0);
    ws16_diff1 = Q6_Wh_vsub_VubVub(vu8_src_n1, vu8_src_p1);

    HVX_Vector vs16_sum0 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_diff0), Q6_V_hi_W(ws16_diff0));

    vs16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_diff1), Q6_V_hi_W(ws16_diff1));
    vs16_com0 = Q6_Vh_vadd_VhVh(Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 2), Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 4));
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11HCore(HVX_VectorPair &ws16_diff0, HVX_VectorPair &ws16_diff1,
                                               HVX_VectorPair &ws16_diff2, HVX_Vector &vs16_com0, HVX_Vector &vs16_sum1,
                                               HVX_Vector &vu8_result, const SumType *row_sum, MI_S32 row_sum_step)
{
    HVX_Vector vs16_sum1_p = vs16_sum1;
    HVX_Vector vs16_sum0_p = vs16_com0;

    vs16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_diff2), Q6_V_hi_W(ws16_diff2));
    vs16_com0 = Q6_Vh_vadd_VhVh(Q6_V_vlalign_VVI(vs16_sum1, vs16_sum1_p, 2), Q6_V_vlalign_VVI(vs16_sum1, vs16_sum1_p, 4));

    HVX_Vector vs16_com3 = Q6_V_valign_VVI(vs16_sum1, vs16_sum1_p, 4);
    HVX_Vector vs16_com2 = Q6_V_valign_VVI(vs16_com0, vs16_sum0_p, 4);

    HVX_Vector vs16_even = Q6_V_vlalign_VVI(Q6_V_hi_W(ws16_diff1), Q6_V_hi_W(ws16_diff0), 6);
    HVX_Vector vs16_odd  = Q6_V_valign_VVI(Q6_V_lo_W(ws16_diff2), Q6_V_lo_W(ws16_diff1), 6);
    HVX_Vector vs16_com  = Q6_Vh_vadd_VhVh(vs16_com2, Q6_Vh_vadd_VhVh(vs16_sum0_p, vs16_com3));

    HVX_Vector *row_sum_data_in  = (HVX_Vector*)(row_sum + row_sum_step);
    HVX_Vector *row_sum_data_out = row_sum_data_in;

    HVX_Vector vs16_com_sume = *row_sum_data_in++;
    HVX_Vector vs16_com_sumo = *row_sum_data_in++;

    vs16_com_sume = Q6_Vh_vadd_VhVh(vs16_com_sume, Q6_Vh_vadd_VhVh(vs16_com, vs16_even));
    vs16_com_sumo = Q6_Vh_vadd_VhVh(vs16_com_sumo, Q6_Vh_vadd_VhVh(vs16_com, vs16_odd));

    *row_sum_data_out++ = vs16_com_sume;
    *row_sum_data_out++ = vs16_com_sumo;

    HVX_Vector vu16_dst_odd  = vdiv_n<MI_U16, 121>(vs16_com_sumo);
    HVX_Vector vu16_dst_even = vdiv_n<MI_U16, 121>(vs16_com_sume);

    vu8_result = Q6_Vub_vsat_VhVh(vu16_dst_odd, vu16_dst_even);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11HCore(HVX_VectorPair &ws16_diff_x0, HVX_VectorPair &ws16_diff_x1,
                                               HVX_VectorPair &ws16_diff_x2, HVX_VectorPair &ws16_diff_x3,
                                               HVX_Vector &vu8_result_x0, HVX_Vector &vu8_result_x1,
                                               SumType *row_sum, MI_S32 row_sum_step, MI_S32 rest)
{
    HVX_Vector vs16_diff_x0_lo = Q6_V_lo_W(ws16_diff_x0);
    HVX_Vector vs16_diff_x0_hi = Q6_V_hi_W(ws16_diff_x0);
    HVX_Vector vs16_diff_x1_lo = Q6_V_lo_W(ws16_diff_x1);
    HVX_Vector vs16_diff_x1_hi = Q6_V_hi_W(ws16_diff_x1);
    HVX_Vector vs16_diff_x2_lo = Q6_V_lo_W(ws16_diff_x2);
    HVX_Vector vs16_diff_x2_hi = Q6_V_hi_W(ws16_diff_x2);
    HVX_Vector vs16_diff_x3_lo = Q6_V_lo_W(ws16_diff_x3);
    HVX_Vector vs16_diff_x3_hi = Q6_V_hi_W(ws16_diff_x3);

    HVX_Vector vs16_diff_l0_lo, vs16_diff_l0_hi, vs16_diff_r0_lo, vs16_diff_r0_hi;
    if (rest & 1)
    {
        MI_S32 align_size0 = (rest / 2) * sizeof(SumType);
        MI_S32 align_size1 = align_size0 + sizeof(SumType);
        vs16_diff_r0_lo = Q6_V_vlalign_safe_VVR(vs16_diff_x3_hi, vs16_diff_x2_hi, align_size1);
        vs16_diff_r0_hi = Q6_V_vlalign_safe_VVR(vs16_diff_x3_lo, vs16_diff_x2_lo, align_size0);
        vs16_diff_l0_lo = Q6_V_valign_safe_VVR(vs16_diff_x1_hi, vs16_diff_x0_hi, align_size0);
        vs16_diff_l0_hi = Q6_V_valign_safe_VVR(vs16_diff_x1_lo, vs16_diff_x0_lo, align_size1);
    }
    else
    {
        MI_S32 align_size = (rest / 2) * sizeof(SumType);
        vs16_diff_r0_lo = Q6_V_vlalign_safe_VVR(vs16_diff_x3_lo, vs16_diff_x2_lo, align_size);
        vs16_diff_r0_hi = Q6_V_vlalign_safe_VVR(vs16_diff_x3_hi, vs16_diff_x2_hi, align_size);
        vs16_diff_l0_lo = Q6_V_valign_safe_VVR(vs16_diff_x1_lo, vs16_diff_x0_lo, align_size);
        vs16_diff_l0_hi = Q6_V_valign_safe_VVR(vs16_diff_x1_hi, vs16_diff_x0_hi, align_size);
    }

    HVX_VectorPair ws16_sum_r0 = Q6_W_vcombine_VV(vs16_diff_r0_hi, vs16_diff_r0_lo);
    HVX_VectorPair ws16_sum_l0 = Q6_W_vcombine_VV(vs16_diff_l0_hi, vs16_diff_l0_lo);

    HVX_Vector vs16_sum0 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_diff_x0), Q6_V_hi_W(ws16_diff_x0));
    HVX_Vector vs16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_diff_x1), Q6_V_hi_W(ws16_diff_x1));
    HVX_Vector vs16_com0 = Q6_Vh_vadd_VhVh(Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 2), Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 4));

    BoxFilter11x11HCore<Tp, SumType>(ws16_diff_x0, ws16_diff_x1, ws16_sum_r0, vs16_com0, vs16_sum1, vu8_result_x0, row_sum, row_sum_step);

    vs16_sum0 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_sum_l0), Q6_V_hi_W(ws16_sum_l0));
    vs16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_diff_x2), Q6_V_hi_W(ws16_diff_x2));
    vs16_com0 = Q6_Vh_vadd_VhVh(Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 2), Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 4));

    BoxFilter11x11HCore<Tp, SumType>(ws16_sum_l0, ws16_diff_x2, ws16_diff_x3, vs16_com0, vs16_sum1, vu8_result_x1, row_sum, row_sum_step + 128);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowVCore(const HVX_Vector &vu8_src_p4, const HVX_Vector &vu8_src_p3,
                                                       const HVX_Vector &vu8_src_p2, const HVX_Vector &vu8_src_p1, const HVX_Vector &vu8_src_p0,
                                                       const HVX_Vector &vu8_src_c,  const HVX_Vector &vu8_src_n0, const HVX_Vector &vu8_src_n1,
                                                       const HVX_Vector &vu8_src_n2, const HVX_Vector &vu8_src_n3, const HVX_Vector &vu8_src_n4,
                                                       HVX_VectorPair &ws16_result)
{
    HVX_VectorPair ws16_sum_p4n4 = Q6_Wh_vadd_VubVub(vu8_src_p4, vu8_src_n4);
    HVX_VectorPair ws16_sum_p3n3 = Q6_Wh_vadd_VubVub(vu8_src_p3, vu8_src_n3);
    HVX_VectorPair ws16_sum1     = Q6_Wh_vadd_WhWh(ws16_sum_p4n4, ws16_sum_p3n3);

    HVX_VectorPair ws16_sum_p2n2 = Q6_Wh_vadd_VubVub(vu8_src_p2, vu8_src_n2);
    HVX_VectorPair ws16_sum_p1n1 = Q6_Wh_vadd_VubVub(vu8_src_p1, vu8_src_n1);
    HVX_VectorPair ws16_sum0     = Q6_Wh_vadd_WhWh(ws16_sum_p2n2, ws16_sum_p1n1);

    ws16_result  = Q6_Wh_vadd_VubVub(vu8_src_p0, vu8_src_n0);
    ws16_sum0    = Q6_Wh_vadd_WhWh(ws16_sum1, ws16_sum0);
    ws16_result  = Q6_Wh_vadd_WhWh(ws16_result, ws16_sum0);
    ws16_result  = Q6_Wh_vmpyacc_WhVubRb(ws16_result, vu8_src_c, 0x01010101);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowVCore(const HVX_Vector &vu8_src_p4l0, const HVX_Vector &vu8_src_p3l0, const HVX_Vector &vu8_src_p2l0,
                                                       const HVX_Vector &vu8_src_p1l0, const HVX_Vector &vu8_src_p0l0, const HVX_Vector &vu8_src_cl0,
                                                       const HVX_Vector &vu8_src_n0l0, const HVX_Vector &vu8_src_n1l0, const HVX_Vector &vu8_src_n2l0,
                                                       const HVX_Vector &vu8_src_n3l0, const HVX_Vector &vu8_src_n4l0, const HVX_Vector &vu8_src_p4c,
                                                       const HVX_Vector &vu8_src_p3c,  const HVX_Vector &vu8_src_p2c,  const HVX_Vector &vu8_src_p1c,
                                                       const HVX_Vector &vu8_src_p0c,  const HVX_Vector &vu8_src_cc,   const HVX_Vector &vu8_src_n0c,
                                                       const HVX_Vector &vu8_src_n1c,  const HVX_Vector &vu8_src_n2c,  const HVX_Vector &vu8_src_n3c,
                                                       const HVX_Vector &vu8_src_n4c,  HVX_VectorPair &ws16_sum0,  HVX_VectorPair &ws16_sum1,
                                                       HVX_Vector &vs16_com0, HVX_Vector &vs16_sum1)
{
    HVX_VectorPair ws16_sum_p4n4 = Q6_Wh_vadd_VubVub(vu8_src_p4l0, vu8_src_n4l0);
    HVX_VectorPair ws16_sum_p3n3 = Q6_Wh_vadd_VubVub(vu8_src_p3l0, vu8_src_n3l0);
    HVX_VectorPair ws16_sum_p1n1 = Q6_Wh_vadd_VubVub(vu8_src_p1l0, vu8_src_n1l0);
    HVX_VectorPair ws16_row_sum0 = Q6_Wh_vadd_WhWh(ws16_sum_p4n4, ws16_sum_p3n3);
    HVX_VectorPair ws16_sum_p2n2 = Q6_Wh_vadd_VubVub(vu8_src_p2l0, vu8_src_n2l0);
    HVX_VectorPair ws16_row_sum1 = Q6_Wh_vadd_WhWh(ws16_sum_p2n2, ws16_sum_p1n1);
    ws16_row_sum0 = Q6_Wh_vadd_WhWh(ws16_row_sum1, ws16_row_sum0);
    ws16_sum0     = Q6_Wh_vadd_VubVub(vu8_src_p0l0, vu8_src_n0l0);
    ws16_sum0     = Q6_Wh_vadd_WhWh(ws16_sum0, ws16_row_sum0);
    ws16_sum0     = Q6_Wh_vmpyacc_WhVubRb(ws16_sum0, vu8_src_cl0, 0x01010101);


    ws16_sum_p4n4 = Q6_Wh_vadd_VubVub(vu8_src_p4c, vu8_src_n4c);
    ws16_sum_p3n3 = Q6_Wh_vadd_VubVub(vu8_src_p3c, vu8_src_n3c);
    ws16_sum_p1n1 = Q6_Wh_vadd_VubVub(vu8_src_p1c, vu8_src_n1c);
    ws16_row_sum0 = Q6_Wh_vadd_WhWh(ws16_sum_p4n4, ws16_sum_p3n3);
    ws16_sum_p2n2 = Q6_Wh_vadd_VubVub(vu8_src_p2c, vu8_src_n2c);
    ws16_row_sum1 = Q6_Wh_vadd_WhWh(ws16_sum_p2n2, ws16_sum_p1n1);
    ws16_sum1     = Q6_Wh_vadd_VubVub(vu8_src_p0c, vu8_src_n0c);
    ws16_row_sum1 = Q6_Wh_vadd_WhWh(ws16_row_sum1, ws16_row_sum0);
    ws16_sum1     = Q6_Wh_vadd_WhWh(ws16_sum1, ws16_row_sum1);
    ws16_sum1     = Q6_Wh_vmpyacc_WhVubRb(ws16_sum1, vu8_src_cc, 0x01010101);

    HVX_Vector vs16_sum0 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_sum0), Q6_V_hi_W(ws16_sum0));

    vs16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_sum1), Q6_V_hi_W(ws16_sum1));
    vs16_com0 = Q6_Vh_vadd_VhVh(Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 2), Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 4));
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowHCore(HVX_VectorPair &ws16_sum0, HVX_VectorPair &ws16_sum1,
                                                       HVX_VectorPair &ws16_sum2, HVX_Vector &vs16_com0, HVX_Vector &vs16_sum1,
                                                       HVX_Vector &vu8_result, const SumType *row_sum, MI_S32 row_sum_step)
{
    HVX_Vector vs16_sum1_p = vs16_sum1;
    HVX_Vector vs16_sum0_p = vs16_com0;

    vs16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_sum2), Q6_V_hi_W(ws16_sum2));
    vs16_com0 = Q6_Vh_vadd_VhVh(Q6_V_vlalign_VVI(vs16_sum1, vs16_sum1_p, 2), Q6_V_vlalign_VVI(vs16_sum1, vs16_sum1_p, 4));

    HVX_Vector vs16_com3 = Q6_V_valign_VVI(vs16_sum1, vs16_sum1_p, 4);
    HVX_Vector vs16_com2 = Q6_V_valign_VVI(vs16_com0, vs16_sum0_p, 4);

    HVX_Vector vs16_even = Q6_V_vlalign_VVI(Q6_V_hi_W(ws16_sum1), Q6_V_hi_W(ws16_sum0), 6);
    HVX_Vector vs16_odd  = Q6_V_valign_VVI(Q6_V_lo_W(ws16_sum2), Q6_V_lo_W(ws16_sum1), 6);
    HVX_Vector vs16_com  = Q6_Vh_vadd_VhVh(vs16_com2, Q6_Vh_vadd_VhVh(vs16_sum0_p, vs16_com3));

    HVX_Vector *row_sum_data_out = (HVX_Vector*)(row_sum + row_sum_step);

    HVX_Vector vs16_com_sume = Q6_Vh_vadd_VhVh(vs16_com, vs16_even);
    HVX_Vector vs16_com_sumo = Q6_Vh_vadd_VhVh(vs16_com, vs16_odd);

    *row_sum_data_out++ = vs16_com_sume;
    *row_sum_data_out++ = vs16_com_sumo;

    HVX_Vector vu16_dst_odd  = vdiv_n<MI_U16, 121>(vs16_com_sumo);
    HVX_Vector vu16_dst_even = vdiv_n<MI_U16, 121>(vs16_com_sume);

    vu8_result = Q6_Vub_vsat_VhVh(vu16_dst_odd, vu16_dst_even);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowHCore(HVX_VectorPair &wu16_sum_x0, HVX_VectorPair &wu16_sum_x1, HVX_VectorPair &wu16_sum_x2, HVX_VectorPair &wu16_sum_x3,
                                                       HVX_Vector &vu8_result_x0, HVX_Vector &vu8_result_x1, SumType *row_sum, MI_S32 row_sum_step, MI_S32 rest)
{
    HVX_Vector vu16_sum_x0_lo = Q6_V_lo_W(wu16_sum_x0);
    HVX_Vector vu16_sum_x0_hi = Q6_V_hi_W(wu16_sum_x0);
    HVX_Vector vu16_sum_x1_lo = Q6_V_lo_W(wu16_sum_x1);
    HVX_Vector vu16_sum_x1_hi = Q6_V_hi_W(wu16_sum_x1);
    HVX_Vector vu16_sum_x2_lo = Q6_V_lo_W(wu16_sum_x2);
    HVX_Vector vu16_sum_x2_hi = Q6_V_hi_W(wu16_sum_x2);
    HVX_Vector vu16_sum_x3_lo = Q6_V_lo_W(wu16_sum_x3);
    HVX_Vector vu16_sum_x3_hi = Q6_V_hi_W(wu16_sum_x3);

    HVX_Vector vu16_sum_l0_lo, vu16_sum_l0_hi, vu16_sum_r0_lo, vu16_sum_r0_hi;
    if (rest & 1)
    {
        MI_S32 align_size0 = (rest / 2) * sizeof(SumType);
        MI_S32 align_size1 = align_size0 + sizeof(SumType);
        vu16_sum_r0_lo = Q6_V_vlalign_safe_VVR(vu16_sum_x3_hi, vu16_sum_x2_hi, align_size1);
        vu16_sum_r0_hi = Q6_V_vlalign_safe_VVR(vu16_sum_x3_lo, vu16_sum_x2_lo, align_size0);
        vu16_sum_l0_lo = Q6_V_valign_safe_VVR(vu16_sum_x1_hi, vu16_sum_x0_hi, align_size0);
        vu16_sum_l0_hi = Q6_V_valign_safe_VVR(vu16_sum_x1_lo, vu16_sum_x0_lo, align_size1);
    }
    else
    {
        MI_S32 align_size = (rest / 2) * sizeof(SumType);
        vu16_sum_r0_lo = Q6_V_vlalign_safe_VVR(vu16_sum_x3_lo, vu16_sum_x2_lo, align_size);
        vu16_sum_r0_hi = Q6_V_vlalign_safe_VVR(vu16_sum_x3_hi, vu16_sum_x2_hi, align_size);
        vu16_sum_l0_lo = Q6_V_valign_safe_VVR(vu16_sum_x1_lo, vu16_sum_x0_lo, align_size);
        vu16_sum_l0_hi = Q6_V_valign_safe_VVR(vu16_sum_x1_hi, vu16_sum_x0_hi, align_size);
    }

    HVX_VectorPair wu16_sum_r0 = Q6_W_vcombine_VV(vu16_sum_r0_hi, vu16_sum_r0_lo);
    HVX_VectorPair wu16_sum_l0 = Q6_W_vcombine_VV(vu16_sum_l0_hi, vu16_sum_l0_lo);

    HVX_Vector vs16_sum0 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(wu16_sum_x0), Q6_V_hi_W(wu16_sum_x0));
    HVX_Vector vs16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(wu16_sum_x1), Q6_V_hi_W(wu16_sum_x1));
    HVX_Vector vs16_com0 = Q6_Vh_vadd_VhVh(Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 2), Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 4));

    BoxFilter11x11FirstRowHCore<Tp, SumType>(wu16_sum_x0, wu16_sum_x1, wu16_sum_r0, vs16_com0, vs16_sum1, vu8_result_x0, row_sum, row_sum_step);

    vs16_sum0 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(wu16_sum_l0), Q6_V_hi_W(wu16_sum_l0));
    vs16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(wu16_sum_x2), Q6_V_hi_W(wu16_sum_x2));
    vs16_com0 = Q6_Vh_vadd_VhVh(Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 2), Q6_V_vlalign_VVI(vs16_sum1, vs16_sum0, 4));

    BoxFilter11x11FirstRowHCore<Tp, SumType>(wu16_sum_l0, wu16_sum_x2, wu16_sum_x3, vs16_com0, vs16_sum1, vu8_result_x1, row_sum, row_sum_step + 128);
}

// using Tp = MI_U16
template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11VCore(const HVX_Vector &vu16_src_p0, const HVX_Vector &vu16_src_n0, HVX_VectorPair &ws32_result)
{
    ws32_result = Q6_Ww_vsub_VuhVuh(vu16_src_n0, vu16_src_p0);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11VCore(const HVX_Vector &vu16_src_p0, const HVX_Vector &vu16_src_n0,
                                               const HVX_Vector &vu16_src_p1, const HVX_Vector &vu16_src_n1,
                                               HVX_VectorPair &ws32_diff0,  HVX_VectorPair &ws32_diff1,
                                               HVX_Vector &vs32_com0, HVX_Vector &vs32_sum1)
{
    ws32_diff0 = Q6_Ww_vsub_VuhVuh(vu16_src_n0, vu16_src_p0);
    ws32_diff1 = Q6_Ww_vsub_VuhVuh(vu16_src_n1, vu16_src_p1);
    vs32_sum1  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_diff1), Q6_V_hi_W(ws32_diff1));

    HVX_VectorPred vq_mask0 = Q6_Q_vsetq_R(8);

    HVX_Vector vs32_sum0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_diff0), Q6_V_hi_W(ws32_diff0));
    HVX_Vector vs32_sum  = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(vs32_sum0, -8), Q6_V_vror_VR(vs32_sum1, -8));

    vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum0, 4), vs32_sum);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11HCore(HVX_VectorPair &ws32_diff0, HVX_VectorPair &ws32_diff1,
                                               HVX_VectorPair &ws32_diff2, HVX_Vector &vs32_com0, HVX_Vector &vs32_sum1,
                                               HVX_Vector &vu16_result,
                                               const SumType *row_sum, MI_S32 row_sum_step)
{
    HVX_VectorPred vq_mask0 = Q6_Q_vsetq_R(12);
    HVX_VectorPred vq_mask1 = Q6_Q_vsetq_R(8);
    HVX_VectorPred vq_mask2 = Q6_Q_vsetq_R(116);
    HVX_VectorPred vq_mask3 = Q6_Q_vsetq_R(120);

    HVX_Vector vs32_sum1_p = vs32_sum1;
    HVX_Vector vs32_sum0_p = vs32_com0;

    HVX_Vector vs32_sum;
    vs32_sum1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_diff2), Q6_V_hi_W(ws32_diff2));
    vs32_sum  = Q6_V_vmux_QVV(vq_mask1, Q6_V_vror_VR(vs32_sum1_p, -8), Q6_V_vror_VR(vs32_sum1, -8));
    vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum1_p, 4), vs32_sum);

    HVX_Vector vs32_com3 = Q6_V_vmux_QVV(vq_mask3, Q6_V_vror_VR(vs32_sum1_p, 8), Q6_V_vror_VR(vs32_sum1, 8));
    HVX_Vector vs32_com2 = Q6_V_vmux_QVV(vq_mask3, Q6_V_vror_VR(vs32_sum0_p, 8), Q6_V_vror_VR(vs32_com0, 8));

    HVX_Vector vs32_even = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(Q6_V_hi_W(ws32_diff0), -12), Q6_V_vror_VR(Q6_V_hi_W(ws32_diff1), -12));
    HVX_Vector vs32_odd  = Q6_V_vmux_QVV(vq_mask2, Q6_V_vror_VR(Q6_V_lo_W(ws32_diff1),  12), Q6_V_vror_VR(Q6_V_lo_W(ws32_diff2),  12));
    HVX_Vector vs32_com  = Q6_Vw_vadd_VwVw(vs32_com2, Q6_Vw_vadd_VwVw(vs32_sum0_p, vs32_com3));

    HVX_Vector *row_sum_data_in  = (HVX_Vector*)(row_sum + row_sum_step);
    HVX_Vector *row_sum_data_out = row_sum_data_in;

    HVX_Vector vs32_sum_even = *row_sum_data_in++;
    HVX_Vector vs32_sum_odd  = *row_sum_data_in++;

    vs32_sum_even = Q6_Vw_vadd_VwVw(vs32_sum_even, Q6_Vw_vadd_VwVw(vs32_com, vs32_even));
    vs32_sum_odd  = Q6_Vw_vadd_VwVw(vs32_sum_odd,  Q6_Vw_vadd_VwVw(vs32_com, vs32_odd));

    *row_sum_data_out++ = vs32_sum_even;
    *row_sum_data_out++ = vs32_sum_odd;

    HVX_Vector vu32_dst_odd  = vdiv_n<MI_U32, 121>(vs32_sum_odd);
    HVX_Vector vu32_dst_even = vdiv_n<MI_U32, 121>(vs32_sum_even);

    vu16_result = Q6_Vuh_vsat_VuwVuw(vu32_dst_odd, vu32_dst_even);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowVCore(const HVX_Vector &vu16_src_p4, const HVX_Vector &vu16_src_p3, const HVX_Vector &vu16_src_p2,
                                                       const HVX_Vector &vu16_src_p1, const HVX_Vector &vu16_src_p0, const HVX_Vector &vu16_src_c,
                                                       const HVX_Vector &vu16_src_n0, const HVX_Vector &vu16_src_n1, const HVX_Vector &vu16_src_n2,
                                                       const HVX_Vector &vu16_src_n3, const HVX_Vector &vu16_src_n4,  HVX_VectorPair &ws32_result)
{
    HVX_VectorPair ws32_sum_p4n4 = Q6_Ww_vadd_VuhVuh(vu16_src_p4, vu16_src_n4);
    HVX_VectorPair ws32_sum_p3n3 = Q6_Ww_vadd_VuhVuh(vu16_src_p3, vu16_src_n3);
    HVX_VectorPair ws32_sum      = Q6_Ww_vadd_WwWw(ws32_sum_p4n4, ws32_sum_p3n3);

    HVX_VectorPair ws32_sum_p2n2 = Q6_Ww_vadd_VuhVuh(vu16_src_p2, vu16_src_n2);
    HVX_VectorPair ws32_sum_p1n1 = Q6_Ww_vadd_VuhVuh(vu16_src_p1, vu16_src_n1);
    ws32_sum                     = Q6_Ww_vadd_WwWw(ws32_sum, Q6_Ww_vadd_WwWw(ws32_sum_p2n2, ws32_sum_p1n1));

    ws32_result = Q6_Ww_vadd_VuhVuh(vu16_src_p0, vu16_src_n0);
    ws32_result = Q6_Ww_vadd_WwWw(ws32_result, ws32_sum);
    ws32_result = Q6_Wuw_vmpyacc_WuwVuhRuh(ws32_result, vu16_src_c, 0x00010001);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowVCore(const HVX_Vector &vu16_src_p4l0, const HVX_Vector &vu16_src_p3l0, const HVX_Vector &vu16_src_p2l0,
                                                       const HVX_Vector &vu16_src_p1l0, const HVX_Vector &vu16_src_p0l0, const HVX_Vector &vu16_src_cl0,
                                                       const HVX_Vector &vu16_src_n0l0, const HVX_Vector &vu16_src_n1l0, const HVX_Vector &vu16_src_n2l0,
                                                       const HVX_Vector &vu16_src_n3l0, const HVX_Vector &vu16_src_n4l0, const HVX_Vector &vu16_src_p4c,
                                                       const HVX_Vector &vu16_src_p3c,  const HVX_Vector &vu16_src_p2c,  const HVX_Vector &vu16_src_p1c,
                                                       const HVX_Vector &vu16_src_p0c,  const HVX_Vector &vu16_src_cc,   const HVX_Vector &vu16_src_n0c,
                                                       const HVX_Vector &vu16_src_n1c,  const HVX_Vector &vu16_src_n2c,  const HVX_Vector &vu16_src_n3c,
                                                       const HVX_Vector &vu16_src_n4c, HVX_VectorPair &ws32_sum0,  HVX_VectorPair &ws32_sum1,
                                                       HVX_Vector &vs32_com0, HVX_Vector &vs32_sum1)
{
    HVX_VectorPair ws32_sum_p4n4 = Q6_Ww_vadd_VuhVuh(vu16_src_p4l0, vu16_src_n4l0);
    HVX_VectorPair ws32_sum_p3n3 = Q6_Ww_vadd_VuhVuh(vu16_src_p3l0, vu16_src_n3l0);
    HVX_VectorPair ws32_sum_p2n2 = Q6_Ww_vadd_VuhVuh(vu16_src_p2l0, vu16_src_n2l0);
    HVX_VectorPair ws32_sum_p1n1 = Q6_Ww_vadd_VuhVuh(vu16_src_p1l0, vu16_src_n1l0);
    ws32_sum0 = Q6_Ww_vadd_WwWw(Q6_Ww_vadd_WwWw(ws32_sum_p4n4, ws32_sum_p3n3), Q6_Ww_vadd_WwWw(ws32_sum_p2n2, ws32_sum_p1n1));
    ws32_sum0 = Q6_Ww_vadd_WwWw(ws32_sum0, Q6_Ww_vadd_VuhVuh(vu16_src_p0l0, vu16_src_n0l0));
    ws32_sum0 = Q6_Wuw_vmpyacc_WuwVuhRuh(ws32_sum0, vu16_src_cl0, 0x00010001);


    ws32_sum_p4n4 = Q6_Ww_vadd_VuhVuh(vu16_src_p4c, vu16_src_n4c);
    ws32_sum_p3n3 = Q6_Ww_vadd_VuhVuh(vu16_src_p3c, vu16_src_n3c);
    ws32_sum_p2n2 = Q6_Ww_vadd_VuhVuh(vu16_src_p2c, vu16_src_n2c);
    ws32_sum_p1n1 = Q6_Ww_vadd_VuhVuh(vu16_src_p1c, vu16_src_n1c);
    ws32_sum1 = Q6_Ww_vadd_WwWw(Q6_Ww_vadd_WwWw(ws32_sum_p4n4, ws32_sum_p3n3), Q6_Ww_vadd_WwWw(ws32_sum_p2n2, ws32_sum_p1n1));
    ws32_sum1 = Q6_Ww_vadd_WwWw(ws32_sum1, Q6_Ww_vadd_VuhVuh(vu16_src_p0c, vu16_src_n0c));
    ws32_sum1 = Q6_Wuw_vmpyacc_WuwVuhRuh(ws32_sum1, vu16_src_cc, 0x00010001);

    vs32_sum1  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_sum1), Q6_V_hi_W(ws32_sum1));

    HVX_VectorPred vq_mask0 = Q6_Q_vsetq_R(8);

    HVX_Vector vs32_sum0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_sum0), Q6_V_hi_W(ws32_sum0));
    HVX_Vector vs32_sum  = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(vs32_sum0, -8), Q6_V_vror_VR(vs32_sum1, -8));

    vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum0, 4), vs32_sum);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowHCore(HVX_VectorPair &ws32_sum0, HVX_VectorPair &ws32_sum1,
                                                       HVX_VectorPair &ws32_sum2, HVX_Vector &vs32_com0, HVX_Vector &vs32_sum1,
                                                       HVX_Vector &vu16_result,
                                                       const SumType *row_sum, MI_S32 row_sum_step)
{
    HVX_VectorPred vq_mask0 = Q6_Q_vsetq_R(12);
    HVX_VectorPred vq_mask1 = Q6_Q_vsetq_R(8);
    HVX_VectorPred vq_mask2 = Q6_Q_vsetq_R(116);
    HVX_VectorPred vq_mask3 = Q6_Q_vsetq_R(120);

    HVX_Vector vs32_sum1_p = vs32_sum1;
    HVX_Vector vs32_sum0_p = vs32_com0;

    HVX_Vector vs32_sum;
    vs32_sum1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_sum2), Q6_V_hi_W(ws32_sum2));
    vs32_sum  = Q6_V_vmux_QVV(vq_mask1, Q6_V_vror_VR(vs32_sum1_p, -8), Q6_V_vror_VR(vs32_sum1, -8));
    vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum1_p, 4), vs32_sum);

    HVX_Vector vs32_com3 = Q6_V_vmux_QVV(vq_mask3, Q6_V_vror_VR(vs32_sum1_p, 8), Q6_V_vror_VR(vs32_sum1, 8));
    HVX_Vector vs32_com2 = Q6_V_vmux_QVV(vq_mask3, Q6_V_vror_VR(vs32_sum0_p, 8), Q6_V_vror_VR(vs32_com0, 8));

    HVX_Vector vs32_even = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(Q6_V_hi_W(ws32_sum0), -12), Q6_V_vror_VR(Q6_V_hi_W(ws32_sum1), -12));
    HVX_Vector vs32_odd  = Q6_V_vmux_QVV(vq_mask2, Q6_V_vror_VR(Q6_V_lo_W(ws32_sum1),  12), Q6_V_vror_VR(Q6_V_lo_W(ws32_sum2),  12));
    HVX_Vector vs32_com  = Q6_Vw_vadd_VwVw(vs32_com2, Q6_Vw_vadd_VwVw(vs32_sum0_p, vs32_com3));

    HVX_Vector *row_sum_data_out = (HVX_Vector*)(row_sum + row_sum_step);

    HVX_Vector vs32_sum_even = Q6_Vw_vadd_VwVw(vs32_com, vs32_even);
    HVX_Vector vs32_sum_odd  = Q6_Vw_vadd_VwVw(vs32_com, vs32_odd);

    *row_sum_data_out++ = vs32_sum_even;
    *row_sum_data_out++ = vs32_sum_odd;

    HVX_Vector vu32_dst_odd  = vdiv_n<MI_U32, 121>(vs32_sum_odd);
    HVX_Vector vu32_dst_even = vdiv_n<MI_U32, 121>(vs32_sum_even);

    vu16_result = Q6_Vuh_vsat_VuwVuw(vu32_dst_odd, vu32_dst_even);
}

// using Tp = MI_S16
template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11VCore(const HVX_Vector &vs16_src_p1, const HVX_Vector &vs16_src_n1, HVX_VectorPair &ws32_result)
{
    ws32_result = Q6_Ww_vsub_VhVh(vs16_src_n1, vs16_src_p1);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11VCore(const HVX_Vector &vs16_src_p0, const HVX_Vector &vs16_src_n0,
                                               const HVX_Vector &vs16_src_p1, const HVX_Vector &vs16_src_n1,
                                               HVX_VectorPair &ws32_diff0,  HVX_VectorPair &ws32_diff1,
                                               HVX_Vector &vs32_com0, HVX_Vector &vs32_sum1)
{
    ws32_diff0 = Q6_Ww_vsub_VhVh(vs16_src_n0, vs16_src_p0);
    ws32_diff1 = Q6_Ww_vsub_VhVh(vs16_src_n1, vs16_src_p1);
    vs32_sum1  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_diff1), Q6_V_hi_W(ws32_diff1));

    HVX_VectorPred vq_mask0 = Q6_Q_vsetq_R(8);

    HVX_Vector vs32_sum0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_diff0), Q6_V_hi_W(ws32_diff0));
    HVX_Vector vs32_sum  = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(vs32_sum0, -8), Q6_V_vror_VR(vs32_sum1, -8));

    vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum0, 4), vs32_sum);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11HCore(HVX_VectorPair &ws32_diff0, HVX_VectorPair &ws32_diff1,
                                               HVX_VectorPair &ws32_diff2, HVX_Vector &vs32_com0, HVX_Vector &vs32_sum1,
                                               HVX_Vector &vs16_result,
                                               const SumType *row_sum, MI_S32 row_sum_step)
{
    HVX_VectorPred vq_mask0 = Q6_Q_vsetq_R(12);
    HVX_VectorPred vq_mask1 = Q6_Q_vsetq_R(8);
    HVX_VectorPred vq_mask2 = Q6_Q_vsetq_R(116);
    HVX_VectorPred vq_mask3 = Q6_Q_vsetq_R(120);

    HVX_Vector vs32_sum1_p = vs32_sum1;
    HVX_Vector vs32_sum0_p = vs32_com0;

    HVX_Vector vs32_sum;
    vs32_sum1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_diff2), Q6_V_hi_W(ws32_diff2));
    vs32_sum  = Q6_V_vmux_QVV(vq_mask1, Q6_V_vror_VR(vs32_sum1_p, -8), Q6_V_vror_VR(vs32_sum1, -8));
    vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum1_p, 4), vs32_sum);

    HVX_Vector vs32_com3 = Q6_V_vmux_QVV(vq_mask3, Q6_V_vror_VR(vs32_sum1_p, 8), Q6_V_vror_VR(vs32_sum1, 8));
    HVX_Vector vs32_com2 = Q6_V_vmux_QVV(vq_mask3, Q6_V_vror_VR(vs32_sum0_p, 8), Q6_V_vror_VR(vs32_com0, 8));

    HVX_Vector vs32_even = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(Q6_V_hi_W(ws32_diff0), -12), Q6_V_vror_VR(Q6_V_hi_W(ws32_diff1), -12));
    HVX_Vector vs32_odd  = Q6_V_vmux_QVV(vq_mask2, Q6_V_vror_VR(Q6_V_lo_W(ws32_diff1),  12), Q6_V_vror_VR(Q6_V_lo_W(ws32_diff2),  12));
    HVX_Vector vs32_com  = Q6_Vw_vadd_VwVw(vs32_com2, Q6_Vw_vadd_VwVw(vs32_sum0_p, vs32_com3));

    HVX_Vector *row_sum_data_in  = (HVX_Vector*)(row_sum + row_sum_step);
    HVX_Vector *row_sum_data_out = row_sum_data_in;

    HVX_Vector vs32_sum_even = *row_sum_data_in++;
    HVX_Vector vs32_sum_odd  = *row_sum_data_in++;

    vs32_sum_even = Q6_Vw_vadd_VwVw(vs32_sum_even, Q6_Vw_vadd_VwVw(vs32_com, vs32_even));
    vs32_sum_odd  = Q6_Vw_vadd_VwVw(vs32_sum_odd,  Q6_Vw_vadd_VwVw(vs32_com, vs32_odd));

    *row_sum_data_out++ = vs32_sum_even;
    *row_sum_data_out++ = vs32_sum_odd;

    HVX_Vector vs32_dst_odd  = vdiv_n<MI_S32, 121>(vs32_sum_odd);
    HVX_Vector vs32_dst_even = vdiv_n<MI_S32, 121>(vs32_sum_even);

    vs16_result = Q6_Vh_vsat_VwVw(vs32_dst_odd, vs32_dst_even);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowVCore(const HVX_Vector &vs16_src_p4, const HVX_Vector &vs16_src_p3, const HVX_Vector &vs16_src_p2,
                                                       const HVX_Vector &vs16_src_p1, const HVX_Vector &vs16_src_p0, const HVX_Vector &vs16_src_c,
                                                       const HVX_Vector &vs16_src_n0, const HVX_Vector &vs16_src_n1, const HVX_Vector &vs16_src_n2,
                                                       const HVX_Vector &vs16_src_n3, const HVX_Vector &vs16_src_n4, HVX_VectorPair &ws32_result)
{
    HVX_VectorPair ws32_sum_p4n4 = Q6_Ww_vadd_VhVh(vs16_src_p4, vs16_src_n4);
    HVX_VectorPair ws32_sum_p3n3 = Q6_Ww_vadd_VhVh(vs16_src_p3, vs16_src_n3);
    HVX_VectorPair ws32_sum_p2n2 = Q6_Ww_vadd_VhVh(vs16_src_p2, vs16_src_n2);
    HVX_VectorPair ws32_sum_p1n1 = Q6_Ww_vadd_VhVh(vs16_src_p1, vs16_src_n1);

    HVX_VectorPair ws32_sum = Q6_Ww_vadd_WwWw(Q6_Ww_vadd_WwWw(ws32_sum_p4n4, ws32_sum_p3n3), Q6_Ww_vadd_WwWw(ws32_sum_p2n2, ws32_sum_p1n1));

    ws32_result = Q6_Ww_vadd_VhVh(vs16_src_p0, vs16_src_n0);
    ws32_result = Q6_Ww_vadd_WwWw(ws32_result, ws32_sum);
    ws32_result = Q6_Ww_vmpyacc_WwVhRh(ws32_result, vs16_src_c, 0x00010001);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowVCore(
                                                       const HVX_Vector &vs16_src_p4l0, const HVX_Vector &vs16_src_p3l0, const HVX_Vector &vs16_src_p2l0,
                                                       const HVX_Vector &vs16_src_p1l0, const HVX_Vector &vs16_src_p0l0, const HVX_Vector &vs16_src_cl0,
                                                       const HVX_Vector &vs16_src_n0l0, const HVX_Vector &vs16_src_n1l0, const HVX_Vector &vs16_src_n2l0,
                                                       const HVX_Vector &vs16_src_n3l0, const HVX_Vector &vs16_src_n4l0, const HVX_Vector &vs16_src_p4c,
                                                       const HVX_Vector &vs16_src_p3c,  const HVX_Vector &vs16_src_p2c,  const HVX_Vector &vs16_src_p1c,
                                                       const HVX_Vector &vs16_src_p0c,  const HVX_Vector &vs16_src_cc,   const HVX_Vector &vs16_src_n0c,
                                                       const HVX_Vector &vs16_src_n1c,  const HVX_Vector &vs16_src_n2c,  const HVX_Vector &vs16_src_n3c,
                                                       const HVX_Vector &vs16_src_n4c, HVX_VectorPair &ws32_sum0,  HVX_VectorPair &ws32_sum1,
                                                       HVX_Vector &vs32_com0, HVX_Vector &vs32_sum1)
{
    HVX_VectorPair ws32_sum_p4n4 = Q6_Ww_vadd_VhVh(vs16_src_p4l0, vs16_src_n4l0);
    HVX_VectorPair ws32_sum_p3n3 = Q6_Ww_vadd_VhVh(vs16_src_p3l0, vs16_src_n3l0);
    HVX_VectorPair ws32_sum_p2n2 = Q6_Ww_vadd_VhVh(vs16_src_p2l0, vs16_src_n2l0);
    HVX_VectorPair ws32_sum_p1n1 = Q6_Ww_vadd_VhVh(vs16_src_p1l0, vs16_src_n1l0);

    ws32_sum0 = Q6_Ww_vadd_WwWw(Q6_Ww_vadd_WwWw(ws32_sum_p4n4, ws32_sum_p3n3), Q6_Ww_vadd_WwWw(ws32_sum_p2n2, ws32_sum_p1n1));
    ws32_sum0 = Q6_Ww_vadd_WwWw(ws32_sum0, Q6_Ww_vadd_VhVh(vs16_src_p0l0, vs16_src_n0l0));
    ws32_sum0 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum0, vs16_src_cl0, 0x00010001);

    ws32_sum_p4n4 = Q6_Ww_vadd_VhVh(vs16_src_p4c, vs16_src_n4c);
    ws32_sum_p3n3 = Q6_Ww_vadd_VhVh(vs16_src_p3c, vs16_src_n3c);
    ws32_sum_p2n2 = Q6_Ww_vadd_VhVh(vs16_src_p2c, vs16_src_n2c);
    ws32_sum_p1n1 = Q6_Ww_vadd_VhVh(vs16_src_p1c, vs16_src_n1c);

    ws32_sum1 = Q6_Ww_vadd_WwWw(Q6_Ww_vadd_WwWw(ws32_sum_p4n4, ws32_sum_p3n3), Q6_Ww_vadd_WwWw(ws32_sum_p2n2, ws32_sum_p1n1));
    ws32_sum1 = Q6_Ww_vadd_WwWw(ws32_sum1, Q6_Ww_vadd_VhVh(vs16_src_p0c, vs16_src_n0c));
    ws32_sum1 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum1, vs16_src_cc, 0x00010001);

    vs32_sum1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_sum1), Q6_V_hi_W(ws32_sum1));

    HVX_VectorPred vq_mask0 = Q6_Q_vsetq_R(8);

    HVX_Vector vs32_sum0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_sum0), Q6_V_hi_W(ws32_sum0));
    HVX_Vector vs32_sum  = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(vs32_sum0, -8), Q6_V_vror_VR(vs32_sum1, -8));

    vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum0, 4), vs32_sum);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowHCore(HVX_VectorPair &ws32_sum0, HVX_VectorPair &ws32_sum1,
                                                       HVX_VectorPair &ws32_sum2, HVX_Vector &vs32_com0, HVX_Vector &vs32_sum1,
                                                       HVX_Vector &vs16_result,
                                                       const SumType *row_sum, MI_S32 row_sum_step)
{
    HVX_VectorPred vq_mask0 = Q6_Q_vsetq_R(12);
    HVX_VectorPred vq_mask1 = Q6_Q_vsetq_R(8);
    HVX_VectorPred vq_mask2 = Q6_Q_vsetq_R(116);
    HVX_VectorPred vq_mask3 = Q6_Q_vsetq_R(120);

    HVX_Vector vs32_sum1_p = vs32_sum1;
    HVX_Vector vs32_sum0_p = vs32_com0;

    HVX_Vector vs32_sum;
    vs32_sum1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_sum2), Q6_V_hi_W(ws32_sum2));
    vs32_sum  = Q6_V_vmux_QVV(vq_mask1, Q6_V_vror_VR(vs32_sum1_p, -8), Q6_V_vror_VR(vs32_sum1, -8));
    vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum1_p, 4), vs32_sum);

    HVX_Vector vs32_com3 = Q6_V_vmux_QVV(vq_mask3, Q6_V_vror_VR(vs32_sum1_p, 8), Q6_V_vror_VR(vs32_sum1, 8));
    HVX_Vector vs32_com2 = Q6_V_vmux_QVV(vq_mask3, Q6_V_vror_VR(vs32_sum0_p, 8), Q6_V_vror_VR(vs32_com0, 8));

    HVX_Vector vs32_even = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(Q6_V_hi_W(ws32_sum0), -12), Q6_V_vror_VR(Q6_V_hi_W(ws32_sum1), -12));
    HVX_Vector vs32_odd  = Q6_V_vmux_QVV(vq_mask2, Q6_V_vror_VR(Q6_V_lo_W(ws32_sum1),  12), Q6_V_vror_VR(Q6_V_lo_W(ws32_sum2),  12));
    HVX_Vector vs32_com  = Q6_Vw_vadd_VwVw(vs32_com2, Q6_Vw_vadd_VwVw(vs32_sum0_p, vs32_com3));

    HVX_Vector *row_sum_data_out = (HVX_Vector*)(row_sum + row_sum_step);

    HVX_Vector vs32_sum_even = Q6_Vw_vadd_VwVw(vs32_com, vs32_even);
    HVX_Vector vs32_sum_odd  = Q6_Vw_vadd_VwVw(vs32_com, vs32_odd);

    *row_sum_data_out++ = vs32_sum_even;
    *row_sum_data_out++ = vs32_sum_odd;

    HVX_Vector vs32_dst_odd  = vdiv_n<MI_S32, 121>(vs32_sum_odd);
    HVX_Vector vs32_dst_even = vdiv_n<MI_S32, 121>(vs32_sum_even);

    vs16_result = Q6_Vh_vsat_VwVw(vs32_dst_odd, vs32_dst_even);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_S16>::value || std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11HCore(HVX_VectorPair &ws32_diff_x0, HVX_VectorPair &ws32_diff_x1,
                                               HVX_VectorPair &ws32_diff_x2, HVX_VectorPair &ws32_diff_x3,
                                               HVX_Vector &vd16_result_x0,   HVX_Vector &vd16_result_x1,
                                               SumType *row_sum, MI_S32 row_sum_step, MI_S32 rest)
{
    HVX_Vector vs32_diff_x0_lo = Q6_V_lo_W(ws32_diff_x0);
    HVX_Vector vs32_diff_x0_hi = Q6_V_hi_W(ws32_diff_x0);
    HVX_Vector vs32_diff_x1_lo = Q6_V_lo_W(ws32_diff_x1);
    HVX_Vector vs32_diff_x1_hi = Q6_V_hi_W(ws32_diff_x1);
    HVX_Vector vs32_diff_x2_lo = Q6_V_lo_W(ws32_diff_x2);
    HVX_Vector vs32_diff_x2_hi = Q6_V_hi_W(ws32_diff_x2);
    HVX_Vector vs32_diff_x3_lo = Q6_V_lo_W(ws32_diff_x3);
    HVX_Vector vs32_diff_x3_hi = Q6_V_hi_W(ws32_diff_x3);

    HVX_Vector vs32_diff_l0_lo, vs32_diff_l0_hi, vs32_diff_r0_lo, vs32_diff_r0_hi;
    if (rest & 1)
    {
        MI_S32 align_size0 = (rest / 2) * sizeof(SumType);
        MI_S32 align_size1 = align_size0 + sizeof(SumType);
        vs32_diff_r0_lo = Q6_V_vlalign_safe_VVR(vs32_diff_x3_hi, vs32_diff_x2_hi, align_size1);
        vs32_diff_r0_hi = Q6_V_vlalign_safe_VVR(vs32_diff_x3_lo, vs32_diff_x2_lo, align_size0);
        vs32_diff_l0_lo = Q6_V_valign_safe_VVR(vs32_diff_x1_hi, vs32_diff_x0_hi, align_size0);
        vs32_diff_l0_hi = Q6_V_valign_safe_VVR(vs32_diff_x1_lo, vs32_diff_x0_lo, align_size1);
    }
    else
    {
        MI_S32 align_size = (rest / 2) * sizeof(SumType);
        vs32_diff_r0_lo = Q6_V_vlalign_safe_VVR(vs32_diff_x3_lo, vs32_diff_x2_lo, align_size);
        vs32_diff_r0_hi = Q6_V_vlalign_safe_VVR(vs32_diff_x3_hi, vs32_diff_x2_hi, align_size);
        vs32_diff_l0_lo = Q6_V_valign_safe_VVR(vs32_diff_x1_lo, vs32_diff_x0_lo, align_size);
        vs32_diff_l0_hi = Q6_V_valign_safe_VVR(vs32_diff_x1_hi, vs32_diff_x0_hi, align_size);
    }

    HVX_VectorPair ws32_diff_r0 = Q6_W_vcombine_VV(vs32_diff_r0_hi, vs32_diff_r0_lo);
    HVX_VectorPair ws32_diff_l0 = Q6_W_vcombine_VV(vs32_diff_l0_hi, vs32_diff_l0_lo);

    HVX_VectorPred vq_mask0 = Q6_Q_vsetq_R(8);

    HVX_Vector vs32_sum0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_diff_x0), Q6_V_hi_W(ws32_diff_x0));
    HVX_Vector vs32_sum1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_diff_x1), Q6_V_hi_W(ws32_diff_x1));
    HVX_Vector vs32_sum  = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(vs32_sum0, -8), Q6_V_vror_VR(vs32_sum1, -8));
    HVX_Vector vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum0, 4), vs32_sum);

    BoxFilter11x11HCore<Tp, SumType>(ws32_diff_x0, ws32_diff_x1, ws32_diff_r0, vs32_com0, vs32_sum1, vd16_result_x0, row_sum, row_sum_step);

    vs32_sum0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_diff_l0), Q6_V_hi_W(ws32_diff_l0));
    vs32_sum1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_diff_x2), Q6_V_hi_W(ws32_diff_x2));
    vs32_sum  = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(vs32_sum0, -8), Q6_V_vror_VR(vs32_sum1, -8));
    vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum0, 4), vs32_sum);

    BoxFilter11x11HCore<Tp, SumType>(ws32_diff_l0, ws32_diff_x2, ws32_diff_x3, vs32_com0, vs32_sum1, vd16_result_x1, row_sum, row_sum_step + 128);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, MI_S16>::value || std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID BoxFilter11x11FirstRowHCore(HVX_VectorPair &wd32_sum_x0, HVX_VectorPair &wd32_sum_x1,
                                                       HVX_VectorPair &wd32_sum_x2, HVX_VectorPair &wd32_sum_x3,
                                                       HVX_Vector &vd16_result_x0, HVX_Vector &vd16_result_x1,
                                                       SumType *row_sum, MI_S32 row_sum_step, MI_S32 rest)
{
    HVX_Vector vd32_sum_x0_lo = Q6_V_lo_W(wd32_sum_x0);
    HVX_Vector vd32_sum_x0_hi = Q6_V_hi_W(wd32_sum_x0);
    HVX_Vector vd32_sum_x1_lo = Q6_V_lo_W(wd32_sum_x1);
    HVX_Vector vd32_sum_x1_hi = Q6_V_hi_W(wd32_sum_x1);
    HVX_Vector vd32_sum_x2_lo = Q6_V_lo_W(wd32_sum_x2);
    HVX_Vector vd32_sum_x2_hi = Q6_V_hi_W(wd32_sum_x2);
    HVX_Vector vd32_sum_x3_lo = Q6_V_lo_W(wd32_sum_x3);
    HVX_Vector vd32_sum_x3_hi = Q6_V_hi_W(wd32_sum_x3);

    HVX_Vector vd32_sum_l0_lo, vd32_sum_l0_hi, vd32_sum_r0_lo, vd32_sum_r0_hi;
    if (rest & 1)
    {
        MI_S32 align_size0 = (rest / 2) * sizeof(SumType);
        MI_S32 align_size1 = align_size0 + sizeof(SumType);
        vd32_sum_r0_lo = Q6_V_vlalign_safe_VVR(vd32_sum_x3_hi, vd32_sum_x2_hi, align_size1);
        vd32_sum_r0_hi = Q6_V_vlalign_safe_VVR(vd32_sum_x3_lo, vd32_sum_x2_lo, align_size0);
        vd32_sum_l0_lo = Q6_V_valign_safe_VVR(vd32_sum_x1_hi, vd32_sum_x0_hi, align_size0);
        vd32_sum_l0_hi = Q6_V_valign_safe_VVR(vd32_sum_x1_lo, vd32_sum_x0_lo, align_size1);
    }
    else
    {
        MI_S32 align_size = (rest / 2) * sizeof(SumType);
        vd32_sum_r0_lo = Q6_V_vlalign_safe_VVR(vd32_sum_x3_lo, vd32_sum_x2_lo, align_size);
        vd32_sum_r0_hi = Q6_V_vlalign_safe_VVR(vd32_sum_x3_hi, vd32_sum_x2_hi, align_size);
        vd32_sum_l0_lo = Q6_V_valign_safe_VVR(vd32_sum_x1_lo, vd32_sum_x0_lo, align_size);
        vd32_sum_l0_hi = Q6_V_valign_safe_VVR(vd32_sum_x1_hi, vd32_sum_x0_hi, align_size);
    }

    HVX_VectorPair wd32_sum_r0 = Q6_W_vcombine_VV(vd32_sum_r0_hi, vd32_sum_r0_lo);
    HVX_VectorPair wd32_sum_l0 = Q6_W_vcombine_VV(vd32_sum_l0_hi, vd32_sum_l0_lo);

    HVX_VectorPred vq_mask0 = Q6_Q_vsetq_R(8);

    HVX_Vector vs32_sum0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wd32_sum_x0), Q6_V_hi_W(wd32_sum_x0));
    HVX_Vector vs32_sum1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wd32_sum_x1), Q6_V_hi_W(wd32_sum_x1));
    HVX_Vector vs32_sum  = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(vs32_sum0, -8), Q6_V_vror_VR(vs32_sum1, -8));
    HVX_Vector vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum0, 4), vs32_sum);

    BoxFilter11x11FirstRowHCore<Tp, SumType>(wd32_sum_x0, wd32_sum_x1, wd32_sum_r0, vs32_com0, vs32_sum1, vd16_result_x0, row_sum, row_sum_step);

    vs32_sum0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wd32_sum_l0), Q6_V_hi_W(wd32_sum_l0));
    vs32_sum1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wd32_sum_x2), Q6_V_hi_W(wd32_sum_x2));
    vs32_sum  = Q6_V_vmux_QVV(vq_mask0, Q6_V_vror_VR(vs32_sum0, -8), Q6_V_vror_VR(vs32_sum1, -8));
    vs32_com0 = Q6_Vw_vadd_VwVw(Q6_V_vlalign_VVI(vs32_sum1, vs32_sum0, 4), vs32_sum);

    BoxFilter11x11FirstRowHCore<Tp, SumType>(wd32_sum_l0, wd32_sum_x2, wd32_sum_x3, vs32_com0, vs32_sum1, vd16_result_x1, row_sum, row_sum_step + 128);
}

template <typename Tp, typename SumType, BorderType BORDER_TYPE, MI_S32 C>
static AURA_VOID BoxFilter11x11Row(const Tp *src_p0, const Tp *src_n0, Tp *dst, MI_S32 width, SumType *row_sum, MI_S32 row_sum_step,
                                 const std::vector<Tp> &border_value)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    MI_S32 shift = 0;
    constexpr MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);
    MI_S32 back_offset = width - ELEM_COUNTS;

    MVType mv_src_p0, mv_src_n0;
    MWType mw_diff0, mw_diff1, mw_diff2;
    MVType mv_com0, mv_sum1;
    MVType mv_result;

    // left border
    {
        vload(src_p0, mv_src_p0);
        vload(src_n0, mv_src_n0);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p0.val[ch], src_p0[ch], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n0.val[ch], src_n0[ch], border_value[ch]);

            BoxFilter11x11VCore<Tp, SumType>(v_border_p1, v_border_n1, mv_src_p0.val[ch], mv_src_n0.val[ch],
                                             mw_diff0.val[ch], mw_diff1.val[ch], mv_com0.val[ch], mv_sum1.val[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (MI_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS, shift += ELEM_COUNTS)
        {
            vload(src_p0 + C * x, mv_src_p0);
            vload(src_n0 + C * x, mv_src_n0);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                BoxFilter11x11VCore<Tp, SumType>(mv_src_p0.val[ch], mv_src_n0.val[ch], mw_diff2.val[ch]);
                BoxFilter11x11HCore<Tp, SumType>(mw_diff0.val[ch], mw_diff1.val[ch], mw_diff2.val[ch], mv_com0.val[ch], mv_sum1.val[ch],
                                                 mv_result.val[ch], row_sum, row_sum_step * ch + shift);
            }

            vstore(dst + C * (x - ELEM_COUNTS), mv_result);

            mw_diff0 = mw_diff1;
            mw_diff1 = mw_diff2;
        }
    }

    // remain
    {
        MI_S32 last = C * (width - 1);
        MI_S32 rest = width % ELEM_COUNTS;
        MVType mv_last;

        vload(src_p0 + C * back_offset, mv_src_p0);
        vload(src_n0 + C * back_offset, mv_src_n0);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0.val[ch], src_p0[last + ch], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0.val[ch], src_n0[last + ch], border_value[ch]);

            HVX_VectorPair w_diff2, w_diff3;
            BoxFilter11x11VCore<Tp, SumType>(mv_src_p0.val[ch], mv_src_n0.val[ch], w_diff2);
            BoxFilter11x11VCore<Tp, SumType>(v_border_p1, v_border_n1, w_diff3);

            BoxFilter11x11HCore<Tp, SumType>(mw_diff0.val[ch], mw_diff1.val[ch], w_diff2, w_diff3,
                                             mv_result.val[ch], mv_last.val[ch], row_sum, row_sum_step * ch + shift, rest);
        }

        vstore(dst + C * (back_offset - rest), mv_result);
        vstore(dst + C * back_offset, mv_last);
    }
}

template <typename Tp, typename SumType, BorderType BORDER_TYPE, MI_S32 C>
static AURA_VOID BoxFilter11x11FirstRow(const Tp *src_p4, const Tp *src_p3, const Tp *src_p2, const Tp *src_p1, const Tp *src_p0, const Tp *src_c,
                                      const Tp *src_n0, const Tp *src_n1, const Tp *src_n2, const Tp *src_n3, const Tp *src_n4, Tp *dst,
                                      MI_S32 width, SumType *row_sum, MI_S32 row_sum_step, const std::vector<Tp> &border_value)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    MI_S32 shift = 0;
    MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);
    MI_S32 back_offset = width - ELEM_COUNTS;

    MVType mv_src_p4, mv_src_p3, mv_src_p2, mv_src_p1, mv_src_p0, mv_src_c, mv_src_n0, mv_src_n1, mv_src_n2, mv_src_n3, mv_src_n4;
    MWType mw_sum0, mw_sum1, mw_sum2;
    MVType mv_com0, mv_sum1;
    MVType mv_result;

    // left border
    {
        vload(src_p4, mv_src_p4);
        vload(src_p3, mv_src_p3);
        vload(src_p2, mv_src_p2);
        vload(src_p1, mv_src_p1);
        vload(src_p0, mv_src_p0);
        vload(src_c,  mv_src_c);
        vload(src_n0, mv_src_n0);
        vload(src_n1, mv_src_n1);
        vload(src_n2, mv_src_n2);
        vload(src_n3, mv_src_n3);
        vload(src_n4, mv_src_n4);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p4 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p4.val[ch], src_p4[ch], border_value[ch]);
            HVX_Vector v_border_p3 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p3.val[ch], src_p3[ch], border_value[ch]);
            HVX_Vector v_border_p2 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p2.val[ch], src_p2[ch], border_value[ch]);
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p1.val[ch], src_p1[ch], border_value[ch]);
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p0.val[ch], src_p0[ch], border_value[ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_c.val[ch],  src_c[ch],  border_value[ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n0.val[ch], src_n0[ch], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n1.val[ch], src_n1[ch], border_value[ch]);
            HVX_Vector v_border_n2 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n2.val[ch], src_n2[ch], border_value[ch]);
            HVX_Vector v_border_n3 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n3.val[ch], src_n3[ch], border_value[ch]);
            HVX_Vector v_border_n4 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n4.val[ch], src_n4[ch], border_value[ch]);

            BoxFilter11x11FirstRowVCore<Tp, SumType>(v_border_p4, v_border_p3, v_border_p2, v_border_p1, v_border_p0, v_border_c, v_border_n0,
                                                     v_border_n1, v_border_n2, v_border_n3, v_border_n4, mv_src_p4.val[ch], mv_src_p3.val[ch],
                                                     mv_src_p2.val[ch], mv_src_p1.val[ch], mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch],
                                                     mv_src_n1.val[ch], mv_src_n2.val[ch], mv_src_n3.val[ch], mv_src_n4.val[ch],
                                                     mw_sum0.val[ch], mw_sum1.val[ch], mv_com0.val[ch], mv_sum1.val[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (MI_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS, shift += ELEM_COUNTS)
        {
            vload(src_p4 + C * x, mv_src_p4);
            vload(src_p3 + C * x, mv_src_p3);
            vload(src_p2 + C * x, mv_src_p2);
            vload(src_p1 + C * x, mv_src_p1);
            vload(src_p0 + C * x, mv_src_p0);
            vload(src_c  + C * x, mv_src_c);
            vload(src_n0 + C * x, mv_src_n0);
            vload(src_n1 + C * x, mv_src_n1);
            vload(src_n2 + C * x, mv_src_n2);
            vload(src_n3 + C * x, mv_src_n3);
            vload(src_n4 + C * x, mv_src_n4);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                BoxFilter11x11FirstRowVCore<Tp, SumType>(mv_src_p4.val[ch], mv_src_p3.val[ch], mv_src_p2.val[ch], mv_src_p1.val[ch],
                                                         mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mv_src_n1.val[ch], mv_src_n2.val[ch],
                                                         mv_src_n3.val[ch], mv_src_n4.val[ch], mw_sum2.val[ch]);
                BoxFilter11x11FirstRowHCore<Tp, SumType>(mw_sum0.val[ch], mw_sum1.val[ch], mw_sum2.val[ch], mv_com0.val[ch], mv_sum1.val[ch],
                                                         mv_result.val[ch], row_sum, row_sum_step * ch + shift);
            }

            vstore(dst + C * (x - ELEM_COUNTS), mv_result);

            mw_sum0 = mw_sum1;
            mw_sum1 = mw_sum2;
        }
    }

    // remain
    {
        MI_S32 last = C * (width - 1);
        MI_S32 rest = width % ELEM_COUNTS;
        MVType mv_last;

        vload(src_p4 + C * back_offset, mv_src_p4);
        vload(src_p3 + C * back_offset, mv_src_p3);
        vload(src_p2 + C * back_offset, mv_src_p2);
        vload(src_p1 + C * back_offset, mv_src_p1);
        vload(src_p0 + C * back_offset, mv_src_p0);
        vload(src_c  + C * back_offset, mv_src_c);
        vload(src_n0 + C * back_offset, mv_src_n0);
        vload(src_n1 + C * back_offset, mv_src_n1);
        vload(src_n2 + C * back_offset, mv_src_n2);
        vload(src_n3 + C * back_offset, mv_src_n3);
        vload(src_n4 + C * back_offset, mv_src_n4);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p4 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p4.val[ch], src_p4[last + ch], border_value[ch]);
            HVX_Vector v_border_p3 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p3.val[ch], src_p3[last + ch], border_value[ch]);
            HVX_Vector v_border_p2 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p2.val[ch], src_p2[last + ch], border_value[ch]);
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p1.val[ch], src_p1[last + ch], border_value[ch]);
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0.val[ch], src_p0[last + ch], border_value[ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_c.val[ch],  src_c[last + ch],  border_value[ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0.val[ch], src_n0[last + ch], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n1.val[ch], src_n1[last + ch], border_value[ch]);
            HVX_Vector v_border_n2 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n2.val[ch], src_n2[last + ch], border_value[ch]);
            HVX_Vector v_border_n3 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n3.val[ch], src_n3[last + ch], border_value[ch]);
            HVX_Vector v_border_n4 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n4.val[ch], src_n4[last + ch], border_value[ch]);

            HVX_VectorPair w_sum2, w_sum3;
            BoxFilter11x11FirstRowVCore<Tp, SumType>(mv_src_p4.val[ch], mv_src_p3.val[ch], mv_src_p2.val[ch], mv_src_p1.val[ch],
                                                     mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mv_src_n1.val[ch], mv_src_n2.val[ch],
                                                     mv_src_n3.val[ch], mv_src_n4.val[ch], w_sum2);
            BoxFilter11x11FirstRowVCore<Tp, SumType>(v_border_p4, v_border_p3, v_border_p2, v_border_p1, v_border_p0, v_border_c, v_border_n0,
                                                     v_border_n1, v_border_n2, v_border_n3, v_border_n4, w_sum3);

            BoxFilter11x11FirstRowHCore<Tp, SumType>(mw_sum0.val[ch], mw_sum1.val[ch], w_sum2, w_sum3,
                                                     mv_result.val[ch], mv_last.val[ch], row_sum, row_sum_step * ch + shift, rest);
        }

        vstore(dst + C * (back_offset - rest), mv_result);
        vstore(dst + C * back_offset, mv_last);
    }
}

template <typename Tp, typename SumType, BorderType BORDER_TYPE, MI_S32 C>
static Status BoxFilter11x11HvxImpl(Context *ctx, const Mat &src, Mat &dst, SumType *row_sum_buffer, const std::vector<Tp> &border_value,
                                    const Tp *border_buffer, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width  = src.GetSizes().m_width;
    MI_S32 height = src.GetSizes().m_height;
    MI_S32 stride = src.GetStrides().m_width;
    MI_S32 channel = src.GetSizes().m_channel;

    MI_S32 thread_idx = ctx->GetWorkerPool()->GetComputeThreadIdx();

    MI_S32 sum_size = (AURA_ALIGN(width, 128) + AURA_HVLEN);
    SumType *row_sum = row_sum_buffer + thread_idx * sum_size * channel;
    if (NULL == row_sum)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
        return Status::ERROR;
    }

    memset(row_sum, 0, sum_size * sizeof(SumType) * C);

    Tp *dst_row      = dst.Ptr<Tp>(start_row);
    const Tp *src_p4 = src.Ptr<Tp, BORDER_TYPE>(start_row - 5, border_buffer);
    const Tp *src_p3 = src.Ptr<Tp, BORDER_TYPE>(start_row - 4, border_buffer);
    const Tp *src_p2 = src.Ptr<Tp, BORDER_TYPE>(start_row - 3, border_buffer);
    const Tp *src_p1 = src.Ptr<Tp, BORDER_TYPE>(start_row - 2, border_buffer);
    const Tp *src_p0 = src.Ptr<Tp, BORDER_TYPE>(start_row - 1, border_buffer);
    const Tp *src_c  = src.Ptr<Tp>(start_row);
    const Tp *src_n0 = src.Ptr<Tp, BORDER_TYPE>(start_row + 1, border_buffer);
    const Tp *src_n1 = src.Ptr<Tp, BORDER_TYPE>(start_row + 2, border_buffer);
    const Tp *src_n2 = src.Ptr<Tp, BORDER_TYPE>(start_row + 3, border_buffer);
    const Tp *src_n3 = src.Ptr<Tp, BORDER_TYPE>(start_row + 4, border_buffer);
    const Tp *src_n4 = src.Ptr<Tp, BORDER_TYPE>(start_row + 5, border_buffer);
    BoxFilter11x11FirstRow<Tp, SumType, BORDER_TYPE, C>(src_p4, src_p3, src_p2, src_p1, src_p0, src_c, src_n0, src_n1, src_n2, src_n3, src_n4,
                                                          dst_row, width, row_sum, sum_size, border_value);

    const Tp *src_p5 = src_p4;
    src_p4 = src_p3;
    src_p3 = src_p2;
    src_p2 = src_p1;
    src_p1 = src_p0;
    src_p0 = src_c;
    src_c  = src_n0;
    src_n0 = src_n1;
    src_n1 = src_n2;
    src_n2 = src_n3;
    src_n3 = src_n4;
    src_n4 = src.Ptr<Tp, BORDER_TYPE>(start_row + 6, border_buffer);
    MI_U64 L2fetch_param = L2PfParam(stride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (MI_S32 y = start_row + 1; y < end_row; y++)
    {
        if (y + 6 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 6)), L2fetch_param);
        }

        Tp *dst_row = dst.Ptr<Tp>(y);
        BoxFilter11x11Row<Tp, SumType, BORDER_TYPE, C>(src_p5, src_n4, dst_row, width, row_sum, sum_size, border_value);

        src_p5 = src_p4;
        src_p4 = src_p3;
        src_p3 = src_p2;
        src_p2 = src_p1;
        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src_n2;
        src_n2 = src_n3;
        src_n3 = src_n4;
        src_n4 = src.Ptr<Tp, BORDER_TYPE>(y + 6, border_buffer);
    }

    return Status::OK;
}

template<typename Tp, typename SumType, BorderType BORDER_TYPE>
static Status BoxFilter11x11HvxHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<Tp> &border_value, const Tp *border_buffer)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    MI_S32 height   = src.GetSizes().m_height;
    MI_S32 width    = src.GetSizes().m_width;
    MI_S32 channel  = src.GetSizes().m_channel;
    MI_S32 sum_size = AURA_ALIGN(width, 128) + AURA_HVLEN;

    AURA_VOID *buffer_sum = AURA_ALLOC(ctx, wp->GetComputeThreadNum() * (sum_size * sizeof(SumType) * channel));
    if (MI_NULL == buffer_sum)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    SumType *buffer_sum_idx = reinterpret_cast<SumType*>(buffer_sum);

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((MI_S32)0, height, BoxFilter11x11HvxImpl<Tp, SumType, BORDER_TYPE, 1>,
                                  ctx, std::cref(src), std::ref(dst), std::ref(buffer_sum_idx), std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((MI_S32)0, height, BoxFilter11x11HvxImpl<Tp, SumType, BORDER_TYPE, 2>,
                                  ctx, std::cref(src), std::ref(dst), std::ref(buffer_sum_idx), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((MI_S32)0, height, BoxFilter11x11HvxImpl<Tp, SumType, BORDER_TYPE, 3>,
                                  ctx, std::cref(src), std::ref(dst), std::ref(buffer_sum_idx), std::cref(border_value), border_buffer);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel");
            break;
        }
    }

    AURA_FREE(ctx, buffer_sum);
    AURA_RETURN(ctx, ret);
}

template <typename Tp, typename SumType>
static Status BoxFilter11x11HvxHelper(Context *ctx, const Mat &src, Mat &dst, BorderType &border_type, const Scalar &border_value)
{
    Tp *border_buffer = MI_NULL;
    std::vector<Tp> vec_border_value = border_value.ToVector<Tp>();

    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 channel = dst.GetSizes().m_channel;

    Status ret = Status::ERROR;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            border_buffer = CreateBorderBuffer(ctx, width, channel, vec_border_value);
            if (MI_NULL == border_buffer)
            {
                AURA_ADD_ERROR_STRING(ctx, "CreateBorderBuffer failed");
                return Status::ERROR;
            }

            ret = BoxFilter11x11HvxHelper<Tp, SumType, BorderType::CONSTANT>(ctx, src, dst, vec_border_value, border_buffer);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = BoxFilter11x11HvxHelper<Tp, SumType, BorderType::REPLICATE>(ctx, src, dst, vec_border_value, border_buffer);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = BoxFilter11x11HvxHelper<Tp, SumType, BorderType::REFLECT_101>(ctx, src, dst, vec_border_value, border_buffer);
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

Status BoxFilter11x11Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = BoxFilter11x11HvxHelper<MI_U8, MI_S16>(ctx, src, dst, border_type, border_value);
            break;
        }

        case ElemType::U16:
        {
            ret = BoxFilter11x11HvxHelper<MI_U16, MI_S32>(ctx, src, dst, border_type, border_value);
            break;
        }

        case ElemType::S16:
        {
            ret = BoxFilter11x11HvxHelper<MI_S16, MI_S32>(ctx, src, dst, border_type, border_value);
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