#include "boxfilter_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/thread_buffer.h"

namespace aura
{

// using Tp = DT_U8
template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5VCore(const HVX_Vector &vu8_src_p0, const HVX_Vector &vu8_src_n0, HVX_VectorPair &ws16_result)
{
    ws16_result = Q6_Wh_vsub_VubVub(vu8_src_n0, vu8_src_p0);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5HCore(HVX_VectorPair &ws16_diff0, HVX_VectorPair &ws16_diff1, HVX_VectorPair &ws16_diff2,
                                             HVX_Vector &vu8_result, const SumType *row_sum, DT_S32 row_sum_step)
{
    HVX_Vector vs16_diff_l0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws16_diff1), Q6_V_hi_W(ws16_diff0), 2);
    HVX_Vector vs16_sum_common0 = Q6_Vh_vadd_VhVh(vs16_diff_l0, Q6_V_lo_W(ws16_diff1));
    HVX_Vector vs16_diff_l1     = Q6_V_vlalign_VVI(Q6_V_lo_W(ws16_diff1), Q6_V_lo_W(ws16_diff0), 2);

    HVX_Vector vs16_diff_r0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws16_diff2), Q6_V_hi_W(ws16_diff1), 2);
    HVX_Vector vs16_sum_common1 = Q6_Vh_vadd_VhVh(vs16_diff_r0, Q6_V_lo_W(ws16_diff2));
               vs16_sum_common1 = Q6_V_valign_VVI(vs16_sum_common1, vs16_sum_common0, 2);

    HVX_Vector vs16_diff_r1     = Q6_V_valign_VVI(Q6_V_hi_W(ws16_diff2), Q6_V_hi_W(ws16_diff1), 2);
    HVX_Vector vs16_sum_common  = Q6_Vh_vadd_VhVh(vs16_sum_common0, vs16_sum_common1);

    HVX_Vector *row_sum_data_in  = (HVX_Vector*)(row_sum + row_sum_step);
    HVX_Vector *row_sum_data_out = row_sum_data_in;

    HVX_Vector vs16_sum_even, vs16_sum_odd;
    vload(row_sum_data_in,   vs16_sum_even);
    vload(++row_sum_data_in, vs16_sum_odd);

    vs16_sum_even = Q6_Vh_vadd_VhVh(vs16_sum_even, Q6_Vh_vadd_VhVh(vs16_sum_common, vs16_diff_l1));
    vs16_sum_odd = Q6_Vh_vadd_VhVh(vs16_sum_odd, Q6_Vh_vadd_VhVh(vs16_sum_common, vs16_diff_r1));

    vstore(row_sum_data_out,   vs16_sum_even);
    vstore(++row_sum_data_out, vs16_sum_odd);

    HVX_Vector vu16_dst_odd  = vdiv_n<DT_U16, 25>(vs16_sum_odd);
    HVX_Vector vu16_dst_even = vdiv_n<DT_U16, 25>(vs16_sum_even);

    vu8_result = Q6_Vub_vsat_VhVh(vu16_dst_odd, vu16_dst_even);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5FirstRowVCore(const HVX_Vector &vu8_src_p1, const HVX_Vector &vu8_src_p0, const HVX_Vector &vu8_src_c,
                                                     const HVX_Vector &vu8_src_n0, const HVX_Vector &vu8_src_n1, HVX_VectorPair &ws16_result)
{
    HVX_VectorPair ws16_result1 = Q6_Wh_vadd_VubVub(vu8_src_p1, vu8_src_n1);
    ws16_result = Q6_Wh_vadd_VubVub(vu8_src_p0, vu8_src_n0);
    ws16_result = Q6_Wh_vadd_WhWh(ws16_result, ws16_result1);
    ws16_result = Q6_Wh_vmpyacc_WhVubRb(ws16_result, vu8_src_c, 0x01010101);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5FirstRowHCore(HVX_VectorPair &ws16_sum0, HVX_VectorPair &ws16_sum1, HVX_VectorPair &ws16_sum2,
                                                     HVX_Vector &vu8_result, const SumType *row_sum, DT_S32 row_sum_step)
{
    HVX_Vector vs16_diff_l0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws16_sum1), Q6_V_hi_W(ws16_sum0), 2);
    HVX_Vector vs16_sum_common0 = Q6_Vh_vadd_VhVh(vs16_diff_l0, Q6_V_lo_W(ws16_sum1));
    HVX_Vector vs16_diff_l1     = Q6_V_vlalign_VVI(Q6_V_lo_W(ws16_sum1), Q6_V_lo_W(ws16_sum0), 2);

    HVX_Vector vs16_diff_r0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws16_sum2), Q6_V_hi_W(ws16_sum1), 2);
    HVX_Vector vs16_sum_common1 = Q6_Vh_vadd_VhVh(vs16_diff_r0, Q6_V_lo_W(ws16_sum2));
               vs16_sum_common1 = Q6_V_valign_VVI(vs16_sum_common1, vs16_sum_common0, 2);

    HVX_Vector vs16_diff_r1     = Q6_V_valign_VVI(Q6_V_hi_W(ws16_sum2), Q6_V_hi_W(ws16_sum1), 2);
    HVX_Vector vs16_sum_common  = Q6_Vh_vadd_VhVh(vs16_sum_common0, vs16_sum_common1);

    HVX_Vector *row_sum_data_out = (HVX_Vector*)(row_sum + row_sum_step);

    HVX_Vector vs16_sum_even = Q6_Vh_vadd_VhVh(vs16_sum_common, vs16_diff_l1);
    HVX_Vector vs16_sum_odd  = Q6_Vh_vadd_VhVh(vs16_sum_common, vs16_diff_r1);

    vstore(row_sum_data_out,   vs16_sum_even);
    vstore(++row_sum_data_out, vs16_sum_odd);

    HVX_Vector vu16_dst_odd  = vdiv_n<DT_U16, 25>(vs16_sum_odd);
    HVX_Vector vu16_dst_even = vdiv_n<DT_U16, 25>(vs16_sum_even);

    vu8_result = Q6_Vub_vsat_VhVh(vu16_dst_odd, vu16_dst_even);
}

// using Tp = DT_U16
template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5VCore(const HVX_Vector &vu16_src_p0, const HVX_Vector &vu16_src_n0, HVX_VectorPair &ws32_result)
{
    ws32_result = Q6_Ww_vsub_VuhVuh(vu16_src_n0, vu16_src_p0);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5HCore(HVX_VectorPair &ws32_diff0, HVX_VectorPair &ws32_diff1, HVX_VectorPair &ws32_diff2,
                                             HVX_Vector &vu16_result, const SumType *row_sum, DT_S32 row_sum_step)
{
    HVX_Vector vs32_diff_l0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws32_diff1), Q6_V_hi_W(ws32_diff0), 4);
    HVX_Vector vs32_sum_common0 = Q6_Vw_vadd_VwVw(vs32_diff_l0, Q6_V_lo_W(ws32_diff1));
    HVX_Vector vs32_diff_l1     = Q6_V_vlalign_VVI(Q6_V_lo_W(ws32_diff1), Q6_V_lo_W(ws32_diff0), 4);

    HVX_Vector vs32_diff_r0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws32_diff2), Q6_V_hi_W(ws32_diff1), 4);
    HVX_Vector vs32_sum_common1 = Q6_Vw_vadd_VwVw(vs32_diff_r0, Q6_V_lo_W(ws32_diff2));
               vs32_sum_common1 = Q6_V_valign_VVI(vs32_sum_common1, vs32_sum_common0, 4);

    HVX_Vector vs32_diff_r1     = Q6_V_valign_VVI(Q6_V_hi_W(ws32_diff2), Q6_V_hi_W(ws32_diff1), 4);
    HVX_Vector vs32_sum_common  = Q6_Vw_vadd_VwVw(vs32_sum_common0, vs32_sum_common1);

    HVX_Vector *row_sum_data_in  = (HVX_Vector*)(row_sum + row_sum_step);
    HVX_Vector *row_sum_data_out = row_sum_data_in;

    HVX_Vector vs32_sum_even, vs32_sum_odd;
    vload(row_sum_data_in,   vs32_sum_even);
    vload(++row_sum_data_in, vs32_sum_odd);

    vs32_sum_even = Q6_Vw_vadd_VwVw(vs32_sum_even, Q6_Vw_vadd_VwVw(vs32_sum_common, vs32_diff_l1));
    vs32_sum_odd  = Q6_Vw_vadd_VwVw(vs32_sum_odd,  Q6_Vw_vadd_VwVw(vs32_sum_common, vs32_diff_r1));

    vstore(row_sum_data_out,   vs32_sum_even);
    vstore(++row_sum_data_out, vs32_sum_odd);

    HVX_Vector vu32_dst_odd  = vdiv_n<DT_U32, 25>(vs32_sum_odd);
    HVX_Vector vu32_dst_even = vdiv_n<DT_U32, 25>(vs32_sum_even);
    vu16_result = Q6_Vuh_vsat_VuwVuw(vu32_dst_odd, vu32_dst_even);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5FirstRowVCore(const HVX_Vector &vu16_src_p1, const HVX_Vector &vu16_src_p0, const HVX_Vector &vu16_src_c,
                                                     const HVX_Vector &vu16_src_n0, const HVX_Vector &vu16_src_n1, HVX_VectorPair &ws32_result)
{
    HVX_VectorPair ws32_result1 = Q6_Ww_vadd_VuhVuh(vu16_src_p1, vu16_src_n1);
    ws32_result = Q6_Ww_vadd_VuhVuh(vu16_src_p0, vu16_src_n0);
    ws32_result = Q6_Ww_vadd_WwWw(ws32_result, ws32_result1);
    ws32_result = Q6_Wuw_vmpyacc_WuwVuhRuh(ws32_result, vu16_src_c, 0x00010001);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5FirstRowHCore(HVX_VectorPair &ws32_sum0, HVX_VectorPair &ws32_sum1, HVX_VectorPair &ws32_sum2,
                                                     HVX_Vector &vu16_result, const SumType *row_sum, DT_S32 row_sum_step)
{
    HVX_Vector vs32_diff_l0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws32_sum1), Q6_V_hi_W(ws32_sum0), 4);
    HVX_Vector vs32_sum_common0 = Q6_Vw_vadd_VwVw(vs32_diff_l0, Q6_V_lo_W(ws32_sum1));
    HVX_Vector vs32_diff_l1     = Q6_V_vlalign_VVI(Q6_V_lo_W(ws32_sum1), Q6_V_lo_W(ws32_sum0), 4);

    HVX_Vector vs32_diff_r0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws32_sum2), Q6_V_hi_W(ws32_sum1), 4);
    HVX_Vector vs32_sum_common1 = Q6_Vw_vadd_VwVw(vs32_diff_r0, Q6_V_lo_W(ws32_sum2));
               vs32_sum_common1 = Q6_V_valign_VVI(vs32_sum_common1, vs32_sum_common0, 4);

    HVX_Vector vs32_diff_r1     = Q6_V_valign_VVI(Q6_V_hi_W(ws32_sum2), Q6_V_hi_W(ws32_sum1), 4);
    HVX_Vector vs32_sum_common  = Q6_Vw_vadd_VwVw(vs32_sum_common0, vs32_sum_common1);

    HVX_Vector *row_sum_data_out = (HVX_Vector*)(row_sum + row_sum_step);

    HVX_Vector vs32_sum_even = Q6_Vw_vadd_VwVw(vs32_sum_common, vs32_diff_l1);
    HVX_Vector vs32_sum_odd  = Q6_Vw_vadd_VwVw(vs32_sum_common, vs32_diff_r1);

    vstore(row_sum_data_out,   vs32_sum_even);
    vstore(++row_sum_data_out, vs32_sum_odd);

    HVX_Vector vu32_dst_odd  = vdiv_n<DT_U32, 25>(vs32_sum_odd);
    HVX_Vector vu32_dst_even = vdiv_n<DT_U32, 25>(vs32_sum_even);
    vu16_result = Q6_Vuh_vsat_VuwVuw(vu32_dst_odd, vu32_dst_even);
}

// using Tp = DT_S16
template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5VCore(const HVX_Vector &vs16_src_p1, const HVX_Vector &vs16_src_n1, HVX_VectorPair &ws32_result)
{
    ws32_result = Q6_Ww_vsub_VhVh(vs16_src_n1, vs16_src_p1);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5HCore(HVX_VectorPair &ws32_diff0, HVX_VectorPair &ws32_diff1, HVX_VectorPair &ws32_diff2,
                                             HVX_Vector &vs16_result, const SumType *row_sum, DT_S32 row_sum_step)
{
    HVX_Vector vs32_diff_l0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws32_diff1), Q6_V_hi_W(ws32_diff0), 4);
    HVX_Vector vs32_sum_common0 = Q6_Vw_vadd_VwVw(vs32_diff_l0, Q6_V_lo_W(ws32_diff1));
    HVX_Vector vs32_diff_l1     = Q6_V_vlalign_VVI(Q6_V_lo_W(ws32_diff1), Q6_V_lo_W(ws32_diff0), 4);

    HVX_Vector vs32_diff_r0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws32_diff2), Q6_V_hi_W(ws32_diff1), 4);
    HVX_Vector vs32_sum_common1 = Q6_Vw_vadd_VwVw(vs32_diff_r0, Q6_V_lo_W(ws32_diff2));
               vs32_sum_common1 = Q6_V_valign_VVI(vs32_sum_common1, vs32_sum_common0, 4);

    HVX_Vector vs32_diff_r1     = Q6_V_valign_VVI(Q6_V_hi_W(ws32_diff2), Q6_V_hi_W(ws32_diff1), 4);
    HVX_Vector vs32_sum_common  = Q6_Vw_vadd_VwVw(vs32_sum_common0, vs32_sum_common1);

    HVX_Vector *row_sum_data_in  = (HVX_Vector*)(row_sum + row_sum_step);
    HVX_Vector *row_sum_data_out = row_sum_data_in;

    HVX_Vector vs32_sum_even, vs32_sum_odd;
    vload(row_sum_data_in,   vs32_sum_even);
    vload(++row_sum_data_in, vs32_sum_odd);

    vs32_sum_even = Q6_Vw_vadd_VwVw(vs32_sum_even, Q6_Vw_vadd_VwVw(vs32_sum_common, vs32_diff_l1));
    vs32_sum_odd  = Q6_Vw_vadd_VwVw(vs32_sum_odd,  Q6_Vw_vadd_VwVw(vs32_sum_common, vs32_diff_r1));

    vstore(row_sum_data_out,   vs32_sum_even);
    vstore(++row_sum_data_out, vs32_sum_odd);

    HVX_Vector vs32_dst_odd  = vdiv_n<DT_S32, 25>(vs32_sum_odd);
    HVX_Vector vs32_dst_even = vdiv_n<DT_S32, 25>(vs32_sum_even);

    vs16_result = Q6_Vh_vsat_VwVw(vs32_dst_odd, vs32_dst_even);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5FirstRowVCore(const HVX_Vector &vs16_src_p1, const HVX_Vector &vs16_src_p0, const HVX_Vector &vs16_src_c,
                                                     const HVX_Vector &vs16_src_n0, const HVX_Vector &vs16_src_n1, HVX_VectorPair &ws32_result)
{
    HVX_VectorPair ws32_result1 = Q6_Ww_vadd_VhVh(vs16_src_p1, vs16_src_n1);
    ws32_result = Q6_Ww_vadd_VhVh(vs16_src_p0, vs16_src_n0);
    ws32_result = Q6_Ww_vadd_WwWw(ws32_result, ws32_result1);
    ws32_result = Q6_Ww_vmpyacc_WwVhRh(ws32_result, vs16_src_c, 0x00010001);
}

template <typename Tp, typename SumType, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5FirstRowHCore(HVX_VectorPair &ws32_sum0, HVX_VectorPair &ws32_sum1, HVX_VectorPair &ws32_sum2,
                                                     HVX_Vector &vs16_result, const SumType *row_sum, DT_S32 row_sum_step)
{
    HVX_Vector vs32_diff_l0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws32_sum1), Q6_V_hi_W(ws32_sum0), 4);
    HVX_Vector vs32_sum_common0 = Q6_Vw_vadd_VwVw(vs32_diff_l0, Q6_V_lo_W(ws32_sum1));
    HVX_Vector vs32_diff_l1     = Q6_V_vlalign_VVI(Q6_V_lo_W(ws32_sum1), Q6_V_lo_W(ws32_sum0), 4);

    HVX_Vector vs32_diff_r0     = Q6_V_vlalign_VVI(Q6_V_hi_W(ws32_sum2), Q6_V_hi_W(ws32_sum1), 4);
    HVX_Vector vs32_sum_common1 = Q6_Vw_vadd_VwVw(vs32_diff_r0, Q6_V_lo_W(ws32_sum2));
               vs32_sum_common1 = Q6_V_valign_VVI(vs32_sum_common1, vs32_sum_common0, 4);

    HVX_Vector vs32_diff_r1     = Q6_V_valign_VVI(Q6_V_hi_W(ws32_sum2), Q6_V_hi_W(ws32_sum1), 4);
    HVX_Vector vs32_sum_common  = Q6_Vw_vadd_VwVw(vs32_sum_common0, vs32_sum_common1);

    HVX_Vector *row_sum_data_out = (HVX_Vector*)(row_sum + row_sum_step);

    HVX_Vector vs32_sum_even = Q6_Vw_vadd_VwVw(vs32_sum_common, vs32_diff_l1);
    HVX_Vector vs32_sum_odd  = Q6_Vw_vadd_VwVw(vs32_sum_common, vs32_diff_r1);

    vstore(row_sum_data_out,   vs32_sum_even);
    vstore(++row_sum_data_out, vs32_sum_odd);

    HVX_Vector vs32_dst_odd  = vdiv_n<DT_S32, 25>(vs32_sum_odd);
    HVX_Vector vs32_dst_even = vdiv_n<DT_S32, 25>(vs32_sum_even);

    vs16_result = Q6_Vh_vsat_VwVw(vs32_dst_odd, vs32_dst_even);
}

template <typename Tp, typename SumType>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5HCore(HVX_VectorPair &w_sum_x0, HVX_VectorPair &w_sum_x1, HVX_VectorPair &w_sum_x2, HVX_VectorPair &w_sum_x3,
                                             HVX_Vector &v_result_x0, HVX_Vector &v_result_x1, SumType *row_sum, DT_S32 row_sum_step, DT_S32 rest)
{
    HVX_Vector v_sum_x0_lo = Q6_V_lo_W(w_sum_x0);
    HVX_Vector v_sum_x0_hi = Q6_V_hi_W(w_sum_x0);
    HVX_Vector v_sum_x1_lo = Q6_V_lo_W(w_sum_x1);
    HVX_Vector v_sum_x1_hi = Q6_V_hi_W(w_sum_x1);
    HVX_Vector v_sum_x2_lo = Q6_V_lo_W(w_sum_x2);
    HVX_Vector v_sum_x2_hi = Q6_V_hi_W(w_sum_x2);
    HVX_Vector v_sum_x3_lo = Q6_V_lo_W(w_sum_x3);
    HVX_Vector v_sum_x3_hi = Q6_V_hi_W(w_sum_x3);

    HVX_Vector v_sum_l0_lo, v_sum_l0_hi, v_sum_r0_lo, v_sum_r0_hi;
    if (rest & 1)
    {
        DT_S32 align_size0 = (rest / 2) * sizeof(SumType);
        DT_S32 align_size1 = align_size0 + sizeof(SumType);
        v_sum_r0_lo = Q6_V_vlalign_safe_VVR(v_sum_x3_hi, v_sum_x2_hi, align_size1);
        v_sum_r0_hi = Q6_V_vlalign_safe_VVR(v_sum_x3_lo, v_sum_x2_lo, align_size0);
        v_sum_l0_lo = Q6_V_valign_safe_VVR(v_sum_x1_hi, v_sum_x0_hi, align_size0);
        v_sum_l0_hi = Q6_V_valign_safe_VVR(v_sum_x1_lo, v_sum_x0_lo, align_size1);
    }
    else
    {
        DT_S32 align_size = (rest / 2) * sizeof(SumType);
        v_sum_r0_lo = Q6_V_vlalign_safe_VVR(v_sum_x3_lo, v_sum_x2_lo, align_size);
        v_sum_r0_hi = Q6_V_vlalign_safe_VVR(v_sum_x3_hi, v_sum_x2_hi, align_size);
        v_sum_l0_lo = Q6_V_valign_safe_VVR(v_sum_x1_lo, v_sum_x0_lo, align_size);
        v_sum_l0_hi = Q6_V_valign_safe_VVR(v_sum_x1_hi, v_sum_x0_hi, align_size);
    }

    HVX_VectorPair w_sum_r0 = Q6_W_vcombine_VV(v_sum_r0_hi, v_sum_r0_lo);
    HVX_VectorPair w_sum_l0 = Q6_W_vcombine_VV(v_sum_l0_hi, v_sum_l0_lo);

    BoxFilter5x5HCore<Tp, SumType>(w_sum_x0, w_sum_x1, w_sum_r0, v_result_x0, row_sum, row_sum_step);
    BoxFilter5x5HCore<Tp, SumType>(w_sum_l0, w_sum_x2, w_sum_x3, v_result_x1, row_sum, row_sum_step + 128);
}

template <typename Tp, typename SumType>
AURA_ALWAYS_INLINE DT_VOID BoxFilter5x5FirstRowHCore(HVX_VectorPair &w_sum_x0, HVX_VectorPair &w_sum_x1, HVX_VectorPair &w_sum_x2, HVX_VectorPair &w_sum_x3,
                                                     HVX_Vector &v_result_x0, HVX_Vector &v_result_x1, SumType *row_sum, DT_S32 row_sum_step, DT_S32 rest)
{
    HVX_Vector v_sum_x0_lo = Q6_V_lo_W(w_sum_x0);
    HVX_Vector v_sum_x0_hi = Q6_V_hi_W(w_sum_x0);
    HVX_Vector v_sum_x1_lo = Q6_V_lo_W(w_sum_x1);
    HVX_Vector v_sum_x1_hi = Q6_V_hi_W(w_sum_x1);
    HVX_Vector v_sum_x2_lo = Q6_V_lo_W(w_sum_x2);
    HVX_Vector v_sum_x2_hi = Q6_V_hi_W(w_sum_x2);
    HVX_Vector v_sum_x3_lo = Q6_V_lo_W(w_sum_x3);
    HVX_Vector v_sum_x3_hi = Q6_V_hi_W(w_sum_x3);

    HVX_Vector v_sum_l0_lo, v_sum_l0_hi, v_sum_r0_lo, v_sum_r0_hi;
    if (rest & 1)
    {
        DT_S32 align_size0 = (rest / 2) * sizeof(SumType);
        DT_S32 align_size1 = align_size0 + sizeof(SumType);
        v_sum_r0_lo = Q6_V_vlalign_safe_VVR(v_sum_x3_hi, v_sum_x2_hi, align_size1);
        v_sum_r0_hi = Q6_V_vlalign_safe_VVR(v_sum_x3_lo, v_sum_x2_lo, align_size0);
        v_sum_l0_lo = Q6_V_valign_safe_VVR(v_sum_x1_hi, v_sum_x0_hi, align_size0);
        v_sum_l0_hi = Q6_V_valign_safe_VVR(v_sum_x1_lo, v_sum_x0_lo, align_size1);
    }
    else
    {
        DT_S32 align_size = (rest / 2) * sizeof(SumType);
        v_sum_r0_lo = Q6_V_vlalign_safe_VVR(v_sum_x3_lo, v_sum_x2_lo, align_size);
        v_sum_r0_hi = Q6_V_vlalign_safe_VVR(v_sum_x3_hi, v_sum_x2_hi, align_size);
        v_sum_l0_lo = Q6_V_valign_safe_VVR(v_sum_x1_lo, v_sum_x0_lo, align_size);
        v_sum_l0_hi = Q6_V_valign_safe_VVR(v_sum_x1_hi, v_sum_x0_hi, align_size);
    }

    HVX_VectorPair w_sum_r0 = Q6_W_vcombine_VV(v_sum_r0_hi, v_sum_r0_lo);
    HVX_VectorPair w_sum_l0 = Q6_W_vcombine_VV(v_sum_l0_hi, v_sum_l0_lo);

    BoxFilter5x5FirstRowHCore<Tp, SumType>(w_sum_x0, w_sum_x1, w_sum_r0, v_result_x0, row_sum, row_sum_step);
    BoxFilter5x5FirstRowHCore<Tp, SumType>(w_sum_l0, w_sum_x2, w_sum_x3, v_result_x1, row_sum, row_sum_step + 128);
}

template <typename Tp, typename SumType, BorderType BORDER_TYPE, DT_S32 C>
static DT_VOID BoxFilter5x5Row(const Tp *src_p0, const Tp *src_n0, Tp *dst, DT_S32 width, SumType *row_sum, DT_S32 row_sum_step,
                               const std::vector<Tp> &border_value)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    DT_S32 shift = 0;
    DT_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);
    DT_S32 back_offset = width - ELEM_COUNTS;

    MVType mv_src_p0, mv_src_n0;
    MWType mw_diff0, mw_diff1, mw_diff2;
    MVType mv_result;

    // left border
    {
        vload(src_p0, mv_src_p0);
        vload(src_n0, mv_src_n0);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p0.val[ch], src_p0[ch], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n0.val[ch], src_n0[ch], border_value[ch]);

            BoxFilter5x5VCore<Tp, SumType>(v_border_p1, v_border_n1, mw_diff0.val[ch]);
            BoxFilter5x5VCore<Tp, SumType>(mv_src_p0.val[ch], mv_src_n0.val[ch], mw_diff1.val[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (DT_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS, shift += ELEM_COUNTS)
        {
            vload(src_p0 + C * x, mv_src_p0);
            vload(src_n0 + C * x, mv_src_n0);

            #pragma unroll(C)
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                BoxFilter5x5VCore<Tp, SumType>(mv_src_p0.val[ch], mv_src_n0.val[ch], mw_diff2.val[ch]);
                BoxFilter5x5HCore<Tp, SumType>(mw_diff0.val[ch], mw_diff1.val[ch], mw_diff2.val[ch], mv_result.val[ch], row_sum, row_sum_step * ch + shift);
            }

            vstore(dst + C * (x - ELEM_COUNTS), mv_result);

            mw_diff0 = mw_diff1;
            mw_diff1 = mw_diff2;
        }
    }

    // remain
    {
        DT_S32 last = C * (width - 1);
        DT_S32 rest = width % ELEM_COUNTS;
        MVType mv_last;

        vload(src_p0 + C * back_offset, mv_src_p0);
        vload(src_n0 + C * back_offset, mv_src_n0);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0.val[ch], src_p0[last + ch], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0.val[ch], src_n0[last + ch], border_value[ch]);

            HVX_VectorPair w_diff2, w_diff3;
            BoxFilter5x5VCore<Tp, SumType>(mv_src_p0.val[ch], mv_src_n0.val[ch], w_diff2);
            BoxFilter5x5VCore<Tp, SumType>(v_border_p1, v_border_n1, w_diff3);

            BoxFilter5x5HCore<Tp, SumType>(mw_diff0.val[ch], mw_diff1.val[ch], w_diff2, w_diff3,
                                           mv_result.val[ch], mv_last.val[ch], row_sum, row_sum_step * ch + shift, rest);
        }

        vstore(dst + C * (back_offset - rest), mv_result);
        vstore(dst + C * back_offset, mv_last);
    }
}

template <typename Tp, typename SumType, BorderType BORDER_TYPE, DT_S32 C>
static DT_VOID BoxFilter5x5FirstRow(const Tp *src_p1, const Tp *src_p0, const Tp *src_c, const Tp *src_n0, const Tp *src_n1, Tp *dst,
                                    DT_S32 width, SumType *row_sum, DT_S32 row_sum_step, const std::vector<Tp> &border_value)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    DT_S32 shift                 = 0;
    constexpr DT_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);
    DT_S32 back_offset           = width - ELEM_COUNTS;

    MVType mv_src_p1, mv_src_p0, mv_src_c, mv_src_n0, mv_src_n1;
    MWType mw_sum0, mw_sum1, mw_sum2;
    MVType mv_result;

    // left border
    {
        vload(src_p1, mv_src_p1);
        vload(src_p0, mv_src_p0);
        vload(src_c,  mv_src_c);
        vload(src_n0, mv_src_n0);
        vload(src_n1, mv_src_n1);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p1.val[ch], src_p1[ch], border_value[ch]);
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p0.val[ch], src_p0[ch], border_value[ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_c.val[ch], src_c[ch], border_value[ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n0.val[ch], src_n0[ch], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n1.val[ch], src_n1[ch], border_value[ch]);

            BoxFilter5x5FirstRowVCore<Tp, SumType>(v_border_p1, v_border_p0, v_border_c, v_border_n0, v_border_n1, mw_sum0.val[ch]);
            BoxFilter5x5FirstRowVCore<Tp, SumType>(mv_src_p1.val[ch], mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mv_src_n1.val[ch], mw_sum1.val[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (DT_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS, shift += ELEM_COUNTS)
        {
            vload(src_p1 + C * x, mv_src_p1);
            vload(src_p0 + C * x, mv_src_p0);
            vload(src_c + C * x,  mv_src_c);
            vload(src_n0 + C * x, mv_src_n0);
            vload(src_n1 + C * x, mv_src_n1);

            #pragma unroll(C)
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                BoxFilter5x5FirstRowVCore<Tp, SumType>(mv_src_p1.val[ch], mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mv_src_n1.val[ch], mw_sum2.val[ch]);
                BoxFilter5x5FirstRowHCore<Tp, SumType>(mw_sum0.val[ch], mw_sum1.val[ch], mw_sum2.val[ch], mv_result.val[ch], row_sum, row_sum_step * ch + shift);
            }

            vstore(dst + C * (x - ELEM_COUNTS), mv_result);

            mw_sum0 = mw_sum1;
            mw_sum1 = mw_sum2;
        }
    }

    // remain
    {
        DT_S32 last = C * (width - 1);
        DT_S32 rest = width % ELEM_COUNTS;
        MVType mv_last;

        vload(src_p1 + C * back_offset, mv_src_p1);
        vload(src_p0 + C * back_offset, mv_src_p0);
        vload(src_c  + C * back_offset, mv_src_c);
        vload(src_n0 + C * back_offset, mv_src_n0);
        vload(src_n1 + C * back_offset, mv_src_n1);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p1.val[ch], src_p1[last + ch], border_value[ch]);
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0.val[ch], src_p0[last + ch], border_value[ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_c.val[ch],  src_c[last + ch],  border_value[ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0.val[ch], src_n0[last + ch], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n1.val[ch], src_n1[last + ch], border_value[ch]);

            HVX_VectorPair w_sum2, w_sum3;
            BoxFilter5x5FirstRowVCore<Tp, SumType>(mv_src_p1.val[ch], mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mv_src_n1.val[ch], w_sum2);
            BoxFilter5x5FirstRowVCore<Tp, SumType>(v_border_p1, v_border_p0, v_border_c, v_border_n0, v_border_n1, w_sum3);

            BoxFilter5x5FirstRowHCore<Tp, SumType>(mw_sum0.val[ch], mw_sum1.val[ch], w_sum2, w_sum3,
                                                   mv_result.val[ch], mv_last.val[ch], row_sum, row_sum_step * ch + shift, rest);
        }

        vstore(dst + C * (back_offset - rest), mv_result);
        vstore(dst + C * back_offset, mv_last);
    }
}

template <typename Tp, typename SumType, BorderType BORDER_TYPE, DT_S32 C>
static Status BoxFilter5x5HvxImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &row_sum_buffer, const std::vector<Tp> &border_value,
                                  const Tp *border_buffer, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width   = src.GetSizes().m_width;
    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    DT_S32 sum_size  = AURA_ALIGN(width, 128) + AURA_HVLEN;
    SumType *row_sum = row_sum_buffer.GetThreadData<SumType>();
    if (NULL == row_sum)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
        return Status::ERROR;
    }

    memset(row_sum, 0, row_sum_buffer.GetBufferSize());

    Tp *dst_row      = dst.Ptr<Tp>(start_row);
    const Tp *src_p1 = src.Ptr<Tp, BORDER_TYPE>(start_row - 2, border_buffer);
    const Tp *src_p0 = src.Ptr<Tp, BORDER_TYPE>(start_row - 1, border_buffer);
    const Tp *src_c  = src.Ptr<Tp>(start_row);
    const Tp *src_n0 = src.Ptr<Tp, BORDER_TYPE>(start_row + 1, border_buffer);
    const Tp *src_n1 = src.Ptr<Tp, BORDER_TYPE>(start_row + 2, border_buffer);
    BoxFilter5x5FirstRow<Tp, SumType, BORDER_TYPE, C>(src_p1, src_p0, src_c, src_n0, src_n1, dst_row, width, row_sum, sum_size, border_value);

    const Tp *src_p2 = src_p1;
    src_p1 = src_p0;
    src_p0 = src_c;
    src_c  = src_n0;
    src_n0 = src_n1;
    src_n1 = src.Ptr<Tp, BORDER_TYPE>(start_row + 3, border_buffer);
    DT_U64 L2fetch_param = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (DT_S32 y = start_row + 1; y < end_row; y++)
    {
        if (y + 3 < height)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<Tp>(y + 3)), L2fetch_param);
        }

        Tp *dst_row = dst.Ptr<Tp>(y);
        BoxFilter5x5Row<Tp, SumType, BORDER_TYPE, C>(src_p2, src_n1, dst_row, width, row_sum, sum_size, border_value);

        src_p2 = src_p1;
        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src.Ptr<Tp, BORDER_TYPE>(y + 3, border_buffer);
    }

    return Status::OK;
}

template<typename Tp, typename SumType, BorderType BORDER_TYPE>
static Status BoxFilter5x5HvxHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<Tp> &border_value, const Tp *border_buffer)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    DT_S32 height   = src.GetSizes().m_height;
    DT_S32 width    = src.GetSizes().m_width;
    DT_S32 channel  = src.GetSizes().m_channel;
    DT_S32 sum_size = AURA_ALIGN(width, 128) + AURA_HVLEN;

    ThreadBuffer row_sum_buffer(ctx, sum_size * sizeof(SumType) * channel);

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((DT_S32)0, height, BoxFilter5x5HvxImpl<Tp, SumType, BORDER_TYPE, 1>,
                                  ctx, std::cref(src), std::ref(dst), std::ref(row_sum_buffer), std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((DT_S32)0, height, BoxFilter5x5HvxImpl<Tp, SumType, BORDER_TYPE, 2>,
                                  ctx, std::cref(src), std::ref(dst), std::ref(row_sum_buffer), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, height, BoxFilter5x5HvxImpl<Tp, SumType, BORDER_TYPE, 3>,
                                  ctx, std::cref(src), std::ref(dst), std::ref(row_sum_buffer), std::cref(border_value), border_buffer);
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

template <typename Tp, typename SumType>
static Status BoxFilter5x5HvxHelper(Context *ctx, const Mat &src, Mat &dst, BorderType &border_type, const Scalar &border_value)
{
    Tp *border_buffer = DT_NULL;
    std::vector<Tp> vec_border_value = border_value.ToVector<Tp>();

    DT_S32 width   = dst.GetSizes().m_width;
    DT_S32 channel = dst.GetSizes().m_channel;

    Status ret = Status::ERROR;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            border_buffer = CreateBorderBuffer(ctx, width, channel, vec_border_value);
            if (DT_NULL == border_buffer)
            {
                AURA_ADD_ERROR_STRING(ctx, "CreateBorderBuffer failed");
                return Status::ERROR;
            }

            ret = BoxFilter5x5HvxHelper<Tp, SumType, BorderType::CONSTANT>(ctx, src, dst, vec_border_value, border_buffer);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = BoxFilter5x5HvxHelper<Tp, SumType, BorderType::REPLICATE>(ctx, src, dst, vec_border_value, border_buffer);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = BoxFilter5x5HvxHelper<Tp, SumType, BorderType::REFLECT_101>(ctx, src, dst, vec_border_value, border_buffer);
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

Status BoxFilter5x5Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = BoxFilter5x5HvxHelper<DT_U8, DT_S16>(ctx, src, dst, border_type, border_value);
            break;
        }

        case ElemType::U16:
        {
            ret = BoxFilter5x5HvxHelper<DT_U16, DT_S32>(ctx, src, dst, border_type, border_value);
            break;
        }

        case ElemType::S16:
        {
            ret = BoxFilter5x5HvxHelper<DT_S16, DT_S32>(ctx, src, dst, border_type, border_value);
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