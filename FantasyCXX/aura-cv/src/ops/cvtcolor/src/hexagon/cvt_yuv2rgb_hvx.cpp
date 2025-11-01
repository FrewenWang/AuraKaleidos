#include "cvtcolor_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{
/**
 * @brief the formula of YUV -> RGB
 * R = 1.164(Y - 16) + 1.596(V - 128)
 * G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
 * B = 1.164(Y - 16)                  + 2.018(U - 128)
 *
 * R = CVTCOLOR_DESCALE(Y2RGB * (Y - 16) + V2R * (V - 128)                  , CVTCOLOR_COEF_BITS)
 * G = CVTCOLOR_DESCALE(Y2RGB * (Y - 16) + V2G * (V - 128) + U2G * (U - 128), CVTCOLOR_COEF_BITS)
 * B = CVTCOLOR_DESCALE(Y2RGB * (Y - 16)                   + U2B * (U - 128), CVTCOLOR_COEF_BITS)
 *
 * Y2RGB :  1220542 =  49 (0x31313131) * 24909 (0x614d614d) + 1;  // Round(1.164f  * (1 << CVTCOLOR_COEF_BITS));
 * V2R   :  1673527 =  67 (0x43434343) * 24978 (0x61926192) + 1;  // Round(1.596f  * (1 << CVTCOLOR_COEF_BITS));
 * V2G   : -852492  = -38 (0xdadadada) * 22434 (0x57a257a2);      // Round(-0.813f * (1 << CVTCOLOR_COEF_BITS));
 * U2G   : -409993  = -13 (0xf3f3f3f3) * 31538 (0x7b327b32) + 1;  // Round(-0.391f * (1 << CVTCOLOR_COEF_BITS));
 * U2B   :  2116026 =  66 (0x42424242) * 32061 (0x7d3d7d3d);      // Round(2.018f  * (1 << CVTCOLOR_COEF_BITS));
 *
 * -213687168(f3436480) : (V2R)       * (-128) + (1 << (CVTCOLOR_COEF_BITS - 1))
 *  162122368(09a9ca80) : (V2G + U2G) * (-128) + (1 << (CVTCOLOR_COEF_BITS - 1))
 * -270327040(efe32300) : (U2B)       * (-128) + (1 << (CVTCOLOR_COEF_BITS - 1))
 */

/**
 * @brief the formula of BT.601 YUV -> RGB
 * R = Y + 1.403(V - 128)
 * G = Y - 0.714(V - 128) - 0.343(U - 128)
 * B = Y                  + 1.770(U - 128)
 *
 * R = CVTCOLOR_DESCALE(Y2RGB * Y + V2R * (V - 128)                  , CVTCOLOR_COEF_BITS)
 * G = CVTCOLOR_DESCALE(Y2RGB * Y + V2G * (V - 128) + U2G * (U - 128), CVTCOLOR_COEF_BITS)
 * B = CVTCOLOR_DESCALE(Y2RGB * Y                   + U2B * (U - 128), CVTCOLOR_COEF_BITS)
 *
 * Y2RGB : 1048576 =  64  (0x40404040) * 16384 (0x40004000);      // Round(1.000f  * (1 << CVTCOLOR_COEF_BITS));
 * V2R   : 1471152 =  48  (0x30303030) * 30649 (0x77b977b9);      // Round(1.403f  * (1 << CVTCOLOR_COEF_BITS));
 * V2G   : -748683 = -27  (0xe5e5e5e5) * 27729 (0x6c516c51);      // Round(-0.714f * (1 << CVTCOLOR_COEF_BITS));
 * U2G   : -359661 = -101 (0x9b9b9b9b) * 3561  (0x0de90de9);      // Round(-0.343f * (1 << CVTCOLOR_COEF_BITS));
 * U2B   : 1855979 =  127 (0x7f7f7f7f) * 14614 (0x39163916) + 1;  // Round(1.770f  * (1 << CVTCOLOR_COEF_BITS));
 *
 * -187783168(f4cea800) : (V2R)       * (-128) + (1 << (CVTCOLOR_COEF_BITS - 1))
 *  142392320(087cbc00) : (V2G + U2G) * (-128) + (1 << (CVTCOLOR_COEF_BITS - 1))
 * -237041024(f1df0a80) : (U2B)       * (-128) + (1 << (CVTCOLOR_COEF_BITS - 1))
 */

AURA_ALWAYS_INLINE AURA_VOID CvtYxuv2XCore(const HVX_VectorPairX2 &w2s32_yy, const HVX_VectorPairX2 &w2s32_xuv, HVX_Vector &vu8_x)
{
    HVX_VectorPair ws32_x_lo = Q6_Ww_vadd_WwWw(w2s32_yy.val[0], w2s32_xuv.val[0]);
    HVX_VectorPair ws32_x_hi = Q6_Ww_vadd_WwWw(w2s32_yy.val[1], w2s32_xuv.val[1]);

    HVX_Vector vs32_x_lo_lo = Q6_Vw_vmax_VwVw(Q6_V_lo_W(ws32_x_lo), Q6_V_vzero());
    HVX_Vector vs32_x_lo_hi = Q6_Vw_vmax_VwVw(Q6_V_hi_W(ws32_x_lo), Q6_V_vzero());
    HVX_Vector vs32_x_hi_lo = Q6_Vw_vmax_VwVw(Q6_V_lo_W(ws32_x_hi), Q6_V_vzero());
    HVX_Vector vs32_x_hi_hi = Q6_Vw_vmax_VwVw(Q6_V_hi_W(ws32_x_hi), Q6_V_vzero());

    HVX_Vector vs16_x_lo = Q6_Vh_vasr_VwVwR(vs32_x_lo_hi, vs32_x_lo_lo, 15);
    HVX_Vector vs16_x_hi = Q6_Vh_vasr_VwVwR(vs32_x_hi_hi, vs32_x_hi_lo, 15);

    vu8_x = Q6_Vub_vasr_VuhVuhR_sat(vs16_x_hi, vs16_x_lo, 5);
}

AURA_ALWAYS_INLINE AURA_VOID CvtShuffU8C3Core(HVX_VectorX3 &v3u8_vec0, HVX_VectorX3 &v3u8_vec1)
{
    HVX_VectorPair wu8_val0 = Q6_W_vshuff_VVR(v3u8_vec1.val[0], v3u8_vec0.val[0], -1);
    HVX_VectorPair wu8_val1 = Q6_W_vshuff_VVR(v3u8_vec1.val[1], v3u8_vec0.val[1], -1);
    HVX_VectorPair wu8_val2 = Q6_W_vshuff_VVR(v3u8_vec1.val[2], v3u8_vec0.val[2], -1);

    v3u8_vec0.val[0] = Q6_V_lo_W(wu8_val0);
    v3u8_vec0.val[1] = Q6_V_lo_W(wu8_val1);
    v3u8_vec0.val[2] = Q6_V_lo_W(wu8_val2);

    v3u8_vec1.val[0] = Q6_V_hi_W(wu8_val0);
    v3u8_vec1.val[1] = Q6_V_hi_W(wu8_val1);
    v3u8_vec1.val[2] = Q6_V_hi_W(wu8_val2);
}

template <MI_U32 MODE, typename std::enable_if<0 == MODE>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CvtUv2RgbuvCore(const HVX_Vector &vu8_src_u, const HVX_Vector &vu8_src_v, HVX_VectorPairX2 &w2s32_ruv,
                                           HVX_VectorPairX2 &w2s32_guv, HVX_VectorPairX2 &w2s32_buv)
{
    HVX_VectorPair ws16_vr = Q6_Wh_vmpy_VubRb(vu8_src_v, 0x43434343);
    HVX_VectorPair ws16_vg = Q6_Wh_vmpy_VubRb(vu8_src_v, 0xdadadada);
    HVX_VectorPair ws16_ub = Q6_Wh_vmpy_VubRb(vu8_src_u, 0x42424242);
    HVX_VectorPair ws16_ug = Q6_Wh_vmpy_VubRb(vu8_src_u, 0xf3f3f3f3);

    HVX_VectorPair ws32_uv2r_neg128_add_rnd = Q6_W_vcombine_VV(Q6_V_vsplat_R(0xf3436480),
                                                               Q6_V_vsplat_R(0xf3436480));
    HVX_VectorPair ws32_uv2g_neg128_add_rnd = Q6_W_vcombine_VV(Q6_V_vsplat_R(0x09a9ca80),
                                                               Q6_V_vsplat_R(0x09a9ca80));
    HVX_VectorPair ws32_uv2b_neg128_add_rnd = Q6_W_vcombine_VV(Q6_V_vsplat_R(0xefe32300),
                                                               Q6_V_vsplat_R(0xefe32300));

    w2s32_ruv.val[0] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2r_neg128_add_rnd, Q6_V_lo_W(ws16_vr), 0x61926192);
    w2s32_guv.val[0] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2g_neg128_add_rnd, Q6_V_lo_W(ws16_vg), 0x57a257a2);
    w2s32_buv.val[0] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2b_neg128_add_rnd, Q6_V_lo_W(ws16_ub), 0x7d3d7d3d);

    w2s32_ruv.val[1] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2r_neg128_add_rnd, Q6_V_hi_W(ws16_vr), 0x61926192);
    w2s32_guv.val[1] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2g_neg128_add_rnd, Q6_V_hi_W(ws16_vg), 0x57a257a2);
    w2s32_buv.val[1] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2b_neg128_add_rnd, Q6_V_hi_W(ws16_ub), 0x7d3d7d3d);

    HVX_VectorPair wu16_v = Q6_Wuh_vzxt_Vub(vu8_src_v);
    HVX_VectorPair wu16_u = Q6_Wuh_vzxt_Vub(vu8_src_u);

    HVX_VectorPair ws32_v_lo = Q6_Wuw_vzxt_Vuh(Q6_V_lo_W(wu16_v));
    HVX_VectorPair ws32_v_hi = Q6_Wuw_vzxt_Vuh(Q6_V_hi_W(wu16_v));

    HVX_VectorPair ws32_u_lo = Q6_Wuw_vzxt_Vuh(Q6_V_lo_W(wu16_u));
    HVX_VectorPair ws32_u_hi = Q6_Wuw_vzxt_Vuh(Q6_V_hi_W(wu16_u));

    w2s32_guv.val[0] = Q6_Ww_vmpyacc_WwVhRh(w2s32_guv.val[0], Q6_V_lo_W(ws16_ug), 0x7b327b32);
    w2s32_guv.val[1] = Q6_Ww_vmpyacc_WwVhRh(w2s32_guv.val[1], Q6_V_hi_W(ws16_ug), 0x7b327b32);

    w2s32_ruv.val[0] = Q6_Ww_vadd_WwWw(w2s32_ruv.val[0], ws32_v_lo);
    w2s32_ruv.val[1] = Q6_Ww_vadd_WwWw(w2s32_ruv.val[1], ws32_v_hi);

    w2s32_guv.val[0] = Q6_Ww_vadd_WwWw(w2s32_guv.val[0], ws32_u_lo);
    w2s32_guv.val[1] = Q6_Ww_vadd_WwWw(w2s32_guv.val[1], ws32_u_hi);
}

template <MI_U32 MODE, typename std::enable_if<1 == MODE>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CvtUv2RgbuvCore(const HVX_Vector &vu8_src_u, const HVX_Vector &vu8_src_v, HVX_VectorPairX2 &w2s32_ruv,
                                           HVX_VectorPairX2 &w2s32_guv, HVX_VectorPairX2 &w2s32_buv)
{
    HVX_VectorPair ws16_vr = Q6_Wh_vmpy_VubRb(vu8_src_v, 0x30303030);
    HVX_VectorPair ws16_vg = Q6_Wh_vmpy_VubRb(vu8_src_v, 0xe5e5e5e5);
    HVX_VectorPair ws16_ub = Q6_Wh_vmpy_VubRb(vu8_src_u, 0x7f7f7f7f);
    HVX_VectorPair ws16_ug = Q6_Wh_vmpy_VubRb(vu8_src_u, 0x9b9b9b9b);

    HVX_VectorPair ws32_uv2r_neg128_add_rnd = Q6_W_vcombine_VV(Q6_V_vsplat_R(0xf4cea800), Q6_V_vsplat_R(0xf4cea800));
    HVX_VectorPair ws32_uv2g_neg128_add_rnd = Q6_W_vcombine_VV(Q6_V_vsplat_R(0x087cbc00), Q6_V_vsplat_R(0x087cbc00));
    HVX_VectorPair ws32_uv2b_neg128_add_rnd = Q6_W_vcombine_VV(Q6_V_vsplat_R(0xf1df0a80), Q6_V_vsplat_R(0xf1df0a80));

    w2s32_ruv.val[0] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2r_neg128_add_rnd, Q6_V_lo_W(ws16_vr), 0x77b977b9);
    w2s32_guv.val[0] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2g_neg128_add_rnd, Q6_V_lo_W(ws16_vg), 0x6c516c51);
    w2s32_buv.val[0] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2b_neg128_add_rnd, Q6_V_lo_W(ws16_ub), 0x39163916);

    w2s32_ruv.val[1] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2r_neg128_add_rnd, Q6_V_hi_W(ws16_vr), 0x77b977b9);
    w2s32_guv.val[1] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2g_neg128_add_rnd, Q6_V_hi_W(ws16_vg), 0x6c516c51);
    w2s32_buv.val[1] = Q6_Ww_vmpyacc_WwVhRh(ws32_uv2b_neg128_add_rnd, Q6_V_hi_W(ws16_ub), 0x39163916);

    w2s32_guv.val[0] = Q6_Ww_vmpyacc_WwVhRh(w2s32_guv.val[0], Q6_V_lo_W(ws16_ug), 0x0de90de9);
    w2s32_guv.val[1] = Q6_Ww_vmpyacc_WwVhRh(w2s32_guv.val[1], Q6_V_hi_W(ws16_ug), 0x0de90de9);

    HVX_VectorPair wu16_u    = Q6_Wuh_vzxt_Vub(vu8_src_u);
    HVX_VectorPair ws32_u_lo = Q6_Wuw_vzxt_Vuh(Q6_V_lo_W(wu16_u));
    HVX_VectorPair ws32_u_hi = Q6_Wuw_vzxt_Vuh(Q6_V_hi_W(wu16_u));

    w2s32_buv.val[0] = Q6_Ww_vadd_WwWw(w2s32_buv.val[0], ws32_u_lo);
    w2s32_buv.val[1] = Q6_Ww_vadd_WwWw(w2s32_buv.val[1], ws32_u_hi);
}

template <MI_U32 MODE, typename std::enable_if<0 == MODE>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CvtY1rgbuv2RgbCore(const HVX_Vector &vu8_src_y, const HVX_VectorPairX2 &w2s32_ruv,
                                              const HVX_VectorPairX2 &w2s32_guv, const HVX_VectorPairX2 &w2s32_buv,
                                              HVX_VectorX3 &v3u8_dst)
{
    HVX_Vector vu8_src_ysub = Q6_Vub_vsub_VubVub_sat(vu8_src_y, Q6_V_vsplat_R(0x10101010));

    HVX_VectorPair wu16_ysub = Q6_Wuh_vzxt_Vub(vu8_src_ysub);
    HVX_VectorPair ws16_yrgb = Q6_Wh_vmpy_VubRb(vu8_src_ysub, 0x31313131);

    HVX_VectorPair ws32_ysub_lo = Q6_Wuw_vzxt_Vuh(Q6_V_lo_W(wu16_ysub));
    HVX_VectorPair ws32_ysub_hi = Q6_Wuw_vzxt_Vuh(Q6_V_hi_W(wu16_ysub));

    HVX_VectorPairX2 w2s32_yrgb;
    w2s32_yrgb.val[0] = Q6_Ww_vmpyacc_WwVhRh(ws32_ysub_lo, Q6_V_lo_W(ws16_yrgb), 0x614d614d);
    w2s32_yrgb.val[1] = Q6_Ww_vmpyacc_WwVhRh(ws32_ysub_hi, Q6_V_hi_W(ws16_yrgb), 0x614d614d);

    CvtYxuv2XCore(w2s32_yrgb, w2s32_ruv, v3u8_dst.val[0]);
    CvtYxuv2XCore(w2s32_yrgb, w2s32_guv, v3u8_dst.val[1]);
    CvtYxuv2XCore(w2s32_yrgb, w2s32_buv, v3u8_dst.val[2]);
}

template <MI_U32 MODE, typename std::enable_if<1 == MODE>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CvtY1rgbuv2RgbCore(const HVX_Vector &vu8_src_y, const HVX_VectorPairX2 &w2s32_ruv,
                                              const HVX_VectorPairX2 &w2s32_guv, const HVX_VectorPairX2 &w2s32_buv,
                                              HVX_VectorX3 &v3u8_dst)
{
    HVX_VectorPair ws16_yrgb = Q6_Wh_vmpy_VubRb(vu8_src_y, 0x40404040);

    HVX_VectorPairX2 w2s32_yrgb;
    w2s32_yrgb.val[0] = Q6_Ww_vmpy_VhRh(Q6_V_lo_W(ws16_yrgb), 0x40004000);
    w2s32_yrgb.val[1] = Q6_Ww_vmpy_VhRh(Q6_V_hi_W(ws16_yrgb), 0x40004000);

    CvtYxuv2XCore(w2s32_yrgb, w2s32_ruv, v3u8_dst.val[0]);
    CvtYxuv2XCore(w2s32_yrgb, w2s32_guv, v3u8_dst.val[1]);
    CvtYxuv2XCore(w2s32_yrgb, w2s32_buv, v3u8_dst.val[2]);
}

template <MI_U32 MODE>
static Status CvtNv2RgbHvxImpl(const Mat &src_y, const Mat &src_uv, Mat &dst, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    // 输出数据的宽度
    const MI_S32 width       = dst.GetSizes().m_width;
    //
    const MI_S32 width_align = width & (-(AURA_HVLEN * 2));

    const MI_S32 iwidth0   = src_y.GetSizes().m_width;
    const MI_S32 iwidth1   = src_uv.GetSizes().m_width;
    const MI_S32 ichannel0 = src_y.GetSizes().m_channel;
    const MI_S32 ichannel1 = src_uv.GetSizes().m_channel;
    const MI_S32 istride0  = src_y.GetStrides().m_width;
    const MI_S32 istride1  = src_uv.GetStrides().m_width;

    const MI_S32 uidx = swapuv;
    const MI_S32 vidx = 1 - uidx;

    MI_U64 L2fetch_param0 = L2PfParam(istride0, iwidth0 * ichannel0 * ElemTypeSize(src_y.GetElemType()), 2, 0);
    MI_U64 L2fetch_param1 = L2PfParam(istride1, iwidth1 * ichannel1 * ElemTypeSize(src_uv.GetElemType()), 1, 0);

    HVX_VectorX2     v2u8_src_y_c, v2u8_src_y_n, v2u8_src_uv;
    HVX_VectorX3     v3u8_dst_c_lo, v3u8_dst_c_hi, v3u8_dst_n_lo, v3u8_dst_n_hi;
    HVX_VectorPairX2 w2s32_ruv, w2s32_guv, w2s32_buv;

    for (MI_S32 uv = start_row; uv < end_row; uv++)
    {
        MI_S32 y = uv * 2;
        if (uv < end_row - 1)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src_uv.Ptr<MI_U8>(uv + 1)), L2fetch_param1);
            L2Fetch(reinterpret_cast<MI_U32>(src_y.Ptr<MI_U8>(y + 2)), L2fetch_param0);
        }

        const MI_U8 *src_y_c  = src_y.Ptr<MI_U8>(y);
        const MI_U8 *src_y_n  = src_y.Ptr<MI_U8>(y + 1);
        const MI_U8 *src_uv_c = src_uv.Ptr<MI_U8>(uv);

        MI_U8 *dst_c = dst.Ptr<MI_U8>(y);
        MI_U8 *dst_n = dst.Ptr<MI_U8>(y + 1);

        MI_S32 x = 0;
        for (; x < width_align; x += (AURA_HVLEN * 2))
        {
LOOP_BODY:
            vload(src_uv_c + x, v2u8_src_uv);
            vload(src_y_c  + x, v2u8_src_y_c);
            vload(src_y_n  + x, v2u8_src_y_n);

            CvtUv2RgbuvCore<MODE>(v2u8_src_uv.val[uidx], v2u8_src_uv.val[vidx], w2s32_ruv, w2s32_guv, w2s32_buv);
            CvtY1rgbuv2RgbCore<MODE>(v2u8_src_y_c.val[0], w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst_c_lo);
            CvtY1rgbuv2RgbCore<MODE>(v2u8_src_y_c.val[1], w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst_c_hi);
            CvtY1rgbuv2RgbCore<MODE>(v2u8_src_y_n.val[0], w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst_n_lo);
            CvtY1rgbuv2RgbCore<MODE>(v2u8_src_y_n.val[1], w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst_n_hi);
            CvtShuffU8C3Core(v3u8_dst_c_lo, v3u8_dst_c_hi);
            CvtShuffU8C3Core(v3u8_dst_n_lo, v3u8_dst_n_hi);

            vstore(dst_c + x * 3,                v3u8_dst_c_lo);
            vstore(dst_c + (x + AURA_HVLEN) * 3, v3u8_dst_c_hi);
            vstore(dst_n + x * 3,                v3u8_dst_n_lo);
            vstore(dst_n + (x + AURA_HVLEN) * 3, v3u8_dst_n_hi);
        }

        if (x < width)
        {
            x = width - (AURA_HVLEN * 2);
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtNv2RgbHvx(Context *ctx, const Mat &src_y, const Mat &src_uv, Mat &dst, MI_BOOL swapuv, CvtColorType type)
{
    Status ret = Status::ERROR;

    if ((src_y.GetSizes().m_width & 1) || (src_y.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats size only support even");
        return ret;
    }

    if (src_y.GetSizes().m_width != dst.GetSizes().m_width || src_y.GetSizes().m_height != dst.GetSizes().m_height ||
        src_uv.GetSizes().m_width != (src_y.GetSizes().m_width >> 1) || src_uv.GetSizes().m_height != (src_y.GetSizes().m_height >> 1) ||
        src_y.GetSizes().m_channel != 1 || src_uv.GetSizes().m_channel != 2 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    MI_S32 height = src_uv.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (type)
    {
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtNv2RgbHvxImpl<0>, std::cref(src_y), std::cref(src_uv), std::ref(dst), swapuv);
            break;
        }

        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtNv2RgbHvxImpl<1>, std::cref(src_y), std::cref(src_uv), std::ref(dst), swapuv);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
static Status CvtY4202RgbHvxImpl(const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    const MI_S32 width       = dst.GetSizes().m_width;
    const MI_S32 width_align = width & (-(AURA_HVLEN * 2));

    const MI_S32 iwidth0  = src_y.GetSizes().m_width;
    const MI_S32 iwidth1  = src_u.GetSizes().m_width;
    const MI_S32 iwidth2  = src_v.GetSizes().m_width;
    const MI_S32 istride0 = src_y.GetStrides().m_width;
    const MI_S32 istride1 = src_u.GetStrides().m_width;
    const MI_S32 istride2 = src_v.GetStrides().m_width;

    MI_U64 L2fetch_param0 = L2PfParam(istride0, iwidth0 * ElemTypeSize(src_y.GetElemType()), 2, 0);
    MI_U64 L2fetch_param1 = L2PfParam(istride1, iwidth1 * ElemTypeSize(src_u.GetElemType()), 1, 0);
    MI_U64 L2fetch_param2 = L2PfParam(istride2, iwidth2 * ElemTypeSize(src_v.GetElemType()), 1, 0);

    HVX_Vector   vu8_src_u, vu8_src_v;
    HVX_VectorX2 v2u8_src_y_c, v2u8_src_y_n;

    HVX_VectorPairX2 w2s32_ruv, w2s32_guv, w2s32_buv;
    HVX_VectorX3     v3u8_dst_c_lo, v3u8_dst_c_hi, v3u8_dst_n_lo, v3u8_dst_n_hi;

    for (MI_S32 uv = start_row; uv < end_row; uv++)
    {
        MI_S32 y = uv * 2;
        if (uv < end_row - 1)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src_y.Ptr<MI_U8>(y  + 2)), L2fetch_param0);
            L2Fetch(reinterpret_cast<MI_U32>(src_u.Ptr<MI_U8>(uv + 1)), L2fetch_param1);
            L2Fetch(reinterpret_cast<MI_U32>(src_v.Ptr<MI_U8>(uv + 1)), L2fetch_param2);
        }

        const MI_U8 *src_y_c = src_y.Ptr<MI_U8>(y);
        const MI_U8 *src_y_n = src_y.Ptr<MI_U8>(y + 1);
        const MI_U8 *src_u_c = swapuv ? src_v.Ptr<MI_U8>(uv) : src_u.Ptr<MI_U8>(uv);
        const MI_U8 *src_v_c = swapuv ? src_u.Ptr<MI_U8>(uv) : src_v.Ptr<MI_U8>(uv);

        MI_U8 *dst_c = dst.Ptr<MI_U8>(y);
        MI_U8 *dst_n = dst.Ptr<MI_U8>(y + 1);

        MI_S32 x = 0;
        for (; x < width_align; x += (AURA_HVLEN * 2))
        {
LOOP_BODY:
            vload(src_u_c + (x >> 1), vu8_src_u);
            vload(src_v_c + (x >> 1), vu8_src_v);
            vload(src_y_c + x,        v2u8_src_y_c);
            vload(src_y_n + x,        v2u8_src_y_n);

            CvtUv2RgbuvCore<MODE>(vu8_src_u, vu8_src_v, w2s32_ruv, w2s32_guv, w2s32_buv);
            CvtY1rgbuv2RgbCore<MODE>(v2u8_src_y_c.val[0], w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst_c_lo);
            CvtY1rgbuv2RgbCore<MODE>(v2u8_src_y_c.val[1], w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst_c_hi);
            CvtY1rgbuv2RgbCore<MODE>(v2u8_src_y_n.val[0], w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst_n_lo);
            CvtY1rgbuv2RgbCore<MODE>(v2u8_src_y_n.val[1], w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst_n_hi);
            CvtShuffU8C3Core(v3u8_dst_c_lo, v3u8_dst_c_hi);
            CvtShuffU8C3Core(v3u8_dst_n_lo, v3u8_dst_n_hi);

            vstore(dst_c + x * 3,                v3u8_dst_c_lo);
            vstore(dst_c + (x + AURA_HVLEN) * 3, v3u8_dst_c_hi);
            vstore(dst_n + x * 3,                v3u8_dst_n_lo);
            vstore(dst_n + (x + AURA_HVLEN) * 3, v3u8_dst_n_hi);
        }

        if (x < width)
        {
            x = width - (AURA_HVLEN * 2);
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtY4202RgbHvx(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, MI_BOOL swapuv, CvtColorType type)
{
    Status ret = Status::ERROR;

    if ((src_y.GetSizes().m_width & 1) || (src_y.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats size only support even");
        return ret;
    }

    if (src_y.GetSizes().m_width != dst.GetSizes().m_width || src_y.GetSizes().m_height != dst.GetSizes().m_height ||
        src_u.GetSizes().m_width != src_v.GetSizes().m_width || src_u.GetSizes().m_height != src_v.GetSizes().m_height ||
        src_u.GetSizes().m_width != (src_y.GetSizes().m_width >> 1) || src_u.GetSizes().m_height != (src_y.GetSizes().m_height >> 1) ||
        src_y.GetSizes().m_channel != 1 || src_u.GetSizes().m_channel != 1 || src_v.GetSizes().m_channel != 1 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    MI_S32 height = src_u.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (type)
    {
        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtY4202RgbHvxImpl<0>, std::cref(src_y), std::cref(src_u), std::cref(src_v),
                                  std::ref(dst), swapuv);
            break;
        }

        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtY4202RgbHvxImpl<1>, std::cref(src_y), std::cref(src_u), std::cref(src_v),
                                  std::ref(dst), swapuv);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
static Status CvtY4222RgbHvxImpl(const Mat &src, Mat &dst, MI_BOOL swapuv, MI_BOOL swapy, MI_S32 start_row, MI_S32 end_row)
{
    const MI_S32 width       = dst.GetSizes().m_width;
    const MI_S32 width_align = width & (-(AURA_HVLEN * 2));

    const MI_S32 ichannel = src.GetSizes().m_channel;
    const MI_S32 istride  = src.GetStrides().m_width;

    const MI_S32 uidx = 1 - swapy + swapuv * 2;
    const MI_S32 vidx = (2 + uidx) % 4;

    MI_U64 L2fetch_param = L2PfParam(istride, width * ichannel * ElemTypeSize(src.GetElemType()), 1, 0);

    HVX_VectorX4     v4u8_src;
    HVX_VectorX3     v3u8_dst0, v3u8_dst1;
    HVX_VectorPairX2 w2s32_ruv, w2s32_guv, w2s32_buv;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y < end_row - 1)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<MI_U8>(y + 1)), L2fetch_param);
        }

        const MI_U8 *src_c = src.Ptr<MI_U8>(y);
        MI_U8       *dst_c = dst.Ptr<MI_U8>(y);

        MI_S32 x = 0;
        for (; x < width_align; x += (AURA_HVLEN * 2))
        {
LOOP_BODY:
            vload(src_c + x * 2, v4u8_src);

            CvtUv2RgbuvCore<MODE>(v4u8_src.val[uidx], v4u8_src.val[vidx], w2s32_ruv, w2s32_guv, w2s32_buv);
            CvtY1rgbuv2RgbCore<MODE>(v4u8_src.val[(MI_S32)swapy], w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst0);
            CvtY1rgbuv2RgbCore<MODE>(v4u8_src.val[(MI_S32)swapy + 2], w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst1);
            CvtShuffU8C3Core(v3u8_dst0, v3u8_dst1);

            vstore(dst_c + x * 3,                v3u8_dst0);
            vstore(dst_c + (x + AURA_HVLEN) * 3, v3u8_dst1);
        }

        if (x < width)
        {
            x = width - (AURA_HVLEN * 2);
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtY4222RgbHvx(Context *ctx, const Mat &src, Mat &dst, MI_BOOL swapuv, MI_BOOL swapy, CvtColorType type)
{
    Status ret = Status::ERROR;

    if (src.GetSizes().m_width & 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "src width only support even");
        return ret;
    }

    if ((src.GetSizes().m_width != dst.GetSizes().m_width) || (src.GetSizes().m_height != dst.GetSizes().m_height) ||
        (src.GetSizes().m_channel != 2) || (dst.GetSizes().m_channel != 3))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    MI_S32 height = src.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (type)
    {
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtY4222RgbHvxImpl<0>, std::cref(src), std::ref(dst), swapuv, swapy);
            break;
        }

        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtY4222RgbHvxImpl<1>, std::cref(src), std::ref(dst), swapuv, swapy);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <MI_U32 MODE>
static Status CvtY4442RgbHvxImpl(const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    const MI_S32 width       = dst.GetSizes().m_width;
    // AURA_HVLEN这个字段是什么样？？
    const MI_S32 width_align = width & (-AURA_HVLEN);
    // Y分量的的宽度 1024
    const MI_S32 istride0 = src_y.GetStrides().m_width;
    // U分量的宽度 512
    const MI_S32 istride1 = src_u.GetStrides().m_width;
    /// V 分量的大小
    const MI_S32 istride2 = src_v.GetStrides().m_width;

    // L2PfParam是高通Hexagon HVX架构中用于配置L2缓存预取器参数的接口，主要针对矢量运算场景优化内存访问效率。返回的MI_U64
    // 该函数通过设置步幅、数据块宽度等参数，指导HVX的L2预取器预测后续内存访问模式并提前加载数据到缓存，从而减少缓存未命中（Cache Miss）带来的延迟
    // 适用于需要连续访问大块数据的场景（如图像处理、矩阵运算）
    // istride0: 内存访问的步长（Stride），即相邻数据行之间的字节偏移量。
    // 
    // 可能表示预取次数或预取模式。例如，设置为1时表示连续预取一次，或启用某种特定的预取策略。
    MI_U64 L2fetch_param0 = L2PfParam(istride0, width * ElemTypeSize(src_y.GetElemType()), 1, 0);
    MI_U64 L2fetch_param1 = L2PfParam(istride1, width * ElemTypeSize(src_u.GetElemType()), 1, 0);
    MI_U64 L2fetch_param2 = L2PfParam(istride2, width * ElemTypeSize(src_v.GetElemType()), 1, 0);

    HVX_Vector       vu8_src_u, vu8_src_v, vu8_src_y;
    HVX_VectorX3     v3u8_dst;
    HVX_VectorPairX2 w2s32_ruv, w2s32_guv, w2s32_buv;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y < end_row - 1)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src_u.Ptr<MI_U8>(y + 1)), L2fetch_param1);
            L2Fetch(reinterpret_cast<MI_U32>(src_v.Ptr<MI_U8>(y + 1)), L2fetch_param2);
            L2Fetch(reinterpret_cast<MI_U32>(src_y.Ptr<MI_U8>(y + 1)), L2fetch_param0);
        }

        const MI_U8 *src_y_c = src_y.Ptr<MI_U8>(y);
        const MI_U8 *src_u_c = src_u.Ptr<MI_U8>(y);
        const MI_U8 *src_v_c = src_v.Ptr<MI_U8>(y);

        MI_U8 *dst_c = dst.Ptr<MI_U8>(y);

        MI_S32 x = 0;
        for (; x < width_align; x += AURA_HVLEN)
        {
LOOP_BODY:
            vload(src_u_c + x, vu8_src_u);
            vload(src_v_c + x, vu8_src_v);
            vload(src_y_c + x, vu8_src_y);

            CvtUv2RgbuvCore<MODE>(vu8_src_u, vu8_src_v, w2s32_ruv, w2s32_guv, w2s32_buv);
            CvtY1rgbuv2RgbCore<MODE>(vu8_src_y, w2s32_ruv, w2s32_guv, w2s32_buv, v3u8_dst);

            vstore(dst_c + x * 3, v3u8_dst);
        }

        if (x < width)
        {
            x = width - AURA_HVLEN;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtY4442RgbHvx(Context *ctx, const Mat &src_y, const Mat &src_u, const Mat &src_v, Mat &dst, CvtColorType type)
{
    Status ret = Status::ERROR;

    if (src_y.GetSizes().m_width != dst.GetSizes().m_width || src_y.GetSizes().m_height != dst.GetSizes().m_height ||
        src_u.GetSizes().m_width != dst.GetSizes().m_width || src_u.GetSizes().m_height != dst.GetSizes().m_height ||
        src_v.GetSizes().m_width != dst.GetSizes().m_width || src_v.GetSizes().m_height != dst.GetSizes().m_height ||
        src_y.GetSizes().m_channel != 1 || src_u.GetSizes().m_channel != 1 || src_v.GetSizes().m_channel != 1 || dst.GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats params do not match");
        return ret;
    }

    MI_S32 height = dst.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (type)
    {
        case CvtColorType::YUV2RGB_Y444:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtY4442RgbHvxImpl<0>, std::cref(src_y), std::cref(src_u), std::cref(src_v),
                                  std::ref(dst));
            break;
        }

        case CvtColorType::YUV2RGB_Y444_601:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtY4442RgbHvxImpl<1>, std::cref(src_y), std::cref(src_u), std::cref(src_v),
                                  std::ref(dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura