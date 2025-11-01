#include "cvtcolor_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{
/**
 * @brief the formula of RGB -> YUV
 *  Y = 16  + 0.257 * r + 0.504 * g + 0.098 * b
 * Cb = 128 - 0.148 * r - 0.291 * g + 0.439 * b
 * Cr = 128 + 0.439 * r - 0.368 * g - 0.071 * b
 *
 * r2y:  269484  =  12  (0x0c) * 22457 (0x57b9);  // round( 0.257f * (1 << CVTCOLOR_COEF_BITS));
 * g2y:  528482  =  89  (0x59) * 5938  (0x1732);  // round( 0.504f * (1 << CVTCOLOR_COEF_BITS));
 * b2y:  102760  =  4   (0x04) * 25690 (0x645a);  // round( 0.098f * (1 << CVTCOLOR_COEF_BITS));
 * r2u: -155188  = -11  (0xf5) * 14108 (0x371c);  // round(-0.148f * (1 << CVTCOLOR_COEF_BITS)) + 1;
 * g2u: -305135  = -5   (0xfb) * 61027 (0xee63);  // round(-0.291f * (1 << CVTCOLOR_COEF_BITS)) + 1;
 * b2u:  460324  =  157 (0x9d) * 2932  (0x0b74);  // round( 0.439f * (1 << CVTCOLOR_COEF_BITS));
 * r2v:  460324  =  157 (0x9d) * 2932  (0x0b74);  // round( 0.439f * (1 << CVTCOLOR_COEF_BITS));
 * g2v: -385875  = -15  (0xf1) * 25725 (0x647d);  // round(-0.368f * (1 << CVTCOLOR_COEF_BITS)) + 1;
 * b2v: -74448   = -3   (0xfd) * 24816 (0x60f0);  // round(-0.071f * (1 << CVTCOLOR_COEF_BITS)) + 1;
 *
 * 16777216  (0x01000000): 16  * (1 << CVTCOLOR_COEF_BITS)
 * 134217728 (0x08000000): 128 * (1 << CVTCOLOR_COEF_BITS)
 */

/**
 * @brief the formula of BT601 RGB -> YUV
 *  Y =  0.299  * r + 0.587  * g + 0.114  * b
 * Cb = -0.1687 * r - 0.3313 * g + 0.5    * b + 128
 * Cr =  0.5    * r - 0.4187 * g - 0.0813 * b + 128
 *
 * @brief the formula of P010 RGB -> YUV
 *  Y = ( 0.299  * r + 0.587  * g + 0.114  * b)       * (1 << 6)
 * Cb = (-0.1687 * r - 0.3313 * g + 0.5    * b + 512) * (1 << 6)
 * Cr = ( 0.5    * r - 0.4187 * g - 0.0813 * b + 512) * (1 << 6)
 *
 * r2y:  313524  =  12  (0x0c) *  26127 (0x660f);  // round( 0.299f * (1 << CVTCOLOR_COEF_BITS));
 * g2y:  615514  =  241 (0xf1) *  2554  (0x09fa);  // round( 0.587f * (1 << CVTCOLOR_COEF_BITS));
 * b2y:  150994  =  6   (0x06) *  19923 (0x4dd3);  // round( 0.144f * (1 << CVTCOLOR_COEF_BITS));
 * r2u: -177208  = -9   (0xf7) *  19655 (0x4cc7);  // round(-0.169f * (1 << CVTCOLOR_COEF_BITS));
 * g2u: -347077  = -37  (0xdb) *  9389  (0x24ad);  // round(-0.331f * (1 << CVTCOLOR_COEF_BITS));
 * b2u:  524288  =  32  (0x20) *  16384 (0x4000);  // round( 0.500f * (1 << CVTCOLOR_COEF_BITS));
 * r2v:  524288  =  32  (0x20) *  16384 (0x4000);  // round( 0.500f * (1 << CVTCOLOR_COEF_BITS));
 * g2v: -439352  = -127 (0x81) *  3457  (0x0d81);  // round(-0.419f * (1 << CVTCOLOR_COEF_BITS));
 * b2v: -84933   =  163 (0xa3) * -523   (0xfdf5);  // round(-0.081f * (1 << CVTCOLOR_COEF_BITS));
 *
 * 134217728 (0x08000000): 128 * (1 << CVTCOLOR_COEF_BITS)
 * 536870912 (0x20000000): 512 * (1 << CVTCOLOR_COEF_BITS)
 */

AURA_ALWAYS_INLINE AURA_VOID CvtAsrShiftCastU8(HVX_VectorPair &ws32_x_lo, HVX_VectorPair &ws32_x_hi, HVX_Vector &vu8_x)
{
    HVX_Vector vs32_x_lo_lo = Q6_Vw_vmax_VwVw(Q6_V_lo_W(ws32_x_lo), Q6_V_vzero());
    HVX_Vector vs32_x_lo_hi = Q6_Vw_vmax_VwVw(Q6_V_hi_W(ws32_x_lo), Q6_V_vzero());
    HVX_Vector vs32_x_hi_lo = Q6_Vw_vmax_VwVw(Q6_V_lo_W(ws32_x_hi), Q6_V_vzero());
    HVX_Vector vs32_x_hi_hi = Q6_Vw_vmax_VwVw(Q6_V_hi_W(ws32_x_hi), Q6_V_vzero());

    HVX_Vector vs16_x_lo = Q6_Vh_vasr_VwVwR(vs32_x_lo_hi, vs32_x_lo_lo, 15);
    HVX_Vector vs16_x_hi = Q6_Vh_vasr_VwVwR(vs32_x_hi_hi, vs32_x_hi_lo, 15);

    vu8_x = Q6_Vub_vasr_VuhVuhR_rnd_sat(vs16_x_hi, vs16_x_lo, 5);
}

AURA_ALWAYS_INLINE AURA_VOID CvtDealRgbCore(HVX_VectorX3 &v3_src0, HVX_VectorX3 &v3_src1, MI_S32 rt)
{
    HVX_VectorPair w_r = Q6_W_vdeal_VVR(v3_src1.val[0], v3_src0.val[0], rt);
    HVX_VectorPair w_g = Q6_W_vdeal_VVR(v3_src1.val[1], v3_src0.val[1], rt);
    HVX_VectorPair w_b = Q6_W_vdeal_VVR(v3_src1.val[2], v3_src0.val[2], rt);

    v3_src0.val[0] = Q6_V_lo_W(w_r);
    v3_src0.val[1] = Q6_V_lo_W(w_g);
    v3_src0.val[2] = Q6_V_lo_W(w_b);

    v3_src1.val[0] = Q6_V_hi_W(w_r);
    v3_src1.val[1] = Q6_V_hi_W(w_g);
    v3_src1.val[2] = Q6_V_hi_W(w_b);
}

template <MI_U32 MODE, typename std::enable_if<0 == MODE>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2YCore(HVX_VectorX3 &v3u8_src, HVX_Vector &vu8_dst_y)
{
    HVX_VectorPair ws32_yc = Q6_W_vcombine_VV(Q6_V_vsplat_R(0x01000000), Q6_V_vsplat_R(0x01000000));

    HVX_VectorPair ws16_yr = Q6_Wh_vmpy_VubRb(v3u8_src.val[0], 0x0c0c0c0c);
    HVX_VectorPair ws16_yg = Q6_Wh_vmpy_VubRb(v3u8_src.val[1], 0x59595959);
    HVX_VectorPair ws16_yb = Q6_Wh_vmpy_VubRb(v3u8_src.val[2], 0x04040404);

    HVX_VectorPair ws32_y_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_yc, Q6_V_lo_W(ws16_yr), 0x57b957b9);
    HVX_VectorPair ws32_y_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_yc, Q6_V_hi_W(ws16_yr), 0x57b957b9);

    ws32_y_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_y_lo, Q6_V_lo_W(ws16_yg), 0x17321732);
    ws32_y_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_y_hi, Q6_V_hi_W(ws16_yg), 0x17321732);
    ws32_y_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_y_lo, Q6_V_lo_W(ws16_yb), 0x645a645a);
    ws32_y_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_y_hi, Q6_V_hi_W(ws16_yb), 0x645a645a);

    CvtAsrShiftCastU8(ws32_y_lo, ws32_y_hi, vu8_dst_y);
}

template <MI_U32 MODE, typename std::enable_if<1 == MODE>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2YCore(HVX_VectorX3 &v3u8_src, HVX_Vector &vu8_dst_y)
{
    HVX_VectorPair ws16_yr = Q6_Wh_vmpy_VubRb(v3u8_src.val[0], 0x0c0c0c0c);
    HVX_VectorPair wu16_yg = Q6_Wuh_vmpy_VubRub(v3u8_src.val[1], 0xf1f1f1f1);
    HVX_VectorPair ws16_yb = Q6_Wh_vmpy_VubRb(v3u8_src.val[2], 0x06060606);

    HVX_VectorPair ws32_y_lo = Q6_Ww_vmpy_VhRh(Q6_V_lo_W(ws16_yr), 0x660f660f);
    HVX_VectorPair ws32_y_hi = Q6_Ww_vmpy_VhRh(Q6_V_hi_W(ws16_yr), 0x660f660f);

    ws32_y_lo = Q6_Ww_vmpyacc_WwVhVuh(ws32_y_lo, Q6_V_vsplat_R(0x09fa09fa), Q6_V_lo_W(wu16_yg));
    ws32_y_hi = Q6_Ww_vmpyacc_WwVhVuh(ws32_y_hi, Q6_V_vsplat_R(0x09fa09fa), Q6_V_hi_W(wu16_yg));
    ws32_y_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_y_lo, Q6_V_lo_W(ws16_yb), 0x4dd34dd3);
    ws32_y_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_y_hi, Q6_V_hi_W(ws16_yb), 0x4dd34dd3);

    CvtAsrShiftCastU8(ws32_y_lo, ws32_y_hi, vu8_dst_y);
}

template <MI_U32 MODE, typename std::enable_if<0 == MODE>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2UVCore(HVX_VectorX3 &v3u8_src, HVX_Vector &vu8_dst_u, HVX_Vector &vu8_dst_v)
{
    HVX_VectorPair ws32_uc = Q6_W_vcombine_VV(Q6_V_vsplat_R(0x08000000), Q6_V_vsplat_R(0x08000000));

    // u
    HVX_VectorPair ws16_ur = Q6_Wh_vmpy_VubRb(v3u8_src.val[0], 0xf5f5f5f5);
    HVX_VectorPair ws16_ug = Q6_Wh_vmpy_VubRb(v3u8_src.val[1], 0xfbfbfbfb);
    HVX_VectorPair wu16_ub = Q6_Wuh_vmpy_VubRub(v3u8_src.val[2], 0x9d9d9d9d);

    HVX_VectorPair ws32_u_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_uc, Q6_V_lo_W(ws16_ur), 0x371c371c);
    HVX_VectorPair ws32_u_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_uc, Q6_V_hi_W(ws16_ur), 0x371c371c);

    ws32_u_lo = Q6_Ww_vmpyacc_WwVhVuh(ws32_u_lo, Q6_V_lo_W(ws16_ug), Q6_V_vsplat_R(0xee63ee63));
    ws32_u_hi = Q6_Ww_vmpyacc_WwVhVuh(ws32_u_hi, Q6_V_hi_W(ws16_ug), Q6_V_vsplat_R(0xee63ee63));
    ws32_u_lo = Q6_Ww_vmpyacc_WwVhVuh(ws32_u_lo, Q6_V_vsplat_R(0x0b740b74), Q6_V_lo_W(wu16_ub));
    ws32_u_hi = Q6_Ww_vmpyacc_WwVhVuh(ws32_u_hi, Q6_V_vsplat_R(0x0b740b74), Q6_V_hi_W(wu16_ub));

    CvtAsrShiftCastU8(ws32_u_lo, ws32_u_hi, vu8_dst_u);

    // v
    HVX_VectorPair wu16_vr = Q6_Wuh_vmpy_VubRub(v3u8_src.val[0], 0x9d9d9d9d);
    HVX_VectorPair ws16_vg = Q6_Wh_vmpy_VubRb(v3u8_src.val[1], 0xf1f1f1f1);
    HVX_VectorPair ws16_vb = Q6_Wh_vmpy_VubRb(v3u8_src.val[2], 0xfdfdfdfd);

    HVX_VectorPair ws32_v_lo = Q6_Ww_vmpyacc_WwVhVuh(ws32_uc, Q6_V_vsplat_R(0x0b740b74), Q6_V_lo_W(wu16_vr));
    HVX_VectorPair ws32_v_hi = Q6_Ww_vmpyacc_WwVhVuh(ws32_uc, Q6_V_vsplat_R(0x0b740b74), Q6_V_hi_W(wu16_vr));

    ws32_v_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_v_lo, Q6_V_lo_W(ws16_vg), 0x647d647d);
    ws32_v_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_v_hi, Q6_V_hi_W(ws16_vg), 0x647d647d);
    ws32_v_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_v_lo, Q6_V_lo_W(ws16_vb), 0x60f060f0);
    ws32_v_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_v_hi, Q6_V_hi_W(ws16_vb), 0x60f060f0);

    CvtAsrShiftCastU8(ws32_v_lo, ws32_v_hi, vu8_dst_v);
}

template <MI_U32 MODE, typename std::enable_if<1 == MODE>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CvtRgb2UVCore(HVX_VectorX3 &v3u8_src, HVX_Vector &vu8_dst_u, HVX_Vector &vu8_dst_v)
{
    HVX_VectorPair ws32_uc = Q6_W_vcombine_VV(Q6_V_vsplat_R(0x08000000), Q6_V_vsplat_R(0x08000000));

    // u
    HVX_VectorPair ws16_ur = Q6_Wh_vmpy_VubRb(v3u8_src.val[0], 0xf7f7f7f7);
    HVX_VectorPair ws16_ug = Q6_Wh_vmpy_VubRb(v3u8_src.val[1], 0xdbdbdbdb);
    HVX_VectorPair ws16_ub = Q6_Wh_vmpy_VubRb(v3u8_src.val[2], 0x20202020);

    HVX_VectorPair ws32_u_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_uc, Q6_V_lo_W(ws16_ur), 0x4cc74cc7);
    HVX_VectorPair ws32_u_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_uc, Q6_V_hi_W(ws16_ur), 0x4cc74cc7);

    ws32_u_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_u_lo, Q6_V_lo_W(ws16_ug), 0x24ad24ad);
    ws32_u_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_u_hi, Q6_V_hi_W(ws16_ug), 0x24ad24ad);
    ws32_u_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_u_lo, Q6_V_lo_W(ws16_ub), 0x40004000);
    ws32_u_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_u_hi, Q6_V_hi_W(ws16_ub), 0x40004000);

    CvtAsrShiftCastU8(ws32_u_lo, ws32_u_hi, vu8_dst_u);

    // v
    HVX_VectorPair ws16_vr = Q6_Wh_vmpy_VubRb(v3u8_src.val[0], 0x20202020);
    HVX_VectorPair ws16_vg = Q6_Wh_vmpy_VubRb(v3u8_src.val[1], 0x81818181);
    HVX_VectorPair ws16_vb = Q6_Wuh_vmpy_VubRub(v3u8_src.val[2], 0xa3a3a3a3);

    HVX_VectorPair ws32_v_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_uc, Q6_V_lo_W(ws16_vr), 0x40004000);
    HVX_VectorPair ws32_v_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_uc, Q6_V_hi_W(ws16_vr), 0x40004000);

    ws32_v_lo = Q6_Ww_vmpyacc_WwVhRh(ws32_v_lo, Q6_V_lo_W(ws16_vg), 0x0d810d81);
    ws32_v_hi = Q6_Ww_vmpyacc_WwVhRh(ws32_v_hi, Q6_V_hi_W(ws16_vg), 0x0d810d81);
    ws32_v_lo = Q6_Ww_vmpyacc_WwVhVuh(ws32_v_lo, Q6_V_vsplat_R(0xfdf5fdf5), Q6_V_lo_W(ws16_vb));
    ws32_v_hi = Q6_Ww_vmpyacc_WwVhVuh(ws32_v_hi, Q6_V_vsplat_R(0xfdf5fdf5), Q6_V_hi_W(ws16_vb));

    CvtAsrShiftCastU8(ws32_v_lo, ws32_v_hi, vu8_dst_v);
}

template <MI_U32 MODE>
static Status CvtRgb2NvHvxImpl(const Mat &src, Mat &dst_y, Mat &dst_uv, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    const MI_S32 width       = src.GetSizes().m_width;
    const MI_S32 ichannel    = src.GetSizes().m_channel;
    const MI_S32 istride     = src.GetStrides().m_width;
    const MI_S32 width_align = width & (-(AURA_HVLEN * 2));
    const MI_S32 uidx        = swapuv;
    const MI_S32 vidx        = 1 - uidx;

    MI_U64 L2fetch_param = L2PfParam(istride, width * ichannel * ElemTypeSize(src.GetElemType()), 2, 0);

    HVX_VectorX3 v3u8_src_c_lo, v3u8_src_c_hi, v3u8_src_n_lo, v3u8_src_n_hi;
    HVX_VectorX2 v2u8_dst_y_c, v2u8_dst_y_n, v2u8_dst_uv;

    for (MI_S32 uv = start_row; uv < end_row; uv++)
    {
        MI_S32 y = uv * 2;
        if (uv < end_row - 1)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<MI_U8>(y + 2)), L2fetch_param);
        }

        MI_U8 *src_c = (MI_U8 *)src.Ptr<MI_U8>(y);
        MI_U8 *src_n = (MI_U8 *)src.Ptr<MI_U8>(y + 1);

        MI_U8 *dst_y_c  = dst_y.Ptr<MI_U8>(y);
        MI_U8 *dst_y_n  = dst_y.Ptr<MI_U8>(y + 1);
        MI_U8 *dst_uv_c = dst_uv.Ptr<MI_U8>(uv);

        MI_S32 x = 0;
        for (; x < width_align; x += (AURA_HVLEN * 2))
        {
LOOP_BODY:
            vload(src_c + x * 3,                v3u8_src_c_lo);
            vload(src_c + (x + AURA_HVLEN) * 3, v3u8_src_c_hi);
            vload(src_n + x * 3,                v3u8_src_n_lo);
            vload(src_n + (x + AURA_HVLEN) * 3, v3u8_src_n_hi);

            CvtDealRgbCore(v3u8_src_c_lo, v3u8_src_c_hi, -1);
            CvtDealRgbCore(v3u8_src_n_lo, v3u8_src_n_hi, -1);

            CvtRgb2YCore<MODE>(v3u8_src_c_lo, v2u8_dst_y_c.val[0]);
            CvtRgb2YCore<MODE>(v3u8_src_c_hi, v2u8_dst_y_c.val[1]);
            CvtRgb2YCore<MODE>(v3u8_src_n_lo, v2u8_dst_y_n.val[0]);
            CvtRgb2YCore<MODE>(v3u8_src_n_hi, v2u8_dst_y_n.val[1]);
            CvtRgb2UVCore<MODE>(v3u8_src_c_lo, v2u8_dst_uv.val[uidx], v2u8_dst_uv.val[vidx]);

            vstore(dst_y_c + x,  v2u8_dst_y_c);
            vstore(dst_y_n + x,  v2u8_dst_y_n);
            vstore(dst_uv_c + x, v2u8_dst_uv);
        }

        if (x < width)
        {
            x = width - (AURA_HVLEN * 2);
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtRgb2NvHvx(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, MI_BOOL swapuv, CvtColorType type)
{
    Status ret = Status::ERROR;

    if ((src.GetSizes().m_width & 1) || (src.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src sizes only support even");
        return ret;
    }

    if (dst_y.GetSizes() * Sizes3(1, 1, 2) != dst_uv.GetSizes() * Sizes3(2, 2, 1) ||
        dst_y.GetSizes() * Sizes3(1, 1, 3) != src.GetSizes() ||
        dst_y.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "the sizes of src and dst do not match");
        return ret;
    }

    MI_S32 height = dst_uv.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (type)
    {
        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2NvHvxImpl<0>, std::cref(src), std::ref(dst_y), std::ref(dst_uv), swapuv);
            break;
        }

        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2NvHvxImpl<1>, std::cref(src), std::ref(dst_y), std::ref(dst_uv), swapuv);
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
static Status CvtRgb2Y420HvxImpl(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    const MI_S32 width       = src.GetSizes().m_width;
    const MI_S32 ichannel    = src.GetSizes().m_channel;
    const MI_S32 istride     = src.GetStrides().m_width;
    const MI_S32 width_align = width & (-(AURA_HVLEN * 2));

    MI_U64 L2fetch_param = L2PfParam(istride, width * ichannel * ElemTypeSize(src.GetElemType()), 2, 0);

    HVX_VectorX3 v3u8_src_c_lo, v3u8_src_c_hi, v3u8_src_n_lo, v3u8_src_n_hi;
    HVX_VectorX2 v2u8_dst_y_c, v2u8_dst_y_n;
    HVX_Vector   vu8_dst_u, vu8_dst_v;

    for (MI_S32 uv = start_row; uv < end_row; uv++)
    {
        MI_S32 y = uv * 2;
        if (uv < end_row - 1)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<MI_U8>(y + 2)), L2fetch_param);
        }

        const MI_U8 *src_c = src.Ptr<MI_U8>(y);
        const MI_U8 *src_n = src.Ptr<MI_U8>(y + 1);

        MI_U8 *dst_y_c = dst_y.Ptr<MI_U8>(y);
        MI_U8 *dst_y_n = dst_y.Ptr<MI_U8>(y + 1);
        MI_U8 *dst_u_c = swapuv ? dst_v.Ptr<MI_U8>(y >> 1) : dst_u.Ptr<MI_U8>(y >> 1);
        MI_U8 *dst_v_c = swapuv ? dst_u.Ptr<MI_U8>(y >> 1) : dst_v.Ptr<MI_U8>(y >> 1);

        MI_S32 x = 0;
        for (; x < width_align; x += (AURA_HVLEN * 2))
        {
LOOP_BODY:
            vload(src_c + x * 3,                v3u8_src_c_lo);
            vload(src_c + (x + AURA_HVLEN) * 3, v3u8_src_c_hi);
            vload(src_n + x * 3,                v3u8_src_n_lo);
            vload(src_n + (x + AURA_HVLEN) * 3, v3u8_src_n_hi);

            CvtDealRgbCore(v3u8_src_c_lo, v3u8_src_c_hi, -1);
            CvtDealRgbCore(v3u8_src_n_lo, v3u8_src_n_hi, -1);

            CvtRgb2YCore<MODE>(v3u8_src_c_lo, v2u8_dst_y_c.val[0]);
            CvtRgb2YCore<MODE>(v3u8_src_c_hi, v2u8_dst_y_c.val[1]);
            CvtRgb2YCore<MODE>(v3u8_src_n_lo, v2u8_dst_y_n.val[0]);
            CvtRgb2YCore<MODE>(v3u8_src_n_hi, v2u8_dst_y_n.val[1]);
            CvtRgb2UVCore<MODE>(v3u8_src_c_lo, vu8_dst_u, vu8_dst_v);

            vstore(dst_y_c + x,        v2u8_dst_y_c);
            vstore(dst_y_n + x,        v2u8_dst_y_n);
            vstore(dst_u_c + (x >> 1), vu8_dst_u);
            vstore(dst_v_c + (x >> 1), vu8_dst_v);
        }

        if (x < width)
        {
            x = width - (AURA_HVLEN * 2);
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtRgb2Y420Hvx(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_BOOL swapuv, CvtColorType type)
{
    Status ret = Status::ERROR;

    if ((src.GetSizes().m_width & 1) || (src.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src and dst mats size only support even");
        return ret;
    }

    if (dst_y.GetSizes() != dst_u.GetSizes() * Sizes3(2, 2, 1) ||
        dst_y.GetSizes() != dst_v.GetSizes() * Sizes3(2, 2, 1) ||
        dst_y.GetSizes() * Sizes3(1, 1, 3) != src.GetSizes() ||
        dst_y.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "the sizes of src and dst do not match");
        return ret;
    }

    MI_S32 height = dst_u.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    switch (type)
    {
        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2Y420HvxImpl<0>, std::cref(src), std::ref(dst_y), std::ref(dst_u),
                                  std::ref(dst_v), swapuv);
            break;
        }

        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2Y420HvxImpl<1>, std::cref(src), std::ref(dst_y), std::ref(dst_u),
                                  std::ref(dst_v), swapuv);
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
static Status CvtRgb2Y444HvxImpl(const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, MI_S32 start_row, MI_S32 end_row)
{
    const MI_S32 width       = src.GetSizes().m_width;
    const MI_S32 ichannel    = src.GetSizes().m_channel;
    const MI_S32 istride     = src.GetStrides().m_width;
    const MI_S32 width_align = width & (-AURA_HVLEN);

    MI_U64 L2fetch_param = L2PfParam(istride, width * ichannel * ElemTypeSize(src.GetElemType()), 1, 0);

    HVX_VectorX3 v3u8_src;
    HVX_Vector   vu8_dst_y, vu8_dst_u, vu8_dst_v;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y < end_row - 1)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<MI_U8>(y + 1)), L2fetch_param);
        }

        const MI_U8 *src_c = src.Ptr<MI_U8>(y);

        MI_U8 *dst_y_c = dst_y.Ptr<MI_U8>(y);
        MI_U8 *dst_u_c = dst_u.Ptr<MI_U8>(y);
        MI_U8 *dst_v_c = dst_v.Ptr<MI_U8>(y);

        MI_S32 x = 0;
        for (; x < width_align; x += AURA_HVLEN)
        {
LOOP_BODY:
            vload(src_c + x * 3, v3u8_src);

            CvtRgb2YCore<MODE>(v3u8_src, vu8_dst_y);
            CvtRgb2UVCore<MODE>(v3u8_src, vu8_dst_u, vu8_dst_v);

            vstore(dst_y_c + x, vu8_dst_y);
            vstore(dst_u_c + x, vu8_dst_u);
            vstore(dst_v_c + x, vu8_dst_v);
        }

        if (x < width)
        {
            x = width - AURA_HVLEN;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtRgb2Y444Hvx(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_u, Mat &dst_v, CvtColorType type)
{
    Status ret = Status::ERROR;

    if (dst_y.GetSizes() != dst_u.GetSizes() ||
        dst_y.GetSizes() != dst_v.GetSizes() ||
        dst_y.GetSizes() * Sizes3(1, 1, 3) != src.GetSizes() ||
        dst_y.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "the sizes of src and dst do not match");
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
        case CvtColorType::RGB2YUV_Y444:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2Y444HvxImpl<0>, std::cref(src), std::ref(dst_y), std::ref(dst_u), std::ref(dst_v));
            break;
        }

        case CvtColorType::RGB2YUV_Y444_601:
        {
            ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2Y444HvxImpl<1>, std::cref(src), std::ref(dst_y), std::ref(dst_u), std::ref(dst_v));
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

AURA_ALWAYS_INLINE AURA_VOID CvtRgb2YP010Core(HVX_VectorX3 &v3u16_src, HVX_VectorX3 &v3s32_rgb2y, HVX_Vector &vu16_y)
{
    HVX_Vector vs32_yr_lo = Q6_Vw_vmpyie_VwVuh(v3s32_rgb2y.val[0], v3u16_src.val[0]);
    HVX_Vector vs32_yg_lo = Q6_Vw_vmpyie_VwVuh(v3s32_rgb2y.val[1], v3u16_src.val[1]);
    HVX_Vector vs32_yb_lo = Q6_Vw_vmpyie_VwVuh(v3s32_rgb2y.val[2], v3u16_src.val[2]);

    HVX_Vector vs32_yr_hi = Q6_Vw_vmpyio_VwVh(v3s32_rgb2y.val[0], v3u16_src.val[0]);
    HVX_Vector vs32_yg_hi = Q6_Vw_vmpyio_VwVh(v3s32_rgb2y.val[1], v3u16_src.val[1]);
    HVX_Vector vs32_yb_hi = Q6_Vw_vmpyio_VwVh(v3s32_rgb2y.val[2], v3u16_src.val[2]);

    HVX_Vector vs32_y_lo = Q6_Vw_vadd_VwVw(vs32_yr_lo, vs32_yg_lo);
    HVX_Vector vs32_y_hi = Q6_Vw_vadd_VwVw(vs32_yr_hi, vs32_yg_hi);

    vs32_y_lo = Q6_Vw_vadd_VwVw(vs32_y_lo, vs32_yb_lo);
    vs32_y_hi = Q6_Vw_vadd_VwVw(vs32_y_hi, vs32_yb_hi);

    vu16_y = Q6_Vuh_vasr_VwVwR_rnd_sat(vs32_y_hi, vs32_y_lo, CVTCOLOR_COEF_BITS - 6);
}

AURA_ALWAYS_INLINE AURA_VOID CvtRgb2UVP010Core(HVX_VectorX3 &v3u16_src, HVX_VectorX3 &v3s32_rgb2u, HVX_VectorX2 &v2s32_rgb2v,
                                             HVX_Vector &vu16_u, HVX_Vector &vu16_v)
{
    HVX_Vector vs32_uc = Q6_V_vsplat_R(0x20000000);

    HVX_Vector vs32_xr_lo = Q6_Vw_vmpyie_VwVuh(v3s32_rgb2u.val[0], v3u16_src.val[0]);
    HVX_Vector vs32_xg_lo = Q6_Vw_vmpyie_VwVuh(v3s32_rgb2u.val[1], v3u16_src.val[1]);
    HVX_Vector vs32_xb_lo = Q6_Vw_vmpyie_VwVuh(v3s32_rgb2u.val[2], v3u16_src.val[2]);

    HVX_Vector vs32_xr_hi = Q6_Vw_vmpyio_VwVh(v3s32_rgb2u.val[0], v3u16_src.val[0]);
    HVX_Vector vs32_xg_hi = Q6_Vw_vmpyio_VwVh(v3s32_rgb2u.val[1], v3u16_src.val[1]);
    HVX_Vector vs32_xb_hi = Q6_Vw_vmpyio_VwVh(v3s32_rgb2u.val[2], v3u16_src.val[2]);

    HVX_Vector vs32_x_lo = Q6_Vw_vadd_VwVw(vs32_xr_lo, vs32_xg_lo);
    HVX_Vector vs32_x_hi = Q6_Vw_vadd_VwVw(vs32_xr_hi, vs32_xg_hi);

    vs32_x_lo = Q6_Vw_vadd_VwVw(vs32_x_lo, vs32_xb_lo);
    vs32_x_hi = Q6_Vw_vadd_VwVw(vs32_x_hi, vs32_xb_hi);

    vs32_x_lo = Q6_Vw_vadd_VwVw(vs32_x_lo, vs32_uc);
    vs32_x_hi = Q6_Vw_vadd_VwVw(vs32_x_hi, vs32_uc);

    vu16_u = Q6_Vuh_vasr_VwVwR_rnd_sat(vs32_x_hi, vs32_x_lo, CVTCOLOR_COEF_BITS - 6);

    vs32_xr_lo = Q6_Vw_vmpyie_VwVuh(v3s32_rgb2u.val[2], v3u16_src.val[0]);
    vs32_xg_lo = Q6_Vw_vmpyie_VwVuh(v2s32_rgb2v.val[0], v3u16_src.val[1]);
    vs32_xb_lo = Q6_Vw_vmpyie_VwVuh(v2s32_rgb2v.val[1], v3u16_src.val[2]);

    vs32_xr_hi = Q6_Vw_vmpyio_VwVh(v3s32_rgb2u.val[2], v3u16_src.val[0]);
    vs32_xg_hi = Q6_Vw_vmpyio_VwVh(v2s32_rgb2v.val[0], v3u16_src.val[1]);
    vs32_xb_hi = Q6_Vw_vmpyio_VwVh(v2s32_rgb2v.val[1], v3u16_src.val[2]);

    vs32_x_lo = Q6_Vw_vadd_VwVw(vs32_xr_lo, vs32_xg_lo);
    vs32_x_hi = Q6_Vw_vadd_VwVw(vs32_xr_hi, vs32_xg_hi);

    vs32_x_lo = Q6_Vw_vadd_VwVw(vs32_x_lo, vs32_xb_lo);
    vs32_x_hi = Q6_Vw_vadd_VwVw(vs32_x_hi, vs32_xb_hi);

    vs32_x_lo = Q6_Vw_vadd_VwVw(vs32_x_lo, vs32_uc);
    vs32_x_hi = Q6_Vw_vadd_VwVw(vs32_x_hi, vs32_uc);

    vu16_v = Q6_Vuh_vasr_VwVwR_rnd_sat(vs32_x_hi, vs32_x_lo, CVTCOLOR_COEF_BITS - 6);
}

template <MI_U32 MODE>
static Status CvtRgb2NvP010HvxImpl(const Mat &src, Mat &dst_y, Mat &dst_uv, MI_BOOL swapuv, MI_S32 start_row, MI_S32 end_row)
{
    constexpr MI_S32 R2Y = Rgb2YuvParamTraits<MODE>::R2Y;
    constexpr MI_S32 G2Y = Rgb2YuvParamTraits<MODE>::G2Y;
    constexpr MI_S32 B2Y = Rgb2YuvParamTraits<MODE>::B2Y;
    constexpr MI_S32 R2U = Rgb2YuvParamTraits<MODE>::R2U;
    constexpr MI_S32 G2U = Rgb2YuvParamTraits<MODE>::G2U;
    constexpr MI_S32 B2U = Rgb2YuvParamTraits<MODE>::B2U;
    constexpr MI_S32 G2V = Rgb2YuvParamTraits<MODE>::G2V;
    constexpr MI_S32 B2V = Rgb2YuvParamTraits<MODE>::B2V;

    const MI_S32 width       = src.GetSizes().m_width;
    const MI_S32 ichannel    = src.GetSizes().m_channel;
    const MI_S32 istride     = src.GetStrides().m_width;
    const MI_S32 width_align = width & (-AURA_HVLEN);
    const MI_S32 uidx        = swapuv;
    const MI_S32 vidx        = 1 - uidx;

    MI_U64 L2fetch_param = L2PfParam(istride, width * ichannel * ElemTypeSize(src.GetElemType()), 2, 0);

    HVX_VectorX3 v3s32_rgb2y, v3s32_rgb2u;
    HVX_VectorX3 v3u16_src_c_lo, v3u16_src_c_hi, v3u16_src_n_lo, v3u16_src_n_hi;
    HVX_VectorX2 v2s32_rgb2v, v2u16_dst_y_c, v2u16_dst_y_n, v2u16_dst_uv;

    v3s32_rgb2y.val[0] = Q6_V_vsplat_R(R2Y);
    v3s32_rgb2y.val[1] = Q6_V_vsplat_R(G2Y);
    v3s32_rgb2y.val[2] = Q6_V_vsplat_R(B2Y);

    v3s32_rgb2u.val[0] = Q6_V_vsplat_R(R2U);
    v3s32_rgb2u.val[1] = Q6_V_vsplat_R(G2U);
    v3s32_rgb2u.val[2] = Q6_V_vsplat_R(B2U); // R2V = B2U

    v2s32_rgb2v.val[0] = Q6_V_vsplat_R(G2V);
    v2s32_rgb2v.val[1] = Q6_V_vsplat_R(B2V);

    for (MI_S32 uv = start_row; uv < end_row; uv++)
    {
        MI_S32 y = uv * 2;
        if (uv < end_row - 1)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<MI_U16>(y + 2)), L2fetch_param);
        }

        MI_U16 *src_c = (MI_U16 *)src.Ptr<MI_U16>(y);
        MI_U16 *src_n = (MI_U16 *)src.Ptr<MI_U16>(y + 1);

        MI_U16 *dst_y_c  = dst_y.Ptr<MI_U16>(y);
        MI_U16 *dst_y_n  = dst_y.Ptr<MI_U16>(y + 1);
        MI_U16 *dst_uv_c = dst_uv.Ptr<MI_U16>(uv);

        MI_S32 x = 0;
        for (; x < width_align; x += AURA_HVLEN)
        {
LOOP_BODY:
            vload(src_c + x * 3,                     v3u16_src_c_lo);
            vload(src_c + (x + AURA_HALF_HVLEN) * 3, v3u16_src_c_hi);
            vload(src_n + x * 3,                     v3u16_src_n_lo);
            vload(src_n + (x + AURA_HALF_HVLEN) * 3, v3u16_src_n_hi);

            CvtDealRgbCore(v3u16_src_c_lo, v3u16_src_c_hi, -2);
            CvtDealRgbCore(v3u16_src_n_lo, v3u16_src_n_hi, -2);

            CvtRgb2YP010Core(v3u16_src_c_lo, v3s32_rgb2y, v2u16_dst_y_c.val[0]);
            CvtRgb2YP010Core(v3u16_src_c_hi, v3s32_rgb2y, v2u16_dst_y_c.val[1]);
            CvtRgb2UVP010Core(v3u16_src_c_lo, v3s32_rgb2u, v2s32_rgb2v, v2u16_dst_uv.val[uidx], v2u16_dst_uv.val[vidx]);
            vstore(dst_y_c + x,  v2u16_dst_y_c);
            vstore(dst_uv_c + x, v2u16_dst_uv);

            CvtRgb2YP010Core(v3u16_src_n_lo, v3s32_rgb2y, v2u16_dst_y_n.val[0]);
            CvtRgb2YP010Core(v3u16_src_n_hi, v3s32_rgb2y, v2u16_dst_y_n.val[1]);
            vstore(dst_y_n + x, v2u16_dst_y_n);
        }

        if (x < width)
        {
            x = width - AURA_HVLEN;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

Status CvtRgb2NvP010Hvx(Context *ctx, const Mat &src, Mat &dst_y, Mat &dst_uv, MI_BOOL swapuv)
{
    Status ret = Status::ERROR;

    if ((src.GetSizes().m_width & 1) || (src.GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(ctx, "src sizes only support even");
        return ret;
    }

    if (dst_y.GetSizes() * Sizes3(1, 1, 2) != dst_uv.GetSizes() * Sizes3(2, 2, 1) ||
        dst_y.GetSizes() * Sizes3(1, 1, 3) != src.GetSizes() ||
        dst_y.GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "the sizes of src and dst do not match");
        return ret;
    }

    MI_S32 height = dst_uv.GetSizes().m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    if ((ret = wp->ParallelFor((MI_S32)0, height, CvtRgb2NvP010HvxImpl<1>, std::cref(src), std::ref(dst_y), std::ref(dst_uv), swapuv)) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
        return ret;
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura