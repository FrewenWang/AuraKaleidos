#include "warp_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

#define AURA_WARP_AFFINE_VTCM_VGATHER_SZ       (AURA_HVLEN * 4)
#define AURA_WARP_AFFINE_VTCM_VGATHER_TOTAL_SZ (AURA_WARP_AFFINE_VTCM_VGATHER_SZ * 4)

namespace aura
{
struct PosInterpCoeffs
{
    HVX_VectorPair ws32_base;
    HVX_VectorPair ws32_dx;
    HVX_VectorPair ws32_dy;
};

struct InterpGatherParam
{
    DT_S32 x_limit;
    DT_S32 y_limit;
    DT_S32 xy_limit;
    DT_S32 xy_mul;

    HVX_Vector  *gather_vec;
    const DT_U8 *gather_base;
    DT_S32       gather_limit;

    DT_U8 border_val;
};

AURA_ALWAYS_INLINE PosInterpCoeffs LoadPosInterpCoeffs8Tile(const DT_S32 *grid_data, DT_S32 grid_step)
{
    HVX_Vector vs32_up_l, vs32_dn_l;
    vload((const DT_U8 *)(grid_data), vs32_up_l);
    vload((const DT_U8 *)(grid_data + grid_step), vs32_dn_l);

    HVX_Vector vs32_up_r = Q6_V_vror_VR(vs32_up_l, 8);
    HVX_Vector vs32_dn_r = Q6_V_vror_VR(vs32_dn_l, 8);

    HVX_Vector vs32_dyl_half = Q6_Vw_vnavg_VwVw(vs32_dn_l, vs32_up_l);
    HVX_Vector vs32_dyh_half = Q6_Vw_vnavg_VwVw(vs32_dn_r, vs32_up_r);

    vs32_up_l = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_Vw_vadd_VwVw(vs32_up_l, vs32_dyl_half), vs32_up_l, 64));
    vs32_up_r = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_Vw_vadd_VwVw(vs32_up_r, vs32_dyh_half), vs32_up_r, 64));
    vs32_dn_l = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_Vw_vadd_VwVw(vs32_dn_l, vs32_dyl_half), vs32_dn_l, 64));
    vs32_dn_r = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_Vw_vadd_VwVw(vs32_dn_r, vs32_dyh_half), vs32_dn_r, 64));

    HVX_Vector vs32_dxu_half = Q6_Vw_vnavg_VwVw(vs32_up_r, vs32_up_l);
    HVX_Vector vs32_dxd_half = Q6_Vw_vnavg_VwVw(vs32_dn_r, vs32_dn_l);

    HVX_Vector vs32_up_l1 = Q6_Vw_vadd_VwVw(vs32_up_l, vs32_dxu_half);
    HVX_Vector vs32_up_r1 = Q6_Vw_vadd_VwVw(vs32_up_r, vs32_dxu_half);
    HVX_Vector vs32_dn_l1 = Q6_Vw_vadd_VwVw(vs32_dn_l, vs32_dxd_half);

    HVX_VectorPair ws32_up_l = Q6_W_vdeal_VVR(vs32_up_l1, vs32_up_l, 4);
    HVX_VectorPair ws32_up_r = Q6_W_vdeal_VVR(vs32_up_r1, vs32_up_r, 4);
    HVX_VectorPair ws32_dn_l = Q6_W_vdeal_VVR(vs32_dn_l1, vs32_dn_l, 4);

    PosInterpCoeffs coeffs;
    coeffs.ws32_base = ws32_up_l;
    coeffs.ws32_dx   = Q6_Ww_vsub_WwWw(ws32_up_r, ws32_up_l);
    coeffs.ws32_dy   = Q6_Ww_vsub_WwWw(ws32_dn_l, ws32_up_l);

    return coeffs;
}

AURA_ALWAYS_INLINE HVX_VectorPair DoInterpForLuma(PosInterpCoeffs &coeffs, DT_S32 h, DT_S32 w)
{
    DT_S32 hh = Q6_R_combine_RlRl(h, h);
    DT_S32 ww = Q6_R_combine_RlRl(w, w);

    HVX_Vector vs32_dyh_lo = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_vsplat_R(8), Q6_V_lo_W(coeffs.ws32_dy), hh);
    HVX_Vector vs32_dyh_hi = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_vsplat_R(8), Q6_V_hi_W(coeffs.ws32_dy), hh);

    HVX_Vector vs32_dxw_lo = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_vsplat_R(8), Q6_V_lo_W(coeffs.ws32_dx), ww);
    HVX_Vector vs32_dxw_hi = Q6_Vw_vmpyiacc_VwVwRh(Q6_V_vsplat_R(8), Q6_V_hi_W(coeffs.ws32_dx), ww);

    HVX_VectorPair ws32_dyh = Q6_W_vcombine_VV(Q6_Vw_vasr_VwR(vs32_dyh_hi, 4), Q6_Vw_vasr_VwR(vs32_dyh_lo, 4));
    HVX_VectorPair ws32_dxw = Q6_W_vcombine_VV(Q6_Vw_vasr_VwR(vs32_dxw_hi, 4), Q6_Vw_vasr_VwR(vs32_dxw_lo, 4));

    return Q6_Ww_vadd_WwWw(coeffs.ws32_base, Q6_Ww_vadd_WwWw(ws32_dxw, ws32_dyh));
}

AURA_ALWAYS_INLINE HVX_VectorPred ClampXYPos(HVX_VectorX2 &v2s16_xypos, DT_S32 x_limit, DT_S32 y_limit)
{
    HVX_VectorPair ws16_xypos = Q6_Wh_vshuffoe_VhVh(v2s16_xypos.val[1], v2s16_xypos.val[0]);

    HVX_VectorPred q_limit_x = Q6_Q_vcmp_gt_VuhVuh(Q6_V_lo_W(ws16_xypos), Q6_Vh_vsplat_R(x_limit));
    HVX_VectorPred q_limit_y = Q6_Q_vcmp_gt_VuhVuh(Q6_V_hi_W(ws16_xypos), Q6_Vh_vsplat_R(y_limit));

    return Q6_Q_or_QQ(q_limit_x, q_limit_y);
}

AURA_ALWAYS_INLINE DT_VOID LimitXYPos(HVX_VectorX2 &v2s16_xypos, DT_S32 xy_limit)
{
    v2s16_xypos.val[0] = Q6_Vh_vmax_VhVh(v2s16_xypos.val[0], Q6_V_vzero());
    v2s16_xypos.val[1] = Q6_Vh_vmax_VhVh(v2s16_xypos.val[1], Q6_V_vzero());

    v2s16_xypos.val[0] = Q6_Vh_vmin_VhVh(v2s16_xypos.val[0], Q6_V_vsplat_R(xy_limit));
    v2s16_xypos.val[1] = Q6_Vh_vmin_VhVh(v2s16_xypos.val[1], Q6_V_vsplat_R(xy_limit));
}

AURA_ALWAYS_INLINE DT_VOID NearestGatherForPos(InterpGatherParam &gather_param, HVX_VectorX2 &v2s16_xypos)
{
    LimitXYPos(v2s16_xypos, gather_param.xy_limit);

    // convert to buffer address
    HVX_Vector vs32_pos_lo = Q6_Vw_vdmpy_VhRh_sat(v2s16_xypos.val[0], gather_param.xy_mul);
    HVX_Vector vs32_pos_hi = Q6_Vw_vdmpy_VhRh_sat(v2s16_xypos.val[1], gather_param.xy_mul);

    Q6_vgather_ARMVw(gather_param.gather_vec + 0, (DT_S32)gather_param.gather_base, gather_param.gather_limit, vs32_pos_lo);
    Q6_vgather_ARMVw(gather_param.gather_vec + 1, (DT_S32)gather_param.gather_base, gather_param.gather_limit, vs32_pos_hi);
}

template <BorderType BORDER_TYPE, typename std::enable_if<BorderType::CONSTANT == BORDER_TYPE>::type * = DT_NULL>
AURA_INLINE DT_VOID NearestGatherForPos(InterpGatherParam &gather_param, HVX_VectorX2 &v2s16_xypos, HVX_Vector &vu16_sum64)
{
    HVX_VectorPred q_limit = ClampXYPos(v2s16_xypos, gather_param.x_limit, gather_param.y_limit);

    NearestGatherForPos(gather_param, v2s16_xypos);

    vu16_sum64 = Q6_Vh_vshuffe_VhVh(gather_param.gather_vec[1], gather_param.gather_vec[0]);
    vu16_sum64 = Q6_V_vmux_QVV(q_limit, Q6_Vh_vsplat_R(gather_param.border_val), vu16_sum64);
}

template <BorderType BORDER_TYPE, typename std::enable_if<BorderType::REPLICATE == BORDER_TYPE>::type * = DT_NULL>
AURA_INLINE DT_VOID NearestGatherForPos(InterpGatherParam &gather_param, HVX_VectorX2 &v2s16_xypos, HVX_Vector &vu16_sum64)
{
    NearestGatherForPos(gather_param, v2s16_xypos);
    vu16_sum64 = Q6_Vh_vshuffe_VhVh(gather_param.gather_vec[1], gather_param.gather_vec[0]);
}

template <InterpType INTERP_TYPE, BorderType BORDER_TYPE, typename std::enable_if<InterpType::NEAREST == INTERP_TYPE>::type * = DT_NULL>
AURA_INLINE DT_VOID InterpGatherForPos(DT_S32 irow, PosInterpCoeffs &coeffs, InterpGatherParam &gather_param, HVX_VectorX2 &v2u8_out)
{
    HVX_VectorX4 v4u16_sum64;
    for (DT_S32 icol = 0; icol < 4; icol++)
    {
        HVX_VectorPair ws32_xypos_lo = DoInterpForLuma(coeffs, irow, icol);     // 0,2,1,3
        HVX_VectorPair ws32_xypos_hi = DoInterpForLuma(coeffs, irow, icol + 4); // 4,6,5,7

        HVX_VectorX2 v2s16_xypos;
        v2s16_xypos.val[0] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_xypos_lo), Q6_V_lo_W(ws32_xypos_lo), 10);
        v2s16_xypos.val[1] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_xypos_hi), Q6_V_lo_W(ws32_xypos_hi), 10);

        NearestGatherForPos<BORDER_TYPE>(gather_param, v2s16_xypos, v4u16_sum64.val[icol]);
    }

    v2u8_out.val[0] = Q6_Vb_vshuffe_VbVb(v4u16_sum64.val[2], v4u16_sum64.val[0]);
    v2u8_out.val[1] = Q6_Vb_vshuffe_VbVb(v4u16_sum64.val[3], v4u16_sum64.val[1]);
}

AURA_ALWAYS_INLINE DT_VOID LinearGatherForPos(InterpGatherParam &gather_param, DT_S32 idx, HVX_VectorX2 &v2s16_xypos, HVX_Vector &vu8_pixels)
{
    HVX_Vector vs32_pos_lo = Q6_Vw_vdmpy_VhRh_sat(v2s16_xypos.val[0], gather_param.xy_mul);
    HVX_Vector vs32_pos_hi = Q6_Vw_vdmpy_VhRh_sat(v2s16_xypos.val[1], gather_param.xy_mul);

    Q6_vgather_ARMVw(gather_param.gather_vec + (idx + 0), (DT_S32)gather_param.gather_base, gather_param.gather_limit, vs32_pos_lo);
    Q6_vgather_ARMVw(gather_param.gather_vec + (idx + 1), (DT_S32)gather_param.gather_base, gather_param.gather_limit, vs32_pos_hi);

    vu8_pixels = Q6_Vh_vshuffe_VhVh(gather_param.gather_vec[idx + 1], gather_param.gather_vec[idx]);
}

template <BorderType BORDER_TYPE, typename std::enable_if<BorderType::CONSTANT == BORDER_TYPE>::type * = DT_NULL>
AURA_INLINE DT_VOID LinearLimitPixels(HVX_VectorPred &q_limit, HVX_Vector &vu8_pixels, DT_S32 border_val)
{
    vu8_pixels = Q6_V_vmux_QVV(q_limit, Q6_Vb_vsplat_R(border_val), vu8_pixels);
}

template <BorderType BORDER_TYPE, typename std::enable_if<BorderType::REPLICATE == BORDER_TYPE>::type * = DT_NULL>
AURA_INLINE DT_VOID LinearLimitPixels(HVX_VectorPred &q_limit, HVX_Vector &vu8_pixels, DT_S32 border_val)
{
    AURA_UNUSED(q_limit);
    AURA_UNUSED(vu8_pixels);
    AURA_UNUSED(border_val);
}

AURA_ALWAYS_INLINE DT_VOID LinearInterpolation(HVX_VectorX2 &v2s16_xyfrac, HVX_VectorX2 &v2u8_pixels, HVX_Vector &vu16_sum64)
{
    HVX_Vector vu16_xyfrac1_lo = Q6_V_vand_VV(v2s16_xyfrac.val[0], Q6_Vh_vsplat_R(0x03ff));
    HVX_Vector vu16_xyfrac1_hi = Q6_V_vand_VV(v2s16_xyfrac.val[1], Q6_Vh_vsplat_R(0x03ff));

    HVX_Vector vu16_xyfrac0_lo = Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(0x0400), vu16_xyfrac1_lo);
    HVX_Vector vu16_xyfrac0_hi = Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(0x0400), vu16_xyfrac1_hi);

    HVX_VectorPair wu16_xyfrac01_lo = Q6_Wh_vshuffoe_VhVh(vu16_xyfrac1_lo, vu16_xyfrac0_lo);
    HVX_VectorPair wu16_xyfrac01_hi = Q6_Wh_vshuffoe_VhVh(vu16_xyfrac1_hi, vu16_xyfrac0_hi);

    HVX_VectorPair wu16_pixels_lo = Q6_W_vshuff_VVR(Q6_V_vzero(), v2u8_pixels.val[0], 3);
    HVX_VectorPair wu16_pixels_hi = Q6_W_vshuff_VVR(Q6_V_vzero(), v2u8_pixels.val[1], 3);

    HVX_Vector vu32_sum32_lo_lo = Q6_Vw_vdmpy_VhVh_sat(Q6_V_lo_W(wu16_pixels_lo), Q6_V_lo_W(wu16_xyfrac01_lo));
    HVX_Vector vu32_sum32_lo_hi = Q6_Vw_vdmpy_VhVh_sat(Q6_V_lo_W(wu16_pixels_hi), Q6_V_lo_W(wu16_xyfrac01_lo));

    HVX_Vector vu32_sum32_hi_lo = Q6_Vw_vdmpy_VhVh_sat(Q6_V_hi_W(wu16_pixels_lo), Q6_V_lo_W(wu16_xyfrac01_hi));
    HVX_Vector vu32_sum32_hi_hi = Q6_Vw_vdmpy_VhVh_sat(Q6_V_hi_W(wu16_pixels_hi), Q6_V_lo_W(wu16_xyfrac01_hi));

    HVX_Vector vu16_sum32_lo = Q6_Vh_vasr_VwVwR(vu32_sum32_lo_hi, vu32_sum32_lo_lo, 3);
    HVX_Vector vu16_sum32_hi = Q6_Vh_vasr_VwVwR(vu32_sum32_hi_hi, vu32_sum32_hi_lo, 3);

    HVX_Vector vu32_sum32_lo = Q6_Vw_vdmpy_VhVh_sat(vu16_sum32_lo, Q6_V_hi_W(wu16_xyfrac01_lo));
    HVX_Vector vu32_sum32_hi = Q6_Vw_vdmpy_VhVh_sat(vu16_sum32_hi, Q6_V_hi_W(wu16_xyfrac01_hi));

    vu16_sum64 = Q6_Vh_vasr_VwVwR(vu32_sum32_hi, vu32_sum32_lo, 15);
}

template <InterpType INTERP_TYPE, BorderType BORDER_TYPE, typename std::enable_if<InterpType::LINEAR == INTERP_TYPE>::type * = DT_NULL>
AURA_INLINE DT_VOID InterpGatherForPos(DT_S32 irow, PosInterpCoeffs &coeffs, InterpGatherParam &gather_param, HVX_VectorX2 &v2u8_out)
{
    HVX_VectorX4 v4u16_sum64;
    for (DT_S32 icol = 0; icol < 4; icol++)
    {
        HVX_VectorPair ws32_xypos_lo = DoInterpForLuma(coeffs, irow, icol);
        HVX_VectorPair ws32_xypos_hi = DoInterpForLuma(coeffs, irow, icol + 4);

        HVX_VectorX2 v2s16_xyfrac;
        v2s16_xyfrac.val[0] = Q6_Vh_vshuffe_VhVh(Q6_V_hi_W(ws32_xypos_lo), Q6_V_lo_W(ws32_xypos_lo));
        v2s16_xyfrac.val[1] = Q6_Vh_vshuffe_VhVh(Q6_V_hi_W(ws32_xypos_hi), Q6_V_lo_W(ws32_xypos_hi));

        HVX_VectorX2 v2s16_xypos_c, v2s16_xypos_n;
        v2s16_xypos_c.val[0] = Q6_Vh_vasr_VwVwR_sat(Q6_V_hi_W(ws32_xypos_lo), Q6_V_lo_W(ws32_xypos_lo), 10);
        v2s16_xypos_c.val[1] = Q6_Vh_vasr_VwVwR_sat(Q6_V_hi_W(ws32_xypos_hi), Q6_V_lo_W(ws32_xypos_hi), 10);
        v2s16_xypos_n.val[0] = Q6_Vh_vadd_VhVh(v2s16_xypos_c.val[0], Q6_V_vsplat_R(0x00010000));
        v2s16_xypos_n.val[1] = Q6_Vh_vadd_VhVh(v2s16_xypos_c.val[1], Q6_V_vsplat_R(0x00010000));

        HVX_VectorPred q_limit_c = ClampXYPos(v2s16_xypos_c, gather_param.x_limit, gather_param.y_limit);
        HVX_VectorPred q_limit_n = ClampXYPos(v2s16_xypos_n, gather_param.x_limit, gather_param.y_limit);

        LimitXYPos(v2s16_xypos_c, gather_param.xy_limit);
        LimitXYPos(v2s16_xypos_n, gather_param.xy_limit);

        HVX_VectorX2 v2u8_pixels;
        LinearGatherForPos(gather_param, 0, v2s16_xypos_c, v2u8_pixels.val[0]);
        LinearGatherForPos(gather_param, 2, v2s16_xypos_n, v2u8_pixels.val[1]);

        LinearLimitPixels<BORDER_TYPE>(q_limit_c, v2u8_pixels.val[0], gather_param.border_val);
        LinearLimitPixels<BORDER_TYPE>(q_limit_n, v2u8_pixels.val[1], gather_param.border_val);

        LinearInterpolation(v2s16_xyfrac, v2u8_pixels, v4u16_sum64.val[icol]);
    }

    v2u8_out.val[0] = Q6_Vub_vasr_VuhVuhR_rnd_sat(v4u16_sum64.val[2], v4u16_sum64.val[0], 2);
    v2u8_out.val[1] = Q6_Vub_vasr_VuhVuhR_rnd_sat(v4u16_sum64.val[3], v4u16_sum64.val[1], 2);
}

template <InterpType INTERP_TYPE, BorderType BORDER_TYPE>
Status WarpAffineU8C1Impl(const Mat &src, const Mat &grid, Mat &dst, Scalar &border_value, DT_U8 *vtcm_buffer,
                          DT_S32 length, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 oheight      = dst.GetSizes().m_height;
    DT_S32 owidth       = dst.GetSizes().m_width;
    DT_S32 ostride      = dst.GetStrides().m_width;

    DT_S32 width_align  = owidth & (-AURA_HVLEN);
    DT_S32 width_remain = owidth - width_align;

    DT_S32 iheight      = src.GetSizes().m_height;
    DT_S32 iwidth       = src.GetSizes().m_width;
    DT_S32 istride      = src.GetStrides().m_width;

    DT_S32 grid_step    = grid.GetStrides().m_width / sizeof(DT_S32);

    InterpGatherParam gather_param;
    gather_param.gather_vec   = (HVX_Vector*)(vtcm_buffer + (start_row / length) * AURA_WARP_AFFINE_VTCM_VGATHER_SZ);
    gather_param.gather_base  = vtcm_buffer + AURA_WARP_AFFINE_VTCM_VGATHER_TOTAL_SZ;
    gather_param.gather_limit = (iheight - 1) * istride + (iwidth - 1);
    gather_param.x_limit      = iwidth - 1;
    gather_param.y_limit      = iheight - 1;
    gather_param.xy_limit     = Q6_R_combine_RlRl(iheight - 1, iwidth - 1);
    gather_param.xy_mul       = Q6_R_combine_RlRl(istride, 1);
    gather_param.border_val   = border_value.m_val[0];

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 max_row = Min(oheight - (y << 4), (DT_S32)16);
        DT_S32 cal_row = Min(max_row, (DT_S32)8);

        for (DT_S32 x = 0; x < owidth; x += AURA_HVLEN)
        {
            const DT_S32 *grid_data  = grid.Ptr<DT_S32>(y) + (x >> 3);
            DT_U8        *out_buffer = dst.Ptr<DT_U8>(y << 4) + x;

            PosInterpCoeffs coeffs = LoadPosInterpCoeffs8Tile(grid_data, grid_step);

            for (DT_S32 irow = 0; irow < cal_row; irow++)
            {
                HVX_VectorX2 v2u8_out;
                InterpGatherForPos<INTERP_TYPE, BORDER_TYPE>(irow, coeffs, gather_param, v2u8_out);

                if (x <= (owidth - AURA_HVLEN))
                {
                    HVX_VectorPair wu8_out = Q6_W_vshuff_VVR(v2u8_out.val[1], v2u8_out.val[0], -1);
                    vstore(out_buffer, Q6_V_lo_W(wu8_out));
                    if (irow + 8 < max_row)
                    {
                        vstore(out_buffer + 8 * ostride, Q6_V_hi_W(wu8_out));
                    }
                }
                else
                {
                    DT_S32 back_offset = width_remain - AURA_HVLEN;

                    HVX_Vector vu8_out_pre;
                    vload(out_buffer - AURA_HVLEN, vu8_out_pre);

                    HVX_VectorPair wu8_out = Q6_W_vshuff_VVR(v2u8_out.val[1], v2u8_out.val[0], -1);
                    HVX_Vector     vu8_out = Q6_V_valign_VVR(Q6_V_lo_W(wu8_out), vu8_out_pre, width_remain);

                    vstore(out_buffer + back_offset, vu8_out);
                    if (irow + 8 < max_row)
                    {
                        DT_U8 *out_line_8 = out_buffer + 8 * ostride;
                        vload(out_line_8 - AURA_HVLEN, vu8_out_pre);

                        vu8_out = Q6_V_valign_VVR(Q6_V_hi_W(wu8_out), vu8_out_pre, width_remain);
                        vstore(out_line_8 + back_offset, vu8_out);
                    }
                }

                out_buffer += ostride;
            }
        }
    }

    return Status::OK;
}

static Status WarpAffineU8C1Hvx(Context *ctx, const Mat &src, const Mat &grid, Mat &dst, InterpType interp_type, BorderType border_type, Scalar &border_value)
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    DT_S32 vtcm_size = src.GetTotalBytes() + AURA_WARP_AFFINE_VTCM_VGATHER_TOTAL_SZ;

    DT_U8 *vtcm_buffer = (DT_U8 *)AURA_ALLOC_PARAM(ctx, AURA_MEM_VTCM, vtcm_size, 1);
    if (DT_NULL == vtcm_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "Could not allocate VTCM.");
        return Status::ABORT;
    }

    AuraMemCopy(vtcm_buffer + AURA_WARP_AFFINE_VTCM_VGATHER_TOTAL_SZ, src.GetData(), src.GetTotalBytes());

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        AURA_FREE(ctx, vtcm_buffer);
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    DT_S32 length  = ((grid.GetSizes().m_height - 1) + (wp->GetComputeThreadNum() - 1)) / wp->GetComputeThreadNum();
    DT_S32 pattern = AURA_MAKE_PATTERN(interp_type, border_type);

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(InterpType::NEAREST, BorderType::CONSTANT):
        {
            ret = wp->ParallelFor((DT_S32)0, grid.GetSizes().m_height - 1, WarpAffineU8C1Impl<InterpType::NEAREST, BorderType::CONSTANT>,
                                  std::cref(src), std::cref(grid), std::ref(dst), border_value, vtcm_buffer, length);
            break;
        }
        case AURA_MAKE_PATTERN(InterpType::NEAREST, BorderType::REPLICATE):
        {
            ret = wp->ParallelFor((DT_S32)0, grid.GetSizes().m_height - 1, WarpAffineU8C1Impl<InterpType::NEAREST, BorderType::REPLICATE>,
                                  std::cref(src), std::cref(grid), std::ref(dst), border_value, vtcm_buffer, length);
            break;
        }
        case AURA_MAKE_PATTERN(InterpType::LINEAR, BorderType::CONSTANT):
        {
            ret = wp->ParallelFor((DT_S32)0, grid.GetSizes().m_height - 1, WarpAffineU8C1Impl<InterpType::LINEAR, BorderType::CONSTANT>,
                                  std::cref(src), std::cref(grid), std::ref(dst), border_value, vtcm_buffer, length);
            break;
        }
        case AURA_MAKE_PATTERN(InterpType::LINEAR, BorderType::REPLICATE):
        {
            ret = wp->ParallelFor((DT_S32)0, grid.GetSizes().m_height - 1, WarpAffineU8C1Impl<InterpType::LINEAR, BorderType::REPLICATE>,
                                  std::cref(src), std::cref(grid), std::ref(dst), border_value, vtcm_buffer, length);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported interp|border type.");
        }
    }

    AURA_FREE(ctx, vtcm_buffer);

    AURA_RETURN(ctx, ret);
}

Status WarpAffineHvx(Context *ctx, const Mat &src, const Mat &matrix, Mat &dst, InterpType interp_type, BorderType border_type, Scalar &border_value)
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    Mat grid = matrix;

    if (2 == matrix.GetSizes().m_height && 3 == matrix.GetSizes().m_width)
    {
        DT_S32 mesh_h = (dst.GetSizes().m_height + 15) / 16 + 1;
        DT_S32 mesh_w = (dst.GetSizes().m_width  + 15) / 16 + 1;
        DT_S32 stride = (mesh_w * 2 * ElemTypeSize(ElemType::S32) + 128) & (-64);

        grid = Mat(ctx, ElemType::S32, aura::Sizes3(mesh_h, mesh_w, 2), AURA_MEM_HEAP, aura::Sizes(mesh_h, stride));
        if (!grid.IsValid())
        {
            AURA_ADD_ERROR_STRING(ctx, "grid is invalid");
            return Status::ERROR;
        }

        ret = InitMapGrid(ctx, matrix, grid, 16);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "InitMapGrid failed");
            return Status::ERROR;
        }
    }

    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetSizes().m_channel, src.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(1, ElemType::U8):
        {
            ret = WarpAffineU8C1Hvx(ctx, src, grid, dst, interp_type, border_type, border_value);
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

AURA_ALWAYS_INLINE HVX_VectorPairX4 Q6_Wx4_vshuff_Wx4R(HVX_VectorPairX4 &w4x4_in, DT_S32 rt)
{
    HVX_VectorPairX4 w4x4_out;
    w4x4_out.val[0] = Q6_W_vshuff_VVR(Q6_V_lo_W(w4x4_in.val[2]), Q6_V_lo_W(w4x4_in.val[0]), rt);
    w4x4_out.val[1] = Q6_W_vshuff_VVR(Q6_V_hi_W(w4x4_in.val[2]), Q6_V_hi_W(w4x4_in.val[0]), rt);
    w4x4_out.val[2] = Q6_W_vshuff_VVR(Q6_V_lo_W(w4x4_in.val[3]), Q6_V_lo_W(w4x4_in.val[1]), rt);
    w4x4_out.val[3] = Q6_W_vshuff_VVR(Q6_V_hi_W(w4x4_in.val[3]), Q6_V_hi_W(w4x4_in.val[1]), rt);

    return w4x4_out;
}

Status WarpAffineCoordImpl(const Mat &grid, Mat &coord, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 coord_height = coord.GetSizes().m_height;
    DT_S32 coord_width  = coord.GetSizes().m_width;
    DT_S32 coord_stride = coord.GetStrides().m_width;
    DT_S32 grid_step    = grid.GetStrides().m_width / sizeof(DT_S32);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 max_row = Min(coord_height - (y << 4), (DT_S32)16);
        DT_S32 cal_row = Min(max_row, (DT_S32)8);

        for (DT_S32 x = 0; x < coord_width; x += AURA_HVLEN)
        {
            const DT_S32 *grid_data    = grid.Ptr<DT_S32>(y) + (x >> 3);
            DT_U8        *coord_buffer = coord.Ptr<DT_U8>(y << 4) + (x * 4);

            PosInterpCoeffs coeffs = LoadPosInterpCoeffs8Tile(grid_data, grid_step);

            for (DT_S32 irow = 0; irow < cal_row; irow++)
            {
                HVX_VectorPairX4 w4s32_xy_c, w4s32_xy_n;
                for (DT_S32 icol = 0; icol < 4; icol++)
                {
                    HVX_VectorPair ws32_xypos_lo = DoInterpForLuma(coeffs, irow, icol);
                    HVX_VectorPair ws32_xypos_hi = DoInterpForLuma(coeffs, irow, icol + 4);

                    HVX_Vector vs16_xy_lo = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_xypos_lo), Q6_V_lo_W(ws32_xypos_lo), 10);
                    HVX_Vector vs16_xy_hi = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_xypos_hi), Q6_V_lo_W(ws32_xypos_hi), 10);

                    w4s32_xy_c.val[icol] = Q6_W_vcombine_VV(vs16_xy_hi, vs16_xy_lo);
                }

                w4s32_xy_n = Q6_Wx4_vshuff_Wx4R(w4s32_xy_c, -4);
                w4s32_xy_c = Q6_Wx4_vshuff_Wx4R(w4s32_xy_n, -4);
                w4s32_xy_n = Q6_Wx4_vshuff_Wx4R(w4s32_xy_c, -16);

                vstore(coord_buffer, w4s32_xy_n.val[0]);
                vstore(coord_buffer + AURA_HVLEN * 2, w4s32_xy_n.val[1]);

                if (irow + 8 < max_row)
                {
                    DT_U8 *coord_line_8 = coord_buffer + 8 * coord_stride;
                    vstore(coord_line_8, w4s32_xy_n.val[2]);
                    vstore(coord_line_8 + AURA_HVLEN * 2, w4s32_xy_n.val[3]);
                }

                coord_buffer += coord_stride;
            }
        }
    }

    return Status::OK;
}

Status WarpAffineCoordHvx(Context *ctx, const Mat &grid, Mat &coord)
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    DT_S32 rows = grid.GetSizes().m_height - 1;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    if (wp->ParallelFor((DT_S32)0, rows, WarpAffineCoordImpl, grid, coord) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor run failed");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace aura