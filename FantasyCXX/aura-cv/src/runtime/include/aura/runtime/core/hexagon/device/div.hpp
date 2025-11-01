#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_DIV_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_DIV_HPP__

#include "aura/runtime/core/types.h"

#include "hexagon_types.h"

namespace aura
{

// u8 = u8 / u8, vu8_u / vu8_v
AURA_INLINE HVX_Vector Q6_Vub_vdiv_VubVub(HVX_Vector vu8_u, HVX_Vector vu8_v)
{
    HVX_VectorPair wu16_divisor_0123 = Q6_Wuh_vmpy_VubRub(vu8_v, 0x80808080);
    HVX_Vector vu16_divisor_02 = Q6_V_lo_W(wu16_divisor_0123);
    HVX_Vector vu16_divisor_13 = Q6_V_hi_W(wu16_divisor_0123);
    HVX_VectorPair wu16_dividend_0213= Q6_Wh_vadd_VubVub(vu8_u, Q6_V_vzero());
    HVX_Vector vu16_dividend_02 = Q6_V_lo_W(wu16_dividend_0213);
    HVX_Vector vu16_dividend_13 = Q6_V_hi_W(wu16_dividend_0213);
    HVX_Vector vd16_result_02 = Q6_V_vzero();
    HVX_Vector vd16_result_13 = Q6_V_vzero();
    HVX_Vector vd16_div_bit = Q6_V_vsplat_R(0x00800080);
    HVX_VectorPred q_0, q_1;

    for (MI_S32 k = 0; k < 8; k++)
    {
        q_0 = Q6_Q_vcmp_gt_VuhVuh(vu16_divisor_02, vu16_dividend_02);
        q_1 = Q6_Q_vcmp_gt_VuhVuh(vu16_divisor_13, vu16_dividend_13);

        vd16_result_02 = Q6_Vh_condacc_QnVhVh(q_0, vd16_result_02, vd16_div_bit);
        vd16_result_13 = Q6_Vh_condacc_QnVhVh(q_1, vd16_result_13, vd16_div_bit);

        vu16_dividend_02 = Q6_Vh_condnac_QnVhVh(q_0, vu16_dividend_02, vu16_divisor_02);
        vu16_dividend_13 = Q6_Vh_condnac_QnVhVh(q_1, vu16_dividend_13, vu16_divisor_13);

        vu16_divisor_02 = Q6_Vh_vasr_VhR(vu16_divisor_02, 1);
        vu16_divisor_13 = Q6_Vh_vasr_VhR(vu16_divisor_13, 1);
        vd16_div_bit    = Q6_Vh_vasr_VhR(vd16_div_bit, 1);
    }
    return Q6_Vb_vshuffe_VbVb(vd16_result_13, vd16_result_02);
}

// s8 = s8 / s8, vs8_u / vs8_v
AURA_INLINE HVX_Vector Q6_Vb_vdiv_VbVb(HVX_Vector vs8_u, HVX_Vector vs8_v)
{
    HVX_Vector vd8_dividend_abs = Q6_Vb_vabs_Vb(vs8_u);
    HVX_Vector vd8_divisor_abs = Q6_Vb_vabs_Vb(vs8_v);

    HVX_VectorPred q_sign = Q6_Q_vcmp_gt_VbVb(Q6_V_vzero(), vs8_u);
    q_sign = Q6_Q_vcmp_gtxacc_QVbVb(q_sign, Q6_V_vzero(), vs8_v);

    HVX_VectorPair wd16_divisor_0123 = Q6_Wuh_vmpy_VubRub(vd8_divisor_abs, 0x80808080);
    HVX_Vector vd16_divisor_02 = Q6_V_lo_W(wd16_divisor_0123);
    HVX_Vector vd16_divisor_13 = Q6_V_hi_W(wd16_divisor_0123);
    HVX_VectorPair wd16_dividend_02_13 = Q6_Wh_vadd_VubVub(vd8_dividend_abs, Q6_V_vzero());
    HVX_Vector vd16_dividend_02 = Q6_V_lo_W(wd16_dividend_02_13);
    HVX_Vector vd16_dividend_13 = Q6_V_hi_W(wd16_dividend_02_13);
    HVX_Vector vd16_result_02 = Q6_V_vzero();
    HVX_Vector vd16_result_13 = Q6_V_vzero();
    HVX_Vector vd16_div_bit = Q6_V_vsplat_R(0x00800080);
    HVX_VectorPred q_0, q_1;

    for (MI_S32 k = 0; k < 8; k++)
    {
        q_0 = Q6_Q_vcmp_gt_VuhVuh(vd16_divisor_02, vd16_dividend_02);
        q_1 = Q6_Q_vcmp_gt_VuhVuh(vd16_divisor_13, vd16_dividend_13);

        vd16_result_02 = Q6_Vh_condacc_QnVhVh(q_0, vd16_result_02, vd16_div_bit);
        vd16_result_13 = Q6_Vh_condacc_QnVhVh(q_1, vd16_result_13, vd16_div_bit);

        vd16_dividend_02 = Q6_Vh_condnac_QnVhVh(q_0, vd16_dividend_02, vd16_divisor_02);
        vd16_dividend_13 = Q6_Vh_condnac_QnVhVh(q_1, vd16_dividend_13, vd16_divisor_13);

        vd16_divisor_02 = Q6_Vh_vasr_VhR(vd16_divisor_02, 1);
        vd16_divisor_13 = Q6_Vh_vasr_VhR(vd16_divisor_13, 1);
        vd16_div_bit = Q6_Vh_vasr_VhR(vd16_div_bit, 1);
    }

    HVX_Vector vd8_out = Q6_Vb_vshuffe_VbVb(vd16_result_13, vd16_result_02);
    HVX_Vector vd8_out_neg = Q6_Vb_vsub_VbVb(Q6_V_vzero(), vd8_out);
    return Q6_V_vmux_QVV(q_sign, vd8_out_neg, vd8_out);
}

// u8 = u16 / u16, vu16_u / vu16_v, output is 8 bit in 16bit vector
AURA_INLINE HVX_Vector Q6_Vuh_vdiv8_VuhVuh(HVX_Vector vu16_u, HVX_Vector vu16_v)
{
    HVX_VectorPair wu32_divisor_0123 = Q6_Wuw_vmpy_VuhRuh(vu16_v, 0x00800080);
    HVX_Vector wu32_divisor_02 = Q6_V_lo_W(wu32_divisor_0123);
    HVX_Vector wu32_divisor_13 = Q6_V_hi_W(wu32_divisor_0123);
    HVX_VectorPair wu32_dividend_0213 = Q6_Ww_vadd_VuhVuh(vu16_u, Q6_V_vzero());
    HVX_Vector wu32_dividend_02 = Q6_V_lo_W(wu32_dividend_0213);
    HVX_Vector wu32_dividend_13 = Q6_V_hi_W(wu32_dividend_0213);
    HVX_Vector wd32_result_02 = Q6_V_vzero();
    HVX_Vector wd32_result_13 = Q6_V_vzero();
    HVX_Vector wd32_div_bit = Q6_V_vsplat_R(0x00000080);
    HVX_VectorPred q_0, q_1;

    for (MI_S32 k = 0; k < 8; k++)
    {
        q_0 = Q6_Q_vcmp_gt_VuwVuw(wu32_divisor_02, wu32_dividend_02);
        q_1 = Q6_Q_vcmp_gt_VuwVuw(wu32_divisor_13, wu32_dividend_13);

        wd32_result_02 = Q6_Vw_condacc_QnVwVw(q_0, wd32_result_02, wd32_div_bit);
        wd32_result_13 = Q6_Vw_condacc_QnVwVw(q_1, wd32_result_13, wd32_div_bit);

        wu32_dividend_02 = Q6_Vw_condnac_QnVwVw(q_0, wu32_dividend_02, wu32_divisor_02);
        wu32_dividend_13 = Q6_Vw_condnac_QnVwVw(q_1, wu32_dividend_13, wu32_divisor_13);

        wu32_divisor_02 = Q6_Vw_vasr_VwR(wu32_divisor_02, 1);
        wu32_divisor_13 = Q6_Vw_vasr_VwR(wu32_divisor_13, 1);
        wd32_div_bit = Q6_Vw_vasr_VwR(wd32_div_bit, 1);
    }

    return Q6_Vh_vshuffe_VhVh(wd32_result_13, wd32_result_02);
}

// s8 = s16 / s16, vs16_u / vs16_v, output is 8 bit in 16bit vector
AURA_INLINE HVX_Vector Q6_Vh_vdiv8_VhVh(HVX_Vector vs16_u, HVX_Vector vs16_v)
{
    HVX_Vector vs16_dividend_abs = Q6_Vh_vabs_Vh(vs16_u);
    HVX_Vector vs16_divisor_abs = Q6_Vh_vabs_Vh(vs16_v);
    HVX_VectorPred q_sign = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), vs16_u);
    q_sign = Q6_Q_vcmp_gtxacc_QVhVh(q_sign, Q6_V_vzero(), vs16_v);

    HVX_VectorPair wd32_divisor_0123 = Q6_Wuw_vmpy_VuhRuh(vs16_divisor_abs, 0x00800080);
    HVX_Vector vd32_divisor_02 = Q6_V_lo_W(wd32_divisor_0123);
    HVX_Vector vd32_divisor_13 = Q6_V_hi_W(wd32_divisor_0123);
    HVX_VectorPair wd32_dividend_0213 = Q6_Ww_vadd_VuhVuh(vs16_dividend_abs, Q6_V_vzero());
    HVX_Vector vd32_dividend_02 = Q6_V_lo_W(wd32_dividend_0213);
    HVX_Vector vd32_dividend_13 = Q6_V_hi_W(wd32_dividend_0213);
    HVX_Vector vd32_result_02 = Q6_V_vzero();
    HVX_Vector vd32_result_13 = Q6_V_vzero();
    HVX_Vector vd32_div_bit = Q6_V_vsplat_R(0x00000080);
    HVX_VectorPred q_0, q_1;

    for (MI_S32 k = 0; k < 8; k++)
    {
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vd32_divisor_02, vd32_dividend_02);
        q_1 = Q6_Q_vcmp_gt_VuwVuw(vd32_divisor_13, vd32_dividend_13);

        vd32_result_02 = Q6_Vw_condacc_QnVwVw(q_0, vd32_result_02, vd32_div_bit);
        vd32_result_13 = Q6_Vw_condacc_QnVwVw(q_1, vd32_result_13, vd32_div_bit);

        vd32_dividend_02 = Q6_Vw_condnac_QnVwVw(q_0, vd32_dividend_02, vd32_divisor_02);
        vd32_dividend_13 = Q6_Vw_condnac_QnVwVw(q_1, vd32_dividend_13, vd32_divisor_13);

        vd32_divisor_02 = Q6_Vw_vasr_VwR(vd32_divisor_02, 1);
        vd32_divisor_13 = Q6_Vw_vasr_VwR(vd32_divisor_13, 1);
        vd32_div_bit = Q6_Vw_vasr_VwR(vd32_div_bit, 1);
    }
    HVX_Vector vd16_out = Q6_Vh_vshuffe_VhVh(vd32_result_13, vd32_result_02);
    HVX_Vector vd16_out_neg = Q6_Vh_vsub_VhVh(Q6_V_vzero(), vd16_out);

    return Q6_V_vmux_QVV(q_sign, vd16_out_neg, vd16_out);
}

// u16 = u16 / u16, vu16_u / vu16_v
AURA_INLINE HVX_Vector Q6_Vuh_vdiv_VuhVuh(HVX_Vector vu16_u, HVX_Vector vu16_v)
{
    HVX_VectorPair wu32_divisor_0123 = Q6_Wuw_vmpy_VuhRuh(vu16_v, 0x80008000);
    HVX_Vector vu32_divisor_02 = Q6_V_lo_W(wu32_divisor_0123);
    HVX_Vector vu32_divisor_13 = Q6_V_hi_W(wu32_divisor_0123);
    HVX_VectorPair wu32_dividend_0213 = Q6_Ww_vadd_VuhVuh(vu16_u, Q6_V_vzero());
    HVX_Vector vu32_dividend_02 = Q6_V_lo_W(wu32_dividend_0213);
    HVX_Vector vu32_dividend_13 = Q6_V_hi_W(wu32_dividend_0213);
    HVX_Vector vu32_result_02 = Q6_V_vzero();
    HVX_Vector vu32_result_13 = Q6_V_vzero();
    HVX_Vector vu32_div_bit = Q6_V_vsplat_R(0x00008000);
    HVX_VectorPred q_0, q_1;

    for (MI_S32 k = 0; k < 16; k++)
    {
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vu32_divisor_02, vu32_dividend_02);
        q_1 = Q6_Q_vcmp_gt_VuwVuw(vu32_divisor_13, vu32_dividend_13);

        vu32_result_02 = Q6_Vw_condacc_QnVwVw(q_0, vu32_result_02, vu32_div_bit);
        vu32_result_13 = Q6_Vw_condacc_QnVwVw(q_1, vu32_result_13, vu32_div_bit);

        vu32_dividend_02 = Q6_Vw_condnac_QnVwVw(q_0, vu32_dividend_02, vu32_divisor_02);
        vu32_dividend_13 = Q6_Vw_condnac_QnVwVw(q_1, vu32_dividend_13, vu32_divisor_13);

        vu32_divisor_02 = Q6_Vw_vasr_VwR(vu32_divisor_02, 1);
        vu32_divisor_13 = Q6_Vw_vasr_VwR(vu32_divisor_13, 1);
        vu32_div_bit = Q6_Vw_vasr_VwR(vu32_div_bit, 1);
    }

    return Q6_Vh_vshuffe_VhVh(vu32_result_13, vu32_result_02);
}

// s16 = s16 / s16, vs16_u / vs16_v
AURA_INLINE HVX_Vector Q6_Vh_vdiv_VhVh(HVX_Vector vs16_u, HVX_Vector vs16_v)
{
    HVX_Vector vs16_dividend_abs = Q6_Vh_vabs_Vh(vs16_u);
    HVX_Vector vs16_divisor_abs = Q6_Vh_vabs_Vh(vs16_v);
    HVX_VectorPred q_sign = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), vs16_u);
    q_sign = Q6_Q_vcmp_gtxacc_QVhVh(q_sign, Q6_V_vzero(), vs16_v);

    HVX_VectorPair wd32_divisor_0123 = Q6_Wuw_vmpy_VuhRuh(vs16_divisor_abs, 0x80008000);
    HVX_Vector vd32_divisor_02 = Q6_V_lo_W(wd32_divisor_0123);
    HVX_Vector vd32_divisor_13 = Q6_V_hi_W(wd32_divisor_0123);
    HVX_VectorPair wd32_dividend_0213 = Q6_Ww_vadd_VuhVuh(vs16_dividend_abs, Q6_V_vzero());
    HVX_Vector vd32_dividend_02 = Q6_V_lo_W(wd32_dividend_0213);
    HVX_Vector vd32_dividend_13 = Q6_V_hi_W(wd32_dividend_0213);
    HVX_Vector vd32_result_02 = Q6_V_vzero();
    HVX_Vector vd32_result_13 = Q6_V_vzero();
    HVX_Vector vd32_div_bit = Q6_V_vsplat_R(0x00008000);
    HVX_VectorPred q_0, q_1;

    for (MI_S32 k = 0; k < 16; k++)
    {
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vd32_divisor_02, vd32_dividend_02);
        q_1 = Q6_Q_vcmp_gt_VuwVuw(vd32_divisor_13, vd32_dividend_13);

        vd32_result_02 = Q6_Vw_condacc_QnVwVw(q_0, vd32_result_02, vd32_div_bit);
        vd32_result_13 = Q6_Vw_condacc_QnVwVw(q_1, vd32_result_13, vd32_div_bit);

        vd32_dividend_02 = Q6_Vw_condnac_QnVwVw(q_0, vd32_dividend_02, vd32_divisor_02);
        vd32_dividend_13 = Q6_Vw_condnac_QnVwVw(q_1, vd32_dividend_13, vd32_divisor_13);

        vd32_divisor_02 = Q6_Vw_vasr_VwR(vd32_divisor_02, 1);
        vd32_divisor_13 = Q6_Vw_vasr_VwR(vd32_divisor_13, 1);
        vd32_div_bit = Q6_Vw_vasr_VwR(vd32_div_bit, 1);
    }
    HVX_Vector vd16_out = Q6_Vh_vshuffe_VhVh(vd32_result_13, vd32_result_02);
    HVX_Vector vd16_out_neg = Q6_Vh_vsub_VhVh(Q6_V_vzero(), vd16_out);

    return Q6_V_vmux_QVV(q_sign, vd16_out_neg, vd16_out);
}

// u8 = u32 / u16, wu32_u / vu16_v
// vu32_u0 = *src++;
// vu32_u1 = *src++;
// wu32_u  = Q6_W_vdeal_VVR(vu32_u1, vu32_u0, -4);
// output is 8 bit in 16bit vector
AURA_INLINE HVX_Vector Q6_Vuh_vdiv8_WuwVuh(HVX_VectorPair wu32_u, HVX_Vector vu16_v)
{
    HVX_VectorPair wu32_denom_0426;
    HVX_Vector vu32_denom_04, vu32_denom_26;
    HVX_Vector vu32_result_04, vu32_result_26;
    HVX_Vector vu32_pixel_sum_04, vu32_pixel_sum_26;
    HVX_Vector vu32_div_bit;
    HVX_VectorPred q_0, q_1;

    wu32_denom_0426 = Q6_Wuw_vmpy_VuhRuh(vu16_v, 0x00800080);

    vu32_denom_04 = Q6_V_lo_W(wu32_denom_0426);
    vu32_denom_26 = Q6_V_hi_W(wu32_denom_0426);

    vu32_pixel_sum_04 = Q6_V_lo_W(wu32_u);
    vu32_pixel_sum_26 = Q6_V_hi_W(wu32_u);

    vu32_result_04 = Q6_V_vzero();
    vu32_result_26 = Q6_V_vzero();

    vu32_div_bit = Q6_V_vsplat_R(0x00800080);

    for (MI_S32 k = 0; k < 8; k++)
    {
        q_0 = Q6_Q_vcmp_gt_VwVw(vu32_denom_04, vu32_pixel_sum_04);
        q_1 = Q6_Q_vcmp_gt_VwVw(vu32_denom_26, vu32_pixel_sum_26);

        vu32_result_04 = Q6_Vw_condacc_QnVwVw(q_0, vu32_result_04, vu32_div_bit);
        vu32_result_26 = Q6_Vw_condacc_QnVwVw(q_1, vu32_result_26, vu32_div_bit);

        vu32_pixel_sum_04 = Q6_Vw_condnac_QnVwVw(q_0, vu32_pixel_sum_04, vu32_denom_04);
        vu32_pixel_sum_26 = Q6_Vw_condnac_QnVwVw(q_1, vu32_pixel_sum_26, vu32_denom_26);

        vu32_denom_04 = Q6_Vw_vasr_VwR(vu32_denom_04, 1);
        vu32_denom_26 = Q6_Vw_vasr_VwR(vu32_denom_26, 1);

        vu32_div_bit  = Q6_Vw_vasr_VwR(vu32_div_bit, 1);
    }

    return Q6_Vh_vshuffe_VhVh(vu32_result_26, vu32_result_04);
}

// s8 = s32 / s16, ws32_u / vs16_v
// vs32_u0 = *src++;
// vs32_u1 = *src++;
// ws32_u  = Q6_W_vdeal_VVR(vs32_u1, vs32_u0, -4);
// output is 8 bit in 16bit vector
AURA_INLINE HVX_Vector Q6_Vh_vdiv8_WwVh(HVX_VectorPair ws32_u, HVX_Vector vs16_v)
{
    HVX_Vector vs16_result;
    HVX_VectorPair wd32_denom_0426;
    HVX_Vector vd32_denom_04, vd32_denom_26;
    HVX_Vector vd32_result_04, vd32_result_26;
    HVX_Vector vd32_pixel_sum_04, vd32_pixel_sum_26;
    HVX_Vector vd32_div_bit;
    HVX_VectorPred q_0, q_1;

    HVX_Vector vs16_divisor_abs = Q6_Vh_vabs_Vh(vs16_v);
    wd32_denom_0426 = Q6_Wuw_vmpy_VuhRuh(vs16_divisor_abs, 0x00800080);

    HVX_Vector vd16_flag_v = Q6_Vh_vasr_VwVwR_sat(Q6_V_hi_W(ws32_u), Q6_V_lo_W(ws32_u), 16);
    HVX_VectorPred q_sign = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), vd16_flag_v);
    q_sign = Q6_Q_vcmp_gtxacc_QVhVh(q_sign, Q6_V_vzero(), vs16_v);

    vd32_denom_04 = Q6_V_lo_W(wd32_denom_0426);
    vd32_denom_26 = Q6_V_hi_W(wd32_denom_0426);

    vd32_pixel_sum_04 = Q6_Vw_vabs_Vw(Q6_V_lo_W(ws32_u));
    vd32_pixel_sum_26 = Q6_Vw_vabs_Vw(Q6_V_hi_W(ws32_u));

    vd32_result_04 = Q6_V_vzero();
    vd32_result_26 = Q6_V_vzero();

    vd32_div_bit = Q6_V_vsplat_R(0x00000080);

    for (MI_S32 k = 0; k < 8; k++)
    {
        q_0 = Q6_Q_vcmp_gt_VwVw(vd32_denom_04, vd32_pixel_sum_04);
        q_1 = Q6_Q_vcmp_gt_VwVw(vd32_denom_26, vd32_pixel_sum_26);

        vd32_result_04 = Q6_Vw_condacc_QnVwVw(q_0, vd32_result_04, vd32_div_bit);
        vd32_result_26 = Q6_Vw_condacc_QnVwVw(q_1, vd32_result_26, vd32_div_bit);

        vd32_pixel_sum_04 = Q6_Vw_condnac_QnVwVw(q_0, vd32_pixel_sum_04, vd32_denom_04);
        vd32_pixel_sum_26 = Q6_Vw_condnac_QnVwVw(q_1, vd32_pixel_sum_26, vd32_denom_26);

        vd32_denom_04 = Q6_Vw_vasr_VwR(vd32_denom_04, 1);
        vd32_denom_26 = Q6_Vw_vasr_VwR(vd32_denom_26, 1);

        vd32_div_bit  = Q6_Vw_vasr_VwR(vd32_div_bit, 1);
    }

    vs16_result = Q6_Vh_vshuffe_VhVh(vd32_result_26, vd32_result_04);
    return Q6_V_vmux_QVV(q_sign, Q6_Vh_vmpyi_VhRb(vs16_result, Q6_R_vsplatb_R(-1)), vs16_result);
}

// u16 = u32 / u16, wu32_u / vu16_v
// vu32_u0 = *src++;
// vu32_u1 = *src++;
// wu32_u  = Q6_W_vdeal_VVR(vu32_u1, vu32_u0, -4);
AURA_INLINE HVX_Vector Q6_Vuh_vdiv_WuwVuh(HVX_VectorPair wu32_u, HVX_Vector vu16_v)
{
    HVX_VectorPair wu32_denom;
    HVX_Vector vu32_denom_02, vu32_denom_13;
    HVX_Vector vu32_result_02, vu32_result_13;
    HVX_Vector vu32_pixel_sum_02, vu32_pixel_sum_13;
    HVX_Vector vu32_div_bit;
    HVX_VectorPred q_0, q_1;

    wu32_denom = Q6_Wuw_vmpy_VuhRuh(vu16_v, 0x80008000);

    vu32_denom_02 = Q6_V_lo_W(wu32_denom);
    vu32_denom_13 = Q6_V_hi_W(wu32_denom);

    vu32_pixel_sum_02 = Q6_V_lo_W(wu32_u);
    vu32_pixel_sum_13 = Q6_V_hi_W(wu32_u);

    vu32_result_02 = Q6_V_vzero();
    vu32_result_13 = Q6_V_vzero();

    vu32_div_bit = Q6_V_vsplat_R(0x00008000);

    for (MI_S32 k = 0; k < 16; k++)
    {
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vu32_denom_02, vu32_pixel_sum_02);
        q_1 = Q6_Q_vcmp_gt_VuwVuw(vu32_denom_13, vu32_pixel_sum_13);

        vu32_result_02 = Q6_Vw_condacc_QnVwVw(q_0, vu32_result_02, vu32_div_bit);
        vu32_result_13 = Q6_Vw_condacc_QnVwVw(q_1, vu32_result_13, vu32_div_bit);

        vu32_pixel_sum_02 = Q6_Vw_condnac_QnVwVw(q_0, vu32_pixel_sum_02, vu32_denom_02);
        vu32_pixel_sum_13 = Q6_Vw_condnac_QnVwVw(q_1, vu32_pixel_sum_13, vu32_denom_13);

        vu32_denom_02 = Q6_Vw_vasr_VwR(vu32_denom_02, 1);
        vu32_denom_13 = Q6_Vw_vasr_VwR(vu32_denom_13, 1);
        vu32_div_bit = Q6_Vw_vasr_VwR(vu32_div_bit, 1);
    }

    return Q6_Vh_vshuffe_VhVh(vu32_result_13, vu32_result_02);
}

// s16 = s32 / s16, ws32_u / vs16_v
// vs32_u0 = *src++;
// vs32_u1 = *src++;
// ws32_u  = Q6_W_vdeal_VVR(vs32_u1, vs32_u0, -4);
AURA_INLINE HVX_Vector Q6_Vh_vdiv_WwVh(HVX_VectorPair ws32_u, HVX_Vector vs16_v)
{
    HVX_Vector vs16_result, vs16_result_rev;
    HVX_VectorPair wd32_denom;
    HVX_Vector vd32_denom_02, vd32_denom_13;
    HVX_Vector vd32_result_02, vd32_result_13;
    HVX_Vector vd32_pixel_sum_02, vd32_pixel_sum_13;
    HVX_Vector vd32_div_bit;
    HVX_VectorPred q_0, q_1, q_sign;
    HVX_Vector vd16_flag_v;

    vd16_flag_v = Q6_Vh_vasr_VwVwR_sat(Q6_V_hi_W(ws32_u), Q6_V_lo_W(ws32_u), 16);
    q_sign = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), vd16_flag_v);
    q_sign = Q6_Q_vcmp_gtxacc_QVhVh(q_sign, Q6_V_vzero(), vs16_v);

    HVX_Vector vu16_divisor_abs = Q6_Vh_vabs_Vh(vs16_v);
    wd32_denom = Q6_Wuw_vmpy_VuhRuh(vu16_divisor_abs, 0x80008000);

    vd32_denom_02 = Q6_V_lo_W(wd32_denom);
    vd32_denom_13 = Q6_V_hi_W(wd32_denom);

    vd32_pixel_sum_02 = Q6_Vw_vabs_Vw(Q6_V_lo_W(ws32_u));
    vd32_pixel_sum_13 = Q6_Vw_vabs_Vw(Q6_V_hi_W(ws32_u));
    vd32_result_02 = Q6_V_vzero();
    vd32_result_13 = Q6_V_vzero();
    vd32_div_bit = Q6_V_vsplat_R(0x80008000);

    for (MI_S32 k = 0; k < 16; k++)
    {
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vd32_denom_02, vd32_pixel_sum_02);
        q_1 = Q6_Q_vcmp_gt_VuwVuw(vd32_denom_13, vd32_pixel_sum_13);

        vd32_result_02 = Q6_Vw_condacc_QnVwVw(q_0, vd32_result_02, vd32_div_bit);
        vd32_result_13 = Q6_Vw_condacc_QnVwVw(q_1, vd32_result_13, vd32_div_bit);

        vd32_pixel_sum_02 = Q6_Vw_condnac_QnVwVw(q_0, vd32_pixel_sum_02, vd32_denom_02);
        vd32_pixel_sum_13 = Q6_Vw_condnac_QnVwVw(q_1, vd32_pixel_sum_13, vd32_denom_13);

        vd32_denom_02 = Q6_Vw_vasr_VwR(vd32_denom_02, 1);
        vd32_denom_13 = Q6_Vw_vasr_VwR(vd32_denom_13, 1);
        vd32_div_bit = Q6_Vw_vasr_VwR(vd32_div_bit, 1);
    }
    vs16_result = Q6_Vh_vshuffe_VhVh(vd32_result_13, vd32_result_02);
    vs16_result_rev = Q6_Vh_vmpyi_VhRb(vs16_result, Q6_R_vsplatb_R(-1));
    vs16_result = Q6_V_vmux_QVV(q_sign, vs16_result_rev, vs16_result);

    return vs16_result;
}

// u8 = u32 / u32, vu32_u / vu32_v
AURA_INLINE HVX_Vector Q6_Vuw_vdiv8_VuwVuw(HVX_Vector vu32_u, HVX_Vector vu32_v)
{
    HVX_Vector vd32_result;
    HVX_Vector vd32_div_bit;
    HVX_VectorPred q_0;
    MI_S32 shift_val = 7;

    HVX_Vector vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vu32_u, 7);
    HVX_Vector vd32_divisor_shift = Q6_Vw_vasl_VwR(vu32_v, shift_val);

    vd32_result = Q6_V_vzero();
    vd32_div_bit = Q6_V_vsplat_R(0x00000080);

    for (MI_S32 k = 0; k < 7; k++)
    {
        shift_val--;
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vu32_v, vd32_dividend_tmp);
        vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);
        vu32_u = Q6_Vw_condnac_QnVwVw(q_0, vu32_u, vd32_divisor_shift);

        vd32_div_bit = Q6_Vuw_vavg_VuwVuw(vd32_div_bit, Q6_V_vzero());
        vd32_divisor_shift = Q6_Vw_vasl_VwR(vu32_v, shift_val);
        vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vu32_u, shift_val);
    }

    q_0 = Q6_Q_vcmp_gt_VuwVuw(vu32_v, vd32_dividend_tmp);
    vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);

    return vd32_result;
}

// s8 = s32 / s32, vs32_u / vs32_v
AURA_INLINE HVX_Vector Q6_Vw_vdiv8_VwVw(HVX_Vector vs32_u, HVX_Vector vs32_v)
{
    HVX_Vector vd32_result;
    HVX_Vector vd32_div_bit;
    HVX_VectorPred q_0;
    MI_S32 shift_val = 7;

    HVX_VectorPred q_sign = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), vs32_u);
    q_sign = Q6_Q_vcmp_gtxacc_QVwVw(q_sign, Q6_V_vzero(), vs32_v);

    vs32_u = Q6_Vw_vabs_Vw(vs32_u);
    vs32_v = Q6_Vw_vabs_Vw(vs32_v);

    HVX_Vector vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vs32_u, shift_val);
    HVX_Vector vd32_divisor_shift = Q6_Vw_vasl_VwR(vs32_v, shift_val);

    vd32_result = Q6_V_vzero();
    vd32_div_bit = Q6_V_vsplat_R(0x00000080);

    for (MI_S32 k = 0; k < 7; k++)
    {
        shift_val--;
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vs32_v, vd32_dividend_tmp);
        vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);
        vs32_u = Q6_Vw_condnac_QnVwVw(q_0, vs32_u, vd32_divisor_shift);

        vd32_div_bit = Q6_Vuw_vavg_VuwVuw(vd32_div_bit, Q6_V_vzero());
        vd32_divisor_shift = Q6_Vw_vasl_VwR(vs32_v, shift_val);
        vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vs32_u, shift_val);
    }
    q_0 = Q6_Q_vcmp_gt_VuwVuw(vs32_v, vd32_dividend_tmp);
    vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);

    HVX_Vector result_neg = Q6_Vw_vsub_VwVw(Q6_V_vzero(), vd32_result);
    vd32_result = Q6_V_vmux_QVV(q_sign, result_neg, vd32_result);

    return vd32_result;
}

// u16 = u32 / u32, vu32_u / vu32_v
AURA_INLINE HVX_Vector Q6_Vuw_vdiv16_VuwVuw(HVX_Vector vu32_u, HVX_Vector vu32_v)
{
    HVX_Vector vd32_result;
    HVX_Vector vd32_div_bit;
    HVX_VectorPred q_0;
    MI_S32 shift_val = 15;

    HVX_Vector vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vu32_u, 15);
    HVX_Vector vd32_divisor_shift = Q6_Vw_vasl_VwR(vu32_v, shift_val);

    vd32_result = Q6_V_vzero();
    vd32_div_bit = Q6_V_vsplat_R(0x00008000);

    for (MI_S32 k = 0; k < 15; k++)
    {
        shift_val--;
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vu32_v, vd32_dividend_tmp);
        vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);
        vu32_u = Q6_Vw_condnac_QnVwVw(q_0, vu32_u, vd32_divisor_shift);

        vd32_div_bit = Q6_Vuw_vavg_VuwVuw(vd32_div_bit, Q6_V_vzero());
        vd32_divisor_shift = Q6_Vw_vasl_VwR(vu32_v, shift_val);
        vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vu32_u, shift_val);
    }
    q_0 = Q6_Q_vcmp_gt_VuwVuw(vu32_v, vd32_dividend_tmp);
    vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);

    return vd32_result;
}

// s16 = s32 / s32, vs32_u / vs32_v
AURA_INLINE HVX_Vector Q6_Vw_vdiv16_VwVw(HVX_Vector vs32_u, HVX_Vector vs32_v)
{
    HVX_Vector vd32_result;
    HVX_Vector vd32_div_bit;
    HVX_VectorPred q_0;
    MI_S32 shift_val = 15;

    HVX_VectorPred q_sign = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), vs32_u);
    q_sign = Q6_Q_vcmp_gtxacc_QVwVw(q_sign, Q6_V_vzero(), vs32_v);

    vs32_u = Q6_Vw_vabs_Vw(vs32_u);
    vs32_v = Q6_Vw_vabs_Vw(vs32_v);

    HVX_Vector vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vs32_u, shift_val);
    HVX_Vector vd32_divisor_shift = Q6_Vw_vasl_VwR(vs32_v, shift_val);

    vd32_result = Q6_V_vzero();
    vd32_div_bit = Q6_V_vsplat_R(0x00008000);

    for (MI_S32 k = 0; k < 15; k++)
    {
        shift_val--;
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vs32_v, vd32_dividend_tmp);
        vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);
        vs32_u = Q6_Vw_condnac_QnVwVw(q_0, vs32_u, vd32_divisor_shift);

        vd32_div_bit = Q6_Vuw_vavg_VuwVuw(vd32_div_bit, Q6_V_vzero());
        vd32_divisor_shift = Q6_Vw_vasl_VwR(vs32_v, shift_val);
        vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vs32_u, shift_val);
    }
    q_0 = Q6_Q_vcmp_gt_VuwVuw(vs32_v, vd32_dividend_tmp);
    vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);

    HVX_Vector vd32_result_neg = Q6_Vw_vsub_VwVw(Q6_V_vzero(), vd32_result);
    vd32_result = Q6_V_vmux_QVV(q_sign, vd32_result_neg, vd32_result);

    return vd32_result;
}

// u32 = u32 / u32, vu32_u / vu32_v
AURA_INLINE HVX_Vector Q6_Vuw_vdiv_VuwVuw(HVX_Vector vu32_u, HVX_Vector vu32_v)
{
    HVX_Vector vd32_result;
    HVX_Vector vd32_div_bit;
    HVX_VectorPred q_0;
    MI_S32 shift_val = 31;

    HVX_Vector vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vu32_u, 31);
    HVX_Vector vd32_divisor_shift = Q6_Vw_vasl_VwR(vu32_v, shift_val);

    vd32_result = Q6_V_vzero();
    vd32_div_bit = Q6_V_vsplat_R(0x80000000);

    for (MI_S32 k = 0; k < 31; k++)
    {
        shift_val--;
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vu32_v, vd32_dividend_tmp);
        vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);
        vu32_u = Q6_Vw_condnac_QnVwVw(q_0, vu32_u, vd32_divisor_shift);

        vd32_div_bit = Q6_Vuw_vavg_VuwVuw(vd32_div_bit, Q6_V_vzero());
        vd32_divisor_shift = Q6_Vw_vasl_VwR(vu32_v, shift_val);
        vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vu32_u, shift_val);
    }
    q_0 = Q6_Q_vcmp_gt_VuwVuw(vu32_v, vd32_dividend_tmp);
    vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);

    return vd32_result;
}

// s32 = s32 / s32, vs32_u / vs32_v
AURA_INLINE HVX_Vector Q6_Vw_vdiv_VwVw(HVX_Vector vs32_u, HVX_Vector vs32_v)
{
    HVX_Vector vd32_result;
    HVX_Vector vd32_div_bit;
    HVX_VectorPred q_0;
    MI_S32 shift_val = 31;

    HVX_VectorPred q_sign = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), vs32_u);
    q_sign = Q6_Q_vcmp_gtxacc_QVwVw(q_sign, Q6_V_vzero(), vs32_v);

    vs32_u = Q6_Vw_vabs_Vw(vs32_u);
    vs32_v = Q6_Vw_vabs_Vw(vs32_v);

    HVX_Vector vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vs32_u, shift_val);
    HVX_Vector vd32_divisor_shift = Q6_Vw_vasl_VwR(vs32_v, shift_val);

    vd32_result = Q6_V_vzero();
    vd32_div_bit = Q6_V_vsplat_R(0x80000000);

    for (MI_S32 k = 0; k < 31; k++)
    {
        shift_val--;
        q_0 = Q6_Q_vcmp_gt_VuwVuw(vs32_v, vd32_dividend_tmp);
        vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);
        vs32_u = Q6_Vw_condnac_QnVwVw(q_0, vs32_u, vd32_divisor_shift);

        vd32_div_bit = Q6_Vuw_vavg_VuwVuw(vd32_div_bit, Q6_V_vzero());
        vd32_divisor_shift = Q6_Vw_vasl_VwR(vs32_v, shift_val);
        vd32_dividend_tmp = Q6_Vuw_vlsr_VuwR(vs32_u, shift_val);
    }
    q_0 = Q6_Q_vcmp_gt_VuwVuw(vs32_v, vd32_dividend_tmp);
    vd32_result = Q6_Vw_condacc_QnVwVw(q_0, vd32_result, vd32_div_bit);

    HVX_Vector vd32_result_neg = Q6_Vw_vsub_VwVw(Q6_V_vzero(), vd32_result);
    vd32_result = Q6_V_vmux_QVV(q_sign, vd32_result_neg, vd32_result);

    return vd32_result;
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_DIV_HPP__