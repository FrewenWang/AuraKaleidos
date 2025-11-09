#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_DIVN_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_DIVN_HPP__

#include "aura/runtime/core/types.h"

#include "hexagon_types.h"

namespace aura
{

/************************************ u8 vdvin ************************************/
template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U8>::value && DENOM == 3>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu8_src)
{
    HVX_VectorPair wu16_mul   = Q6_Wuh_vmpy_VubRub(vu8_src, 0xABABABAB);
    HVX_Vector     vu8_result = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(wu16_mul), Q6_V_lo_W(wu16_mul));
                   vu8_result = Q6_Vub_vlsr_VubR(vu8_result, 1);

    return vu8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U8>::value && DENOM == 5>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu8_src)
{
    HVX_VectorPair wu16_mul   = Q6_Wuh_vmpy_VubRub(vu8_src,  0xCDCDCDCD);
    HVX_Vector     vu8_result = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(wu16_mul), Q6_V_lo_W(wu16_mul));
                   vu8_result = Q6_Vub_vlsr_VubR(vu8_result, 2);

    return vu8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U8>::value && DENOM == 7>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu8_src)
{
    HVX_VectorPair wu16_mul   = Q6_Wuh_vmpy_VubRub(vu8_src,  0x25252525);
    HVX_VectorPair wu16_add   = Q6_Wh_vadd_VubVub(Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(wu16_mul), Q6_V_lo_W(wu16_mul)), vu8_src);
    HVX_Vector     vu8_result = Q6_Vb_vshuffe_VbVb(Q6_Vuh_vlsr_VuhR(Q6_V_hi_W(wu16_add), 3), Q6_Vuh_vlsr_VuhR(Q6_V_lo_W(wu16_add), 3));

    return vu8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U8>::value && DENOM == 9>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu8_src)
{
    HVX_VectorPair wu16_mul   = Q6_Wuh_vmpy_VubRub(vu8_src, 0x39393939);
    HVX_Vector     vu8_result = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(wu16_mul), Q6_V_lo_W(wu16_mul));
                   vu8_result = Q6_Vub_vlsr_VubR(vu8_result, 1);

    return vu8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U8>::value && DENOM == 25>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu8_src)
{
    HVX_VectorPair wu16_mul   = Q6_Wuh_vmpy_VubRub(vu8_src, 0x29292929);
    HVX_Vector     vu8_result = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(wu16_mul), Q6_V_lo_W(wu16_mul));
                   vu8_result = Q6_Vub_vlsr_VubR(vu8_result, 2);

    return vu8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U8>::value && DENOM == 49>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu8_src)
{
    HVX_VectorPair wu16_mul   = Q6_Wuh_vmpy_VubRub(vu8_src,  0x4F4F4F4F);
    HVX_VectorPair wu16_add   = Q6_Wh_vadd_VubVub(Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(wu16_mul), Q6_V_lo_W(wu16_mul)), vu8_src);
    HVX_Vector     vu8_result = Q6_Vb_vshuffe_VbVb(Q6_Vuh_vlsr_VuhR(Q6_V_hi_W(wu16_add), 6), Q6_Vuh_vlsr_VuhR(Q6_V_lo_W(wu16_add), 6));

    return vu8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U8>::value &&
                                  (DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                  (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu8_src)
{
    constexpr DT_U8 LENGTH = 32 - __builtin_clz(DENOM - 1);
    constexpr DT_U8 MUL    = (0 == DENOM) ? 0 : (((DT_U16)256) * ((1 << LENGTH) - DENOM)) / DENOM + 1;
    constexpr DT_U8 SHIFT0 = (LENGTH > 1) ? 1 : LENGTH;
    constexpr DT_U8 SHIFT1 = (0 == DENOM) ? 7 : ((DT_S32)LENGTH > 1) ? (DT_S8)(LENGTH - 1) : 0;

    const static HVX_Vector vu8_mul = Q6_Vb_vsplat_R(MUL);

    HVX_Vector vu8_result;
    HVX_VectorPair wu16_mul;
    HVX_Vector vu8_shu, vu8_sub;

    wu16_mul   = Q6_Wuh_vmpy_VubVub(vu8_src, vu8_mul);
    vu8_shu    = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(wu16_mul), Q6_V_lo_W(wu16_mul));
    vu8_sub    = Q6_Vub_vsub_VubVub_sat(vu8_src, vu8_shu);
    vu8_sub    = Q6_Vub_vlsr_VubR(vu8_sub, SHIFT0);
    vu8_result = Q6_Vub_vadd_VubVub_sat(vu8_sub, vu8_shu);
    vu8_result = Q6_Vub_vlsr_VubR(vu8_result, SHIFT1);

    return vu8_result;
}

/************************************ s8 vdvin ************************************/
template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S8>::value && DENOM == 3>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs8_src)
{
    HVX_VectorPair ws16_mul     = Q6_Wh_vmpy_VubVb(Q6_Vb_vsplat_R(171), vs8_src);
    HVX_VectorPred qs16_pred_lo = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_lo_W(ws16_mul));
    HVX_VectorPred qs16_pred_hi = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_hi_W(ws16_mul));
    HVX_Vector     vs16_add_lo  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_mul), Q6_V_vand_QR(qs16_pred_lo, 0x1ff01ff));
    HVX_Vector     vs16_add_hi  = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_mul), Q6_V_vand_QR(qs16_pred_hi, 0x1ff01ff));
    HVX_Vector     vs8_result   = Q6_Vb_vshuffe_VbVb(Q6_Vh_vasr_VhR(vs16_add_hi, 9), Q6_Vh_vasr_VhR(vs16_add_lo, 9));

    return vs8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S8>::value && DENOM == 5>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs8_src)
{
    HVX_VectorPair ws16_mul     = Q6_Wh_vmpy_VubVb(Q6_Vb_vsplat_R(205), vs8_src);
    HVX_VectorPred qs16_pred_lo = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_lo_W(ws16_mul));
    HVX_VectorPred qs16_pred_hi = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_hi_W(ws16_mul));
    HVX_Vector     vs16_add_lo  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_mul), Q6_V_vand_QR(qs16_pred_lo, 0x3ff03ff));
    HVX_Vector     vs16_add_hi  = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_mul), Q6_V_vand_QR(qs16_pred_hi, 0x3ff03ff));
    HVX_Vector     vs8_result   = Q6_Vb_vshuffe_VbVb(Q6_Vh_vasr_VhR(vs16_add_hi, 10), Q6_Vh_vasr_VhR(vs16_add_lo, 10));

    return vs8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S8>::value && DENOM == 7>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs8_src)
{
    HVX_VectorPair ws16_mul     = Q6_Wh_vmpy_VubVb(Q6_Vb_vsplat_R(147), vs8_src);
    HVX_VectorPred qs16_pred_lo = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_lo_W(ws16_mul));
    HVX_VectorPred qs16_pred_hi = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_hi_W(ws16_mul));
    HVX_Vector     vs16_add_lo  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_mul), Q6_V_vand_QR(qs16_pred_lo, 0x3ff03ff));
    HVX_Vector     vs16_add_hi  = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_mul), Q6_V_vand_QR(qs16_pred_hi, 0x3ff03ff));
    HVX_Vector     vs8_result   = Q6_Vb_vshuffe_VbVb(Q6_Vh_vasr_VhR(vs16_add_hi, 10), Q6_Vh_vasr_VhR(vs16_add_lo, 10));

    return vs8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S8>::value && DENOM == 9>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs8_src)
{
    HVX_VectorPair ws16_mul     = Q6_Wh_vmpy_VubVb(Q6_Vb_vsplat_R(57), vs8_src);
    HVX_VectorPred qs16_pred_lo = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_lo_W(ws16_mul));
    HVX_VectorPred qs16_pred_hi = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_hi_W(ws16_mul));
    HVX_Vector     vs16_add_lo  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_mul), Q6_V_vand_QR(qs16_pred_lo, 0x1ff01ff));
    HVX_Vector     vs16_add_hi  = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_mul), Q6_V_vand_QR(qs16_pred_hi, 0x1ff01ff));
    HVX_Vector     vs8_result   = Q6_Vb_vshuffe_VbVb(Q6_Vh_vasr_VhR(vs16_add_hi, 9), Q6_Vh_vasr_VhR(vs16_add_lo, 9));

    return vs8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S8>::value && DENOM == 25>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs8_src)
{
    HVX_VectorPair ws16_mul     = Q6_Wh_vmpy_VubVb(Q6_Vb_vsplat_R(41), vs8_src);
    HVX_VectorPred qs16_pred_lo = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_lo_W(ws16_mul));
    HVX_VectorPred qs16_pred_hi = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_hi_W(ws16_mul));
    HVX_Vector     vs16_add_lo  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_mul), Q6_V_vand_QR(qs16_pred_lo, 0x3ff03ff));
    HVX_Vector     vs16_add_hi  = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_mul), Q6_V_vand_QR(qs16_pred_hi, 0x3ff03ff));
    HVX_Vector     vs8_result   = Q6_Vb_vshuffe_VbVb(Q6_Vh_vasr_VhR(vs16_add_hi, 10), Q6_Vh_vasr_VhR(vs16_add_lo, 10));

    return vs8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S8>::value && DENOM == 49>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs8_src)
{
    HVX_VectorPair ws16_mul     = Q6_Wh_vmpy_VubVb(Q6_Vb_vsplat_R(42), vs8_src);
    HVX_VectorPred qs16_pred_lo = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_lo_W(ws16_mul));
    HVX_VectorPred qs16_pred_hi = Q6_Q_vcmp_gt_VhVh(Q6_V_vzero(), Q6_V_hi_W(ws16_mul));
    HVX_Vector     vs16_add_lo  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_mul), Q6_V_vand_QR(qs16_pred_lo, 0x7ff07ff));
    HVX_Vector     vs16_add_hi  = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_mul), Q6_V_vand_QR(qs16_pred_hi, 0x7ff07ff));
    HVX_Vector     vs8_result   = Q6_Vb_vshuffe_VbVb(Q6_Vh_vasr_VhR(vs16_add_hi, 11), Q6_Vh_vasr_VhR(vs16_add_lo, 11));

    return vs8_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S8>::value &&
                                  (DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                  (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs8_src)
{
    constexpr DT_U8  ABS_D  = (DENOM >= 0) ? DENOM : -DENOM;
    constexpr DT_S8  BITS   = static_cast<DT_S8>(32 - __builtin_clz(ABS_D - 1));
    constexpr DT_S8  LENGTH = (BITS > 1) ? BITS : 1;
    constexpr DT_S16 MUL    = (0 == DENOM) ? 0 : 1 + ((DT_U16(1)) << (8 + LENGTH - 1)) / ABS_D - ((DT_U16)256);
    constexpr DT_S8  SHIFT  = (0 == DENOM) ? 7 : LENGTH - 1;

    const static HVX_Vector vs8_mul  = Q6_Vb_vsplat_R(MUL);
    const static HVX_Vector vs8_sign = Q6_Vb_vsplat_R(DENOM >> (7));

    HVX_Vector vs8_result;
    HVX_VectorPair ws16_mul, wu16_add, wu16_input;
    HVX_Vector vs8_add, vs8_sub0, vs8_sub1;

    ws16_mul   = Q6_Wh_vmpy_VbVb(vs8_src, vs8_mul);
    vs8_add    = Q6_Vb_vadd_VbVb(vs8_src, Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(ws16_mul), Q6_V_lo_W(ws16_mul)));
    wu16_add   = Q6_Wh_vunpack_Vb(vs8_add);
    wu16_input = Q6_Wh_vunpack_Vb(vs8_src);
    vs8_sub0   = Q6_Vb_vpack_VhVh_sat(Q6_Vh_vasr_VhR(Q6_V_hi_W(wu16_add), SHIFT), Q6_Vh_vasr_VhR(Q6_V_lo_W(wu16_add), SHIFT));
    vs8_sub1   = Q6_Vb_vpack_VhVh_sat(Q6_Vh_vasr_VhR(Q6_V_hi_W(wu16_input), 7), Q6_Vh_vasr_VhR(Q6_V_lo_W(wu16_input), 7));
    vs8_add    = Q6_Vb_vsub_VbVb(vs8_sub0, vs8_sub1);
    vs8_result = Q6_Vb_vsub_VbVb(Q6_V_vxor_VV(vs8_add, vs8_sign), vs8_sign);

    return vs8_result;
}

/************************************ u16 vdvin ************************************/
template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U16>::value && DENOM == 3>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu16_src)
{
    HVX_VectorPair wu32_mul    = Q6_Wuw_vmpy_VuhRuh(vu16_src, 0xAAABAAAB);
    HVX_Vector     vu16_result = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wu32_mul), Q6_V_lo_W(wu32_mul));
                   vu16_result = Q6_Vuh_vlsr_VuhR(vu16_result, 1);

    return vu16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U16>::value && DENOM == 5>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu16_src)
{
    HVX_VectorPair wu32_mul    = Q6_Wuw_vmpy_VuhRuh(vu16_src, 0xCCCDCCCD);
    HVX_Vector     vu16_result = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wu32_mul), Q6_V_lo_W(wu32_mul));
                   vu16_result = Q6_Vuh_vlsr_VuhR(vu16_result, 2);

    return vu16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U16>::value && DENOM == 7>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu16_src)
{
    HVX_VectorPair wu32_mul    = Q6_Wuw_vmpy_VuhRuh(vu16_src, 0x24932493);
    HVX_VectorPair wu32_add    = Q6_Ww_vadd_VuhVuh(Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wu32_mul), Q6_V_lo_W(wu32_mul)), vu16_src);
    HVX_Vector     vu16_result = Q6_Vh_vshuffe_VhVh(Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(wu32_add), 3), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wu32_add), 3));

    return vu16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U16>::value && DENOM == 9>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu16_src)
{
    HVX_VectorPair wu32_mul    = Q6_Wuw_vmpy_VuhRuh(vu16_src, 0xE38FE38F);
    HVX_Vector     vu16_result = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wu32_mul), Q6_V_lo_W(wu32_mul));
                   vu16_result = Q6_Vuh_vlsr_VuhR(vu16_result, 3);

    return vu16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U16>::value && DENOM == 25>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu16_src)
{
    HVX_VectorPair wu32_mul    = Q6_Wuw_vmpy_VuhRuh(vu16_src, 0x47AF47AF);
    HVX_VectorPair wu32_add    = Q6_Ww_vadd_VuhVuh(Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wu32_mul), Q6_V_lo_W(wu32_mul)), vu16_src);
    HVX_Vector     vu16_result = Q6_Vh_vshuffe_VhVh(Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(wu32_add), 5), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wu32_add), 5));

    return vu16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U16>::value && DENOM == 49>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu16_src)
{
    HVX_VectorPair wu32_mul    = Q6_Wuw_vmpy_VuhRuh(vu16_src, 0x4E5F4E5F);
    HVX_VectorPair wu32_add    = Q6_Ww_vadd_VuhVuh(Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wu32_mul), Q6_V_lo_W(wu32_mul)), vu16_src);
    HVX_Vector     vu16_result = Q6_Vh_vshuffe_VhVh(Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(wu32_add), 6), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wu32_add), 6));

    return vu16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U16>::value &&
                                  (DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                  (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu16_src)
{
    constexpr DT_U16 LENGTH = 32 - __builtin_clz(DENOM - 1);
    constexpr DT_U16 MUL    = (0 == DENOM) ? 0 : ((65536) * ((1 << LENGTH) - DENOM)) / DENOM + 1;
    constexpr DT_U16 SHIFT0 = (LENGTH > 1) ? 1 : LENGTH;
    constexpr DT_U16 SHIFT1 = (0 == DENOM) ? 15 : (((DT_S32)LENGTH > 1) ? LENGTH - 1 : 0);

    const static HVX_Vector vu16_mul = Q6_Vh_vsplat_R(MUL);

    HVX_Vector vu16_result, vu16_shu, vu16_sub;
    HVX_VectorPair wu32_mul;

    wu32_mul    = Q6_Wuw_vmpy_VuhVuh(vu16_src, vu16_mul);
    vu16_shu    = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wu32_mul), Q6_V_lo_W(wu32_mul));
    vu16_sub    = Q6_Vuh_vsub_VuhVuh_sat(vu16_src, vu16_shu);
    vu16_sub    = Q6_Vuh_vlsr_VuhR(vu16_sub, SHIFT0);
    vu16_result = Q6_Vuh_vadd_VuhVuh_sat(vu16_sub, vu16_shu);
    vu16_result = Q6_Vuh_vlsr_VuhR(vu16_result, SHIFT1);

    return vu16_result;
}

/************************************ s16 vdvin ************************************/
template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S16>::value && DENOM == 3>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs16_src)
{
    HVX_VectorPair ws32_mul     = Q6_Ww_vmpy_VhVuh(vs16_src, Q6_Vh_vsplat_R(43691));
    HVX_VectorPred qs32_pred_lo = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_lo_W(ws32_mul));
    HVX_VectorPred qs32_pred_hi = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_hi_W(ws32_mul));
    HVX_Vector     vs32_add_lo  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_mul), Q6_V_vand_QR(qs32_pred_lo, 0x1FFFF));
    HVX_Vector     vs32_add_hi  = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_mul), Q6_V_vand_QR(qs32_pred_hi, 0x1FFFF));
    HVX_Vector     vs16_result  = Q6_Vh_vshuffe_VhVh(Q6_Vw_vasr_VwR(vs32_add_hi, 17), Q6_Vw_vasr_VwR(vs32_add_lo, 17));

    return vs16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S16>::value && DENOM == 5>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs16_src)
{
    HVX_VectorPair ws32_mul     = Q6_Ww_vmpy_VhVuh(vs16_src, Q6_Vh_vsplat_R(52429));
    HVX_VectorPred qs32_pred_lo = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_lo_W(ws32_mul));
    HVX_VectorPred qs32_pred_hi = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_hi_W(ws32_mul));
    HVX_Vector     vs32_add_lo  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_mul), Q6_V_vand_QR(qs32_pred_lo, 0x3FFFF));
    HVX_Vector     vs32_add_hi  = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_mul), Q6_V_vand_QR(qs32_pred_hi, 0x3FFFF));
    HVX_Vector     vs16_result  = Q6_Vh_vshuffe_VhVh(Q6_Vw_vasr_VwR(vs32_add_hi, 18), Q6_Vw_vasr_VwR(vs32_add_lo, 18));

    return vs16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S16>::value && DENOM == 7>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs16_src)
{
    HVX_VectorPair ws32_mul     = Q6_Ww_vmpy_VhVuh(vs16_src, Q6_Vh_vsplat_R(18725));
    HVX_VectorPred qs32_pred_lo = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_lo_W(ws32_mul));
    HVX_VectorPred qs32_pred_hi = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_hi_W(ws32_mul));
    HVX_Vector     vs32_add_lo  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_mul), Q6_V_vand_QR(qs32_pred_lo, 0x1FFFF));
    HVX_Vector     vs32_add_hi  = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_mul), Q6_V_vand_QR(qs32_pred_hi, 0x1FFFF));
    HVX_Vector     vs16_result  = Q6_Vh_vshuffe_VhVh(Q6_Vw_vasr_VwR(vs32_add_hi, 17), Q6_Vw_vasr_VwR(vs32_add_lo, 17));

    return vs16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S16>::value && DENOM == 9>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs16_src)
{
    HVX_VectorPair ws32_mul     = Q6_Ww_vmpy_VhVuh(vs16_src, Q6_Vh_vsplat_R(58255));
    HVX_VectorPred qs32_pred_lo = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_lo_W(ws32_mul));
    HVX_VectorPred qs32_pred_hi = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_hi_W(ws32_mul));
    HVX_Vector     vs32_add_lo  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_mul), Q6_V_vand_QR(qs32_pred_lo, 0x7FFFF));
    HVX_Vector     vs32_add_hi  = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_mul), Q6_V_vand_QR(qs32_pred_hi, 0x7FFFF));
    HVX_Vector     vs16_result  = Q6_Vh_vshuffe_VhVh(Q6_Vw_vasr_VwR(vs32_add_hi, 19), Q6_Vw_vasr_VwR(vs32_add_lo, 19));

    return vs16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S16>::value && DENOM == 25>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs16_src)
{
    HVX_VectorPair ws32_mul     = Q6_Ww_vmpy_VhVuh(vs16_src, Q6_Vh_vsplat_R(20972));
    HVX_VectorPred qs32_pred_lo = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_lo_W(ws32_mul));
    HVX_VectorPred qs32_pred_hi = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_hi_W(ws32_mul));
    HVX_Vector     vs32_add_lo  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_mul), Q6_V_vand_QR(qs32_pred_lo, 0x7FFFF));
    HVX_Vector     vs32_add_hi  = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_mul), Q6_V_vand_QR(qs32_pred_hi, 0x7FFFF));
    HVX_Vector     vs16_result  = Q6_Vh_vshuffe_VhVh(Q6_Vw_vasr_VwR(vs32_add_hi, 19), Q6_Vw_vasr_VwR(vs32_add_lo, 19));

    return vs16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S16>::value && DENOM == 49>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs16_src)
{
    HVX_VectorPair ws32_mul     = Q6_Ww_vmpy_VhVuh(vs16_src, Q6_Vh_vsplat_R(2675));
    HVX_VectorPred qs32_pred_lo = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_lo_W(ws32_mul));
    HVX_VectorPred qs32_pred_hi = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), Q6_V_hi_W(ws32_mul));
    HVX_Vector     vs32_add_lo  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_mul), Q6_V_vand_QR(qs32_pred_lo, 0x1FFFF));
    HVX_Vector     vs32_add_hi  = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_mul), Q6_V_vand_QR(qs32_pred_hi, 0x1FFFF));
    HVX_Vector     vs16_result  = Q6_Vh_vshuffe_VhVh(Q6_Vw_vasr_VwR(vs32_add_hi, 17), Q6_Vw_vasr_VwR(vs32_add_lo, 17));

    return vs16_result;
}

template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S16>::value &&
                                  (DENOM != 3) && (DENOM != 5)  && (DENOM != 7) &&
                                  (DENOM != 9) && (DENOM != 25) && (DENOM != 49)>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs16_src)
{
    constexpr DT_U16 ABS_D  = (DENOM >= 0) ? DENOM : -DENOM;
    constexpr DT_S16 BITS   = static_cast<DT_S16>(32 - __builtin_clz(ABS_D - 1));
    constexpr DT_S16 LENGTH = (BITS > 1) ? BITS : 1;
    constexpr DT_S32 MUL    = (0 == DENOM) ? 0 : 1 + ((DT_U32(1)) << (16 + LENGTH - 1)) / ABS_D - (65536);
    constexpr DT_S16 SHIFT  = (0 == DENOM) ? 15 : (LENGTH - 1);

    const static HVX_Vector vs16_mul  = Q6_Vh_vsplat_R(MUL);
    const static HVX_Vector vs16_sign = Q6_Vh_vsplat_R(DENOM >> 15);
    const static HVX_Vector vshift    = Q6_Vh_vsplat_R(SHIFT);

    HVX_Vector vs16_result;
    HVX_VectorPair ws32_mul;
    HVX_Vector vs16_mul_hi, vs16_add;

    ws32_mul    = Q6_Ww_vmpy_VhVh(vs16_src, vs16_mul);
    vs16_mul_hi = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(ws32_mul), Q6_V_lo_W(ws32_mul));
    vs16_add    = Q6_Vh_vadd_VhVh(vs16_src, vs16_mul_hi);
    vs16_add    = Q6_Vh_vsub_VhVh(Q6_Vh_vasr_VhVh(vs16_add, vshift), Q6_Vh_vasr_VhR(vs16_src, 15));
    vs16_result = Q6_Vh_vsub_VhVh(Q6_V_vxor_VV(vs16_add, vs16_sign), vs16_sign);

    return vs16_result;
}

/************************************ u32 vdvin ************************************/
template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_U32>::value>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vu32_src)
{
    constexpr DT_U32 LENGTH = 32 - __builtin_clz(DENOM - 1);
    constexpr DT_U32 MUL    = (0 == DENOM) ? 0 : ((DT_U64)(4294967296) * (((DT_U64)1 << LENGTH) - (DT_U64)DENOM) / DENOM + 1);
    constexpr DT_U32 MUL_R  = (MUL << 16) | (MUL >> 16);
    constexpr DT_U32 SHIFT0 = (LENGTH > 1) ? 1 : LENGTH;
    constexpr DT_U32 SHIFT1 = (0 == DENOM) ? 31 : (((DT_S32)LENGTH > 1) ? ((DT_S32)LENGTH - 1) : 0);

    HVX_Vector vu32_result;
    HVX_VectorPair vu32_albh_ahbl, vu32_albl_ahbh;
    HVX_Vector vu32_add, vu32_sub;

    // div  v_mul_xy = v_mul_xy / max_ro
    vu32_albh_ahbl = Q6_Wuw_vmpy_VuhRuh(vu32_src, MUL_R); // al_bh, ah_bl
    vu32_albl_ahbh = Q6_Wuw_vmpy_VuhRuh(vu32_src, MUL);   // al_bl, ah_bh

    // bh and ah at least one is lt (1 << 15), so al_bh + ah_bl will not overflow
    vu32_add = Q6_Vuw_vadd_VuwVuw_sat(Q6_V_lo_W(vu32_albh_ahbl), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(vu32_albl_ahbh), 16));
    vu32_add = Q6_Vuw_vlsr_VuwR(Q6_Vuw_vavg_VuwVuw(vu32_add, Q6_V_hi_W(vu32_albh_ahbl)), 15);
    vu32_add = Q6_Vuw_vadd_VuwVuw_sat(vu32_add, Q6_V_hi_W(vu32_albl_ahbh));

    vu32_sub    = Q6_Vuw_vsub_VuwVuw_sat(vu32_src, vu32_add);
    vu32_sub    = Q6_Vuw_vlsr_VuwR(vu32_sub, SHIFT0);
    vu32_result = Q6_Vuw_vadd_VuwVuw_sat(vu32_sub, vu32_add);
    vu32_result = Q6_Vuw_vlsr_VuwR(vu32_result, SHIFT1);

    return vu32_result;
}

/************************************ s32 vdvin ************************************/
template <typename Tp, Tp DENOM,
          typename std::enable_if<std::is_same<Tp, DT_S32>::value>::type* = DT_NULL>
inline HVX_Vector vdiv_n(const HVX_Vector &vs32_src)
{
    constexpr DT_U32 ABS_D  = (DENOM >= 0) ? DENOM : -DENOM;
    constexpr DT_S32 BITS   = 32 - __builtin_clz(ABS_D - 1);
    constexpr DT_S32 LENGTH = (BITS > 1) ? BITS : 1;
    constexpr DT_S64 MUL    = (0 == ABS_D) ? 0 : ((1 + ((DT_U64(1)) << (32 + LENGTH - 1)) / ABS_D) - (4294967296));
    constexpr DT_S32 SHIFT  = (0 == DENOM) ? 31 : (LENGTH - 1);

    const static HVX_Vector vs32_mul   = Q6_V_vsplat_R(MUL);
    const static HVX_Vector vs32_sign  = Q6_V_vsplat_R(DENOM >> (31));
    const static HVX_Vector vs32_shift = Q6_V_vsplat_R(SHIFT);

    HVX_Vector vs32_result, vs32_result0, vs32_mul_hi;
    HVX_VectorPair ws32_mul;

    ws32_mul     = Q6_Wd_vmul_VwVw(vs32_src, vs32_mul);
    vs32_mul_hi  = Q6_V_hi_W(ws32_mul);
    vs32_result0 = Q6_Vw_vadd_VwVw(vs32_src, vs32_mul_hi);
    vs32_result0 = Q6_Vw_vsub_VwVw(Q6_Vw_vasr_VwVw(vs32_result0, vs32_shift), Q6_Vw_vasr_VwR(vs32_src, 31));
    vs32_result  = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(vs32_result0, vs32_sign), vs32_sign);

    return vs32_result;
}

template <typename Tp, Tp DENOM>
static HVX_VectorPair vdiv_n(const HVX_VectorPair &w_src)
{
    HVX_Vector v_result0 = vdiv_n<Tp, DENOM>(Q6_V_lo_W(w_src));
    HVX_Vector v_result1 = vdiv_n<Tp, DENOM>(Q6_V_hi_W(w_src));
    return Q6_W_vcombine_VV(v_result1, v_result0);
}

/************************************ common vdvin ************************************/
#define W_VDIVN_COMM                                              \
    HVX_VectorPair operator()(const HVX_VectorPair &w_src)        \
    {                                                             \
        HVX_Vector v_result0 = operator()(Q6_V_lo_W(w_src));      \
        HVX_Vector v_result1 = operator()(Q6_V_hi_W(w_src));      \
        return Q6_W_vcombine_VV(v_result1, v_result0);            \
    }

template <typename Tp> class HvxVdivnHelper;

template <>
class HvxVdivnHelper<DT_U8>
{
public:
    explicit HvxVdivnHelper(DT_U8 denom) : m_denom(denom)
    {
        DT_U8 length = 32 - __builtin_clz(m_denom - 1);
        DT_U8 mul    = (0 == m_denom) ? 0 : (((DT_U16)256) * ((1 << length) - m_denom)) / m_denom + 1;
        m_vu8_mul    = Q6_Vb_vsplat_R(mul);
        m_shift0     = Min<DT_U8>(length, (DT_U8)1);
        m_shift1     = Max<DT_S8>((DT_S8)length - 1, (DT_S8)0);
        m_shift1     = (0 == m_denom) ? 7 : m_shift1;
    }

    HVX_Vector operator()(const HVX_Vector &vu8_src)
    {
        HVX_Vector vu8_result;
        HVX_VectorPair wu16_mul;
        HVX_Vector vu8_mul_hi, vu8_sub;

        wu16_mul   = Q6_Wuh_vmpy_VubVub(vu8_src, m_vu8_mul);
        vu8_mul_hi = Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(wu16_mul), Q6_V_lo_W(wu16_mul));
        vu8_sub    = Q6_Vub_vsub_VubVub_sat(vu8_src, vu8_mul_hi);
        vu8_sub    = Q6_Vub_vlsr_VubR(vu8_sub, m_shift0);
        vu8_result = Q6_Vub_vadd_VubVub_sat(vu8_sub, vu8_mul_hi);
        vu8_result = Q6_Vub_vlsr_VubR(vu8_result, m_shift1);

        return vu8_result;
    }

    W_VDIVN_COMM;

private:
    DT_U8      m_denom;
    DT_U8      m_shift0;
    DT_U8      m_shift1;
    HVX_Vector m_vu8_mul;
};

template <>
class HvxVdivnHelper<DT_S8>
{
public:
    explicit HvxVdivnHelper(DT_S8 denom) : m_denom(denom)
    {
        DT_U8 abs_d  = Abs(m_denom);
        DT_S8 length = 32 - __builtin_clz(abs_d - 1);
              length = Max<DT_U8>(length, (DT_S8)1);

        DT_S16 mul = (0 == m_denom) ? 0 : (1 + ((DT_U16(1)) << (8 + length - 1)) / abs_d) - (256);
        m_vs8_mul  = Q6_Vb_vsplat_R(mul);
        m_vs8_sign = Q6_Vb_vsplat_R(m_denom >> (7));
        m_shift    = (0 == m_denom) ? 7 : length - 1;
    }

    HVX_Vector operator()(const HVX_Vector &vs8_src)
    {
        HVX_Vector vs8_result;
        HVX_VectorPair ws16_mul, wu16_add, wu16_input;
        HVX_Vector vs8_add, vs8_sub0, vs8_sub1;

        ws16_mul   = Q6_Wh_vmpy_VbVb(vs8_src, m_vs8_mul);
        vs8_add    = Q6_Vb_vadd_VbVb(vs8_src, Q6_Vb_vshuffo_VbVb(Q6_V_hi_W(ws16_mul), Q6_V_lo_W(ws16_mul)));
        wu16_add   = Q6_Wh_vunpack_Vb(vs8_add);
        wu16_input = Q6_Wh_vunpack_Vb(vs8_src);
        vs8_sub0   = Q6_Vb_vpack_VhVh_sat(Q6_Vh_vasr_VhR(Q6_V_hi_W(wu16_add), m_shift), Q6_Vh_vasr_VhR(Q6_V_lo_W(wu16_add), m_shift));
        vs8_sub1   = Q6_Vb_vpack_VhVh_sat(Q6_Vh_vasr_VhR(Q6_V_hi_W(wu16_input), 7), Q6_Vh_vasr_VhR(Q6_V_lo_W(wu16_input), 7));
        vs8_add    = Q6_Vb_vsub_VbVb(vs8_sub0, vs8_sub1);
        vs8_result = Q6_Vb_vsub_VbVb(Q6_V_vxor_VV(vs8_add, m_vs8_sign), m_vs8_sign);

        return vs8_result;
    }

    W_VDIVN_COMM;

private:
    DT_S8      m_denom;
    DT_S8      m_shift;
    HVX_Vector m_vs8_mul;
    HVX_Vector m_vs8_sign;
};

template <>
class HvxVdivnHelper<DT_U16>
{
public:
    explicit HvxVdivnHelper(DT_U16 denom) : m_denom(denom)
    {
        DT_U16 length = 32 - __builtin_clz(m_denom - 1);
        DT_U16 mul    = (0 == m_denom) ? 0 : ((65536) * ((1 << length) - m_denom)) / m_denom + 1;
        m_vu16_mul    = Q6_Vh_vsplat_R(mul);
        m_shift0      = Min<DT_U16>(length, (DT_U16)1);
        m_shift1      = Max<DT_S16>((DT_S16)length - 1, (DT_S16)0);
        m_shift1      = (0 == m_denom) ? 15 : m_shift1;
    }

    HVX_Vector operator()(const HVX_Vector &vu16_src)
    {
        HVX_Vector vu16_result, vu16_mul_hi, vu16_sub;
        HVX_VectorPair wu32_mul;

        wu32_mul    = Q6_Wuw_vmpy_VuhVuh(vu16_src, m_vu16_mul);
        vu16_mul_hi = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(wu32_mul), Q6_V_lo_W(wu32_mul));
        vu16_sub    = Q6_Vuh_vsub_VuhVuh_sat(vu16_src, vu16_mul_hi);
        vu16_sub    = Q6_Vuh_vlsr_VuhR(vu16_sub, m_shift0);
        vu16_result = Q6_Vuh_vadd_VuhVuh_sat(vu16_sub, vu16_mul_hi);
        vu16_result = Q6_Vuh_vlsr_VuhR(vu16_result, m_shift1);

        return vu16_result;
    }

    W_VDIVN_COMM;

private:
    DT_U16     m_denom;
    DT_U16     m_shift0;
    DT_U16     m_shift1;
    HVX_Vector m_vu16_mul;
};

template <>
class HvxVdivnHelper<DT_S16>
{
public:
    explicit HvxVdivnHelper(DT_S16 denom) : m_denom(denom)
    {
        DT_U16 abs_d  = Abs(m_denom);
        DT_S16 length = 32 - __builtin_clz(abs_d - 1);
               length = Max<DT_U16>(length, (DT_S16)1);

        DT_S32 MUL   = (0 == m_denom) ? 0 : (1 + ((DT_U32(1)) << (16 + length - 1)) / abs_d) - (65536);
        DT_S16 SHIFT = (0 == m_denom) ? 15 : (length - 1);
        m_vs16_mul   = Q6_Vh_vsplat_R(MUL);
        m_vs16_sign  = Q6_Vh_vsplat_R(m_denom >> (15));
        m_shift      = Q6_Vh_vsplat_R(SHIFT);
    }

    HVX_Vector operator()(const HVX_Vector &vs16_src)
    {
        HVX_Vector vs16_result;
        HVX_VectorPair ws32_mul;
        HVX_Vector vs16_mul_hi, vs16_add;

        ws32_mul    = Q6_Ww_vmpy_VhVh(vs16_src, m_vs16_mul);
        vs16_mul_hi = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(ws32_mul), Q6_V_lo_W(ws32_mul));
        vs16_add    = Q6_Vh_vadd_VhVh(vs16_src, vs16_mul_hi);
        vs16_add    = Q6_Vh_vsub_VhVh(Q6_Vh_vasr_VhVh(vs16_add, m_shift), Q6_Vh_vasr_VhR(vs16_src, 15));
        vs16_result = Q6_Vh_vsub_VhVh(Q6_V_vxor_VV(vs16_add, m_vs16_sign), m_vs16_sign);

        return vs16_result;
    }

    W_VDIVN_COMM;

private:
    DT_S16     m_denom;
    HVX_Vector m_vs16_mul;
    HVX_Vector m_vs16_sign;
    HVX_Vector m_shift;
};

template <>
class HvxVdivnHelper<DT_U32>
{
public:
    explicit HvxVdivnHelper(DT_U32 denom) : m_denom(denom)
    {
        DT_U32 length = 32 - __builtin_clz(m_denom - 1);
        m_mul = (0 == m_denom) ? 0 : ((DT_U64)(4294967296) * (((DT_U64)1 << length) - (DT_U64)m_denom) / m_denom + 1);

        m_mul_r  = (m_mul << 16) | (m_mul >> 16);
        m_shift0 = Min<DT_U32>(length, (DT_U32)1);
        m_shift1 = (0 == m_denom) ? 31 : (Max<DT_S32>((DT_S32)length - 1, (DT_S32)0));
    }

    HVX_Vector operator()(const HVX_Vector &vu32_src)
    {
        HVX_Vector vu32_result;
        HVX_Vector vu32_add, vu32_sub;

        // div  v_mul_xy = v_mul_xy / max_ro
        HVX_VectorPair wu32_albh_ahbl = Q6_Wuw_vmpy_VuhRuh(vu32_src, m_mul_r); // al_bh, ah_bl
        HVX_VectorPair wu32_albl_ahbh = Q6_Wuw_vmpy_VuhRuh(vu32_src, m_mul);   // al_bl, ah_bh

        // bh and ah at least one is lt (1 << 15), so al_bh + ah_bl will not overflow
        vu32_add = Q6_Vuw_vadd_VuwVuw_sat(Q6_V_lo_W(wu32_albh_ahbl), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wu32_albl_ahbh), 16));
        vu32_add = Q6_Vuw_vlsr_VuwR(Q6_Vuw_vavg_VuwVuw(vu32_add, Q6_V_hi_W(wu32_albh_ahbl)), 15);
        vu32_add = Q6_Vuw_vadd_VuwVuw_sat(vu32_add, Q6_V_hi_W(wu32_albl_ahbh));

        vu32_sub    = Q6_Vuw_vsub_VuwVuw_sat(vu32_src, vu32_add);
        vu32_sub    = Q6_Vuw_vlsr_VuwR(vu32_sub, m_shift0);
        vu32_result = Q6_Vuw_vadd_VuwVuw_sat(vu32_sub, vu32_add);
        vu32_result = Q6_Vuw_vlsr_VuwR(vu32_result, m_shift1);

        return vu32_result;
    }

    W_VDIVN_COMM;

private:
    DT_U32 m_mul;
    DT_U32 m_mul_r;
    DT_U32 m_shift0;
    DT_U32 m_shift1;
    DT_U32 m_denom;
};

template <>
class HvxVdivnHelper<DT_S32>
{
public:
    explicit HvxVdivnHelper(DT_S32 denom) : m_denom(denom)
    {
        DT_U32 abs_d  = Abs(m_denom);
        DT_S32 length = 32 - __builtin_clz(abs_d - 1);
               length = Max<DT_U32>(length, (DT_S32)1);
        DT_S64 mul    = (0 == abs_d) ? 0 : (1 + ((DT_U64(1)) << (32 + length - 1)) / abs_d) - (4294967296);

        m_vs32_mul  = Q6_V_vsplat_R(mul);
        m_vs32_sign = Q6_V_vsplat_R(m_denom >> (31));
        m_shift     = Q6_V_vsplat_R(length - 1);
        m_shift     = (0 == m_denom) ? 31 : m_shift;
    }

    HVX_Vector operator()(const HVX_Vector &vs32_src)
    {
        HVX_Vector vs32_result, vs32_result0, vs32_mul_hi;
        HVX_VectorPair ws32_mul;

        ws32_mul     = Q6_Wd_vmul_VwVw(vs32_src, m_vs32_mul);
        vs32_mul_hi  = Q6_V_hi_W(ws32_mul);
        vs32_result0 = Q6_Vw_vadd_VwVw(vs32_src, vs32_mul_hi);
        vs32_result0 = Q6_Vw_vsub_VwVw(Q6_Vw_vasr_VwVw(vs32_result0, m_shift), Q6_Vw_vasr_VwR(vs32_src, 31));
        vs32_result  = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(vs32_result0, m_vs32_sign), m_vs32_sign);

        return vs32_result;
    }

    W_VDIVN_COMM;

private:
    DT_S32     m_denom;
    HVX_Vector m_vs32_mul;
    HVX_Vector m_vs32_sign;
    HVX_Vector m_shift;
};

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_DIVN_HPP__