#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

static const MI_S32 BETA_S16[10][4] = {
    { 2273, -225,   0,     0},
    { 2195, -147,   0,     0},
    { 1834,  235,  -21,    0},
    { 1310,  873,  -135,   0},
    { 738,   1535, -225,   0},
    { 214,   1981, -147,   0},
    {-147,   1981,  235,  -21 },
    {-225,   1535,  873,  -135},
    {-135,   873,   1535, -225},
    {-21,    235,   1981, -147}
};

static const MI_S32 BETA_S32[10][4] = {
    { 36368, -3600,   0,      0},
    { 35120, -2352,   0,      0},
    { 29344,  3760,  -336,    0},
    { 20960,  13968, -2160,   0},
    { 11808,  24560, -3600,   0},
    { 3424,   31696, -2352,   0},
    {-2352,   31696,  3760,  -336 },
    {-3600,   24560,  13968, -2160},
    {-2160,   13968,  24560, -3600},
    {-336,    3760,   31696, -2352}
};

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX2HCore(HVX_Vector &vu8_src,
                                             HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                             HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1,
                                             HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair wu8_twin = Q6_W_vshuff_VVR(vu8_src, vu8_src, -1);
    HVX_Vector vu8_x0_twin  = Q6_V_lo_W(wu8_twin);
    vu8_src                 = Q6_V_hi_W(wu8_twin);

    HVX_Vector vu8_r0_src      = Q6_V_valign_VVI(vu8_src, vu8_x0_twin, 2);
    HVX_Vector vu8_r1_src      = Q6_V_valign_VVI(vu8_src, vu8_x0_twin, 4);
    HVX_Vector vu8_r2_src      = Q6_V_valign_VVI(vu8_src, vu8_x0_twin, 6);
    HVX_VectorPair wu16_c_src  = Q6_Wuh_vunpack_Vub(vu8_x0_twin);
    HVX_VectorPair wu16_r0_src = Q6_Wuh_vunpack_Vub(vu8_r0_src);
    HVX_VectorPair wu16_r1_src = Q6_Wuh_vunpack_Vub(vu8_r1_src);
    HVX_VectorPair wu16_r2_src = Q6_Wuh_vunpack_Vub(vu8_r2_src);

    ws32_result_l = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_lo_W(wu16_c_src));
    ws32_result_h = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_hi_W(wu16_c_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(wu16_r0_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(wu16_r0_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(wu16_r1_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(wu16_r1_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(wu16_r2_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(wu16_r2_src));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX2HCore(HVX_Vector &vs8_src,
                                             HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                             HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1,
                                             HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair ws8_twin = Q6_W_vshuff_VVR(vs8_src, vs8_src, -1);
    HVX_Vector vs8_x0_twin  = Q6_V_lo_W(ws8_twin);
    vs8_src                 = Q6_V_hi_W(ws8_twin);

    HVX_Vector vs8_r0_src      = Q6_V_valign_VVI(vs8_src, vs8_x0_twin, 2);
    HVX_Vector vs8_r1_src      = Q6_V_valign_VVI(vs8_src, vs8_x0_twin, 4);
    HVX_Vector vs8_r2_src      = Q6_V_valign_VVI(vs8_src, vs8_x0_twin, 6);
    HVX_VectorPair ws16_c_src  = Q6_Wh_vunpack_Vb(vs8_x0_twin);
    HVX_VectorPair ws16_r0_src = Q6_Wh_vunpack_Vb(vs8_r0_src);
    HVX_VectorPair ws16_r1_src = Q6_Wh_vunpack_Vb(vs8_r1_src);
    HVX_VectorPair ws16_r2_src = Q6_Wh_vunpack_Vb(vs8_r2_src);

    ws32_result_l = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_lo_W(ws16_c_src));
    ws32_result_h = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_hi_W(ws16_c_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(ws16_r0_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(ws16_r0_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(ws16_r1_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(ws16_r1_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(ws16_r2_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(ws16_r2_src));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX2HCore(HVX_Vector &vu16_src, HVX_VectorPair &ws32_result,
                                             HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1,
                                             HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair wu16_twin = Q6_W_vshuff_VVR(vu16_src, vu16_src, -2);
    HVX_Vector vu16_x0_twin  = Q6_V_lo_W(wu16_twin);
    vu16_src                 = Q6_V_hi_W(wu16_twin);

    HVX_Vector vu16_r0_src = Q6_V_valign_VVI(vu16_src, vu16_x0_twin, 4);
    HVX_Vector vu16_r1_src = Q6_V_valign_VVR(vu16_src, vu16_x0_twin, 8);
    HVX_Vector vu16_r2_src = Q6_V_valign_VVR(vu16_src, vu16_x0_twin, 12);

    ws32_result = Q6_Ww_vmpy_VhVuh(vs16_alpha0, vu16_x0_twin);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha1, vu16_r0_src);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha2, vu16_r1_src);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha3, vu16_r2_src);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX2HCore(HVX_Vector &vs16_src, HVX_VectorPair &ws32_result,
                                             HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1,
                                             HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair ws16_twin = Q6_W_vshuff_VVR(vs16_src, vs16_src, -2);
    HVX_Vector vs16_x0_twin  = Q6_V_lo_W(ws16_twin);
    vs16_src                 = Q6_V_hi_W(ws16_twin);

    HVX_Vector vs16_r0_src = Q6_V_valign_VVI(vs16_src, vs16_x0_twin, 4);
    HVX_Vector vs16_r1_src = Q6_V_valign_VVR(vs16_src, vs16_x0_twin, 8);
    HVX_Vector vs16_r2_src = Q6_V_valign_VVR(vs16_src, vs16_x0_twin, 12);

    ws32_result = Q6_Ww_vmpy_VhVh(vs16_alpha0, vs16_x0_twin);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha1, vs16_r0_src);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha2, vs16_r1_src);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha3, vs16_r2_src);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX2HCore(HVX_Vector &vu8_x0_twin, HVX_Vector &vu8_x1_src,
                                             HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                             HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1,
                                             HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair wu8_twin = Q6_W_vshuff_VVR(vu8_x1_src, vu8_x1_src, -1);
    HVX_Vector vu8_x1_twin  = Q6_V_lo_W(wu8_twin);

    HVX_Vector vu8_r0_src      = Q6_V_valign_VVI(vu8_x1_twin, vu8_x0_twin, 2);
    HVX_Vector vu8_r1_src      = Q6_V_valign_VVI(vu8_x1_twin, vu8_x0_twin, 4);
    HVX_Vector vu8_r2_src      = Q6_V_valign_VVI(vu8_x1_twin, vu8_x0_twin, 6);
    HVX_VectorPair wu16_c_src  = Q6_Wuh_vunpack_Vub(vu8_x0_twin);
    HVX_VectorPair wu16_r0_src = Q6_Wuh_vunpack_Vub(vu8_r0_src);
    HVX_VectorPair wu16_r1_src = Q6_Wuh_vunpack_Vub(vu8_r1_src);
    HVX_VectorPair wu16_r2_src = Q6_Wuh_vunpack_Vub(vu8_r2_src);

    ws32_result_l = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_lo_W(wu16_c_src));
    ws32_result_h = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_hi_W(wu16_c_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(wu16_r0_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(wu16_r0_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(wu16_r1_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(wu16_r1_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(wu16_r2_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(wu16_r2_src));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX2HCore(HVX_Vector &vs8_x0_twin, HVX_Vector &vs8_x1_src,
                                             HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                             HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1,
                                             HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair ws8_twin = Q6_W_vshuff_VVR(vs8_x1_src, vs8_x1_src, -1);
    HVX_Vector vs8_x1_twin  = Q6_V_lo_W(ws8_twin);

    HVX_Vector vs8_r0_src      = Q6_V_valign_VVI(vs8_x1_twin, vs8_x0_twin, 2);
    HVX_Vector vs8_r1_src      = Q6_V_valign_VVI(vs8_x1_twin, vs8_x0_twin, 4);
    HVX_Vector vs8_r2_src      = Q6_V_valign_VVI(vs8_x1_twin, vs8_x0_twin, 6);
    HVX_VectorPair ws16_c_src  = Q6_Wh_vunpack_Vb(vs8_x0_twin);
    HVX_VectorPair ws16_r0_src = Q6_Wh_vunpack_Vb(vs8_r0_src);
    HVX_VectorPair ws16_r1_src = Q6_Wh_vunpack_Vb(vs8_r1_src);
    HVX_VectorPair ws16_r2_src = Q6_Wh_vunpack_Vb(vs8_r2_src);

    ws32_result_l = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_lo_W(ws16_c_src));
    ws32_result_h = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_hi_W(ws16_c_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(ws16_r0_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(ws16_r0_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(ws16_r1_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(ws16_r1_src));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(ws16_r2_src));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(ws16_r2_src));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX2HCore(HVX_Vector &vu16_x0_twin, HVX_Vector &vu16_x1_src,
                                             HVX_VectorPair &ws32_result,
                                             HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1,
                                             HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair wu16_twin = Q6_W_vshuff_VVR(vu16_x1_src, vu16_x1_src, -2);
    HVX_Vector vu16_x1_twin  = Q6_V_lo_W(wu16_twin);

    HVX_Vector vu16_r0_src = Q6_V_valign_VVI(vu16_x1_twin, vu16_x0_twin, 4);
    HVX_Vector vu16_r1_src = Q6_V_valign_VVR(vu16_x1_twin, vu16_x0_twin, 8);
    HVX_Vector vu16_r2_src = Q6_V_valign_VVR(vu16_x1_twin, vu16_x0_twin, 12);

    ws32_result = Q6_Ww_vmpy_VhVuh(vs16_alpha0, vu16_x0_twin);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha1, vu16_r0_src);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha2, vu16_r1_src);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha3, vu16_r2_src);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX2HCore(HVX_Vector &vs16_x0_twin, HVX_Vector &vs16_x1_src,
                                             HVX_VectorPair &ws32_result,
                                             HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1,
                                             HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair ws16_twin = Q6_W_vshuff_VVR(vs16_x1_src, vs16_x1_src, -2);
    HVX_Vector vs16_x1_twin  = Q6_V_lo_W(ws16_twin);

    HVX_Vector vs16_r0_src = Q6_V_valign_VVI(vs16_x1_twin, vs16_x0_twin, 4);
    HVX_Vector vs16_r1_src = Q6_V_valign_VVR(vs16_x1_twin, vs16_x0_twin, 8);
    HVX_Vector vs16_r2_src = Q6_V_valign_VVR(vs16_x1_twin, vs16_x0_twin, 12);

    ws32_result = Q6_Ww_vmpy_VhVh(vs16_alpha0, vs16_x0_twin);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha1, vs16_r0_src);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha2, vs16_r1_src);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha3, vs16_r2_src);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H0Core(HVX_Vector &vu8_twin_src,
                                              HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair wu8_quad = Q6_W_vshuff_VVR(vu8_twin_src, vu8_twin_src, -2);
    HVX_Vector vu8_x0_quad  = Q6_V_lo_W(wu8_quad);
    vu8_twin_src            = Q6_V_hi_W(wu8_quad);

    HVX_Vector vu8_r0_quad      = Q6_V_valign_VVR(vu8_twin_src, vu8_x0_quad, 4);
    HVX_Vector vu8_r1_quad      = Q6_V_valign_VVR(vu8_twin_src, vu8_x0_quad, 8);
    HVX_Vector vu8_r2_quad      = Q6_V_valign_VVR(vu8_twin_src, vu8_x0_quad, 12);
    HVX_VectorPair wu16_c_quad  = Q6_Wuh_vunpack_Vub(vu8_x0_quad);
    HVX_VectorPair wu16_r0_quad = Q6_Wuh_vunpack_Vub(vu8_r0_quad);
    HVX_VectorPair wu16_r1_quad = Q6_Wuh_vunpack_Vub(vu8_r1_quad);
    HVX_VectorPair wu16_r2_quad = Q6_Wuh_vunpack_Vub(vu8_r2_quad);

    ws32_result_l = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_lo_W(wu16_c_quad));
    ws32_result_h = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_hi_W(wu16_c_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(wu16_r0_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(wu16_r0_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(wu16_r1_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(wu16_r1_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(wu16_r2_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(wu16_r2_quad));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H0Core(HVX_Vector &vs8_twin_src,
                                              HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair ws8_quad = Q6_W_vshuff_VVR(vs8_twin_src, vs8_twin_src, -2);
    HVX_Vector vs8_x0_quad  = Q6_V_lo_W(ws8_quad);
    vs8_twin_src            = Q6_V_hi_W(ws8_quad);

    HVX_Vector vs8_r0_quad      = Q6_V_valign_VVR(vs8_twin_src, vs8_x0_quad, 4);
    HVX_Vector vs8_r1_quad      = Q6_V_valign_VVR(vs8_twin_src, vs8_x0_quad, 8);
    HVX_Vector vs8_r2_quad      = Q6_V_valign_VVR(vs8_twin_src, vs8_x0_quad, 12);
    HVX_VectorPair ws16_c_quad  = Q6_Wh_vunpack_Vb(vs8_x0_quad);
    HVX_VectorPair ws16_r0_quad = Q6_Wh_vunpack_Vb(vs8_r0_quad);
    HVX_VectorPair ws16_r1_quad = Q6_Wh_vunpack_Vb(vs8_r1_quad);
    HVX_VectorPair ws16_r2_quad = Q6_Wh_vunpack_Vb(vs8_r2_quad);

    ws32_result_l = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_lo_W(ws16_c_quad));
    ws32_result_h = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_hi_W(ws16_c_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(ws16_r0_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(ws16_r0_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(ws16_r1_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(ws16_r1_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(ws16_r2_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(ws16_r2_quad));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H0Core(HVX_Vector &vu16_twin_src, HVX_VectorPair &ws32_result,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair wu16_quad = Q6_W_vshuff_VVR(vu16_twin_src, vu16_twin_src, -4);
    HVX_Vector vu16_x0_quad  = Q6_V_lo_W(wu16_quad);
    vu16_twin_src            = Q6_V_hi_W(wu16_quad);

    HVX_Vector vu16_r0_quad = Q6_V_valign_VVR(vu16_twin_src, vu16_x0_quad, 8);
    HVX_Vector vu16_r1_quad = Q6_V_valign_VVR(vu16_twin_src, vu16_x0_quad, 16);
    HVX_Vector vu16_r2_quad = Q6_V_valign_VVR(vu16_twin_src, vu16_x0_quad, 24);

    ws32_result = Q6_Ww_vmpy_VhVuh(vs16_alpha0, vu16_x0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha1, vu16_r0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha2, vu16_r1_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha3, vu16_r2_quad);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H0Core(HVX_Vector &vs16_twin_src, HVX_VectorPair &ws32_result,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair ws16_quad = Q6_W_vshuff_VVR(vs16_twin_src, vs16_twin_src, -4);
    HVX_Vector vs16_x0_quad  = Q6_V_lo_W(ws16_quad);
    vs16_twin_src            = Q6_V_hi_W(ws16_quad);

    HVX_Vector vs16_r0_quad = Q6_V_valign_VVR(vs16_twin_src, vs16_x0_quad, 8);
    HVX_Vector vs16_r1_quad = Q6_V_valign_VVR(vs16_twin_src, vs16_x0_quad, 16);
    HVX_Vector vs16_r2_quad = Q6_V_valign_VVR(vs16_twin_src, vs16_x0_quad, 24);

    ws32_result = Q6_Ww_vmpy_VhVh(vs16_alpha0, vs16_x0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha1, vs16_r0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha2, vs16_r1_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha3, vs16_r2_quad);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H1Core(HVX_Vector &vu8_x0_quad, HVX_Vector &vu8_x1_twin,
                                              HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair wu8_quad = Q6_W_vshuff_VVR(vu8_x1_twin, vu8_x1_twin, -2);
    HVX_Vector vu8_x1_quad  = Q6_V_lo_W(wu8_quad);
    vu8_x1_twin             = Q6_V_hi_W(wu8_quad);

    HVX_Vector vu8_r0_quad      = Q6_V_valign_VVR(vu8_x1_quad, vu8_x0_quad, 4);
    HVX_Vector vu8_r1_quad      = Q6_V_valign_VVR(vu8_x1_quad, vu8_x0_quad, 8);
    HVX_Vector vu8_r2_quad      = Q6_V_valign_VVR(vu8_x1_quad, vu8_x0_quad, 12);
    HVX_VectorPair wu16_c_quad  = Q6_Wuh_vunpack_Vub(vu8_x0_quad);
    HVX_VectorPair wu16_r0_quad = Q6_Wuh_vunpack_Vub(vu8_r0_quad);
    HVX_VectorPair wu16_r1_quad = Q6_Wuh_vunpack_Vub(vu8_r1_quad);
    HVX_VectorPair wu16_r2_quad = Q6_Wuh_vunpack_Vub(vu8_r2_quad);

    ws32_result_l = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_lo_W(wu16_c_quad));
    ws32_result_h = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_hi_W(wu16_c_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(wu16_r0_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(wu16_r0_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(wu16_r1_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(wu16_r1_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(wu16_r2_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(wu16_r2_quad));

    vu8_x0_quad = vu8_x1_quad;
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H1Core(HVX_Vector &vs8_x0_quad, HVX_Vector &vs8_x1_twin,
                                              HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair ws8_quad = Q6_W_vshuff_VVR(vs8_x1_twin, vs8_x1_twin, -2);
    HVX_Vector vs8_x1_quad  = Q6_V_lo_W(ws8_quad);
    vs8_x1_twin             = Q6_V_hi_W(ws8_quad);

    HVX_Vector vs8_r0_quad      = Q6_V_valign_VVR(vs8_x1_quad, vs8_x0_quad, 4);
    HVX_Vector vs8_r1_quad      = Q6_V_valign_VVR(vs8_x1_quad, vs8_x0_quad, 8);
    HVX_Vector vs8_r2_quad      = Q6_V_valign_VVR(vs8_x1_quad, vs8_x0_quad, 12);
    HVX_VectorPair ws16_c_quad  = Q6_Wh_vunpack_Vb(vs8_x0_quad);
    HVX_VectorPair ws16_r0_quad = Q6_Wh_vunpack_Vb(vs8_r0_quad);
    HVX_VectorPair ws16_r1_quad = Q6_Wh_vunpack_Vb(vs8_r1_quad);
    HVX_VectorPair ws16_r2_quad = Q6_Wh_vunpack_Vb(vs8_r2_quad);

    ws32_result_l = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_lo_W(ws16_c_quad));
    ws32_result_h = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_hi_W(ws16_c_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(ws16_r0_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(ws16_r0_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(ws16_r1_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(ws16_r1_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(ws16_r2_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(ws16_r2_quad));

    vs8_x0_quad = vs8_x1_quad;
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H1Core(HVX_Vector &vu16_x0_quad, HVX_Vector &vu16_x1_twin, HVX_VectorPair &ws32_result,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair wu16_quad = Q6_W_vshuff_VVR(vu16_x1_twin, vu16_x1_twin, -4);
    HVX_Vector vu16_x1_quad  = Q6_V_lo_W(wu16_quad);
    vu16_x1_twin             = Q6_V_hi_W(wu16_quad);

    HVX_Vector vu16_r0_quad = Q6_V_valign_VVR(vu16_x1_quad, vu16_x0_quad, 8);
    HVX_Vector vu16_r1_quad = Q6_V_valign_VVR(vu16_x1_quad, vu16_x0_quad, 16);
    HVX_Vector vu16_r2_quad = Q6_V_valign_VVR(vu16_x1_quad, vu16_x0_quad, 24);

    ws32_result = Q6_Ww_vmpy_VhVuh(vs16_alpha0, vu16_x0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha1, vu16_r0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha2, vu16_r1_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha3, vu16_r2_quad);

    vu16_x0_quad = vu16_x1_quad;
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H1Core(HVX_Vector &vs16_x0_quad, HVX_Vector &vs16_x1_twin, HVX_VectorPair &ws32_result,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair ws16_quad = Q6_W_vshuff_VVR(vs16_x1_twin, vs16_x1_twin, -4);
    HVX_Vector vs16_x1_quad  = Q6_V_lo_W(ws16_quad);
    vs16_x1_twin             = Q6_V_hi_W(ws16_quad);

    HVX_Vector vs16_r0_quad = Q6_V_valign_VVR(vs16_x1_quad, vs16_x0_quad, 8);
    HVX_Vector vs16_r1_quad = Q6_V_valign_VVR(vs16_x1_quad, vs16_x0_quad, 16);
    HVX_Vector vs16_r2_quad = Q6_V_valign_VVR(vs16_x1_quad, vs16_x0_quad, 24);

    ws32_result = Q6_Ww_vmpy_VhVh(vs16_alpha0, vs16_x0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha1, vs16_r0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha2, vs16_r1_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha3, vs16_r2_quad);

    vs16_x0_quad = vs16_x1_quad;
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H2Core(HVX_Vector &vu8_x0_quad, HVX_Vector &vu8_x1_quad,
                                              HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_Vector vu8_r0_quad      = Q6_V_valign_VVR(vu8_x1_quad, vu8_x0_quad, 4);
    HVX_Vector vu8_r1_quad      = Q6_V_valign_VVR(vu8_x1_quad, vu8_x0_quad, 8);
    HVX_Vector vu8_r2_quad      = Q6_V_valign_VVR(vu8_x1_quad, vu8_x0_quad, 12);
    HVX_VectorPair wu16_c_quad  = Q6_Wuh_vunpack_Vub(vu8_x0_quad);
    HVX_VectorPair wu16_r0_quad = Q6_Wuh_vunpack_Vub(vu8_r0_quad);
    HVX_VectorPair wu16_r1_quad = Q6_Wuh_vunpack_Vub(vu8_r1_quad);
    HVX_VectorPair wu16_r2_quad = Q6_Wuh_vunpack_Vub(vu8_r2_quad);

    ws32_result_l = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_lo_W(wu16_c_quad));
    ws32_result_h = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_hi_W(wu16_c_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(wu16_r0_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(wu16_r0_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(wu16_r1_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(wu16_r1_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(wu16_r2_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(wu16_r2_quad));

    vu8_x0_quad = vu8_x1_quad;
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H2Core(HVX_Vector &vs8_x0_quad, HVX_Vector &vs8_x1_quad,
                                              HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_Vector vs8_r0_quad      = Q6_V_valign_VVR(vs8_x1_quad, vs8_x0_quad, 4);
    HVX_Vector vs8_r1_quad      = Q6_V_valign_VVR(vs8_x1_quad, vs8_x0_quad, 8);
    HVX_Vector vs8_r2_quad      = Q6_V_valign_VVR(vs8_x1_quad, vs8_x0_quad, 12);
    HVX_VectorPair ws16_c_quad  = Q6_Wh_vunpack_Vb(vs8_x0_quad);
    HVX_VectorPair ws16_r0_quad = Q6_Wh_vunpack_Vb(vs8_r0_quad);
    HVX_VectorPair ws16_r1_quad = Q6_Wh_vunpack_Vb(vs8_r1_quad);
    HVX_VectorPair ws16_r2_quad = Q6_Wh_vunpack_Vb(vs8_r2_quad);

    ws32_result_l = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_lo_W(ws16_c_quad));
    ws32_result_h = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_hi_W(ws16_c_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(ws16_r0_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(ws16_r0_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(ws16_r1_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(ws16_r1_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(ws16_r2_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(ws16_r2_quad));

    vs8_x0_quad = vs8_x1_quad;
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H2Core(HVX_Vector &vu16_x0_quad, HVX_Vector &vu16_x1_quad, HVX_VectorPair &ws32_result,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_Vector vu16_r0_quad = Q6_V_valign_VVR(vu16_x1_quad, vu16_x0_quad, 8);
    HVX_Vector vu16_r1_quad = Q6_V_valign_VVR(vu16_x1_quad, vu16_x0_quad, 16);
    HVX_Vector vu16_r2_quad = Q6_V_valign_VVR(vu16_x1_quad, vu16_x0_quad, 24);

    ws32_result = Q6_Ww_vmpy_VhVuh(vs16_alpha0, vu16_x0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha1, vu16_r0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha2, vu16_r1_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha3, vu16_r2_quad);

    vu16_x0_quad = vu16_x1_quad;
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H2Core(HVX_Vector &vs16_x0_quad, HVX_Vector &vs16_x1_quad, HVX_VectorPair &ws32_result,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_Vector vs16_r0_quad = Q6_V_valign_VVR(vs16_x1_quad, vs16_x0_quad, 8);
    HVX_Vector vs16_r1_quad = Q6_V_valign_VVR(vs16_x1_quad, vs16_x0_quad, 16);
    HVX_Vector vs16_r2_quad = Q6_V_valign_VVR(vs16_x1_quad, vs16_x0_quad, 24);

    ws32_result = Q6_Ww_vmpy_VhVh(vs16_alpha0, vs16_x0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha1, vs16_r0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha2, vs16_r1_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha3, vs16_r2_quad);

    vs16_x0_quad = vs16_x1_quad;
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H3Core(HVX_Vector &vu8_x0_quad, HVX_Vector &vu8_x1_twin,
                                              HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair wu8_quad = Q6_W_vshuff_VVR(vu8_x1_twin, vu8_x1_twin, -2);
    HVX_Vector vu8_x1_quad  = Q6_V_lo_W(wu8_quad);

    HVX_Vector vu8_r0_quad      = Q6_V_valign_VVR(vu8_x1_quad, vu8_x0_quad, 4);
    HVX_Vector vu8_r1_quad      = Q6_V_valign_VVR(vu8_x1_quad, vu8_x0_quad, 8);
    HVX_Vector vu8_r2_quad      = Q6_V_valign_VVR(vu8_x1_quad, vu8_x0_quad, 12);
    HVX_VectorPair wu16_c_quad  = Q6_Wuh_vunpack_Vub(vu8_x0_quad);
    HVX_VectorPair wu16_r0_quad = Q6_Wuh_vunpack_Vub(vu8_r0_quad);
    HVX_VectorPair wu16_r1_quad = Q6_Wuh_vunpack_Vub(vu8_r1_quad);
    HVX_VectorPair wu16_r2_quad = Q6_Wuh_vunpack_Vub(vu8_r2_quad);

    ws32_result_l = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_lo_W(wu16_c_quad));
    ws32_result_h = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_hi_W(wu16_c_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(wu16_r0_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(wu16_r0_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(wu16_r1_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(wu16_r1_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(wu16_r2_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(wu16_r2_quad));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H3Core(HVX_Vector &vs8_x0_quad, HVX_Vector &vs8_x1_twin,
                                              HVX_VectorPair &ws32_result_l, HVX_VectorPair &ws32_result_h,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair ws8_quad = Q6_W_vshuff_VVR(vs8_x1_twin, vs8_x1_twin, -2);
    HVX_Vector vs8_x1_quad  = Q6_V_lo_W(ws8_quad);

    HVX_Vector vs8_r0_quad      = Q6_V_valign_VVR(vs8_x1_quad, vs8_x0_quad, 4);
    HVX_Vector vs8_r1_quad      = Q6_V_valign_VVR(vs8_x1_quad, vs8_x0_quad, 8);
    HVX_Vector vs8_r2_quad      = Q6_V_valign_VVR(vs8_x1_quad, vs8_x0_quad, 12);
    HVX_VectorPair ws16_c_quad  = Q6_Wh_vunpack_Vb(vs8_x0_quad);
    HVX_VectorPair ws16_r0_quad = Q6_Wh_vunpack_Vb(vs8_r0_quad);
    HVX_VectorPair ws16_r1_quad = Q6_Wh_vunpack_Vb(vs8_r1_quad);
    HVX_VectorPair ws16_r2_quad = Q6_Wh_vunpack_Vb(vs8_r2_quad);

    ws32_result_l = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_lo_W(ws16_c_quad));
    ws32_result_h = Q6_Ww_vmpy_VhVh(vs16_alpha0, Q6_V_hi_W(ws16_c_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha1, Q6_V_lo_W(ws16_r0_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha1, Q6_V_hi_W(ws16_r0_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha2, Q6_V_lo_W(ws16_r1_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha2, Q6_V_hi_W(ws16_r1_quad));
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha3, Q6_V_lo_W(ws16_r2_quad));
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha3, Q6_V_hi_W(ws16_r2_quad));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H3Core(HVX_Vector &vu16_x0_quad, HVX_Vector &vu16_x1_twin, HVX_VectorPair &ws32_result,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair wu16_quad = Q6_W_vshuff_VVR(vu16_x1_twin, vu16_x1_twin, -4);
    HVX_Vector vu16_x1_quad  = Q6_V_lo_W(wu16_quad);

    HVX_Vector vu16_r0_quad = Q6_V_valign_VVR(vu16_x1_quad, vu16_x0_quad, 8);
    HVX_Vector vu16_r1_quad = Q6_V_valign_VVR(vu16_x1_quad, vu16_x0_quad, 16);
    HVX_Vector vu16_r2_quad = Q6_V_valign_VVR(vu16_x1_quad, vu16_x0_quad, 24);

    ws32_result = Q6_Ww_vmpy_VhVuh(vs16_alpha0, vu16_x0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha1, vu16_r0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha2, vu16_r1_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVuh(ws32_result, vs16_alpha3, vu16_r2_quad);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuUpX4H3Core(HVX_Vector &vs16_x0_quad, HVX_Vector &vs16_x1_twin, HVX_VectorPair &ws32_result,
                                              HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1, HVX_Vector &vs16_alpha2, HVX_Vector &vs16_alpha3)
{
    HVX_VectorPair ws16_quad = Q6_W_vshuff_VVR(vs16_x1_twin, vs16_x1_twin, -4);
    HVX_Vector vs16_x1_quad  = Q6_V_lo_W(ws16_quad);

    HVX_Vector vs16_r0_quad = Q6_V_valign_VVR(vs16_x1_quad, vs16_x0_quad, 8);
    HVX_Vector vs16_r1_quad = Q6_V_valign_VVR(vs16_x1_quad, vs16_x0_quad, 16);
    HVX_Vector vs16_r2_quad = Q6_V_valign_VVR(vs16_x1_quad, vs16_x0_quad, 24);

    ws32_result = Q6_Ww_vmpy_VhVh(vs16_alpha0, vs16_x0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha1, vs16_r0_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha2, vs16_r1_quad);
    ws32_result = Q6_Ww_vmpyacc_WwVhVh(ws32_result, vs16_alpha3, vs16_r2_quad);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result_l,  HVX_VectorPair &ws32_c_result_h,
                                               HVX_VectorPair &ws32_n0_result_l, HVX_VectorPair &ws32_n0_result_h,
                                               HVX_Vector &vu8_dst, MI_S32 beta0, MI_S32 beta1)
{
    HVX_Vector vs16_beta0 = Q6_Vh_vsplat_R(beta0);
    HVX_Vector vs16_beta1 = Q6_Vh_vsplat_R(beta1);

    HVX_Vector vs32_result_ll = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_lh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_hl = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_h), vs16_beta0);
    HVX_Vector vs32_result_hh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_h), vs16_beta0);
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_h), vs16_beta1));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_h), vs16_beta1));

    HVX_Vector vu16_result_l = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_lh, vs32_result_ll, 2));
    HVX_Vector vu16_result_h = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_hh, vs32_result_hl, 2));
    vu8_dst = Q6_Vb_vdeal_Vb(Q6_Vub_vasr_VhVhR_rnd_sat(vu16_result_h, vu16_result_l, 6));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result_l,  HVX_VectorPair &ws32_c_result_h,
                                               HVX_VectorPair &ws32_n0_result_l, HVX_VectorPair &ws32_n0_result_h,
                                               HVX_Vector &vs8_dst, MI_S32 beta0, MI_S32 beta1)
{
    HVX_Vector vs16_beta0 = Q6_Vh_vsplat_R(beta0);
    HVX_Vector vs16_beta1 = Q6_Vh_vsplat_R(beta1);

    HVX_Vector vs32_result_ll = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_lh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_hl = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_h), vs16_beta0);
    HVX_Vector vs32_result_hh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_h), vs16_beta0);
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_h), vs16_beta1));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_h), vs16_beta1));

    HVX_Vector vs16_result_l = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_lh, vs32_result_ll, 2));
    HVX_Vector vs16_result_h = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_hh, vs32_result_hl, 2));
    vs8_dst = Q6_Vb_vdeal_Vb(Q6_Vb_vasr_VhVhR_rnd_sat(vs16_result_h, vs16_result_l, 6));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result, HVX_VectorPair &ws32_n0_result,
                                               HVX_Vector &vu16_dst, MI_S32 beta0, MI_S32 beta1)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(beta0);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(beta1);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    HVX_VectorPair ws64_result_l = Q6_Wd_vmul_VwVw(Q6_V_lo_W(ws32_c_result), vs32_beta0);
    HVX_VectorPair ws64_result_h = Q6_Wd_vmul_VwVw(Q6_V_hi_W(ws32_c_result), vs32_beta0);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n0_result), vs32_beta1);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n0_result), vs32_beta1);
    ws64_result_l = Q6_Wd_vadd_WdWd(ws64_result_l, ws64_rnd);
    ws64_result_h = Q6_Wd_vadd_WdWd(ws64_result_h, ws64_rnd);

    HVX_Vector vu32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_l), 30));
    HVX_Vector vu32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_h), 30));
    vu32_sum_l = Q6_Vw_vmax_VwVw(vu32_sum_l, Q6_V_vzero());
    vu32_sum_h = Q6_Vw_vmax_VwVw(vu32_sum_h, Q6_V_vzero());
    vu16_dst = Q6_Vuh_vsat_VuwVuw(vu32_sum_h, vu32_sum_l);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result, HVX_VectorPair &ws32_n0_result,
                                               HVX_Vector &vs16_dst, MI_S32 beta0, MI_S32 beta1)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(beta0);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(beta1);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    HVX_VectorPair ws64_result_l = Q6_Wd_vmul_VwVw(Q6_V_lo_W(ws32_c_result), vs32_beta0);
    HVX_VectorPair ws64_result_h = Q6_Wd_vmul_VwVw(Q6_V_hi_W(ws32_c_result), vs32_beta0);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n0_result), vs32_beta1);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n0_result), vs32_beta1);
    ws64_result_l = Q6_Wd_vadd_WdWd(ws64_result_l, ws64_rnd);
    ws64_result_h = Q6_Wd_vadd_WdWd(ws64_result_h, ws64_rnd);

    HVX_Vector vs32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_l), 30));
    HVX_Vector vs32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_h), 30));
    vs16_dst = Q6_Vh_vsat_VwVw(vs32_sum_h, vs32_sum_l);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result_l,  HVX_VectorPair &ws32_c_result_h,
                                               HVX_VectorPair &ws32_n0_result_l, HVX_VectorPair &ws32_n0_result_h,
                                               HVX_VectorPair &ws32_n1_result_l, HVX_VectorPair &ws32_n1_result_h,
                                               HVX_Vector &vu8_dst, MI_S32 beta0, MI_S32 beta1, MI_S32 beta2)
{
    HVX_Vector vs16_beta0 = Q6_Vh_vsplat_R(beta0);
    HVX_Vector vs16_beta1 = Q6_Vh_vsplat_R(beta1);
    HVX_Vector vs16_beta2 = Q6_Vh_vsplat_R(beta2);

    HVX_Vector vs32_result_ll = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_lh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_hl = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_h), vs16_beta0);
    HVX_Vector vs32_result_hh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_h), vs16_beta0);
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_h), vs16_beta1));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_h), vs16_beta1));
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_l), vs16_beta2));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_l), vs16_beta2));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_h), vs16_beta2));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_h), vs16_beta2));

    HVX_Vector vu16_result_l = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_lh, vs32_result_ll, 2));
    HVX_Vector vu16_result_h = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_hh, vs32_result_hl, 2));
    vu8_dst = Q6_Vb_vdeal_Vb(Q6_Vub_vasr_VhVhR_rnd_sat(vu16_result_h, vu16_result_l, 6));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result_l,  HVX_VectorPair &ws32_c_result_h,
                                               HVX_VectorPair &ws32_n0_result_l, HVX_VectorPair &ws32_n0_result_h,
                                               HVX_VectorPair &ws32_n1_result_l, HVX_VectorPair &ws32_n1_result_h,
                                               HVX_Vector &vs8_dst, MI_S32 beta0, MI_S32 beta1, MI_S32 beta2)
{
    HVX_Vector vs16_beta0 = Q6_Vh_vsplat_R(beta0);
    HVX_Vector vs16_beta1 = Q6_Vh_vsplat_R(beta1);
    HVX_Vector vs16_beta2 = Q6_Vh_vsplat_R(beta2);

    HVX_Vector vs32_result_ll = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_lh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_hl = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_h), vs16_beta0);
    HVX_Vector vs32_result_hh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_h), vs16_beta0);
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_h), vs16_beta1));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_h), vs16_beta1));
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_l), vs16_beta2));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_l), vs16_beta2));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_h), vs16_beta2));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_h), vs16_beta2));

    HVX_Vector vs16_result_l = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_lh, vs32_result_ll, 2));
    HVX_Vector vs16_result_h = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_hh, vs32_result_hl, 2));
    vs8_dst = Q6_Vb_vdeal_Vb(Q6_Vb_vasr_VhVhR_rnd_sat(vs16_result_h, vs16_result_l, 6));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result, HVX_VectorPair &ws32_n0_result, HVX_VectorPair &ws32_n1_result,
                                               HVX_Vector &vu16_dst, MI_S32 beta0, MI_S32 beta1, MI_S32 beta2)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(beta0);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(beta1);
    HVX_Vector vs32_beta2   = Q6_V_vsplat_R(beta2);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    HVX_VectorPair ws64_result_l = Q6_Wd_vmul_VwVw(Q6_V_lo_W(ws32_c_result), vs32_beta0);
    HVX_VectorPair ws64_result_h = Q6_Wd_vmul_VwVw(Q6_V_hi_W(ws32_c_result), vs32_beta0);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n0_result), vs32_beta1);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n0_result), vs32_beta1);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n1_result), vs32_beta2);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n1_result), vs32_beta2);
    ws64_result_l = Q6_Wud_vadd_WudWud(ws64_result_l, ws64_rnd);
    ws64_result_h = Q6_Wud_vadd_WudWud(ws64_result_h, ws64_rnd);

    HVX_Vector vu32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_l), 30));
    HVX_Vector vu32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_h), 30));
    vu32_sum_l = Q6_Vw_vmax_VwVw(vu32_sum_l, Q6_V_vzero());
    vu32_sum_h = Q6_Vw_vmax_VwVw(vu32_sum_h, Q6_V_vzero());
    vu16_dst = Q6_Vuh_vsat_VuwVuw(vu32_sum_h, vu32_sum_l);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result, HVX_VectorPair &ws32_n0_result, HVX_VectorPair &ws32_n1_result,
                                               HVX_Vector &vs16_dst, MI_S32 beta0, MI_S32 beta1, MI_S32 beta2)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(beta0);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(beta1);
    HVX_Vector vs32_beta2   = Q6_V_vsplat_R(beta2);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    HVX_VectorPair ws64_result_l = Q6_Wd_vmul_VwVw(Q6_V_lo_W(ws32_c_result), vs32_beta0);
    HVX_VectorPair ws64_result_h = Q6_Wd_vmul_VwVw(Q6_V_hi_W(ws32_c_result), vs32_beta0);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n0_result), vs32_beta1);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n0_result), vs32_beta1);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n1_result), vs32_beta2);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n1_result), vs32_beta2);
    ws64_result_l = Q6_Wd_vadd_WdWd(ws64_result_l, ws64_rnd);
    ws64_result_h = Q6_Wd_vadd_WdWd(ws64_result_h, ws64_rnd);

    HVX_Vector vs32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_l), 30));
    HVX_Vector vs32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_h), 30));
    vs16_dst = Q6_Vh_vsat_VwVw(vs32_sum_h, vs32_sum_l);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result_l,  HVX_VectorPair &ws32_c_result_h,
                                               HVX_VectorPair &ws32_n0_result_l, HVX_VectorPair &ws32_n0_result_h,
                                               HVX_VectorPair &ws32_n1_result_l, HVX_VectorPair &ws32_n1_result_h,
                                               HVX_VectorPair &ws32_n2_result_l, HVX_VectorPair &ws32_n2_result_h,
                                               HVX_Vector &vu8_dst, MI_S32 beta0, MI_S32 beta1, MI_S32 beta2, MI_S32 beta3)
{
    HVX_Vector vs16_beta0 = Q6_Vh_vsplat_R(beta0);
    HVX_Vector vs16_beta1 = Q6_Vh_vsplat_R(beta1);
    HVX_Vector vs16_beta2 = Q6_Vh_vsplat_R(beta2);
    HVX_Vector vs16_beta3 = Q6_Vh_vsplat_R(beta3);

    HVX_Vector vs32_result_ll = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_lh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_hl = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_h), vs16_beta0);
    HVX_Vector vs32_result_hh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_h), vs16_beta0);
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_h), vs16_beta1));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_h), vs16_beta1));
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_l), vs16_beta2));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_l), vs16_beta2));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_h), vs16_beta2));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_h), vs16_beta2));
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n2_result_l), vs16_beta3));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n2_result_l), vs16_beta3));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n2_result_h), vs16_beta3));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n2_result_h), vs16_beta3));

    HVX_Vector vu16_result_l = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_lh, vs32_result_ll, 2));
    HVX_Vector vu16_result_h = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_hh, vs32_result_hl, 2));
    vu8_dst = Q6_Vb_vdeal_Vb(Q6_Vub_vasr_VhVhR_rnd_sat(vu16_result_h, vu16_result_l, 6));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result_l,  HVX_VectorPair &ws32_c_result_h,
                                               HVX_VectorPair &ws32_n0_result_l, HVX_VectorPair &ws32_n0_result_h,
                                               HVX_VectorPair &ws32_n1_result_l, HVX_VectorPair &ws32_n1_result_h,
                                               HVX_VectorPair &ws32_n2_result_l, HVX_VectorPair &ws32_n2_result_h,
                                               HVX_Vector &vs8_dst, MI_S32 beta0, MI_S32 beta1, MI_S32 beta2, MI_S32 beta3)
{
    HVX_Vector vs16_beta0 = Q6_Vh_vsplat_R(beta0);
    HVX_Vector vs16_beta1 = Q6_Vh_vsplat_R(beta1);
    HVX_Vector vs16_beta2 = Q6_Vh_vsplat_R(beta2);
    HVX_Vector vs16_beta3 = Q6_Vh_vsplat_R(beta3);

    HVX_Vector vs32_result_ll = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_lh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_l), vs16_beta0);
    HVX_Vector vs32_result_hl = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_c_result_h), vs16_beta0);
    HVX_Vector vs32_result_hh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_c_result_h), vs16_beta0);
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_l), vs16_beta1));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_h), vs16_beta1));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_h), vs16_beta1));
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_l), vs16_beta2));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_l), vs16_beta2));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_h), vs16_beta2));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_h), vs16_beta2));
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n2_result_l), vs16_beta3));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n2_result_l), vs16_beta3));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n2_result_h), vs16_beta3));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n2_result_h), vs16_beta3));

    HVX_Vector vs16_result_l = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_lh, vs32_result_ll, 2));
    HVX_Vector vs16_result_h = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_hh, vs32_result_hl, 2));
    vs8_dst = Q6_Vb_vdeal_Vb(Q6_Vb_vasr_VhVhR_rnd_sat(vs16_result_h, vs16_result_l, 6));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result,  HVX_VectorPair &ws32_n0_result,
                                               HVX_VectorPair &ws32_n1_result, HVX_VectorPair &ws32_n2_result,
                                               HVX_Vector &vu16_dst, MI_S32 beta0, MI_S32 beta1, MI_S32 beta2, MI_S32 beta3)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(beta0);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(beta1);
    HVX_Vector vs32_beta2   = Q6_V_vsplat_R(beta2);
    HVX_Vector vs32_beta3   = Q6_V_vsplat_R(beta3);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    HVX_VectorPair ws64_result_l = Q6_Wd_vmul_VwVw(Q6_V_lo_W(ws32_c_result), vs32_beta0);
    HVX_VectorPair ws64_result_h = Q6_Wd_vmul_VwVw(Q6_V_hi_W(ws32_c_result), vs32_beta0);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n0_result), vs32_beta1);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n0_result), vs32_beta1);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n1_result), vs32_beta2);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n1_result), vs32_beta2);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n2_result), vs32_beta3);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n2_result), vs32_beta3);
    ws64_result_l = Q6_Wud_vadd_WudWud(ws64_result_l, ws64_rnd);
    ws64_result_h = Q6_Wud_vadd_WudWud(ws64_result_h, ws64_rnd);

    HVX_Vector vu32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_l), 30));
    HVX_Vector vu32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_h), 30));
    vu32_sum_l = Q6_Vw_vmax_VwVw(vu32_sum_l, Q6_V_vzero());
    vu32_sum_h = Q6_Vw_vmax_VwVw(vu32_sum_h, Q6_V_vzero());

    vu16_dst = Q6_Vuh_vsat_VuwVuw(vu32_sum_h, vu32_sum_l);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeCuFastUpVCore(HVX_VectorPair &ws32_c_result,  HVX_VectorPair &ws32_n0_result,
                                               HVX_VectorPair &ws32_n1_result, HVX_VectorPair &ws32_n2_result,
                                               HVX_Vector &vs16_dst, MI_S32 beta0, MI_S32 beta1, MI_S32 beta2, MI_S32 beta3)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(beta0);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(beta1);
    HVX_Vector vs32_beta2   = Q6_V_vsplat_R(beta2);
    HVX_Vector vs32_beta3   = Q6_V_vsplat_R(beta3);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    HVX_VectorPair ws64_result_l = Q6_Wd_vmul_VwVw(Q6_V_lo_W(ws32_c_result), vs32_beta0);
    HVX_VectorPair ws64_result_h = Q6_Wd_vmul_VwVw(Q6_V_hi_W(ws32_c_result), vs32_beta0);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n0_result), vs32_beta1);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n0_result), vs32_beta1);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n1_result), vs32_beta2);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n1_result), vs32_beta2);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, Q6_V_lo_W(ws32_n2_result), vs32_beta3);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, Q6_V_hi_W(ws32_n2_result), vs32_beta3);
    ws64_result_l = Q6_Wd_vadd_WdWd(ws64_result_l, ws64_rnd);
    ws64_result_h = Q6_Wd_vadd_WdWd(ws64_result_h, ws64_rnd);

    HVX_Vector vs32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_l), 30));
    HVX_Vector vs32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_h), 30));
    vs16_dst = Q6_Vh_vsat_VwVw(vs32_sum_h, vs32_sum_l);
}

// Tp = MI_U8 / MI_S8
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
ResizeCuUpX2ZeroRow(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 istride, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));
    const Tp *src_n2 = src_n1 + (istride / sizeof(Tp));
    Tp *dst_n0 = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1 = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2 = dst_n1 + (ostride / sizeof(Tp));
    Tp *dst_n3 = dst_n2 + (ostride / sizeof(Tp));

    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S32 *row_head      = reinterpret_cast<MI_S32*>(vtcm_buffer->row_head);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src, mv_n2_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src, mv_n2_x1_src;
    MVType mv_c_dst, mv_n0_dst, mv_n1_dst, mv_n2_dst, mv_n3_dst;
    HVX_Vector v_alpha0 = Q6_V_vsplat_R(0xFFB8FF28);    // (-72  << 16) + (-216)
    HVX_Vector v_alpha1 = Q6_V_vsplat_R(0x02180708);    // (536  << 16) + (1800)
    HVX_Vector v_alpha2 = Q6_V_vsplat_R(0x07080218);    // (1800 << 16) + (536)
    HVX_Vector v_alpha3 = Q6_V_vsplat_R(0xFF28FFB8);    // (-216 << 16) + (-72)

    MI_S32 *row1_head = row_head + C * 6;
    MI_S32 *row2_head = row_head + C * 12;
    MI_S32 *row3_head = row_head + C * 18;

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 row0_head[3];
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;

        row0_head[0]   = src_c[id0]  * 2264 + src_c[id1]  * (-216);
        row1_head[id0] = src_n0[id0] * 2264 + src_n0[id1] * (-216);
        row2_head[id0] = src_n1[id0] * 2264 + src_n1[id1] * (-216);
        row3_head[id0] = src_n2[id0] * 2264 + src_n2[id1] * (-216);
        row0_head[1]   = src_c[id0]  * 1584 + src_c[id1]  * 536  + src_c[id2]  * (-72);
        row1_head[id1] = src_n0[id0] * 1584 + src_n0[id1] * 536  + src_n0[id2] * (-72);
        row2_head[id1] = src_n1[id0] * 1584 + src_n1[id1] * 536  + src_n1[id2] * (-72);
        row3_head[id1] = src_n2[id0] * 1584 + src_n2[id1] * 536  + src_n2[id2] * (-72);
        row0_head[2]   = src_c[id0]  * 464  + src_c[id1]  * 1800 + src_c[id2]  * (-216);
        row1_head[id2] = src_n0[id0] * 464  + src_n0[id1] * 1800 + src_n0[id2] * (-216);
        row2_head[id2] = src_n1[id0] * 464  + src_n1[id1] * 1800 + src_n1[id2] * (-216);
        row3_head[id2] = src_n2[id0] * 464  + src_n2[id1] * 1800 + src_n2[id2] * (-216);

        dst_c[id0]  = SaturateCast<Tp>((row0_head[0] * 2264   + row1_head[id0] * (-216) + 2097152) >> 22);
        dst_c[id1]  = SaturateCast<Tp>((row0_head[1] * 2264   + row1_head[id1] * (-216) + 2097152) >> 22);
        dst_c[id2]  = SaturateCast<Tp>((row0_head[2] * 2264   + row1_head[id2] * (-216) + 2097152) >> 22);
        dst_n0[id0] = SaturateCast<Tp>((row0_head[0] * 1584   + row1_head[id0] * 536  + row2_head[id0] * (-72)  + 2097152) >> 22);
        dst_n0[id1] = SaturateCast<Tp>((row0_head[1] * 1584   + row1_head[id1] * 536  + row2_head[id1] * (-72)  + 2097152) >> 22);
        dst_n0[id2] = SaturateCast<Tp>((row0_head[2] * 1584   + row1_head[id2] * 536  + row2_head[id2] * (-72)  + 2097152) >> 22);
        dst_n1[id0] = SaturateCast<Tp>((row0_head[0] * 464    + row1_head[id0] * 1800 + row2_head[id0] * (-216) + 2097152) >> 22);
        dst_n1[id1] = SaturateCast<Tp>((row0_head[1] * 464    + row1_head[id1] * 1800 + row2_head[id1] * (-216) + 2097152) >> 22);
        dst_n1[id2] = SaturateCast<Tp>((row0_head[2] * 464    + row1_head[id2] * 1800 + row2_head[id2] * (-216) + 2097152) >> 22);
        dst_n2[id0] = SaturateCast<Tp>((row0_head[0] * (-216) + row1_head[id0] * 1800 + row2_head[id0] * 536  + row3_head[id0] *  (-72) + 2097152) >> 22);
        dst_n2[id1] = SaturateCast<Tp>((row0_head[1] * (-216) + row1_head[id1] * 1800 + row2_head[id1] * 536  + row3_head[id1] *  (-72) + 2097152) >> 22);
        dst_n2[id2] = SaturateCast<Tp>((row0_head[2] * (-216) + row1_head[id2] * 1800 + row2_head[id2] * 536  + row3_head[id2] *  (-72) + 2097152) >> 22);
        dst_n3[id0] = SaturateCast<Tp>((row0_head[0] * (-72)  + row1_head[id0] * 536  + row2_head[id0] * 1800 + row3_head[id0] * (-216) + 2097152) >> 22);
        dst_n3[id1] = SaturateCast<Tp>((row0_head[1] * (-72)  + row1_head[id1] * 536  + row2_head[id1] * 1800 + row3_head[id1] * (-216) + 2097152) >> 22);
        dst_n3[id2] = SaturateCast<Tp>((row0_head[2] * (-72)  + row1_head[id2] * 536  + row2_head[id2] * 1800 + row3_head[id2] * (-216) + 2097152) >> 22);
    }

    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_c_result_l, ws32_n0_result_l, ws32_n1_result_l, ws32_n2_result_l;
    HVX_VectorPair ws32_c_result_h, ws32_n0_result_h, ws32_n1_result_h, ws32_n2_result_h;

    auto resize_cu_up2_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h,
                                mv_c_dst.val[ch], 2264, -216);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                mv_n0_dst.val[ch], 1584, 536, -72);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                mv_n1_dst.val[ch], 464, 1800, -216);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_n2_dst.val[ch], -216, 1800, 536, -72);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_n3_dst.val[ch], -72, 536, 1800, -216);
    };

    vload(src_c , mv_c_x0_src);
    vload(src_n0, mv_n0_x0_src);
    vload(src_n1, mv_n1_x0_src);
    vload(src_n2, mv_n2_x0_src);

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        vload(src_c  + (x + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (x + elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (x + elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (x + elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + 3) * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + elem_counts + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + elem_counts + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + elem_counts + 3) * C, mv_n3_dst);

        mv_c_x0_src  = mv_c_x1_src;
        mv_n0_x0_src = mv_n0_x1_src;
        mv_n1_x0_src = mv_n1_x1_src;
        mv_n2_x0_src = mv_n2_x1_src;
    }

    if (width_align < iwidth - 3)
    {
        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + 3) * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border  = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  width_align + elem_counts - iwidth);
            HVX_Vector v_n0_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n0_x1_src.val[ch], width_align + elem_counts - iwidth);
            HVX_Vector v_n1_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n1_x1_src.val[ch], width_align + elem_counts - iwidth);
            HVX_Vector v_n2_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n2_x1_src.val[ch], width_align + elem_counts - iwidth);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + elem_counts + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + elem_counts + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + elem_counts + 3) * C, mv_n3_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 3 - (elem_counts << 1);

        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_n1 + sx * C, mv_n1_x0_src);
        vload(src_n2 + sx * C, mv_n2_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);
        vstore(dst_n3 + dx * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border  = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch],  3);
            HVX_Vector v_n0_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n0_x1_src.val[ch], 3);
            HVX_Vector v_n1_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n1_x1_src.val[ch], 3);
            HVX_Vector v_n2_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n2_x1_src.val[ch], 3);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + elem_counts) * C, mv_n3_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 row0_head[3];
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[0]          = src_c[id2]  * (-216) + src_c[id1]  * 1800 + src_c[id0]  * 464;
        row1_head[3 * C + ch] = src_n0[id2] * (-216) + src_n0[id1] * 1800 + src_n0[id0] * 464;
        row2_head[3 * C + ch] = src_n1[id2] * (-216) + src_n1[id1] * 1800 + src_n1[id0] * 464;
        row3_head[3 * C + ch] = src_n2[id2] * (-216) + src_n2[id1] * 1800 + src_n2[id0] * 464;
        row0_head[1]          = src_c[id2]  * (-72)  + src_c[id1]  * 536  + src_c[id0]  * 1584;
        row1_head[4 * C + ch] = src_n0[id2] * (-72)  + src_n0[id1] * 536  + src_n0[id0] * 1584;
        row2_head[4 * C + ch] = src_n1[id2] * (-72)  + src_n1[id1] * 536  + src_n1[id0] * 1584;
        row3_head[4 * C + ch] = src_n2[id2] * (-72)  + src_n2[id1] * 536  + src_n2[id0] * 1584;
        row0_head[2]          = src_c[id1]  * (-216) + src_c[id0]  * 2264;
        row1_head[5 * C + ch] = src_n0[id1] * (-216) + src_n0[id0] * 2264;
        row2_head[5 * C + ch] = src_n1[id1] * (-216) + src_n1[id0] * 2264;
        row3_head[5 * C + ch] = src_n2[id1] * (-216) + src_n2[id0] * 2264;

        dst_c[(owidth - 3) * C + ch]  = SaturateCast<Tp>((row0_head[0] * 2264   + row1_head[3 * C + ch] * (-216) + 2097152) >> 22);
        dst_c[(owidth - 2) * C + ch]  = SaturateCast<Tp>((row0_head[1] * 2264   + row1_head[4 * C + ch] * (-216) + 2097152) >> 22);
        dst_c[(owidth - 1) * C + ch]  = SaturateCast<Tp>((row0_head[2] * 2264   + row1_head[5 * C + ch] * (-216) + 2097152) >> 22);
        dst_n0[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[0] * 1584   + row1_head[3 * C + ch] * 536  + row2_head[3 * C + ch] * (-72)  + 2097152) >> 22);
        dst_n0[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[1] * 1584   + row1_head[4 * C + ch] * 536  + row2_head[4 * C + ch] * (-72)  + 2097152) >> 22);
        dst_n0[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[2] * 1584   + row1_head[5 * C + ch] * 536  + row2_head[5 * C + ch] * (-72)  + 2097152) >> 22);
        dst_n1[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[0] * 464    + row1_head[3 * C + ch] * 1800 + row2_head[3 * C + ch] * (-216) + 2097152) >> 22);
        dst_n1[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[1] * 464    + row1_head[4 * C + ch] * 1800 + row2_head[4 * C + ch] * (-216) + 2097152) >> 22);
        dst_n1[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[2] * 464    + row1_head[5 * C + ch] * 1800 + row2_head[5 * C + ch] * (-216) + 2097152) >> 22);
        dst_n2[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[0] * (-216) + row1_head[3 * C + ch] * 1800 + row2_head[3 * C + ch] * 536  + row3_head[3 * C + ch] *  (-72) + 2097152) >> 22);
        dst_n2[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[1] * (-216) + row1_head[4 * C + ch] * 1800 + row2_head[4 * C + ch] * 536  + row3_head[4 * C + ch] *  (-72) + 2097152) >> 22);
        dst_n2[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[2] * (-216) + row1_head[5 * C + ch] * 1800 + row2_head[5 * C + ch] * 536  + row3_head[5 * C + ch] *  (-72) + 2097152) >> 22);
        dst_n3[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[0] * (-72)  + row1_head[3 * C + ch] * 536  + row2_head[3 * C + ch] * 1800 + row3_head[3 * C + ch] * (-216) + 2097152) >> 22);
        dst_n3[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[1] * (-72)  + row1_head[4 * C + ch] * 536  + row2_head[4 * C + ch] * 1800 + row3_head[4 * C + ch] * (-216) + 2097152) >> 22);
        dst_n3[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[2] * (-72)  + row1_head[5 * C + ch] * 536  + row2_head[5 * C + ch] * 1800 + row3_head[5 * C + ch] * (-216) + 2097152) >> 22);
    }
}

// Tp = MI_U16 / MI_S16
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
ResizeCuUpX2ZeroRow(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 istride, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));
    const Tp *src_n2 = src_n1 + (istride / sizeof(Tp));
    Tp *dst_n0 = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1 = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2 = dst_n1 + (ostride / sizeof(Tp));
    Tp *dst_n3 = dst_n2 + (ostride / sizeof(Tp));

    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S64 *row_head      = reinterpret_cast<MI_S64*>(vtcm_buffer->row_head);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src, mv_n2_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src, mv_n2_x1_src;
    MVType mv_c_dst, mv_n0_dst, mv_n1_dst, mv_n2_dst, mv_n3_dst;
    HVX_Vector v_alpha0 = Q6_V_vsplat_R(0xFB80F280);    // (-1152 << 16) + (-3456)
    HVX_Vector v_alpha1 = Q6_V_vsplat_R(0x21807080);    // (8576  << 16) + (28800)
    HVX_Vector v_alpha2 = Q6_V_vsplat_R(0x70802180);    // (28800 << 16) + (8576)
    HVX_Vector v_alpha3 = Q6_V_vsplat_R(0xF280FB80);    // (-3456 << 16) + (-1152)

    MI_S64 *row1_head = row_head + C * 6;
    MI_S64 *row2_head = row_head + C * 12;
    MI_S64 *row3_head = row_head + C * 18;

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S64 row0_head[3];
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;

        row0_head[0]   = src_c[id0]  * 36224 + src_c[id1]  * (-3456);
        row1_head[id0] = src_n0[id0] * 36224 + src_n0[id1] * (-3456);
        row2_head[id0] = src_n1[id0] * 36224 + src_n1[id1] * (-3456);
        row3_head[id0] = src_n2[id0] * 36224 + src_n2[id1] * (-3456);
        row0_head[1]   = src_c[id0]  * 25344 + src_c[id1]  * 8576  + src_c[id2]  * (-1152);
        row1_head[id1] = src_n0[id0] * 25344 + src_n0[id1] * 8576  + src_n0[id2] * (-1152);
        row2_head[id1] = src_n1[id0] * 25344 + src_n1[id1] * 8576  + src_n1[id2] * (-1152);
        row3_head[id1] = src_n2[id0] * 25344 + src_n2[id1] * 8576  + src_n2[id2] * (-1152);
        row0_head[2]   = src_c[id0]  * 7424  + src_c[id1]  * 28800 + src_c[id2]  * (-3456);
        row1_head[id2] = src_n0[id0] * 7424  + src_n0[id1] * 28800 + src_n0[id2] * (-3456);
        row2_head[id2] = src_n1[id0] * 7424  + src_n1[id1] * 28800 + src_n1[id2] * (-3456);
        row3_head[id2] = src_n2[id0] * 7424  + src_n2[id1] * 28800 + src_n2[id2] * (-3456);

        dst_c[id0]  = SaturateCast<Tp>((row0_head[0] * 36224   + row1_head[id0] * (-3456) + 536870912) >> 30);
        dst_c[id1]  = SaturateCast<Tp>((row0_head[1] * 36224   + row1_head[id1] * (-3456) + 536870912) >> 30);
        dst_c[id2]  = SaturateCast<Tp>((row0_head[2] * 36224   + row1_head[id2] * (-3456) + 536870912) >> 30);
        dst_n0[id0] = SaturateCast<Tp>((row0_head[0] * 25344   + row1_head[id0] * 8576  + row2_head[id0] * (-1152) + 536870912) >> 30);
        dst_n0[id1] = SaturateCast<Tp>((row0_head[1] * 25344   + row1_head[id1] * 8576  + row2_head[id1] * (-1152) + 536870912) >> 30);
        dst_n0[id2] = SaturateCast<Tp>((row0_head[2] * 25344   + row1_head[id2] * 8576  + row2_head[id2] * (-1152) + 536870912) >> 30);
        dst_n1[id0] = SaturateCast<Tp>((row0_head[0] * 7424    + row1_head[id0] * 28800 + row2_head[id0] * (-3456) + 536870912) >> 30);
        dst_n1[id1] = SaturateCast<Tp>((row0_head[1] * 7424    + row1_head[id1] * 28800 + row2_head[id1] * (-3456) + 536870912) >> 30);
        dst_n1[id2] = SaturateCast<Tp>((row0_head[2] * 7424    + row1_head[id2] * 28800 + row2_head[id2] * (-3456) + 536870912) >> 30);
        dst_n2[id0] = SaturateCast<Tp>((row0_head[0] * (-3456) + row1_head[id0] * 28800 + row2_head[id0] * 8576  + row3_head[id0] * (-1152) + 536870912) >> 30);
        dst_n2[id1] = SaturateCast<Tp>((row0_head[1] * (-3456) + row1_head[id1] * 28800 + row2_head[id1] * 8576  + row3_head[id1] * (-1152) + 536870912) >> 30);
        dst_n2[id2] = SaturateCast<Tp>((row0_head[2] * (-3456) + row1_head[id2] * 28800 + row2_head[id2] * 8576  + row3_head[id2] * (-1152) + 536870912) >> 30);
        dst_n3[id0] = SaturateCast<Tp>((row0_head[0] * (-1152) + row1_head[id0] * 8576  + row2_head[id0] * 28800 + row3_head[id0] * (-3456) + 536870912) >> 30);
        dst_n3[id1] = SaturateCast<Tp>((row0_head[1] * (-1152) + row1_head[id1] * 8576  + row2_head[id1] * 28800 + row3_head[id1] * (-3456) + 536870912) >> 30);
        dst_n3[id2] = SaturateCast<Tp>((row0_head[2] * (-1152) + row1_head[id2] * 8576  + row2_head[id2] * 28800 + row3_head[id2] * (-3456) + 536870912) >> 30);
    }

    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result;

    auto resize_cu_up2_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, mv_c_dst.val[ch], 36224, -3456);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, mv_n0_dst.val[ch], 25344, 8576, -1152);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, mv_n1_dst.val[ch], 7424, 28800, -3456);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_n2_dst.val[ch], -3456, 28800, 8576, -1152);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_n3_dst.val[ch], -1152, 8576, 28800, -3456);
    };

    vload(src_c , mv_c_x0_src);
    vload(src_n0, mv_n0_x0_src);
    vload(src_n1, mv_n1_x0_src);
    vload(src_n2, mv_n2_x0_src);

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        vload(src_c  + (x + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (x + elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (x + elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (x + elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + 3) * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + elem_counts + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + elem_counts + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + elem_counts + 3) * C, mv_n3_dst);

        mv_c_x0_src  = mv_c_x1_src;
        mv_n0_x0_src = mv_n0_x1_src;
        mv_n1_x0_src = mv_n1_x1_src;
        mv_n2_x0_src = mv_n2_x1_src;
    }

    if (width_align < iwidth - 3)
    {
        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + 3) * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border  = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  (width_align + elem_counts - iwidth) * 2);
            HVX_Vector v_n0_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n0_x1_src.val[ch], (width_align + elem_counts - iwidth) * 2);
            HVX_Vector v_n1_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n1_x1_src.val[ch], (width_align + elem_counts - iwidth) * 2);
            HVX_Vector v_n2_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n2_x1_src.val[ch], (width_align + elem_counts - iwidth) * 2);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + elem_counts + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + elem_counts + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + elem_counts + 3) * C, mv_n3_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 3 - (elem_counts << 1);

        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_n1 + sx * C, mv_n1_x0_src);
        vload(src_n2 + sx * C, mv_n2_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);
        vstore(dst_n3 + dx * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border  = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch],  6);
            HVX_Vector v_n0_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n0_x1_src.val[ch], 6);
            HVX_Vector v_n1_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n1_x1_src.val[ch], 6);
            HVX_Vector v_n2_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n2_x1_src.val[ch], 6);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + elem_counts) * C, mv_n3_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S64 row0_head[3];
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[0]          = src_c[id2]  * (-3456) + src_c[id1]  * 28800 + src_c[id0]  * 7424;
        row1_head[3 * C + ch] = src_n0[id2] * (-3456) + src_n0[id1] * 28800 + src_n0[id0] * 7424;
        row2_head[3 * C + ch] = src_n1[id2] * (-3456) + src_n1[id1] * 28800 + src_n1[id0] * 7424;
        row3_head[3 * C + ch] = src_n2[id2] * (-3456) + src_n2[id1] * 28800 + src_n2[id0] * 7424;
        row0_head[1]          = src_c[id2]  * (-1152) + src_c[id1]  * 8576  + src_c[id0]  * 25344;
        row1_head[4 * C + ch] = src_n0[id2] * (-1152) + src_n0[id1] * 8576  + src_n0[id0] * 25344;
        row2_head[4 * C + ch] = src_n1[id2] * (-1152) + src_n1[id1] * 8576  + src_n1[id0] * 25344;
        row3_head[4 * C + ch] = src_n2[id2] * (-1152) + src_n2[id1] * 8576  + src_n2[id0] * 25344;
        row0_head[2]          = src_c[id1]  * (-3456) + src_c[id0]  * 36224;
        row1_head[5 * C + ch] = src_n0[id1] * (-3456) + src_n0[id0] * 36224;
        row2_head[5 * C + ch] = src_n1[id1] * (-3456) + src_n1[id0] * 36224;
        row3_head[5 * C + ch] = src_n2[id1] * (-3456) + src_n2[id0] * 36224;

        dst_c[(owidth - 3) * C + ch]  = SaturateCast<Tp>((row0_head[0] * 36224   + row1_head[3 * C + ch] * (-3456) + 536870912) >> 30);
        dst_c[(owidth - 2) * C + ch]  = SaturateCast<Tp>((row0_head[1] * 36224   + row1_head[4 * C + ch] * (-3456) + 536870912) >> 30);
        dst_c[(owidth - 1) * C + ch]  = SaturateCast<Tp>((row0_head[2] * 36224   + row1_head[5 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n0[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[0] * 25344   + row1_head[3 * C + ch] * 8576  + row2_head[3 * C + ch] * (-1152) + 536870912) >> 30);
        dst_n0[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[1] * 25344   + row1_head[4 * C + ch] * 8576  + row2_head[4 * C + ch] * (-1152) + 536870912) >> 30);
        dst_n0[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[2] * 25344   + row1_head[5 * C + ch] * 8576  + row2_head[5 * C + ch] * (-1152) + 536870912) >> 30);
        dst_n1[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[0] * 7424    + row1_head[3 * C + ch] * 28800 + row2_head[3 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n1[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[1] * 7424    + row1_head[4 * C + ch] * 28800 + row2_head[4 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n1[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[2] * 7424    + row1_head[5 * C + ch] * 28800 + row2_head[5 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n2[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[0] * (-3456) + row1_head[3 * C + ch] * 28800 + row2_head[3 * C + ch] * 8576  + row3_head[3 * C + ch] * (-1152) + 536870912) >> 30);
        dst_n2[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[1] * (-3456) + row1_head[4 * C + ch] * 28800 + row2_head[4 * C + ch] * 8576  + row3_head[4 * C + ch] * (-1152) + 536870912) >> 30);
        dst_n2[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[2] * (-3456) + row1_head[5 * C + ch] * 28800 + row2_head[5 * C + ch] * 8576  + row3_head[5 * C + ch] * (-1152) + 536870912) >> 30);
        dst_n3[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[0] * (-1152) + row1_head[3 * C + ch] * 8576  + row2_head[3 * C + ch] * 28800 + row3_head[3 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n3[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[1] * (-1152) + row1_head[4 * C + ch] * 8576  + row2_head[4 * C + ch] * 28800 + row3_head[4 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n3[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[2] * (-1152) + row1_head[5 * C + ch] * 8576  + row2_head[5 * C + ch] * 28800 + row3_head[5 * C + ch] * (-3456) + 536870912) >> 30);
    }
}

// Tp = MI_U8 / MI_S8
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
ResizeCuUpX2UpBorder(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 istride, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));
    const Tp *src_n2 = src_n1 + (istride / sizeof(Tp));
    Tp *dst_n0 = dst_c + (ostride / sizeof(Tp));

    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S32 *row_head      = reinterpret_cast<MI_S32*>(vtcm_buffer->row_head);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src, mv_n2_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src, mv_n2_x1_src;
    MVType mv_c_dst, mv_n0_dst;
    HVX_Vector v_alpha0 = Q6_V_vsplat_R(0xFFB8FF28);    // (-72 << 16)  + (-216)
    HVX_Vector v_alpha1 = Q6_V_vsplat_R(0x2180708);     // (536 << 16)  + (1800)
    HVX_Vector v_alpha2 = Q6_V_vsplat_R(0x07080218);    // (1800 << 16) + (536)
    HVX_Vector v_alpha3 = Q6_V_vsplat_R(0xFF28FFB8);    // (-216 << 16) + (-72)

    MI_S32 *row1_head = row_head + C * 6;
    MI_S32 *row2_head = row_head + C * 12;
    MI_S32 *row3_head = row_head + C * 18;

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 row0_head[3];
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;

        row0_head[0]   = src_c[id0]  * 2264 + src_c[id1]  * (-216);
        row1_head[id0] = src_n0[id0] * 2264 + src_n0[id1] * (-216);
        row2_head[id0] = src_n1[id0] * 2264 + src_n1[id1] * (-216);
        row3_head[id0] = src_n2[id0] * 2264 + src_n2[id1] * (-216);
        row0_head[1]   = src_c[id0]  * 1584 + src_c[id1]  * 536  + src_c[id2]  * (-72);
        row1_head[id1] = src_n0[id0] * 1584 + src_n0[id1] * 536  + src_n0[id2] * (-72);
        row2_head[id1] = src_n1[id0] * 1584 + src_n1[id1] * 536  + src_n1[id2] * (-72);
        row3_head[id1] = src_n2[id0] * 1584 + src_n2[id1] * 536  + src_n2[id2] * (-72);
        row0_head[2]   = src_c[id0]  * 464  + src_c[id1]  * 1800 + src_c[id2]  * (-216);
        row1_head[id2] = src_n0[id0] * 464  + src_n0[id1] * 1800 + src_n0[id2] * (-216);
        row2_head[id2] = src_n1[id0] * 464  + src_n1[id1] * 1800 + src_n1[id2] * (-216);
        row3_head[id2] = src_n2[id0] * 464  + src_n2[id1] * 1800 + src_n2[id2] * (-216);

        dst_c[id0]  = SaturateCast<Tp>((row0_head[0] * (-216) + row1_head[id0] * 1800 + row2_head[id0] * 536  + row3_head[id0] * (-72)  + 2097152) >> 22);
        dst_c[id1]  = SaturateCast<Tp>((row0_head[1] * (-216) + row1_head[id1] * 1800 + row2_head[id1] * 536  + row3_head[id1] * (-72)  + 2097152) >> 22);
        dst_c[id2]  = SaturateCast<Tp>((row0_head[2] * (-216) + row1_head[id2] * 1800 + row2_head[id2] * 536  + row3_head[id2] * (-72)  + 2097152) >> 22);
        dst_n0[id0] = SaturateCast<Tp>((row0_head[0] * (-72)  + row1_head[id0] * 536  + row2_head[id0] * 1800 + row3_head[id0] * (-216) + 2097152) >> 22);
        dst_n0[id1] = SaturateCast<Tp>((row0_head[1] * (-72)  + row1_head[id1] * 536  + row2_head[id1] * 1800 + row3_head[id1] * (-216) + 2097152) >> 22);
        dst_n0[id2] = SaturateCast<Tp>((row0_head[2] * (-72)  + row1_head[id2] * 536  + row2_head[id2] * 1800 + row3_head[id2] * (-216) + 2097152) >> 22);
    }

    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_c_result_l, ws32_n0_result_l, ws32_n1_result_l, ws32_n2_result_l;
    HVX_VectorPair ws32_c_result_h, ws32_n0_result_h, ws32_n1_result_h, ws32_n2_result_h;

    auto resize_cu_up2_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_c_dst.val[ch], -216, 1800, 536, -72);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_n0_dst.val[ch], -72, 536, 1800, -216);
    };

    vload(src_c , mv_c_x0_src);
    vload(src_n0, mv_n0_x0_src);
    vload(src_n1, mv_n1_x0_src);
    vload(src_n2, mv_n2_x0_src);

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        vload(src_c  + (x + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (x + elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (x + elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (x + elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);

        mv_c_x0_src  = mv_c_x1_src;
        mv_n0_x0_src = mv_n0_x1_src;
        mv_n1_x0_src = mv_n1_x1_src;
        mv_n2_x0_src = mv_n2_x1_src;
    }

    if (width_align < iwidth - 3)
    {
        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);


        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border  = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  width_align + elem_counts - iwidth);
            HVX_Vector v_n0_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n0_x1_src.val[ch], width_align + elem_counts - iwidth);
            HVX_Vector v_n1_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n1_x1_src.val[ch], width_align + elem_counts - iwidth);
            HVX_Vector v_n2_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n2_x1_src.val[ch], width_align + elem_counts - iwidth);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 3 - (elem_counts << 1);

        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_n1 + sx * C, mv_n1_x0_src);
        vload(src_n2 + sx * C, mv_n2_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border  = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch],  3);
            HVX_Vector v_n0_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n0_x1_src.val[ch], 3);
            HVX_Vector v_n1_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n1_x1_src.val[ch], 3);
            HVX_Vector v_n2_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n2_x1_src.val[ch], 3);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 row0_head[3];
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[0]          = src_c[id2]  * (-216) + src_c[id1]  * 1800 + src_c[id0]  * 464;
        row1_head[3 * C + ch] = src_n0[id2] * (-216) + src_n0[id1] * 1800 + src_n0[id0] * 464;
        row2_head[3 * C + ch] = src_n1[id2] * (-216) + src_n1[id1] * 1800 + src_n1[id0] * 464;
        row3_head[3 * C + ch] = src_n2[id2] * (-216) + src_n2[id1] * 1800 + src_n2[id0] * 464;
        row0_head[1]          = src_c[id2]  * (-72)  + src_c[id1]  * 536  + src_c[id0]  * 1584;
        row1_head[4 * C + ch] = src_n0[id2] * (-72)  + src_n0[id1] * 536  + src_n0[id0] * 1584;
        row2_head[4 * C + ch] = src_n1[id2] * (-72)  + src_n1[id1] * 536  + src_n1[id0] * 1584;
        row3_head[4 * C + ch] = src_n2[id2] * (-72)  + src_n2[id1] * 536  + src_n2[id0] * 1584;
        row0_head[2]          = src_c[id1]  * (-216) + src_c[id0]  * 2264;
        row1_head[5 * C + ch] = src_n0[id1] * (-216) + src_n0[id0] * 2264;
        row2_head[5 * C + ch] = src_n1[id1] * (-216) + src_n1[id0] * 2264;
        row3_head[5 * C + ch] = src_n2[id1] * (-216) + src_n2[id0] * 2264;

        dst_c[(owidth - 3)  * C + ch] = SaturateCast<Tp>((row0_head[0] * (-216) + row1_head[3 * C + ch] * 1800 + row2_head[3 * C + ch] * 536  + row3_head[3 * C + ch] *  (-72) + 2097152) >> 22);
        dst_c[(owidth - 2)  * C + ch] = SaturateCast<Tp>((row0_head[1] * (-216) + row1_head[4 * C + ch] * 1800 + row2_head[4 * C + ch] * 536  + row3_head[4 * C + ch] *  (-72) + 2097152) >> 22);
        dst_c[(owidth - 1)  * C + ch] = SaturateCast<Tp>((row0_head[2] * (-216) + row1_head[5 * C + ch] * 1800 + row2_head[5 * C + ch] * 536  + row3_head[5 * C + ch] *  (-72) + 2097152) >> 22);
        dst_n0[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[0] * (-72)  + row1_head[3 * C + ch] * 536  + row2_head[3 * C + ch] * 1800 + row3_head[3 * C + ch] * (-216) + 2097152) >> 22);
        dst_n0[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[1] * (-72)  + row1_head[4 * C + ch] * 536  + row2_head[4 * C + ch] * 1800 + row3_head[4 * C + ch] * (-216) + 2097152) >> 22);
        dst_n0[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[2] * (-72)  + row1_head[5 * C + ch] * 536  + row2_head[5 * C + ch] * 1800 + row3_head[5 * C + ch] * (-216) + 2097152) >> 22);
    }
}

// Tp = MI_U16 / MI_S16
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
ResizeCuUpX2UpBorder(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 istride, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));
    const Tp *src_n2 = src_n1 + (istride / sizeof(Tp));
    Tp *dst_n0 = dst_c + (ostride / sizeof(Tp));

    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S64 *row_head      = reinterpret_cast<MI_S64*>(vtcm_buffer->row_head);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src, mv_n2_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src, mv_n2_x1_src;
    MVType mv_c_dst, mv_n0_dst;
    HVX_Vector v_alpha0 = Q6_V_vsplat_R(0xFB80F280);    // (-1152 << 16) + (-3456)
    HVX_Vector v_alpha1 = Q6_V_vsplat_R(0x21807080);    // (8576  << 16) + (28800)
    HVX_Vector v_alpha2 = Q6_V_vsplat_R(0x70802180);    // (28800 << 16) + (8576)
    HVX_Vector v_alpha3 = Q6_V_vsplat_R(0xF280FB80);    // (-3456 << 16) + (-1152)

    MI_S64 *row1_head = row_head + C * 6;
    MI_S64 *row2_head = row_head + C * 12;
    MI_S64 *row3_head = row_head + C * 18;

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S64 row0_head[3];
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;

        row0_head[0]   = src_c[id0]  * 36224 + src_c[id1]  * (-3456);
        row1_head[id0] = src_n0[id0] * 36224 + src_n0[id1] * (-3456);
        row2_head[id0] = src_n1[id0] * 36224 + src_n1[id1] * (-3456);
        row3_head[id0] = src_n2[id0] * 36224 + src_n2[id1] * (-3456);
        row0_head[1]   = src_c[id0]  * 25344 + src_c[id1]  * 8576  + src_c[id2]  * (-1152);
        row1_head[id1] = src_n0[id0] * 25344 + src_n0[id1] * 8576  + src_n0[id2] * (-1152);
        row2_head[id1] = src_n1[id0] * 25344 + src_n1[id1] * 8576  + src_n1[id2] * (-1152);
        row3_head[id1] = src_n2[id0] * 25344 + src_n2[id1] * 8576  + src_n2[id2] * (-1152);
        row0_head[2]   = src_c[id0]  * 7424  + src_c[id1]  * 28800 + src_c[id2]  * (-3456);
        row1_head[id2] = src_n0[id0] * 7424  + src_n0[id1] * 28800 + src_n0[id2] * (-3456);
        row2_head[id2] = src_n1[id0] * 7424  + src_n1[id1] * 28800 + src_n1[id2] * (-3456);
        row3_head[id2] = src_n2[id0] * 7424  + src_n2[id1] * 28800 + src_n2[id2] * (-3456);

        dst_c[id0]  = SaturateCast<Tp>((row0_head[0] * (-3456) + row1_head[id0] * 28800 + row2_head[id0] * 8576  + row3_head[id0] * (-1152) + 536870912) >> 30);
        dst_c[id1]  = SaturateCast<Tp>((row0_head[1] * (-3456) + row1_head[id1] * 28800 + row2_head[id1] * 8576  + row3_head[id1] * (-1152) + 536870912) >> 30);
        dst_c[id2]  = SaturateCast<Tp>((row0_head[2] * (-3456) + row1_head[id2] * 28800 + row2_head[id2] * 8576  + row3_head[id2] * (-1152) + 536870912) >> 30);
        dst_n0[id0] = SaturateCast<Tp>((row0_head[0] * (-1152) + row1_head[id0] * 8576  + row2_head[id0] * 28800 + row3_head[id0] * (-3456) + 536870912) >> 30);
        dst_n0[id1] = SaturateCast<Tp>((row0_head[1] * (-1152) + row1_head[id1] * 8576  + row2_head[id1] * 28800 + row3_head[id1] * (-3456) + 536870912) >> 30);
        dst_n0[id2] = SaturateCast<Tp>((row0_head[2] * (-1152) + row1_head[id2] * 8576  + row2_head[id2] * 28800 + row3_head[id2] * (-3456) + 536870912) >> 30);
    }

    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result;

    auto resize_cu_up2_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_c_dst.val[ch],  -3456, 28800, 8576, -1152);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_n0_dst.val[ch], -1152, 8576, 28800, -3456);
    };

    vload(src_c , mv_c_x0_src);
    vload(src_n0, mv_n0_x0_src);
    vload(src_n1, mv_n1_x0_src);
    vload(src_n2, mv_n2_x0_src);

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        vload(src_c  + (x + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (x + elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (x + elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (x + elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);

        mv_c_x0_src  = mv_c_x1_src;
        mv_n0_x0_src = mv_n0_x1_src;
        mv_n1_x0_src = mv_n1_x1_src;
        mv_n2_x0_src = mv_n2_x1_src;
    }

    if (width_align < iwidth - 3)
    {
        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border  = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  (width_align + elem_counts - iwidth) * 2);
            HVX_Vector v_n0_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n0_x1_src.val[ch], (width_align + elem_counts - iwidth) * 2);
            HVX_Vector v_n1_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n1_x1_src.val[ch], (width_align + elem_counts - iwidth) * 2);
            HVX_Vector v_n2_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n2_x1_src.val[ch], (width_align + elem_counts - iwidth) * 2);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 3 - (elem_counts << 1);

        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_n1 + sx * C, mv_n1_x0_src);
        vload(src_n2 + sx * C, mv_n2_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border  = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch],  6);
            HVX_Vector v_n0_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n0_x1_src.val[ch], 6);
            HVX_Vector v_n1_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n1_x1_src.val[ch], 6);
            HVX_Vector v_n2_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n2_x1_src.val[ch], 6);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX2HCore<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S64 row0_head[3];
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[0]          = src_c[id2]  * (-3456) + src_c[id1]  * 28800 + src_c[id0]  * 7424;
        row1_head[3 * C + ch] = src_n0[id2] * (-3456) + src_n0[id1] * 28800 + src_n0[id0] * 7424;
        row2_head[3 * C + ch] = src_n1[id2] * (-3456) + src_n1[id1] * 28800 + src_n1[id0] * 7424;
        row3_head[3 * C + ch] = src_n2[id2] * (-3456) + src_n2[id1] * 28800 + src_n2[id0] * 7424;
        row0_head[1]          = src_c[id2]  * (-1152) + src_c[id1]  * 8576  + src_c[id0]  * 25344;
        row1_head[4 * C + ch] = src_n0[id2] * (-1152) + src_n0[id1] * 8576  + src_n0[id0] * 25344;
        row2_head[4 * C + ch] = src_n1[id2] * (-1152) + src_n1[id1] * 8576  + src_n1[id0] * 25344;
        row3_head[4 * C + ch] = src_n2[id2] * (-1152) + src_n2[id1] * 8576  + src_n2[id0] * 25344;
        row0_head[2]          = src_c[id1]  * (-3456) + src_c[id0]  * 36224;
        row1_head[5 * C + ch] = src_n0[id1] * (-3456) + src_n0[id0] * 36224;
        row2_head[5 * C + ch] = src_n1[id1] * (-3456) + src_n1[id0] * 36224;
        row3_head[5 * C + ch] = src_n2[id1] * (-3456) + src_n2[id0] * 36224;

        dst_c[(owidth - 3)  * C + ch] = SaturateCast<Tp>((row0_head[0] * (-3456) + row1_head[3 * C + ch] * 28800 + row2_head[3 * C + ch] * 8576  + row3_head[3 * C + ch] * (-1152) + 536870912) >> 30);
        dst_c[(owidth - 2)  * C + ch] = SaturateCast<Tp>((row0_head[1] * (-3456) + row1_head[4 * C + ch] * 28800 + row2_head[4 * C + ch] * 8576  + row3_head[4 * C + ch] * (-1152) + 536870912) >> 30);
        dst_c[(owidth - 1)  * C + ch] = SaturateCast<Tp>((row0_head[2] * (-3456) + row1_head[5 * C + ch] * 28800 + row2_head[5 * C + ch] * 8576  + row3_head[5 * C + ch] * (-1152) + 536870912) >> 30);
        dst_n0[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[0] * (-1152) + row1_head[3 * C + ch] * 8576  + row2_head[3 * C + ch] * 28800 + row3_head[3 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n0[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[1] * (-1152) + row1_head[4 * C + ch] * 8576  + row2_head[4 * C + ch] * 28800 + row3_head[4 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n0[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[2] * (-1152) + row1_head[5 * C + ch] * 8576  + row2_head[5 * C + ch] * 28800 + row3_head[5 * C + ch] * (-3456) + 536870912) >> 30);
    }
}

// Tp = MI_U8 / MI_S8
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
ResizeCuUpX2Row(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    Tp *dst_n0            = dst_c + (ostride / sizeof(Tp));
    MI_S32 *row0_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row0_ptr);
    MI_S32 *row_head      = reinterpret_cast<MI_S32*>(vtcm_buffer->row_head);
    MI_S32 *row0_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row0_ptr + row_pre_size);
    vtcm_buffer->row0_ptr = reinterpret_cast<MI_U8*>(row0_ptr);
    vtcm_buffer->row1_ptr = reinterpret_cast<MI_U8*>(row1_ptr);
    vtcm_buffer->row2_ptr = reinterpret_cast<MI_U8*>(row2_ptr);
    vtcm_buffer->row3_ptr = reinterpret_cast<MI_U8*>(row3_ptr);

    MVType mv_c_x0_src, mv_c_x1_src;
    MVType mv_c_dst, mv_n0_dst;
    HVX_Vector v_alpha0 = Q6_V_vsplat_R(0xFFB8FF28);    // (-72 << 16)  + (-216)
    HVX_Vector v_alpha1 = Q6_V_vsplat_R(0x2180708);     // (536 << 16)  + (1800)
    HVX_Vector v_alpha2 = Q6_V_vsplat_R(0x07080218);    // (1800 << 16) + (536)
    HVX_Vector v_alpha3 = Q6_V_vsplat_R(0xFF28FFB8);    // (-216 << 16) + (-72)

    MI_S32 *row0_head = row_head;
    MI_S32 *row1_head = row_head + C * 6;
    MI_S32 *row2_head = row_head + C * 12;
    MI_S32 *row3_head = row_head + C * 18;

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;

        row0_head[id0] = row1_head[id0]; row1_head[id0] = row2_head[id0]; row2_head[id0] = row3_head[id0];
        row0_head[id1] = row1_head[id1]; row1_head[id1] = row2_head[id1]; row2_head[id1] = row3_head[id1];
        row0_head[id2] = row1_head[id2]; row1_head[id2] = row2_head[id2]; row2_head[id2] = row3_head[id2];
        row3_head[id0] = src_c[id0] * 2264 + src_c[id1] * (-216);
        row3_head[id1] = src_c[id0] * 1584 + src_c[id1] * 536  + src_c[id2] * (-72);
        row3_head[id2] = src_c[id0] * 464  + src_c[id1] * 1800 + src_c[id2] * (-216);

        dst_c[id0]  = SaturateCast<Tp>((row0_head[id0] * (-216) + row1_head[id0] * 1800 + row2_head[id0] * 536  + row3_head[id0] * (-72)  + 2097152) >> 22);
        dst_c[id1]  = SaturateCast<Tp>((row0_head[id1] * (-216) + row1_head[id1] * 1800 + row2_head[id1] * 536  + row3_head[id1] * (-72)  + 2097152) >> 22);
        dst_c[id2]  = SaturateCast<Tp>((row0_head[id2] * (-216) + row1_head[id2] * 1800 + row2_head[id2] * 536  + row3_head[id2] * (-72)  + 2097152) >> 22);
        dst_n0[id0] = SaturateCast<Tp>((row0_head[id0] * (-72)  + row1_head[id0] * 536  + row2_head[id0] * 1800 + row3_head[id0] * (-216) + 2097152) >> 22);
        dst_n0[id1] = SaturateCast<Tp>((row0_head[id1] * (-72)  + row1_head[id1] * 536  + row2_head[id1] * 1800 + row3_head[id1] * (-216) + 2097152) >> 22);
        dst_n0[id2] = SaturateCast<Tp>((row0_head[id2] * (-72)  + row1_head[id2] * 536  + row2_head[id2] * 1800 + row3_head[id2] * (-216) + 2097152) >> 22);
    }

    HVX_VectorPair *v_row0 = (HVX_VectorPair *)(row0_ptr);
    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_p2_result_l, ws32_p1_result_l, ws32_p0_result_l, ws32_c_result_l;
    HVX_VectorPair ws32_p2_result_h, ws32_p1_result_h, ws32_p0_result_h, ws32_c_result_h;

    auto resize_cu_up2_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_c_dst.val[ch], -216, 1800, 536, -72);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_n0_dst.val[ch], -72, 536, 1800, -216);
    };

    vload(src_c, mv_c_x0_src);

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        vload(src_c + (x + elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);

        mv_c_x0_src  = mv_c_x1_src;
    }

    if (width_align < iwidth - 3)
    {
        vload(src_c + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch], width_align + elem_counts - iwidth);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 3 - (elem_counts << 1);

        HVX_VectorPair *v_back_row0 = (HVX_VectorPair *)(row0_back_ptr);
        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c + sx * C, mv_c_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;
            *v_back_row3++   = ws32_c_result_l;
            *v_back_row3++   = ws32_c_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch], 3);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;
            *v_back_row3++   = ws32_c_result_l;
            *v_back_row3++   = ws32_c_result_h;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[3 * C + ch] = row1_head[3 * C + ch];
        row0_head[4 * C + ch] = row1_head[4 * C + ch];
        row0_head[5 * C + ch] = row1_head[5 * C + ch];
        row1_head[3 * C + ch] = row2_head[3 * C + ch];
        row1_head[4 * C + ch] = row2_head[4 * C + ch];
        row1_head[5 * C + ch] = row2_head[5 * C + ch];
        row2_head[3 * C + ch] = row3_head[3 * C + ch];
        row2_head[4 * C + ch] = row3_head[4 * C + ch];
        row2_head[5 * C + ch] = row3_head[5 * C + ch];
        row3_head[3 * C + ch] = src_c[id2] * (-216) + src_c[id1] * 1800 + src_c[id0] * 464;
        row3_head[4 * C + ch] = src_c[id2] * (-72)  + src_c[id1] * 536  + src_c[id0] * 1584;
        row3_head[5 * C + ch] = src_c[id1] * (-216) + src_c[id0] * 2264;

        dst_c[(owidth - 3) * C + ch]  = SaturateCast<Tp>((row0_head[3 * C + ch]  * (-216) + row1_head[3 * C + ch] * 1800 + row2_head[3 * C + ch] * 536  + row3_head[3 * C + ch] *  (-72) + 2097152) >> 22);
        dst_c[(owidth - 2) * C + ch]  = SaturateCast<Tp>((row0_head[4 * C + ch]  * (-216) + row1_head[4 * C + ch] * 1800 + row2_head[4 * C + ch] * 536  + row3_head[4 * C + ch] *  (-72) + 2097152) >> 22);
        dst_c[(owidth - 1) * C + ch]  = SaturateCast<Tp>((row0_head[5 * C + ch]  * (-216) + row1_head[5 * C + ch] * 1800 + row2_head[5 * C + ch] * 536  + row3_head[5 * C + ch] *  (-72) + 2097152) >> 22);
        dst_n0[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[3 * C + ch]  * (-72)  + row1_head[3 * C + ch] * 536  + row2_head[3 * C + ch] * 1800 + row3_head[3 * C + ch] * (-216) + 2097152) >> 22);
        dst_n0[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[4 * C + ch]  * (-72)  + row1_head[4 * C + ch] * 536  + row2_head[4 * C + ch] * 1800 + row3_head[4 * C + ch] * (-216) + 2097152) >> 22);
        dst_n0[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[5 * C + ch]  * (-72)  + row1_head[5 * C + ch] * 536  + row2_head[5 * C + ch] * 1800 + row3_head[5 * C + ch] * (-216) + 2097152) >> 22);
    }
}

// Tp = MI_U16 / MI_S16
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
ResizeCuUpX2Row(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    Tp *dst_n0            = dst_c + (ostride / sizeof(Tp));
    MI_S32 *row0_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row0_ptr);
    MI_S64 *row_head      = reinterpret_cast<MI_S64*>(vtcm_buffer->row_head);
    MI_S32 *row0_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row0_ptr + row_pre_size);
    vtcm_buffer->row0_ptr = reinterpret_cast<MI_U8*>(row0_ptr);
    vtcm_buffer->row1_ptr = reinterpret_cast<MI_U8*>(row1_ptr);
    vtcm_buffer->row2_ptr = reinterpret_cast<MI_U8*>(row2_ptr);
    vtcm_buffer->row3_ptr = reinterpret_cast<MI_U8*>(row3_ptr);

    MVType mv_c_x0_src, mv_c_x1_src;
    MVType mv_c_dst, mv_n0_dst;
    HVX_Vector v_alpha0 = Q6_V_vsplat_R(0xFB80F280);    // (-1152 << 16) + (-3456)
    HVX_Vector v_alpha1 = Q6_V_vsplat_R(0x21807080);    // (8576  << 16) + (28800)
    HVX_Vector v_alpha2 = Q6_V_vsplat_R(0x70802180);    // (28800 << 16) + (8576)
    HVX_Vector v_alpha3 = Q6_V_vsplat_R(0xF280FB80);    // (-3456 << 16) + (-1152)

    MI_S64 *row0_head = row_head;
    MI_S64 *row1_head = row_head + C * 6;
    MI_S64 *row2_head = row_head + C * 12;
    MI_S64 *row3_head = row_head + C * 18;

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;

        row0_head[id0] = row1_head[id0]; row1_head[id0] = row2_head[id0]; row2_head[id0] = row3_head[id0];
        row0_head[id1] = row1_head[id1]; row1_head[id1] = row2_head[id1]; row2_head[id1] = row3_head[id1];
        row0_head[id2] = row1_head[id2]; row1_head[id2] = row2_head[id2]; row2_head[id2] = row3_head[id2];
        row3_head[id0] = src_c[id0] * 36224 + src_c[id1] * (-3456);
        row3_head[id1] = src_c[id0] * 25344 + src_c[id1] * 8576  + src_c[id2] * (-1152);
        row3_head[id2] = src_c[id0] * 7424  + src_c[id1] * 28800 + src_c[id2] * (-3456);

        dst_c[id0]  = SaturateCast<Tp>((row0_head[id0] * (-3456) + row1_head[id0] * 28800 + row2_head[id0] * 8576  + row3_head[id0] * (-1152) + 536870912) >> 30);
        dst_c[id1]  = SaturateCast<Tp>((row0_head[id1] * (-3456) + row1_head[id1] * 28800 + row2_head[id1] * 8576  + row3_head[id1] * (-1152) + 536870912) >> 30);
        dst_c[id2]  = SaturateCast<Tp>((row0_head[id2] * (-3456) + row1_head[id2] * 28800 + row2_head[id2] * 8576  + row3_head[id2] * (-1152) + 536870912) >> 30);
        dst_n0[id0] = SaturateCast<Tp>((row0_head[id0] * (-1152) + row1_head[id0] * 8576  + row2_head[id0] * 28800 + row3_head[id0] * (-3456) + 536870912) >> 30);
        dst_n0[id1] = SaturateCast<Tp>((row0_head[id1] * (-1152) + row1_head[id1] * 8576  + row2_head[id1] * 28800 + row3_head[id1] * (-3456) + 536870912) >> 30);
        dst_n0[id2] = SaturateCast<Tp>((row0_head[id2] * (-1152) + row1_head[id2] * 8576  + row2_head[id2] * 28800 + row3_head[id2] * (-3456) + 536870912) >> 30);
    }

    HVX_VectorPair *v_row0 = (HVX_VectorPair *)(row0_ptr);
    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result;

    auto resize_cu_up2_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_c_dst.val[ch],  -3456, 28800, 8576, -1152);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n0_dst.val[ch], -1152, 8576, 28800, -3456);
    };

    vload(src_c, mv_c_x0_src);

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        vload(src_c + (x + elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);

        mv_c_x0_src  = mv_c_x1_src;
    }

    if (width_align < iwidth - 3)
    {
        vload(src_c + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch], (width_align + elem_counts - iwidth) * 2);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 3 - (elem_counts << 1);

        HVX_VectorPair *v_back_row0 = (HVX_VectorPair *)(row0_back_ptr);
        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c + sx * C, mv_c_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;
            *v_back_row3++ = ws32_c_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch], 6);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;
            *v_back_row3++ = ws32_c_result;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[3 * C + ch] = row1_head[3 * C + ch];
        row0_head[4 * C + ch] = row1_head[4 * C + ch];
        row0_head[5 * C + ch] = row1_head[5 * C + ch];
        row1_head[3 * C + ch] = row2_head[3 * C + ch];
        row1_head[4 * C + ch] = row2_head[4 * C + ch];
        row1_head[5 * C + ch] = row2_head[5 * C + ch];
        row2_head[3 * C + ch] = row3_head[3 * C + ch];
        row2_head[4 * C + ch] = row3_head[4 * C + ch];
        row2_head[5 * C + ch] = row3_head[5 * C + ch];
        row3_head[3 * C + ch] = src_c[id2] * (-3456) + src_c[id1] * 28800 + src_c[id0] * 7424;
        row3_head[4 * C + ch] = src_c[id2] * (-1152) + src_c[id1] * 8576  + src_c[id0] * 25344;
        row3_head[5 * C + ch] = src_c[id1] * (-3456) + src_c[id0] * 36224;

        dst_c[(owidth - 3) * C + ch]  = SaturateCast<Tp>((row0_head[3 * C + ch]  * (-3456) + row1_head[3 * C + ch] * 28800 + row2_head[3 * C + ch] * 8576  + row3_head[3 * C + ch] * (-1152) + 536870912) >> 30);
        dst_c[(owidth - 2) * C + ch]  = SaturateCast<Tp>((row0_head[4 * C + ch]  * (-3456) + row1_head[4 * C + ch] * 28800 + row2_head[4 * C + ch] * 8576  + row3_head[4 * C + ch] * (-1152) + 536870912) >> 30);
        dst_c[(owidth - 1) * C + ch]  = SaturateCast<Tp>((row0_head[5 * C + ch]  * (-3456) + row1_head[5 * C + ch] * 28800 + row2_head[5 * C + ch] * 8576  + row3_head[5 * C + ch] * (-1152) + 536870912) >> 30);
        dst_n0[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[3 * C + ch]  * (-1152) + row1_head[3 * C + ch] * 8576  + row2_head[3 * C + ch] * 28800 + row3_head[3 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n0[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[4 * C + ch]  * (-1152) + row1_head[4 * C + ch] * 8576  + row2_head[4 * C + ch] * 28800 + row3_head[4 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n0[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[5 * C + ch]  * (-1152) + row1_head[5 * C + ch] * 8576  + row2_head[5 * C + ch] * 28800 + row3_head[5 * C + ch] * (-3456) + 536870912) >> 30);
    }
}

// Tp = MI_U8 / MI_S8
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
ResizeCuUpX2BottomBorder(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    Tp *dst_n0 = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1 = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2 = dst_n1 + (ostride / sizeof(Tp));
    Tp *dst_n3 = dst_n2 + (ostride / sizeof(Tp));

    MI_S32 *row0_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S32 *row_head      = reinterpret_cast<MI_S32*>(vtcm_buffer->row_head);
    MI_S32 *row0_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);
    vtcm_buffer->row0_ptr = reinterpret_cast<MI_U8*>(row0_ptr);
    vtcm_buffer->row1_ptr = reinterpret_cast<MI_U8*>(row1_ptr);
    vtcm_buffer->row2_ptr = reinterpret_cast<MI_U8*>(row2_ptr);

    MVType mv_c_x0_src, mv_c_x1_src;
    MVType mv_c_dst, mv_n0_dst, mv_n1_dst, mv_n2_dst, mv_n3_dst;
    HVX_Vector v_alpha0 = Q6_V_vsplat_R(0xFFB8FF28);    // (-72 << 16)  + (-216)
    HVX_Vector v_alpha1 = Q6_V_vsplat_R(0x2180708);     // (536 << 16)  + (1800)
    HVX_Vector v_alpha2 = Q6_V_vsplat_R(0x07080218);    // (1800 << 16) + (536)
    HVX_Vector v_alpha3 = Q6_V_vsplat_R(0xFF28FFB8);    // (-216 << 16) + (-72)

    MI_S32 *row0_head = row_head;
    MI_S32 *row1_head = row_head + C * 6;
    MI_S32 *row2_head = row_head + C * 12;
    MI_S32 *row3_head = row_head + C * 18;

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;

        row0_head[id0] = row1_head[id0];
        row0_head[id1] = row1_head[id1];
        row0_head[id2] = row1_head[id2];
        row1_head[id0] = row2_head[id0];
        row1_head[id1] = row2_head[id1];
        row1_head[id2] = row2_head[id2];
        row2_head[id0] = row3_head[id0];
        row2_head[id1] = row3_head[id1];
        row2_head[id2] = row3_head[id2];
        row3_head[id0] = src_c[id0] * 2264 + src_c[id1] * (-216);
        row3_head[id1] = src_c[id0] * 1584 + src_c[id1] * 536  + src_c[id2] * (-72);
        row3_head[id2] = src_c[id0] * 464  + src_c[id1] * 1800 + src_c[id2] * (-216);

        dst_c[id0]  = SaturateCast<Tp>((row0_head[id0] * (-216) + row1_head[id0] * 1800 + row2_head[id0] * 536  + row3_head[id0] * (-72)  + 2097152) >> 22);
        dst_c[id1]  = SaturateCast<Tp>((row0_head[id1] * (-216) + row1_head[id1] * 1800 + row2_head[id1] * 536  + row3_head[id1] * (-72)  + 2097152) >> 22);
        dst_c[id2]  = SaturateCast<Tp>((row0_head[id2] * (-216) + row1_head[id2] * 1800 + row2_head[id2] * 536  + row3_head[id2] * (-72)  + 2097152) >> 22);
        dst_n0[id0] = SaturateCast<Tp>((row0_head[id0] * (-72)  + row1_head[id0] * 536  + row2_head[id0] * 1800 + row3_head[id0] * (-216) + 2097152) >> 22);
        dst_n0[id1] = SaturateCast<Tp>((row0_head[id1] * (-72)  + row1_head[id1] * 536  + row2_head[id1] * 1800 + row3_head[id1] * (-216) + 2097152) >> 22);
        dst_n0[id2] = SaturateCast<Tp>((row0_head[id2] * (-72)  + row1_head[id2] * 536  + row2_head[id2] * 1800 + row3_head[id2] * (-216) + 2097152) >> 22);
        dst_n1[id0] = SaturateCast<Tp>((row1_head[id0] * (-216) + row2_head[id0] * 1800 + row3_head[id0] * 464  + 2097152) >> 22);
        dst_n1[id1] = SaturateCast<Tp>((row1_head[id1] * (-216) + row2_head[id1] * 1800 + row3_head[id1] * 464  + 2097152) >> 22);
        dst_n1[id2] = SaturateCast<Tp>((row1_head[id2] * (-216) + row2_head[id2] * 1800 + row3_head[id2] * 464  + 2097152) >> 22);
        dst_n2[id0] = SaturateCast<Tp>((row1_head[id0] * (-72)  + row2_head[id0] * 536  + row3_head[id0] * 1584 + 2097152) >> 22);
        dst_n2[id1] = SaturateCast<Tp>((row1_head[id1] * (-72)  + row2_head[id1] * 536  + row3_head[id1] * 1584 + 2097152) >> 22);
        dst_n2[id2] = SaturateCast<Tp>((row1_head[id2] * (-72)  + row2_head[id2] * 536  + row3_head[id2] * 1584 + 2097152) >> 22);
        dst_n3[id0] = SaturateCast<Tp>((row2_head[id0] * (-216) + row3_head[id0] * 2264 + 2097152) >> 22);
        dst_n3[id1] = SaturateCast<Tp>((row2_head[id1] * (-216) + row3_head[id1] * 2264 + 2097152) >> 22);
        dst_n3[id2] = SaturateCast<Tp>((row2_head[id2] * (-216) + row3_head[id2] * 2264 + 2097152) >> 22);
    }

    HVX_VectorPair *v_row0 = (HVX_VectorPair *)(row0_ptr);
    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair ws32_p2_result_l, ws32_p1_result_l, ws32_p0_result_l, ws32_c_result_l;
    HVX_VectorPair ws32_p2_result_h, ws32_p1_result_h, ws32_p0_result_h, ws32_c_result_h;

    auto resize_cu_up2_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_c_dst.val[ch], -216, 1800, 536, -72);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_n0_dst.val[ch], -72, 536, 1800, -216);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h, ws32_c_result_l, ws32_c_result_h,
                                mv_n1_dst.val[ch], -216, 1800, 464);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h, ws32_c_result_l, ws32_c_result_h,
                                mv_n2_dst.val[ch], -72, 536, 1584);
        ResizeCuFastUpVCore<Tp>(ws32_p0_result_l, ws32_p0_result_h, ws32_c_result_l, ws32_c_result_h, mv_n3_dst.val[ch], -216, 2264);
    };

    vload(src_c, mv_c_x0_src);

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        vload(src_c + (x + elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + 3) * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + elem_counts + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + elem_counts + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + elem_counts + 3) * C, mv_n3_dst);

        mv_c_x0_src  = mv_c_x1_src;
    }

    if (width_align < iwidth - 3)
    {
        vload(src_c + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + 3) * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch], width_align + elem_counts - iwidth);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + elem_counts + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + elem_counts + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + elem_counts + 3) * C, mv_n3_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 3 - (elem_counts << 1);

        HVX_VectorPair *v_back_row0 = (HVX_VectorPair *)(row0_back_ptr);
        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);

        vload(src_c + sx * C, mv_c_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);
        vstore(dst_n3 + dx * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch], 3);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + elem_counts) * C, mv_n3_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[3 * C + ch] = row1_head[3 * C + ch];
        row0_head[4 * C + ch] = row1_head[4 * C + ch];
        row0_head[5 * C + ch] = row1_head[5 * C + ch];
        row1_head[3 * C + ch] = row2_head[3 * C + ch];
        row1_head[4 * C + ch] = row2_head[4 * C + ch];
        row1_head[5 * C + ch] = row2_head[5 * C + ch];
        row2_head[3 * C + ch] = row3_head[3 * C + ch];
        row2_head[4 * C + ch] = row3_head[4 * C + ch];
        row2_head[5 * C + ch] = row3_head[5 * C + ch];
        row3_head[3 * C + ch] = src_c[id2] * (-216) + src_c[id1] * 1800 + src_c[id0] * 464;
        row3_head[4 * C + ch] = src_c[id2] * (-72)  + src_c[id1] * 536  + src_c[id0] * 1584;
        row3_head[5 * C + ch] = src_c[id1] * (-216) + src_c[id0] * 2264;

        dst_c[(owidth - 3)  * C + ch] = SaturateCast<Tp>((row0_head[3 * C + ch] * (-216) + row1_head[3 * C + ch] * 1800 + row2_head[3 * C + ch] * 536  + row3_head[3 * C + ch] *  (-72) + 2097152) >> 22);
        dst_c[(owidth - 2)  * C + ch] = SaturateCast<Tp>((row0_head[4 * C + ch] * (-216) + row1_head[4 * C + ch] * 1800 + row2_head[4 * C + ch] * 536  + row3_head[4 * C + ch] *  (-72) + 2097152) >> 22);
        dst_c[(owidth - 1)  * C + ch] = SaturateCast<Tp>((row0_head[5 * C + ch] * (-216) + row1_head[5 * C + ch] * 1800 + row2_head[5 * C + ch] * 536  + row3_head[5 * C + ch] *  (-72) + 2097152) >> 22);
        dst_n0[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[3 * C + ch] * (-72)  + row1_head[3 * C + ch] * 536  + row2_head[3 * C + ch] * 1800 + row3_head[3 * C + ch] * (-216) + 2097152) >> 22);
        dst_n0[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[4 * C + ch] * (-72)  + row1_head[4 * C + ch] * 536  + row2_head[4 * C + ch] * 1800 + row3_head[4 * C + ch] * (-216) + 2097152) >> 22);
        dst_n0[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[5 * C + ch] * (-72)  + row1_head[5 * C + ch] * 536  + row2_head[5 * C + ch] * 1800 + row3_head[5 * C + ch] * (-216) + 2097152) >> 22);
        dst_n1[(owidth - 3) * C + ch] = SaturateCast<Tp>((row1_head[3 * C + ch] * (-216) + row2_head[3 * C + ch] * 1800 + row3_head[3 * C + ch] * 464  + 2097152) >> 22);
        dst_n1[(owidth - 2) * C + ch] = SaturateCast<Tp>((row1_head[4 * C + ch] * (-216) + row2_head[4 * C + ch] * 1800 + row3_head[4 * C + ch] * 464  + 2097152) >> 22);
        dst_n1[(owidth - 1) * C + ch] = SaturateCast<Tp>((row1_head[5 * C + ch] * (-216) + row2_head[5 * C + ch] * 1800 + row3_head[5 * C + ch] * 464  + 2097152) >> 22);
        dst_n2[(owidth - 3) * C + ch] = SaturateCast<Tp>((row1_head[3 * C + ch] * (-72)  + row2_head[3 * C + ch] * 536  + row3_head[3 * C + ch] * 1584 + 2097152) >> 22);
        dst_n2[(owidth - 2) * C + ch] = SaturateCast<Tp>((row1_head[4 * C + ch] * (-72)  + row2_head[4 * C + ch] * 536  + row3_head[4 * C + ch] * 1584 + 2097152) >> 22);
        dst_n2[(owidth - 1) * C + ch] = SaturateCast<Tp>((row1_head[5 * C + ch] * (-72)  + row2_head[5 * C + ch] * 536  + row3_head[5 * C + ch] * 1584 + 2097152) >> 22);
        dst_n3[(owidth - 3) * C + ch] = SaturateCast<Tp>((row2_head[3 * C + ch] * (-216) + row3_head[3 * C + ch] * 2264 + 2097152) >> 22);
        dst_n3[(owidth - 2) * C + ch] = SaturateCast<Tp>((row2_head[4 * C + ch] * (-216) + row3_head[4 * C + ch] * 2264 + 2097152) >> 22);
        dst_n3[(owidth - 1) * C + ch] = SaturateCast<Tp>((row2_head[5 * C + ch] * (-216) + row3_head[5 * C + ch] * 2264 + 2097152) >> 22);
    }
}

// Tp = MI_U16 / MI_S16
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
ResizeCuUpX2BottomBorder(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    Tp *dst_n0 = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1 = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2 = dst_n1 + (ostride / sizeof(Tp));
    Tp *dst_n3 = dst_n2 + (ostride / sizeof(Tp));

    MI_S32 *row0_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S64 *row_head      = reinterpret_cast<MI_S64*>(vtcm_buffer->row_head);
    MI_S32 *row0_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);
    vtcm_buffer->row0_ptr = reinterpret_cast<MI_U8*>(row0_ptr);
    vtcm_buffer->row1_ptr = reinterpret_cast<MI_U8*>(row1_ptr);
    vtcm_buffer->row2_ptr = reinterpret_cast<MI_U8*>(row2_ptr);

    MVType mv_c_x0_src, mv_c_x1_src;
    MVType mv_c_dst, mv_n0_dst, mv_n1_dst, mv_n2_dst, mv_n3_dst;
    HVX_Vector v_alpha0 = Q6_V_vsplat_R(0xFB80F280);    // (-1152 << 16) + (-3456)
    HVX_Vector v_alpha1 = Q6_V_vsplat_R(0x21807080);    // (8576  << 16) + (28800)
    HVX_Vector v_alpha2 = Q6_V_vsplat_R(0x70802180);    // (28800 << 16) + (8576)
    HVX_Vector v_alpha3 = Q6_V_vsplat_R(0xF280FB80);    // (-3456 << 16) + (-1152)

    MI_S64 *row0_head = row_head;
    MI_S64 *row1_head = row_head + C * 6;
    MI_S64 *row2_head = row_head + C * 12;
    MI_S64 *row3_head = row_head + C * 18;

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;

        row0_head[id0] = row1_head[id0];
        row0_head[id1] = row1_head[id1];
        row0_head[id2] = row1_head[id2];
        row1_head[id0] = row2_head[id0];
        row1_head[id1] = row2_head[id1];
        row1_head[id2] = row2_head[id2];
        row2_head[id0] = row3_head[id0];
        row2_head[id1] = row3_head[id1];
        row2_head[id2] = row3_head[id2];
        row3_head[id0] = src_c[id0] * 36224 + src_c[id1] * (-3456);
        row3_head[id1] = src_c[id0] * 25344 + src_c[id1] * 8576  + src_c[id2] * (-1152);
        row3_head[id2] = src_c[id0] * 7424  + src_c[id1] * 28800 + src_c[id2] * (-3456);

        dst_c[id0]  = SaturateCast<Tp>((row0_head[id0] * (-3456) + row1_head[id0] * 28800 + row2_head[id0] * 8576  + row3_head[id0] * (-1152) + 536870912) >> 30);
        dst_c[id1]  = SaturateCast<Tp>((row0_head[id1] * (-3456) + row1_head[id1] * 28800 + row2_head[id1] * 8576  + row3_head[id1] * (-1152) + 536870912) >> 30);
        dst_c[id2]  = SaturateCast<Tp>((row0_head[id2] * (-3456) + row1_head[id2] * 28800 + row2_head[id2] * 8576  + row3_head[id2] * (-1152) + 536870912) >> 30);
        dst_n0[id0] = SaturateCast<Tp>((row0_head[id0] * (-1152) + row1_head[id0] * 8576  + row2_head[id0] * 28800 + row3_head[id0] * (-3456) + 536870912) >> 30);
        dst_n0[id1] = SaturateCast<Tp>((row0_head[id1] * (-1152) + row1_head[id1] * 8576  + row2_head[id1] * 28800 + row3_head[id1] * (-3456) + 536870912) >> 30);
        dst_n0[id2] = SaturateCast<Tp>((row0_head[id2] * (-1152) + row1_head[id2] * 8576  + row2_head[id2] * 28800 + row3_head[id2] * (-3456) + 536870912) >> 30);
        dst_n1[id0] = SaturateCast<Tp>((row1_head[id0] * (-3456) + row2_head[id0] * 28800 + row3_head[id0] * 7424  + 536870912) >> 30);
        dst_n1[id1] = SaturateCast<Tp>((row1_head[id1] * (-3456) + row2_head[id1] * 28800 + row3_head[id1] * 7424  + 536870912) >> 30);
        dst_n1[id2] = SaturateCast<Tp>((row1_head[id2] * (-3456) + row2_head[id2] * 28800 + row3_head[id2] * 7424  + 536870912) >> 30);
        dst_n2[id0] = SaturateCast<Tp>((row1_head[id0] * (-1152) + row2_head[id0] * 8576  + row3_head[id0] * 25344 + 536870912) >> 30);
        dst_n2[id1] = SaturateCast<Tp>((row1_head[id1] * (-1152) + row2_head[id1] * 8576  + row3_head[id1] * 25344 + 536870912) >> 30);
        dst_n2[id2] = SaturateCast<Tp>((row1_head[id2] * (-1152) + row2_head[id2] * 8576  + row3_head[id2] * 25344 + 536870912) >> 30);
        dst_n3[id0] = SaturateCast<Tp>((row2_head[id0] * (-3456) + row3_head[id0] * 36224 + 536870912) >> 30);
        dst_n3[id1] = SaturateCast<Tp>((row2_head[id1] * (-3456) + row3_head[id1] * 36224 + 536870912) >> 30);
        dst_n3[id2] = SaturateCast<Tp>((row2_head[id2] * (-3456) + row3_head[id2] * 36224 + 536870912) >> 30);
    }

    HVX_VectorPair *v_row0 = (HVX_VectorPair *)(row0_ptr);
    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result;

    auto resize_cu_up2_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_c_dst.val[ch],  -3456, 28800, 8576, -1152);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n0_dst.val[ch], -1152, 8576, 28800, -3456);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n1_dst.val[ch], -3456, 28800, 7424);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n2_dst.val[ch], -1152, 8576, 25344);
        ResizeCuFastUpVCore<Tp>(ws32_p0_result, ws32_c_result,  mv_n3_dst.val[ch], -3456, 36224);
    };

    vload(src_c, mv_c_x0_src);

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        vload(src_c + (x + elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + 3) * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + elem_counts + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + elem_counts + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + elem_counts + 3) * C, mv_n3_dst);

        mv_c_x0_src  = mv_c_x1_src;
    }

    if (width_align < iwidth - 3)
    {
        vload(src_c + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + 3) * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch], (width_align + elem_counts - iwidth) * 2);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + ((x << 1) + elem_counts + 3) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 1) + elem_counts + 3) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 1) + elem_counts + 3) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 1) + elem_counts + 3) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 1) + elem_counts + 3) * C, mv_n3_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 3 - (elem_counts << 1);

        HVX_VectorPair *v_back_row0 = (HVX_VectorPair *)(row0_back_ptr);
        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);

        vload(src_c + sx * C, mv_c_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);
        vstore(dst_n3 + dx * C, mv_n3_dst);

        for (ch = 0; ch < C; ch++)
        {
            HVX_Vector v_c_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch], 6);

            // row
            ResizeCuUpX2HCore<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;

            // col
            resize_cu_up2_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + elem_counts) * C, mv_n3_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[3 * C + ch] = row1_head[3 * C + ch];
        row0_head[4 * C + ch] = row1_head[4 * C + ch];
        row0_head[5 * C + ch] = row1_head[5 * C + ch];
        row1_head[3 * C + ch] = row2_head[3 * C + ch];
        row1_head[4 * C + ch] = row2_head[4 * C + ch];
        row1_head[5 * C + ch] = row2_head[5 * C + ch];
        row2_head[3 * C + ch] = row3_head[3 * C + ch];
        row2_head[4 * C + ch] = row3_head[4 * C + ch];
        row2_head[5 * C + ch] = row3_head[5 * C + ch];
        row3_head[3 * C + ch] = src_c[id2] * (-3456) + src_c[id1] * 28800 + src_c[id0] * 7424;
        row3_head[4 * C + ch] = src_c[id2] * (-1152) + src_c[id1] * 8576  + src_c[id0] * 25344;
        row3_head[5 * C + ch] = src_c[id1] * (-3456) + src_c[id0] * 36224;

        dst_c[(owidth - 3)  * C + ch] = SaturateCast<Tp>((row0_head[3 * C + ch] * (-3456) + row1_head[3 * C + ch] * 28800 + row2_head[3 * C + ch] * 8576  + row3_head[3 * C + ch] * (-1152) + 536870912) >> 30);
        dst_c[(owidth - 2)  * C + ch] = SaturateCast<Tp>((row0_head[4 * C + ch] * (-3456) + row1_head[4 * C + ch] * 28800 + row2_head[4 * C + ch] * 8576  + row3_head[4 * C + ch] * (-1152) + 536870912) >> 30);
        dst_c[(owidth - 1)  * C + ch] = SaturateCast<Tp>((row0_head[5 * C + ch] * (-3456) + row1_head[5 * C + ch] * 28800 + row2_head[5 * C + ch] * 8576  + row3_head[5 * C + ch] * (-1152) + 536870912) >> 30);
        dst_n0[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[3 * C + ch] * (-1152) + row1_head[3 * C + ch] * 8576  + row2_head[3 * C + ch] * 28800 + row3_head[3 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n0[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[4 * C + ch] * (-1152) + row1_head[4 * C + ch] * 8576  + row2_head[4 * C + ch] * 28800 + row3_head[4 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n0[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[5 * C + ch] * (-1152) + row1_head[5 * C + ch] * 8576  + row2_head[5 * C + ch] * 28800 + row3_head[5 * C + ch] * (-3456) + 536870912) >> 30);
        dst_n1[(owidth - 3) * C + ch] = SaturateCast<Tp>((row1_head[3 * C + ch] * (-3456) + row2_head[3 * C + ch] * 28800 + row3_head[3 * C + ch] * 7424  + 536870912) >> 30);
        dst_n1[(owidth - 2) * C + ch] = SaturateCast<Tp>((row1_head[4 * C + ch] * (-3456) + row2_head[4 * C + ch] * 28800 + row3_head[4 * C + ch] * 7424  + 536870912) >> 30);
        dst_n1[(owidth - 1) * C + ch] = SaturateCast<Tp>((row1_head[5 * C + ch] * (-3456) + row2_head[5 * C + ch] * 28800 + row3_head[5 * C + ch] * 7424  + 536870912) >> 30);
        dst_n2[(owidth - 3) * C + ch] = SaturateCast<Tp>((row1_head[3 * C + ch] * (-1152) + row2_head[3 * C + ch] * 8576  + row3_head[3 * C + ch] * 25344 + 536870912) >> 30);
        dst_n2[(owidth - 2) * C + ch] = SaturateCast<Tp>((row1_head[4 * C + ch] * (-1152) + row2_head[4 * C + ch] * 8576  + row3_head[4 * C + ch] * 25344 + 536870912) >> 30);
        dst_n2[(owidth - 1) * C + ch] = SaturateCast<Tp>((row1_head[5 * C + ch] * (-1152) + row2_head[5 * C + ch] * 8576  + row3_head[5 * C + ch] * 25344 + 536870912) >> 30);
        dst_n3[(owidth - 3) * C + ch] = SaturateCast<Tp>((row2_head[3 * C + ch] * (-3456) + row3_head[3 * C + ch] * 36224 + 536870912) >> 30);
        dst_n3[(owidth - 2) * C + ch] = SaturateCast<Tp>((row2_head[4 * C + ch] * (-3456) + row3_head[4 * C + ch] * 36224 + 536870912) >> 30);
        dst_n3[(owidth - 1) * C + ch] = SaturateCast<Tp>((row2_head[5 * C + ch] * (-3456) + row3_head[5 * C + ch] * 36224 + 536870912) >> 30);
    }
}

// Tp = MI_U8 / MI_S8
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
ResizeCuUpX4ZeroRow(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 istride, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));
    const Tp *src_n2 = src_n1 + (istride / sizeof(Tp));
    Tp *dst_n0       = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1       = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2       = dst_n1 + (ostride / sizeof(Tp));
    Tp *dst_n3       = dst_n2 + (ostride / sizeof(Tp));
    Tp *dst_n4       = dst_n3 + (ostride / sizeof(Tp));
    Tp *dst_n5       = dst_n4 + (ostride / sizeof(Tp));
    Tp *dst_n6       = dst_n5 + (ostride / sizeof(Tp));
    Tp *dst_n7       = dst_n6 + (ostride / sizeof(Tp));
    Tp *dst_n8       = dst_n7 + (ostride / sizeof(Tp));

    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S32 *row_head      = reinterpret_cast<MI_S32*>(vtcm_buffer->row_head);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src, mv_n2_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src, mv_n2_x1_src;
    MVType mv_c_dst,  mv_n0_dst, mv_n1_dst, mv_n2_dst, mv_n3_dst;
    MVType mv_n4_dst, mv_n5_dst, mv_n6_dst, mv_n7_dst, mv_n8_dst;
    HVX_Vector v_alpha0 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFFEBFF79), Q6_V_vsplat_R(0xFF1FFF6D), -4));    // (-21 << 16) + (-135)  (-225 << 16) + (-147)
    HVX_Vector v_alpha1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x00EB0369), Q6_V_vsplat_R(0x05FF07BD), -4));    // (235 << 16) + (873 )  (1535 << 16) + (1981)
    HVX_Vector v_alpha2 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x07BD05FF), Q6_V_vsplat_R(0x036900EB), -4));    // (1981 << 16) + (1535)  (873 << 16) + (235)
    HVX_Vector v_alpha3 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFF6DFF1F), Q6_V_vsplat_R(0xFF79FFEB), -4));    // (-147 << 16) + (-225)  (-135 << 16) + (-21)

    MI_S32 *row1_head = row_head + C * 12;
    MI_S32 *row2_head = row_head + C * 24;
    MI_S32 *row3_head = row_head + C * 36;

    vload(src_c , mv_c_x0_src);
    vload(src_n0, mv_n0_x0_src);
    vload(src_n1, mv_n1_x0_src);
    vload(src_n2, mv_n2_x0_src);

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 row0_head[6];
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;
        MI_S32 id3 = 3 * C + ch;
        MI_S32 id4 = 4 * C + ch;
        MI_S32 id5 = 5 * C + ch;

        row0_head[0]   = src_c[id0]  * 2273 + src_c[id1]  * (-225);
        row1_head[id0] = src_n0[id0] * 2273 + src_n0[id1] * (-225);
        row2_head[id0] = src_n1[id0] * 2273 + src_n1[id1] * (-225);
        row3_head[id0] = src_n2[id0] * 2273 + src_n2[id1] * (-225);
        row0_head[1]   = src_c[id0]  * 2195 + src_c[id1]  * (-147);
        row1_head[id1] = src_n0[id0] * 2195 + src_n0[id1] * (-147);
        row2_head[id1] = src_n1[id0] * 2195 + src_n1[id1] * (-147);
        row3_head[id1] = src_n2[id0] * 2195 + src_n2[id1] * (-147);
        row0_head[2]   = src_c[id0]  * 1834 + src_c[id1]  * 235  + src_c[id2]  * (-21);
        row1_head[id2] = src_n0[id0] * 1834 + src_n0[id1] * 235  + src_n0[id2] * (-21);
        row2_head[id2] = src_n1[id0] * 1834 + src_n1[id1] * 235  + src_n1[id2] * (-21);
        row3_head[id2] = src_n2[id0] * 1834 + src_n2[id1] * 235  + src_n2[id2] * (-21);
        row0_head[3]   = src_c[id0]  * 1310 + src_c[id1]  * 873  + src_c[id2]  * (-135);
        row1_head[id3] = src_n0[id0] * 1310 + src_n0[id1] * 873  + src_n0[id2] * (-135);
        row2_head[id3] = src_n1[id0] * 1310 + src_n1[id1] * 873  + src_n1[id2] * (-135);
        row3_head[id3] = src_n2[id0] * 1310 + src_n2[id1] * 873  + src_n2[id2] * (-135);
        row0_head[4]   = src_c[id0]  * 738  + src_c[id1]  * 1535 + src_c[id2]  * (-225);
        row1_head[id4] = src_n0[id0] * 738  + src_n0[id1] * 1535 + src_n0[id2] * (-225);
        row2_head[id4] = src_n1[id0] * 738  + src_n1[id1] * 1535 + src_n1[id2] * (-225);
        row3_head[id4] = src_n2[id0] * 738  + src_n2[id1] * 1535 + src_n2[id2] * (-225);
        row0_head[5]   = src_c[id0]  * 214  + src_c[id1]  * 1981 + src_c[id2]  * (-147);
        row1_head[id5] = src_n0[id0] * 214  + src_n0[id1] * 1981 + src_n0[id2] * (-147);
        row2_head[id5] = src_n1[id0] * 214  + src_n1[id1] * 1981 + src_n1[id2] * (-147);
        row3_head[id5] = src_n2[id0] * 214  + src_n2[id1] * 1981 + src_n2[id2] * (-147);

        Tp *dst = dst_c;
        #pragma unroll(10)
        for (MI_S32 y = 0; y < 10; y++)
        {
            dst[id0] = SaturateCast<Tp>((row0_head[0] * BETA_S16[y][0] + row1_head[id0] * BETA_S16[y][1] + row2_head[id0] * BETA_S16[y][2] + row3_head[id0] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id1] = SaturateCast<Tp>((row0_head[1] * BETA_S16[y][0] + row1_head[id1] * BETA_S16[y][1] + row2_head[id1] * BETA_S16[y][2] + row3_head[id1] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id2] = SaturateCast<Tp>((row0_head[2] * BETA_S16[y][0] + row1_head[id2] * BETA_S16[y][1] + row2_head[id2] * BETA_S16[y][2] + row3_head[id2] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id3] = SaturateCast<Tp>((row0_head[3] * BETA_S16[y][0] + row1_head[id3] * BETA_S16[y][1] + row2_head[id3] * BETA_S16[y][2] + row3_head[id3] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id4] = SaturateCast<Tp>((row0_head[4] * BETA_S16[y][0] + row1_head[id4] * BETA_S16[y][1] + row2_head[id4] * BETA_S16[y][2] + row3_head[id4] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id5] = SaturateCast<Tp>((row0_head[5] * BETA_S16[y][0] + row1_head[id5] * BETA_S16[y][1] + row2_head[id5] * BETA_S16[y][2] + row3_head[id5] * BETA_S16[y][3] + 2097152) >> 22);

            dst += (ostride / sizeof(Tp));
        }

        HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch],  mv_c_x0_src.val[ch],  -1);
        HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x0_src.val[ch], mv_n0_x0_src.val[ch], -1);
        HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x0_src.val[ch], mv_n1_x0_src.val[ch], -1);
        HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x0_src.val[ch], mv_n2_x0_src.val[ch], -1);
        mv_c_x0_src.val[ch]      = Q6_V_lo_W(w_c_twin);
        mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
        mv_n0_x0_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
        mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
        mv_n1_x0_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
        mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
        mv_n2_x0_src.val[ch]     = Q6_V_lo_W(w_n2_twin);
        mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);
    }

    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_c_result_l, ws32_n0_result_l, ws32_n1_result_l, ws32_n2_result_l;
    HVX_VectorPair ws32_c_result_h, ws32_n0_result_h, ws32_n1_result_h, ws32_n2_result_h;

    auto resize_cu_up4_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h,
                                mv_c_dst.val[ch], 2273, -225);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h,
                                mv_n0_dst.val[ch], 2195, -147);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                mv_n1_dst.val[ch], 1834, 235, -21);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                mv_n2_dst.val[ch], 1310, 873, -135);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                mv_n3_dst.val[ch], 738, 1535, -225);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                mv_n4_dst.val[ch], 214, 1981, -147);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_n5_dst.val[ch], -147, 1981, 235, -21);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_n6_dst.val[ch], -225, 1535, 873, -135);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_n7_dst.val[ch], -135, 873, 1535, -225);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_n8_dst.val[ch], -21, 235, 1981, -147);
    };

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n8_dst);

        vload(src_c  + (x + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (x + elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (x + elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (x + elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -1);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -1);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -1);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -1);
            mv_c_x1_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_lo_W(w_n2_twin);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();

            mv_c_x0_src.val[ch]  = mv_c_x1_src.val[ch];
            mv_n0_x0_src.val[ch] = mv_n0_x1_src.val[ch];
            mv_n1_x0_src.val[ch] = mv_n1_x1_src.val[ch];
            mv_n2_x0_src.val[ch] = mv_n2_x1_src.val[ch];

            mv_c_x1_src.val[ch]  = Q6_V_hi_W(w_c_twin);
            mv_n0_x1_src.val[ch] = Q6_V_hi_W(w_n0_twin);
            mv_n1_x1_src.val[ch] = Q6_V_hi_W(w_n1_twin);
            mv_n2_x1_src.val[ch] = Q6_V_hi_W(w_n2_twin);
        }

        vstore(dst_c  + ((x << 2) + elem_counts * 3 + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n8_dst);
    }

    if (width_align < iwidth - 3)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n8_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -1);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -1);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -1);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -1);
            mv_c_x1_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_lo_W(w_n2_twin);

            HVX_Vector v_c_border  = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  (width_align + elem_counts - iwidth) << 1);
            HVX_Vector v_n0_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n0_x1_src.val[ch], (width_align + elem_counts - iwidth) << 1);
            HVX_Vector v_n1_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n1_x1_src.val[ch], (width_align + elem_counts - iwidth) << 1);
            HVX_Vector v_n2_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n2_x1_src.val[ch], (width_align + elem_counts - iwidth) << 1);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts * 3) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n8_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 6 - (elem_counts << 2);

        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_n1 + sx * C, mv_n1_x0_src);
        vload(src_n2 + sx * C, mv_n2_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch],  mv_c_x0_src.val[ch],  -1);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x0_src.val[ch], mv_n0_x0_src.val[ch], -1);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x0_src.val[ch], mv_n1_x0_src.val[ch], -1);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x0_src.val[ch], mv_n2_x0_src.val[ch], -1);
            mv_c_x0_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
            mv_n0_x0_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
            mv_n1_x0_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
            mv_n2_x0_src.val[ch]     = Q6_V_lo_W(w_n2_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);

            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);
        vstore(dst_n3 + dx * C, mv_n3_dst);
        vstore(dst_n4 + dx * C, mv_n4_dst);
        vstore(dst_n5 + dx * C, mv_n5_dst);
        vstore(dst_n6 + dx * C, mv_n6_dst);
        vstore(dst_n7 + dx * C, mv_n7_dst);
        vstore(dst_n8 + dx * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + elem_counts) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + elem_counts) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + elem_counts) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + elem_counts) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + elem_counts) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + elem_counts) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts << 1)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts << 1)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts << 1)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts << 1)) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + (elem_counts << 1)) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + (elem_counts << 1)) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + (elem_counts << 1)) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + (elem_counts << 1)) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + (elem_counts << 1)) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + (elem_counts << 1)) * C, mv_n8_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -1);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -1);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -1);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -1);
            mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);

            HVX_Vector v_c_border  = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch],  6);
            HVX_Vector v_n0_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n0_x1_src.val[ch], 6);
            HVX_Vector v_n1_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n1_x1_src.val[ch], 6);
            HVX_Vector v_n2_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n2_x1_src.val[ch], 6);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts * 3)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts * 3)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts * 3)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts * 3)) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + (elem_counts * 3)) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + (elem_counts * 3)) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + (elem_counts * 3)) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + (elem_counts * 3)) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + (elem_counts * 3)) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + (elem_counts * 3)) * C, mv_n8_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 row0_head[6];
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[0]           = src_c[id2]  * (-147) + src_c[id1]  * 1981 + src_c[id0]  * 214;
        row1_head[6 * C + ch]  = src_n0[id2] * (-147) + src_n0[id1] * 1981 + src_n0[id0] * 214;
        row2_head[6 * C + ch]  = src_n1[id2] * (-147) + src_n1[id1] * 1981 + src_n1[id0] * 214;
        row3_head[6 * C + ch]  = src_n2[id2] * (-147) + src_n2[id1] * 1981 + src_n2[id0] * 214;
        row0_head[1]           = src_c[id2]  * (-225) + src_c[id1]  * 1535 + src_c[id0]  * 738;
        row1_head[7 * C + ch]  = src_n0[id2] * (-225) + src_n0[id1] * 1535 + src_n0[id0] * 738;
        row2_head[7 * C + ch]  = src_n1[id2] * (-225) + src_n1[id1] * 1535 + src_n1[id0] * 738;
        row3_head[7 * C + ch]  = src_n2[id2] * (-225) + src_n2[id1] * 1535 + src_n2[id0] * 738;
        row0_head[2]           = src_c[id2]  * (-135) + src_c[id1]  * 873  + src_c[id0]  * 1310;
        row1_head[8 * C + ch]  = src_n0[id2] * (-135) + src_n0[id1] * 873  + src_n0[id0] * 1310;
        row2_head[8 * C + ch]  = src_n1[id2] * (-135) + src_n1[id1] * 873  + src_n1[id0] * 1310;
        row3_head[8 * C + ch]  = src_n2[id2] * (-135) + src_n2[id1] * 873  + src_n2[id0] * 1310;
        row0_head[3]           = src_c[id2]  * (-21)  + src_c[id1]  * 235  + src_c[id0]  * 1834;
        row1_head[9 * C + ch]  = src_n0[id2] * (-21)  + src_n0[id1] * 235  + src_n0[id0] * 1834;
        row2_head[9 * C + ch]  = src_n1[id2] * (-21)  + src_n1[id1] * 235  + src_n1[id0] * 1834;
        row3_head[9 * C + ch]  = src_n2[id2] * (-21)  + src_n2[id1] * 235  + src_n2[id0] * 1834;
        row0_head[4]           = src_c[id1]  * (-147) + src_c[id0]  * 2195;
        row1_head[10 * C + ch] = src_n0[id1] * (-147) + src_n0[id0] * 2195;
        row2_head[10 * C + ch] = src_n1[id1] * (-147) + src_n1[id0] * 2195;
        row3_head[10 * C + ch] = src_n2[id1] * (-147) + src_n2[id0] * 2195;
        row0_head[5]           = src_c[id1]  * (-225) + src_c[id0]  * 2273;
        row1_head[11 * C + ch] = src_n0[id1] * (-225) + src_n0[id0] * 2273;
        row2_head[11 * C + ch] = src_n1[id1] * (-225) + src_n1[id0] * 2273;
        row3_head[11 * C + ch] = src_n2[id1] * (-225) + src_n2[id0] * 2273;

        Tp *dst = dst_c;
        #pragma unroll(10)
        for (MI_S32 y = 0; y < 10; y++)
        {
            dst[(owidth - 6) * C + ch] = SaturateCast<Tp>((row0_head[0] * BETA_S16[y][0] + row1_head[6 * C + ch]  * BETA_S16[y][1] + row2_head[6 * C + ch]  * BETA_S16[y][2] + row3_head[6 * C + ch]  * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 5) * C + ch] = SaturateCast<Tp>((row0_head[1] * BETA_S16[y][0] + row1_head[7 * C + ch]  * BETA_S16[y][1] + row2_head[7 * C + ch]  * BETA_S16[y][2] + row3_head[7 * C + ch]  * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 4) * C + ch] = SaturateCast<Tp>((row0_head[2] * BETA_S16[y][0] + row1_head[8 * C + ch]  * BETA_S16[y][1] + row2_head[8 * C + ch]  * BETA_S16[y][2] + row3_head[8 * C + ch]  * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[3] * BETA_S16[y][0] + row1_head[9 * C + ch]  * BETA_S16[y][1] + row2_head[9 * C + ch]  * BETA_S16[y][2] + row3_head[9 * C + ch]  * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[4] * BETA_S16[y][0] + row1_head[10 * C + ch] * BETA_S16[y][1] + row2_head[10 * C + ch] * BETA_S16[y][2] + row3_head[10 * C + ch] * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[5] * BETA_S16[y][0] + row1_head[11 * C + ch] * BETA_S16[y][1] + row2_head[11 * C + ch] * BETA_S16[y][2] + row3_head[11 * C + ch] * BETA_S16[y][3] + 2097152) >> 22);

            dst += (ostride / sizeof(Tp));
        }
    }
}

// Tp = MI_U8 / MI_S8
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
ResizeCuUpX4UpBorder(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 istride, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));
    const Tp *src_n2 = src_n1 + (istride / sizeof(Tp));
    Tp *dst_n0       = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1       = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2       = dst_n1 + (ostride / sizeof(Tp));

    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S32 *row_head      = reinterpret_cast<MI_S32*>(vtcm_buffer->row_head);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src, mv_n2_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src, mv_n2_x1_src;
    MVType mv_c_dst,  mv_n0_dst, mv_n1_dst, mv_n2_dst;
    HVX_Vector v_alpha0 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFFEBFF79), Q6_V_vsplat_R(0xFF1FFF6D), -4));    // (-21 << 16) + (-135)  (-225 << 16) + (-147)
    HVX_Vector v_alpha1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x00EB0369), Q6_V_vsplat_R(0x05FF07BD), -4));    // (235 << 16) + (873 )  (1535 << 16) + (1981)
    HVX_Vector v_alpha2 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x07BD05FF), Q6_V_vsplat_R(0x036900EB), -4));    // (1981 << 16) + (1535)  (873 << 16) + (235)
    HVX_Vector v_alpha3 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFF6DFF1F), Q6_V_vsplat_R(0xFF79FFEB), -4));    // (-147 << 16) + (-225)  (-135 << 16) + (-21)

    MI_S32 *row1_head = row_head + C * 12;
    MI_S32 *row2_head = row_head + C * 24;
    MI_S32 *row3_head = row_head + C * 36;

    vload(src_c , mv_c_x0_src);
    vload(src_n0, mv_n0_x0_src);
    vload(src_n1, mv_n1_x0_src);
    vload(src_n2, mv_n2_x0_src);

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 row0_head[6];
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;
        MI_S32 id3 = 3 * C + ch;
        MI_S32 id4 = 4 * C + ch;
        MI_S32 id5 = 5 * C + ch;

        row0_head[0]   = src_c[id0]  * 2273 + src_c[id1]  * (-225);
        row1_head[id0] = src_n0[id0] * 2273 + src_n0[id1] * (-225);
        row2_head[id0] = src_n1[id0] * 2273 + src_n1[id1] * (-225);
        row3_head[id0] = src_n2[id0] * 2273 + src_n2[id1] * (-225);
        row0_head[1]   = src_c[id0]  * 2195 + src_c[id1]  * (-147);
        row1_head[id1] = src_n0[id0] * 2195 + src_n0[id1] * (-147);
        row2_head[id1] = src_n1[id0] * 2195 + src_n1[id1] * (-147);
        row3_head[id1] = src_n2[id0] * 2195 + src_n2[id1] * (-147);
        row0_head[2]   = src_c[id0]  * 1834 + src_c[id1]  * 235  + src_c[id2]  * (-21);
        row1_head[id2] = src_n0[id0] * 1834 + src_n0[id1] * 235  + src_n0[id2] * (-21);
        row2_head[id2] = src_n1[id0] * 1834 + src_n1[id1] * 235  + src_n1[id2] * (-21);
        row3_head[id2] = src_n2[id0] * 1834 + src_n2[id1] * 235  + src_n2[id2] * (-21);
        row0_head[3]   = src_c[id0]  * 1310 + src_c[id1]  * 873  + src_c[id2]  * (-135);
        row1_head[id3] = src_n0[id0] * 1310 + src_n0[id1] * 873  + src_n0[id2] * (-135);
        row2_head[id3] = src_n1[id0] * 1310 + src_n1[id1] * 873  + src_n1[id2] * (-135);
        row3_head[id3] = src_n2[id0] * 1310 + src_n2[id1] * 873  + src_n2[id2] * (-135);
        row0_head[4]   = src_c[id0]  * 738  + src_c[id1]  * 1535 + src_c[id2]  * (-225);
        row1_head[id4] = src_n0[id0] * 738  + src_n0[id1] * 1535 + src_n0[id2] * (-225);
        row2_head[id4] = src_n1[id0] * 738  + src_n1[id1] * 1535 + src_n1[id2] * (-225);
        row3_head[id4] = src_n2[id0] * 738  + src_n2[id1] * 1535 + src_n2[id2] * (-225);
        row0_head[5]   = src_c[id0]  * 214  + src_c[id1]  * 1981 + src_c[id2]  * (-147);
        row1_head[id5] = src_n0[id0] * 214  + src_n0[id1] * 1981 + src_n0[id2] * (-147);
        row2_head[id5] = src_n1[id0] * 214  + src_n1[id1] * 1981 + src_n1[id2] * (-147);
        row3_head[id5] = src_n2[id0] * 214  + src_n2[id1] * 1981 + src_n2[id2] * (-147);

        Tp *dst = dst_c;
        #pragma unroll(4)
        for (MI_S32 y = 6; y < 10; y++)
        {
            dst[id0] = SaturateCast<Tp>((row0_head[0] * BETA_S16[y][0] + row1_head[id0] * BETA_S16[y][1] + row2_head[id0] * BETA_S16[y][2] + row3_head[id0] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id1] = SaturateCast<Tp>((row0_head[1] * BETA_S16[y][0] + row1_head[id1] * BETA_S16[y][1] + row2_head[id1] * BETA_S16[y][2] + row3_head[id1] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id2] = SaturateCast<Tp>((row0_head[2] * BETA_S16[y][0] + row1_head[id2] * BETA_S16[y][1] + row2_head[id2] * BETA_S16[y][2] + row3_head[id2] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id3] = SaturateCast<Tp>((row0_head[3] * BETA_S16[y][0] + row1_head[id3] * BETA_S16[y][1] + row2_head[id3] * BETA_S16[y][2] + row3_head[id3] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id4] = SaturateCast<Tp>((row0_head[4] * BETA_S16[y][0] + row1_head[id4] * BETA_S16[y][1] + row2_head[id4] * BETA_S16[y][2] + row3_head[id4] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id5] = SaturateCast<Tp>((row0_head[5] * BETA_S16[y][0] + row1_head[id5] * BETA_S16[y][1] + row2_head[id5] * BETA_S16[y][2] + row3_head[id5] * BETA_S16[y][3] + 2097152) >> 22);

            dst += (ostride / sizeof(Tp));
        }

        HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch],  mv_c_x0_src.val[ch],  -1);
        HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x0_src.val[ch], mv_n0_x0_src.val[ch], -1);
        HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x0_src.val[ch], mv_n1_x0_src.val[ch], -1);
        HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x0_src.val[ch], mv_n2_x0_src.val[ch], -1);
        mv_c_x0_src.val[ch]      = Q6_V_lo_W(w_c_twin);
        mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
        mv_n0_x0_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
        mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
        mv_n1_x0_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
        mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
        mv_n2_x0_src.val[ch]     = Q6_V_lo_W(w_n2_twin);
        mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);
    }

    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_c_result_l, ws32_n0_result_l, ws32_n1_result_l, ws32_n2_result_l;
    HVX_VectorPair ws32_c_result_h, ws32_n0_result_h, ws32_n1_result_h, ws32_n2_result_h;

    auto resize_cu_up4_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_c_dst.val[ch], -147, 1981, 235, -21);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_n0_dst.val[ch], -225, 1535, 873, -135);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_n1_dst.val[ch], -135, 873, 1535, -225);
        ResizeCuFastUpVCore<Tp>(ws32_c_result_l, ws32_c_result_h, ws32_n0_result_l, ws32_n0_result_h, ws32_n1_result_l, ws32_n1_result_h,
                                ws32_n2_result_l, ws32_n2_result_h, mv_n2_dst.val[ch], -21, 235, 1981, -147);
    };

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);

        vload(src_c  + (x + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (x + elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (x + elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (x + elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -1);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -1);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -1);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -1);
            mv_c_x1_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_lo_W(w_n2_twin);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();

            mv_c_x0_src.val[ch]  = mv_c_x1_src.val[ch];
            mv_n0_x0_src.val[ch] = mv_n0_x1_src.val[ch];
            mv_n1_x0_src.val[ch] = mv_n1_x1_src.val[ch];
            mv_n2_x0_src.val[ch] = mv_n2_x1_src.val[ch];

            mv_c_x1_src.val[ch]  = Q6_V_hi_W(w_c_twin);
            mv_n0_x1_src.val[ch] = Q6_V_hi_W(w_n0_twin);
            mv_n1_x1_src.val[ch] = Q6_V_hi_W(w_n1_twin);
            mv_n2_x1_src.val[ch] = Q6_V_hi_W(w_n2_twin);
        }

        vstore(dst_c  + ((x << 2) + elem_counts * 3 + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n2_dst);
    }

    if (width_align < iwidth - 3)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -1);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -1);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -1);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -1);
            mv_c_x1_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_lo_W(w_n2_twin);

            HVX_Vector v_c_border  = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  (width_align + elem_counts - iwidth) << 1);
            HVX_Vector v_n0_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n0_x1_src.val[ch], (width_align + elem_counts - iwidth) << 1);
            HVX_Vector v_n1_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n1_x1_src.val[ch], (width_align + elem_counts - iwidth) << 1);
            HVX_Vector v_n2_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n2_x1_src.val[ch], (width_align + elem_counts - iwidth) << 1);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result_l;
            *v_row1++ = ws32_n0_result_h;
            *v_row2++ = ws32_n1_result_l;
            *v_row2++ = ws32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts * 3) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n2_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 6 - (elem_counts << 2);

        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_n1 + sx * C, mv_n1_x0_src);
        vload(src_n2 + sx * C, mv_n2_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch],  mv_c_x0_src.val[ch],  -1);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x0_src.val[ch], mv_n0_x0_src.val[ch], -1);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x0_src.val[ch], mv_n1_x0_src.val[ch], -1);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x0_src.val[ch], mv_n2_x0_src.val[ch], -1);
            mv_c_x0_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
            mv_n0_x0_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
            mv_n1_x0_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
            mv_n2_x0_src.val[ch]     = Q6_V_lo_W(w_n2_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);

            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts << 1)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts << 1)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts << 1)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts << 1)) * C, mv_n2_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -1);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -1);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -1);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -1);
            mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);

            HVX_Vector v_c_border  = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch],  6);
            HVX_Vector v_n0_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n0_x1_src.val[ch], 6);
            HVX_Vector v_n1_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n1_x1_src.val[ch], 6);
            HVX_Vector v_n2_border = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_n2_x1_src.val[ch], 6);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result_l, ws32_n0_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result_l, ws32_n1_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result_l, ws32_n2_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result_l;
            *v_back_row1++ = ws32_n0_result_h;
            *v_back_row2++ = ws32_n1_result_l;
            *v_back_row2++ = ws32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts * 3)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts * 3)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts * 3)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts * 3)) * C, mv_n2_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 row0_head[6];
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[0]           = src_c[id2]  * (-147) + src_c[id1]  * 1981 + src_c[id0]  * 214;
        row1_head[6 * C + ch]  = src_n0[id2] * (-147) + src_n0[id1] * 1981 + src_n0[id0] * 214;
        row2_head[6 * C + ch]  = src_n1[id2] * (-147) + src_n1[id1] * 1981 + src_n1[id0] * 214;
        row3_head[6 * C + ch]  = src_n2[id2] * (-147) + src_n2[id1] * 1981 + src_n2[id0] * 214;
        row0_head[1]           = src_c[id2]  * (-225) + src_c[id1]  * 1535 + src_c[id0]  * 738;
        row1_head[7 * C + ch]  = src_n0[id2] * (-225) + src_n0[id1] * 1535 + src_n0[id0] * 738;
        row2_head[7 * C + ch]  = src_n1[id2] * (-225) + src_n1[id1] * 1535 + src_n1[id0] * 738;
        row3_head[7 * C + ch]  = src_n2[id2] * (-225) + src_n2[id1] * 1535 + src_n2[id0] * 738;
        row0_head[2]           = src_c[id2]  * (-135) + src_c[id1]  * 873  + src_c[id0]  * 1310;
        row1_head[8 * C + ch]  = src_n0[id2] * (-135) + src_n0[id1] * 873  + src_n0[id0] * 1310;
        row2_head[8 * C + ch]  = src_n1[id2] * (-135) + src_n1[id1] * 873  + src_n1[id0] * 1310;
        row3_head[8 * C + ch]  = src_n2[id2] * (-135) + src_n2[id1] * 873  + src_n2[id0] * 1310;
        row0_head[3]           = src_c[id2]  * (-21)  + src_c[id1]  * 235  + src_c[id0]  * 1834;
        row1_head[9 * C + ch]  = src_n0[id2] * (-21)  + src_n0[id1] * 235  + src_n0[id0] * 1834;
        row2_head[9 * C + ch]  = src_n1[id2] * (-21)  + src_n1[id1] * 235  + src_n1[id0] * 1834;
        row3_head[9 * C + ch]  = src_n2[id2] * (-21)  + src_n2[id1] * 235  + src_n2[id0] * 1834;
        row0_head[4]           = src_c[id1]  * (-147) + src_c[id0]  * 2195;
        row1_head[10 * C + ch] = src_n0[id1] * (-147) + src_n0[id0] * 2195;
        row2_head[10 * C + ch] = src_n1[id1] * (-147) + src_n1[id0] * 2195;
        row3_head[10 * C + ch] = src_n2[id1] * (-147) + src_n2[id0] * 2195;
        row0_head[5]           = src_c[id1]  * (-225) + src_c[id0]  * 2273;
        row1_head[11 * C + ch] = src_n0[id1] * (-225) + src_n0[id0] * 2273;
        row2_head[11 * C + ch] = src_n1[id1] * (-225) + src_n1[id0] * 2273;
        row3_head[11 * C + ch] = src_n2[id1] * (-225) + src_n2[id0] * 2273;

        Tp *dst = dst_c;
        #pragma unroll(4)
        for (MI_S32 y = 6; y < 10; y++)
        {
            dst[(owidth - 6) * C + ch] = SaturateCast<Tp>((row0_head[0] * BETA_S16[y][0] + row1_head[6 * C + ch]  * BETA_S16[y][1] + row2_head[6 * C + ch]  * BETA_S16[y][2] + row3_head[6 * C + ch]  * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 5) * C + ch] = SaturateCast<Tp>((row0_head[1] * BETA_S16[y][0] + row1_head[7 * C + ch]  * BETA_S16[y][1] + row2_head[7 * C + ch]  * BETA_S16[y][2] + row3_head[7 * C + ch]  * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 4) * C + ch] = SaturateCast<Tp>((row0_head[2] * BETA_S16[y][0] + row1_head[8 * C + ch]  * BETA_S16[y][1] + row2_head[8 * C + ch]  * BETA_S16[y][2] + row3_head[8 * C + ch]  * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[3] * BETA_S16[y][0] + row1_head[9 * C + ch]  * BETA_S16[y][1] + row2_head[9 * C + ch]  * BETA_S16[y][2] + row3_head[9 * C + ch]  * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[4] * BETA_S16[y][0] + row1_head[10 * C + ch] * BETA_S16[y][1] + row2_head[10 * C + ch] * BETA_S16[y][2] + row3_head[10 * C + ch] * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[5] * BETA_S16[y][0] + row1_head[11 * C + ch] * BETA_S16[y][1] + row2_head[11 * C + ch] * BETA_S16[y][2] + row3_head[11 * C + ch] * BETA_S16[y][3] + 2097152) >> 22);

            dst += (ostride / sizeof(Tp));
        }
    }
}

// Tp = MI_U8 / MI_S8
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
ResizeCuUpX4Row(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    Tp *dst_n0 = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1 = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2 = dst_n1 + (ostride / sizeof(Tp));

    MI_S32 *row0_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row0_ptr);
    MI_S32 *row_head      = reinterpret_cast<MI_S32*>(vtcm_buffer->row_head);
    MI_S32 *row0_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row0_ptr + row_pre_size);
    vtcm_buffer->row0_ptr = reinterpret_cast<MI_U8*>(row0_ptr);
    vtcm_buffer->row1_ptr = reinterpret_cast<MI_U8*>(row1_ptr);
    vtcm_buffer->row2_ptr = reinterpret_cast<MI_U8*>(row2_ptr);
    vtcm_buffer->row3_ptr = reinterpret_cast<MI_U8*>(row3_ptr);

    MVType mv_c_x0_src, mv_c_x1_src;
    MVType mv_c_dst, mv_n0_dst, mv_n1_dst, mv_n2_dst;
    HVX_Vector v_alpha0 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFFEBFF79), Q6_V_vsplat_R(0xFF1FFF6D), -4));    // (-21 << 16) + (-135)  (-225 << 16) + (-147)
    HVX_Vector v_alpha1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x00EB0369), Q6_V_vsplat_R(0x05FF07BD), -4));    // (235 << 16) + (873 )  (1535 << 16) + (1981)
    HVX_Vector v_alpha2 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x07BD05FF), Q6_V_vsplat_R(0x036900EB), -4));    // (1981 << 16) + (1535)  (873 << 16) + (235)
    HVX_Vector v_alpha3 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFF6DFF1F), Q6_V_vsplat_R(0xFF79FFEB), -4));    // (-147 << 16) + (-225)  (-135 << 16) + (-21)

    MI_S32 *row0_head = row_head;
    MI_S32 *row1_head = row_head + C * 12;
    MI_S32 *row2_head = row_head + C * 24;
    MI_S32 *row3_head = row_head + C * 36;

    vload(src_c , mv_c_x0_src);

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;
        MI_S32 id3 = 3 * C + ch;
        MI_S32 id4 = 4 * C + ch;
        MI_S32 id5 = 5 * C + ch;

        row0_head[id0] = row1_head[id0]; row1_head[id0] = row2_head[id0]; row2_head[id0] = row3_head[id0];
        row0_head[id1] = row1_head[id1]; row1_head[id1] = row2_head[id1]; row2_head[id1] = row3_head[id1];
        row0_head[id2] = row1_head[id2]; row1_head[id2] = row2_head[id2]; row2_head[id2] = row3_head[id2];
        row0_head[id3] = row1_head[id3]; row1_head[id3] = row2_head[id3]; row2_head[id3] = row3_head[id3];
        row0_head[id4] = row1_head[id4]; row1_head[id4] = row2_head[id4]; row2_head[id4] = row3_head[id4];
        row0_head[id5] = row1_head[id5]; row1_head[id5] = row2_head[id5]; row2_head[id5] = row3_head[id5];
        row3_head[id0] = src_c[id0] * 2273 + src_c[id1] * (-225);
        row3_head[id1] = src_c[id0] * 2195 + src_c[id1] * (-147);
        row3_head[id2] = src_c[id0] * 1834 + src_c[id1] * 235  + src_c[id2] * (-21);
        row3_head[id3] = src_c[id0] * 1310 + src_c[id1] * 873  + src_c[id2] * (-135);
        row3_head[id4] = src_c[id0] * 738  + src_c[id1] * 1535 + src_c[id2] * (-225);
        row3_head[id5] = src_c[id0] * 214  + src_c[id1] * 1981 + src_c[id2] * (-147);

        Tp *dst = dst_c;
        #pragma unroll(4)
        for (MI_S32 y = 6; y < 10; y++)
        {
            dst[id0] = SaturateCast<Tp>((row0_head[id0] * BETA_S16[y][0] + row1_head[id0] * BETA_S16[y][1] + row2_head[id0] * BETA_S16[y][2] + row3_head[id0] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id1] = SaturateCast<Tp>((row0_head[id1] * BETA_S16[y][0] + row1_head[id1] * BETA_S16[y][1] + row2_head[id1] * BETA_S16[y][2] + row3_head[id1] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id2] = SaturateCast<Tp>((row0_head[id2] * BETA_S16[y][0] + row1_head[id2] * BETA_S16[y][1] + row2_head[id2] * BETA_S16[y][2] + row3_head[id2] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id3] = SaturateCast<Tp>((row0_head[id3] * BETA_S16[y][0] + row1_head[id3] * BETA_S16[y][1] + row2_head[id3] * BETA_S16[y][2] + row3_head[id3] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id4] = SaturateCast<Tp>((row0_head[id4] * BETA_S16[y][0] + row1_head[id4] * BETA_S16[y][1] + row2_head[id4] * BETA_S16[y][2] + row3_head[id4] * BETA_S16[y][3] + 2097152) >> 22);
            dst[id5] = SaturateCast<Tp>((row0_head[id5] * BETA_S16[y][0] + row1_head[id5] * BETA_S16[y][1] + row2_head[id5] * BETA_S16[y][2] + row3_head[id5] * BETA_S16[y][3] + 2097152) >> 22);

            dst += (ostride / sizeof(Tp));
        }

        HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch], mv_c_x0_src.val[ch], -1);
        mv_c_x0_src.val[ch]     = Q6_V_lo_W(w_c_twin);
        mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);
    }

    HVX_VectorPair *v_row0 = (HVX_VectorPair *)(row0_ptr);
    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_p2_result_l, ws32_p1_result_l, ws32_p0_result_l, ws32_c_result_l;
    HVX_VectorPair ws32_p2_result_h, ws32_p1_result_h, ws32_p0_result_h, ws32_c_result_h;

    auto resize_cu_up4_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_c_dst.val[ch], -147, 1981, 235, -21);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_n0_dst.val[ch], -225, 1535, 873, -135);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_n1_dst.val[ch], -135, 873, 1535, -225);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_n2_dst.val[ch], -21, 235, 1981, -147);
    };

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);

        vload(src_c + (x + elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch], mv_c_x1_src.val[ch], -1);
            mv_c_x1_src.val[ch]     = Q6_V_lo_W(w_c_twin);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();

            mv_c_x0_src.val[ch] = mv_c_x1_src.val[ch];
            mv_c_x1_src.val[ch] = Q6_V_hi_W(w_c_twin);
        }

        vstore(dst_c  + ((x << 2) + elem_counts * 3 + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n2_dst);
    }

    if (width_align < iwidth - 3)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch], mv_c_x1_src.val[ch], -1);
            mv_c_x1_src.val[ch]     = Q6_V_lo_W(w_c_twin);
            HVX_Vector v_c_border   = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  (width_align + elem_counts - iwidth) << 1);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;
            *v_row3++ = ws32_c_result_l;
            *v_row3++ = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts * 3) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n2_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 6 - (elem_counts << 2);

        HVX_VectorPair *v_back_row0 = (HVX_VectorPair *)(row0_back_ptr);
        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c + sx * C, mv_c_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch], mv_c_x0_src.val[ch], -1);
            mv_c_x0_src.val[ch]     = Q6_V_lo_W(w_c_twin);
            mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);

            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;
            *v_back_row3++   = ws32_c_result_l;
            *v_back_row3++   = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;
            *v_back_row3++   = ws32_c_result_l;
            *v_back_row3++   = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;
            *v_back_row3++   = ws32_c_result_l;
            *v_back_row3++   = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts << 1)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts << 1)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts << 1)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts << 1)) * C, mv_n2_dst);

        vload(src_c + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch], mv_c_x1_src.val[ch], -1);
            mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);
            HVX_Vector v_c_border   = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch],  6);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;
            *v_back_row3++   = ws32_c_result_l;
            *v_back_row3++   = ws32_c_result_h;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts * 3)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts * 3)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts * 3)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts * 3)) * C, mv_n2_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[6  * C + ch] = row1_head[6  * C + ch]; row1_head[6  * C + ch] = row2_head[6  * C + ch];
        row0_head[7  * C + ch] = row1_head[7  * C + ch]; row1_head[7  * C + ch] = row2_head[7  * C + ch];
        row0_head[8  * C + ch] = row1_head[8  * C + ch]; row1_head[8  * C + ch] = row2_head[8  * C + ch];
        row0_head[9  * C + ch] = row1_head[9  * C + ch]; row1_head[9  * C + ch] = row2_head[9  * C + ch];
        row0_head[10 * C + ch] = row1_head[10 * C + ch]; row1_head[10 * C + ch] = row2_head[10 * C + ch];
        row0_head[11 * C + ch] = row1_head[11 * C + ch]; row1_head[11 * C + ch] = row2_head[11 * C + ch];
        row2_head[6  * C + ch] = row3_head[6  * C + ch];
        row2_head[7  * C + ch] = row3_head[7  * C + ch];
        row2_head[8  * C + ch] = row3_head[8  * C + ch];
        row2_head[9  * C + ch] = row3_head[9  * C + ch];
        row2_head[10 * C + ch] = row3_head[10 * C + ch];
        row2_head[11 * C + ch] = row3_head[11 * C + ch];
        row3_head[6  * C + ch] = src_c[id2] * (-147) + src_c[id1] * 1981 + src_c[id0] * 214;
        row3_head[7  * C + ch] = src_c[id2] * (-225) + src_c[id1] * 1535 + src_c[id0] * 738;
        row3_head[8  * C + ch] = src_c[id2] * (-135) + src_c[id1] * 873  + src_c[id0] * 1310;
        row3_head[9  * C + ch] = src_c[id2] * (-21)  + src_c[id1] * 235  + src_c[id0] * 1834;
        row3_head[10 * C + ch] = src_c[id1] * (-147) + src_c[id0] * 2195;
        row3_head[11 * C + ch] = src_c[id1] * (-225) + src_c[id0] * 2273;

        Tp *dst = dst_c;
        #pragma unroll(4)
        for (MI_S32 y = 6; y < 10; y++)
        {
            dst[(owidth - 6) * C + ch] = SaturateCast<Tp>((row0_head[6  * C + ch] * BETA_S16[y][0] + row1_head[6  * C + ch] * BETA_S16[y][1] + row2_head[6  * C + ch] * BETA_S16[y][2] + row3_head[6  * C + ch] * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 5) * C + ch] = SaturateCast<Tp>((row0_head[7  * C + ch] * BETA_S16[y][0] + row1_head[7  * C + ch] * BETA_S16[y][1] + row2_head[7  * C + ch] * BETA_S16[y][2] + row3_head[7  * C + ch] * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 4) * C + ch] = SaturateCast<Tp>((row0_head[8  * C + ch] * BETA_S16[y][0] + row1_head[8  * C + ch] * BETA_S16[y][1] + row2_head[8  * C + ch] * BETA_S16[y][2] + row3_head[8  * C + ch] * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[9  * C + ch] * BETA_S16[y][0] + row1_head[9  * C + ch] * BETA_S16[y][1] + row2_head[9  * C + ch] * BETA_S16[y][2] + row3_head[9  * C + ch] * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[10 * C + ch] * BETA_S16[y][0] + row1_head[10 * C + ch] * BETA_S16[y][1] + row2_head[10 * C + ch] * BETA_S16[y][2] + row3_head[10 * C + ch] * BETA_S16[y][3] + 2097152) >> 22);
            dst[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[11 * C + ch] * BETA_S16[y][0] + row1_head[11 * C + ch] * BETA_S16[y][1] + row2_head[11 * C + ch] * BETA_S16[y][2] + row3_head[11 * C + ch] * BETA_S16[y][3] + 2097152) >> 22);

            dst += (ostride / sizeof(Tp));
        }
    }
}

// Tp = MI_U8 / MI_S8
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
ResizeCuUpX4BottomBorder(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    Tp *dst_n0 = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1 = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2 = dst_n1 + (ostride / sizeof(Tp));
    Tp *dst_n3 = dst_n2 + (ostride / sizeof(Tp));
    Tp *dst_n4 = dst_n3 + (ostride / sizeof(Tp));
    Tp *dst_n5 = dst_n4 + (ostride / sizeof(Tp));
    Tp *dst_n6 = dst_n5 + (ostride / sizeof(Tp));
    Tp *dst_n7 = dst_n6 + (ostride / sizeof(Tp));
    Tp *dst_n8 = dst_n7 + (ostride / sizeof(Tp));

    MI_S32 *row0_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S32 *row_head      = reinterpret_cast<MI_S32*>(vtcm_buffer->row_head);
    MI_S32 *row0_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);
    vtcm_buffer->row0_ptr = reinterpret_cast<MI_U8*>(row0_ptr);
    vtcm_buffer->row1_ptr = reinterpret_cast<MI_U8*>(row1_ptr);
    vtcm_buffer->row2_ptr = reinterpret_cast<MI_U8*>(row2_ptr);

    MVType mv_c_x0_src, mv_c_x1_src;
    MVType mv_c_dst,  mv_n0_dst, mv_n1_dst, mv_n2_dst, mv_n3_dst;
    MVType mv_n4_dst, mv_n5_dst, mv_n6_dst, mv_n7_dst, mv_n8_dst;
    HVX_Vector v_alpha0 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFFEBFF79), Q6_V_vsplat_R(0xFF1FFF6D), -4));    // (-21 << 16) + (-135)  (-225 << 16) + (-147)
    HVX_Vector v_alpha1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x00EB0369), Q6_V_vsplat_R(0x05FF07BD), -4));    // (235 << 16) + (873 )  (1535 << 16) + (1981)
    HVX_Vector v_alpha2 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x07BD05FF), Q6_V_vsplat_R(0x036900EB), -4));    // (1981 << 16) + (1535)  (873 << 16) + (235)
    HVX_Vector v_alpha3 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFF6DFF1F), Q6_V_vsplat_R(0xFF79FFEB), -4));    // (-147 << 16) + (-225)  (-135 << 16) + (-21)

    MI_S32 *row0_head = row_head;
    MI_S32 *row1_head = row_head + C * 12;
    MI_S32 *row2_head = row_head + C * 24;
    MI_S32 *row3_head = row_head + C * 36;

    vload(src_c , mv_c_x0_src);

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;
        MI_S32 id3 = 3 * C + ch;
        MI_S32 id4 = 4 * C + ch;
        MI_S32 id5 = 5 * C + ch;

        row0_head[id0] = row1_head[id0]; row1_head[id0] = row2_head[id0]; row2_head[id0] = row3_head[id0];
        row0_head[id1] = row1_head[id1]; row1_head[id1] = row2_head[id1]; row2_head[id1] = row3_head[id1];
        row0_head[id2] = row1_head[id2]; row1_head[id2] = row2_head[id2]; row2_head[id2] = row3_head[id2];
        row0_head[id3] = row1_head[id3]; row1_head[id3] = row2_head[id3]; row2_head[id3] = row3_head[id3];
        row0_head[id4] = row1_head[id4]; row1_head[id4] = row2_head[id4]; row2_head[id4] = row3_head[id4];
        row0_head[id5] = row1_head[id5]; row1_head[id5] = row2_head[id5]; row2_head[id5] = row3_head[id5];
        row3_head[id0] = src_c[id0] * 2273 + src_c[id1] * (-225);
        row3_head[id1] = src_c[id0] * 2195 + src_c[id1] * (-147);
        row3_head[id2] = src_c[id0] * 1834 + src_c[id1] * 235  + src_c[id2] * (-21);
        row3_head[id3] = src_c[id0] * 1310 + src_c[id1] * 873  + src_c[id2] * (-135);
        row3_head[id4] = src_c[id0] * 738  + src_c[id1] * 1535 + src_c[id2] * (-225);
        row3_head[id5] = src_c[id0] * 214  + src_c[id1] * 1981 + src_c[id2] * (-147);

        Tp *dst = dst_c;
        #pragma unroll(10)
        for (MI_S32 y = 0; y < 10; y++)
        {
            dst[id0] = SaturateCast<Tp>((row0_head[id0] * BETA_S16[9 - y][3] + row1_head[id0] * BETA_S16[9 - y][2] + row2_head[id0] * BETA_S16[9 - y][1] + row3_head[id0] * BETA_S16[9 - y][0] + 2097152) >> 22);
            dst[id1] = SaturateCast<Tp>((row0_head[id1] * BETA_S16[9 - y][3] + row1_head[id1] * BETA_S16[9 - y][2] + row2_head[id1] * BETA_S16[9 - y][1] + row3_head[id1] * BETA_S16[9 - y][0] + 2097152) >> 22);
            dst[id2] = SaturateCast<Tp>((row0_head[id2] * BETA_S16[9 - y][3] + row1_head[id2] * BETA_S16[9 - y][2] + row2_head[id2] * BETA_S16[9 - y][1] + row3_head[id2] * BETA_S16[9 - y][0] + 2097152) >> 22);
            dst[id3] = SaturateCast<Tp>((row0_head[id3] * BETA_S16[9 - y][3] + row1_head[id3] * BETA_S16[9 - y][2] + row2_head[id3] * BETA_S16[9 - y][1] + row3_head[id3] * BETA_S16[9 - y][0] + 2097152) >> 22);
            dst[id4] = SaturateCast<Tp>((row0_head[id4] * BETA_S16[9 - y][3] + row1_head[id4] * BETA_S16[9 - y][2] + row2_head[id4] * BETA_S16[9 - y][1] + row3_head[id4] * BETA_S16[9 - y][0] + 2097152) >> 22);
            dst[id5] = SaturateCast<Tp>((row0_head[id5] * BETA_S16[9 - y][3] + row1_head[id5] * BETA_S16[9 - y][2] + row2_head[id5] * BETA_S16[9 - y][1] + row3_head[id5] * BETA_S16[9 - y][0] + 2097152) >> 22);

            dst += (ostride / sizeof(Tp));
        }

        HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch], mv_c_x0_src.val[ch], -1);
        mv_c_x0_src.val[ch]     = Q6_V_lo_W(w_c_twin);
        mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);
    }

    HVX_VectorPair *v_row0 = (HVX_VectorPair *)(row0_ptr);
    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair ws32_p2_result_l, ws32_p1_result_l, ws32_p0_result_l, ws32_c_result_l;
    HVX_VectorPair ws32_p2_result_h, ws32_p1_result_h, ws32_p0_result_h, ws32_c_result_h;

    auto resize_cu_up4_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_c_dst.val[ch], -147, 1981, 235, -21);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_n0_dst.val[ch], -225, 1535, 873, -135);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_n1_dst.val[ch], -135, 873, 1535, -225);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result_l, ws32_p2_result_h, ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h,
                                ws32_c_result_l, ws32_c_result_h, mv_n2_dst.val[ch], -21, 235, 1981, -147);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h, ws32_c_result_l, ws32_c_result_h,
                                mv_n3_dst.val[ch], -147, 1981, 214);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h, ws32_c_result_l, ws32_c_result_h,
                                mv_n4_dst.val[ch], -225, 1535, 738);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h, ws32_c_result_l, ws32_c_result_h,
                                mv_n5_dst.val[ch], -135, 873, 1310);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h, ws32_c_result_l, ws32_c_result_h,
                                mv_n6_dst.val[ch], -21, 235, 1834);
        ResizeCuFastUpVCore<Tp>(ws32_p0_result_l, ws32_p0_result_h, ws32_c_result_l, ws32_c_result_h,
                                mv_n7_dst.val[ch], -147, 2195);
        ResizeCuFastUpVCore<Tp>(ws32_p0_result_l, ws32_p0_result_h, ws32_c_result_l, ws32_c_result_h,
                                mv_n8_dst.val[ch], -225, 2273);
    };

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n8_dst);

        vload(src_c + (x + elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -1);
            mv_c_x1_src.val[ch]     = Q6_V_lo_W(w_c_twin);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up4_vfunc();

            mv_c_x0_src.val[ch] = mv_c_x1_src.val[ch];
            mv_c_x1_src.val[ch] = Q6_V_hi_W(w_c_twin);
        }

        vstore(dst_c  + ((x << 2) + elem_counts * 3 + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n8_dst);
    }

    if (width_align < iwidth - 3)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n8_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -1);
            mv_c_x1_src.val[ch]     = Q6_V_lo_W(w_c_twin);
            HVX_Vector v_c_border   = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  (width_align + elem_counts - iwidth) << 1);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_row0++;
            ws32_p2_result_h = *v_row0++;
            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts * 3) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n8_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 6 - (elem_counts << 2);

        HVX_VectorPair *v_back_row0 = (HVX_VectorPair *)(row0_back_ptr);
        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);

        vload(src_c + sx * C, mv_c_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch],  mv_c_x0_src.val[ch],  -1);
            mv_c_x0_src.val[ch]     = Q6_V_lo_W(w_c_twin);
            mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);

            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);
        vstore(dst_n3 + dx * C, mv_n3_dst);
        vstore(dst_n4 + dx * C, mv_n4_dst);
        vstore(dst_n5 + dx * C, mv_n5_dst);
        vstore(dst_n6 + dx * C, mv_n6_dst);
        vstore(dst_n7 + dx * C, mv_n7_dst);
        vstore(dst_n8 + dx * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + elem_counts) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + elem_counts) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + elem_counts) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + elem_counts) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + elem_counts) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + elem_counts) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result_l, ws32_c_result_h, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts << 1)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts << 1)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts << 1)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts << 1)) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + (elem_counts << 1)) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + (elem_counts << 1)) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + (elem_counts << 1)) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + (elem_counts << 1)) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + (elem_counts << 1)) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + (elem_counts << 1)) * C, mv_n8_dst);

        vload(src_c + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -1);
            mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);
            HVX_Vector v_c_border   = Q6_V_vlalign_VVI(Q6_V_vzero(), mv_c_x1_src.val[ch],  6);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result_l,  ws32_c_result_h,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result_l = *v_back_row0++;
            ws32_p2_result_h = *v_back_row0++;
            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts * 3)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts * 3)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts * 3)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts * 3)) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + (elem_counts * 3)) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + (elem_counts * 3)) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + (elem_counts * 3)) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + (elem_counts * 3)) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + (elem_counts * 3)) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + (elem_counts * 3)) * C, mv_n8_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[6  * C + ch] = row1_head[6  * C + ch]; row1_head[6  * C + ch] = row2_head[6  * C + ch];
        row0_head[7  * C + ch] = row1_head[7  * C + ch]; row1_head[7  * C + ch] = row2_head[7  * C + ch];
        row0_head[8  * C + ch] = row1_head[8  * C + ch]; row1_head[8  * C + ch] = row2_head[8  * C + ch];
        row0_head[9  * C + ch] = row1_head[9  * C + ch]; row1_head[9  * C + ch] = row2_head[9  * C + ch];
        row0_head[10 * C + ch] = row1_head[10 * C + ch]; row1_head[10 * C + ch] = row2_head[10 * C + ch];
        row0_head[11 * C + ch] = row1_head[11 * C + ch]; row1_head[11 * C + ch] = row2_head[11 * C + ch];
        row2_head[6  * C + ch] = row3_head[6  * C + ch];
        row2_head[7  * C + ch] = row3_head[7  * C + ch];
        row2_head[8  * C + ch] = row3_head[8  * C + ch];
        row2_head[9  * C + ch] = row3_head[9  * C + ch];
        row2_head[10 * C + ch] = row3_head[10 * C + ch];
        row2_head[11 * C + ch] = row3_head[11 * C + ch];
        row3_head[6  * C + ch] = src_c[id2] * (-147) + src_c[id1] * 1981 + src_c[id0] * 214;
        row3_head[7  * C + ch] = src_c[id2] * (-225) + src_c[id1] * 1535 + src_c[id0] * 738;
        row3_head[8  * C + ch] = src_c[id2] * (-135) + src_c[id1] * 873  + src_c[id0] * 1310;
        row3_head[9  * C + ch] = src_c[id2] * (-21)  + src_c[id1] * 235  + src_c[id0] * 1834;
        row3_head[10 * C + ch] = src_c[id1] * (-147) + src_c[id0] * 2195;
        row3_head[11 * C + ch] = src_c[id1] * (-225) + src_c[id0] * 2273;

        Tp *dst = dst_c;
        #pragma unroll(10)
        for (MI_S32 y = 0; y < 10; y++)
        {
            dst[(owidth - 6) * C + ch] = SaturateCast<Tp>((row0_head[6  * C + ch] * BETA_S16[9 - y][3] + row1_head[6  * C + ch] * BETA_S16[9 - y][2] + row2_head[6  * C + ch] * BETA_S16[9 - y][1] + row3_head[6  * C + ch] * BETA_S16[9 - y][0] + 2097152) >> 22);
            dst[(owidth - 5) * C + ch] = SaturateCast<Tp>((row0_head[7  * C + ch] * BETA_S16[9 - y][3] + row1_head[7  * C + ch] * BETA_S16[9 - y][2] + row2_head[7  * C + ch] * BETA_S16[9 - y][1] + row3_head[7  * C + ch] * BETA_S16[9 - y][0] + 2097152) >> 22);
            dst[(owidth - 4) * C + ch] = SaturateCast<Tp>((row0_head[8  * C + ch] * BETA_S16[9 - y][3] + row1_head[8  * C + ch] * BETA_S16[9 - y][2] + row2_head[8  * C + ch] * BETA_S16[9 - y][1] + row3_head[8  * C + ch] * BETA_S16[9 - y][0] + 2097152) >> 22);
            dst[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[9  * C + ch] * BETA_S16[9 - y][3] + row1_head[9  * C + ch] * BETA_S16[9 - y][2] + row2_head[9  * C + ch] * BETA_S16[9 - y][1] + row3_head[9  * C + ch] * BETA_S16[9 - y][0] + 2097152) >> 22);
            dst[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[10 * C + ch] * BETA_S16[9 - y][3] + row1_head[10 * C + ch] * BETA_S16[9 - y][2] + row2_head[10 * C + ch] * BETA_S16[9 - y][1] + row3_head[10 * C + ch] * BETA_S16[9 - y][0] + 2097152) >> 22);
            dst[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[11 * C + ch] * BETA_S16[9 - y][3] + row1_head[11 * C + ch] * BETA_S16[9 - y][2] + row2_head[11 * C + ch] * BETA_S16[9 - y][1] + row3_head[11 * C + ch] * BETA_S16[9 - y][0] + 2097152) >> 22);

            dst += (ostride / sizeof(Tp));
        }
    }
}

// Tp = MI_U16 / MI_S16
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
ResizeCuUpX4ZeroRow(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 istride, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));
    const Tp *src_n2 = src_n1 + (istride / sizeof(Tp));
    Tp *dst_n0       = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1       = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2       = dst_n1 + (ostride / sizeof(Tp));
    Tp *dst_n3       = dst_n2 + (ostride / sizeof(Tp));
    Tp *dst_n4       = dst_n3 + (ostride / sizeof(Tp));
    Tp *dst_n5       = dst_n4 + (ostride / sizeof(Tp));
    Tp *dst_n6       = dst_n5 + (ostride / sizeof(Tp));
    Tp *dst_n7       = dst_n6 + (ostride / sizeof(Tp));
    Tp *dst_n8       = dst_n7 + (ostride / sizeof(Tp));

    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S64 *row_head      = reinterpret_cast<MI_S64*>(vtcm_buffer->row_head);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src, mv_n2_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src, mv_n2_x1_src;
    MVType mv_c_dst,  mv_n0_dst, mv_n1_dst, mv_n2_dst, mv_n3_dst;
    MVType mv_n4_dst, mv_n5_dst, mv_n6_dst, mv_n7_dst, mv_n8_dst;
    HVX_Vector v_alpha0 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFEB0F790), Q6_V_vsplat_R(0xF1F0F6D0), -4));    // (-336  << 16) + (-2160) (-3600 << 16) + (-2352)
    HVX_Vector v_alpha1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x0EB03690), Q6_V_vsplat_R(0x5FF07BD0), -4));    // (3760  << 16) + (13968) (24560 << 16) + (31696)
    HVX_Vector v_alpha2 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x7BD05FF0), Q6_V_vsplat_R(0x36900EB0), -4));    // (31696 << 16) + (24560) (13968 << 16) + (3760)
    HVX_Vector v_alpha3 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xF6D0F1F0), Q6_V_vsplat_R(0xF790FEB0), -4));    // (-2385 << 16) + (-3600) (-2160 << 16) + (-336)

    MI_S64 *row1_head = row_head + C * 12;
    MI_S64 *row2_head = row_head + C * 24;
    MI_S64 *row3_head = row_head + C * 36;

    vload(src_c , mv_c_x0_src);
    vload(src_n0, mv_n0_x0_src);
    vload(src_n1, mv_n1_x0_src);
    vload(src_n2, mv_n2_x0_src);

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S64 row0_head[6];
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;
        MI_S32 id3 = 3 * C + ch;
        MI_S32 id4 = 4 * C + ch;
        MI_S32 id5 = 5 * C + ch;

        row0_head[0]   = src_c[id0]  * 36368 + src_c[id1]  * (-3600);
        row1_head[id0] = src_n0[id0] * 36368 + src_n0[id1] * (-3600);
        row2_head[id0] = src_n1[id0] * 36368 + src_n1[id1] * (-3600);
        row3_head[id0] = src_n2[id0] * 36368 + src_n2[id1] * (-3600);
        row0_head[1]   = src_c[id0]  * 35120 + src_c[id1]  * (-2352);
        row1_head[id1] = src_n0[id0] * 35120 + src_n0[id1] * (-2352);
        row2_head[id1] = src_n1[id0] * 35120 + src_n1[id1] * (-2352);
        row3_head[id1] = src_n2[id0] * 35120 + src_n2[id1] * (-2352);
        row0_head[2]   = src_c[id0]  * 29344 + src_c[id1]  * 3760  + src_c[id2]  * (-336);
        row1_head[id2] = src_n0[id0] * 29344 + src_n0[id1] * 3760  + src_n0[id2] * (-336);
        row2_head[id2] = src_n1[id0] * 29344 + src_n1[id1] * 3760  + src_n1[id2] * (-336);
        row3_head[id2] = src_n2[id0] * 29344 + src_n2[id1] * 3760  + src_n2[id2] * (-336);
        row0_head[3]   = src_c[id0]  * 20960 + src_c[id1]  * 13968 + src_c[id2]  * (-2160);
        row1_head[id3] = src_n0[id0] * 20960 + src_n0[id1] * 13968 + src_n0[id2] * (-2160);
        row2_head[id3] = src_n1[id0] * 20960 + src_n1[id1] * 13968 + src_n1[id2] * (-2160);
        row3_head[id3] = src_n2[id0] * 20960 + src_n2[id1] * 13968 + src_n2[id2] * (-2160);
        row0_head[4]   = src_c[id0]  * 11808 + src_c[id1]  * 24560 + src_c[id2]  * (-3600);
        row1_head[id4] = src_n0[id0] * 11808 + src_n0[id1] * 24560 + src_n0[id2] * (-3600);
        row2_head[id4] = src_n1[id0] * 11808 + src_n1[id1] * 24560 + src_n1[id2] * (-3600);
        row3_head[id4] = src_n2[id0] * 11808 + src_n2[id1] * 24560 + src_n2[id2] * (-3600);
        row0_head[5]   = src_c[id0]  * 3424  + src_c[id1]  * 31696 + src_c[id2]  * (-2352);
        row1_head[id5] = src_n0[id0] * 3424  + src_n0[id1] * 31696 + src_n0[id2] * (-2352);
        row2_head[id5] = src_n1[id0] * 3424  + src_n1[id1] * 31696 + src_n1[id2] * (-2352);
        row3_head[id5] = src_n2[id0] * 3424  + src_n2[id1] * 31696 + src_n2[id2] * (-2352);

        Tp *dst = dst_c;
        #pragma unroll(10)
        for (MI_S32 y = 0; y < 10; y++)
        {
            dst[id0] = SaturateCast<Tp>((row0_head[0] * BETA_S32[y][0] + row1_head[id0] * BETA_S32[y][1] + row2_head[id0] * BETA_S32[y][2] + row3_head[id0] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id1] = SaturateCast<Tp>((row0_head[1] * BETA_S32[y][0] + row1_head[id1] * BETA_S32[y][1] + row2_head[id1] * BETA_S32[y][2] + row3_head[id1] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id2] = SaturateCast<Tp>((row0_head[2] * BETA_S32[y][0] + row1_head[id2] * BETA_S32[y][1] + row2_head[id2] * BETA_S32[y][2] + row3_head[id2] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id3] = SaturateCast<Tp>((row0_head[3] * BETA_S32[y][0] + row1_head[id3] * BETA_S32[y][1] + row2_head[id3] * BETA_S32[y][2] + row3_head[id3] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id4] = SaturateCast<Tp>((row0_head[4] * BETA_S32[y][0] + row1_head[id4] * BETA_S32[y][1] + row2_head[id4] * BETA_S32[y][2] + row3_head[id4] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id5] = SaturateCast<Tp>((row0_head[5] * BETA_S32[y][0] + row1_head[id5] * BETA_S32[y][1] + row2_head[id5] * BETA_S32[y][2] + row3_head[id5] * BETA_S32[y][3] + 536870912) >> 30);

            dst += (ostride / sizeof(Tp));
        }

        HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch],  mv_c_x0_src.val[ch],  -2);
        HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x0_src.val[ch], mv_n0_x0_src.val[ch], -2);
        HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x0_src.val[ch], mv_n1_x0_src.val[ch], -2);
        HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x0_src.val[ch], mv_n2_x0_src.val[ch], -2);
        mv_c_x0_src.val[ch]      = Q6_V_lo_W(w_c_twin);
        mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
        mv_n0_x0_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
        mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
        mv_n1_x0_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
        mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
        mv_n2_x0_src.val[ch]     = Q6_V_lo_W(w_n2_twin);
        mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);
    }

    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result;

    auto resize_cu_up4_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, mv_c_dst.val[ch],  36368, -3600);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, mv_n0_dst.val[ch], 35120, -2352);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, mv_n1_dst.val[ch], 29344, 3760,  -336);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, mv_n2_dst.val[ch], 20960, 13968, -2160);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, mv_n3_dst.val[ch], 11808, 24560, -3600);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, mv_n4_dst.val[ch], 3424,  31696, -2352);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_n5_dst.val[ch], -2352, 31696, 3760,  -336);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_n6_dst.val[ch], -3600, 24560, 13968, -2160);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_n7_dst.val[ch], -2160, 13968, 24560, -3600);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_n8_dst.val[ch], -336,  3760,  31696, -2352);
    };

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n8_dst);

        vload(src_c  + (x + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (x + elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (x + elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (x + elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -2);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -2);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -2);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_lo_W(w_n2_twin);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();

            mv_c_x0_src.val[ch]  = mv_c_x1_src.val[ch];
            mv_n0_x0_src.val[ch] = mv_n0_x1_src.val[ch];
            mv_n1_x0_src.val[ch] = mv_n1_x1_src.val[ch];
            mv_n2_x0_src.val[ch] = mv_n2_x1_src.val[ch];

            mv_c_x1_src.val[ch]  = Q6_V_hi_W(w_c_twin);
            mv_n0_x1_src.val[ch] = Q6_V_hi_W(w_n0_twin);
            mv_n1_x1_src.val[ch] = Q6_V_hi_W(w_n1_twin);
            mv_n2_x1_src.val[ch] = Q6_V_hi_W(w_n2_twin);
        }

        vstore(dst_c  + ((x << 2) + elem_counts * 3 + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n8_dst);
    }

    if (width_align < iwidth - 3)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n8_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -2);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -2);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -2);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_lo_W(w_n2_twin);

            HVX_Vector v_c_border  = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  (width_align + elem_counts - iwidth) << 2);
            HVX_Vector v_n0_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n0_x1_src.val[ch], (width_align + elem_counts - iwidth) << 2);
            HVX_Vector v_n1_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n1_x1_src.val[ch], (width_align + elem_counts - iwidth) << 2);
            HVX_Vector v_n2_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n2_x1_src.val[ch], (width_align + elem_counts - iwidth) << 2);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts * 3) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n8_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 6 - (elem_counts << 2);

        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_n1 + sx * C, mv_n1_x0_src);
        vload(src_n2 + sx * C, mv_n2_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch],  mv_c_x0_src.val[ch],  -2);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x0_src.val[ch], mv_n0_x0_src.val[ch], -2);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x0_src.val[ch], mv_n1_x0_src.val[ch], -2);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x0_src.val[ch], mv_n2_x0_src.val[ch], -2);
            mv_c_x0_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
            mv_n0_x0_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
            mv_n1_x0_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
            mv_n2_x0_src.val[ch]     = Q6_V_lo_W(w_n2_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);

            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);
        vstore(dst_n3 + dx * C, mv_n3_dst);
        vstore(dst_n4 + dx * C, mv_n4_dst);
        vstore(dst_n5 + dx * C, mv_n5_dst);
        vstore(dst_n6 + dx * C, mv_n6_dst);
        vstore(dst_n7 + dx * C, mv_n7_dst);
        vstore(dst_n8 + dx * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + elem_counts) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + elem_counts) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + elem_counts) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + elem_counts) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + elem_counts) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + elem_counts) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts << 1)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts << 1)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts << 1)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts << 1)) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + (elem_counts << 1)) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + (elem_counts << 1)) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + (elem_counts << 1)) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + (elem_counts << 1)) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + (elem_counts << 1)) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + (elem_counts << 1)) * C, mv_n8_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -2);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -2);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -2);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);

            HVX_Vector v_c_border  = Q6_V_vlalign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  12);
            HVX_Vector v_n0_border = Q6_V_vlalign_VVR(Q6_V_vzero(), mv_n0_x1_src.val[ch], 12);
            HVX_Vector v_n1_border = Q6_V_vlalign_VVR(Q6_V_vzero(), mv_n1_x1_src.val[ch], 12);
            HVX_Vector v_n2_border = Q6_V_vlalign_VVR(Q6_V_vzero(), mv_n2_x1_src.val[ch], 12);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts * 3)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts * 3)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts * 3)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts * 3)) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + (elem_counts * 3)) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + (elem_counts * 3)) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + (elem_counts * 3)) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + (elem_counts * 3)) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + (elem_counts * 3)) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + (elem_counts * 3)) * C, mv_n8_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S64 row0_head[6];
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[0]           = src_c[id2]  * (-2352) + src_c[id1]  * 31696 + src_c[id0]  * 3424;
        row1_head[6 * C + ch]  = src_n0[id2] * (-2352) + src_n0[id1] * 31696 + src_n0[id0] * 3424;
        row2_head[6 * C + ch]  = src_n1[id2] * (-2352) + src_n1[id1] * 31696 + src_n1[id0] * 3424;
        row3_head[6 * C + ch]  = src_n2[id2] * (-2352) + src_n2[id1] * 31696 + src_n2[id0] * 3424;
        row0_head[1]           = src_c[id2]  * (-3600) + src_c[id1]  * 24560 + src_c[id0]  * 11808;
        row1_head[7 * C + ch]  = src_n0[id2] * (-3600) + src_n0[id1] * 24560 + src_n0[id0] * 11808;
        row2_head[7 * C + ch]  = src_n1[id2] * (-3600) + src_n1[id1] * 24560 + src_n1[id0] * 11808;
        row3_head[7 * C + ch]  = src_n2[id2] * (-3600) + src_n2[id1] * 24560 + src_n2[id0] * 11808;
        row0_head[2]           = src_c[id2]  * (-2160) + src_c[id1]  * 13968 + src_c[id0]  * 20960;
        row1_head[8 * C + ch]  = src_n0[id2] * (-2160) + src_n0[id1] * 13968 + src_n0[id0] * 20960;
        row2_head[8 * C + ch]  = src_n1[id2] * (-2160) + src_n1[id1] * 13968 + src_n1[id0] * 20960;
        row3_head[8 * C + ch]  = src_n2[id2] * (-2160) + src_n2[id1] * 13968 + src_n2[id0] * 20960;
        row0_head[3]           = src_c[id2]  * (-336)  + src_c[id1]  * 3760  + src_c[id0]  * 29344;
        row1_head[9 * C + ch]  = src_n0[id2] * (-336)  + src_n0[id1] * 3760  + src_n0[id0] * 29344;
        row2_head[9 * C + ch]  = src_n1[id2] * (-336)  + src_n1[id1] * 3760  + src_n1[id0] * 29344;
        row3_head[9 * C + ch]  = src_n2[id2] * (-336)  + src_n2[id1] * 3760  + src_n2[id0] * 29344;
        row0_head[4]           = src_c[id1]  * (-2352) + src_c[id0]  * 35120;
        row1_head[10 * C + ch] = src_n0[id1] * (-2352) + src_n0[id0] * 35120;
        row2_head[10 * C + ch] = src_n1[id1] * (-2352) + src_n1[id0] * 35120;
        row3_head[10 * C + ch] = src_n2[id1] * (-2352) + src_n2[id0] * 35120;
        row0_head[5]           = src_c[id1]  * (-3600) + src_c[id0]  * 36368;
        row1_head[11 * C + ch] = src_n0[id1] * (-3600) + src_n0[id0] * 36368;
        row2_head[11 * C + ch] = src_n1[id1] * (-3600) + src_n1[id0] * 36368;
        row3_head[11 * C + ch] = src_n2[id1] * (-3600) + src_n2[id0] * 36368;

        Tp *dst = dst_c;
        #pragma unroll(10)
        for (MI_S32 y = 0; y < 10; y++)
        {
            dst[(owidth - 6) * C + ch] = SaturateCast<Tp>((row0_head[0] * BETA_S32[y][0] + row1_head[6 * C + ch]  * BETA_S32[y][1] + row2_head[6 * C + ch]  * BETA_S32[y][2] + row3_head[6 * C + ch]  * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 5) * C + ch] = SaturateCast<Tp>((row0_head[1] * BETA_S32[y][0] + row1_head[7 * C + ch]  * BETA_S32[y][1] + row2_head[7 * C + ch]  * BETA_S32[y][2] + row3_head[7 * C + ch]  * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 4) * C + ch] = SaturateCast<Tp>((row0_head[2] * BETA_S32[y][0] + row1_head[8 * C + ch]  * BETA_S32[y][1] + row2_head[8 * C + ch]  * BETA_S32[y][2] + row3_head[8 * C + ch]  * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[3] * BETA_S32[y][0] + row1_head[9 * C + ch]  * BETA_S32[y][1] + row2_head[9 * C + ch]  * BETA_S32[y][2] + row3_head[9 * C + ch]  * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[4] * BETA_S32[y][0] + row1_head[10 * C + ch] * BETA_S32[y][1] + row2_head[10 * C + ch] * BETA_S32[y][2] + row3_head[10 * C + ch] * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[5] * BETA_S32[y][0] + row1_head[11 * C + ch] * BETA_S32[y][1] + row2_head[11 * C + ch] * BETA_S32[y][2] + row3_head[11 * C + ch] * BETA_S32[y][3] + 536870912) >> 30);

            dst += (ostride / sizeof(Tp));
        }
    }
}

// Tp = MI_U16 / MI_S16
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
ResizeCuUpX4UpBorder(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 istride, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));
    const Tp *src_n2 = src_n1 + (istride / sizeof(Tp));
    Tp *dst_n0       = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1       = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2       = dst_n1 + (ostride / sizeof(Tp));

    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S64 *row_head      = reinterpret_cast<MI_S64*>(vtcm_buffer->row_head);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src, mv_n2_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src, mv_n2_x1_src;
    MVType mv_c_dst,  mv_n0_dst, mv_n1_dst, mv_n2_dst;
    HVX_Vector v_alpha0 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFEB0F790), Q6_V_vsplat_R(0xF1F0F6D0), -4));    // (-336  << 16) + (-2160) (-3600 << 16) + (-2352)
    HVX_Vector v_alpha1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x0EB03690), Q6_V_vsplat_R(0x5FF07BD0), -4));    // (3760  << 16) + (13968) (24560 << 16) + (31696)
    HVX_Vector v_alpha2 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x7BD05FF0), Q6_V_vsplat_R(0x36900EB0), -4));    // (31696 << 16) + (24560) (13968 << 16) + (3760)
    HVX_Vector v_alpha3 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xF6D0F1F0), Q6_V_vsplat_R(0xF790FEB0), -4));    // (-2385 << 16) + (-3600) (-2160 << 16) + (-336)

    MI_S64 *row1_head = row_head + C * 12;
    MI_S64 *row2_head = row_head + C * 24;
    MI_S64 *row3_head = row_head + C * 36;

    vload(src_c , mv_c_x0_src);
    vload(src_n0, mv_n0_x0_src);
    vload(src_n1, mv_n1_x0_src);
    vload(src_n2, mv_n2_x0_src);

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S64 row0_head[6];
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;
        MI_S32 id3 = 3 * C + ch;
        MI_S32 id4 = 4 * C + ch;
        MI_S32 id5 = 5 * C + ch;

        row0_head[0]   = src_c[id0]  * 36368 + src_c[id1]  * (-3600);
        row1_head[id0] = src_n0[id0] * 36368 + src_n0[id1] * (-3600);
        row2_head[id0] = src_n1[id0] * 36368 + src_n1[id1] * (-3600);
        row3_head[id0] = src_n2[id0] * 36368 + src_n2[id1] * (-3600);
        row0_head[1]   = src_c[id0]  * 35120 + src_c[id1]  * (-2352);
        row1_head[id1] = src_n0[id0] * 35120 + src_n0[id1] * (-2352);
        row2_head[id1] = src_n1[id0] * 35120 + src_n1[id1] * (-2352);
        row3_head[id1] = src_n2[id0] * 35120 + src_n2[id1] * (-2352);
        row0_head[2]   = src_c[id0]  * 29344 + src_c[id1]  * 3760  + src_c[id2]  * (-336);
        row1_head[id2] = src_n0[id0] * 29344 + src_n0[id1] * 3760  + src_n0[id2] * (-336);
        row2_head[id2] = src_n1[id0] * 29344 + src_n1[id1] * 3760  + src_n1[id2] * (-336);
        row3_head[id2] = src_n2[id0] * 29344 + src_n2[id1] * 3760  + src_n2[id2] * (-336);
        row0_head[3]   = src_c[id0]  * 20960 + src_c[id1]  * 13968 + src_c[id2]  * (-2160);
        row1_head[id3] = src_n0[id0] * 20960 + src_n0[id1] * 13968 + src_n0[id2] * (-2160);
        row2_head[id3] = src_n1[id0] * 20960 + src_n1[id1] * 13968 + src_n1[id2] * (-2160);
        row3_head[id3] = src_n2[id0] * 20960 + src_n2[id1] * 13968 + src_n2[id2] * (-2160);
        row0_head[4]   = src_c[id0]  * 11808 + src_c[id1]  * 24560 + src_c[id2]  * (-3600);
        row1_head[id4] = src_n0[id0] * 11808 + src_n0[id1] * 24560 + src_n0[id2] * (-3600);
        row2_head[id4] = src_n1[id0] * 11808 + src_n1[id1] * 24560 + src_n1[id2] * (-3600);
        row3_head[id4] = src_n2[id0] * 11808 + src_n2[id1] * 24560 + src_n2[id2] * (-3600);
        row0_head[5]   = src_c[id0]  * 3424  + src_c[id1]  * 31696 + src_c[id2]  * (-2352);
        row1_head[id5] = src_n0[id0] * 3424  + src_n0[id1] * 31696 + src_n0[id2] * (-2352);
        row2_head[id5] = src_n1[id0] * 3424  + src_n1[id1] * 31696 + src_n1[id2] * (-2352);
        row3_head[id5] = src_n2[id0] * 3424  + src_n2[id1] * 31696 + src_n2[id2] * (-2352);

        Tp *dst = dst_c;
        #pragma unroll(4)
        for (MI_S32 y = 6; y < 10; y++)
        {
            dst[id0] = SaturateCast<Tp>((row0_head[0] * BETA_S32[y][0] + row1_head[id0] * BETA_S32[y][1] + row2_head[id0] * BETA_S32[y][2] + row3_head[id0] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id1] = SaturateCast<Tp>((row0_head[1] * BETA_S32[y][0] + row1_head[id1] * BETA_S32[y][1] + row2_head[id1] * BETA_S32[y][2] + row3_head[id1] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id2] = SaturateCast<Tp>((row0_head[2] * BETA_S32[y][0] + row1_head[id2] * BETA_S32[y][1] + row2_head[id2] * BETA_S32[y][2] + row3_head[id2] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id3] = SaturateCast<Tp>((row0_head[3] * BETA_S32[y][0] + row1_head[id3] * BETA_S32[y][1] + row2_head[id3] * BETA_S32[y][2] + row3_head[id3] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id4] = SaturateCast<Tp>((row0_head[4] * BETA_S32[y][0] + row1_head[id4] * BETA_S32[y][1] + row2_head[id4] * BETA_S32[y][2] + row3_head[id4] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id5] = SaturateCast<Tp>((row0_head[5] * BETA_S32[y][0] + row1_head[id5] * BETA_S32[y][1] + row2_head[id5] * BETA_S32[y][2] + row3_head[id5] * BETA_S32[y][3] + 536870912) >> 30);

            dst += (ostride / sizeof(Tp));
        }

        HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch],  mv_c_x0_src.val[ch],  -2);
        HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x0_src.val[ch], mv_n0_x0_src.val[ch], -2);
        HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x0_src.val[ch], mv_n1_x0_src.val[ch], -2);
        HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x0_src.val[ch], mv_n2_x0_src.val[ch], -2);
        mv_c_x0_src.val[ch]      = Q6_V_lo_W(w_c_twin);
        mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
        mv_n0_x0_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
        mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
        mv_n1_x0_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
        mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
        mv_n2_x0_src.val[ch]     = Q6_V_lo_W(w_n2_twin);
        mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);
    }

    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result;

    auto resize_cu_up4_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_c_dst.val[ch],  -2352, 31696, 3760,  -336);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_n0_dst.val[ch], -3600, 24560, 13968, -2160);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_n1_dst.val[ch], -2160, 13968, 24560, -3600);
        ResizeCuFastUpVCore<Tp>(ws32_c_result, ws32_n0_result, ws32_n1_result, ws32_n2_result, mv_n2_dst.val[ch], -336,  3760,  31696, -2352);
    };

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);

        vload(src_c  + (x + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (x + elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (x + elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (x + elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -2);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -2);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -2);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_lo_W(w_n2_twin);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();

            mv_c_x0_src.val[ch]  = mv_c_x1_src.val[ch];
            mv_n0_x0_src.val[ch] = mv_n0_x1_src.val[ch];
            mv_n1_x0_src.val[ch] = mv_n1_x1_src.val[ch];
            mv_n2_x0_src.val[ch] = mv_n2_x1_src.val[ch];

            mv_c_x1_src.val[ch]  = Q6_V_hi_W(w_c_twin);
            mv_n0_x1_src.val[ch] = Q6_V_hi_W(w_n0_twin);
            mv_n1_x1_src.val[ch] = Q6_V_hi_W(w_n1_twin);
            mv_n2_x1_src.val[ch] = Q6_V_hi_W(w_n2_twin);
        }

        vstore(dst_c  + ((x << 2) + elem_counts * 3 + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n2_dst);
    }

    if (width_align < iwidth - 3)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -2);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -2);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -2);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_lo_W(w_n2_twin);

            HVX_Vector v_c_border  = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  (width_align + elem_counts - iwidth) << 2);
            HVX_Vector v_n0_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n0_x1_src.val[ch], (width_align + elem_counts - iwidth) << 2);
            HVX_Vector v_n1_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n1_x1_src.val[ch], (width_align + elem_counts - iwidth) << 2);
            HVX_Vector v_n2_border = Q6_V_valign_VVR(Q6_V_vzero(), mv_n2_x1_src.val[ch], (width_align + elem_counts - iwidth) << 2);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_row1++ = ws32_n0_result;
            *v_row2++ = ws32_n1_result;
            *v_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts * 3) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n2_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 6 - (elem_counts << 2);

        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_n1 + sx * C, mv_n1_x0_src);
        vload(src_n2 + sx * C, mv_n2_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch],  mv_c_x0_src.val[ch],  -2);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x0_src.val[ch], mv_n0_x0_src.val[ch], -2);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x0_src.val[ch], mv_n1_x0_src.val[ch], -2);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x0_src.val[ch], mv_n2_x0_src.val[ch], -2);
            mv_c_x0_src.val[ch]      = Q6_V_lo_W(w_c_twin);
            mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
            mv_n0_x0_src.val[ch]     = Q6_V_lo_W(w_n0_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
            mv_n1_x0_src.val[ch]     = Q6_V_lo_W(w_n1_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
            mv_n2_x0_src.val[ch]     = Q6_V_lo_W(w_n2_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);

            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n0_x0_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n1_x0_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H0Core<Tp>(mv_n2_x0_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H1Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H2Core<Tp>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts << 1)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts << 1)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts << 1)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts << 1)) * C, mv_n2_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (iwidth - elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (iwidth - elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (iwidth - elem_counts) * C, mv_n2_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin  = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch],  mv_c_x1_src.val[ch],  -2);
            HVX_VectorPair w_n0_twin = Q6_W_vshuff_VVR(mv_n0_x1_src.val[ch], mv_n0_x1_src.val[ch], -2);
            HVX_VectorPair w_n1_twin = Q6_W_vshuff_VVR(mv_n1_x1_src.val[ch], mv_n1_x1_src.val[ch], -2);
            HVX_VectorPair w_n2_twin = Q6_W_vshuff_VVR(mv_n2_x1_src.val[ch], mv_n2_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]      = Q6_V_hi_W(w_c_twin);
            mv_n0_x1_src.val[ch]     = Q6_V_hi_W(w_n0_twin);
            mv_n1_x1_src.val[ch]     = Q6_V_hi_W(w_n1_twin);
            mv_n2_x1_src.val[ch]     = Q6_V_hi_W(w_n2_twin);

            HVX_Vector v_c_border  = Q6_V_vlalign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  12);
            HVX_Vector v_n0_border = Q6_V_vlalign_VVR(Q6_V_vzero(), mv_n0_x1_src.val[ch], 12);
            HVX_Vector v_n1_border = Q6_V_vlalign_VVR(Q6_V_vzero(), mv_n1_x1_src.val[ch], 12);
            HVX_Vector v_n2_border = Q6_V_vlalign_VVR(Q6_V_vzero(), mv_n2_x1_src.val[ch], 12);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch],  v_c_border,  ws32_c_result,  v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n0_x0_src.val[ch], v_n0_border, ws32_n0_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n1_x0_src.val[ch], v_n1_border, ws32_n1_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);
            ResizeCuUpX4H3Core<Tp>(mv_n2_x0_src.val[ch], v_n2_border, ws32_n2_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            *v_back_row1++ = ws32_n0_result;
            *v_back_row2++ = ws32_n1_result;
            *v_back_row3++ = ws32_n2_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts * 3)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts * 3)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts * 3)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts * 3)) * C, mv_n2_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S64 row0_head[6];
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[0]           = src_c[id2]  * (-2352) + src_c[id1]  * 31696 + src_c[id0]  * 3424;
        row1_head[6 * C + ch]  = src_n0[id2] * (-2352) + src_n0[id1] * 31696 + src_n0[id0] * 3424;
        row2_head[6 * C + ch]  = src_n1[id2] * (-2352) + src_n1[id1] * 31696 + src_n1[id0] * 3424;
        row3_head[6 * C + ch]  = src_n2[id2] * (-2352) + src_n2[id1] * 31696 + src_n2[id0] * 3424;
        row0_head[1]           = src_c[id2]  * (-3600) + src_c[id1]  * 24560 + src_c[id0]  * 11808;
        row1_head[7 * C + ch]  = src_n0[id2] * (-3600) + src_n0[id1] * 24560 + src_n0[id0] * 11808;
        row2_head[7 * C + ch]  = src_n1[id2] * (-3600) + src_n1[id1] * 24560 + src_n1[id0] * 11808;
        row3_head[7 * C + ch]  = src_n2[id2] * (-3600) + src_n2[id1] * 24560 + src_n2[id0] * 11808;
        row0_head[2]           = src_c[id2]  * (-2160) + src_c[id1]  * 13968 + src_c[id0]  * 20960;
        row1_head[8 * C + ch]  = src_n0[id2] * (-2160) + src_n0[id1] * 13968 + src_n0[id0] * 20960;
        row2_head[8 * C + ch]  = src_n1[id2] * (-2160) + src_n1[id1] * 13968 + src_n1[id0] * 20960;
        row3_head[8 * C + ch]  = src_n2[id2] * (-2160) + src_n2[id1] * 13968 + src_n2[id0] * 20960;
        row0_head[3]           = src_c[id2]  * (-336)  + src_c[id1]  * 3760  + src_c[id0]  * 29344;
        row1_head[9 * C + ch]  = src_n0[id2] * (-336)  + src_n0[id1] * 3760  + src_n0[id0] * 29344;
        row2_head[9 * C + ch]  = src_n1[id2] * (-336)  + src_n1[id1] * 3760  + src_n1[id0] * 29344;
        row3_head[9 * C + ch]  = src_n2[id2] * (-336)  + src_n2[id1] * 3760  + src_n2[id0] * 29344;
        row0_head[4]           = src_c[id1]  * (-2352) + src_c[id0]  * 35120;
        row1_head[10 * C + ch] = src_n0[id1] * (-2352) + src_n0[id0] * 35120;
        row2_head[10 * C + ch] = src_n1[id1] * (-2352) + src_n1[id0] * 35120;
        row3_head[10 * C + ch] = src_n2[id1] * (-2352) + src_n2[id0] * 35120;
        row0_head[5]           = src_c[id1]  * (-3600) + src_c[id0]  * 36368;
        row1_head[11 * C + ch] = src_n0[id1] * (-3600) + src_n0[id0] * 36368;
        row2_head[11 * C + ch] = src_n1[id1] * (-3600) + src_n1[id0] * 36368;
        row3_head[11 * C + ch] = src_n2[id1] * (-3600) + src_n2[id0] * 36368;

        Tp *dst = dst_c;
        #pragma unroll(4)
        for (MI_S32 y = 6; y < 10; y++)
        {
            dst[(owidth - 6) * C + ch] = SaturateCast<Tp>((row0_head[0] * BETA_S32[y][0] + row1_head[6 * C + ch]  * BETA_S32[y][1] + row2_head[6 * C + ch]  * BETA_S32[y][2] + row3_head[6 * C + ch]  * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 5) * C + ch] = SaturateCast<Tp>((row0_head[1] * BETA_S32[y][0] + row1_head[7 * C + ch]  * BETA_S32[y][1] + row2_head[7 * C + ch]  * BETA_S32[y][2] + row3_head[7 * C + ch]  * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 4) * C + ch] = SaturateCast<Tp>((row0_head[2] * BETA_S32[y][0] + row1_head[8 * C + ch]  * BETA_S32[y][1] + row2_head[8 * C + ch]  * BETA_S32[y][2] + row3_head[8 * C + ch]  * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[3] * BETA_S32[y][0] + row1_head[9 * C + ch]  * BETA_S32[y][1] + row2_head[9 * C + ch]  * BETA_S32[y][2] + row3_head[9 * C + ch]  * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[4] * BETA_S32[y][0] + row1_head[10 * C + ch] * BETA_S32[y][1] + row2_head[10 * C + ch] * BETA_S32[y][2] + row3_head[10 * C + ch] * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[5] * BETA_S32[y][0] + row1_head[11 * C + ch] * BETA_S32[y][1] + row2_head[11 * C + ch] * BETA_S32[y][2] + row3_head[11 * C + ch] * BETA_S32[y][3] + 536870912) >> 30);

            dst += (ostride / sizeof(Tp));
        }
    }
}

// Tp = MI_U16 / MI_S16
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
ResizeCuUpX4Row(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    Tp *dst_n0 = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1 = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2 = dst_n1 + (ostride / sizeof(Tp));

    MI_S32 *row0_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S32 *row3_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row0_ptr);
    MI_S64 *row_head      = reinterpret_cast<MI_S64*>(vtcm_buffer->row_head);
    MI_S32 *row0_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);
    MI_S32 *row3_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row0_ptr + row_pre_size);
    vtcm_buffer->row0_ptr = reinterpret_cast<MI_U8*>(row0_ptr);
    vtcm_buffer->row1_ptr = reinterpret_cast<MI_U8*>(row1_ptr);
    vtcm_buffer->row2_ptr = reinterpret_cast<MI_U8*>(row2_ptr);
    vtcm_buffer->row3_ptr = reinterpret_cast<MI_U8*>(row3_ptr);

    MVType mv_c_x0_src, mv_c_x1_src;
    MVType mv_c_dst, mv_n0_dst, mv_n1_dst, mv_n2_dst;
    HVX_Vector v_alpha0 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFEB0F790), Q6_V_vsplat_R(0xF1F0F6D0), -4));    // (-336  << 16) + (-2160) (-3600 << 16) + (-2352)
    HVX_Vector v_alpha1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x0EB03690), Q6_V_vsplat_R(0x5FF07BD0), -4));    // (3760  << 16) + (13968) (24560 << 16) + (31696)
    HVX_Vector v_alpha2 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x7BD05FF0), Q6_V_vsplat_R(0x36900EB0), -4));    // (31696 << 16) + (24560) (13968 << 16) + (3760)
    HVX_Vector v_alpha3 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xF6D0F1F0), Q6_V_vsplat_R(0xF790FEB0), -4));    // (-2385 << 16) + (-3600) (-2160 << 16) + (-336)

    MI_S64 *row0_head = row_head;
    MI_S64 *row1_head = row_head + C * 12;
    MI_S64 *row2_head = row_head + C * 24;
    MI_S64 *row3_head = row_head + C * 36;

    vload(src_c , mv_c_x0_src);

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;
        MI_S32 id3 = 3 * C + ch;
        MI_S32 id4 = 4 * C + ch;
        MI_S32 id5 = 5 * C + ch;

        row0_head[id0] = row1_head[id0]; row1_head[id0] = row2_head[id0]; row2_head[id0] = row3_head[id0];
        row0_head[id1] = row1_head[id1]; row1_head[id1] = row2_head[id1]; row2_head[id1] = row3_head[id1];
        row0_head[id2] = row1_head[id2]; row1_head[id2] = row2_head[id2]; row2_head[id2] = row3_head[id2];
        row0_head[id3] = row1_head[id3]; row1_head[id3] = row2_head[id3]; row2_head[id3] = row3_head[id3];
        row0_head[id4] = row1_head[id4]; row1_head[id4] = row2_head[id4]; row2_head[id4] = row3_head[id4];
        row0_head[id5] = row1_head[id5]; row1_head[id5] = row2_head[id5]; row2_head[id5] = row3_head[id5];
        row3_head[id0] = src_c[id0] * 36368 + src_c[id1] * (-3600);
        row3_head[id1] = src_c[id0] * 35120 + src_c[id1] * (-2352);
        row3_head[id2] = src_c[id0] * 29344 + src_c[id1] * 3760  + src_c[id2] * (-336);
        row3_head[id3] = src_c[id0] * 20960 + src_c[id1] * 13968 + src_c[id2] * (-2160);
        row3_head[id4] = src_c[id0] * 11808 + src_c[id1] * 24560 + src_c[id2] * (-3600);
        row3_head[id5] = src_c[id0] * 3424  + src_c[id1] * 31696 + src_c[id2] * (-2352);

        Tp *dst = dst_c;
        #pragma unroll(4)
        for (MI_S32 y = 6; y < 10; y++)
        {
            dst[id0] = SaturateCast<Tp>((row0_head[id0] * BETA_S32[y][0] + row1_head[id0] * BETA_S32[y][1] + row2_head[id0] * BETA_S32[y][2] + row3_head[id0] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id1] = SaturateCast<Tp>((row0_head[id1] * BETA_S32[y][0] + row1_head[id1] * BETA_S32[y][1] + row2_head[id1] * BETA_S32[y][2] + row3_head[id1] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id2] = SaturateCast<Tp>((row0_head[id2] * BETA_S32[y][0] + row1_head[id2] * BETA_S32[y][1] + row2_head[id2] * BETA_S32[y][2] + row3_head[id2] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id3] = SaturateCast<Tp>((row0_head[id3] * BETA_S32[y][0] + row1_head[id3] * BETA_S32[y][1] + row2_head[id3] * BETA_S32[y][2] + row3_head[id3] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id4] = SaturateCast<Tp>((row0_head[id4] * BETA_S32[y][0] + row1_head[id4] * BETA_S32[y][1] + row2_head[id4] * BETA_S32[y][2] + row3_head[id4] * BETA_S32[y][3] + 536870912) >> 30);
            dst[id5] = SaturateCast<Tp>((row0_head[id5] * BETA_S32[y][0] + row1_head[id5] * BETA_S32[y][1] + row2_head[id5] * BETA_S32[y][2] + row3_head[id5] * BETA_S32[y][3] + 536870912) >> 30);

            dst += (ostride / sizeof(Tp));
        }

        HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch], mv_c_x0_src.val[ch], -2);
        mv_c_x0_src.val[ch]     = Q6_V_lo_W(w_c_twin);
        mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);
    }

    HVX_VectorPair *v_row0 = (HVX_VectorPair *)(row0_ptr);
    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair *v_row3 = (HVX_VectorPair *)(row3_ptr);
    HVX_VectorPair ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result;

    auto resize_cu_up4_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_c_dst.val[ch],  -2352, 31696, 3760,  -336);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n0_dst.val[ch], -3600, 24560, 13968, -2160);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n1_dst.val[ch], -2160, 13968, 24560, -3600);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n2_dst.val[ch], -336,  3760,  31696, -2352);
    };

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);

        vload(src_c + (x + elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch], mv_c_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]     = Q6_V_lo_W(w_c_twin);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();

            mv_c_x0_src.val[ch] = mv_c_x1_src.val[ch];
            mv_c_x1_src.val[ch] = Q6_V_hi_W(w_c_twin);
        }

        vstore(dst_c  + ((x << 2) + elem_counts * 3 + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n2_dst);
    }

    if (width_align < iwidth - 3)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch], mv_c_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]     = Q6_V_lo_W(w_c_twin);
            HVX_Vector v_c_border   = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  (width_align + elem_counts - iwidth) << 2);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;
            *v_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts * 3) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n2_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 6 - (elem_counts << 2);

        HVX_VectorPair *v_back_row0 = (HVX_VectorPair *)(row0_back_ptr);
        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);
        HVX_VectorPair *v_back_row3 = (HVX_VectorPair *)(row3_back_ptr);

        vload(src_c + sx * C, mv_c_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch], mv_c_x0_src.val[ch], -2);
            mv_c_x0_src.val[ch]     = Q6_V_lo_W(w_c_twin);
            mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);

            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;
            *v_back_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;
            *v_back_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;
            *v_back_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts << 1)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts << 1)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts << 1)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts << 1)) * C, mv_n2_dst);

        vload(src_c + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch], mv_c_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);
            HVX_Vector v_c_border   = Q6_V_vlalign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch], 12);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;
            *v_back_row3++ = ws32_c_result;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts * 3)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts * 3)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts * 3)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts * 3)) * C, mv_n2_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[6  * C + ch] = row1_head[6  * C + ch]; row1_head[6  * C + ch] = row2_head[6  * C + ch];
        row0_head[7  * C + ch] = row1_head[7  * C + ch]; row1_head[7  * C + ch] = row2_head[7  * C + ch];
        row0_head[8  * C + ch] = row1_head[8  * C + ch]; row1_head[8  * C + ch] = row2_head[8  * C + ch];
        row0_head[9  * C + ch] = row1_head[9  * C + ch]; row1_head[9  * C + ch] = row2_head[9  * C + ch];
        row0_head[10 * C + ch] = row1_head[10 * C + ch]; row1_head[10 * C + ch] = row2_head[10 * C + ch];
        row0_head[11 * C + ch] = row1_head[11 * C + ch]; row1_head[11 * C + ch] = row2_head[11 * C + ch];
        row2_head[6  * C + ch] = row3_head[6  * C + ch];
        row2_head[7  * C + ch] = row3_head[7  * C + ch];
        row2_head[8  * C + ch] = row3_head[8  * C + ch];
        row2_head[9  * C + ch] = row3_head[9  * C + ch];
        row2_head[10 * C + ch] = row3_head[10 * C + ch];
        row2_head[11 * C + ch] = row3_head[11 * C + ch];
        row3_head[6  * C + ch] = src_c[id2] * (-2352) + src_c[id1] * 31696 + src_c[id0] * 3424;
        row3_head[7  * C + ch] = src_c[id2] * (-3600) + src_c[id1] * 24560 + src_c[id0] * 11808;
        row3_head[8  * C + ch] = src_c[id2] * (-2160) + src_c[id1] * 13968 + src_c[id0] * 20960;
        row3_head[9  * C + ch] = src_c[id2] * (-336)  + src_c[id1] * 3760  + src_c[id0] * 29344;
        row3_head[10 * C + ch] = src_c[id1] * (-2352) + src_c[id0] * 35120;
        row3_head[11 * C + ch] = src_c[id1] * (-3600) + src_c[id0] * 36368;

        Tp *dst = dst_c;
        #pragma unroll(4)
        for (MI_S32 y = 6; y < 10; y++)
        {
            dst[(owidth - 6) * C + ch] = SaturateCast<Tp>((row0_head[6  * C + ch] * BETA_S32[y][0] + row1_head[6  * C + ch] * BETA_S32[y][1] + row2_head[6  * C + ch] * BETA_S32[y][2] + row3_head[6  * C + ch] * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 5) * C + ch] = SaturateCast<Tp>((row0_head[7  * C + ch] * BETA_S32[y][0] + row1_head[7  * C + ch] * BETA_S32[y][1] + row2_head[7  * C + ch] * BETA_S32[y][2] + row3_head[7  * C + ch] * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 4) * C + ch] = SaturateCast<Tp>((row0_head[8  * C + ch] * BETA_S32[y][0] + row1_head[8  * C + ch] * BETA_S32[y][1] + row2_head[8  * C + ch] * BETA_S32[y][2] + row3_head[8  * C + ch] * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[9  * C + ch] * BETA_S32[y][0] + row1_head[9  * C + ch] * BETA_S32[y][1] + row2_head[9  * C + ch] * BETA_S32[y][2] + row3_head[9  * C + ch] * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[10 * C + ch] * BETA_S32[y][0] + row1_head[10 * C + ch] * BETA_S32[y][1] + row2_head[10 * C + ch] * BETA_S32[y][2] + row3_head[10 * C + ch] * BETA_S32[y][3] + 536870912) >> 30);
            dst[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[11 * C + ch] * BETA_S32[y][0] + row1_head[11 * C + ch] * BETA_S32[y][1] + row2_head[11 * C + ch] * BETA_S32[y][2] + row3_head[11 * C + ch] * BETA_S32[y][3] + 536870912) >> 30);

            dst += (ostride / sizeof(Tp));
        }
    }
}

// Tp = MI_U16 / MI_S16
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
ResizeCuUpX4BottomBorder(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, MI_S32 iwidth, MI_S32 owidth, MI_S32 ostride)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align  = (iwidth - 3) & (-elem_counts);
    MI_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN);

    Tp *dst_n0 = dst_c  + (ostride / sizeof(Tp));
    Tp *dst_n1 = dst_n0 + (ostride / sizeof(Tp));
    Tp *dst_n2 = dst_n1 + (ostride / sizeof(Tp));
    Tp *dst_n3 = dst_n2 + (ostride / sizeof(Tp));
    Tp *dst_n4 = dst_n3 + (ostride / sizeof(Tp));
    Tp *dst_n5 = dst_n4 + (ostride / sizeof(Tp));
    Tp *dst_n6 = dst_n5 + (ostride / sizeof(Tp));
    Tp *dst_n7 = dst_n6 + (ostride / sizeof(Tp));
    Tp *dst_n8 = dst_n7 + (ostride / sizeof(Tp));

    MI_S32 *row0_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr);
    MI_S32 *row1_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr);
    MI_S32 *row2_ptr      = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr);
    MI_S64 *row_head      = reinterpret_cast<MI_S64*>(vtcm_buffer->row_head);
    MI_S32 *row0_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    MI_S32 *row1_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    MI_S32 *row2_back_ptr = reinterpret_cast<MI_S32*>(vtcm_buffer->row3_ptr + row_pre_size);
    vtcm_buffer->row0_ptr = reinterpret_cast<MI_U8*>(row0_ptr);
    vtcm_buffer->row1_ptr = reinterpret_cast<MI_U8*>(row1_ptr);
    vtcm_buffer->row2_ptr = reinterpret_cast<MI_U8*>(row2_ptr);

    MVType mv_c_x0_src, mv_c_x1_src;
    MVType mv_c_dst,  mv_n0_dst, mv_n1_dst, mv_n2_dst, mv_n3_dst;
    MVType mv_n4_dst, mv_n5_dst, mv_n6_dst, mv_n7_dst, mv_n8_dst;
    HVX_Vector v_alpha0 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xFEB0F790), Q6_V_vsplat_R(0xF1F0F6D0), -4));    // (-336  << 16) + (-2160) (-3600 << 16) + (-2352)
    HVX_Vector v_alpha1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x0EB03690), Q6_V_vsplat_R(0x5FF07BD0), -4));    // (3760  << 16) + (13968) (24560 << 16) + (31696)
    HVX_Vector v_alpha2 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0x7BD05FF0), Q6_V_vsplat_R(0x36900EB0), -4));    // (31696 << 16) + (24560) (13968 << 16) + (3760)
    HVX_Vector v_alpha3 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(0xF6D0F1F0), Q6_V_vsplat_R(0xF790FEB0), -4));    // (-2385 << 16) + (-3600) (-2160 << 16) + (-336)

    MI_S64 *row0_head = row_head;
    MI_S64 *row1_head = row_head + C * 12;
    MI_S64 *row2_head = row_head + C * 24;
    MI_S64 *row3_head = row_head + C * 36;

    vload(src_c , mv_c_x0_src);

    MI_S32 ch = 0;
    #pragma unroll(C)
    for (; ch < C; ch++)
    {
        MI_S32 id0 = ch;
        MI_S32 id1 = C + ch;
        MI_S32 id2 = 2 * C + ch;
        MI_S32 id3 = 3 * C + ch;
        MI_S32 id4 = 4 * C + ch;
        MI_S32 id5 = 5 * C + ch;

        row0_head[id0] = row1_head[id0]; row1_head[id0] = row2_head[id0]; row2_head[id0] = row3_head[id0];
        row0_head[id1] = row1_head[id1]; row1_head[id1] = row2_head[id1]; row2_head[id1] = row3_head[id1];
        row0_head[id2] = row1_head[id2]; row1_head[id2] = row2_head[id2]; row2_head[id2] = row3_head[id2];
        row0_head[id3] = row1_head[id3]; row1_head[id3] = row2_head[id3]; row2_head[id3] = row3_head[id3];
        row0_head[id4] = row1_head[id4]; row1_head[id4] = row2_head[id4]; row2_head[id4] = row3_head[id4];
        row0_head[id5] = row1_head[id5]; row1_head[id5] = row2_head[id5]; row2_head[id5] = row3_head[id5];
        row3_head[id0] = src_c[id0] * 36368 + src_c[id1] * (-3600);
        row3_head[id1] = src_c[id0] * 35120 + src_c[id1] * (-2352);
        row3_head[id2] = src_c[id0] * 29344 + src_c[id1] * 3760  + src_c[id2] * (-336);
        row3_head[id3] = src_c[id0] * 20960 + src_c[id1] * 13968 + src_c[id2] * (-2160);
        row3_head[id4] = src_c[id0] * 11808 + src_c[id1] * 24560 + src_c[id2] * (-3600);
        row3_head[id5] = src_c[id0] * 3424  + src_c[id1] * 31696 + src_c[id2] * (-2352);

        Tp *dst = dst_c;
        #pragma unroll(10)
        for (MI_S32 y = 0; y < 10; y++)
        {
            dst[id0] = SaturateCast<Tp>((row0_head[id0] * BETA_S32[9 - y][3] + row1_head[id0] * BETA_S32[9 - y][2] + row2_head[id0] * BETA_S32[9 - y][1] + row3_head[id0] * BETA_S32[9 - y][0] + 536870912) >> 30);
            dst[id1] = SaturateCast<Tp>((row0_head[id1] * BETA_S32[9 - y][3] + row1_head[id1] * BETA_S32[9 - y][2] + row2_head[id1] * BETA_S32[9 - y][1] + row3_head[id1] * BETA_S32[9 - y][0] + 536870912) >> 30);
            dst[id2] = SaturateCast<Tp>((row0_head[id2] * BETA_S32[9 - y][3] + row1_head[id2] * BETA_S32[9 - y][2] + row2_head[id2] * BETA_S32[9 - y][1] + row3_head[id2] * BETA_S32[9 - y][0] + 536870912) >> 30);
            dst[id3] = SaturateCast<Tp>((row0_head[id3] * BETA_S32[9 - y][3] + row1_head[id3] * BETA_S32[9 - y][2] + row2_head[id3] * BETA_S32[9 - y][1] + row3_head[id3] * BETA_S32[9 - y][0] + 536870912) >> 30);
            dst[id4] = SaturateCast<Tp>((row0_head[id4] * BETA_S32[9 - y][3] + row1_head[id4] * BETA_S32[9 - y][2] + row2_head[id4] * BETA_S32[9 - y][1] + row3_head[id4] * BETA_S32[9 - y][0] + 536870912) >> 30);
            dst[id5] = SaturateCast<Tp>((row0_head[id5] * BETA_S32[9 - y][3] + row1_head[id5] * BETA_S32[9 - y][2] + row2_head[id5] * BETA_S32[9 - y][1] + row3_head[id5] * BETA_S32[9 - y][0] + 536870912) >> 30);

            dst += (ostride / sizeof(Tp));
        }

        HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch], mv_c_x0_src.val[ch], -2);
        mv_c_x0_src.val[ch]     = Q6_V_lo_W(w_c_twin);
        mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);
    }

    HVX_VectorPair *v_row0 = (HVX_VectorPair *)(row0_ptr);
    HVX_VectorPair *v_row1 = (HVX_VectorPair *)(row1_ptr);
    HVX_VectorPair *v_row2 = (HVX_VectorPair *)(row2_ptr);
    HVX_VectorPair ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result;

    auto resize_cu_up4_vfunc = [&]()
    {
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_c_dst.val[ch],  -2352, 31696, 3760,  -336);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n0_dst.val[ch], -3600, 24560, 13968, -2160);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n1_dst.val[ch], -2160, 13968, 24560, -3600);
        ResizeCuFastUpVCore<Tp>(ws32_p2_result, ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n2_dst.val[ch], -336,  3760,  31696, -2352);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n3_dst.val[ch], -2352, 31696, 3424);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n4_dst.val[ch], -3600, 24560, 11808);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n5_dst.val[ch], -2160, 13968, 20960);
        ResizeCuFastUpVCore<Tp>(ws32_p1_result, ws32_p0_result, ws32_c_result, mv_n6_dst.val[ch], -336,  3760,  29344);
        ResizeCuFastUpVCore<Tp>(ws32_p0_result, ws32_c_result, mv_n7_dst.val[ch], -2352, 35120);
        ResizeCuFastUpVCore<Tp>(ws32_p0_result, ws32_c_result, mv_n8_dst.val[ch], -3600, 36368);
    };

    MI_S32 x = 0;
    for (; x < width_align - elem_counts; x += elem_counts)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n8_dst);

        vload(src_c + (x + elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch], mv_c_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]     = Q6_V_lo_W(w_c_twin);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up4_vfunc();

            mv_c_x0_src.val[ch] = mv_c_x1_src.val[ch];
            mv_c_x1_src.val[ch] = Q6_V_hi_W(w_c_twin);
        }

        vstore(dst_c  + ((x << 2) + elem_counts * 3 + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts * 3 + 6) * C, mv_n8_dst);
    }

    if (width_align < iwidth - 3)
    {
        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + elem_counts + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + elem_counts + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + elem_counts + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + elem_counts + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + elem_counts + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + elem_counts + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + elem_counts + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + elem_counts + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + elem_counts + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + elem_counts + 6) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts << 1) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts << 1) + 6) * C, mv_n8_dst);

        vload(src_c  + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch], mv_c_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]     = Q6_V_lo_W(w_c_twin);
            HVX_Vector v_c_border   = Q6_V_valign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch],  (width_align + elem_counts - iwidth) << 2);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_row0++;
            ws32_p1_result = *v_row1++;
            ws32_p0_result = *v_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + ((x << 2) + (elem_counts * 3) + 6) * C, mv_c_dst);
        vstore(dst_n0 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n0_dst);
        vstore(dst_n1 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n1_dst);
        vstore(dst_n2 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n2_dst);
        vstore(dst_n3 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n3_dst);
        vstore(dst_n4 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n4_dst);
        vstore(dst_n5 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n5_dst);
        vstore(dst_n6 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n6_dst);
        vstore(dst_n7 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n7_dst);
        vstore(dst_n8 + ((x << 2) + (elem_counts * 3) + 6) * C, mv_n8_dst);

        MI_S32 sx = iwidth - 3 - elem_counts;
        MI_S32 dx = owidth - 6 - (elem_counts << 2);

        HVX_VectorPair *v_back_row0 = (HVX_VectorPair *)(row0_back_ptr);
        HVX_VectorPair *v_back_row1 = (HVX_VectorPair *)(row1_back_ptr);
        HVX_VectorPair *v_back_row2 = (HVX_VectorPair *)(row2_back_ptr);

        vload(src_c + sx * C, mv_c_x0_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x0_src.val[ch], mv_c_x0_src.val[ch], -2);
            mv_c_x0_src.val[ch]     = Q6_V_lo_W(w_c_twin);
            mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);

            // row
            ResizeCuUpX4H0Core<Tp>(mv_c_x0_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + dx * C, mv_c_dst);
        vstore(dst_n0 + dx * C, mv_n0_dst);
        vstore(dst_n1 + dx * C, mv_n1_dst);
        vstore(dst_n2 + dx * C, mv_n2_dst);
        vstore(dst_n3 + dx * C, mv_n3_dst);
        vstore(dst_n4 + dx * C, mv_n4_dst);
        vstore(dst_n5 + dx * C, mv_n5_dst);
        vstore(dst_n6 + dx * C, mv_n6_dst);
        vstore(dst_n7 + dx * C, mv_n7_dst);
        vstore(dst_n8 + dx * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H1Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + elem_counts) * C, mv_c_dst);
        vstore(dst_n0 + (dx + elem_counts) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + elem_counts) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + elem_counts) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + elem_counts) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + elem_counts) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + elem_counts) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + elem_counts) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + elem_counts) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + elem_counts) * C, mv_n8_dst);

        for (ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuUpX4H2Core<Tp>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts << 1)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts << 1)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts << 1)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts << 1)) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + (elem_counts << 1)) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + (elem_counts << 1)) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + (elem_counts << 1)) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + (elem_counts << 1)) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + (elem_counts << 1)) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + (elem_counts << 1)) * C, mv_n8_dst);

        vload(src_c + (iwidth - elem_counts) * C, mv_c_x1_src);

        for (ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_twin = Q6_W_vshuff_VVR(mv_c_x1_src.val[ch], mv_c_x1_src.val[ch], -2);
            mv_c_x1_src.val[ch]     = Q6_V_hi_W(w_c_twin);
            HVX_Vector v_c_border   = Q6_V_vlalign_VVR(Q6_V_vzero(), mv_c_x1_src.val[ch], 12);

            // row
            ResizeCuUpX4H3Core<Tp>(mv_c_x0_src.val[ch], v_c_border, ws32_c_result, v_alpha0, v_alpha1, v_alpha2, v_alpha3);

            ws32_p2_result = *v_back_row0++;
            ws32_p1_result = *v_back_row1++;
            ws32_p0_result = *v_back_row2++;

            // col
            resize_cu_up4_vfunc();
        }

        vstore(dst_c  + (dx + (elem_counts * 3)) * C, mv_c_dst);
        vstore(dst_n0 + (dx + (elem_counts * 3)) * C, mv_n0_dst);
        vstore(dst_n1 + (dx + (elem_counts * 3)) * C, mv_n1_dst);
        vstore(dst_n2 + (dx + (elem_counts * 3)) * C, mv_n2_dst);
        vstore(dst_n3 + (dx + (elem_counts * 3)) * C, mv_n3_dst);
        vstore(dst_n4 + (dx + (elem_counts * 3)) * C, mv_n4_dst);
        vstore(dst_n5 + (dx + (elem_counts * 3)) * C, mv_n5_dst);
        vstore(dst_n6 + (dx + (elem_counts * 3)) * C, mv_n6_dst);
        vstore(dst_n7 + (dx + (elem_counts * 3)) * C, mv_n7_dst);
        vstore(dst_n8 + (dx + (elem_counts * 3)) * C, mv_n8_dst);
    }

    #pragma unroll(C)
    for (ch = 0; ch < C; ch++)
    {
        MI_S32 id0 = (iwidth - 1) * C + ch;
        MI_S32 id1 = (iwidth - 2) * C + ch;
        MI_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[6  * C + ch] = row1_head[6  * C + ch]; row1_head[6  * C + ch] = row2_head[6  * C + ch];
        row0_head[7  * C + ch] = row1_head[7  * C + ch]; row1_head[7  * C + ch] = row2_head[7  * C + ch];
        row0_head[8  * C + ch] = row1_head[8  * C + ch]; row1_head[8  * C + ch] = row2_head[8  * C + ch];
        row0_head[9  * C + ch] = row1_head[9  * C + ch]; row1_head[9  * C + ch] = row2_head[9  * C + ch];
        row0_head[10 * C + ch] = row1_head[10 * C + ch]; row1_head[10 * C + ch] = row2_head[10 * C + ch];
        row0_head[11 * C + ch] = row1_head[11 * C + ch]; row1_head[11 * C + ch] = row2_head[11 * C + ch];
        row2_head[6  * C + ch] = row3_head[6  * C + ch];
        row2_head[7  * C + ch] = row3_head[7  * C + ch];
        row2_head[8  * C + ch] = row3_head[8  * C + ch];
        row2_head[9  * C + ch] = row3_head[9  * C + ch];
        row2_head[10 * C + ch] = row3_head[10 * C + ch];
        row2_head[11 * C + ch] = row3_head[11 * C + ch];
        row3_head[6  * C + ch] = src_c[id2] * (-2352) + src_c[id1] * 31696 + src_c[id0] * 3424;
        row3_head[7  * C + ch] = src_c[id2] * (-3600) + src_c[id1] * 24560 + src_c[id0] * 11808;
        row3_head[8  * C + ch] = src_c[id2] * (-2160) + src_c[id1] * 13968 + src_c[id0] * 20960;
        row3_head[9  * C + ch] = src_c[id2] * (-336)  + src_c[id1] * 3760  + src_c[id0] * 29344;
        row3_head[10 * C + ch] = src_c[id1] * (-2352) + src_c[id0] * 35120;
        row3_head[11 * C + ch] = src_c[id1] * (-3600) + src_c[id0] * 36368;

        Tp *dst = dst_c;
        #pragma unroll(10)
        for (MI_S32 y = 0; y < 10; y++)
        {
            dst[(owidth - 6) * C + ch] = SaturateCast<Tp>((row0_head[6  * C + ch] * BETA_S32[9 - y][3] + row1_head[6  * C + ch] * BETA_S32[9 - y][2] + row2_head[6  * C + ch] * BETA_S32[9 - y][1] + row3_head[6  * C + ch] * BETA_S32[9 - y][0] + 536870912) >> 30);
            dst[(owidth - 5) * C + ch] = SaturateCast<Tp>((row0_head[7  * C + ch] * BETA_S32[9 - y][3] + row1_head[7  * C + ch] * BETA_S32[9 - y][2] + row2_head[7  * C + ch] * BETA_S32[9 - y][1] + row3_head[7  * C + ch] * BETA_S32[9 - y][0] + 536870912) >> 30);
            dst[(owidth - 4) * C + ch] = SaturateCast<Tp>((row0_head[8  * C + ch] * BETA_S32[9 - y][3] + row1_head[8  * C + ch] * BETA_S32[9 - y][2] + row2_head[8  * C + ch] * BETA_S32[9 - y][1] + row3_head[8  * C + ch] * BETA_S32[9 - y][0] + 536870912) >> 30);
            dst[(owidth - 3) * C + ch] = SaturateCast<Tp>((row0_head[9  * C + ch] * BETA_S32[9 - y][3] + row1_head[9  * C + ch] * BETA_S32[9 - y][2] + row2_head[9  * C + ch] * BETA_S32[9 - y][1] + row3_head[9  * C + ch] * BETA_S32[9 - y][0] + 536870912) >> 30);
            dst[(owidth - 2) * C + ch] = SaturateCast<Tp>((row0_head[10 * C + ch] * BETA_S32[9 - y][3] + row1_head[10 * C + ch] * BETA_S32[9 - y][2] + row2_head[10 * C + ch] * BETA_S32[9 - y][1] + row3_head[10 * C + ch] * BETA_S32[9 - y][0] + 536870912) >> 30);
            dst[(owidth - 1) * C + ch] = SaturateCast<Tp>((row0_head[11 * C + ch] * BETA_S32[9 - y][3] + row1_head[11 * C + ch] * BETA_S32[9 - y][2] + row2_head[11 * C + ch] * BETA_S32[9 - y][1] + row3_head[11 * C + ch] * BETA_S32[9 - y][0] + 536870912) >> 30);

            dst += (ostride / sizeof(Tp));
        }
    }
}

template<typename Tp, MI_S32 C>
static Status ResizeCuUpX2HvxImpl(const Mat &src, Mat &dst, ResizeCuFastVtcmBuffer vtcm_buffer, MI_S32 thread_num, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 iwidth    = src.GetSizes().m_width;
    MI_S32 iheight   = src.GetSizes().m_height;
    MI_S32 owidth    = dst.GetSizes().m_width;
    MI_S32 oheight   = dst.GetSizes().m_height;
    MI_S32 istride   = src.GetStrides().m_width;
    MI_S32 ostride   = dst.GetStrides().m_width;
    MI_S32 thread_id = SaturateCast<MI_S32>(static_cast<MI_F32>(start_row) * thread_num / iheight);
    MI_S32 head_size = AURA_HVLEN * ((owidth / iwidth) >> 1) * sizeof(MI_S32) * sizeof(Tp);
    MI_S32 back_size = AURA_HVLEN * (owidth / iwidth) * C * sizeof(MI_S32);
    MI_S32 row_size  = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN) + back_size;

    vtcm_buffer.row0_ptr = vtcm_buffer.row0_ptr + row_size  * thread_id;
    vtcm_buffer.row1_ptr = vtcm_buffer.row1_ptr + row_size  * thread_id;
    vtcm_buffer.row2_ptr = vtcm_buffer.row2_ptr + row_size  * thread_id;
    vtcm_buffer.row3_ptr = vtcm_buffer.row3_ptr + row_size  * thread_id;
    vtcm_buffer.row_head = vtcm_buffer.row_head + head_size * thread_id;

    MI_U64 l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 4, 0);
    MI_S32 y        = start_row;
    MI_S32 loop_row = end_row - 4 * (iheight == end_row);

    if (0 == y)
    {
        const Tp *src_c = src.Ptr<Tp>(0);
        Tp *dst_c       = dst.Ptr<Tp>(0);
        L2Fetch(reinterpret_cast<MI_U32>(src_c), l2fetch_param);
        ResizeCuUpX2ZeroRow<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, istride, ostride);
        y = 1;
    }
    else
    {
        const Tp *src_c = src.Ptr<Tp>(y);
        Tp *dst_c       = dst.Ptr<Tp>((y << 1) + 3);
        L2Fetch(reinterpret_cast<MI_U32>(src_c), l2fetch_param);
        ResizeCuUpX2UpBorder<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, istride, ostride);
        y++;
    }

    MI_S32 yofs   = (y << 1) + 3;
    l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 1, 0);
    L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 3)), l2fetch_param);

    for (; y < loop_row; y++)
    {
        if (y + 4 < loop_row)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 4)), l2fetch_param);
        }

        const Tp *src_c = src.Ptr<Tp>(y + 3);
        Tp *dst_c       = dst.Ptr<Tp>(yofs);
        ResizeCuUpX2Row<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, ostride);

        yofs += 2;
    }

    if (iheight == end_row)
    {
        const Tp *src_c = src.Ptr<Tp>(iheight - 1);
        Tp *dst_c       = dst.Ptr<Tp>(oheight - 5);
        ResizeCuUpX2BottomBorder<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, ostride);
    }

    return Status::OK;
}

template<typename Tp, MI_S32 C>
static Status ResizeCuUpX4HvxImpl(const Mat &src, Mat &dst, ResizeCuFastVtcmBuffer vtcm_buffer, MI_S32 thread_num, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 iwidth    = src.GetSizes().m_width;
    MI_S32 iheight   = src.GetSizes().m_height;
    MI_S32 owidth    = dst.GetSizes().m_width;
    MI_S32 oheight   = dst.GetSizes().m_height;
    MI_S32 istride   = src.GetStrides().m_width;
    MI_S32 ostride   = dst.GetStrides().m_width;
    MI_S32 thread_id = SaturateCast<MI_S32>(static_cast<MI_F32>(start_row) * thread_num / iheight);
    MI_S32 head_size = AURA_HVLEN * ((owidth / iwidth) >> 1) * sizeof(MI_S32) * sizeof(Tp);
    MI_S32 back_size = AURA_HVLEN * (owidth / iwidth) * C * sizeof(MI_S32);
    MI_S32 row_size  = AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN) + back_size;

    vtcm_buffer.row0_ptr = vtcm_buffer.row0_ptr + row_size  * thread_id;
    vtcm_buffer.row1_ptr = vtcm_buffer.row1_ptr + row_size  * thread_id;
    vtcm_buffer.row2_ptr = vtcm_buffer.row2_ptr + row_size  * thread_id;
    vtcm_buffer.row3_ptr = vtcm_buffer.row3_ptr + row_size  * thread_id;
    vtcm_buffer.row_head = vtcm_buffer.row_head + head_size * thread_id;

    MI_U64 l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 4, 0);
    MI_S32 y             = start_row;
    MI_S32 loop_row      = end_row - 4 * (iheight == end_row);

    if (0 == y)
    {
        const Tp *src_c = src.Ptr<Tp>(0);
        Tp *dst_c       = dst.Ptr<Tp>(0);
        L2Fetch(reinterpret_cast<MI_U32>(src_c), l2fetch_param);
        ResizeCuUpX4ZeroRow<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, istride, ostride);
        y = 1;
    }
    else
    {
        const Tp *src_c = src.Ptr<Tp>(y);
        Tp *dst_c = dst.Ptr<Tp>((y << 2) + 6);
        L2Fetch(reinterpret_cast<MI_U32>(src_c), l2fetch_param);
        ResizeCuUpX4UpBorder<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, istride, ostride);
        y++;
    }

    MI_S32 yofs   = (y << 2) + 6;
    l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 1, 0);
    L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 3)), l2fetch_param);

    for (; y < loop_row; y++)
    {
        if (y + 4 < loop_row)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 4)), l2fetch_param);
        }

        const Tp *src_c = src.Ptr<Tp>(y + 3);
        Tp *dst_c       = dst.Ptr<Tp>(yofs);
        ResizeCuUpX4Row<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, ostride);

        yofs += 4;
    }

    if (iheight == end_row)
    {
        const Tp *src_c = src.Ptr<Tp>(iheight - 1);
        Tp *dst_c       = dst.Ptr<Tp>(oheight - 10);
        ResizeCuUpX4BottomBorder<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, ostride);
    }

    return Status::OK;
}

template <typename Tp, MI_S32 C>
static Status ResizeCuFastUpHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool fail");
        return Status::ERROR;
    }

    Status ret               = Status::ERROR;
    MI_S32 iwidth            = src.GetSizes().m_width;
    MI_S32 owidth            = dst.GetSizes().m_width;
    MI_S32 thread_num        = wp->GetComputeThreadNum();
    MI_S32 head_buffer_size  = AURA_HVLEN * ((owidth / iwidth) >> 1) * sizeof(MI_S32) * sizeof(Tp) * thread_num;
    MI_S32 back_buffer_size  = AURA_HVLEN * (owidth / iwidth) * C * sizeof(MI_S32);
    MI_S32 row_buffer_size   = (AURA_ALIGN(owidth * C * sizeof(MI_S32), AURA_HVLEN) + back_buffer_size) * thread_num;
    MI_S32 total_buffer_size = head_buffer_size + row_buffer_size * 4;

    MI_U8 *vtcm_mem = static_cast<MI_U8*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_VTCM, total_buffer_size, AURA_HVLEN));
    if (MI_NULL == vtcm_mem)
    {
        AURA_ADD_ERROR_STRING(ctx, "alloc vtcm memory failed");
        AURA_FREE(ctx, vtcm_mem);
        return Status::ABORT;
    }

    struct ResizeCuFastVtcmBuffer vtcm_buffer;
    vtcm_buffer.row0_ptr = vtcm_mem;
    vtcm_buffer.row1_ptr = vtcm_buffer.row0_ptr + row_buffer_size;
    vtcm_buffer.row2_ptr = vtcm_buffer.row1_ptr + row_buffer_size;
    vtcm_buffer.row3_ptr = vtcm_buffer.row2_ptr + row_buffer_size;
    vtcm_buffer.row_head = vtcm_buffer.row3_ptr + row_buffer_size;

    if (owidth == 2 * iwidth)
    {
        ret = wp->ParallelFor((MI_S32)0, src.GetSizes().m_height, ResizeCuUpX2HvxImpl<Tp, C>, std::cref(src), std::ref(dst), vtcm_buffer, thread_num);
    }
    else if (owidth == 4 * iwidth)
    {
        ret = wp->ParallelFor((MI_S32)0, src.GetSizes().m_height, ResizeCuUpX4HvxImpl<Tp, C>, std::cref(src), std::ref(dst), vtcm_buffer, thread_num);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "only support scale 0.5, 0.25");
        AURA_FREE(ctx, vtcm_mem);
        return Status::ERROR;
    }

    AURA_FREE(ctx, vtcm_mem);
    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status ResizeCuFastUpHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret     = Status::ERROR;
    MI_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = ResizeCuFastUpHvxHelper<Tp, 1>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastHvxHelper failed for c1");
            }
            break;
        }

        case 2:
        {
            ret = ResizeCuFastUpHvxHelper<Tp, 2>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastHvxHelper failed for c2");
            }
            break;
        }

        case 3:
        {
            ret = ResizeCuFastUpHvxHelper<Tp, 3>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastHvxHelper failed for c3");
            }
            break;
        }

        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "only support channel 1,2,3");
        }
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeCuFastUpHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeCuFastUpHvxHelper<MI_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastUpHvxHelper run failed, type: MI_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeCuFastUpHvxHelper<MI_S8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastUpHvxHelper run failed, type: MI_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeCuFastUpHvxHelper<MI_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastUpHvxHelper run failed, type: MI_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeCuFastUpHvxHelper<MI_S16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastUpHvxHelper run failed, type: MI_S16");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type is not supported.");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
