#include "laplacian_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename St, typename std::enable_if<std::is_same<St, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian7x7Sum(const HVX_Vector &vu8_src_p_x0, const HVX_Vector &vu8_src_p_x1, const HVX_Vector &vu8_src_p_x2,
                                           const HVX_Vector &vu8_src_n_x0, const HVX_Vector &vu8_src_n_x1, const HVX_Vector &vu8_src_n_x2,
                                           HVX_VectorPair &wu16_sum_l2r2, HVX_VectorPair &wu16_sum_l1r1, HVX_VectorPair &wu16_sum_l0r0,
                                           HVX_VectorPair &wu16_sum_c, DT_S16 k0, DT_S16 k1, DT_S16 k2, DT_S16 k3)
{
    DT_U32 k0k0 = (k0 << 24) | (k0 << 16) | (k0 << 8) | k0;
    DT_U32 k1k1 = (k1 << 24) | (k1 << 16) | (k1 << 8) | k1;
    DT_U32 k2k2 = (k2 << 24) | (k2 << 16) | (k2 << 8) | k2;
    DT_U32 k3k3 = (k3 << 24) | (k3 << 16) | (k3 << 8) | k3;

    HVX_Vector vu8_src_p_l2 = Q6_V_vlalign_VVR(vu8_src_p_x1, vu8_src_p_x0, 3);
    HVX_Vector vu8_src_p_l1 = Q6_V_vlalign_VVR(vu8_src_p_x1, vu8_src_p_x0, 2);
    HVX_Vector vu8_src_p_l0 = Q6_V_vlalign_VVR(vu8_src_p_x1, vu8_src_p_x0, 1);
    HVX_Vector vu8_src_p_r0 = Q6_V_valign_VVR(vu8_src_p_x2, vu8_src_p_x1, 1);
    HVX_Vector vu8_src_p_r1 = Q6_V_valign_VVR(vu8_src_p_x2, vu8_src_p_x1, 2);
    HVX_Vector vu8_src_p_r2 = Q6_V_valign_VVR(vu8_src_p_x2, vu8_src_p_x1, 3);

    wu16_sum_l2r2 = Q6_Wuh_vmpy_VubRub(vu8_src_p_l2, k0k0);
    wu16_sum_l2r2 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l2r2, vu8_src_p_r2, k0k0);
    wu16_sum_l1r1 = Q6_Wuh_vmpy_VubRub(vu8_src_p_l1, k1k1);
    wu16_sum_l1r1 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_p_r1, k1k1);
    wu16_sum_l0r0 = Q6_Wuh_vmpy_VubRub(vu8_src_p_l0, k2k2);
    wu16_sum_l0r0 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l0r0, vu8_src_p_r0, k2k2);
    wu16_sum_c    = Q6_Wuh_vmpy_VubRub(vu8_src_p_x1, k3k3);

    HVX_Vector vu8_src_n_l2 = Q6_V_vlalign_VVR(vu8_src_n_x1, vu8_src_n_x0, 3);
    HVX_Vector vu8_src_n_l1 = Q6_V_vlalign_VVR(vu8_src_n_x1, vu8_src_n_x0, 2);
    HVX_Vector vu8_src_n_l0 = Q6_V_vlalign_VVR(vu8_src_n_x1, vu8_src_n_x0, 1);
    HVX_Vector vu8_src_n_r0 = Q6_V_valign_VVR(vu8_src_n_x2, vu8_src_n_x1, 1);
    HVX_Vector vu8_src_n_r1 = Q6_V_valign_VVR(vu8_src_n_x2, vu8_src_n_x1, 2);
    HVX_Vector vu8_src_n_r2 = Q6_V_valign_VVR(vu8_src_n_x2, vu8_src_n_x1, 3);

    wu16_sum_l2r2 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l2r2, vu8_src_n_l2, k0k0);
    wu16_sum_l2r2 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l2r2, vu8_src_n_r2, k0k0);
    wu16_sum_l1r1 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_n_l1, k1k1);
    wu16_sum_l1r1 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_n_r1, k1k1);
    wu16_sum_l0r0 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l0r0, vu8_src_n_l0, k2k2);
    wu16_sum_l0r0 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l0r0, vu8_src_n_r0, k2k2);
    wu16_sum_c    = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_c, vu8_src_n_x1, k3k3);
}

template <typename St, typename std::enable_if<std::is_same<St, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian7x7Sum(const HVX_Vector &vu16_src_p_x0, const HVX_Vector &vu16_src_p_x1, const HVX_Vector &vu16_src_p_x2,
                                           const HVX_Vector &vu16_src_n_x0, const HVX_Vector &vu16_src_n_x1, const HVX_Vector &vu16_src_n_x2,
                                           HVX_VectorPair &wu32_sum_l2r2, HVX_VectorPair &wu32_sum_l1r1, HVX_VectorPair &wu32_sum_l0r0,
                                           HVX_VectorPair &wu32_sum_c, DT_S16 k0, DT_S16 k1, DT_S16 k2, DT_S16 k3)
{
    DT_U32 k0k0 = k0 << 16 | k0;
    DT_U32 k1k1 = k1 << 16 | k1;
    DT_U32 k2k2 = k2 << 16 | k2;
    DT_U32 k3k3 = k3 << 16 | k3;

    HVX_Vector vu16_src_p_l2 = Q6_V_vlalign_VVR(vu16_src_p_x1, vu16_src_p_x0, 6);
    HVX_Vector vu16_src_p_l1 = Q6_V_vlalign_VVR(vu16_src_p_x1, vu16_src_p_x0, 4);
    HVX_Vector vu16_src_p_l0 = Q6_V_vlalign_VVR(vu16_src_p_x1, vu16_src_p_x0, 2);
    HVX_Vector vu16_src_p_r0 = Q6_V_valign_VVR(vu16_src_p_x2, vu16_src_p_x1, 2);
    HVX_Vector vu16_src_p_r1 = Q6_V_valign_VVR(vu16_src_p_x2, vu16_src_p_x1, 4);
    HVX_Vector vu16_src_p_r2 = Q6_V_valign_VVR(vu16_src_p_x2, vu16_src_p_x1, 6);

    wu32_sum_l2r2 = Q6_Wuw_vmpy_VuhRuh(vu16_src_p_l2, k0k0);
    wu32_sum_l2r2 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l2r2, vu16_src_p_r2, k0k0);
    wu32_sum_l1r1 = Q6_Wuw_vmpy_VuhRuh(vu16_src_p_l1, k1k1);
    wu32_sum_l1r1 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_p_r1, k1k1);
    wu32_sum_l0r0 = Q6_Wuw_vmpy_VuhRuh(vu16_src_p_l0, k2k2);
    wu32_sum_l0r0 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l0r0, vu16_src_p_r0, k2k2);
    wu32_sum_c    = Q6_Wuw_vmpy_VuhRuh(vu16_src_p_x1, k3k3);

    HVX_Vector vu16_src_n_l2 = Q6_V_vlalign_VVR(vu16_src_n_x1, vu16_src_n_x0, 6);
    HVX_Vector vu16_src_n_l1 = Q6_V_vlalign_VVR(vu16_src_n_x1, vu16_src_n_x0, 4);
    HVX_Vector vu16_src_n_l0 = Q6_V_vlalign_VVR(vu16_src_n_x1, vu16_src_n_x0, 2);
    HVX_Vector vu16_src_n_r0 = Q6_V_valign_VVR(vu16_src_n_x2, vu16_src_n_x1, 2);
    HVX_Vector vu16_src_n_r1 = Q6_V_valign_VVR(vu16_src_n_x2, vu16_src_n_x1, 4);
    HVX_Vector vu16_src_n_r2 = Q6_V_valign_VVR(vu16_src_n_x2, vu16_src_n_x1, 6);

    wu32_sum_l2r2 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l2r2, vu16_src_n_l2, k0k0);
    wu32_sum_l2r2 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l2r2, vu16_src_n_r2, k0k0);
    wu32_sum_l1r1 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_n_l1, k1k1);
    wu32_sum_l1r1 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_n_r1, k1k1);
    wu32_sum_l0r0 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l0r0, vu16_src_n_l0, k2k2);
    wu32_sum_l0r0 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l0r0, vu16_src_n_r0, k2k2);
    wu32_sum_c    = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_c, vu16_src_n_x1, k3k3);
}

template <typename St, typename std::enable_if<std::is_same<St, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian7x7Sum(const HVX_Vector &vs16_src_p_x0, const HVX_Vector &vs16_src_p_x1, const HVX_Vector &vs16_src_p_x2,
                                           const HVX_Vector &vs16_src_n_x0, const HVX_Vector &vs16_src_n_x1, const HVX_Vector &vs16_src_n_x2,
                                           HVX_VectorPair &ws32_sum_l2r2, HVX_VectorPair &ws32_sum_l1r1, HVX_VectorPair &ws32_sum_l0r0,
                                           HVX_VectorPair &ws32_sum_c, DT_S16 k0, DT_S16 k1, DT_S16 k2, DT_S16 k3)
{
    DT_U32 k0k0 = k0 << 16 | k0;
    DT_U32 k1k1 = k1 << 16 | k1;
    DT_U32 k2k2 = k2 << 16 | k2;
    DT_U32 k3k3 = k3 << 16 | k3;

    HVX_Vector vs16_src_p_l2 = Q6_V_vlalign_VVR(vs16_src_p_x1, vs16_src_p_x0, 6);
    HVX_Vector vs16_src_p_l1 = Q6_V_vlalign_VVR(vs16_src_p_x1, vs16_src_p_x0, 4);
    HVX_Vector vs16_src_p_l0 = Q6_V_vlalign_VVR(vs16_src_p_x1, vs16_src_p_x0, 2);
    HVX_Vector vs16_src_p_r0 = Q6_V_valign_VVR(vs16_src_p_x2, vs16_src_p_x1, 2);
    HVX_Vector vs16_src_p_r1 = Q6_V_valign_VVR(vs16_src_p_x2, vs16_src_p_x1, 4);
    HVX_Vector vs16_src_p_r2 = Q6_V_valign_VVR(vs16_src_p_x2, vs16_src_p_x1, 6);

    ws32_sum_l2r2 = Q6_Ww_vmpy_VhRh(vs16_src_p_l2, k0k0);
    ws32_sum_l2r2 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l2r2, vs16_src_p_r2, k0k0);
    ws32_sum_l1r1 = Q6_Ww_vmpy_VhRh(vs16_src_p_l1, k1k1);
    ws32_sum_l1r1 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_p_r1, k1k1);
    ws32_sum_l0r0 = Q6_Ww_vmpy_VhRh(vs16_src_p_l0, k2k2);
    ws32_sum_l0r0 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l0r0, vs16_src_p_r0, k2k2);
    ws32_sum_c    = Q6_Ww_vmpy_VhRh(vs16_src_p_x1, k3k3);

    HVX_Vector vs16_src_n_l2 = Q6_V_vlalign_VVR(vs16_src_n_x1, vs16_src_n_x0, 6);
    HVX_Vector vs16_src_n_l1 = Q6_V_vlalign_VVR(vs16_src_n_x1, vs16_src_n_x0, 4);
    HVX_Vector vs16_src_n_l0 = Q6_V_vlalign_VVR(vs16_src_n_x1, vs16_src_n_x0, 2);
    HVX_Vector vs16_src_n_r0 = Q6_V_valign_VVR(vs16_src_n_x2, vs16_src_n_x1, 2);
    HVX_Vector vs16_src_n_r1 = Q6_V_valign_VVR(vs16_src_n_x2, vs16_src_n_x1, 4);
    HVX_Vector vs16_src_n_r2 = Q6_V_valign_VVR(vs16_src_n_x2, vs16_src_n_x1, 6);

    ws32_sum_l2r2 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l2r2, vs16_src_n_l2, k0k0);
    ws32_sum_l2r2 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l2r2, vs16_src_n_r2, k0k0);
    ws32_sum_l1r1 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_n_l1, k1k1);
    ws32_sum_l1r1 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_n_r1, k1k1);
    ws32_sum_l0r0 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l0r0, vs16_src_n_l0, k2k2);
    ws32_sum_l0r0 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l0r0, vs16_src_n_r0, k2k2);
    ws32_sum_c    = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_c, vs16_src_n_x1, k3k3);
}

// using St = DT_U8
template <typename St, typename std::enable_if<std::is_same<St, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian7x7Core(const HVX_Vector &vu8_src_p2_x0, const HVX_Vector &vu8_src_p2_x1, const HVX_Vector &vu8_src_p2_x2,
                                            const HVX_Vector &vu8_src_p1_x0, const HVX_Vector &vu8_src_p1_x1, const HVX_Vector &vu8_src_p1_x2,
                                            const HVX_Vector &vu8_src_p0_x0, const HVX_Vector &vu8_src_p0_x1, const HVX_Vector &vu8_src_p0_x2,
                                            const HVX_Vector &vu8_src_c_x0,  const HVX_Vector &vu8_src_c_x1,  const HVX_Vector &vu8_src_c_x2,
                                            const HVX_Vector &vu8_src_n0_x0, const HVX_Vector &vu8_src_n0_x1, const HVX_Vector &vu8_src_n0_x2,
                                            const HVX_Vector &vu8_src_n1_x0, const HVX_Vector &vu8_src_n1_x1, const HVX_Vector &vu8_src_n1_x2,
                                            const HVX_Vector &vu8_src_n2_x0, const HVX_Vector &vu8_src_n2_x1, const HVX_Vector &vu8_src_n2_x2,
                                            HVX_Vector &vs16_dst_lo, HVX_Vector &vs16_dst_hi)
{
    HVX_VectorPair wu16_sum_l2r2, wu16_sum_l1r1, wu16_sum_l0r0, wu16_sum_c;
    HVX_VectorPair wu16_sum_p2n2, wu16_sum_p1n1, ws16_sum_p0n0, ws16_sum_c, ws16_sum;

    // p2n2
    Laplacian7x7Sum<St>(vu8_src_p2_x0, vu8_src_p2_x1, vu8_src_p2_x2, vu8_src_n2_x0, vu8_src_n2_x1, vu8_src_n2_x2,
                        wu16_sum_l2r2, wu16_sum_l1r1, wu16_sum_l0r0, wu16_sum_c, 1, 4, 7, 8);

    wu16_sum_p2n2 = Q6_Wh_vadd_WhWh(wu16_sum_l2r2, wu16_sum_l1r1);
    wu16_sum_p2n2 = Q6_Wh_vadd_WhWh(wu16_sum_p2n2, wu16_sum_l0r0);
    wu16_sum_p2n2 = Q6_Wh_vadd_WhWh(wu16_sum_p2n2, wu16_sum_c);

    // p1n1
    Laplacian7x7Sum<St>(vu8_src_p1_x0, vu8_src_p1_x1, vu8_src_p1_x2, vu8_src_n1_x0, vu8_src_n1_x1, vu8_src_n1_x2,
                        wu16_sum_l2r2, wu16_sum_l1r1, wu16_sum_l0r0, wu16_sum_c, 4, 12, 12, 8);

    wu16_sum_p1n1 = Q6_Wh_vadd_WhWh(wu16_sum_l2r2, wu16_sum_l1r1);
    wu16_sum_p1n1 = Q6_Wh_vadd_WhWh(wu16_sum_p1n1, wu16_sum_l0r0);
    wu16_sum_p1n1 = Q6_Wh_vadd_WhWh(wu16_sum_p1n1, wu16_sum_c);

    // p0n0
    Laplacian7x7Sum<St>(vu8_src_p0_x0, vu8_src_p0_x1, vu8_src_p0_x2, vu8_src_n0_x0, vu8_src_n0_x1, vu8_src_n0_x2,
                        wu16_sum_l2r2, wu16_sum_l1r1, wu16_sum_l0r0, wu16_sum_c, 7, 12, 15, 40);

    ws16_sum_p0n0 = Q6_Wh_vadd_WhWh(wu16_sum_l2r2, wu16_sum_l1r1);
    ws16_sum_p0n0 = Q6_Wh_vsub_WhWh(ws16_sum_p0n0, wu16_sum_l0r0);
    ws16_sum_p0n0 = Q6_Wh_vsub_WhWh(ws16_sum_p0n0, wu16_sum_c);

    // c
    HVX_Vector vu8_src_l2 = Q6_V_vlalign_VVR(vu8_src_c_x1, vu8_src_c_x0, 3);
    HVX_Vector vu8_src_l1 = Q6_V_vlalign_VVR(vu8_src_c_x1, vu8_src_c_x0, 2);
    HVX_Vector vu8_src_l0 = Q6_V_vlalign_VVR(vu8_src_c_x1, vu8_src_c_x0, 1);
    HVX_Vector vu8_src_r0 = Q6_V_valign_VVR(vu8_src_c_x2, vu8_src_c_x1, 1);
    HVX_Vector vu8_src_r1 = Q6_V_valign_VVR(vu8_src_c_x2, vu8_src_c_x1, 2);
    HVX_Vector vu8_src_r2 = Q6_V_valign_VVR(vu8_src_c_x2, vu8_src_c_x1, 3);

    wu16_sum_l2r2 = Q6_Wuh_vmpy_VubRub(vu8_src_l2, 0x08080808);
    wu16_sum_l2r2 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l2r2, vu8_src_r2, 0x08080808);
    wu16_sum_l1r1 = Q6_Wuh_vmpy_VubRub(vu8_src_l1, 0x08080808);
    wu16_sum_l1r1 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_r1, 0x08080808);
    wu16_sum_l0r0 = Q6_Wuh_vmpy_VubRub(vu8_src_l0, 0x28282828);
    wu16_sum_l0r0 = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l0r0, vu8_src_r0, 0x28282828);
    wu16_sum_c    = Q6_Wuh_vmpy_VubRub(vu8_src_c_x1, 0x50505050);

    ws16_sum_c = Q6_Wh_vadd_WhWh(wu16_sum_l2r2, wu16_sum_l1r1);
    ws16_sum_c = Q6_Wh_vsub_WhWh(ws16_sum_c, wu16_sum_l0r0);
    ws16_sum_c = Q6_Wh_vsub_WhWh(ws16_sum_c, wu16_sum_c);

    // sum
    ws16_sum    = Q6_Wh_vadd_WhWh(wu16_sum_p2n2, wu16_sum_p1n1);
    ws16_sum    = Q6_Wh_vadd_WhWh(ws16_sum, ws16_sum_p0n0);
    ws16_sum    = Q6_Wh_vadd_WhWh(ws16_sum, ws16_sum_c);
    ws16_sum    = Q6_W_vshuff_VVR(Q6_V_hi_W(ws16_sum), Q6_V_lo_W(ws16_sum), -2);
    vs16_dst_lo = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_sum), Q6_V_lo_W(ws16_sum));
    vs16_dst_hi = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_sum), Q6_V_hi_W(ws16_sum));
}

// using St = DT_U16
template <typename St, typename std::enable_if<std::is_same<St, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian7x7Core(const HVX_Vector &vu16_src_p2_x0, const HVX_Vector &vu16_src_p2_x1, const HVX_Vector &vu16_src_p2_x2,
                                            const HVX_Vector &vu16_src_p1_x0, const HVX_Vector &vu16_src_p1_x1, const HVX_Vector &vu16_src_p1_x2,
                                            const HVX_Vector &vu16_src_p0_x0, const HVX_Vector &vu16_src_p0_x1, const HVX_Vector &vu16_src_p0_x2,
                                            const HVX_Vector &vu16_src_c_x0,  const HVX_Vector &vu16_src_c_x1,  const HVX_Vector &vu16_src_c_x2,
                                            const HVX_Vector &vu16_src_n0_x0, const HVX_Vector &vu16_src_n0_x1, const HVX_Vector &vu16_src_n0_x2,
                                            const HVX_Vector &vu16_src_n1_x0, const HVX_Vector &vu16_src_n1_x1, const HVX_Vector &vu16_src_n1_x2,
                                            const HVX_Vector &vu16_src_n2_x0, const HVX_Vector &vu16_src_n2_x1, const HVX_Vector &vu16_src_n2_x2,
                                            HVX_Vector &vu16_dst)
{
    HVX_VectorPair wu32_sum_l2r2, wu32_sum_l1r1, wu32_sum_l0r0, wu32_sum_c;
    HVX_VectorPair wu32_sum_p2n2, wu32_sum_p1n1, ws32_sum_p0n0, ws32_sum_c, ws32_sum;

    // p2
    Laplacian7x7Sum<St>(vu16_src_p2_x0, vu16_src_p2_x1, vu16_src_p2_x2, vu16_src_n2_x0, vu16_src_n2_x1, vu16_src_n2_x2,
                        wu32_sum_l2r2, wu32_sum_l1r1, wu32_sum_l0r0, wu32_sum_c, 2, 8, 14, 16);

    wu32_sum_p2n2 = Q6_Ww_vadd_WwWw(wu32_sum_l2r2, wu32_sum_l1r1);
    wu32_sum_p2n2 = Q6_Ww_vadd_WwWw(wu32_sum_p2n2, wu32_sum_l0r0);
    wu32_sum_p2n2 = Q6_Ww_vadd_WwWw(wu32_sum_p2n2, wu32_sum_c);

    // p1n1
    Laplacian7x7Sum<St>(vu16_src_p1_x0, vu16_src_p1_x1, vu16_src_p1_x2, vu16_src_n1_x0, vu16_src_n1_x1, vu16_src_n1_x2,
                        wu32_sum_l2r2, wu32_sum_l1r1, wu32_sum_l0r0, wu32_sum_c, 8, 24, 24, 16);

    wu32_sum_p1n1 = Q6_Ww_vadd_WwWw(wu32_sum_l2r2, wu32_sum_l1r1);
    wu32_sum_p1n1 = Q6_Ww_vadd_WwWw(wu32_sum_p1n1, wu32_sum_l0r0);
    wu32_sum_p1n1 = Q6_Ww_vadd_WwWw(wu32_sum_p1n1, wu32_sum_c);

    // p0n0
    Laplacian7x7Sum<St>(vu16_src_p0_x0, vu16_src_p0_x1, vu16_src_p0_x2, vu16_src_n0_x0, vu16_src_n0_x1, vu16_src_n0_x2,
                        wu32_sum_l2r2, wu32_sum_l1r1, wu32_sum_l0r0, wu32_sum_c, 14, 24, 30, 80);

    ws32_sum_p0n0 = Q6_Ww_vadd_WwWw(wu32_sum_l2r2, wu32_sum_l1r1);
    ws32_sum_p0n0 = Q6_Ww_vsub_WwWw(ws32_sum_p0n0, wu32_sum_l0r0);
    ws32_sum_p0n0 = Q6_Ww_vsub_WwWw(ws32_sum_p0n0, wu32_sum_c);

    // c
    HVX_Vector vu16_src_l2 = Q6_V_vlalign_VVR(vu16_src_c_x1, vu16_src_c_x0, 6);
    HVX_Vector vu16_src_l1 = Q6_V_vlalign_VVR(vu16_src_c_x1, vu16_src_c_x0, 4);
    HVX_Vector vu16_src_l0 = Q6_V_vlalign_VVR(vu16_src_c_x1, vu16_src_c_x0, 2);
    HVX_Vector vu16_src_r0 = Q6_V_valign_VVR(vu16_src_c_x2, vu16_src_c_x1, 2);
    HVX_Vector vu16_src_r1 = Q6_V_valign_VVR(vu16_src_c_x2, vu16_src_c_x1, 4);
    HVX_Vector vu16_src_r2 = Q6_V_valign_VVR(vu16_src_c_x2, vu16_src_c_x1, 6);

    wu32_sum_l2r2 = Q6_Wuw_vmpy_VuhRuh(vu16_src_l2, 0x00100010);
    wu32_sum_l2r2 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l2r2, vu16_src_r2, 0x00100010);
    wu32_sum_l1r1 = Q6_Wuw_vmpy_VuhRuh(vu16_src_l1, 0x00100010);
    wu32_sum_l1r1 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_r1, 0x00100010);
    wu32_sum_l0r0 = Q6_Wuw_vmpy_VuhRuh(vu16_src_l0, 0x00500050);
    wu32_sum_l0r0 = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l0r0, vu16_src_r0, 0x00500050);
    wu32_sum_c    = Q6_Wuw_vmpy_VuhRuh(vu16_src_c_x1, 0x00a000a0);

    ws32_sum_c = Q6_Ww_vadd_WwWw(wu32_sum_l2r2, wu32_sum_l1r1);
    ws32_sum_c = Q6_Ww_vsub_WwWw(ws32_sum_c,    wu32_sum_l0r0);
    ws32_sum_c = Q6_Ww_vsub_WwWw(ws32_sum_c,    wu32_sum_c);

    // sum
    ws32_sum = Q6_Ww_vadd_WwWw(wu32_sum_p2n2, wu32_sum_p1n1);
    ws32_sum = Q6_Ww_vadd_WwWw(ws32_sum, ws32_sum_p0n0);
    ws32_sum = Q6_Ww_vadd_WwWw(ws32_sum, ws32_sum_c);
    ws32_sum = Q6_W_vshuff_VVR(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum), -4);
    vu16_dst = Q6_Vuh_vpack_VwVw_sat(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum));
}

// using St = DT_S16
template <typename St, typename std::enable_if<std::is_same<St, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian7x7Core(const HVX_Vector &vs16_src_p2_x0, const HVX_Vector &vs16_src_p2_x1, const HVX_Vector &vs16_src_p2_x2,
                                            const HVX_Vector &vs16_src_p1_x0, const HVX_Vector &vs16_src_p1_x1, const HVX_Vector &vs16_src_p1_x2,
                                            const HVX_Vector &vs16_src_p0_x0, const HVX_Vector &vs16_src_p0_x1, const HVX_Vector &vs16_src_p0_x2,
                                            const HVX_Vector &vs16_src_c_x0,  const HVX_Vector &vs16_src_c_x1,  const HVX_Vector &vs16_src_c_x2,
                                            const HVX_Vector &vs16_src_n0_x0, const HVX_Vector &vs16_src_n0_x1, const HVX_Vector &vs16_src_n0_x2,
                                            const HVX_Vector &vs16_src_n1_x0, const HVX_Vector &vs16_src_n1_x1, const HVX_Vector &vs16_src_n1_x2,
                                            const HVX_Vector &vs16_src_n2_x0, const HVX_Vector &vs16_src_n2_x1, const HVX_Vector &vs16_src_n2_x2,
                                            HVX_Vector &vs16_dst)
{
    HVX_VectorPair ws32_sum_l2r2, ws32_sum_l1r1, ws32_sum_l0r0, ws32_sum_c;
    HVX_VectorPair ws32_sum_p2n2, ws32_sum_p1n1, ws32_sum_p0n0, ws32_sum_c_c, ws32_sum;

    // p2n2
    Laplacian7x7Sum<St>(vs16_src_p2_x0, vs16_src_p2_x1, vs16_src_p2_x2, vs16_src_n2_x0, vs16_src_n2_x1, vs16_src_n2_x2,
                        ws32_sum_l2r2, ws32_sum_l1r1, ws32_sum_l0r0, ws32_sum_c, 2, 8, 14, 16);

    ws32_sum_p2n2 = Q6_Ww_vadd_WwWw(ws32_sum_l2r2, ws32_sum_l1r1);
    ws32_sum_p2n2 = Q6_Ww_vadd_WwWw(ws32_sum_p2n2, ws32_sum_l0r0);
    ws32_sum_p2n2 = Q6_Ww_vadd_WwWw(ws32_sum_p2n2, ws32_sum_c);

    // p1n1
    Laplacian7x7Sum<St>(vs16_src_p1_x0, vs16_src_p1_x1, vs16_src_p1_x2, vs16_src_n1_x0, vs16_src_n1_x1, vs16_src_n1_x2,
                        ws32_sum_l2r2, ws32_sum_l1r1, ws32_sum_l0r0, ws32_sum_c, 8, 24, 24, 16);

    ws32_sum_p1n1 = Q6_Ww_vadd_WwWw(ws32_sum_l2r2, ws32_sum_l1r1);
    ws32_sum_p1n1 = Q6_Ww_vadd_WwWw(ws32_sum_p1n1, ws32_sum_l0r0);
    ws32_sum_p1n1 = Q6_Ww_vadd_WwWw(ws32_sum_p1n1, ws32_sum_c);

    Laplacian7x7Sum<St>(vs16_src_p0_x0, vs16_src_p0_x1, vs16_src_p0_x2, vs16_src_n0_x0, vs16_src_n0_x1, vs16_src_n0_x2,
                        ws32_sum_l2r2, ws32_sum_l1r1, ws32_sum_l0r0, ws32_sum_c, 14, 24, 30, 80);

    ws32_sum_p0n0 = Q6_Ww_vadd_WwWw(ws32_sum_l2r2, ws32_sum_l1r1);
    ws32_sum_p0n0 = Q6_Ww_vsub_WwWw(ws32_sum_p0n0, ws32_sum_l0r0);
    ws32_sum_p0n0 = Q6_Ww_vsub_WwWw(ws32_sum_p0n0, ws32_sum_c);

    // c
    HVX_Vector vs16_src_l2 = Q6_V_vlalign_VVR(vs16_src_c_x1, vs16_src_c_x0, 6);
    HVX_Vector vs16_src_l1 = Q6_V_vlalign_VVR(vs16_src_c_x1, vs16_src_c_x0, 4);
    HVX_Vector vs16_src_l0 = Q6_V_vlalign_VVR(vs16_src_c_x1, vs16_src_c_x0, 2);
    HVX_Vector vs16_src_r0 = Q6_V_valign_VVR(vs16_src_c_x2, vs16_src_c_x1, 2);
    HVX_Vector vs16_src_r1 = Q6_V_valign_VVR(vs16_src_c_x2, vs16_src_c_x1, 4);
    HVX_Vector vs16_src_r2 = Q6_V_valign_VVR(vs16_src_c_x2, vs16_src_c_x1, 6);

    ws32_sum_l2r2 = Q6_Ww_vmpy_VhRh(vs16_src_l2, 0x00100010);
    ws32_sum_l2r2 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l2r2, vs16_src_r2, 0x00100010);
    ws32_sum_l1r1 = Q6_Ww_vmpy_VhRh(vs16_src_l1, 0x00100010);
    ws32_sum_l1r1 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_r1, 0x00100010);
    ws32_sum_l0r0 = Q6_Ww_vmpy_VhRh(vs16_src_l0, 0x00500050);
    ws32_sum_l0r0 = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l0r0, vs16_src_r0, 0x00500050);
    ws32_sum_c    = Q6_Ww_vmpy_VhRh(vs16_src_c_x1, 0x00a000a0);

    ws32_sum_c_c  = Q6_Ww_vadd_WwWw(ws32_sum_l2r2, ws32_sum_l1r1);
    ws32_sum_c_c  = Q6_Ww_vsub_WwWw(ws32_sum_c_c,  ws32_sum_l0r0);
    ws32_sum_c_c  = Q6_Ww_vsub_WwWw(ws32_sum_c_c,  ws32_sum_c);

    // sum
    ws32_sum = Q6_Ww_vadd_WwWw(ws32_sum_p2n2, ws32_sum_p1n1);
    ws32_sum = Q6_Ww_vadd_WwWw(ws32_sum, ws32_sum_p0n0);
    ws32_sum = Q6_Ww_vadd_WwWw(ws32_sum, ws32_sum_c_c);
    ws32_sum = Q6_W_vshuff_VVR(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum), -4);
    vs16_dst = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum));
}

template <typename St, typename std::enable_if<std::is_same<St, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian7x7Core(const HVX_Vector &vu8_src_p2_x0, const HVX_Vector &vu8_src_p2_x1, const HVX_Vector &vu8_src_p2_x2, const HVX_Vector &vu8_border_p2,
                                            const HVX_Vector &vu8_src_p1_x0, const HVX_Vector &vu8_src_p1_x1, const HVX_Vector &vu8_src_p1_x2, const HVX_Vector &vu8_border_p1,
                                            const HVX_Vector &vu8_src_p0_x0, const HVX_Vector &vu8_src_p0_x1, const HVX_Vector &vu8_src_p0_x2, const HVX_Vector &vu8_border_p0,
                                            const HVX_Vector &vu8_src_c_x0,  const HVX_Vector &vu8_src_c_x1,  const HVX_Vector &vu8_src_c_x2,  const HVX_Vector &vu8_border_c,
                                            const HVX_Vector &vu8_src_n0_x0, const HVX_Vector &vu8_src_n0_x1, const HVX_Vector &vu8_src_n0_x2, const HVX_Vector &vu8_border_n0,
                                            const HVX_Vector &vu8_src_n1_x0, const HVX_Vector &vu8_src_n1_x1, const HVX_Vector &vu8_src_n1_x2, const HVX_Vector &vu8_border_n1,
                                            const HVX_Vector &vu8_src_n2_x0, const HVX_Vector &vu8_src_n2_x1, const HVX_Vector &vu8_src_n2_x2, const HVX_Vector &vu8_border_n2,
                                            HVX_Vector &vs16_dst_x0_lo, HVX_Vector &vs16_dst_x0_hi, HVX_Vector &vs16_dst_x1_lo, HVX_Vector &vs16_dst_x1_hi,
                                            DT_S32 align_size)
{
    HVX_Vector vu8_src_p2_r = Q6_V_vlalign_VVR(vu8_border_p2, vu8_src_p2_x2, align_size);
    HVX_Vector vu8_src_p1_r = Q6_V_vlalign_VVR(vu8_border_p1, vu8_src_p1_x2, align_size);
    HVX_Vector vu8_src_p0_r = Q6_V_vlalign_VVR(vu8_border_p0, vu8_src_p0_x2, align_size);
    HVX_Vector vu8_src_c_r  = Q6_V_vlalign_VVR(vu8_border_c,  vu8_src_c_x2,  align_size);
    HVX_Vector vu8_src_n0_r = Q6_V_vlalign_VVR(vu8_border_n0, vu8_src_n0_x2, align_size);
    HVX_Vector vu8_src_n1_r = Q6_V_vlalign_VVR(vu8_border_n1, vu8_src_n1_x2, align_size);
    HVX_Vector vu8_src_n2_r = Q6_V_vlalign_VVR(vu8_border_n2, vu8_src_n2_x2, align_size);
    Laplacian7x7Core<St>(vu8_src_p2_x0, vu8_src_p2_x1, vu8_src_p2_r,
                         vu8_src_p1_x0, vu8_src_p1_x1, vu8_src_p1_r,
                         vu8_src_p0_x0, vu8_src_p0_x1, vu8_src_p0_r,
                         vu8_src_c_x0,  vu8_src_c_x1,  vu8_src_c_r,
                         vu8_src_n0_x0, vu8_src_n0_x1, vu8_src_n0_r,
                         vu8_src_n1_x0, vu8_src_n1_x1, vu8_src_n1_r,
                         vu8_src_n2_x0, vu8_src_n2_x1, vu8_src_n2_r,
                         vs16_dst_x0_lo, vs16_dst_x0_hi);

    HVX_Vector vu8_src_p2_l = Q6_V_valign_VVR(vu8_src_p2_x1, vu8_src_p2_x0, align_size);
    HVX_Vector vu8_src_p1_l = Q6_V_valign_VVR(vu8_src_p1_x1, vu8_src_p1_x0, align_size);
    HVX_Vector vu8_src_p0_l = Q6_V_valign_VVR(vu8_src_p0_x1, vu8_src_p0_x0, align_size);
    HVX_Vector vu8_src_c_l  = Q6_V_valign_VVR(vu8_src_c_x1,  vu8_src_c_x0,  align_size);
    HVX_Vector vu8_src_n0_l = Q6_V_valign_VVR(vu8_src_n0_x1, vu8_src_n0_x0, align_size);
    HVX_Vector vu8_src_n1_l = Q6_V_valign_VVR(vu8_src_n1_x1, vu8_src_n1_x0, align_size);
    HVX_Vector vu8_src_n2_l = Q6_V_valign_VVR(vu8_src_n2_x1, vu8_src_n2_x0, align_size);
    Laplacian7x7Core<St>(vu8_src_p2_l, vu8_src_p2_x2, vu8_border_p2,
                         vu8_src_p1_l, vu8_src_p1_x2, vu8_border_p1,
                         vu8_src_p0_l, vu8_src_p0_x2, vu8_border_p0,
                         vu8_src_c_l,  vu8_src_c_x2,  vu8_border_c,
                         vu8_src_n0_l, vu8_src_n0_x2, vu8_border_n0,
                         vu8_src_n1_l, vu8_src_n1_x2, vu8_border_n1,
                         vu8_src_n2_l, vu8_src_n2_x2, vu8_border_n2,
                         vs16_dst_x1_lo, vs16_dst_x1_hi);
}

template <typename St, typename std::enable_if<std::is_same<St, DT_U16>::value || std::is_same<St, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian7x7Core(const HVX_Vector &vd16_src_p2_x0, const HVX_Vector &vd16_src_p2_x1, const HVX_Vector &vd16_src_p2_x2, const HVX_Vector &vd16_border_p2,
                                            const HVX_Vector &vd16_src_p1_x0, const HVX_Vector &vd16_src_p1_x1, const HVX_Vector &vd16_src_p1_x2, const HVX_Vector &vd16_border_p1,
                                            const HVX_Vector &vd16_src_p0_x0, const HVX_Vector &vd16_src_p0_x1, const HVX_Vector &vd16_src_p0_x2, const HVX_Vector &vd16_border_p0,
                                            const HVX_Vector &vd16_src_c_x0,  const HVX_Vector &vd16_src_c_x1,  const HVX_Vector &vd16_src_c_x2,  const HVX_Vector &vd16_border_c,
                                            const HVX_Vector &vd16_src_n0_x0, const HVX_Vector &vd16_src_n0_x1, const HVX_Vector &vd16_src_n0_x2, const HVX_Vector &vd16_border_n0,
                                            const HVX_Vector &vd16_src_n1_x0, const HVX_Vector &vd16_src_n1_x1, const HVX_Vector &vd16_src_n1_x2, const HVX_Vector &vd16_border_n1,
                                            const HVX_Vector &vd16_src_n2_x0, const HVX_Vector &vd16_src_n2_x1, const HVX_Vector &vd16_src_n2_x2, const HVX_Vector &vd16_border_n2,
                                            HVX_Vector &vd16_dst_x0, HVX_Vector &vd16_dst_x1,
                                            DT_S32 align_size)
{
    HVX_Vector vd16_src_p2_c = Q6_V_vlalign_VVR(vd16_border_p2, vd16_src_p2_x2, align_size);
    HVX_Vector vd16_src_p1_c = Q6_V_vlalign_VVR(vd16_border_p1, vd16_src_p1_x2, align_size);
    HVX_Vector vd16_src_p0_c = Q6_V_vlalign_VVR(vd16_border_p0, vd16_src_p0_x2, align_size);
    HVX_Vector vd16_src_c_c  = Q6_V_vlalign_VVR(vd16_border_c,  vd16_src_c_x2,  align_size);
    HVX_Vector vd16_src_n0_c = Q6_V_vlalign_VVR(vd16_border_n0, vd16_src_n0_x2, align_size);
    HVX_Vector vd16_src_n1_c = Q6_V_vlalign_VVR(vd16_border_n1, vd16_src_n1_x2, align_size);
    HVX_Vector vd16_src_n2_c = Q6_V_vlalign_VVR(vd16_border_n2, vd16_src_n2_x2, align_size);
    Laplacian7x7Core<St>(vd16_src_p2_x0, vd16_src_p2_x1, vd16_src_p2_c,
                         vd16_src_p1_x0, vd16_src_p1_x1, vd16_src_p1_c,
                         vd16_src_p0_x0, vd16_src_p0_x1, vd16_src_p0_c,
                         vd16_src_c_x0,  vd16_src_c_x1,  vd16_src_c_c,
                         vd16_src_n0_x0, vd16_src_n0_x1, vd16_src_n0_c,
                         vd16_src_n1_x0, vd16_src_n1_x1, vd16_src_n1_c,
                         vd16_src_n2_x0, vd16_src_n2_x1, vd16_src_n2_c,
                         vd16_dst_x0);

    HVX_Vector vd16_src_p2_r0 = Q6_V_valign_VVR(vd16_src_p2_x1, vd16_src_p2_x0, align_size);
    HVX_Vector vd16_src_p1_r0 = Q6_V_valign_VVR(vd16_src_p1_x1, vd16_src_p1_x0, align_size);
    HVX_Vector vd16_src_p0_r0 = Q6_V_valign_VVR(vd16_src_p0_x1, vd16_src_p0_x0, align_size);
    HVX_Vector vd16_src_c_r0  = Q6_V_valign_VVR(vd16_src_c_x1,  vd16_src_c_x0,  align_size);
    HVX_Vector vd16_src_n0_r0 = Q6_V_valign_VVR(vd16_src_n0_x1, vd16_src_n0_x0, align_size);
    HVX_Vector vd16_src_n1_r0 = Q6_V_valign_VVR(vd16_src_n1_x1, vd16_src_n1_x0, align_size);
    HVX_Vector vd16_src_n2_r0 = Q6_V_valign_VVR(vd16_src_n2_x1, vd16_src_n2_x0, align_size);
    Laplacian7x7Core<St>(vd16_src_p2_r0, vd16_src_p2_x2, vd16_border_p2,
                         vd16_src_p1_r0, vd16_src_p1_x2, vd16_border_p1,
                         vd16_src_p0_r0, vd16_src_p0_x2, vd16_border_p0,
                         vd16_src_c_r0,  vd16_src_c_x2,  vd16_border_c,
                         vd16_src_n0_r0, vd16_src_n0_x2, vd16_border_n0,
                         vd16_src_n1_r0, vd16_src_n1_x2, vd16_border_n1,
                         vd16_src_n2_r0, vd16_src_n2_x2, vd16_border_n2,
                         vd16_dst_x1);
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C,
          typename std::enable_if<std::is_same<St, DT_U8>::value>::type* = DT_NULL>
static DT_VOID Laplacian7x7Row(const St *src_p2, const St *src_p1, const St *src_p0, const St *src_c,
                               const St *src_n0, const St *src_n1, const St *src_n2, Dt *dst_c,
                               const std::vector<St> &border_value, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    constexpr DT_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(St);
    DT_S32 back_offset = width - ELEM_COUNTS;

    MVType mvu8_src_p2[3], mvu8_src_p1[3], mvu8_src_p0[3], mvu8_src_c[3], mvu8_src_n0[3], mvu8_src_n1[3], mvu8_src_n2[3];
    MVType mvs16_dst_lo, mvs16_dst_hi;

    // left border
    {
        vload(src_p2, mvu8_src_p2[1]);
        vload(src_p1, mvu8_src_p1[1]);
        vload(src_p0, mvu8_src_p0[1]);
        vload(src_c,  mvu8_src_c[1]);
        vload(src_n0, mvu8_src_n0[1]);
        vload(src_n1, mvu8_src_n1[1]);
        vload(src_n2, mvu8_src_n2[1]);

        #pragma unroll
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvu8_src_p2[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_p2[1].val[ch], src_p2[ch], border_value[ch]);
            mvu8_src_p1[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mvu8_src_p0[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mvu8_src_c[0].val[ch]  = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_c[1].val[ch],  src_c[ch],  border_value[ch]);
            mvu8_src_n0[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mvu8_src_n1[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_n1[1].val[ch], src_n1[ch], border_value[ch]);
            mvu8_src_n2[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_n2[1].val[ch], src_n2[ch], border_value[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (DT_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS)
        {
            vload(src_p2 + C * x, mvu8_src_p2[2]);
            vload(src_p1 + C * x, mvu8_src_p1[2]);
            vload(src_p0 + C * x, mvu8_src_p0[2]);
            vload(src_c  + C * x, mvu8_src_c[2]);
            vload(src_n0 + C * x, mvu8_src_n0[2]);
            vload(src_n1 + C * x, mvu8_src_n1[2]);
            vload(src_n2 + C * x, mvu8_src_n2[2]);

            #pragma unroll
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Laplacian7x7Core<St>(mvu8_src_p2[0].val[ch], mvu8_src_p2[1].val[ch], mvu8_src_p2[2].val[ch],
                                     mvu8_src_p1[0].val[ch], mvu8_src_p1[1].val[ch], mvu8_src_p1[2].val[ch],
                                     mvu8_src_p0[0].val[ch], mvu8_src_p0[1].val[ch], mvu8_src_p0[2].val[ch],
                                     mvu8_src_c[0].val[ch],  mvu8_src_c[1].val[ch],  mvu8_src_c[2].val[ch],
                                     mvu8_src_n0[0].val[ch], mvu8_src_n0[1].val[ch], mvu8_src_n0[2].val[ch],
                                     mvu8_src_n1[0].val[ch], mvu8_src_n1[1].val[ch], mvu8_src_n1[2].val[ch],
                                     mvu8_src_n2[0].val[ch], mvu8_src_n2[1].val[ch], mvu8_src_n2[2].val[ch],
                                     mvs16_dst_lo.val[ch], mvs16_dst_hi.val[ch]);
            }

            vstore(dst_c + C * (x - ELEM_COUNTS),        mvs16_dst_lo);
            vstore(dst_c + C * (x - (ELEM_COUNTS >> 1)), mvs16_dst_hi);

            mvu8_src_p2[0] = mvu8_src_p2[1];
            mvu8_src_p1[0] = mvu8_src_p1[1];
            mvu8_src_p0[0] = mvu8_src_p0[1];
            mvu8_src_c[0]  = mvu8_src_c[1];
            mvu8_src_n0[0] = mvu8_src_n0[1];
            mvu8_src_n1[0] = mvu8_src_n1[1];
            mvu8_src_n2[0] = mvu8_src_n2[1];

            mvu8_src_p2[1] = mvu8_src_p2[2];
            mvu8_src_p1[1] = mvu8_src_p1[2];
            mvu8_src_p0[1] = mvu8_src_p0[2];
            mvu8_src_c[1]  = mvu8_src_c[2];
            mvu8_src_n0[1] = mvu8_src_n0[2];
            mvu8_src_n1[1] = mvu8_src_n1[2];
            mvu8_src_n2[1] = mvu8_src_n2[2];
        }
    }

    // remain
    {
        DT_S32 last = (width - 1) * C;
        DT_S32 rest = width % ELEM_COUNTS;
        MVType mvs16_last_l, mvs16_last_h;

        vload(src_p2 + C * back_offset, mvu8_src_p2[2]);
        vload(src_p1 + C * back_offset, mvu8_src_p1[2]);
        vload(src_p0 + C * back_offset, mvu8_src_p0[2]);
        vload(src_c  + C * back_offset, mvu8_src_c[2]);
        vload(src_n0 + C * back_offset, mvu8_src_n0[2]);
        vload(src_n1 + C * back_offset, mvu8_src_n1[2]);
        vload(src_n2 + C * back_offset, mvu8_src_n2[2]);

        #pragma unroll
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector vu8_border_p2 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_p2[2].val[ch], src_p2[last + ch], border_value[ch]);
            HVX_Vector vu8_border_p1 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_p1[2].val[ch], src_p1[last + ch], border_value[ch]);
            HVX_Vector vu8_border_p0 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_p0[2].val[ch], src_p0[last + ch], border_value[ch]);
            HVX_Vector vu8_border_c  = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_c[2].val[ch],  src_c[last + ch],  border_value[ch]);
            HVX_Vector vu8_border_n0 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_n0[2].val[ch], src_n0[last + ch], border_value[ch]);
            HVX_Vector vu8_border_n1 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_n1[2].val[ch], src_n1[last + ch], border_value[ch]);
            HVX_Vector vu8_border_n2 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_n2[2].val[ch], src_n2[last + ch], border_value[ch]);

            Laplacian7x7Core<St>(mvu8_src_p2[0].val[ch], mvu8_src_p2[1].val[ch], mvu8_src_p2[2].val[ch], vu8_border_p2,
                                 mvu8_src_p1[0].val[ch], mvu8_src_p1[1].val[ch], mvu8_src_p1[2].val[ch], vu8_border_p1,
                                 mvu8_src_p0[0].val[ch], mvu8_src_p0[1].val[ch], mvu8_src_p0[2].val[ch], vu8_border_p0,
                                 mvu8_src_c[0].val[ch],  mvu8_src_c[1].val[ch],  mvu8_src_c[2].val[ch],  vu8_border_c,
                                 mvu8_src_n0[0].val[ch], mvu8_src_n0[1].val[ch], mvu8_src_n0[2].val[ch], vu8_border_n0,
                                 mvu8_src_n1[0].val[ch], mvu8_src_n1[1].val[ch], mvu8_src_n1[2].val[ch], vu8_border_n1,
                                 mvu8_src_n2[0].val[ch], mvu8_src_n2[1].val[ch], mvu8_src_n2[2].val[ch], vu8_border_n2,
                                 mvs16_dst_lo.val[ch],   mvs16_dst_hi.val[ch],   mvs16_last_l.val[ch],   mvs16_last_h.val[ch],
                                 rest * sizeof(St));
        }

        vstore(dst_c + C * (back_offset - rest),                      mvs16_dst_lo);
        vstore(dst_c + C * (back_offset - rest + (ELEM_COUNTS >> 1)), mvs16_dst_hi);
        vstore(dst_c + C * back_offset,                               mvs16_last_l);
        vstore(dst_c + C * (back_offset + (ELEM_COUNTS >> 1)),        mvs16_last_h);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C,
          typename std::enable_if<std::is_same<St, DT_U16>::value || std::is_same<St, DT_S16>::value>::type* = DT_NULL>
static DT_VOID Laplacian7x7Row(const St *src_p2, const St *src_p1, const St *src_p0, const St *src_c,
                               const St *src_n0, const St *src_n1, const St *src_n2, Dt *dst_c,
                               const std::vector<St> &border_value, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    constexpr DT_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(St);
    DT_S32 back_offset = width - ELEM_COUNTS;

    MVType mvd16_src_p2[3], mvd16_src_p1[3], mvd16_src_p0[3], mvd16_src_c[3], mvd16_src_n0[3], mvd16_src_n1[3], mvd16_src_n2[3];
    MVType mvd16_dst;

    // left border
    {
        vload(src_p2, mvd16_src_p2[1]);
        vload(src_p1, mvd16_src_p1[1]);
        vload(src_p0, mvd16_src_p0[1]);
        vload(src_c,  mvd16_src_c[1]);
        vload(src_n0, mvd16_src_n0[1]);
        vload(src_n1, mvd16_src_n1[1]);
        vload(src_n2, mvd16_src_n2[1]);

        #pragma unroll
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvd16_src_p2[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_p2[1].val[ch], src_p2[ch], border_value[ch]);
            mvd16_src_p1[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mvd16_src_p0[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mvd16_src_c[0].val[ch]  = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_c[1].val[ch],  src_c[ch],  border_value[ch]);
            mvd16_src_n0[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mvd16_src_n1[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_n1[1].val[ch], src_n1[ch], border_value[ch]);
            mvd16_src_n2[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_n2[1].val[ch], src_n2[ch], border_value[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (DT_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS)
        {
            vload(src_p2 + C * x, mvd16_src_p2[2]);
            vload(src_p1 + C * x, mvd16_src_p1[2]);
            vload(src_p0 + C * x, mvd16_src_p0[2]);
            vload(src_c  + C * x, mvd16_src_c[2]);
            vload(src_n0 + C * x, mvd16_src_n0[2]);
            vload(src_n1 + C * x, mvd16_src_n1[2]);
            vload(src_n2 + C * x, mvd16_src_n2[2]);

            #pragma unroll
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Laplacian7x7Core<St>(mvd16_src_p2[0].val[ch], mvd16_src_p2[1].val[ch], mvd16_src_p2[2].val[ch],
                                     mvd16_src_p1[0].val[ch], mvd16_src_p1[1].val[ch], mvd16_src_p1[2].val[ch],
                                     mvd16_src_p0[0].val[ch], mvd16_src_p0[1].val[ch], mvd16_src_p0[2].val[ch],
                                     mvd16_src_c[0].val[ch],  mvd16_src_c[1].val[ch],  mvd16_src_c[2].val[ch],
                                     mvd16_src_n0[0].val[ch], mvd16_src_n0[1].val[ch], mvd16_src_n0[2].val[ch],
                                     mvd16_src_n1[0].val[ch], mvd16_src_n1[1].val[ch], mvd16_src_n1[2].val[ch],
                                     mvd16_src_n2[0].val[ch], mvd16_src_n2[1].val[ch], mvd16_src_n2[2].val[ch],
                                     mvd16_dst.val[ch]);
            }

            vstore(dst_c + C * (x - ELEM_COUNTS), mvd16_dst);

            mvd16_src_p2[0] = mvd16_src_p2[1];
            mvd16_src_p1[0] = mvd16_src_p1[1];
            mvd16_src_p0[0] = mvd16_src_p0[1];
            mvd16_src_c[0]  = mvd16_src_c[1];
            mvd16_src_n0[0] = mvd16_src_n0[1];
            mvd16_src_n1[0] = mvd16_src_n1[1];
            mvd16_src_n2[0] = mvd16_src_n2[1];

            mvd16_src_p2[1] = mvd16_src_p2[2];
            mvd16_src_p1[1] = mvd16_src_p1[2];
            mvd16_src_p0[1] = mvd16_src_p0[2];
            mvd16_src_c[1]  = mvd16_src_c[2];
            mvd16_src_n0[1] = mvd16_src_n0[2];
            mvd16_src_n1[1] = mvd16_src_n1[2];
            mvd16_src_n2[1] = mvd16_src_n2[2];
        }
    }

    // remain
    {
        DT_S32 last = (width - 1) * C;
        DT_S32 rest = width % ELEM_COUNTS;
        MVType mvd16_last;

        vload(src_p2 + C * back_offset, mvd16_src_p2[2]);
        vload(src_p1 + C * back_offset, mvd16_src_p1[2]);
        vload(src_p0 + C * back_offset, mvd16_src_p0[2]);
        vload(src_c  + C * back_offset, mvd16_src_c[2]);
        vload(src_n0 + C * back_offset, mvd16_src_n0[2]);
        vload(src_n1 + C * back_offset, mvd16_src_n1[2]);
        vload(src_n2 + C * back_offset, mvd16_src_n2[2]);

        #pragma unroll
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector vd16_border_p2 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_p2[2].val[ch], src_p2[last + ch], border_value[ch]);
            HVX_Vector vd16_border_p1 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_p1[2].val[ch], src_p1[last + ch], border_value[ch]);
            HVX_Vector vd16_border_p0 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_p0[2].val[ch], src_p0[last + ch], border_value[ch]);
            HVX_Vector vd16_border_c  = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_c[2].val[ch],  src_c[last + ch],  border_value[ch]);
            HVX_Vector vd16_border_n0 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_n0[2].val[ch], src_n0[last + ch], border_value[ch]);
            HVX_Vector vd16_border_n1 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_n1[2].val[ch], src_n1[last + ch], border_value[ch]);
            HVX_Vector vd16_border_n2 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_n2[2].val[ch], src_n2[last + ch], border_value[ch]);

            Laplacian7x7Core<St>(mvd16_src_p2[0].val[ch], mvd16_src_p2[1].val[ch], mvd16_src_p2[2].val[ch], vd16_border_p2,
                                 mvd16_src_p1[0].val[ch], mvd16_src_p1[1].val[ch], mvd16_src_p1[2].val[ch], vd16_border_p1,
                                 mvd16_src_p0[0].val[ch], mvd16_src_p0[1].val[ch], mvd16_src_p0[2].val[ch], vd16_border_p0,
                                 mvd16_src_c[0].val[ch],  mvd16_src_c[1].val[ch],  mvd16_src_c[2].val[ch],  vd16_border_c,
                                 mvd16_src_n0[0].val[ch], mvd16_src_n0[1].val[ch], mvd16_src_n0[2].val[ch], vd16_border_n0,
                                 mvd16_src_n1[0].val[ch], mvd16_src_n1[1].val[ch], mvd16_src_n1[2].val[ch], vd16_border_n1,
                                 mvd16_src_n2[0].val[ch], mvd16_src_n2[1].val[ch], mvd16_src_n2[2].val[ch], vd16_border_n2,
                                 mvd16_dst.val[ch],       mvd16_last.val[ch],
                                 rest * sizeof(St));
        }

        vstore(dst_c + C * (back_offset - rest), mvd16_dst);
        vstore(dst_c + C * back_offset,          mvd16_last);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C>
static Status Laplacian7x7HvxImpl(const Mat &src, Mat &dst, const std::vector<St> &border_value,
                                  const St *border_buffer, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width   = src.GetSizes().m_width;
    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    const St *src_p2 = src.Ptr<St, BORDER_TYPE>(start_row - 3, border_buffer);
    const St *src_p1 = src.Ptr<St, BORDER_TYPE>(start_row - 2, border_buffer);
    const St *src_p0 = src.Ptr<St, BORDER_TYPE>(start_row - 1, border_buffer);
    const St *src_c  = src.Ptr<St>(start_row);
    const St *src_n0 = src.Ptr<St, BORDER_TYPE>(start_row + 1, border_buffer);
    const St *src_n1 = src.Ptr<St, BORDER_TYPE>(start_row + 2, border_buffer);
    const St *src_n2 = src.Ptr<St, BORDER_TYPE>(start_row + 3, border_buffer);

    DT_U64 L2fetch_row_param = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if (y + 4 < height)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<St>(y + 4)), L2fetch_row_param);
        }

        Dt *dst_c = dst.Ptr<Dt>(y);
        Laplacian7x7Row<St, Dt, BORDER_TYPE, C>(src_p2, src_p1, src_p0, src_c, src_n0, src_n1, src_n2, dst_c, border_value, width);

        src_p2 = src_p1;
        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src_n2;
        src_n2 = src.Ptr<St, BORDER_TYPE>(y + 4, border_buffer);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE>
static Status Laplacian7x7HvxHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<St> &border_value, const St *border_buffer)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Laplacian7x7HvxImpl<St, Dt, BORDER_TYPE, 1>,
                                  std::cref(src), std::ref(dst), std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Laplacian7x7HvxImpl<St, Dt, BORDER_TYPE, 2>,
                                  std::cref(src), std::ref(dst), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Laplacian7x7HvxImpl<St, Dt, BORDER_TYPE, 3>,
                                  std::cref(src), std::ref(dst), std::cref(border_value), border_buffer);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename St, typename Dt>
static Status Laplacian7x7HvxHelper(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    St *border_buffer = DT_NULL;
    std::vector<St> vec_border_value = border_value.ToVector<St>();

    DT_S32 width   = src.GetSizes().m_width;
    DT_S32 channel = src.GetSizes().m_channel;

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

            ret = Laplacian7x7HvxHelper<St, Dt, BorderType::CONSTANT>(ctx, src, dst, vec_border_value, border_buffer);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Laplacian7x7HvxHelper<St, Dt, BorderType::REPLICATE>(ctx, src, dst, vec_border_value, border_buffer);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Laplacian7x7HvxHelper<St, Dt, BorderType::REFLECT_101>(ctx, src, dst, vec_border_value, border_buffer);
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

Status Laplacian7x7Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = Laplacian7x7HvxHelper<DT_U8, DT_S16>(ctx, src, dst, border_type, border_value);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = Laplacian7x7HvxHelper<DT_U16, DT_U16>(ctx, src, dst, border_type, border_value);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = Laplacian7x7HvxHelper<DT_S16, DT_S16>(ctx, src, dst, border_type, border_value);
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
