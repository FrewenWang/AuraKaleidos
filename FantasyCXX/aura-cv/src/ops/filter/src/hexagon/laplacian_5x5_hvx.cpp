#include "laplacian_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

// using St = DT_U8
template <typename St, typename std::enable_if<std::is_same<St, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian5x5Core(const HVX_Vector &vu8_src_p1_x0, const HVX_Vector &vu8_src_p1_x1, const HVX_Vector &vu8_src_p1_x2,
                                            const HVX_Vector &vu8_src_p0_x0, const HVX_Vector &vu8_src_p0_x1, const HVX_Vector &vu8_src_p0_x2,
                                            const HVX_Vector &vu8_src_c_x0,  const HVX_Vector &vu8_src_c_x1,  const HVX_Vector &vu8_src_c_x2,
                                            const HVX_Vector &vu8_src_n0_x0, const HVX_Vector &vu8_src_n0_x1, const HVX_Vector &vu8_src_n0_x2,
                                            const HVX_Vector &vu8_src_n1_x0, const HVX_Vector &vu8_src_n1_x1, const HVX_Vector &vu8_src_n1_x2,
                                            HVX_Vector &vs16_dst_lo, HVX_Vector &vs16_dst_hi)
{
    HVX_Vector vu8_src_p1_l1, vu8_src_p1_l0, vu8_src_p1_r0, vu8_src_p1_r1;
    HVX_Vector vu8_src_p0_l1, vu8_src_p0_r1;
    HVX_Vector vu8_src_c_l1,  vu8_src_c_l0,  vu8_src_c_r0,  vu8_src_c_r1;
    HVX_Vector vu8_src_n0_l1, vu8_src_n0_r1;
    HVX_Vector vu8_src_n1_l1, vu8_src_n1_l0, vu8_src_n1_r0, vu8_src_n1_r1;
    HVX_VectorPair wu16_sum_l1r1,  wu16_sum_l0r0,  wu16_sum_c;
    HVX_VectorPair wu16_sum_p1_n1, ws16_sum_p0_n0, ws16_sum_c, ws16_sum;

    // p1_n1
    vu8_src_p1_l1  = Q6_V_vlalign_VVR(vu8_src_p1_x1, vu8_src_p1_x0, 2);
    vu8_src_p1_l0  = Q6_V_vlalign_VVR(vu8_src_p1_x1, vu8_src_p1_x0, 1);
    vu8_src_p1_r0  = Q6_V_valign_VVR(vu8_src_p1_x2, vu8_src_p1_x1, 1);
    vu8_src_p1_r1  = Q6_V_valign_VVR(vu8_src_p1_x2, vu8_src_p1_x1, 2);

    wu16_sum_l1r1  = Q6_Wuh_vmpy_VubRub(vu8_src_p1_l1, 0x02020202);
    wu16_sum_l1r1  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_p1_r1, 0x02020202);
    wu16_sum_l0r0  = Q6_Wuh_vmpy_VubRub(vu8_src_p1_l0, 0x04040404);
    wu16_sum_l0r0  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l0r0, vu8_src_p1_r0, 0x04040404);
    wu16_sum_c     = Q6_Wuh_vmpy_VubRub(vu8_src_p1_x1, 0x04040404);

    vu8_src_n1_l1  = Q6_V_vlalign_VVR(vu8_src_n1_x1, vu8_src_n1_x0, 2);
    vu8_src_n1_l0  = Q6_V_vlalign_VVR(vu8_src_n1_x1, vu8_src_n1_x0, 1);
    vu8_src_n1_r0  = Q6_V_valign_VVR(vu8_src_n1_x2, vu8_src_n1_x1, 1);
    vu8_src_n1_r1  = Q6_V_valign_VVR(vu8_src_n1_x2, vu8_src_n1_x1, 2);

    wu16_sum_l1r1  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_n1_l1, 0x02020202);
    wu16_sum_l1r1  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_n1_r1, 0x02020202);
    wu16_sum_l0r0  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l0r0, vu8_src_n1_l0, 0x04040404);
    wu16_sum_l0r0  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l0r0, vu8_src_n1_r0, 0x04040404);
    wu16_sum_c     = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_c,    vu8_src_n1_x1, 0x04040404);

    wu16_sum_p1_n1 = Q6_Wh_vadd_WhWh(wu16_sum_l1r1, wu16_sum_l0r0);
    wu16_sum_p1_n1 = Q6_Wh_vadd_WhWh(wu16_sum_p1_n1, wu16_sum_c);

    // p0_n0
    vu8_src_p0_l1  = Q6_V_vlalign_VVR(vu8_src_p0_x1, vu8_src_p0_x0, 2);
    vu8_src_p0_r1  = Q6_V_valign_VVR(vu8_src_p0_x2, vu8_src_p0_x1, 2);

    wu16_sum_l1r1  = Q6_Wuh_vmpy_VubRub(vu8_src_p0_l1, 0x04040404);
    wu16_sum_l1r1  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_p0_r1, 0x04040404);
    wu16_sum_c     = Q6_Wuh_vmpy_VubRub(vu8_src_p0_x1, 0x08080808);

    vu8_src_n0_l1  = Q6_V_vlalign_VVR(vu8_src_n0_x1, vu8_src_n0_x0, 2);
    vu8_src_n0_r1  = Q6_V_valign_VVR(vu8_src_n0_x2, vu8_src_n0_x1, 2);

    wu16_sum_l1r1  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_n0_l1, 0x04040404);
    wu16_sum_l1r1  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_n0_r1, 0x04040404);
    wu16_sum_c     = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_c, vu8_src_n0_x1, 0x08080808);

    ws16_sum_p0_n0 = Q6_Wh_vsub_WhWh(wu16_sum_l1r1, wu16_sum_c);

    // c
    vu8_src_c_l1   = Q6_V_vlalign_VVR(vu8_src_c_x1, vu8_src_c_x0, 2);
    vu8_src_c_l0   = Q6_V_vlalign_VVR(vu8_src_c_x1, vu8_src_c_x0, 1);
    vu8_src_c_r0   = Q6_V_valign_VVR(vu8_src_c_x2, vu8_src_c_x1, 1);
    vu8_src_c_r1   = Q6_V_valign_VVR(vu8_src_c_x2, vu8_src_c_x1, 2);

    wu16_sum_l1r1  = Q6_Wuh_vmpy_VubRub(vu8_src_c_l1, 0x04040404);
    wu16_sum_l1r1  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l1r1, vu8_src_c_r1, 0x04040404);
    wu16_sum_l0r0  = Q6_Wuh_vmpy_VubRub(vu8_src_c_l0, 0x08080808);
    wu16_sum_l0r0  = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum_l0r0, vu8_src_c_r0, 0x08080808);
    wu16_sum_c     = Q6_Wuh_vmpy_VubRub(vu8_src_c_x1, 0x18181818);

    ws16_sum_c     = Q6_Wh_vsub_WhWh(wu16_sum_l1r1, wu16_sum_l0r0);
    ws16_sum_c     = Q6_Wh_vsub_WhWh(ws16_sum_c, wu16_sum_c);

    // sum
    ws16_sum       = Q6_Wh_vadd_WhWh(wu16_sum_p1_n1, ws16_sum_p0_n0);
    ws16_sum       = Q6_Wh_vadd_WhWh(ws16_sum, ws16_sum_c);
    ws16_sum       = Q6_W_vshuff_VVR(Q6_V_hi_W(ws16_sum), Q6_V_lo_W(ws16_sum), -2);
    vs16_dst_lo    = Q6_V_lo_W(ws16_sum);
    vs16_dst_hi    = Q6_V_hi_W(ws16_sum);
}

// using St = DT_U16
template <typename St, typename std::enable_if<std::is_same<St, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian5x5Core(const HVX_Vector &vu16_src_p1_x0, const HVX_Vector &vu16_src_p1_x1, const HVX_Vector &vu16_src_p1_x2,
                                            const HVX_Vector &vu16_src_p0_x0, const HVX_Vector &vu16_src_p0_x1, const HVX_Vector &vu16_src_p0_x2,
                                            const HVX_Vector &vu16_src_c_x0,  const HVX_Vector &vu16_src_c_x1,  const HVX_Vector &vu16_src_c_x2,
                                            const HVX_Vector &vu16_src_n0_x0, const HVX_Vector &vu16_src_n0_x1, const HVX_Vector &vu16_src_n0_x2,
                                            const HVX_Vector &vu16_src_n1_x0, const HVX_Vector &vu16_src_n1_x1, const HVX_Vector &vu16_src_n1_x2,
                                            HVX_Vector &vu16_dst)
{
    HVX_Vector     vu16_src_p1_l1, vu16_src_p1_l0, vu16_src_p1_r0, vu16_src_p1_r1;
    HVX_Vector     vu16_src_p0_l1, vu16_src_p0_r1;
    HVX_Vector     vu16_src_c_l1,  vu16_src_c_l0,  vu16_src_c_r0,  vu16_src_c_r1;
    HVX_Vector     vu16_src_n0_l1, vu16_src_n0_r1;
    HVX_Vector     vu16_src_n1_l1, vu16_src_n1_l0, vu16_src_n1_r0, vu16_src_n1_r1;
    HVX_VectorPair wu32_sum_l1r1,  wu32_sum_l0r0,  wu32_sum_c;
    HVX_VectorPair wu32_sum_p1_n1, ws32_sum_p0_n0, ws32_sum_c,     ws32_sum;

    // p1_n1
    vu16_src_p1_l1 = Q6_V_vlalign_VVR(vu16_src_p1_x1, vu16_src_p1_x0, 4);
    vu16_src_p1_l0 = Q6_V_vlalign_VVR(vu16_src_p1_x1, vu16_src_p1_x0, 2);
    vu16_src_p1_r0 = Q6_V_valign_VVR(vu16_src_p1_x2, vu16_src_p1_x1, 2);
    vu16_src_p1_r1 = Q6_V_valign_VVR(vu16_src_p1_x2, vu16_src_p1_x1, 4);

    wu32_sum_l1r1  = Q6_Wuw_vmpy_VuhRuh(vu16_src_p1_l1, 0x00020002);
    wu32_sum_l1r1  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_p1_r1, 0x00020002);  
    wu32_sum_l0r0  = Q6_Wuw_vmpy_VuhRuh(vu16_src_p1_l0, 0x00040004);
    wu32_sum_l0r0  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l0r0, vu16_src_p1_r0, 0x00040004);
    wu32_sum_c     = Q6_Wuw_vmpy_VuhRuh(vu16_src_p1_x1, 0x00040004);

    vu16_src_n1_l1 = Q6_V_vlalign_VVR(vu16_src_n1_x1, vu16_src_n1_x0, 4);
    vu16_src_n1_l0 = Q6_V_vlalign_VVR(vu16_src_n1_x1, vu16_src_n1_x0, 2);
    vu16_src_n1_r0 = Q6_V_valign_VVR(vu16_src_n1_x2,  vu16_src_n1_x1, 2);
    vu16_src_n1_r1 = Q6_V_valign_VVR(vu16_src_n1_x2,  vu16_src_n1_x1, 4);

    wu32_sum_l1r1  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_n1_l1, 0x00020002);
    wu32_sum_l1r1  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_n1_r1, 0x00020002);
    wu32_sum_l0r0  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l0r0, vu16_src_n1_l0, 0x00040004);
    wu32_sum_l0r0  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l0r0, vu16_src_n1_r0, 0x00040004);
    wu32_sum_c     = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_c,    vu16_src_n1_x1, 0x00040004);

    wu32_sum_p1_n1 = Q6_Ww_vadd_WwWw(wu32_sum_l1r1,  wu32_sum_l0r0);
    wu32_sum_p1_n1 = Q6_Ww_vadd_WwWw(wu32_sum_p1_n1, wu32_sum_c);

    // p0_n0
    vu16_src_p0_l1 = Q6_V_vlalign_VVR(vu16_src_p0_x1, vu16_src_p0_x0, 4);
    vu16_src_p0_r1 = Q6_V_valign_VVR(vu16_src_p0_x2,  vu16_src_p0_x1, 4);

    wu32_sum_l1r1  = Q6_Wuw_vmpy_VuhRuh(vu16_src_p0_l1, 0x00040004);
    wu32_sum_l1r1  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_p0_r1, 0x00040004);
    wu32_sum_c     = Q6_Wuw_vmpy_VuhRuh(vu16_src_p0_x1, 0x00080008);

    vu16_src_n0_l1 = Q6_V_vlalign_VVR(vu16_src_n0_x1, vu16_src_n0_x0, 4);
    vu16_src_n0_r1 = Q6_V_valign_VVR(vu16_src_n0_x2,  vu16_src_n0_x1, 4);

    wu32_sum_l1r1  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_n0_l1, 0x00040004);
    wu32_sum_l1r1  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_n0_r1, 0x00040004);
    wu32_sum_c     = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_c,    vu16_src_n0_x1, 0x00080008);

    ws32_sum_p0_n0 = Q6_Ww_vsub_WwWw(wu32_sum_l1r1, wu32_sum_c);

    // c
    vu16_src_c_l1  = Q6_V_vlalign_VVR(vu16_src_c_x1, vu16_src_c_x0, 4);
    vu16_src_c_l0  = Q6_V_vlalign_VVR(vu16_src_c_x1, vu16_src_c_x0, 2);
    vu16_src_c_r0  = Q6_V_valign_VVR(vu16_src_c_x2,  vu16_src_c_x1, 2);
    vu16_src_c_r1  = Q6_V_valign_VVR(vu16_src_c_x2,  vu16_src_c_x1, 4);

    wu32_sum_l1r1  = Q6_Wuw_vmpy_VuhRuh(vu16_src_c_l1, 0x00040004);
    wu32_sum_l1r1  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l1r1, vu16_src_c_r1, 0x00040004);
    wu32_sum_l0r0  = Q6_Wuw_vmpy_VuhRuh(vu16_src_c_l0, 0x00080008);
    wu32_sum_l0r0  = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_l0r0, vu16_src_c_r0, 0x00080008);
    wu32_sum_c     = Q6_Wuw_vmpy_VuhRuh(vu16_src_c_x1, 0x00180018);

    ws32_sum_c     = Q6_Ww_vsub_WwWw(wu32_sum_l1r1, wu32_sum_l0r0);
    ws32_sum_c     = Q6_Ww_vsub_WwWw(ws32_sum_c,    wu32_sum_c);

    // sum
    ws32_sum       = Q6_Ww_vadd_WwWw(wu32_sum_p1_n1, ws32_sum_p0_n0);
    ws32_sum       = Q6_Ww_vadd_WwWw(ws32_sum,       ws32_sum_c);
    ws32_sum       = Q6_W_vshuff_VVR(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum), -4);
    vu16_dst       = Q6_Vuh_vpack_VwVw_sat(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum));
}

// using St = DT_S16
template <typename St, typename std::enable_if<std::is_same<St, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian5x5Core(const HVX_Vector &vs16_src_p1_x0, const HVX_Vector &vs16_src_p1_x1, const HVX_Vector &vs16_src_p1_x2,
                                            const HVX_Vector &vs16_src_p0_x0, const HVX_Vector &vs16_src_p0_x1, const HVX_Vector &vs16_src_p0_x2,
                                            const HVX_Vector &vs16_src_c_x0,  const HVX_Vector &vs16_src_c_x1,  const HVX_Vector &vs16_src_c_x2,
                                            const HVX_Vector &vs16_src_n0_x0, const HVX_Vector &vs16_src_n0_x1, const HVX_Vector &vs16_src_n0_x2,
                                            const HVX_Vector &vs16_src_n1_x0, const HVX_Vector &vs16_src_n1_x1, const HVX_Vector &vs16_src_n1_x2,
                                            HVX_Vector &vs16_dst)
{
    HVX_Vector vs16_src_p1_l1, vs16_src_p1_l0, vs16_src_p1_r0, vs16_src_p1_r1;
    HVX_Vector vs16_src_p0_l1, vs16_src_p0_r1;
    HVX_Vector vs16_src_c_l1,  vs16_src_c_l0,  vs16_src_c_r0,  vs16_src_c_r1;
    HVX_Vector vs16_src_n0_l1, vs16_src_n0_r1;
    HVX_Vector vs16_src_n1_l1, vs16_src_n1_l0, vs16_src_n1_r0, vs16_src_n1_r1;

    HVX_VectorPair ws32_sum_l1r1,  ws32_sum_l0r0,  ws32_sum_c;
    HVX_VectorPair ws32_sum_p1_n1, ws32_sum_p0_n0, ws32_sum_c_c, ws32_sum;

    // p1_n1
    vs16_src_p1_l1 = Q6_V_vlalign_VVR(vs16_src_p1_x1, vs16_src_p1_x0, 4);
    vs16_src_p1_l0 = Q6_V_vlalign_VVR(vs16_src_p1_x1, vs16_src_p1_x0, 2);
    vs16_src_p1_r0 = Q6_V_valign_VVR(vs16_src_p1_x2,  vs16_src_p1_x1, 2);
    vs16_src_p1_r1 = Q6_V_valign_VVR(vs16_src_p1_x2,  vs16_src_p1_x1, 4);

    ws32_sum_l1r1  = Q6_Ww_vmpy_VhRh(vs16_src_p1_l1, 0x00020002);
    ws32_sum_l1r1  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_p1_r1, 0x00020002);  
    ws32_sum_l0r0  = Q6_Ww_vmpy_VhRh(vs16_src_p1_l0, 0x00040004);
    ws32_sum_l0r0  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l0r0, vs16_src_p1_r0, 0x00040004);
    ws32_sum_c     = Q6_Ww_vmpy_VhRh(vs16_src_p1_x1, 0x00040004);

    vs16_src_n1_l1 = Q6_V_vlalign_VVR(vs16_src_n1_x1, vs16_src_n1_x0, 4);
    vs16_src_n1_l0 = Q6_V_vlalign_VVR(vs16_src_n1_x1, vs16_src_n1_x0, 2);
    vs16_src_n1_r0 = Q6_V_valign_VVR(vs16_src_n1_x2,  vs16_src_n1_x1, 2);
    vs16_src_n1_r1 = Q6_V_valign_VVR(vs16_src_n1_x2,  vs16_src_n1_x1, 4);

    ws32_sum_l1r1  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_n1_l1, 0x00020002);
    ws32_sum_l1r1  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_n1_r1, 0x00020002);
    ws32_sum_l0r0  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l0r0, vs16_src_n1_l0, 0x00040004);
    ws32_sum_l0r0  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l0r0, vs16_src_n1_r0, 0x00040004);
    ws32_sum_c     = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_c,    vs16_src_n1_x1, 0x00040004);

    ws32_sum_p1_n1 = Q6_Ww_vadd_WwWw(ws32_sum_l1r1,  ws32_sum_l0r0);
    ws32_sum_p1_n1 = Q6_Ww_vadd_WwWw(ws32_sum_p1_n1, ws32_sum_c);

    // p0_n0
    vs16_src_p0_l1 = Q6_V_vlalign_VVR(vs16_src_p0_x1, vs16_src_p0_x0, 4);
    vs16_src_p0_r1 = Q6_V_valign_VVR(vs16_src_p0_x2,  vs16_src_p0_x1, 4);

    ws32_sum_l1r1  = Q6_Ww_vmpy_VhRh(vs16_src_p0_l1, 0x00040004);
    ws32_sum_l1r1  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_p0_r1, 0x00040004);
    ws32_sum_c     = Q6_Ww_vmpy_VhRh(vs16_src_p0_x1, 0x00080008);

    vs16_src_n0_l1 = Q6_V_vlalign_VVR(vs16_src_n0_x1, vs16_src_n0_x0, 4);
    vs16_src_n0_r1 = Q6_V_valign_VVR(vs16_src_n0_x2,  vs16_src_n0_x1, 4);

    ws32_sum_l1r1  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_n0_l1, 0x00040004);
    ws32_sum_l1r1  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_n0_r1, 0x00040004);
    ws32_sum_c     = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_c,    vs16_src_n0_x1, 0x00080008);

    ws32_sum_p0_n0 = Q6_Ww_vsub_WwWw(ws32_sum_l1r1, ws32_sum_c);

    // c
    vs16_src_c_l1  = Q6_V_vlalign_VVR(vs16_src_c_x1, vs16_src_c_x0, 4);
    vs16_src_c_l0  = Q6_V_vlalign_VVR(vs16_src_c_x1, vs16_src_c_x0, 2);
    vs16_src_c_r0  = Q6_V_valign_VVR(vs16_src_c_x2,  vs16_src_c_x1, 2);
    vs16_src_c_r1  = Q6_V_valign_VVR(vs16_src_c_x2,  vs16_src_c_x1, 4);

    ws32_sum_l1r1  = Q6_Ww_vmpy_VhRh(vs16_src_c_l1, 0x00040004);
    ws32_sum_l1r1  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l1r1, vs16_src_c_r1, 0x00040004);
    ws32_sum_l0r0  = Q6_Ww_vmpy_VhRh(vs16_src_c_l0, 0x00080008);
    ws32_sum_l0r0  = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_l0r0, vs16_src_c_r0, 0x00080008);
    ws32_sum_c     = Q6_Ww_vmpy_VhRh(vs16_src_c_x1, 0x00180018);

    ws32_sum_c_c   = Q6_Ww_vsub_WwWw(ws32_sum_l1r1, ws32_sum_l0r0);
    ws32_sum_c_c   = Q6_Ww_vsub_WwWw(ws32_sum_c_c,  ws32_sum_c);

    // sum
    ws32_sum       = Q6_Ww_vadd_WwWw(ws32_sum_p1_n1, ws32_sum_p0_n0);
    ws32_sum       = Q6_Ww_vadd_WwWw(ws32_sum, ws32_sum_c_c);
    ws32_sum       = Q6_W_vshuff_VVR(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum), -4);
    vs16_dst       = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum));
}

// using St = DT_U8
template <typename St, typename std::enable_if<std::is_same<St, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian5x5Core(const HVX_Vector &vu8_src_p1_x0, const HVX_Vector &vu8_src_p1_x1, const HVX_Vector &vu8_src_p1_x2, const HVX_Vector &vu8_border_p1,
                                            const HVX_Vector &vu8_src_p0_x0, const HVX_Vector &vu8_src_p0_x1, const HVX_Vector &vu8_src_p0_x2, const HVX_Vector &vu8_border_p0,
                                            const HVX_Vector &vu8_src_c_x0,  const HVX_Vector &vu8_src_c_x1,  const HVX_Vector &vu8_src_c_x2,  const HVX_Vector &vu8_border_c,
                                            const HVX_Vector &vu8_src_n0_x0, const HVX_Vector &vu8_src_n0_x1, const HVX_Vector &vu8_src_n0_x2, const HVX_Vector &vu8_border_n0,
                                            const HVX_Vector &vu8_src_n1_x0, const HVX_Vector &vu8_src_n1_x1, const HVX_Vector &vu8_src_n1_x2, const HVX_Vector &vu8_border_n1,
                                            HVX_Vector &vs16_dst_x0_l, HVX_Vector &vs16_dst_x0_h, HVX_Vector &vs16_dst_x1_lo, HVX_Vector &vs16_dst_x1_hi,
                                            DT_S32 align_size)
{
    HVX_Vector vu8_src_p1_r = Q6_V_vlalign_VVR(vu8_border_p1, vu8_src_p1_x2, align_size);
    HVX_Vector vu8_src_p0_r = Q6_V_vlalign_VVR(vu8_border_p0, vu8_src_p0_x2, align_size);
    HVX_Vector vu8_src_c_r  = Q6_V_vlalign_VVR(vu8_border_c,  vu8_src_c_x2,  align_size);
    HVX_Vector vu8_src_n0_r = Q6_V_vlalign_VVR(vu8_border_n0, vu8_src_n0_x2, align_size);
    HVX_Vector vu8_src_n1_r = Q6_V_vlalign_VVR(vu8_border_n1, vu8_src_n1_x2, align_size);
    Laplacian5x5Core<St>(vu8_src_p1_x0, vu8_src_p1_x1, vu8_src_p1_r,
                         vu8_src_p0_x0, vu8_src_p0_x1, vu8_src_p0_r,
                         vu8_src_c_x0,  vu8_src_c_x1,  vu8_src_c_r,
                         vu8_src_n0_x0, vu8_src_n0_x1, vu8_src_n0_r,
                         vu8_src_n1_x0, vu8_src_n1_x1, vu8_src_n1_r,
                         vs16_dst_x0_l, vs16_dst_x0_h);

    HVX_Vector vu8_src_p1_l = Q6_V_valign_VVR(vu8_src_p1_x1, vu8_src_p1_x0, align_size);
    HVX_Vector vu8_src_p0_l = Q6_V_valign_VVR(vu8_src_p0_x1, vu8_src_p0_x0, align_size);
    HVX_Vector vu8_src_c_l  = Q6_V_valign_VVR(vu8_src_c_x1,  vu8_src_c_x0,  align_size);
    HVX_Vector vu8_src_n0_l = Q6_V_valign_VVR(vu8_src_n0_x1, vu8_src_n0_x0, align_size);
    HVX_Vector vu8_src_n1_l = Q6_V_valign_VVR(vu8_src_n1_x1, vu8_src_n1_x0, align_size);
    Laplacian5x5Core<St>(vu8_src_p1_l,   vu8_src_p1_x2, vu8_border_p1,
                         vu8_src_p0_l,   vu8_src_p0_x2, vu8_border_p0,
                         vu8_src_c_l,    vu8_src_c_x2,  vu8_border_c,
                         vu8_src_n0_l,   vu8_src_n0_x2, vu8_border_n0,
                         vu8_src_n1_l,   vu8_src_n1_x2, vu8_border_n1,
                         vs16_dst_x1_lo, vs16_dst_x1_hi);
}

// using St = DT_U16 / DT_S16
template <typename St, typename std::enable_if<std::is_same<St, DT_U16>::value || std::is_same<St, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Laplacian5x5Core(const HVX_Vector &vd16_p1_x0, const HVX_Vector &vd16_p1_x1, const HVX_Vector &vd16_p1_x2, const HVX_Vector &vd16_border_p1,
                                            const HVX_Vector &vd16_p0_x0, const HVX_Vector &vd16_p0_x1, const HVX_Vector &vd16_p0_x2, const HVX_Vector &vd16_border_p0,
                                            const HVX_Vector &vd16_c_x0,  const HVX_Vector &vd16_c_x1,  const HVX_Vector &vd16_c_x2,  const HVX_Vector &vd16_border_c,
                                            const HVX_Vector &vd16_n0_x0, const HVX_Vector &vd16_n0_x1, const HVX_Vector &vd16_n0_x2, const HVX_Vector &vd16_border_n0,
                                            const HVX_Vector &vd16_n1_x0, const HVX_Vector &vd16_n1_x1, const HVX_Vector &vd16_n1_x2, const HVX_Vector &vd16_border_n1,
                                            HVX_Vector &vd16_dst_x0, HVX_Vector &vd16_dst_x1,
                                            DT_S32 align_size)
{
    HVX_Vector vd16_src_p1_r = Q6_V_vlalign_VVR(vd16_border_p1, vd16_p1_x2, align_size);
    HVX_Vector vd16_src_p0_r = Q6_V_vlalign_VVR(vd16_border_p0, vd16_p0_x2, align_size);
    HVX_Vector vd16_src_c_r  = Q6_V_vlalign_VVR(vd16_border_c,  vd16_c_x2,  align_size);
    HVX_Vector vd16_src_n0_r = Q6_V_vlalign_VVR(vd16_border_n0, vd16_n0_x2, align_size);
    HVX_Vector vd16_src_n1_r = Q6_V_vlalign_VVR(vd16_border_n1, vd16_n1_x2, align_size);
    Laplacian5x5Core<St>(vd16_p1_x0, vd16_p1_x1, vd16_src_p1_r,
                         vd16_p0_x0, vd16_p0_x1, vd16_src_p0_r,
                         vd16_c_x0,  vd16_c_x1,  vd16_src_c_r,
                         vd16_n0_x0, vd16_n0_x1, vd16_src_n0_r,
                         vd16_n1_x0, vd16_n1_x1, vd16_src_n1_r,
                         vd16_dst_x0);

    HVX_Vector vd16_src_p1_l = Q6_V_valign_VVR(vd16_p1_x1, vd16_p1_x0, align_size);
    HVX_Vector vd16_src_p0_l = Q6_V_valign_VVR(vd16_p0_x1, vd16_p0_x0, align_size);
    HVX_Vector vd16_src_c_l  = Q6_V_valign_VVR(vd16_c_x1,  vd16_c_x0,  align_size);
    HVX_Vector vd16_src_n0_l = Q6_V_valign_VVR(vd16_n0_x1, vd16_n0_x0, align_size);
    HVX_Vector vd16_src_n1_l = Q6_V_valign_VVR(vd16_n1_x1, vd16_n1_x0, align_size);
    Laplacian5x5Core<St>(vd16_src_p1_l, vd16_p1_x2, vd16_border_p1,
                         vd16_src_p0_l, vd16_p0_x2, vd16_border_p0,
                         vd16_src_c_l,  vd16_c_x2,  vd16_border_c,
                         vd16_src_n0_l, vd16_n0_x2, vd16_border_n0,
                         vd16_src_n1_l, vd16_n1_x2, vd16_border_n1,
                         vd16_dst_x1);
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C,
          typename std::enable_if<std::is_same<St, DT_U8>::value>::type* = DT_NULL>
static DT_VOID Laplacian5x5Row(const St *src_p1, const St *src_p0, const St *src_c, const St *src_n0, const St *src_n1,
                               Dt *dst_c, const std::vector<St> &border_value, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    constexpr DT_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(St);
    DT_S32 back_offset = width - ELEM_COUNTS;

    MVType mvu8_src_p1[3], mvu8_src_p0[3], mvu8_src_c[3], mvu8_src_n0[3], mvu8_src_n1[3];
    MVType mvs16_dst_lo, mvs16_dst_hi;

    // left border
    {
        vload(src_p1, mvu8_src_p1[1]);
        vload(src_p0, mvu8_src_p0[1]);
        vload(src_c,  mvu8_src_c[1]);
        vload(src_n0, mvu8_src_n0[1]);
        vload(src_n1, mvu8_src_n1[1]);

        #pragma unroll
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvu8_src_p1[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mvu8_src_p0[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mvu8_src_c[0].val[ch]  = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_c[1].val[ch],  src_c[ch],  border_value[ch]);
            mvu8_src_n0[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mvu8_src_n1[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_n1[1].val[ch], src_n1[ch], border_value[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (DT_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS)
        {
            vload(src_p1 + C * x, mvu8_src_p1[2]);
            vload(src_p0 + C * x, mvu8_src_p0[2]);
            vload(src_c  + C * x, mvu8_src_c[2]);
            vload(src_n0 + C * x, mvu8_src_n0[2]);
            vload(src_n1 + C * x, mvu8_src_n1[2]);

            #pragma unroll
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Laplacian5x5Core<St>(mvu8_src_p1[0].val[ch], mvu8_src_p1[1].val[ch], mvu8_src_p1[2].val[ch],
                                     mvu8_src_p0[0].val[ch], mvu8_src_p0[1].val[ch], mvu8_src_p0[2].val[ch],
                                     mvu8_src_c[0].val[ch],  mvu8_src_c[1].val[ch],  mvu8_src_c[2].val[ch],
                                     mvu8_src_n0[0].val[ch], mvu8_src_n0[1].val[ch], mvu8_src_n0[2].val[ch],
                                     mvu8_src_n1[0].val[ch], mvu8_src_n1[1].val[ch], mvu8_src_n1[2].val[ch],
                                     mvs16_dst_lo.val[ch], mvs16_dst_hi.val[ch]);
            }

            vstore(dst_c + C * (x - ELEM_COUNTS),        mvs16_dst_lo);
            vstore(dst_c + C * (x - (ELEM_COUNTS >> 1)), mvs16_dst_hi);

            mvu8_src_p1[0] = mvu8_src_p1[1];
            mvu8_src_p0[0] = mvu8_src_p0[1];
            mvu8_src_c[0]  = mvu8_src_c[1];
            mvu8_src_n0[0] = mvu8_src_n0[1];
            mvu8_src_n1[0] = mvu8_src_n1[1];

            mvu8_src_p1[1] = mvu8_src_p1[2];
            mvu8_src_p0[1] = mvu8_src_p0[2];
            mvu8_src_c[1]  = mvu8_src_c[2];
            mvu8_src_n0[1] = mvu8_src_n0[2];
            mvu8_src_n1[1] = mvu8_src_n1[2];
        }
    }

    // remain
    {
        DT_S32 last = (width - 1) * C;
        DT_S32 rest = width % ELEM_COUNTS;
        MVType mvs16_last_lo, mvs16_last_hi;

        vload(src_p1 + C * back_offset, mvu8_src_p1[2]);
        vload(src_p0 + C * back_offset, mvu8_src_p0[2]);
        vload(src_c  + C * back_offset, mvu8_src_c[2]);
        vload(src_n0 + C * back_offset, mvu8_src_n0[2]);
        vload(src_n1 + C * back_offset, mvu8_src_n1[2]);

        #pragma unroll
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector vu8_border_p1 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_p1[2].val[ch], src_p1[last + ch], border_value[ch]);
            HVX_Vector vu8_border_p0 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_p0[2].val[ch], src_p0[last + ch], border_value[ch]);
            HVX_Vector vu8_border_c  = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_c[2].val[ch],  src_c[last + ch],  border_value[ch]);
            HVX_Vector vu8_border_n0 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_n0[2].val[ch], src_n0[last + ch], border_value[ch]);
            HVX_Vector vu8_border_n1 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_n1[2].val[ch], src_n1[last + ch], border_value[ch]);

            Laplacian5x5Core<St>(mvu8_src_p1[0].val[ch], mvu8_src_p1[1].val[ch], mvu8_src_p1[2].val[ch], vu8_border_p1,
                                 mvu8_src_p0[0].val[ch], mvu8_src_p0[1].val[ch], mvu8_src_p0[2].val[ch], vu8_border_p0,
                                 mvu8_src_c[0].val[ch],  mvu8_src_c[1].val[ch],  mvu8_src_c[2].val[ch],  vu8_border_c,
                                 mvu8_src_n0[0].val[ch], mvu8_src_n0[1].val[ch], mvu8_src_n0[2].val[ch], vu8_border_n0,
                                 mvu8_src_n1[0].val[ch], mvu8_src_n1[1].val[ch], mvu8_src_n1[2].val[ch], vu8_border_n1,
                                 mvs16_dst_lo.val[ch],   mvs16_dst_hi.val[ch],   mvs16_last_lo.val[ch],  mvs16_last_hi.val[ch],
                                 rest * sizeof(St));
        }

        vstore(dst_c + C * (back_offset - rest),                      mvs16_dst_lo);
        vstore(dst_c + C * (back_offset - rest + (ELEM_COUNTS >> 1)), mvs16_dst_hi);
        vstore(dst_c + C * back_offset,                               mvs16_last_lo);
        vstore(dst_c + C * (back_offset + (ELEM_COUNTS >> 1)),        mvs16_last_hi);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C,
          typename std::enable_if<std::is_same<St, DT_U16>::value || std::is_same<St, DT_S16>::value>::type* = DT_NULL>
static DT_VOID Laplacian5x5Row(const St *src_p1, const St *src_p0, const St *src_c, const St *src_n0, const St *src_n1,
                               Dt *dst_c, const std::vector<St> &border_value, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    constexpr DT_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(St);
    DT_S32 back_offset = width - ELEM_COUNTS;

    MVType mvd16_src_p1[3], mvd16_src_p0[3], mvd16_src_c[3], mvd16_src_n0[3], mvd16_src_n1[3];
    MVType mvd16_dst;

    // left border
    {
        vload(src_p1, mvd16_src_p1[1]);
        vload(src_p0, mvd16_src_p0[1]);
        vload(src_c,  mvd16_src_c[1]);
        vload(src_n0, mvd16_src_n0[1]);
        vload(src_n1, mvd16_src_n1[1]);

        #pragma unroll
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvd16_src_p1[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mvd16_src_p0[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mvd16_src_c[0].val[ch]  = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_c[1].val[ch],  src_c[ch],  border_value[ch]);
            mvd16_src_n0[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mvd16_src_n1[0].val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_n1[1].val[ch], src_n1[ch], border_value[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (DT_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS)
        {
            vload(src_p1 + C * x, mvd16_src_p1[2]);
            vload(src_p0 + C * x, mvd16_src_p0[2]);
            vload(src_c  + C * x, mvd16_src_c[2]);
            vload(src_n0 + C * x, mvd16_src_n0[2]);
            vload(src_n1 + C * x, mvd16_src_n1[2]);

            #pragma unroll
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Laplacian5x5Core<St>(mvd16_src_p1[0].val[ch], mvd16_src_p1[1].val[ch], mvd16_src_p1[2].val[ch],
                                     mvd16_src_p0[0].val[ch], mvd16_src_p0[1].val[ch], mvd16_src_p0[2].val[ch],
                                     mvd16_src_c[0].val[ch],  mvd16_src_c[1].val[ch],  mvd16_src_c[2].val[ch],
                                     mvd16_src_n0[0].val[ch], mvd16_src_n0[1].val[ch], mvd16_src_n0[2].val[ch],
                                     mvd16_src_n1[0].val[ch], mvd16_src_n1[1].val[ch], mvd16_src_n1[2].val[ch],
                                     mvd16_dst.val[ch]);
            }

            vstore(dst_c + C * (x - ELEM_COUNTS), mvd16_dst);

            mvd16_src_p1[0] = mvd16_src_p1[1];
            mvd16_src_p0[0] = mvd16_src_p0[1];
            mvd16_src_c[0]  = mvd16_src_c[1];
            mvd16_src_n0[0] = mvd16_src_n0[1];
            mvd16_src_n1[0] = mvd16_src_n1[1];

            mvd16_src_p1[1] = mvd16_src_p1[2];
            mvd16_src_p0[1] = mvd16_src_p0[2];
            mvd16_src_c[1]  = mvd16_src_c[2];
            mvd16_src_n0[1] = mvd16_src_n0[2];
            mvd16_src_n1[1] = mvd16_src_n1[2];
        }
    }

    // remain
    {
        DT_S32 last = (width - 1) * C;
        DT_S32 rest = width % ELEM_COUNTS;
        MVType mvd16_last;

        vload(src_p1 + C * back_offset, mvd16_src_p1[2]);
        vload(src_p0 + C * back_offset, mvd16_src_p0[2]);
        vload(src_c  + C * back_offset, mvd16_src_c[2]);
        vload(src_n0 + C * back_offset, mvd16_src_n0[2]);
        vload(src_n1 + C * back_offset, mvd16_src_n1[2]);

        #pragma unroll
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector vd16_border_p1 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_p1[2].val[ch], src_p1[last + ch], border_value[ch]);
            HVX_Vector vd16_border_p0 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_p0[2].val[ch], src_p0[last + ch], border_value[ch]);
            HVX_Vector vd16_border_c  = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_c[2].val[ch],  src_c[last + ch],  border_value[ch]);
            HVX_Vector vd16_border_n0 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_n0[2].val[ch], src_n0[last + ch], border_value[ch]);
            HVX_Vector vd16_border_n1 = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_n1[2].val[ch], src_n1[last + ch], border_value[ch]);

            Laplacian5x5Core<St>(mvd16_src_p1[0].val[ch], mvd16_src_p1[1].val[ch], mvd16_src_p1[2].val[ch], vd16_border_p1,
                                 mvd16_src_p0[0].val[ch], mvd16_src_p0[1].val[ch], mvd16_src_p0[2].val[ch], vd16_border_p0,
                                 mvd16_src_c[0].val[ch],  mvd16_src_c[1].val[ch],  mvd16_src_c[2].val[ch],  vd16_border_c,
                                 mvd16_src_n0[0].val[ch], mvd16_src_n0[1].val[ch], mvd16_src_n0[2].val[ch], vd16_border_n0,
                                 mvd16_src_n1[0].val[ch], mvd16_src_n1[1].val[ch], mvd16_src_n1[2].val[ch], vd16_border_n1,
                                 mvd16_dst.val[ch],       mvd16_last.val[ch],
                                 rest * sizeof(St));
        }

        vstore(dst_c + C * (back_offset - rest), mvd16_dst);
        vstore(dst_c + C * back_offset,          mvd16_last);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C>
static Status Laplacian5x5HvxImpl(const Mat &src, Mat &dst, const std::vector<St> &border_value,
                                  const St *border_buffer, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width   = src.GetSizes().m_width;
    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    const St *src_p1 = src.Ptr<St, BORDER_TYPE>(start_row - 2, border_buffer);
    const St *src_p0 = src.Ptr<St, BORDER_TYPE>(start_row - 1, border_buffer);
    const St *src_c  = src.Ptr<St>(start_row);
    const St *src_n0 = src.Ptr<St, BORDER_TYPE>(start_row + 1, border_buffer);
    const St *src_n1 = src.Ptr<St, BORDER_TYPE>(start_row + 2, border_buffer);

    DT_U64 L2fetch_row_param = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if (y + 3 < height)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<St>(y + 3)), L2fetch_row_param);
        }

        Dt *dst_c = dst.Ptr<Dt>(y);
        Laplacian5x5Row<St, Dt, BORDER_TYPE, C>(src_p1, src_p0, src_c, src_n0, src_n1, dst_c, border_value, width);

        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src.Ptr<St, BORDER_TYPE>(y + 3, border_buffer);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE>
static Status Laplacian5x5HvxHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<St> &border_value, const St *border_buffer)
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
            ret = wp->ParallelFor((DT_S32)0, height, Laplacian5x5HvxImpl<St, Dt, BORDER_TYPE, 1>,
                                  std::cref(src), std::ref(dst), std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Laplacian5x5HvxImpl<St, Dt, BORDER_TYPE, 2>,
                                  std::cref(src), std::ref(dst), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Laplacian5x5HvxImpl<St, Dt, BORDER_TYPE, 3>,
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
static Status Laplacian5x5HvxHelper(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value)
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

            ret = Laplacian5x5HvxHelper<St, Dt, BorderType::CONSTANT>(ctx, src, dst, vec_border_value, border_buffer);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Laplacian5x5HvxHelper<St, Dt, BorderType::REPLICATE>(ctx, src, dst, vec_border_value, border_buffer);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Laplacian5x5HvxHelper<St, Dt, BorderType::REFLECT_101>(ctx, src, dst, vec_border_value, border_buffer);
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

Status Laplacian5x5Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = Laplacian5x5HvxHelper<DT_U8, DT_S16>(ctx, src, dst, border_type, border_value);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = Laplacian5x5HvxHelper<DT_U16, DT_U16>(ctx, src, dst, border_type, border_value);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = Laplacian5x5HvxHelper<DT_S16, DT_S16>(ctx, src, dst, border_type, border_value);
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
