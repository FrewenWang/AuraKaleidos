#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_LOAD_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_LOAD_HPP__

#include "aura/runtime/core/hexagon/device/core.hpp"
#include "aura/runtime/core/hexagon/device/traits.hpp"

typedef long HEXAGON_Vect_UN __attribute__((__vector_size__(AURA_HVLEN)))__attribute__((aligned(4)));
#define vmemu(addr) *((HEXAGON_Vect_UN*)(addr))

namespace aura
{

extern const DT_U8 vdelta_load_d8[]    __attribute__((aligned(128)));
extern const DT_U8 vrdelta_store_d8[]  __attribute__((aligned(128)));
extern const DT_U8 vdelta_load_d16[]   __attribute__((aligned(128)));
extern const DT_U8 vrdelta_store_d16[] __attribute__((aligned(128)));
extern const DT_U8 vdelta_load_d32[]   __attribute__((aligned(128)));
extern const DT_U8 vrdelta_store_d32[] __attribute__((aligned(128)));

// =============================================== HVX_Vector ===============================================
AURA_INLINE DT_VOID vload(const DT_VOID *addr, HVX_Vector &v_out)
{
    v_out = vmemu(addr);
}

AURA_INLINE DT_VOID vstore(DT_VOID *addr, const HVX_Vector &v_in)
{
    vmemu(addr) = v_in;
}

// =============================================== HVX_VectorPair ===============================================
AURA_INLINE DT_VOID vload(const DT_VOID *addr, HVX_VectorPair &mv_out)
{
    HVX_Vector v_x0 = vmemu(addr);
    HVX_Vector v_x1 = vmemu((DT_U8*)addr + AURA_HVLEN);

    mv_out = Q6_W_vcombine_VV(v_x1, v_x0);
}

AURA_INLINE DT_VOID vstore(DT_VOID *addr, const HVX_VectorPair &mv_in)
{
    vmemu(addr)                      = Q6_V_lo_W(mv_in);
    vmemu((DT_U8*)addr + AURA_HVLEN) = Q6_V_hi_W(mv_in);
}

// =============================================== HVX_VectorX1 ===============================================
AURA_INLINE DT_VOID vload(const DT_VOID *addr, HVX_VectorX1 &v_out)
{
    vload(addr, v_out.val[0]);
}

AURA_INLINE DT_VOID vstore(DT_VOID *addr, const HVX_VectorX1 &v_in)
{
    vstore(addr, v_in.val[0]);
}

// =============================================== HVX_VectorX2 ===============================================
template <typename Tp>
AURA_INLINE DT_VOID vload(const Tp *addr, HVX_VectorX2 &v_out)
{
    constexpr DT_S32 BYTES = static_cast<DT_S32>(sizeof(Tp));

    HVX_Vector v_x0 = vmemu(addr);
    HVX_Vector v_x1 = vmemu(addr + AURA_HVLEN / BYTES);

    HVX_VectorPair w_u_v = Q6_W_vdeal_VVR(v_x1, v_x0, -BYTES);
    v_out.val[0] = Q6_V_lo_W(w_u_v);
    v_out.val[1] = Q6_V_hi_W(w_u_v);
}

template <typename Tp>
AURA_INLINE DT_VOID vstore(Tp *addr, const HVX_VectorX2 &mv_in)
{
    constexpr DT_S32 BYTES = static_cast<DT_S32>(sizeof(Tp));
    HVX_VectorPair w_uv = Q6_W_vshuff_VVR(mv_in.val[1], mv_in.val[0], -BYTES);

    vmemu(addr)                      = Q6_V_lo_W(w_uv);
    vmemu(addr + AURA_HVLEN / BYTES) = Q6_V_hi_W(w_uv);
}

// =============================================== HVX_VectorX4 ===============================================
template <typename Tp>
AURA_INLINE DT_VOID vload(const Tp *addr, HVX_VectorX4 &mv_out)
{
    constexpr DT_S32 BYTES   = static_cast<DT_S32>(sizeof(Tp));
    constexpr DT_S32 BYTESX2 = BYTES * 2;

    HVX_Vector v_x0 = vmemu(addr);
    HVX_Vector v_x1 = vmemu(addr + AURA_HVLEN / BYTES);
    HVX_Vector v_x2 = vmemu(addr + AURA_HVLEN / BYTES * 2);
    HVX_Vector v_x3 = vmemu(addr + AURA_HVLEN / BYTES * 3);

    HVX_VectorPair w_uv0 = Q6_W_vdeal_VVR(v_x1, v_x0, -BYTESX2);
    HVX_VectorPair w_uv1 = Q6_W_vdeal_VVR(v_x3, v_x2, -BYTESX2);
    HVX_VectorPair w_uv2 = Q6_W_vdeal_VVR(Q6_V_lo_W(w_uv1), Q6_V_lo_W(w_uv0), -BYTES);
    HVX_VectorPair w_uv3 = Q6_W_vdeal_VVR(Q6_V_hi_W(w_uv1), Q6_V_hi_W(w_uv0), -BYTES);

    mv_out.val[0] = Q6_V_lo_W(w_uv2);
    mv_out.val[1] = Q6_V_hi_W(w_uv2);
    mv_out.val[2] = Q6_V_lo_W(w_uv3);
    mv_out.val[3] = Q6_V_hi_W(w_uv3);
}

template <typename Tp>
AURA_INLINE DT_VOID vstore(Tp *addr, const HVX_VectorX4 &v_in)
{
    constexpr DT_S32 BYTES   = static_cast<DT_S32>(sizeof(Tp));
    constexpr DT_S32 BYTESX2 = BYTES * 2;
    HVX_VectorPair w_uv0 = Q6_W_vshuff_VVR(v_in.val[1], v_in.val[0], -BYTES);
    HVX_VectorPair w_uv1 = Q6_W_vshuff_VVR(v_in.val[3], v_in.val[2], -BYTES);
    HVX_VectorPair w_uv2 = Q6_W_vshuff_VVR(Q6_V_lo_W(w_uv1), Q6_V_lo_W(w_uv0), -BYTESX2);
    HVX_VectorPair w_uv3 = Q6_W_vshuff_VVR(Q6_V_hi_W(w_uv1), Q6_V_hi_W(w_uv0), -BYTESX2);

    vmemu(addr)                          = Q6_V_lo_W(w_uv2);
    vmemu(addr + AURA_HVLEN / BYTES)     = Q6_V_hi_W(w_uv2);
    vmemu(addr + AURA_HVLEN / BYTES * 2) = Q6_V_lo_W(w_uv3);
    vmemu(addr + AURA_HVLEN / BYTES * 3) = Q6_V_hi_W(w_uv3);
}

// =============================================== HVX_VectorX3 ===============================================
// sizeof(Tp) == 1
template <typename Tp, typename std::enable_if<sizeof(Tp) == 1>::type* = DT_NULL>
AURA_INLINE DT_VOID vload(const Tp *addr, HVX_VectorX3 &mvd8_out)
{
    HVX_Vector vd8_x0 = vmemu(addr);
    HVX_Vector vd8_x1 = vmemu(addr + AURA_HVLEN);
    HVX_Vector vd8_x2 = vmemu(addr + AURA_HVLEN * 2);

    HVX_Vector v_vdelta = vmemu(vdelta_load_d8);
    HVX_Vector vd8_x0_rgbx = Q6_V_vdelta_VV(vd8_x0, v_vdelta);
    HVX_Vector vd8_x1_rgbx = Q6_V_vdelta_VV(Q6_V_vlalign_VVR(vd8_x1, vd8_x0, 32), v_vdelta);
    HVX_Vector vd8_x2_rgbx = Q6_V_vdelta_VV(Q6_V_vlalign_VVR(vd8_x2, vd8_x1, 64), v_vdelta);
    HVX_Vector vd8_x3_rgbx = Q6_V_vdelta_VV(Q6_V_vror_VR(vd8_x2, 32), v_vdelta);

    HVX_VectorPair wd8_x0_rg_bx = Q6_W_vdeal_VVR(vd8_x1_rgbx, vd8_x0_rgbx, -2);
    HVX_VectorPair wd8_x1_rg_bx = Q6_W_vdeal_VVR(vd8_x3_rgbx, vd8_x2_rgbx, -2);
    HVX_VectorPair wd8_r_g      = Q6_W_vdeal_VVR(Q6_V_lo_W(wd8_x1_rg_bx), Q6_V_lo_W(wd8_x0_rg_bx), -1);
    HVX_VectorPair wd8_b_x      = Q6_W_vdeal_VVR(Q6_V_hi_W(wd8_x1_rg_bx), Q6_V_hi_W(wd8_x0_rg_bx), -1);

    mvd8_out.val[0] = Q6_V_lo_W(wd8_r_g);
    mvd8_out.val[1] = Q6_V_hi_W(wd8_r_g);
    mvd8_out.val[2] = Q6_V_lo_W(wd8_b_x);
}

template <typename Tp, typename std::enable_if<sizeof(Tp) == 1>::type* = DT_NULL>
AURA_INLINE DT_VOID vstore(Tp *addr, const HVX_VectorX3 &mvd8_in)
{
    HVX_VectorPair wd8_rg_rg   = Q6_W_vshuff_VVR(mvd8_in.val[1], mvd8_in.val[0], -1);
    HVX_VectorPair wd8_b0_b0   = Q6_W_vshuff_VVR(Q6_V_vzero(), mvd8_in.val[2], -1);
    HVX_VectorPair wd8_x0_rgb0 = Q6_W_vshuff_VVR(Q6_V_lo_W(wd8_b0_b0), Q6_V_lo_W(wd8_rg_rg), -2);
    HVX_VectorPair wd8_x1_rgb0 = Q6_W_vshuff_VVR(Q6_V_hi_W(wd8_b0_b0), Q6_V_hi_W(wd8_rg_rg), -2);

    HVX_Vector v_vrdelta = vmemu(vrdelta_store_d8);
    HVX_Vector vd8_x0_rgb = Q6_V_vrdelta_VV(Q6_V_lo_W(wd8_x0_rgb0), v_vrdelta);
    HVX_Vector vd8_x1_rgb = Q6_V_vrdelta_VV(Q6_V_hi_W(wd8_x0_rgb0), v_vrdelta);
    HVX_Vector vd8_x2_rgb = Q6_V_vrdelta_VV(Q6_V_lo_W(wd8_x1_rgb0), v_vrdelta);
    HVX_Vector vd8_x3_rgb = Q6_V_vrdelta_VV(Q6_V_hi_W(wd8_x1_rgb0), v_vrdelta);

    HVX_Vector vd8_r = Q6_V_valign_VVR(vd8_x1_rgb, Q6_V_vror_VR(vd8_x0_rgb, 96), 32);
    HVX_Vector vd8_g = Q6_V_valign_VVR(vd8_x2_rgb, Q6_V_vror_VR(vd8_x1_rgb, 96), 64);
    HVX_Vector vd8_b = Q6_V_valign_VVR(vd8_x3_rgb, Q6_V_vror_VR(vd8_x2_rgb, 96), 96);

    vmemu(addr)                  = vd8_r;
    vmemu(addr + AURA_HVLEN)     = vd8_g;
    vmemu(addr + AURA_HVLEN * 2) = vd8_b;
}

// sizeof(Tp) == 2
template <typename Tp, typename std::enable_if<sizeof(Tp) == 2>::type* = DT_NULL>
AURA_INLINE DT_VOID vload(const Tp *addr, HVX_VectorX3 &mvd16_out)
{
    HVX_Vector vd16_x0 = vmemu(addr);
    HVX_Vector vd16_x1 = vmemu(addr + AURA_HVLEN / 2);
    HVX_Vector vd16_x2 = vmemu(addr + AURA_HVLEN);

    HVX_Vector v_vdelta = vmemu(vdelta_load_d16);
    HVX_Vector vd16_x0_rgbx = Q6_V_vdelta_VV(vd16_x0, v_vdelta);
    HVX_Vector vd16_x1_rgbx = Q6_V_vdelta_VV(Q6_V_vlalign_VVR(vd16_x1, vd16_x0, 32), v_vdelta);
    HVX_Vector vd16_x2_rgbx = Q6_V_vdelta_VV(Q6_V_vlalign_VVR(vd16_x2, vd16_x1, 64), v_vdelta);
    HVX_Vector vd16_x3_rgbx = Q6_V_vdelta_VV(Q6_V_vror_VR(vd16_x2, 32), v_vdelta);

    HVX_VectorPair wd16_x0_rg_bx = Q6_W_vdeal_VVR(vd16_x1_rgbx, vd16_x0_rgbx, -4);
    HVX_VectorPair wd16_x1_rg_bx = Q6_W_vdeal_VVR(vd16_x3_rgbx, vd16_x2_rgbx, -4);
    HVX_VectorPair wd16_r_g      = Q6_W_vdeal_VVR(Q6_V_lo_W(wd16_x1_rg_bx), Q6_V_lo_W(wd16_x0_rg_bx), -2);
    HVX_VectorPair wd16_b_x      = Q6_W_vdeal_VVR(Q6_V_hi_W(wd16_x1_rg_bx), Q6_V_hi_W(wd16_x0_rg_bx), -2);

    mvd16_out.val[0] = Q6_V_lo_W(wd16_r_g);
    mvd16_out.val[1] = Q6_V_hi_W(wd16_r_g);
    mvd16_out.val[2] = Q6_V_lo_W(wd16_b_x);
}

template <typename Tp, typename std::enable_if<sizeof(Tp) == 2>::type* = DT_NULL>
AURA_INLINE DT_VOID vstore(Tp *addr, const HVX_VectorX3 &mvd16_in)
{
    HVX_VectorPair wd16_rg_rg   = Q6_W_vshuff_VVR(mvd16_in.val[1], mvd16_in.val[0], -2);
    HVX_VectorPair wd16_b0_b0   = Q6_W_vshuff_VVR(Q6_V_vzero(), mvd16_in.val[2], -2);
    HVX_VectorPair wd16_x0_rgb0 = Q6_W_vshuff_VVR(Q6_V_lo_W(wd16_b0_b0), Q6_V_lo_W(wd16_rg_rg), -4);
    HVX_VectorPair wd16_x1_rgb0 = Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_b0_b0), Q6_V_hi_W(wd16_rg_rg), -4);

    HVX_Vector v_vrdelta = vmemu(vrdelta_store_d16);
    HVX_Vector vd16_x0_rgb = Q6_V_vrdelta_VV(Q6_V_lo_W(wd16_x0_rgb0), v_vrdelta);
    HVX_Vector vd16_x1_rgb = Q6_V_vrdelta_VV(Q6_V_hi_W(wd16_x0_rgb0), v_vrdelta);
    HVX_Vector vd16_x2_rgb = Q6_V_vrdelta_VV(Q6_V_lo_W(wd16_x1_rgb0), v_vrdelta);
    HVX_Vector vd16_x3_rgb = Q6_V_vrdelta_VV(Q6_V_hi_W(wd16_x1_rgb0), v_vrdelta);

    HVX_Vector vd16_r = Q6_V_valign_VVR(vd16_x1_rgb, Q6_V_vror_VR(vd16_x0_rgb, 96), 32);
    HVX_Vector vd16_g = Q6_V_valign_VVR(vd16_x2_rgb, Q6_V_vror_VR(vd16_x1_rgb, 96), 64);
    HVX_Vector vd16_b = Q6_V_valign_VVR(vd16_x3_rgb, Q6_V_vror_VR(vd16_x2_rgb, 96), 96);

    vmemu(addr)                  = vd16_r;
    vmemu(addr + AURA_HVLEN / 2) = vd16_g;
    vmemu(addr + AURA_HVLEN)     = vd16_b;
}

// sizeof(Tp) == 4
template <typename Tp, typename std::enable_if<sizeof(Tp) == 4>::type* = DT_NULL>
AURA_INLINE DT_VOID vload(const Tp *addr, HVX_VectorX3 &mvd32_out)
{
    HVX_Vector vd32_x0 = vmemu(addr);
    HVX_Vector vd32_x1 = vmemu(addr + AURA_HALF_HVLEN / 2);
    HVX_Vector vd32_x2 = vmemu(addr + AURA_HALF_HVLEN);

    HVX_Vector v_vdelta = vmemu(vdelta_load_d32);
    HVX_Vector vd32_x0_rgbx = Q6_V_vdelta_VV(vd32_x0, v_vdelta);
    HVX_Vector vd32_x1_rgbx = Q6_V_vdelta_VV(Q6_V_vlalign_VVR(vd32_x1, vd32_x0, 32), v_vdelta);
    HVX_Vector vd32_x2_rgbx = Q6_V_vdelta_VV(Q6_V_vlalign_VVR(vd32_x2, vd32_x1, 64), v_vdelta);
    HVX_Vector vd32_x3_rgbx = Q6_V_vdelta_VV(Q6_V_vror_VR(vd32_x2, 32), v_vdelta);

    HVX_VectorPair wd32_x0_rg_bx = Q6_W_vdeal_VVR(vd32_x1_rgbx, vd32_x0_rgbx, -8);
    HVX_VectorPair wd32_x1_rg_bx = Q6_W_vdeal_VVR(vd32_x3_rgbx, vd32_x2_rgbx, -8);
    HVX_VectorPair wd32_r_g = Q6_W_vdeal_VVR(Q6_V_lo_W(wd32_x1_rg_bx), Q6_V_lo_W(wd32_x0_rg_bx), -4);
    HVX_VectorPair wd32_b_x = Q6_W_vdeal_VVR(Q6_V_hi_W(wd32_x1_rg_bx), Q6_V_hi_W(wd32_x0_rg_bx), -4);

    mvd32_out.val[0] = Q6_V_lo_W(wd32_r_g);
    mvd32_out.val[1] = Q6_V_hi_W(wd32_r_g);
    mvd32_out.val[2] = Q6_V_lo_W(wd32_b_x);
}

template <typename Tp, typename std::enable_if<sizeof(Tp) == 4>::type* = DT_NULL>
AURA_INLINE DT_VOID vstore(Tp *addr, const HVX_VectorX3 &mvd32_in)
{
    HVX_VectorPair wd32_rg_rg = Q6_W_vshuff_VVR(mvd32_in.val[1], mvd32_in.val[0], -4);
    HVX_VectorPair wd32_b0_b0 = Q6_W_vshuff_VVR(Q6_V_vzero(), mvd32_in.val[2], -4);
    HVX_VectorPair wd32_x0_rgb0 = Q6_W_vshuff_VVR(Q6_V_lo_W(wd32_b0_b0), Q6_V_lo_W(wd32_rg_rg), -8);
    HVX_VectorPair wd32_x1_rgb0 = Q6_W_vshuff_VVR(Q6_V_hi_W(wd32_b0_b0), Q6_V_hi_W(wd32_rg_rg), -8);

    HVX_Vector v_vrdelta = vmemu(vrdelta_store_d32);
    HVX_Vector vd32_x0_rgb = Q6_V_vrdelta_VV(Q6_V_lo_W(wd32_x0_rgb0), v_vrdelta);
    HVX_Vector vd32_x1_rgb = Q6_V_vrdelta_VV(Q6_V_hi_W(wd32_x0_rgb0), v_vrdelta);
    HVX_Vector vd32_x2_rgb = Q6_V_vrdelta_VV(Q6_V_lo_W(wd32_x1_rgb0), v_vrdelta);
    HVX_Vector vd32_x3_rgb = Q6_V_vrdelta_VV(Q6_V_hi_W(wd32_x1_rgb0), v_vrdelta);

    HVX_Vector vd32_r = Q6_V_valign_VVR(vd32_x1_rgb, Q6_V_vror_VR(vd32_x0_rgb, 96), 32);
    HVX_Vector vd32_g = Q6_V_valign_VVR(vd32_x2_rgb, Q6_V_vror_VR(vd32_x1_rgb, 96), 64);
    HVX_Vector vd32_b = Q6_V_valign_VVR(vd32_x3_rgb, Q6_V_vror_VR(vd32_x2_rgb, 96), 96);

    vmemu(addr)                       = vd32_r;
    vmemu(addr + AURA_HALF_HVLEN / 2) = vd32_g;
    vmemu(addr + AURA_HALF_HVLEN)     = vd32_b;
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_LOAD_HPP__