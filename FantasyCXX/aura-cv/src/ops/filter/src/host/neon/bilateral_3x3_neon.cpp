#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename MVTp>
AURA_ALWAYS_INLINE DT_VOID Bilateral3x3U8Prepare(MVTp *mvdu8_p0_src, MVTp *mvdu8_c_src, MVTp *mvdu8_n0_src)
{
    mvdu8_p0_src[0] = mvdu8_p0_src[1];
    mvdu8_c_src[0]  = mvdu8_c_src[1];
    mvdu8_n0_src[0] = mvdu8_n0_src[1];

    mvdu8_p0_src[1] = mvdu8_p0_src[2];
    mvdu8_c_src[1]  = mvdu8_c_src[2];
    mvdu8_n0_src[1] = mvdu8_n0_src[2];
}

AURA_ALWAYS_INLINE uint8x8x1_t Bilateral3x3U8VectorCore(uint8x8x1_t &vdu8_p0x1_src, uint8x8x1_t &vdu8_cx0_src, uint8x8x1_t &vdu8_cx1_src,
                                                        uint8x8x1_t &vdu8_cx2_src, uint8x8x1_t &vdu8_n0x1_src,
                                                        uint8x8x4_t &v4du8_color_weight_tbl, uint8x8_t *vdu8_space_scale)
{
    uint8x8x1_t vdu8_cl0_src, vdu8_cr0_src;

    vdu8_cl0_src.val[0] = neon::vext<7>(vdu8_cx0_src.val[0], vdu8_cx1_src.val[0]);
    vdu8_cr0_src.val[0] = neon::vext<1>(vdu8_cx1_src.val[0], vdu8_cx2_src.val[0]);

    // top
    uint8x8_t vdu8_color_w;
    neon::vdup(vdu8_color_w, 0);
    uint8x8_t vdu8_idx       = neon::vabd(vdu8_p0x1_src.val[0], vdu8_cx1_src.val[0]);
    vdu8_color_w             = neon::vtbx(vdu8_color_w, v4du8_color_weight_tbl, vdu8_idx);
    uint16x8_t vqu16_w       = neon::vmull(vdu8_color_w, vdu8_space_scale[0]);
    uint32x4_t vqu16_w_sum_l = neon::vmovl(neon::vgetlow(vqu16_w));
    uint32x4_t vqu16_w_sum_h = neon::vmovl(neon::vgethigh(vqu16_w));
    uint16x8_t vqu16_p0c_src = neon::vmovl(vdu8_p0x1_src.val[0]);
    uint32x4_t vqu32_sum_l   = neon::vmull(neon::vgetlow(vqu16_p0c_src),  neon::vgetlow(vqu16_w));
    uint32x4_t vqu32_sum_h   = neon::vmull(neon::vgethigh(vqu16_p0c_src), neon::vgethigh(vqu16_w));

    // left
    neon::vdup(vdu8_color_w, 0);
    vdu8_idx                 = neon::vabd(vdu8_cl0_src.val[0], vdu8_cx1_src.val[0]);
    vdu8_color_w             = neon::vtbx(vdu8_color_w, v4du8_color_weight_tbl, vdu8_idx);
    vqu16_w                  = neon::vmull(vdu8_color_w, vdu8_space_scale[1]);
    vqu16_w_sum_l            = neon::vaddw(vqu16_w_sum_l, neon::vgetlow(vqu16_w));
    vqu16_w_sum_h            = neon::vaddw(vqu16_w_sum_h, neon::vgethigh(vqu16_w));
    uint16x8_t vqu16_cl0_src = neon::vmovl(vdu8_cl0_src.val[0]);
    vqu32_sum_l              = neon::vmlal(vqu32_sum_l, neon::vgetlow(vqu16_cl0_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h              = neon::vmlal(vqu32_sum_h, neon::vgethigh(vqu16_cl0_src), neon::vgethigh(vqu16_w));

    // center
    neon::vdup(vdu8_color_w, neon::vgetlane<0>(v4du8_color_weight_tbl.val[0]));
    vqu16_w                  = neon::vmull(vdu8_color_w, vdu8_space_scale[2]);
    vqu16_w_sum_l            = neon::vaddw(vqu16_w_sum_l, neon::vgetlow(vqu16_w));
    vqu16_w_sum_h            = neon::vaddw(vqu16_w_sum_h, neon::vgethigh(vqu16_w));
    uint16x8_t vqu16_cc_src  = neon::vmovl(vdu8_cx1_src.val[0]);
    vqu32_sum_l              = neon::vmlal(vqu32_sum_l, neon::vgetlow(vqu16_cc_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h              = neon::vmlal(vqu32_sum_h, neon::vgethigh(vqu16_cc_src), neon::vgethigh(vqu16_w));

    // right
    neon::vdup(vdu8_color_w, 0);
    vdu8_idx                 = neon::vabd(vdu8_cr0_src.val[0], vdu8_cx1_src.val[0]);
    vdu8_color_w             = neon::vtbx(vdu8_color_w, v4du8_color_weight_tbl, vdu8_idx);
    vqu16_w                  = neon::vmull(vdu8_color_w, vdu8_space_scale[3]);
    vqu16_w_sum_l            = neon::vaddw(vqu16_w_sum_l, neon::vgetlow(vqu16_w));
    vqu16_w_sum_h            = neon::vaddw(vqu16_w_sum_h, neon::vgethigh(vqu16_w));
    uint16x8_t vqu16_cr0_src = neon::vmovl(vdu8_cr0_src.val[0]);
    vqu32_sum_l              = neon::vmlal(vqu32_sum_l, neon::vgetlow(vqu16_cr0_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h              = neon::vmlal(vqu32_sum_h, neon::vgethigh(vqu16_cr0_src), neon::vgethigh(vqu16_w));

    // bottom
    neon::vdup(vdu8_color_w, 0);
    vdu8_idx                 = neon::vabd(vdu8_n0x1_src.val[0], vdu8_cx1_src.val[0]);
    vdu8_color_w             = neon::vtbx(vdu8_color_w, v4du8_color_weight_tbl, vdu8_idx);
    vqu16_w                  = neon::vmull(vdu8_color_w, vdu8_space_scale[4]);
    vqu16_w_sum_l            = neon::vaddw(vqu16_w_sum_l, neon::vgetlow(vqu16_w));
    vqu16_w_sum_h            = neon::vaddw(vqu16_w_sum_h, neon::vgethigh(vqu16_w));
    uint16x8_t vqu16_n0c_src = neon::vmovl(vdu8_n0x1_src.val[0]);
    vqu32_sum_l              = neon::vmlal(vqu32_sum_l, neon::vgetlow(vqu16_n0c_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h              = neon::vmlal(vqu32_sum_h, neon::vgethigh(vqu16_n0c_src), neon::vgethigh(vqu16_w));

    float32x4_t vqf32_wei_sum_l = neon::vcvt<DT_F32>(vqu16_w_sum_l);
    float32x4_t vqf32_wei_sum_h = neon::vcvt<DT_F32>(vqu16_w_sum_h);
    float32x4_t vqf32_sum_l     = neon::vcvt<DT_F32>(vqu32_sum_l);
    float32x4_t vqf32_sum_h     = neon::vcvt<DT_F32>(vqu32_sum_h);

    uint16x4_t  vdu16_dst_l, vdu16_dst_h;
    float32x4_t vqf32_reciprocal;
    vqf32_reciprocal = neon::vreciprocal_newton(vqf32_wei_sum_l);
    vdu16_dst_l      = neon::vmovn(neon::vcvt<DT_U32>(neon::vrndn(neon::vmul(vqf32_sum_l, vqf32_reciprocal))));

    vqf32_reciprocal = neon::vreciprocal_newton(vqf32_wei_sum_h);
    vdu16_dst_h      = neon::vmovn(neon::vcvt<DT_U32>(neon::vrndn(neon::vmul(vqf32_sum_h, vqf32_reciprocal))));

    uint8x8x1_t vdu8_dst;
    vdu8_dst.val[0]  = neon::vqmovn(neon::vcombine(vdu16_dst_l, vdu16_dst_h));

    return vdu8_dst;
}

AURA_ALWAYS_INLINE uint8x8x3_t Bilateral3x3U8VectorCore(uint8x8x3_t &vdu8_p0x1_src, uint8x8x3_t &vdu8_cx0_src, uint8x8x3_t &vdu8_cx1_src,
                                                        uint8x8x3_t &vdu8_cx2_src, uint8x8x3_t &vdu8_n0x1_src,
                                                        uint8x8x4_t &v4du8_color_weight_tbl, uint8x8_t *vdu8_space_scale)
{
    uint8x8x3_t vdu8_cl0_src, vdu8_cr0_src;

    vdu8_cl0_src.val[0] = neon::vext<7>(vdu8_cx0_src.val[0], vdu8_cx1_src.val[0]);
    vdu8_cl0_src.val[1] = neon::vext<7>(vdu8_cx0_src.val[1], vdu8_cx1_src.val[1]);
    vdu8_cl0_src.val[2] = neon::vext<7>(vdu8_cx0_src.val[2], vdu8_cx1_src.val[2]);

    vdu8_cr0_src.val[0] = neon::vext<1>(vdu8_cx1_src.val[0], vdu8_cx2_src.val[0]);
    vdu8_cr0_src.val[1] = neon::vext<1>(vdu8_cx1_src.val[1], vdu8_cx2_src.val[1]);
    vdu8_cr0_src.val[2] = neon::vext<1>(vdu8_cx1_src.val[2], vdu8_cx2_src.val[2]);

    // top
    uint8x8_t vdu8_color_w;
    neon::vdup(vdu8_color_w, 0);
    uint8x8_t vdu8_idx_r       = neon::vabd(vdu8_p0x1_src.val[0], vdu8_cx1_src.val[0]);
    uint8x8_t vdu8_idx_g       = neon::vabd(vdu8_p0x1_src.val[1], vdu8_cx1_src.val[1]);
    uint8x8_t vdu8_idx_b       = neon::vabd(vdu8_p0x1_src.val[2], vdu8_cx1_src.val[2]);
    uint8x8_t vdu8_idx         = neon::vqadd(vdu8_idx_r, neon::vqadd(vdu8_idx_g, vdu8_idx_b));
    vdu8_color_w               = neon::vtbx(vdu8_color_w, v4du8_color_weight_tbl, vdu8_idx);
    uint16x8_t vqu16_w         = neon::vmull(vdu8_color_w, vdu8_space_scale[0]);
    uint32x4_t vqu16_w_sum_l   = neon::vmovl(neon::vgetlow(vqu16_w));
    uint32x4_t vqu16_w_sum_h   = neon::vmovl(neon::vgethigh(vqu16_w));
    uint16x8_t vqu16_p0c_src   = neon::vmovl(vdu8_p0x1_src.val[0]);
    uint32x4_t vqu32_sum_l_r   = neon::vmull(neon::vgetlow(vqu16_p0c_src),  neon::vgetlow(vqu16_w));
    uint32x4_t vqu32_sum_h_r   = neon::vmull(neon::vgethigh(vqu16_p0c_src), neon::vgethigh(vqu16_w));
    vqu16_p0c_src              = neon::vmovl(vdu8_p0x1_src.val[1]);
    uint32x4_t vqu32_sum_l_g   = neon::vmull(neon::vgetlow(vqu16_p0c_src),  neon::vgetlow(vqu16_w));
    uint32x4_t vqu32_sum_h_g   = neon::vmull(neon::vgethigh(vqu16_p0c_src), neon::vgethigh(vqu16_w));
    vqu16_p0c_src              = neon::vmovl(vdu8_p0x1_src.val[2]);
    uint32x4_t vqu32_sum_l_b   = neon::vmull(neon::vgetlow(vqu16_p0c_src),  neon::vgetlow(vqu16_w));
    uint32x4_t vqu32_sum_h_b   = neon::vmull(neon::vgethigh(vqu16_p0c_src), neon::vgethigh(vqu16_w));

    // left
    neon::vdup(vdu8_color_w, 0);
    vdu8_idx_r               = neon::vabd(vdu8_cl0_src.val[0], vdu8_cx1_src.val[0]);
    vdu8_idx_g               = neon::vabd(vdu8_cl0_src.val[1], vdu8_cx1_src.val[1]);
    vdu8_idx_b               = neon::vabd(vdu8_cl0_src.val[2], vdu8_cx1_src.val[2]);
    vdu8_idx                 = neon::vqadd(vdu8_idx_r, neon::vqadd(vdu8_idx_g, vdu8_idx_b));
    vdu8_color_w             = neon::vtbx(vdu8_color_w, v4du8_color_weight_tbl, vdu8_idx);
    vqu16_w                  = neon::vmull(vdu8_color_w, vdu8_space_scale[1]);
    vqu16_w_sum_l            = neon::vaddw(vqu16_w_sum_l, neon::vgetlow(vqu16_w));
    vqu16_w_sum_h            = neon::vaddw(vqu16_w_sum_h, neon::vgethigh(vqu16_w));
    uint16x8_t vqu16_cl0_src = neon::vmovl(vdu8_cl0_src.val[0]);
    vqu32_sum_l_r            = neon::vmlal(vqu32_sum_l_r, neon::vgetlow(vqu16_cl0_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_r            = neon::vmlal(vqu32_sum_h_r, neon::vgethigh(vqu16_cl0_src), neon::vgethigh(vqu16_w));
    vqu16_cl0_src            = neon::vmovl(vdu8_cl0_src.val[1]);
    vqu32_sum_l_g            = neon::vmlal(vqu32_sum_l_g, neon::vgetlow(vqu16_cl0_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_g            = neon::vmlal(vqu32_sum_h_g, neon::vgethigh(vqu16_cl0_src), neon::vgethigh(vqu16_w));
    vqu16_cl0_src            = neon::vmovl(vdu8_cl0_src.val[2]);
    vqu32_sum_l_b            = neon::vmlal(vqu32_sum_l_b, neon::vgetlow(vqu16_cl0_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_b            = neon::vmlal(vqu32_sum_h_b, neon::vgethigh(vqu16_cl0_src), neon::vgethigh(vqu16_w));

    // center
    neon::vdup(vdu8_color_w, neon::vgetlane<0>(v4du8_color_weight_tbl.val[0]));
    vqu16_w                  = neon::vmull(vdu8_color_w, vdu8_space_scale[2]);
    vqu16_w_sum_l            = neon::vaddw(vqu16_w_sum_l, neon::vgetlow(vqu16_w));
    vqu16_w_sum_h            = neon::vaddw(vqu16_w_sum_h, neon::vgethigh(vqu16_w));
    uint16x8_t vqu16_cc_src  = neon::vmovl(vdu8_cx1_src.val[0]);
    vqu32_sum_l_r            = neon::vmlal(vqu32_sum_l_r, neon::vgetlow(vqu16_cc_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_r            = neon::vmlal(vqu32_sum_h_r, neon::vgethigh(vqu16_cc_src), neon::vgethigh(vqu16_w));
    vqu16_cc_src             = neon::vmovl(vdu8_cx1_src.val[1]);
    vqu32_sum_l_g            = neon::vmlal(vqu32_sum_l_g, neon::vgetlow(vqu16_cc_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_g            = neon::vmlal(vqu32_sum_h_g, neon::vgethigh(vqu16_cc_src), neon::vgethigh(vqu16_w));
    vqu16_cc_src             = neon::vmovl(vdu8_cx1_src.val[2]);
    vqu32_sum_l_b            = neon::vmlal(vqu32_sum_l_b, neon::vgetlow(vqu16_cc_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_b            = neon::vmlal(vqu32_sum_h_b, neon::vgethigh(vqu16_cc_src), neon::vgethigh(vqu16_w));

    // right
    neon::vdup(vdu8_color_w, 0);
    vdu8_idx_r               = neon::vabd(vdu8_cr0_src.val[0], vdu8_cx1_src.val[0]);
    vdu8_idx_g               = neon::vabd(vdu8_cr0_src.val[1], vdu8_cx1_src.val[1]);
    vdu8_idx_b               = neon::vabd(vdu8_cr0_src.val[2], vdu8_cx1_src.val[2]);
    vdu8_idx                 = neon::vqadd(vdu8_idx_r, neon::vqadd(vdu8_idx_g, vdu8_idx_b));
    vdu8_color_w             = neon::vtbx(vdu8_color_w, v4du8_color_weight_tbl, vdu8_idx);
    vqu16_w                  = neon::vmull(vdu8_color_w, vdu8_space_scale[3]);
    vqu16_w_sum_l            = neon::vaddw(vqu16_w_sum_l, neon::vgetlow(vqu16_w));
    vqu16_w_sum_h            = neon::vaddw(vqu16_w_sum_h, neon::vgethigh(vqu16_w));
    uint16x8_t vqu16_cr0_src = neon::vmovl(vdu8_cr0_src.val[0]);
    vqu32_sum_l_r            = neon::vmlal(vqu32_sum_l_r, neon::vgetlow(vqu16_cr0_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_r            = neon::vmlal(vqu32_sum_h_r, neon::vgethigh(vqu16_cr0_src), neon::vgethigh(vqu16_w));
    vqu16_cr0_src            = neon::vmovl(vdu8_cr0_src.val[1]);
    vqu32_sum_l_g            = neon::vmlal(vqu32_sum_l_g, neon::vgetlow(vqu16_cr0_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_g            = neon::vmlal(vqu32_sum_h_g, neon::vgethigh(vqu16_cr0_src), neon::vgethigh(vqu16_w));
    vqu16_cr0_src            = neon::vmovl(vdu8_cr0_src.val[2]);
    vqu32_sum_l_b            = neon::vmlal(vqu32_sum_l_b, neon::vgetlow(vqu16_cr0_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_b            = neon::vmlal(vqu32_sum_h_b, neon::vgethigh(vqu16_cr0_src), neon::vgethigh(vqu16_w));

    // bottom
    neon::vdup(vdu8_color_w, 0);
    vdu8_idx_r               = neon::vabd(vdu8_n0x1_src.val[0], vdu8_cx1_src.val[0]);
    vdu8_idx_g               = neon::vabd(vdu8_n0x1_src.val[1], vdu8_cx1_src.val[1]);
    vdu8_idx_b               = neon::vabd(vdu8_n0x1_src.val[2], vdu8_cx1_src.val[2]);
    vdu8_idx                 = neon::vqadd(vdu8_idx_r, neon::vqadd(vdu8_idx_g, vdu8_idx_b));
    vdu8_color_w             = neon::vtbx(vdu8_color_w, v4du8_color_weight_tbl, vdu8_idx);
    vqu16_w                  = neon::vmull(vdu8_color_w, vdu8_space_scale[4]);
    vqu16_w_sum_l            = neon::vaddw(vqu16_w_sum_l, neon::vgetlow(vqu16_w));
    vqu16_w_sum_h            = neon::vaddw(vqu16_w_sum_h, neon::vgethigh(vqu16_w));
    uint16x8_t vqu16_n0c_src = neon::vmovl(vdu8_n0x1_src.val[0]);
    vqu32_sum_l_r            = neon::vmlal(vqu32_sum_l_r, neon::vgetlow(vqu16_n0c_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_r            = neon::vmlal(vqu32_sum_h_r, neon::vgethigh(vqu16_n0c_src), neon::vgethigh(vqu16_w));
    vqu16_n0c_src            = neon::vmovl(vdu8_n0x1_src.val[1]);
    vqu32_sum_l_g            = neon::vmlal(vqu32_sum_l_g, neon::vgetlow(vqu16_n0c_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_g            = neon::vmlal(vqu32_sum_h_g, neon::vgethigh(vqu16_n0c_src), neon::vgethigh(vqu16_w));
    vqu16_n0c_src            = neon::vmovl(vdu8_n0x1_src.val[2]);
    vqu32_sum_l_b            = neon::vmlal(vqu32_sum_l_b, neon::vgetlow(vqu16_n0c_src), neon::vgetlow(vqu16_w));
    vqu32_sum_h_b            = neon::vmlal(vqu32_sum_h_b, neon::vgethigh(vqu16_n0c_src), neon::vgethigh(vqu16_w));

    uint8x8x3_t v3du8_dst;
    // r
    float32x4_t vqf32_wei_sum_l = neon::vcvt<DT_F32>(vqu16_w_sum_l);
    float32x4_t vqf32_wei_sum_h = neon::vcvt<DT_F32>(vqu16_w_sum_h);
    float32x4_t vqf32_sum_l     = neon::vcvt<DT_F32>(vqu32_sum_l_r);
    float32x4_t vqf32_sum_h     = neon::vcvt<DT_F32>(vqu32_sum_h_r);

    uint16x4_t  vdu16_dst_l, vdu16_dst_h;
    float32x4_t vqf32_reciprocal_l, vqf32_reciprocal_h;
    vqf32_reciprocal_l = neon::vreciprocal_newton(vqf32_wei_sum_l);
    vdu16_dst_l        = neon::vmovn(neon::vcvt<DT_U32>(neon::vrndn(neon::vmul(vqf32_sum_l, vqf32_reciprocal_l))));
    vqf32_reciprocal_h = neon::vreciprocal_newton(vqf32_wei_sum_h);
    vdu16_dst_h        = neon::vmovn(neon::vcvt<DT_U32>(neon::vrndn(neon::vmul(vqf32_sum_h, vqf32_reciprocal_h))));

    v3du8_dst.val[0]   = neon::vqmovn(neon::vcombine(vdu16_dst_l, vdu16_dst_h));

    // g
    vqf32_sum_l      = neon::vcvt<DT_F32>(vqu32_sum_l_g);
    vqf32_sum_h      = neon::vcvt<DT_F32>(vqu32_sum_h_g);
    vdu16_dst_l      = neon::vmovn(neon::vcvt<DT_U32>(neon::vrndn(neon::vmul(vqf32_sum_l, vqf32_reciprocal_l))));
    vdu16_dst_h      = neon::vmovn(neon::vcvt<DT_U32>(neon::vrndn(neon::vmul(vqf32_sum_h, vqf32_reciprocal_h))));
    v3du8_dst.val[1] = neon::vqmovn(neon::vcombine(vdu16_dst_l, vdu16_dst_h));

    // b
    vqf32_sum_l      = neon::vcvt<DT_F32>(vqu32_sum_l_b);
    vqf32_sum_h      = neon::vcvt<DT_F32>(vqu32_sum_h_b);
    vdu16_dst_l      = neon::vmovn(neon::vcvt<DT_U32>(neon::vrndn(neon::vmul(vqf32_sum_l, vqf32_reciprocal_l))));
    vdu16_dst_h      = neon::vmovn(neon::vcvt<DT_U32>(neon::vrndn(neon::vmul(vqf32_sum_h, vqf32_reciprocal_h))));
    v3du8_dst.val[2] = neon::vqmovn(neon::vcombine(vdu16_dst_l, vdu16_dst_h));

    return v3du8_dst;
}

template <BorderType BORDER_TYPE, DT_S32 C>
static DT_VOID Bilateral3x3U8Row(const DT_U8 *src_p0, const DT_U8 *src_c, const DT_U8 *src_n0, DT_U8 *dst, const std::vector<DT_U8> &space_weight,
                                 uint8x8x4_t &v4du8_color_weight_tbl, const std::vector<DT_U8> &border_value, DT_S32 width)
{
    constexpr DT_S32 elem_counts = 8;
    constexpr DT_S32 voffset     = elem_counts * C;
    const DT_S32 width_align8    = (width & -elem_counts) * C;

    uint8x8_t vdu8_space_scale[5];
    neon::vdup(vdu8_space_scale[0], space_weight[0]);
    neon::vdup(vdu8_space_scale[1], space_weight[1]);
    neon::vdup(vdu8_space_scale[2], space_weight[2]);
    neon::vdup(vdu8_space_scale[3], space_weight[3]);
    neon::vdup(vdu8_space_scale[4], space_weight[4]);

    using MVType = typename neon::MDVector<DT_U8, C>::MVType;

    MVType mvdu8_p0_src[3], mvdu8_c_src[3], mvdu8_n0_src[3];
    MVType mvdu8_dst;

    // left border
    {
        neon::vload(src_p0, mvdu8_p0_src[1]);
        neon::vload(src_c,  mvdu8_c_src[1]);
        neon::vload(src_n0, mvdu8_n0_src[1]);
        neon::vload(src_p0 + voffset, mvdu8_p0_src[2]);
        neon::vload(src_c  + voffset, mvdu8_c_src[2]);
        neon::vload(src_n0 + voffset, mvdu8_n0_src[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvdu8_p0_src[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvdu8_p0_src[1].val[ch], src_p0[ch], border_value[ch]);
            mvdu8_c_src[0].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvdu8_c_src[1].val[ch], src_c[ch], border_value[ch]);
            mvdu8_n0_src[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mvdu8_n0_src[1].val[ch], src_n0[ch], border_value[ch]);
        }

        mvdu8_dst = Bilateral3x3U8VectorCore(mvdu8_p0_src[1], mvdu8_c_src[0], mvdu8_c_src[1], mvdu8_c_src[2],
                                             mvdu8_n0_src[1], v4du8_color_weight_tbl, vdu8_space_scale);

        neon::vstore(dst, mvdu8_dst);

        Bilateral3x3U8Prepare(mvdu8_p0_src, mvdu8_c_src, mvdu8_n0_src);
    }

    // Middle
    {
        for (DT_S32 x = voffset; x < (width_align8 - voffset); x += voffset)
        {
            neon::vload(src_p0 + x + voffset, mvdu8_p0_src[2]);
            neon::vload(src_c  + x + voffset, mvdu8_c_src[2]);
            neon::vload(src_n0 + x + voffset, mvdu8_n0_src[2]);

            mvdu8_dst = Bilateral3x3U8VectorCore(mvdu8_p0_src[1], mvdu8_c_src[0], mvdu8_c_src[1], mvdu8_c_src[2],
                                                 mvdu8_n0_src[1], v4du8_color_weight_tbl, vdu8_space_scale);

            neon::vstore(dst + x, mvdu8_dst);

            Bilateral3x3U8Prepare(mvdu8_p0_src, mvdu8_c_src, mvdu8_n0_src);
        }
    }

    // back
    {
        if (width_align8 != width * C)
        {
            DT_S32 x = (width - elem_counts * 2) * C;

            neon::vload(src_p0 + x - voffset, mvdu8_p0_src[0]);
            neon::vload(src_p0 + x, mvdu8_p0_src[1]);
            neon::vload(src_p0 + x + voffset, mvdu8_p0_src[2]);
            neon::vload(src_c + x - voffset, mvdu8_c_src[0]);
            neon::vload(src_c + x, mvdu8_c_src[1]);
            neon::vload(src_c + x + voffset, mvdu8_c_src[2]);
            neon::vload(src_n0 + x - voffset, mvdu8_n0_src[0]);
            neon::vload(src_n0 + x, mvdu8_n0_src[1]);
            neon::vload(src_n0 + x + voffset, mvdu8_n0_src[2]);

            mvdu8_dst = Bilateral3x3U8VectorCore(mvdu8_p0_src[1], mvdu8_c_src[0], mvdu8_c_src[1], mvdu8_c_src[2],
                                                 mvdu8_n0_src[1], v4du8_color_weight_tbl, vdu8_space_scale);

            neon::vstore(dst + x, mvdu8_dst);

            Bilateral3x3U8Prepare(mvdu8_p0_src, mvdu8_c_src, mvdu8_n0_src);
        }
    }
    // right
    {
        DT_S32 x = (width - 8) * C;
        DT_S32 last = (width - 1) * C;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mvdu8_p0_src[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvdu8_p0_src[1].val[ch], src_p0[last], border_value[ch]);
            mvdu8_c_src[2].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvdu8_c_src[1].val[ch], src_c[last], border_value[ch]);
            mvdu8_n0_src[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mvdu8_n0_src[1].val[ch], src_n0[last], border_value[ch]);

            last++;
        }

        mvdu8_dst = Bilateral3x3U8VectorCore(mvdu8_p0_src[1], mvdu8_c_src[0], mvdu8_c_src[1], mvdu8_c_src[2],
                                             mvdu8_n0_src[1], v4du8_color_weight_tbl, vdu8_space_scale);

        neon::vstore(dst + x, mvdu8_dst);
    }
}

template <BorderType BORDER_TYPE, DT_S32 C>
static Status Bilateral3x3U8NeonImpl(const Mat &src, Mat &dst, std::vector<DT_U8> &space_weight, std::vector<DT_U8> &color_weight,
                                     const std::vector<DT_U8> &border_value, const DT_U8 *border_buffer, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width = dst.GetSizes().m_width;

    uint8x8x4_t v4du8_color_weight_tbl;
    for (DT_S32 j = 0; j < 4; j++)
    {
        v4du8_color_weight_tbl.val[j] = neon::vload1(color_weight.data() + j * 8);
    }

    const DT_U8 *src_p = DT_NULL, *src_c = DT_NULL, *src_n = DT_NULL;
    DT_U8 *dst_c = DT_NULL;

    DT_S32 y = start_row;

    src_p = src.Ptr<DT_U8, BORDER_TYPE>(y - 1, border_buffer);
    src_c = src.Ptr<DT_U8>(y);
    src_n = src.Ptr<DT_U8, BORDER_TYPE>(y + 1, border_buffer);

    for (; y < end_row; y++)
    {
        dst_c = dst.Ptr<DT_U8>(y);
        Bilateral3x3U8Row<BORDER_TYPE, C>(src_p, src_c, src_n, dst_c, space_weight, v4du8_color_weight_tbl, border_value, width);

        src_p = src_c;
        src_c = src_n;
        src_n = src.Ptr<DT_U8, BORDER_TYPE>(y + 2, border_buffer);
    }

    return Status::OK;
}

template<BorderType BORDER_TYPE>
static Status Bilateral3x3U8NeonHelper(Context *ctx, const Mat &src, Mat &dst, std::vector<DT_U8> &space_weight, std::vector<DT_U8> &color_weight,
                                       const std::vector<DT_U8> &border_value, const DT_U8 *border_buffer, const OpTarget &target)
{
    AURA_UNUSED(target);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    DT_S32 height  = dst.GetSizes().m_height;
    DT_S32 channel = dst.GetSizes().m_channel;

    switch(channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, Bilateral3x3U8NeonImpl<BORDER_TYPE, 1>, std::cref(src), std::ref(dst),
                                  std::ref(space_weight), std::ref(color_weight), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Bilateral3x3U8NeonImpl<BORDER_TYPE, 3>, std::cref(src), std::ref(dst),
                                  std::ref(space_weight), std::ref(color_weight), std::cref(border_value), border_buffer);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported channel");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

static Status GetBilateral3x3U8Weight(const DT_F32 *space_data, const DT_F32 *color_data, std::vector<DT_U8> &space, std::vector<DT_U8> &color)
{
    if ((DT_NULL == space_data) || (DT_NULL == color_data))
    {
        return Status::ERROR;
    }

    for (DT_S32 i = 0; i < static_cast<DT_S32>(space.size()); i++)
    {
        space[i] = static_cast<DT_U8>(255 * space_data[i]);
    }

    for (DT_S32 i = 0; i < static_cast<DT_S32>(color.size()); i++)
    {
        color[i] = static_cast<DT_U8>(255 * color_data[i]);
    }

    return Status::OK;
}

static Status Bilateral3x3U8NeonHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &space_weight, const Mat &color_weight,
                                       DT_S32 valid_num, BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    // weight quant
    std::vector<DT_U8> vec_space_weight(valid_num, 0);
    std::vector<DT_U8> vec_color_weight(32, 0);

    DT_U8 *border_buffer                = DT_NULL;
    std::vector<DT_U8> vec_border_value = border_value.ToVector<DT_U8>();
    DT_S32 width                        = dst.GetSizes().m_width;
    DT_S32 channel                      = dst.GetSizes().m_channel;

    ret = GetBilateral3x3U8Weight(space_weight.Ptr<DT_F32>(0), color_weight.Ptr<DT_F32>(0), vec_space_weight, vec_color_weight);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetBilateral3x3U8Weight failed");
        return ret;
    }

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

            ret = Bilateral3x3U8NeonHelper<BorderType::CONSTANT>(ctx, src, dst, vec_space_weight, vec_color_weight, vec_border_value, border_buffer, target);

            AURA_FREE(ctx, border_buffer);

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Bilateral3x3U8NeonHelper<BorderType::CONSTANT> failed");
            }
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Bilateral3x3U8NeonHelper<BorderType::REPLICATE>(ctx, src, dst, vec_space_weight, vec_color_weight, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Bilateral3x3U8NeonHelper<BorderType::REPLICATE> failed");
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Bilateral3x3U8NeonHelper<BorderType::REFLECT_101>(ctx, src, dst, vec_space_weight, vec_color_weight, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Bilateral3x3U8NeonHelper<BorderType::REFLECT_101> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported border_type.");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Bilateral3x3Neon(Context *ctx, const Mat &src, Mat &dst, const Mat &space_weight, const Mat &color_weight,
                        DT_S32 valid_num, BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if (valid_num != 5)
    {
        AURA_ADD_ERROR_STRING(ctx, "valid_num error!");
        return Status::ERROR;
    }

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Bilateral3x3U8NeonHelper(ctx, src, dst, space_weight, color_weight, valid_num, border_type, border_value, target);
            break;
        }

        default :
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura