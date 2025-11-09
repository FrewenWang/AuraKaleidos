#include "morph_impl.hpp"
#include "aura/runtime/worker_pool.h"

namespace aura
{

#define QVECTOR_NUM      (7)

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 ELEM_COUNTS,
          typename VqTpye = typename neon::QVector<Tp>::VType,
          typename std::enable_if<MorphShape::RECT == MORPH_SHAPE, Tp>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Morph7x7Vector(VqTpye &vq_src_p2x0, VqTpye &vq_src_p2x1, VqTpye &vq_src_p2x2,
                                          VqTpye &vq_src_p1x0, VqTpye &vq_src_p1x1, VqTpye &vq_src_p1x2,
                                          VqTpye &vq_src_p0x0, VqTpye &vq_src_p0x1, VqTpye &vq_src_p0x2,
                                          VqTpye &vq_src_cx0,  VqTpye &vq_src_cx1,  VqTpye &vq_src_cx2,
                                          VqTpye &vq_src_n0x0, VqTpye &vq_src_n0x1, VqTpye &vq_src_n0x2,
                                          VqTpye &vq_src_n1x0, VqTpye &vq_src_n1x1, VqTpye &vq_src_n1x2,
                                          VqTpye &vq_src_n2x0, VqTpye &vq_src_n2x1, VqTpye &vq_src_n2x2,
                                          VqTpye &vq_result,   VqTpye *vq_vertical_results)
{
    // vertical results
    vq_vertical_results[0] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p2x0, vq_src_p1x0, vq_src_p0x0, vq_src_cx0, vq_src_n0x0, vq_src_n1x0, vq_src_n2x0);
    vq_vertical_results[1] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p2x1, vq_src_p1x1, vq_src_p0x1, vq_src_cx1, vq_src_n0x1, vq_src_n1x1, vq_src_n2x1);
    vq_vertical_results[2] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p2x2, vq_src_p1x2, vq_src_p0x2, vq_src_cx2, vq_src_n0x2, vq_src_n1x2, vq_src_n2x2);

    // horizonal results
    VqTpye vq_vertical_l2 = neon::vext<ELEM_COUNTS - 3>(vq_vertical_results[0], vq_vertical_results[1]);
    VqTpye vq_vertical_l1 = neon::vext<ELEM_COUNTS - 2>(vq_vertical_results[0], vq_vertical_results[1]);
    VqTpye vq_vertical_l0 = neon::vext<ELEM_COUNTS - 1>(vq_vertical_results[0], vq_vertical_results[1]);
    VqTpye vq_vertical_r0 = neon::vext<1>(vq_vertical_results[1], vq_vertical_results[2]);
    VqTpye vq_vertical_r1 = neon::vext<2>(vq_vertical_results[1], vq_vertical_results[2]);
    VqTpye vq_vertical_r2 = neon::vext<3>(vq_vertical_results[1], vq_vertical_results[2]);

    vq_result = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_vertical_l2, vq_vertical_l1, vq_vertical_l0, vq_vertical_results[1],
                                                vq_vertical_r0, vq_vertical_r1, vq_vertical_r2);

    // slide results
    vq_vertical_results[0] = vq_vertical_results[1];
    vq_vertical_results[1] = vq_vertical_results[2];
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 ELEM_COUNTS,
          typename VqTpye = typename neon::QVector<Tp>::VType,
          typename std::enable_if<MorphShape::RECT == MORPH_SHAPE, Tp>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Morph7x7Vector(VqTpye &vq_src_p2x2, VqTpye &vq_src_p1x2, VqTpye &vq_src_p0x2,
                                          VqTpye &vq_src_cx0,  VqTpye &vq_src_cx1,  VqTpye &vq_src_cx2,
                                          VqTpye &vq_src_n0x2, VqTpye &vq_src_n1x2, VqTpye &vq_src_n2x2,
                                          VqTpye &vq_result,   VqTpye *vq_vertical_results)
{
    AURA_UNUSED(vq_src_cx0);
    AURA_UNUSED(vq_src_cx1);

    // vertical results
    vq_vertical_results[2] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p2x2, vq_src_p1x2, vq_src_p0x2, vq_src_cx2, vq_src_n0x2, vq_src_n1x2, vq_src_n2x2);

    // horizonal results
    VqTpye vq_vertical_l2 = neon::vext<ELEM_COUNTS - 3>(vq_vertical_results[0], vq_vertical_results[1]);
    VqTpye vq_vertical_l1 = neon::vext<ELEM_COUNTS - 2>(vq_vertical_results[0], vq_vertical_results[1]);
    VqTpye vq_vertical_l0 = neon::vext<ELEM_COUNTS - 1>(vq_vertical_results[0], vq_vertical_results[1]);
    VqTpye vq_vertical_r0 = neon::vext<1>(vq_vertical_results[1], vq_vertical_results[2]);
    VqTpye vq_vertical_r1 = neon::vext<2>(vq_vertical_results[1], vq_vertical_results[2]);
    VqTpye vq_vertical_r2 = neon::vext<3>(vq_vertical_results[1], vq_vertical_results[2]);

    vq_result = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_vertical_l2, vq_vertical_l1, vq_vertical_l0, vq_vertical_results[1],
                                                vq_vertical_r0, vq_vertical_r1, vq_vertical_r2);

    // slide results
    vq_vertical_results[0] = vq_vertical_results[1];
    vq_vertical_results[1] = vq_vertical_results[2];
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 ELEM_COUNTS,
          typename VqTpye = typename neon::QVector<Tp>::VType,
          typename std::enable_if<MorphShape::CROSS == MORPH_SHAPE, Tp>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Morph7x7Vector(VqTpye &vq_src_p2x0, VqTpye &vq_src_p2x1, VqTpye &vq_src_p2x2,
                                          VqTpye &vq_src_p1x0, VqTpye &vq_src_p1x1, VqTpye &vq_src_p1x2,
                                          VqTpye &vq_src_p0x0, VqTpye &vq_src_p0x1, VqTpye &vq_src_p0x2,
                                          VqTpye &vq_src_cx0,  VqTpye &vq_src_cx1,  VqTpye &vq_src_cx2,
                                          VqTpye &vq_src_n0x0, VqTpye &vq_src_n0x1, VqTpye &vq_src_n0x2,
                                          VqTpye &vq_src_n1x0, VqTpye &vq_src_n1x1, VqTpye &vq_src_n1x2,
                                          VqTpye &vq_src_n2x0, VqTpye &vq_src_n2x1, VqTpye &vq_src_n2x2,
                                          VqTpye &vq_result,   VqTpye *vq_vertical_results)
{
    AURA_UNUSED(vq_src_p2x0);
    AURA_UNUSED(vq_src_p1x0);
    AURA_UNUSED(vq_src_p0x0);
    AURA_UNUSED(vq_src_n0x0);
    AURA_UNUSED(vq_src_n1x0);
    AURA_UNUSED(vq_src_n2x0);

    // vertical results
    vq_vertical_results[1] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p2x1, vq_src_p1x1, vq_src_p0x1, vq_src_cx1, vq_src_n0x1, vq_src_n1x1, vq_src_n2x1);

    // horizonal results
    VqTpye vq_src_l2 = neon::vext<ELEM_COUNTS - 3>(vq_src_cx0, vq_src_cx1);
    VqTpye vq_src_l1 = neon::vext<ELEM_COUNTS - 2>(vq_src_cx0, vq_src_cx1);
    VqTpye vq_src_l0 = neon::vext<ELEM_COUNTS - 1>(vq_src_cx0, vq_src_cx1);
    VqTpye vq_src_r0 = neon::vext<1>(vq_src_cx1, vq_src_cx2);
    VqTpye vq_src_r1 = neon::vext<2>(vq_src_cx1, vq_src_cx2);
    VqTpye vq_src_r2 = neon::vext<3>(vq_src_cx1, vq_src_cx2);

    vq_result = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_l2, vq_src_l1, vq_src_l0, vq_vertical_results[1], vq_src_r0, vq_src_r1, vq_src_r2);

    // vertical results
    vq_vertical_results[1] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p2x2, vq_src_p1x2, vq_src_p0x2, vq_src_cx2, vq_src_n0x2, vq_src_n1x2, vq_src_n2x2);

    // slide src
    vq_src_cx0 = vq_src_cx1;
    vq_src_cx1 = vq_src_cx2;
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 ELEM_COUNTS,
          typename VqTpye = typename neon::QVector<Tp>::VType,
          typename std::enable_if<MorphShape::CROSS == MORPH_SHAPE, Tp>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Morph7x7Vector(VqTpye &vq_src_p2x2, VqTpye &vq_src_p1x2, VqTpye &vq_src_p0x2,
                                          VqTpye &vq_src_cx0,  VqTpye &vq_src_cx1,  VqTpye &vq_src_cx2,
                                          VqTpye &vq_src_n0x2, VqTpye &vq_src_n1x2, VqTpye &vq_src_n2x2,
                                          VqTpye &vq_result,   VqTpye *vq_vertical_results)
{
    // horizonal results
    VqTpye vq_src_l2 = neon::vext<ELEM_COUNTS - 3>(vq_src_cx0, vq_src_cx1);
    VqTpye vq_src_l1 = neon::vext<ELEM_COUNTS - 2>(vq_src_cx0, vq_src_cx1);
    VqTpye vq_src_l0 = neon::vext<ELEM_COUNTS - 1>(vq_src_cx0, vq_src_cx1);
    VqTpye vq_src_r0 = neon::vext<1>(vq_src_cx1, vq_src_cx2);
    VqTpye vq_src_r1 = neon::vext<2>(vq_src_cx1, vq_src_cx2);
    VqTpye vq_src_r2 = neon::vext<3>(vq_src_cx1, vq_src_cx2);

    vq_result = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_l2, vq_src_l1, vq_src_l0, vq_vertical_results[1], vq_src_r0, vq_src_r1, vq_src_r2);

    // vertical results
    vq_vertical_results[1] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p2x2, vq_src_p1x2, vq_src_p0x2, vq_src_cx2, vq_src_n0x2, vq_src_n1x2, vq_src_n2x2);

    // slide src
    vq_src_cx0 = vq_src_cx1;
    vq_src_cx1 = vq_src_cx2;
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 ELEM_COUNTS,
          typename VqTpye = typename neon::QVector<Tp>::VType,
          typename std::enable_if<MorphShape::ELLIPSE == MORPH_SHAPE, Tp>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Morph7x7Vector(VqTpye &vq_src_p2x0, VqTpye &vq_src_p2x1, VqTpye &vq_src_p2x2,
                                          VqTpye &vq_src_p1x0, VqTpye &vq_src_p1x1, VqTpye &vq_src_p1x2,
                                          VqTpye &vq_src_p0x0, VqTpye &vq_src_p0x1, VqTpye &vq_src_p0x2,
                                          VqTpye &vq_src_cx0,  VqTpye &vq_src_cx1,  VqTpye &vq_src_cx2,
                                          VqTpye &vq_src_n0x0, VqTpye &vq_src_n0x1, VqTpye &vq_src_n0x2,
                                          VqTpye &vq_src_n1x0, VqTpye &vq_src_n1x1, VqTpye &vq_src_n1x2,
                                          VqTpye &vq_src_n2x0, VqTpye &vq_src_n2x1, VqTpye &vq_src_n2x2,
                                          VqTpye &vq_result,   VqTpye *vq_vertical_results)
{
    AURA_UNUSED(vq_src_p2x0);
    AURA_UNUSED(vq_src_n2x0);

    // vertical results
    vq_vertical_results[0] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p0x0, vq_src_cx0,  vq_src_n0x0);
    vq_vertical_results[1] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p0x1, vq_src_cx1,  vq_src_n0x1);
    vq_vertical_results[2] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p0x2, vq_src_cx2,  vq_src_n0x2);
    vq_vertical_results[3] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p1x0, vq_src_p0x0, vq_src_cx0,  vq_src_n0x0, vq_src_n1x0);
    vq_vertical_results[4] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p1x1, vq_src_p0x1, vq_src_cx1,  vq_src_n0x1, vq_src_n1x1);
    vq_vertical_results[5] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p1x2, vq_src_p0x2, vq_src_cx2,  vq_src_n0x2, vq_src_n1x2);
    vq_vertical_results[6] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p2x1, vq_src_p1x1, vq_src_p0x1, vq_src_cx1,  vq_src_n0x1, vq_src_n1x1, vq_src_n2x1);

    // horizonal results
    VqTpye vq_vertical_l2 = neon::vext<ELEM_COUNTS - 3>(vq_vertical_results[0], vq_vertical_results[1]);
    VqTpye vq_vertical_l1 = neon::vext<ELEM_COUNTS - 2>(vq_vertical_results[3], vq_vertical_results[4]);
    VqTpye vq_vertical_l0 = neon::vext<ELEM_COUNTS - 1>(vq_vertical_results[3], vq_vertical_results[4]);
    VqTpye vq_vertical_r0 = neon::vext<1>(vq_vertical_results[4], vq_vertical_results[5]);
    VqTpye vq_vertical_r1 = neon::vext<2>(vq_vertical_results[4], vq_vertical_results[5]);
    VqTpye vq_vertical_r2 = neon::vext<3>(vq_vertical_results[1], vq_vertical_results[2]);

    vq_result = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_vertical_l2, vq_vertical_l1, vq_vertical_l0, vq_vertical_results[6],
                                                vq_vertical_r0, vq_vertical_r1, vq_vertical_r2);

    // vertical results
    vq_vertical_results[6] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p2x2, vq_src_p1x2, vq_src_p0x2, vq_src_cx2, vq_src_n0x2, vq_src_n1x2, vq_src_n2x2);

    // slide results
    vq_vertical_results[0] = vq_vertical_results[1];
    vq_vertical_results[1] = vq_vertical_results[2];
    vq_vertical_results[3] = vq_vertical_results[4];
    vq_vertical_results[4] = vq_vertical_results[5];
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 ELEM_COUNTS,
          typename VqTpye = typename neon::QVector<Tp>::VType,
          typename std::enable_if<MorphShape::ELLIPSE == MORPH_SHAPE, Tp>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Morph7x7Vector(VqTpye &vq_src_p2x2, VqTpye &vq_src_p1x2, VqTpye &vq_src_p0x2,
                                          VqTpye &vq_src_cx0,  VqTpye &vq_src_cx1,  VqTpye &vq_src_cx2,
                                          VqTpye &vq_src_n0x2, VqTpye &vq_src_n1x2, VqTpye &vq_src_n2x2,
                                          VqTpye &vq_result, VqTpye *vq_vertical_results)
{
    AURA_UNUSED(vq_src_cx0);
    AURA_UNUSED(vq_src_cx1);

    // vertical results
    vq_vertical_results[2] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p0x2, vq_src_cx2, vq_src_n0x2);
    vq_vertical_results[5] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p1x2, vq_src_p0x2, vq_src_cx2, vq_src_n0x2, vq_src_n1x2);

    // horizonal results
    VqTpye vq_vertical_l2 = neon::vext<ELEM_COUNTS - 3>(vq_vertical_results[0], vq_vertical_results[1]);
    VqTpye vq_vertical_l1 = neon::vext<ELEM_COUNTS - 2>(vq_vertical_results[3], vq_vertical_results[4]);
    VqTpye vq_vertical_l0 = neon::vext<ELEM_COUNTS - 1>(vq_vertical_results[3], vq_vertical_results[4]);
    VqTpye vq_vertical_r0 = neon::vext<1>(vq_vertical_results[4], vq_vertical_results[5]);
    VqTpye vq_vertical_r1 = neon::vext<2>(vq_vertical_results[4], vq_vertical_results[5]);
    VqTpye vq_vertical_r2 = neon::vext<3>(vq_vertical_results[1], vq_vertical_results[2]);

    vq_result = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_vertical_l2, vq_vertical_l1, vq_vertical_l0, vq_vertical_results[6],
                                                vq_vertical_r0, vq_vertical_r1, vq_vertical_r2);

    // vertical results
    vq_vertical_results[6] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_p2x2, vq_src_p1x2, vq_src_p0x2, vq_src_cx2, vq_src_n0x2, vq_src_n1x2, vq_src_n2x2);

    // slide results
    vq_vertical_results[0] = vq_vertical_results[1];
    vq_vertical_results[1] = vq_vertical_results[2];
    vq_vertical_results[3] = vq_vertical_results[4];
    vq_vertical_results[4] = vq_vertical_results[5];
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 C>
static DT_VOID Morph7x7Row(const Tp *src_p2, const Tp *src_p1, const Tp *src_p0, const Tp *src_c,
                           const Tp *src_n0, const Tp *src_n1, const Tp *src_n2, Tp *dst, DT_S32 width)
{
    using MVqTpye = typename neon::MQVector<Tp, C>::MVType;
    using VqTpye  = typename neon::QVector<Tp>::VType;

    constexpr DT_S32 ELEM_COUNTS = static_cast<DT_S32>(16 / sizeof(Tp));
    constexpr DT_S32 VOFFSET     = ELEM_COUNTS * C;
    const DT_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVqTpye mvq_src_p2[3], mvq_src_p1[3], mvq_src_p0[3], mvq_src_c[3], mvq_src_n0[3], mvq_src_n1[3], mvq_src_n2[3], mvq_result;
    VqTpye vq_vertical_results[QVECTOR_NUM * C];

    // left
    {
        neon::vload(src_p2,           mvq_src_p2[1]);
        neon::vload(src_p2 + VOFFSET, mvq_src_p2[2]);
        neon::vload(src_p1,           mvq_src_p1[1]);
        neon::vload(src_p1 + VOFFSET, mvq_src_p1[2]);
        neon::vload(src_p0,           mvq_src_p0[1]);
        neon::vload(src_p0 + VOFFSET, mvq_src_p0[2]);
        neon::vload(src_c,            mvq_src_c[1]);
        neon::vload(src_c  + VOFFSET, mvq_src_c[2]);
        neon::vload(src_n0,           mvq_src_n0[1]);
        neon::vload(src_n0 + VOFFSET, mvq_src_n0[2]);
        neon::vload(src_n1,           mvq_src_n1[1]);
        neon::vload(src_n1 + VOFFSET, mvq_src_n1[2]);
        neon::vload(src_n2,           mvq_src_n2[1]);
        neon::vload(src_n2 + VOFFSET, mvq_src_n2[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            neon::vdup(mvq_src_p2[0].val[ch], src_p2[ch]);
            neon::vdup(mvq_src_p1[0].val[ch], src_p1[ch]);
            neon::vdup(mvq_src_p0[0].val[ch], src_p0[ch]);
            neon::vdup(mvq_src_c[0].val[ch],  src_c[ch]);
            neon::vdup(mvq_src_n0[0].val[ch], src_n0[ch]);
            neon::vdup(mvq_src_n1[0].val[ch], src_n1[ch]);
            neon::vdup(mvq_src_n2[0].val[ch], src_n2[ch]);

            Morph7x7Vector<Tp, MORPH_SHAPE, MORPH_TYPE, ELEM_COUNTS>(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                                     mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                                     mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                                     mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                                                     mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                                     mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                                     mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                                                     mvq_result.val[ch], vq_vertical_results + ch * QVECTOR_NUM);
        }
        neon::vstore(dst, mvq_result);
    }

    // middle
    {
        for (DT_S32 x = VOFFSET; x < width_align - VOFFSET; x += VOFFSET)
        {
            neon::vload(src_p2 + x + VOFFSET, mvq_src_p2[2]);
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c  + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);
            neon::vload(src_n2 + x + VOFFSET, mvq_src_n2[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Morph7x7Vector<Tp, MORPH_SHAPE, MORPH_TYPE, ELEM_COUNTS>(mvq_src_p2[2].val[ch], mvq_src_p1[2].val[ch], mvq_src_p0[2].val[ch],
                                                                         mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                                                         mvq_src_n0[2].val[ch], mvq_src_n1[2].val[ch], mvq_src_n2[2].val[ch],
                                                                         mvq_result.val[ch], vq_vertical_results + ch * QVECTOR_NUM);
            }
            neon::vstore(dst + x, mvq_result);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            DT_S32 x = (width - (ELEM_COUNTS << 1)) * C;

            neon::vload(src_p2 + x - VOFFSET, mvq_src_p2[0]);
            neon::vload(src_p2 + x,           mvq_src_p2[1]);
            neon::vload(src_p2 + x + VOFFSET, mvq_src_p2[2]);
            neon::vload(src_p1 + x - VOFFSET, mvq_src_p1[0]);
            neon::vload(src_p1 + x,           mvq_src_p1[1]);
            neon::vload(src_p1 + x + VOFFSET, mvq_src_p1[2]);
            neon::vload(src_p0 + x - VOFFSET, mvq_src_p0[0]);
            neon::vload(src_p0 + x,           mvq_src_p0[1]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c  + x - VOFFSET, mvq_src_c[0]);
            neon::vload(src_c  + x,           mvq_src_c[1]);
            neon::vload(src_c  + x + VOFFSET, mvq_src_c[2]);
            neon::vload(src_n0 + x - VOFFSET, mvq_src_n0[0]);
            neon::vload(src_n0 + x,           mvq_src_n0[1]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);
            neon::vload(src_n1 + x - VOFFSET, mvq_src_n1[0]);
            neon::vload(src_n1 + x,           mvq_src_n1[1]);
            neon::vload(src_n1 + x + VOFFSET, mvq_src_n1[2]);
            neon::vload(src_n2 + x - VOFFSET, mvq_src_n2[0]);
            neon::vload(src_n2 + x,           mvq_src_n2[1]);
            neon::vload(src_n2 + x + VOFFSET, mvq_src_n2[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Morph7x7Vector<Tp, MORPH_SHAPE, MORPH_TYPE, ELEM_COUNTS>(mvq_src_p2[0].val[ch], mvq_src_p2[1].val[ch], mvq_src_p2[2].val[ch],
                                                                         mvq_src_p1[0].val[ch], mvq_src_p1[1].val[ch], mvq_src_p1[2].val[ch],
                                                                         mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                                                         mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                                                         mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                                                         mvq_src_n1[0].val[ch], mvq_src_n1[1].val[ch], mvq_src_n1[2].val[ch],
                                                                         mvq_src_n2[0].val[ch], mvq_src_n2[1].val[ch], mvq_src_n2[2].val[ch],
                                                                         mvq_result.val[ch], vq_vertical_results + ch * QVECTOR_NUM);
            }
            neon::vstore(dst + x, mvq_result);
        }
    }

    // right
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;

        mvq_src_p2[1] = mvq_src_p2[2];
        mvq_src_p1[1] = mvq_src_p1[2];
        mvq_src_p0[1] = mvq_src_p0[2];
        mvq_src_c[1]  = mvq_src_c[2];
        mvq_src_n0[1] = mvq_src_n0[2];
        mvq_src_n1[1] = mvq_src_n1[2];
        mvq_src_n2[1] = mvq_src_n2[2];

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            neon::vdup(mvq_src_p2[2].val[ch], src_p2[last]);
            neon::vdup(mvq_src_p1[2].val[ch], src_p1[last]);
            neon::vdup(mvq_src_p0[2].val[ch], src_p0[last]);
            neon::vdup(mvq_src_c[2].val[ch],  src_c[last]);
            neon::vdup(mvq_src_n0[2].val[ch], src_n0[last]);
            neon::vdup(mvq_src_n1[2].val[ch], src_n1[last]);
            neon::vdup(mvq_src_n2[2].val[ch], src_n2[last]);

            Morph7x7Vector<Tp, MORPH_SHAPE, MORPH_TYPE, ELEM_COUNTS>(mvq_src_p2[2].val[ch], mvq_src_p1[2].val[ch], mvq_src_p0[2].val[ch],
                                                                     mvq_src_c[0].val[ch],  mvq_src_c[1].val[ch],  mvq_src_c[2].val[ch],
                                                                     mvq_src_n0[2].val[ch], mvq_src_n1[2].val[ch], mvq_src_n2[2].val[ch],
                                                                     mvq_result.val[ch], vq_vertical_results + ch * QVECTOR_NUM);
            last++;
        }
        neon::vstore(dst + x, mvq_result);
    }
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 C>
static Status Morph7x7NeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);

    DT_S32 width = dst.GetSizes().m_width;

    DT_S32 y = start_row;

    const Tp *src_p2 = src.Ptr<Tp, BorderType::REPLICATE>(y - 3, DT_NULL);
    const Tp *src_p1 = src.Ptr<Tp, BorderType::REPLICATE>(y - 2, DT_NULL);
    const Tp *src_p0 = src.Ptr<Tp, BorderType::REPLICATE>(y - 1, DT_NULL);
    const Tp *src_c  = src.Ptr<Tp>(y);
    const Tp *src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(y + 1, DT_NULL);
    const Tp *src_n1 = src.Ptr<Tp, BorderType::REPLICATE>(y + 2, DT_NULL);
    const Tp *src_n2 = src.Ptr<Tp, BorderType::REPLICATE>(y + 3, DT_NULL);

    for (; y < end_row; y++)
    {
        Tp *dst_c = dst.Ptr<Tp>(y);
        Morph7x7Row<Tp, MORPH_SHAPE, MORPH_TYPE, C>(src_p2, src_p1, src_p0, src_c, src_n0, src_n1, src_n2, dst_c, width);

        src_p2 = src_p1;
        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src_n2;
        src_n2 = src.Ptr<Tp, BorderType::REPLICATE>(y + 4, DT_NULL);
    }

    return Status::OK;
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE>
static Status Morph7x7NeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    DT_S32 height = dst.GetSizes().m_height;

    switch (dst.GetSizes().m_channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, Morph7x7NeonImpl<Tp, MORPH_SHAPE, MORPH_TYPE, 1>, ctx, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonImpl<Tp, MORPH_SHAPE, 1> run failed!");
            }
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Morph7x7NeonImpl<Tp, MORPH_SHAPE, MORPH_TYPE, 2>, ctx, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonImpl<Tp, MORPH_SHAPE, 2> run failed!");
            }
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Morph7x7NeonImpl<Tp, MORPH_SHAPE, MORPH_TYPE, 3>, ctx, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonImpl<Tp, MORPH_SHAPE, 3> run failed!");
            }
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

template <typename Tp, MorphShape MORPH_SHAPE>
static Status Morph7x7NeonHelper(Context *ctx, const Mat &src, Mat &dst, MorphType type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case MorphType::ERODE:
        {
            ret = Morph7x7NeonHelper<Tp, MORPH_SHAPE, MorphType::ERODE>(ctx, src, dst, target);
            break;
        }

        case MorphType::DILATE:
        {
            ret = Morph7x7NeonHelper<Tp, MORPH_SHAPE, MorphType::DILATE>(ctx, src, dst, target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported morph type");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status Morph7x7NeonHelper(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch(shape)
    {
        case MorphShape::RECT:
        {
            ret = Morph7x7NeonHelper<Tp, MorphShape::RECT>(ctx, src, dst, type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonHelper<Tp, MorphShape::RECT> run failed!");
            }
            break;
        }

        case MorphShape::CROSS:
        {
            ret = Morph7x7NeonHelper<Tp, MorphShape::CROSS>(ctx, src, dst, type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonHelper<Tp, MorphShape::CROSS> run failed!");
            }
            break;
        }

        case MorphShape::ELLIPSE:
        {
            ret = Morph7x7NeonHelper<Tp, MorphShape::ELLIPSE>(ctx, src, dst, type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonHelper<Tp, MorphShape::ELLIPSE> run failed!");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported morph shape");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Morph7x7Neon(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Morph7x7NeonHelper<DT_U8>(ctx, src, dst, type, shape, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonHelper<DT_U8> run failed!");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = Morph7x7NeonHelper<DT_U16>(ctx, src, dst, type, shape, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonHelper<DT_U16> run failed!");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = Morph7x7NeonHelper<DT_S16>(ctx,src, dst, type, shape, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonHelper<DT_S16> run failed!");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = Morph7x7NeonHelper<MI_F16>(ctx, src, dst, type, shape, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonHelper<MI_F16> run failed!");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = Morph7x7NeonHelper<DT_F32>(ctx, src, dst, type, shape, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph7x7NeonHelper<DT_F32> run failed!");
            }
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
