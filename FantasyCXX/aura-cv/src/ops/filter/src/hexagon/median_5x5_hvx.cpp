#include "median_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
AURA_ALWAYS_INLINE DT_VOID Median5x5Core(HVX_Vector &v_src_p1x0, HVX_Vector &v_src_p1x1, HVX_Vector &v_src_p1x2,
                                         HVX_Vector &v_src_p0x0, HVX_Vector &v_src_p0x1, HVX_Vector &v_src_p0x2,
                                         HVX_Vector &v_src_c0x0, HVX_Vector &v_src_c0x1, HVX_Vector &v_src_c0x2,
                                         HVX_Vector &v_src_c1x0, HVX_Vector &v_src_c1x1, HVX_Vector &v_src_c1x2,
                                         HVX_Vector &v_src_n0x0, HVX_Vector &v_src_n0x1, HVX_Vector &v_src_n0x2,
                                         HVX_Vector &v_src_n1x0, HVX_Vector &v_src_n1x1, HVX_Vector &v_src_n1x2,
                                         HVX_Vector &v_result0,  HVX_Vector &v_result1)
{
    HVX_Vector v_src_p1l1 = Q6_V_vlalign_VVR(v_src_p1x1, v_src_p1x0, sizeof(Tp) << 1);
    HVX_Vector v_src_p1l0 = Q6_V_vlalign_VVR(v_src_p1x1, v_src_p1x0, sizeof(Tp));
    HVX_Vector v_src_p1c  = v_src_p1x1;
    HVX_Vector v_src_p1r0 = Q6_V_valign_VVR(v_src_p1x2, v_src_p1x1, sizeof(Tp));
    HVX_Vector v_src_p1r1 = Q6_V_valign_VVR(v_src_p1x2, v_src_p1x1, sizeof(Tp) << 1);

    HVX_Vector v_src_p0l1 = Q6_V_vlalign_VVR(v_src_p0x1, v_src_p0x0, sizeof(Tp) << 1);
    HVX_Vector v_src_p0l0 = Q6_V_vlalign_VVR(v_src_p0x1, v_src_p0x0, sizeof(Tp));
    HVX_Vector v_src_p0c  = v_src_p0x1;
    HVX_Vector v_src_p0r0 = Q6_V_valign_VVR(v_src_p0x2, v_src_p0x1, sizeof(Tp));
    HVX_Vector v_src_p0r1 = Q6_V_valign_VVR(v_src_p0x2, v_src_p0x1, sizeof(Tp) << 1);

    HVX_Vector v_src_c0l1 = Q6_V_vlalign_VVR(v_src_c0x1, v_src_c0x0, sizeof(Tp) << 1);
    HVX_Vector v_src_c0l0 = Q6_V_vlalign_VVR(v_src_c0x1, v_src_c0x0, sizeof(Tp));
    HVX_Vector v_src_c0c  = v_src_c0x1;
    HVX_Vector v_src_c0r0 = Q6_V_valign_VVR(v_src_c0x2, v_src_c0x1, sizeof(Tp));
    HVX_Vector v_src_c0r1 = Q6_V_valign_VVR(v_src_c0x2, v_src_c0x1, sizeof(Tp) << 1);

    HVX_Vector v_src_c1l1 = Q6_V_vlalign_VVR(v_src_c1x1, v_src_c1x0, sizeof(Tp) << 1);
    HVX_Vector v_src_c1l0 = Q6_V_vlalign_VVR(v_src_c1x1, v_src_c1x0, sizeof(Tp));
    HVX_Vector v_src_c1c  = v_src_c1x1;
    HVX_Vector v_src_c1r0 = Q6_V_valign_VVR(v_src_c1x2, v_src_c1x1, sizeof(Tp));
    HVX_Vector v_src_c1r1 = Q6_V_valign_VVR(v_src_c1x2, v_src_c1x1, sizeof(Tp) << 1);

    HVX_Vector v_src_n0l1 = Q6_V_vlalign_VVR(v_src_n0x1, v_src_n0x0, sizeof(Tp) << 1);
    HVX_Vector v_src_n0l0 = Q6_V_vlalign_VVR(v_src_n0x1, v_src_n0x0, sizeof(Tp));
    HVX_Vector v_src_n0c  = v_src_n0x1;
    HVX_Vector v_src_n0r0 = Q6_V_valign_VVR(v_src_n0x2, v_src_n0x1, sizeof(Tp));
    HVX_Vector v_src_n0r1 = Q6_V_valign_VVR(v_src_n0x2, v_src_n0x1, sizeof(Tp) << 1);

    HVX_Vector v_src_n1l1 = Q6_V_vlalign_VVR(v_src_n1x1, v_src_n1x0, sizeof(Tp) << 1);
    HVX_Vector v_src_n1l0 = Q6_V_vlalign_VVR(v_src_n1x1, v_src_n1x0, sizeof(Tp));
    HVX_Vector v_src_n1c  = v_src_n1x1;
    HVX_Vector v_src_n1r0 = Q6_V_valign_VVR(v_src_n1x2, v_src_n1x1, sizeof(Tp));
    HVX_Vector v_src_n1r1 = Q6_V_valign_VVR(v_src_n1x2, v_src_n1x1, sizeof(Tp) << 1);

    // step1 sort main body row1-row5
    // row0 row1
    VectorMinMax<Tp>(v_src_p0l1, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p0l0, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_c0c);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0r1);

    // row2 row3
    VectorMinMax<Tp>(v_src_c1l1, v_src_n0l1);
    VectorMinMax<Tp>(v_src_c1l0, v_src_n0l0);
    VectorMinMax<Tp>(v_src_c1c,  v_src_n0c);
    VectorMinMax<Tp>(v_src_c1r0, v_src_n0r0);
    VectorMinMax<Tp>(v_src_c1r1, v_src_n0r1);

    // row0 row2
    VectorMinMax<Tp>(v_src_p0l1, v_src_c1l1);
    VectorMinMax<Tp>(v_src_p0l0, v_src_c1l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_c1c);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c1r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c1r1);

    // row1 row3
    VectorMinMax<Tp>(v_src_c0l1, v_src_n0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_n0l0);
    VectorMinMax<Tp>(v_src_c0c,  v_src_n0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_n0r0);
    VectorMinMax<Tp>(v_src_c0r1, v_src_n0r1);

    // row1 row2
    VectorMinMax<Tp>(v_src_c0l1, v_src_c1l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c1l0);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c1c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_c1r0);
    VectorMinMax<Tp>(v_src_c0r1, v_src_c1r1);

    // step2 get middle 10 from 4x4 box
    HVX_Vector v_border_result;
    HVX_Vector v_reuse[6];

    //2.1sort hori row1
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r0, v_src_p0r1);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r1);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);

    //2.2sort hori row2
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_c0r1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r0);

    //2.3sort hori row3
    VectorMinMax<Tp>(v_src_c1l0, v_src_c1c);
    VectorMinMax<Tp>(v_src_c1r0, v_src_c1r1);
    VectorMinMax<Tp>(v_src_c1l0, v_src_c1r0);
    VectorMinMax<Tp>(v_src_c1c,  v_src_c1r1);
    VectorMinMax<Tp>(v_src_c1c,  v_src_c1r0);

    //2.4sort hori row4
    VectorMinMax<Tp>(v_src_n0l0, v_src_n0c);
    VectorMinMax<Tp>(v_src_n0r0, v_src_n0r1);
    VectorMinMax<Tp>(v_src_n0l0, v_src_n0r0);
    VectorMinMax<Tp>(v_src_n0c,  v_src_n0r1);
    VectorMinMax<Tp>(v_src_n0c,  v_src_n0r0);

    //2.5sort diagonal
    VectorMinMax<Tp>(v_src_c0l0, v_src_p0c);
    VectorMinMax<Tp>(v_src_c1l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_c1l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_n0l0, v_src_c1c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_p0r1);
    VectorMinMax<Tp>(v_src_n0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_c1c,  v_src_p0r1);
    VectorMinMax<Tp>(v_src_c1c,  v_src_c0r0);
    VectorMinMax<Tp>(v_src_n0c,  v_src_c1r0);
    VectorMinMax<Tp>(v_src_c1r0, v_src_c0r1);
    VectorMinMax<Tp>(v_src_n0c,  v_src_c1r0);
    VectorMinMax<Tp>(v_src_n0r0, v_src_c1r1);

    //find a min and a max
    VectorMinMax<Tp>(v_src_p0c,  v_src_c1l0);
    VectorMinMax<Tp>(v_src_c0r1, v_src_n0r0);
    VectorMinMax<Tp>(v_src_c1l0, v_src_n0l0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0r1);

    //sort 10 nums
    VectorMinMax<Tp>(v_src_p0r1, v_src_c1r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_n0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_p0r1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_n0l0);
    VectorMinMax<Tp>(v_src_p0r0, v_src_n0l0);
    VectorMinMax<Tp>(v_src_n0l0, v_src_c1c);
    VectorMinMax<Tp>(v_src_c1c,  v_src_c0r0);

    // step3 get middle most6 from14
    VectorMinMax<Tp>(v_src_p0l1, v_src_c1l0);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c1l0);
    VectorMinMax<Tp>(v_src_c1r0, v_src_c1l1);
    VectorMinMax<Tp>(v_src_c0r1, v_src_n0l1);
    VectorMinMax<Tp>(v_src_c0r1, v_src_c1l1);
    VectorMinMax<Tp>(v_src_c1l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_c1r0, v_src_c0r1);
    VectorMinMax<Tp>(v_src_c1l0, v_src_c1r0);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c1r0);
    VectorMinMax<Tp>(v_src_c1l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_c0c,  v_src_n0l0);
    VectorMinMax<Tp>(v_src_c0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c1r0);
    VectorMinMax<Tp>(v_src_n0c,  v_src_c0r1);
    VectorMinMax<Tp>(v_src_n0c,  v_src_c1r0);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c1c);
    VectorMinMax<Tp>(v_src_p0r0, v_src_p0r1);
    VectorMinMax<Tp>(v_src_c1c,  v_src_p0r1);
    VectorMinMax<Tp>(v_src_n0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_n0l0, v_src_n0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_n0c);
    VectorMinMax<Tp>(v_src_n0l0, v_src_p0r1);

    v_reuse[0] = v_src_p0r0;
    v_reuse[1] = v_src_c1c;
    v_reuse[2] = v_src_n0l0;
    v_reuse[3] = v_src_p0r1;
    v_reuse[4] = v_src_c0r0;
    v_reuse[5] = v_src_n0c;

    // step4 get middle most 2 from10
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p1r0);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p1c,  v_src_c1c);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_c0r0, v_src_p1r0);
    VectorMinMax<Tp>(v_src_n0c,  v_src_p1r1);
    VectorMinMax<Tp>(v_src_n0c,  v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0r0, v_src_n0l0);
    VectorMinMax<Tp>(v_src_c1c,  v_src_p0r1);
    VectorMinMax<Tp>(v_src_c1c,  v_src_n0l0);
    VectorMinMax<Tp>(v_src_n0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_n0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0r0);
    VectorMinMax<Tp>(v_src_p0r0, v_src_n0l0);
    VectorMinMax<Tp>(v_src_c1c,  v_src_p0r1);
    VectorMinMax<Tp>(v_src_c1c,  v_src_n0l0);

    v_result0       = v_src_n0l0;
    v_border_result = v_src_p0r1;

    VectorMinMax<Tp>(v_src_p1l1, v_result0);
    VectorMinMax<Tp>(v_result0,  v_border_result);
    VectorMinMax<Tp>(v_src_n1l0, v_src_n1c);
    VectorMinMax<Tp>(v_src_n1r0, v_src_n1r1);
    VectorMinMax<Tp>(v_src_n1l0, v_src_n1r0);
    VectorMinMax<Tp>(v_src_n1c,  v_src_n1r1);
    VectorMinMax<Tp>(v_src_n1c,  v_src_n1r0);
    VectorMinMax<Tp>(v_src_n1l0, v_reuse[0]);
    VectorMinMax<Tp>(v_src_n1c,  v_reuse[1]);
    VectorMinMax<Tp>(v_src_n1c,  v_reuse[0]);
    VectorMinMax<Tp>(v_reuse[4], v_src_n1r0);
    VectorMinMax<Tp>(v_reuse[5], v_src_n1r1);
    VectorMinMax<Tp>(v_reuse[5], v_src_n1r0);
    VectorMinMax<Tp>(v_reuse[0], v_reuse[2]);
    VectorMinMax<Tp>(v_reuse[1], v_reuse[3]);
    VectorMinMax<Tp>(v_reuse[1], v_reuse[2]);
    VectorMinMax<Tp>(v_reuse[2], v_reuse[4]);
    VectorMinMax<Tp>(v_reuse[3], v_reuse[5]);
    VectorMinMax<Tp>(v_reuse[3], v_reuse[4]);
    VectorMinMax<Tp>(v_reuse[0], v_reuse[2]);
    VectorMinMax<Tp>(v_reuse[1], v_reuse[3]);
    VectorMinMax<Tp>(v_reuse[1], v_reuse[2]);

    v_result1       = v_reuse[2];
    v_border_result = v_reuse[3];

    VectorMinMax<Tp>(v_src_n1l1, v_result1);
    VectorMinMax<Tp>(v_result1, v_border_result);
}

template <typename Tp, DT_S32 C>
static DT_VOID Median5x5TwoRow(const Tp *src_p1, const Tp *src_p0, const Tp *src_c0, const Tp *src_c1, const Tp *src_n0,
                               const Tp *src_n1, Tp *dst_c0, Tp *dst_c1, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    DT_S32 back_offset = width - elem_counts;

    MVType mv_src_p1x0, mv_src_p1x1, mv_src_p1x2;
    MVType mv_src_p0x0, mv_src_p0x1, mv_src_p0x2;
    MVType mv_src_c0x0, mv_src_c0x1, mv_src_c0x2;
    MVType mv_src_c1x0, mv_src_c1x1, mv_src_c1x2;
    MVType mv_src_n0x0, mv_src_n0x1, mv_src_n0x2;
    MVType mv_src_n1x0, mv_src_n1x1, mv_src_n1x2;

    MVType mv_result0, mv_result1;

    // left
    {
        vload(src_p1, mv_src_p1x1);
        vload(src_p0, mv_src_p0x1);
        vload(src_c0, mv_src_c0x1);
        vload(src_c1, mv_src_c1x1);
        vload(src_n0, mv_src_n0x1);
        vload(src_n1, mv_src_n1x1);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p1x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_p1x1.val[ch], src_p1[ch], 2);
            mv_src_p0x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_p0x1.val[ch], src_p0[ch], 2);
            mv_src_c0x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_c0x1.val[ch], src_c0[ch], 2);
            mv_src_c1x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_c1x1.val[ch], src_c1[ch], 2);
            mv_src_n0x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_n0x1.val[ch], src_n0[ch], 2);
            mv_src_n1x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_n1x1.val[ch], src_n1[ch], 2);
        }
    }

    // middle
    for (DT_S32 x = elem_counts; x <= back_offset; x += elem_counts)
    {
        vload(src_p1 + C * x, mv_src_p1x2);
        vload(src_p0 + C * x, mv_src_p0x2);
        vload(src_c0 + C * x, mv_src_c0x2);
        vload(src_c1 + C * x, mv_src_c1x2);
        vload(src_n0 + C * x, mv_src_n0x2);
        vload(src_n1 + C * x, mv_src_n1x2);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            Median5x5Core<Tp>(mv_src_p1x0.val[ch], mv_src_p1x1.val[ch], mv_src_p1x2.val[ch],
                              mv_src_p0x0.val[ch], mv_src_p0x1.val[ch], mv_src_p0x2.val[ch],
                              mv_src_c0x0.val[ch], mv_src_c0x1.val[ch], mv_src_c0x2.val[ch],
                              mv_src_c1x0.val[ch], mv_src_c1x1.val[ch], mv_src_c1x2.val[ch],
                              mv_src_n0x0.val[ch], mv_src_n0x1.val[ch], mv_src_n0x2.val[ch],
                              mv_src_n1x0.val[ch], mv_src_n1x1.val[ch], mv_src_n1x2.val[ch],
                              mv_result0.val[ch],  mv_result1.val[ch]);
        }
        vstore(dst_c0 + C * (x - elem_counts), mv_result0);
        vstore(dst_c1 + C * (x - elem_counts), mv_result1);

        mv_src_p1x0 = mv_src_p1x1;
        mv_src_p0x0 = mv_src_p0x1;
        mv_src_c0x0 = mv_src_c0x1;
        mv_src_c1x0 = mv_src_c1x1;
        mv_src_n0x0 = mv_src_n0x1;
        mv_src_n1x0 = mv_src_n1x1;
        mv_src_p1x1 = mv_src_p1x2;
        mv_src_p0x1 = mv_src_p0x2;
        mv_src_c0x1 = mv_src_c0x2;
        mv_src_c1x1 = mv_src_c1x2;
        mv_src_n0x1 = mv_src_n0x2;
        mv_src_n1x1 = mv_src_n1x2;
    }

    // right
    {
        DT_S32 last = (width - 1) * C;
        DT_S32 rest = width % elem_counts;
        MVType mv_last_result0, mv_last_result1;

        vload(src_p1 + C * back_offset, mv_src_p1x2);
        vload(src_p0 + C * back_offset, mv_src_p0x2);
        vload(src_c0 + C * back_offset, mv_src_c0x2);
        vload(src_c1 + C * back_offset, mv_src_c1x2);
        vload(src_n0 + C * back_offset, mv_src_n0x2);
        vload(src_n1 + C * back_offset, mv_src_n1x2);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_p1x2.val[ch], src_p1[last + ch], 1);
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_p0x2.val[ch], src_p0[last + ch], 1);
            HVX_Vector v_border_c0 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_c0x2.val[ch], src_c0[last + ch], 1);
            HVX_Vector v_border_c1 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_c1x2.val[ch], src_c1[last + ch], 1);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_n0x2.val[ch], src_n0[last + ch], 1);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_n1x2.val[ch], src_n1[last + ch], 1);

            HVX_Vector v_src_p1r = Q6_V_vlalign_VVR(v_border_p1, mv_src_p1x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_p0r = Q6_V_vlalign_VVR(v_border_p0, mv_src_p0x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_c0r = Q6_V_vlalign_VVR(v_border_c0, mv_src_c0x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_c1r = Q6_V_vlalign_VVR(v_border_c1, mv_src_c1x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_n0r = Q6_V_vlalign_VVR(v_border_n0, mv_src_n0x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_n1r = Q6_V_vlalign_VVR(v_border_n1, mv_src_n1x2.val[ch], rest * sizeof(Tp));

            Median5x5Core<Tp>(mv_src_p1x0.val[ch], mv_src_p1x1.val[ch], v_src_p1r,
                              mv_src_p0x0.val[ch], mv_src_p0x1.val[ch], v_src_p0r,
                              mv_src_c0x0.val[ch], mv_src_c0x1.val[ch], v_src_c0r,
                              mv_src_c1x0.val[ch], mv_src_c1x1.val[ch], v_src_c1r,
                              mv_src_n0x0.val[ch], mv_src_n0x1.val[ch], v_src_n0r,
                              mv_src_n1x0.val[ch], mv_src_n1x1.val[ch], v_src_n1r,
                              mv_result0.val[ch],  mv_result1.val[ch]);

            HVX_Vector v_src_p1l = Q6_V_valign_VVR(mv_src_p1x1.val[ch], mv_src_p1x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_p0l = Q6_V_valign_VVR(mv_src_p0x1.val[ch], mv_src_p0x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_c0l = Q6_V_valign_VVR(mv_src_c0x1.val[ch], mv_src_c0x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_c1l = Q6_V_valign_VVR(mv_src_c1x1.val[ch], mv_src_c1x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_n0l = Q6_V_valign_VVR(mv_src_n0x1.val[ch], mv_src_n0x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_n1l = Q6_V_valign_VVR(mv_src_n1x1.val[ch], mv_src_n1x0.val[ch], rest * sizeof(Tp));

            Median5x5Core<Tp>(v_src_p1l, mv_src_p1x2.val[ch], v_border_p1,
                              v_src_p0l, mv_src_p0x2.val[ch], v_border_p0,
                              v_src_c0l, mv_src_c0x2.val[ch], v_border_c0,
                              v_src_c1l, mv_src_c1x2.val[ch], v_border_c1,
                              v_src_n0l, mv_src_n0x2.val[ch], v_border_n0,
                              v_src_n1l, mv_src_n1x2.val[ch], v_border_n1,
                              mv_last_result0.val[ch], mv_last_result1.val[ch]);
        }
        vstore(dst_c0 + C * (back_offset - rest), mv_result0);
        vstore(dst_c0 + C * back_offset,          mv_last_result0);
        vstore(dst_c1 + C * (back_offset - rest), mv_result1);
        vstore(dst_c1 + C * back_offset,          mv_last_result1);
    }
}

template <typename Tp, DT_S32 C>
static Status Median5x5HvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width   = src.GetSizes().m_width;
    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    const Tp *src_p1 = src.Ptr<Tp, BorderType::REPLICATE>(start_row - 2);
    const Tp *src_p0 = src.Ptr<Tp, BorderType::REPLICATE>(start_row - 1);
    const Tp *src_c0 = src.Ptr<Tp>(start_row);
    const Tp *src_c1 = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 1);
    const Tp *src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 2);
    const Tp *src_n1 = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 3);

    DT_U64 L2fetch_param = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 2, 0);
    DT_S32 y;
    for (y = start_row; y < end_row - 1; y += 2)
    {
        if (y + 3 < height)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<Tp>(y + 3)), L2fetch_param);
        }

        Tp *dst_c0 = dst.Ptr<Tp>(y);
        Tp *dst_c1 = dst.Ptr<Tp, BorderType::REPLICATE>(y + 1);
        Median5x5TwoRow<Tp, C>(src_p1, src_p0, src_c0, src_c1, src_n0, src_n1, dst_c0, dst_c1, width);

        src_p1 = src_c0;
        src_p0 = src_c1;
        src_c0 = src_n0;
        src_c1 = src_n1;
        src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(y + 4);
        src_n1 = src.Ptr<Tp, BorderType::REPLICATE>(y + 5);
    }

    if (y == end_row - 1)
    {
        src_p1 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 4);
        src_p0 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 3);
        src_c0 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 2);
        src_c1 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 1);
        src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(end_row);
        src_n1 = src.Ptr<Tp, BorderType::REPLICATE>(end_row + 1);

        Tp *dst_c0 = dst.Ptr<Tp, BorderType::REPLICATE>(end_row - 2);
        Tp *dst_c1 = dst.Ptr<Tp, BorderType::REPLICATE>(end_row - 1);

        Median5x5TwoRow<Tp, C>(src_p1, src_p0, src_c0, src_c1, src_n0, src_n1, dst_c0, dst_c1, width);
    }

    return Status::OK;
}

template<typename Tp>
static Status Median5x5HvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Median5x5HvxImpl<Tp, 1>, src, dst);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Median5x5HvxImpl<Tp, 2>, src, dst);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Median5x5HvxImpl<Tp, 3>, src, dst);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Median5x5Hvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Median5x5HvxHelper<DT_U8>(ctx, src, dst);
            break;
        }

        case ElemType::S8:
        {
            ret = Median5x5HvxHelper<DT_S8>(ctx, src, dst);
            break;
        }

        case ElemType::U16:
        {
            ret = Median5x5HvxHelper<DT_U16>(ctx, src, dst);
            break;
        }

        case ElemType::S16:
        {
            ret = Median5x5HvxHelper<DT_S16>(ctx, src, dst);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported data type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura