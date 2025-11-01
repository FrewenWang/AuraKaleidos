#include "median_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
AURA_ALWAYS_INLINE AURA_VOID Median7x7Core(HVX_Vector &v_src_p2x0, HVX_Vector &v_src_p2x1, HVX_Vector &v_src_p2x2,
                                         HVX_Vector &v_src_p1x0, HVX_Vector &v_src_p1x1, HVX_Vector &v_src_p1x2,
                                         HVX_Vector &v_src_p0x0, HVX_Vector &v_src_p0x1, HVX_Vector &v_src_p0x2,
                                         HVX_Vector &v_src_c0x0, HVX_Vector &v_src_c0x1, HVX_Vector &v_src_c0x2,
                                         HVX_Vector &v_src_c1x0, HVX_Vector &v_src_c1x1, HVX_Vector &v_src_c1x2,
                                         HVX_Vector &v_src_n0x0, HVX_Vector &v_src_n0x1, HVX_Vector &v_src_n0x2,
                                         HVX_Vector &v_src_n1x0, HVX_Vector &v_src_n1x1, HVX_Vector &v_src_n1x2,
                                         HVX_Vector &v_src_n2x0, HVX_Vector &v_src_n2x1, HVX_Vector &v_src_n2x2,
                                         HVX_Vector &v_result0, HVX_Vector &v_result1)
{
    HVX_Vector v_src_p2l2 = Q6_V_vlalign_VVR(v_src_p2x1, v_src_p2x0, sizeof(Tp) * 3);
    HVX_Vector v_src_p2l1 = Q6_V_vlalign_VVR(v_src_p2x1, v_src_p2x0, sizeof(Tp) << 1);
    HVX_Vector v_src_p2l0 = Q6_V_vlalign_VVR(v_src_p2x1, v_src_p2x0, sizeof(Tp));
    HVX_Vector v_src_p2c  = v_src_p2x1;
    HVX_Vector v_src_p2r0 = Q6_V_valign_VVR(v_src_p2x2, v_src_p2x1, sizeof(Tp));
    HVX_Vector v_src_p2r1 = Q6_V_valign_VVR(v_src_p2x2, v_src_p2x1, sizeof(Tp) << 1);
    HVX_Vector v_src_p2r2 = Q6_V_valign_VVR(v_src_p2x2, v_src_p2x1, sizeof(Tp) * 3);

    HVX_Vector v_src_p1l2 = Q6_V_vlalign_VVR(v_src_p1x1, v_src_p1x0, sizeof(Tp) * 3);
    HVX_Vector v_src_p1l1 = Q6_V_vlalign_VVR(v_src_p1x1, v_src_p1x0, sizeof(Tp) << 1);
    HVX_Vector v_src_p1l0 = Q6_V_vlalign_VVR(v_src_p1x1, v_src_p1x0, sizeof(Tp));
    HVX_Vector v_src_p1c  = v_src_p1x1;
    HVX_Vector v_src_p1r0 = Q6_V_valign_VVR(v_src_p1x2, v_src_p1x1, sizeof(Tp));
    HVX_Vector v_src_p1r1 = Q6_V_valign_VVR(v_src_p1x2, v_src_p1x1, sizeof(Tp) << 1);
    HVX_Vector v_src_p1r2 = Q6_V_valign_VVR(v_src_p1x2, v_src_p1x1, sizeof(Tp) * 3);

    HVX_Vector v_src_p0l2 = Q6_V_vlalign_VVR(v_src_p0x1, v_src_p0x0, sizeof(Tp) * 3);
    HVX_Vector v_src_p0l1 = Q6_V_vlalign_VVR(v_src_p0x1, v_src_p0x0, sizeof(Tp) << 1);
    HVX_Vector v_src_p0l0 = Q6_V_vlalign_VVR(v_src_p0x1, v_src_p0x0, sizeof(Tp));
    HVX_Vector v_src_p0c  = v_src_p0x1;
    HVX_Vector v_src_p0r0 = Q6_V_valign_VVR(v_src_p0x2, v_src_p0x1, sizeof(Tp));
    HVX_Vector v_src_p0r1 = Q6_V_valign_VVR(v_src_p0x2, v_src_p0x1, sizeof(Tp) << 1);
    HVX_Vector v_src_p0r2 = Q6_V_valign_VVR(v_src_p0x2, v_src_p0x1, sizeof(Tp) * 3);

    HVX_Vector v_src_c0l2 = Q6_V_vlalign_VVR(v_src_c0x1, v_src_c0x0, sizeof(Tp) * 3);
    HVX_Vector v_src_c0l1 = Q6_V_vlalign_VVR(v_src_c0x1, v_src_c0x0, sizeof(Tp) << 1);
    HVX_Vector v_src_c0l0 = Q6_V_vlalign_VVR(v_src_c0x1, v_src_c0x0, sizeof(Tp));
    HVX_Vector v_src_c0c  = v_src_c0x1;
    HVX_Vector v_src_c0r0 = Q6_V_valign_VVR(v_src_c0x2, v_src_c0x1, sizeof(Tp));
    HVX_Vector v_src_c0r1 = Q6_V_valign_VVR(v_src_c0x2, v_src_c0x1, sizeof(Tp) << 1);
    HVX_Vector v_src_c0r2 = Q6_V_valign_VVR(v_src_c0x2, v_src_c0x1, sizeof(Tp) * 3);

    HVX_Vector v_src_c1l2 = Q6_V_vlalign_VVR(v_src_c1x1, v_src_c1x0, sizeof(Tp) * 3);
    HVX_Vector v_src_c1l1 = Q6_V_vlalign_VVR(v_src_c1x1, v_src_c1x0, sizeof(Tp) << 1);
    HVX_Vector v_src_c1l0 = Q6_V_vlalign_VVR(v_src_c1x1, v_src_c1x0, sizeof(Tp));
    HVX_Vector v_src_c1c  = v_src_c1x1;
    HVX_Vector v_src_c1r0 = Q6_V_valign_VVR(v_src_c1x2, v_src_c1x1, sizeof(Tp));
    HVX_Vector v_src_c1r1 = Q6_V_valign_VVR(v_src_c1x2, v_src_c1x1, sizeof(Tp) << 1);
    HVX_Vector v_src_c1r2 = Q6_V_valign_VVR(v_src_c1x2, v_src_c1x1, sizeof(Tp) * 3);

    HVX_Vector v_src_n0l2 = Q6_V_vlalign_VVR(v_src_n0x1, v_src_n0x0, sizeof(Tp) * 3);
    HVX_Vector v_src_n0l1 = Q6_V_vlalign_VVR(v_src_n0x1, v_src_n0x0, sizeof(Tp) << 1);
    HVX_Vector v_src_n0l0 = Q6_V_vlalign_VVR(v_src_n0x1, v_src_n0x0, sizeof(Tp));
    HVX_Vector v_src_n0c  = v_src_n0x1;
    HVX_Vector v_src_n0r0 = Q6_V_valign_VVR(v_src_n0x2, v_src_n0x1, sizeof(Tp));
    HVX_Vector v_src_n0r1 = Q6_V_valign_VVR(v_src_n0x2, v_src_n0x1, sizeof(Tp) << 1);
    HVX_Vector v_src_n0r2 = Q6_V_valign_VVR(v_src_n0x2, v_src_n0x1, sizeof(Tp) * 3);

    HVX_Vector v_src_n1l2 = Q6_V_vlalign_VVR(v_src_n1x1, v_src_n1x0, sizeof(Tp) * 3);
    HVX_Vector v_src_n1l1 = Q6_V_vlalign_VVR(v_src_n1x1, v_src_n1x0, sizeof(Tp) << 1);
    HVX_Vector v_src_n1l0 = Q6_V_vlalign_VVR(v_src_n1x1, v_src_n1x0, sizeof(Tp));
    HVX_Vector v_src_n1c  = v_src_n1x1;
    HVX_Vector v_src_n1r0 = Q6_V_valign_VVR(v_src_n1x2, v_src_n1x1, sizeof(Tp));
    HVX_Vector v_src_n1r1 = Q6_V_valign_VVR(v_src_n1x2, v_src_n1x1, sizeof(Tp) << 1);
    HVX_Vector v_src_n1r2 = Q6_V_valign_VVR(v_src_n1x2, v_src_n1x1, sizeof(Tp) * 3);

    HVX_Vector v_src_n2l2 = Q6_V_vlalign_VVR(v_src_n2x1, v_src_n2x0, sizeof(Tp) * 3);
    HVX_Vector v_src_n2l1 = Q6_V_vlalign_VVR(v_src_n2x1, v_src_n2x0, sizeof(Tp) << 1);
    HVX_Vector v_src_n2l0 = Q6_V_vlalign_VVR(v_src_n2x1, v_src_n2x0, sizeof(Tp));
    HVX_Vector v_src_n2c  = v_src_n2x1;
    HVX_Vector v_src_n2r0 = Q6_V_valign_VVR(v_src_n2x2, v_src_n2x1, sizeof(Tp));
    HVX_Vector v_src_n2r1 = Q6_V_valign_VVR(v_src_n2x2, v_src_n2x1, sizeof(Tp) << 1);
    HVX_Vector v_src_n2r2 = Q6_V_valign_VVR(v_src_n2x2, v_src_n2x1, sizeof(Tp) * 3);

    // step1 Get minmax  from 26   delete 7 32
    VectorMinMax<Tp>(v_src_p1l2, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_c0r1);
    VectorMinMax<Tp>(v_src_c0r2, v_src_c1l2);
    VectorMinMax<Tp>(v_src_c1l1, v_src_c1l0);
    VectorMinMax<Tp>(v_src_c1c,  v_src_c1r0);
    VectorMinMax<Tp>(v_src_p1l2, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_c0r2, v_src_c1l1);
    VectorMinMax<Tp>(v_src_p1l2, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r2);
    VectorMinMax<Tp>(v_src_p1l2, v_src_p0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c1c);
    VectorMinMax<Tp>(v_src_p1l2, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r1);
    VectorMinMax<Tp>(v_src_c1l2, v_src_c1l0);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0r1, v_src_c1l0);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c1l0, v_src_c1r0);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c1r0);

    //Get minmax  from 27   delete 33 31
    VectorMinMax<Tp>(v_src_c1r1, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_c0r1);
    VectorMinMax<Tp>(v_src_c0r2, v_src_c1l2);
    VectorMinMax<Tp>(v_src_c1l1, v_src_c1l0);
    VectorMinMax<Tp>(v_src_c1r1, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_c0r2, v_src_c1l1);
    VectorMinMax<Tp>(v_src_c1r1, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r2);
    VectorMinMax<Tp>(v_src_c1r1, v_src_p0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c1c);
    VectorMinMax<Tp>(v_src_c1r1, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r1);
    VectorMinMax<Tp>(v_src_c1l2, v_src_c1l0);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0r1, v_src_c1l0);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c1l0, v_src_c1c);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c1c);

    //Get minmax  from 26   delete 34 30
    VectorMinMax<Tp>(v_src_c1r2, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_c0r1);
    VectorMinMax<Tp>(v_src_c0r2, v_src_c1l2);
    VectorMinMax<Tp>(v_src_c1l1, v_src_c1l0);
    VectorMinMax<Tp>(v_src_c1r2, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_c0r2, v_src_c1l1);
    VectorMinMax<Tp>(v_src_c1r2, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r2);
    VectorMinMax<Tp>(v_src_c1r2, v_src_p0l1);
    VectorMinMax<Tp>(v_src_c1r2, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r1);
    VectorMinMax<Tp>(v_src_c1l2, v_src_c1l0);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0r1, v_src_c1l0);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c1l0);

    //Get minmax  from 25   delete 35 29
    VectorMinMax<Tp>(v_src_n0l2, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_c0r1);
    VectorMinMax<Tp>(v_src_c0r2, v_src_c1l2);
    VectorMinMax<Tp>(v_src_n0l2, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_c0r2, v_src_c1l1);
    VectorMinMax<Tp>(v_src_n0l2, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r2);
    VectorMinMax<Tp>(v_src_n0l2, v_src_p0l1);
    VectorMinMax<Tp>(v_src_n0l2, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r1);
    VectorMinMax<Tp>(v_src_c1l2, v_src_c1l1);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0r1, v_src_c1l1);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c1l1);

    //Get minmax  from 24   delete 36 28
    VectorMinMax<Tp>(v_src_n0l1, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_c0r1);
    VectorMinMax<Tp>(v_src_c0r2, v_src_c1l2);
    VectorMinMax<Tp>(v_src_n0l1, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_n0l1, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r2);
    VectorMinMax<Tp>(v_src_n0l1, v_src_p0l1);
    VectorMinMax<Tp>(v_src_n0l1, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r1);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0r1, v_src_c1l2);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c1l2);

    //Get minmax  from 23   delete 37 27
    VectorMinMax<Tp>(v_src_n0l0, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_c0r1);
    VectorMinMax<Tp>(v_src_n0l0, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_n0l0, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r2);
    VectorMinMax<Tp>(v_src_n0l0, v_src_p0l1);
    VectorMinMax<Tp>(v_src_n0l0, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r1);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0r1, v_src_c0r2);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c0r2);

    //Get minmax  from 22   delete 38 26
    VectorMinMax<Tp>(v_src_n0c,  v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r0, v_src_c0r1);
    VectorMinMax<Tp>(v_src_n0c,  v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_n0c,  v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_n0c,  v_src_p0l1);
    VectorMinMax<Tp>(v_src_n0c,  v_src_c0l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r1);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c0r1);

    //Get minmax  from 21   delete 39 25
    VectorMinMax<Tp>(v_src_n0r0, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_n0r0, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0r0);
    VectorMinMax<Tp>(v_src_n0r0, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_n0r0, v_src_p0l1);
    VectorMinMax<Tp>(v_src_n0r0, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0c,  v_src_c0r0);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c0r0);

   //Get minmax  from 20   delete 40 24
    VectorMinMax<Tp>(v_src_n0r1, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l0, v_src_c0c);
    VectorMinMax<Tp>(v_src_n0r1, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_n0r1, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_n0r1, v_src_p0l1);
    VectorMinMax<Tp>(v_src_n0r1, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c0c);

    //Get minmax  from 19   delete 41 23
    VectorMinMax<Tp>(v_src_n0r2, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_n0r2, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_n0r2, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_n0r2, v_src_p0l1);
    VectorMinMax<Tp>(v_src_n0r2, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c0l0);

    // step1 Get minmax  from 18   delete 42 22
    VectorMinMax<Tp>(v_src_n1l2, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_n1l2, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_n1l2, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_n1l2, v_src_p0l1);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l1);

    // step1 Get minmax  from 17   delete 43 21
    VectorMinMax<Tp>(v_src_n1l1, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_n1l1, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_n1l1, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_n1l1, v_src_p0l1);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l2);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l2);
    VectorMinMax<Tp>(v_src_p0l2, v_src_c0l2);

    // step1 Get minmax  from 16   delete 44 20
    VectorMinMax<Tp>(v_src_n1l0, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_n1l0, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_n1l0, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_n1l0, v_src_p0l1);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_p0r2);
    VectorMinMax<Tp>(v_src_p0l2, v_src_p0r2);

    // step1 Get minmax  from 15   delete 45 19
    VectorMinMax<Tp>(v_src_n1c, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_n1c,  v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_n1c,  v_src_p1r0);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_n1c,  v_src_p0l1);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_p0r1);
    VectorMinMax<Tp>(v_src_p0l2, v_src_p0r1);

    // step1 Get minmax  from 14   delete 46 18
    VectorMinMax<Tp>(v_src_n1r0, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_n1r0, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_n1r0, v_src_p1r0);
    VectorMinMax<Tp>(v_src_n1r0, v_src_p0l1);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l2, v_src_p0r0);

    // step1 Get minmax  from 13   delete 47 17
    VectorMinMax<Tp>(v_src_n1r1, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_n1r1, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_n1r1, v_src_p1r0);
    VectorMinMax<Tp>(v_src_n1r1, v_src_p0l1);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0c);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l2, v_src_p0c);

    // step1 Get minmax  from 12   delete 48 16
    VectorMinMax<Tp>(v_src_n1r2, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_n1r2, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_n1r2, v_src_p1r0);
    VectorMinMax<Tp>(v_src_n1r2, v_src_p0l1);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l2, v_src_p0l0);

    // step2 copy elem
    v_src_p0l0 = v_src_p1l1;
    v_src_p0c = v_src_p1l0;
    v_src_p0r0 = v_src_p1c;
    v_src_p0r1 = v_src_p1r0;
    v_src_p0r2 = v_src_p1r1;
    v_src_c0l2 = v_src_p1r2;
    v_src_c0l1 = v_src_p0l2;
    v_src_c0l0 = v_src_p0l1;

    // step3 Get minmax  from 9   delete 0 15
    VectorMinMax<Tp>(v_src_p2l2, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p2l2, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p2l2, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p2l2, v_src_p0l1);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);
    VectorMinMax<Tp>(v_src_p0l2, v_src_p0l1);

    // step3 Get minmax  from 8   delete 1 14
    VectorMinMax<Tp>(v_src_p2l1, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p1r2, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p2l1, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p2l1, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p0l2);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p0l2);

    // step3 Get minmax  from 8   delete 2 13
    VectorMinMax<Tp>(v_src_p2l0, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p2l0, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p2l0, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r1, v_src_p1r2);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p1r2);

    // step3 Get minmax  from 8   delete 3 12
    VectorMinMax<Tp>(v_src_p2c,  v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1r0, v_src_p1r1);
    VectorMinMax<Tp>(v_src_p2c,  v_src_p1l0);
    VectorMinMax<Tp>(v_src_p2c,  v_src_p1r0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p1r1);

    // step3 Get minmax  from 8   delete 4 11
    VectorMinMax<Tp>(v_src_p2r0, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p2r0, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p2r0, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);
    VectorMinMax<Tp>(v_src_p1c,  v_src_p1r0);

    // step3 Get minmax  from 8   delete 5 10
    VectorMinMax<Tp>(v_src_p2r1, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p1l0, v_src_p1c);
    VectorMinMax<Tp>(v_src_p2r1, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p2r1, v_src_p1r0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1c);

    // step3 Get minmax  from 8   delete 6 9
    VectorMinMax<Tp>(v_src_p2r2, v_src_p1l1);
    VectorMinMax<Tp>(v_src_p2r2, v_src_p1l0);
    VectorMinMax<Tp>(v_src_p1l1, v_src_p1l0);
    VectorMinMax<Tp>(v_src_n2l2, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_n2l2, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_n2l2, v_src_p0r1);
    VectorMinMax<Tp>(v_src_n2l2, v_src_c0l0);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);
    VectorMinMax<Tp>(v_src_c0l1, v_src_c0l0);

    // step3 Get minmax  from 8   delete 1 14
    VectorMinMax<Tp>(v_src_n2l1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_c0l2, v_src_c0l1);

    VectorMinMax<Tp>(v_src_n2l1, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_n2l1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l1);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l1);

    // step3 Get minmax  from 8   delete 2 13
    VectorMinMax<Tp>(v_src_n2l0, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_n2l0, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0r1, v_src_c0l2);
    VectorMinMax<Tp>(v_src_n2l0, v_src_p0r1);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r2, v_src_c0l2);
    VectorMinMax<Tp>(v_src_p0r0, v_src_c0l2);

    // step3 Get minmax  from 8   delete 3 12
    VectorMinMax<Tp>(v_src_n2c,  v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r1, v_src_p0r2);
    VectorMinMax<Tp>(v_src_n2c,  v_src_p0c);
    VectorMinMax<Tp>(v_src_n2c,  v_src_p0r1);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r0, v_src_p0r2);

    // step3 Get minmax  from 8   delete 4 11
    VectorMinMax<Tp>(v_src_n2r0, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_n2r0, v_src_p0c);
    VectorMinMax<Tp>(v_src_n2r0, v_src_p0r1);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);
    VectorMinMax<Tp>(v_src_p0r0, v_src_p0r1);

    // step3 Get minmax  from 8   delete 5 10
    VectorMinMax<Tp>(v_src_n2r1, v_src_p0l0);
    VectorMinMax<Tp>(v_src_p0c,  v_src_p0r0);
    VectorMinMax<Tp>(v_src_n2r1, v_src_p0c);
    VectorMinMax<Tp>(v_src_n2r1, v_src_p0r1);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0r0);

    // step3 Get minmax  from 8   delete 6 9
    VectorMinMax<Tp>(v_src_n2r2, v_src_p0l0);
    VectorMinMax<Tp>(v_src_n2r2, v_src_p0c);
    VectorMinMax<Tp>(v_src_p0l0, v_src_p0c);

    v_result0 = v_src_p1l1;
    v_result1 = v_src_p0l0;
}

template <typename Tp, MI_S32 C>
static AURA_VOID Median7x7TwoRow(const Tp *src_p2, const Tp *src_p1, const Tp *src_p0, const Tp *src_c0, const Tp *src_c1, const Tp *src_n0,
                               const Tp *src_n1, const Tp *src_n2, Tp *dst_c0, Tp *dst_c1, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 back_offset = width - elem_counts;

    MVType mv_src_p2x0, mv_src_p2x1, mv_src_p2x2;
    MVType mv_src_p1x0, mv_src_p1x1, mv_src_p1x2;
    MVType mv_src_p0x0, mv_src_p0x1, mv_src_p0x2;
    MVType mv_src_c0x0, mv_src_c0x1, mv_src_c0x2;
    MVType mv_src_c1x0, mv_src_c1x1, mv_src_c1x2;
    MVType mv_src_n0x0, mv_src_n0x1, mv_src_n0x2;
    MVType mv_src_n1x0, mv_src_n1x1, mv_src_n1x2;
    MVType mv_src_n2x0, mv_src_n2x1, mv_src_n2x2;

    MVType mv_result0 , mv_result1;

    // left
    {
        vload(src_p2, mv_src_p2x1);
        vload(src_p1, mv_src_p1x1);
        vload(src_p0, mv_src_p0x1);
        vload(src_c0, mv_src_c0x1);
        vload(src_c1, mv_src_c1x1);
        vload(src_n0, mv_src_n0x1);
        vload(src_n1, mv_src_n1x1);
        vload(src_n2, mv_src_n2x1);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p2x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_p2x1.val[ch], src_p2[ch], 3);
            mv_src_p1x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_p1x1.val[ch], src_p1[ch], 3);
            mv_src_p0x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_p0x1.val[ch], src_p0[ch], 3);
            mv_src_c0x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_c0x1.val[ch], src_c0[ch], 3);
            mv_src_c1x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_c1x1.val[ch], src_c1[ch], 3);
            mv_src_n0x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_n0x1.val[ch], src_n0[ch], 3);
            mv_src_n1x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_n1x1.val[ch], src_n1[ch], 3);
            mv_src_n2x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_n2x1.val[ch], src_n2[ch], 3);
        }
    }

    // middle
    for (MI_S32 x = elem_counts; x <= back_offset; x += elem_counts)
    {
        vload(src_p2 + C * x, mv_src_p2x2);
        vload(src_p1 + C * x, mv_src_p1x2);
        vload(src_p0 + C * x, mv_src_p0x2);
        vload(src_c0 + C * x, mv_src_c0x2);
        vload(src_c1 + C * x, mv_src_c1x2);
        vload(src_n0 + C * x, mv_src_n0x2);
        vload(src_n1 + C * x, mv_src_n1x2);
        vload(src_n2 + C * x, mv_src_n2x2);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            Median7x7Core<Tp>(mv_src_p2x0.val[ch], mv_src_p2x1.val[ch], mv_src_p2x2.val[ch],
                              mv_src_p1x0.val[ch], mv_src_p1x1.val[ch], mv_src_p1x2.val[ch],
                              mv_src_p0x0.val[ch], mv_src_p0x1.val[ch], mv_src_p0x2.val[ch],
                              mv_src_c0x0.val[ch], mv_src_c0x1.val[ch], mv_src_c0x2.val[ch],
                              mv_src_c1x0.val[ch], mv_src_c1x1.val[ch], mv_src_c1x2.val[ch],
                              mv_src_n0x0.val[ch], mv_src_n0x1.val[ch], mv_src_n0x2.val[ch],
                              mv_src_n1x0.val[ch], mv_src_n1x1.val[ch], mv_src_n1x2.val[ch],
                              mv_src_n2x0.val[ch], mv_src_n2x1.val[ch], mv_src_n2x2.val[ch],
                              mv_result0.val[ch],  mv_result1.val[ch]);
        }
        vstore(dst_c0 + C * (x - elem_counts), mv_result0);
        vstore(dst_c1 + C * (x - elem_counts), mv_result1);

        mv_src_p2x0 = mv_src_p2x1;
        mv_src_p1x0 = mv_src_p1x1;
        mv_src_p0x0 = mv_src_p0x1;
        mv_src_c0x0 = mv_src_c0x1;
        mv_src_c1x0 = mv_src_c1x1;
        mv_src_n0x0 = mv_src_n0x1;
        mv_src_n1x0 = mv_src_n1x1;
        mv_src_n2x0 = mv_src_n2x1;

        mv_src_p2x1 = mv_src_p2x2;
        mv_src_p1x1 = mv_src_p1x2;
        mv_src_p0x1 = mv_src_p0x2;
        mv_src_c0x1 = mv_src_c0x2;
        mv_src_c1x1 = mv_src_c1x2;
        mv_src_n0x1 = mv_src_n0x2;
        mv_src_n1x1 = mv_src_n1x2;
        mv_src_n2x1 = mv_src_n2x2;
    }

    // right
    {
        MI_S32 last = (width - 1) * C;
        MI_S32 rest = width % elem_counts;
        MVType mv_last_result0, mv_last_result1;

        vload(src_p2 + C * back_offset, mv_src_p2x2);
        vload(src_p1 + C * back_offset, mv_src_p1x2);
        vload(src_p0 + C * back_offset, mv_src_p0x2);
        vload(src_c0 + C * back_offset, mv_src_c0x2);
        vload(src_c1 + C * back_offset, mv_src_c1x2);
        vload(src_n0 + C * back_offset, mv_src_n0x2);
        vload(src_n1 + C * back_offset, mv_src_n1x2);
        vload(src_n2 + C * back_offset, mv_src_n2x2);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_p2_border = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_p2x2.val[ch], src_p2[last + ch], 1);
            HVX_Vector v_p1_border = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_p1x2.val[ch], src_p1[last + ch], 1);
            HVX_Vector v_p0_border = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_p0x2.val[ch], src_p0[last + ch], 1);
            HVX_Vector v_c0_border = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_c0x2.val[ch], src_c0[last + ch], 1);
            HVX_Vector v_c1_border = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_c1x2.val[ch], src_c1[last + ch], 1);
            HVX_Vector v_n0_border = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_n0x2.val[ch], src_n0[last + ch], 1);
            HVX_Vector v_n1_border = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_n1x2.val[ch], src_n1[last + ch], 1);
            HVX_Vector v_n2_border = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_n2x2.val[ch], src_n2[last + ch], 1);

            HVX_Vector v_p2r_src = Q6_V_vlalign_VVR(v_p2_border, mv_src_p2x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_p1r_src = Q6_V_vlalign_VVR(v_p1_border, mv_src_p1x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_p0r_src = Q6_V_vlalign_VVR(v_p0_border, mv_src_p0x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_c0r_src = Q6_V_vlalign_VVR(v_c0_border, mv_src_c0x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_c1r_src = Q6_V_vlalign_VVR(v_c1_border, mv_src_c1x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_n0r_src = Q6_V_vlalign_VVR(v_n0_border, mv_src_n0x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_n1r_src = Q6_V_vlalign_VVR(v_n1_border, mv_src_n1x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_n2r_src = Q6_V_vlalign_VVR(v_n2_border, mv_src_n2x2.val[ch], rest * sizeof(Tp));

            Median7x7Core<Tp>(mv_src_p2x0.val[ch], mv_src_p2x1.val[ch], v_p2r_src,
                              mv_src_p1x0.val[ch], mv_src_p1x1.val[ch], v_p1r_src,
                              mv_src_p0x0.val[ch], mv_src_p0x1.val[ch], v_p0r_src,
                              mv_src_c0x0.val[ch], mv_src_c0x1.val[ch], v_c0r_src,
                              mv_src_c1x0.val[ch], mv_src_c1x1.val[ch], v_c1r_src,
                              mv_src_n0x0.val[ch], mv_src_n0x1.val[ch], v_n0r_src,
                              mv_src_n1x0.val[ch], mv_src_n1x1.val[ch], v_n1r_src,
                              mv_src_n2x0.val[ch], mv_src_n2x1.val[ch], v_n2r_src,
                              mv_result0.val[ch],  mv_result1.val[ch]);

            HVX_Vector v_p2l_src = Q6_V_valign_VVR(mv_src_p2x1.val[ch], mv_src_p2x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_p1l_src = Q6_V_valign_VVR(mv_src_p1x1.val[ch], mv_src_p1x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_p0l_src = Q6_V_valign_VVR(mv_src_p0x1.val[ch], mv_src_p0x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_c0l_src = Q6_V_valign_VVR(mv_src_c0x1.val[ch], mv_src_c0x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_c1l_src = Q6_V_valign_VVR(mv_src_c1x1.val[ch], mv_src_c1x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_n0l_src = Q6_V_valign_VVR(mv_src_n0x1.val[ch], mv_src_n0x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_n1l_src = Q6_V_valign_VVR(mv_src_n1x1.val[ch], mv_src_n1x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_n2l_src = Q6_V_valign_VVR(mv_src_n2x1.val[ch], mv_src_n2x0.val[ch], rest * sizeof(Tp));

            Median7x7Core<Tp>(v_p2l_src, mv_src_p2x2.val[ch], v_p2_border,
                              v_p1l_src, mv_src_p1x2.val[ch], v_p1_border,
                              v_p0l_src, mv_src_p0x2.val[ch], v_p0_border,
                              v_c0l_src, mv_src_c0x2.val[ch], v_c0_border,
                              v_c1l_src, mv_src_c1x2.val[ch], v_c1_border,
                              v_n0l_src, mv_src_n0x2.val[ch], v_n0_border,
                              v_n1l_src, mv_src_n1x2.val[ch], v_n1_border,
                              v_n2l_src, mv_src_n2x2.val[ch], v_n2_border,
                              mv_last_result0.val[ch], mv_last_result1.val[ch]);
        }
        vstore(dst_c0 + C * (back_offset - rest), mv_result0);
        vstore(dst_c0 + C * back_offset, mv_last_result0);
        vstore(dst_c1 + C * (back_offset - rest), mv_result1);
        vstore(dst_c1 + C * back_offset, mv_last_result1);
    }
}

template <typename Tp, MI_S32 C>
static Status Median7x7HvxImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width   = src.GetSizes().m_width;
    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;

    const Tp *src_p2 = src.Ptr<Tp, BorderType::REPLICATE>(start_row - 3);
    const Tp *src_p1 = src.Ptr<Tp, BorderType::REPLICATE>(start_row - 2);
    const Tp *src_p0 = src.Ptr<Tp, BorderType::REPLICATE>(start_row - 1);
    const Tp *src_c0 = src.Ptr<Tp>(start_row);
    const Tp *src_c1 = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 1);
    const Tp *src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 2);
    const Tp *src_n1 = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 3);
    const Tp *src_n2 = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 4);

    MI_U64 L2fetch_param = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 2, 0);
    MI_S32 y;
    for (y = start_row; y < end_row - 1; y += 2)
    {
        if (y + 4 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 3)), L2fetch_param);
        }

        Tp *dst_c0 = dst.Ptr<Tp>(y);
        Tp *dst_c1 = dst.Ptr<Tp, BorderType::REPLICATE>(y + 1);
        Median7x7TwoRow<Tp, C>(src_p2, src_p1, src_p0, src_c0, src_c1, src_n0, src_n1, src_n2, dst_c0, dst_c1, width);

        src_p2 = src_p0;
        src_p1 = src_c0;
        src_p0 = src_c1;
        src_c0 = src_n0;
        src_c1 = src_n1;
        src_n0 = src_n2;
        src_n1 = src.Ptr<Tp, BorderType::REPLICATE>(y + 5);
        src_n2 = src.Ptr<Tp, BorderType::REPLICATE>(y + 6);
    }

    if (y == end_row - 1)
    {
        src_p2 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 5);
        src_p1 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 4);
        src_p0 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 3);
        src_c0 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 2);
        src_c1 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 1);
        src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(end_row);
        src_n1 = src.Ptr<Tp, BorderType::REPLICATE>(end_row + 1);
        src_n2 = src.Ptr<Tp, BorderType::REPLICATE>(end_row + 2);

        Tp *dst_c0 = dst.Ptr<Tp, BorderType::REPLICATE>(end_row - 2);
        Tp *dst_c1 = dst.Ptr<Tp, BorderType::REPLICATE>(end_row - 1);

        Median7x7TwoRow<Tp, C>(src_p2, src_p1, src_p0, src_c0, src_c1, src_n0, src_n1, src_n2, dst_c0, dst_c1, width);
    }

    return Status::OK;
}

template<typename Tp>
static Status Median7x7HvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Median7x7HvxImpl<Tp, 1>, src, dst);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Median7x7HvxImpl<Tp, 2>, src, dst);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Median7x7HvxImpl<Tp, 3>, src, dst);
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

Status Median7x7Hvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Median7x7HvxHelper<MI_U8>(ctx, src, dst);
            break;
        }

        case ElemType::S8:
        {
            ret = Median7x7HvxHelper<MI_S8>(ctx, src, dst);
            break;
        }

        case ElemType::U16:
        {
            ret = Median7x7HvxHelper<MI_U16>(ctx, src, dst);
            break;
        }

        case ElemType::S16:
        {
            ret = Median7x7HvxHelper<MI_S16>(ctx, src, dst);
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