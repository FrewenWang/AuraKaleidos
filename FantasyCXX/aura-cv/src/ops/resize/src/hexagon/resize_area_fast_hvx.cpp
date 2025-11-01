#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

struct ResizeAreaVtcmBuffer
{
    MI_U8 *xofs;
    MI_U8 *yofs;
    MI_U8 *src_buffer;
    MI_S32 src_buffer_pitch;
    MI_U8 *gather_buffer;
};

template <typename Tp, typename Rt>
static Status GetResizeAreaFastDnOffset(MI_S32 owidth, MI_S32 oheight, MI_S32 int_scale_x, MI_S32 int_scale_y, ResizeAreaVtcmBuffer *vctm_buffer)
{
    if (MI_NULL == vctm_buffer)
    {
        return Status::ERROR;
    }

    Rt *xofs = reinterpret_cast<Rt *>(vctm_buffer->xofs);
    Rt *yofs = reinterpret_cast<Rt *>(vctm_buffer->yofs);

    for (MI_S32 dx = 0; dx < owidth; dx++)
    {
        xofs[dx] = static_cast<Rt>((int_scale_x * sizeof(Rt)) * dx);
    }

    for (MI_S32 dy = 0; dy < oheight; dy++)
    {
        yofs[dy]  = static_cast<Rt>(int_scale_y * dy);
    }

    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX2Core(HVX_Vector &vu8_result, const HVX_Vector &vu8_c_x0_src, const HVX_Vector &vu8_c_x1_src,
                                              const HVX_Vector &vu8_n0_x0_src, const HVX_Vector &vu8_n0_x1_src)
{
    HVX_Vector vu16_c_x0_sum = Q6_Vh_vdmpy_VubRb(vu8_c_x0_src, 0x01010101);
    HVX_Vector vu16_c_x1_sum = Q6_Vh_vdmpy_VubRb(vu8_c_x1_src, 0x01010101);
    HVX_Vector vu16_x0_sum   = Q6_Vh_vdmpyacc_VhVubRb(vu16_c_x0_sum, vu8_n0_x0_src, 0x01010101);
    HVX_Vector vu16_x1_sum   = Q6_Vh_vdmpyacc_VhVubRb(vu16_c_x1_sum, vu8_n0_x1_src, 0x01010101);

    vu8_result = Q6_Vb_vdeal_Vb(Q6_Vub_vasr_VuhVuhR_rnd_sat(vu16_x1_sum, vu16_x0_sum, 2));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX2Core(HVX_Vector &vs8_result, const HVX_Vector &vs8_c_x0_src, const HVX_Vector &vs8_c_x1_src,
                                              const HVX_Vector &vs8_n0_x0_src, const HVX_Vector &vs8_n0_x1_src)
{
    HVX_VectorPair ws16_c_x0_src  = Q6_Wh_vsxt_Vb(vs8_c_x0_src);
    HVX_VectorPair ws16_c_x1_src  = Q6_Wh_vsxt_Vb(vs8_c_x1_src);
    HVX_VectorPair ws16_n0_x0_src = Q6_Wh_vsxt_Vb(vs8_n0_x0_src);
    HVX_VectorPair ws16_n0_x1_src = Q6_Wh_vsxt_Vb(vs8_n0_x1_src);

    HVX_Vector vs16_c_x0_sum  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_c_x0_src),  Q6_V_hi_W(ws16_c_x0_src));
    HVX_Vector vs16_c_x1_sum  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_c_x1_src),  Q6_V_hi_W(ws16_c_x1_src));
    HVX_Vector vs16_n0_x0_sum = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_n0_x0_src), Q6_V_hi_W(ws16_n0_x0_src));
    HVX_Vector vs16_n0_x1_sum = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_n0_x1_src), Q6_V_hi_W(ws16_n0_x1_src));
    HVX_Vector vs16_x0_sum    = Q6_Vh_vadd_VhVh(vs16_c_x0_sum, vs16_n0_x0_sum);
    HVX_Vector vs16_x1_sum    = Q6_Vh_vadd_VhVh(vs16_c_x1_sum, vs16_n0_x1_sum);

    vs8_result = Q6_Vb_vdeal_Vb(Q6_Vb_vasr_VhVhR_rnd_sat(vs16_x1_sum, vs16_x0_sum, 2));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX2Core(HVX_Vector &vu16_result, const HVX_Vector &vu16_c_x0_src, const HVX_Vector &vu16_c_x1_src,
                                              const HVX_Vector &vu16_n0_x0_src, const HVX_Vector &vu16_n0_x1_src)
{
    HVX_VectorPair wu32_x0_sum = Q6_Ww_vadd_VuhVuh(vu16_c_x0_src, vu16_n0_x0_src);
    HVX_VectorPair wu32_x1_sum = Q6_Ww_vadd_VuhVuh(vu16_c_x1_src, vu16_n0_x1_src);
    HVX_Vector vu32_x0_sum     = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_x0_sum), Q6_V_hi_W(wu32_x0_sum));
    HVX_Vector vu32_x1_sum     = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_x1_sum), Q6_V_hi_W(wu32_x1_sum));

    HVX_Vector vu16_sum = Q6_Vuh_vasr_VwVwR_rnd_sat(vu32_x1_sum, vu32_x0_sum, 2);
    vu16_result = Q6_Vh_vdeal_Vh(vu16_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX2Core(HVX_Vector &vs16_result, const HVX_Vector &vs16_c_x0_src, const HVX_Vector &vs16_c_x1_src,
                                              const HVX_Vector &vs16_n0_x0_src, const HVX_Vector &vs16_n0_x1_src)
{
    HVX_Vector vs32_c_x0_sum = Q6_Vw_vdmpy_VhRb(vs16_c_x0_src, 0x01010101);
    HVX_Vector vs32_c_x1_sum = Q6_Vw_vdmpy_VhRb(vs16_c_x1_src, 0x01010101);
    HVX_Vector vs32_x0_sum   = Q6_Vw_vdmpyacc_VwVhRb(vs32_c_x0_sum, vs16_n0_x0_src, 0x01010101);
    HVX_Vector vs32_x1_sum   = Q6_Vw_vdmpyacc_VwVhRb(vs32_c_x1_sum, vs16_n0_x1_src, 0x01010101);

    vs16_result = Q6_Vh_vdeal_Vh(Q6_Vh_vasr_VwVwR_rnd_sat(vs32_x1_sum, vs32_x0_sum, 2));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4VCore(HVX_Vector &vu32_result, const HVX_Vector &vu8_c_src, const HVX_Vector &vu8_n0_src,
                                               const HVX_Vector &vu8_n1_src, const HVX_Vector &vu8_n2_src)
{
    HVX_VectorPair wu16_sum0 = Q6_Wh_vadd_VubVub(vu8_c_src, vu8_n0_src);
    HVX_VectorPair wu16_sum1 = Q6_Wh_vadd_VubVub(vu8_n1_src, vu8_n2_src);

    HVX_Vector vu16_sum_h = Q6_Vh_vadd_VhVh(Q6_V_lo_W(wu16_sum0), Q6_V_lo_W(wu16_sum1));
    HVX_Vector vu16_sum_l = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wu16_sum0), Q6_V_hi_W(wu16_sum1));

    HVX_Vector vu32_sum_h = Q6_Vw_vdmpy_VhRb(vu16_sum_h, 0x01010101);
    HVX_Vector vu32_sum_l = Q6_Vw_vdmpy_VhRb(vu16_sum_l, 0x01010101);

    vu32_result = Q6_Vw_vadd_VwVw(vu32_sum_h, vu32_sum_l);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4VCore(HVX_Vector &vs32_result, const HVX_Vector &vs8_c_src, const HVX_Vector &vs8_n0_src,
                                               const HVX_Vector &vs8_n1_src, const HVX_Vector &vs8_n2_src)
{
    HVX_VectorPair ws16_c_src  = Q6_Wh_vsxt_Vb(vs8_c_src);
    HVX_VectorPair ws16_n0_src = Q6_Wh_vsxt_Vb(vs8_n0_src);
    HVX_VectorPair ws16_n1_src = Q6_Wh_vsxt_Vb(vs8_n1_src);
    HVX_VectorPair ws16_n2_src = Q6_Wh_vsxt_Vb(vs8_n2_src);

    HVX_Vector vs16_c_sum_l  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_c_src),  Q6_V_lo_W(ws16_n0_src));
    HVX_Vector vs16_c_sum_h  = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_c_src),  Q6_V_hi_W(ws16_n0_src));
    HVX_Vector vs16_n0_sum_l = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_n1_src), Q6_V_lo_W(ws16_n2_src));
    HVX_Vector vs16_n0_sum_h = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_n1_src), Q6_V_hi_W(ws16_n2_src));

    HVX_Vector vs16_sum_l = Q6_Vh_vadd_VhVh(vs16_c_sum_l, vs16_n0_sum_l);
    HVX_Vector vs16_sum_h = Q6_Vh_vadd_VhVh(vs16_c_sum_h, vs16_n0_sum_h);
    HVX_Vector vs32_sum_h = Q6_Vw_vdmpy_VhRb(vs16_sum_h, 0x01010101);
    HVX_Vector vs32_sum_l = Q6_Vw_vdmpy_VhRb(vs16_sum_l, 0x01010101);

    vs32_result = Q6_Vw_vadd_VwVw(vs32_sum_h, vs32_sum_l);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4Core(HVX_Vector &vd32_result0, HVX_Vector &vd32_result1, const HVX_Vector &vd8_c_x0_src, const HVX_Vector &vd8_n0_x0_src,
                                              const HVX_Vector &vd8_n1_x0_src, const HVX_Vector &vd8_n2_x0_src, const HVX_Vector &vd8_c_x1_src,
                                              const HVX_Vector &vd8_n0_x1_src, const HVX_Vector &vd8_n1_x1_src, const HVX_Vector &vd8_n2_x1_src)
{
    ResizeAreaDnX4VCore<Tp>(vd32_result0, vd8_c_x0_src, vd8_n0_x0_src, vd8_n1_x0_src, vd8_n2_x0_src);
    ResizeAreaDnX4VCore<Tp>(vd32_result1, vd8_c_x1_src, vd8_n0_x1_src, vd8_n1_x1_src, vd8_n2_x1_src);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4Core(HVX_Vector &vu32_result0, HVX_Vector &vu32_result1,
                                              const HVX_Vector &vu16_c_x0_src, const HVX_Vector &vu16_n0_x0_src, const HVX_Vector &vu16_n1_x0_src, const HVX_Vector &vu16_n2_x0_src,
                                              const HVX_Vector &vu16_c_x1_src, const HVX_Vector &vu16_n0_x1_src, const HVX_Vector &vu16_n1_x1_src, const HVX_Vector &vu16_n2_x1_src)
{
    AURA_UNUSED(vu32_result1);

    HVX_VectorPair wu32_n0_x0_sum = Q6_Ww_vadd_VuhVuh(vu16_c_x0_src, vu16_n0_x0_src);
    HVX_VectorPair wu32_n1_x0_sum = Q6_Ww_vadd_VuhVuh(vu16_n1_x0_src, vu16_n2_x0_src);
    HVX_VectorPair wu32_n0_x1_sum = Q6_Ww_vadd_VuhVuh(vu16_c_x1_src, vu16_n0_x1_src);
    HVX_VectorPair wu32_n1_x1_sum = Q6_Ww_vadd_VuhVuh(vu16_n1_x1_src, vu16_n2_x1_src);

    HVX_Vector vu32_c_x0_l_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_n0_x0_sum), Q6_V_lo_W(wu32_n1_x0_sum));
    HVX_Vector vu32_c_x0_h_sum = Q6_Vw_vadd_VwVw(Q6_V_hi_W(wu32_n0_x0_sum), Q6_V_hi_W(wu32_n1_x0_sum));
    HVX_Vector vu32_c_x1_l_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_n0_x1_sum), Q6_V_lo_W(wu32_n1_x1_sum));
    HVX_Vector vu32_c_x1_h_sum = Q6_Vw_vadd_VwVw(Q6_V_hi_W(wu32_n0_x1_sum), Q6_V_hi_W(wu32_n1_x1_sum));

    HVX_Vector vu32_c_x0_sum = Q6_Vw_vadd_VwVw(vu32_c_x0_l_sum, vu32_c_x0_h_sum);
    HVX_Vector vu32_c_x1_sum = Q6_Vw_vadd_VwVw(vu32_c_x1_l_sum, vu32_c_x1_h_sum);

    HVX_VectorPair wu32_sum = Q6_W_vdeal_VVR(vu32_c_x1_sum, vu32_c_x0_sum, -4);

    vu32_result0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_sum), Q6_V_hi_W(wu32_sum));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4Core(HVX_Vector &vs32_result0, HVX_Vector &vs32_result1,
                                              const HVX_Vector &vs16_c_x0_src, const HVX_Vector &vs16_n0_x0_src, const HVX_Vector &vs16_n1_x0_src, const HVX_Vector &vs16_n2_x0_src,
                                              const HVX_Vector &vs16_c_x1_src, const HVX_Vector &vs16_n0_x1_src, const HVX_Vector &vs16_n1_x1_src, const HVX_Vector &vs16_n2_x1_src)
{
    AURA_UNUSED(vs32_result1);

    HVX_VectorPair ws32_n0_x0_sum = Q6_Ww_vadd_VhVh(vs16_c_x0_src, vs16_n0_x0_src);
    HVX_VectorPair ws32_n1_x0_sum = Q6_Ww_vadd_VhVh(vs16_n1_x0_src, vs16_n2_x0_src);
    HVX_VectorPair ws32_n0_x1_sum = Q6_Ww_vadd_VhVh(vs16_c_x1_src, vs16_n0_x1_src);
    HVX_VectorPair ws32_n1_x1_sum = Q6_Ww_vadd_VhVh(vs16_n1_x1_src, vs16_n2_x1_src);

    HVX_Vector vs32_c_x0_l_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_n0_x0_sum), Q6_V_lo_W(ws32_n1_x0_sum));
    HVX_Vector vs32_c_x0_h_sum = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_n0_x0_sum), Q6_V_hi_W(ws32_n1_x0_sum));
    HVX_Vector vs32_c_x1_l_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_n0_x1_sum), Q6_V_lo_W(ws32_n1_x1_sum));
    HVX_Vector vs32_c_x1_h_sum = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_n0_x1_sum), Q6_V_hi_W(ws32_n1_x1_sum));

    HVX_Vector vs32_c_x0_sum = Q6_Vw_vadd_VwVw(vs32_c_x0_l_sum, vs32_c_x0_h_sum);
    HVX_Vector vs32_c_x1_sum = Q6_Vw_vadd_VwVw(vs32_c_x1_l_sum, vs32_c_x1_h_sum);

    HVX_VectorPair ws32_x_sum = Q6_W_vdeal_VVR(vs32_c_x1_sum, vs32_c_x0_sum, -4);

    vs32_result0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_x_sum), Q6_V_hi_W(ws32_x_sum));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4SumCore(HVX_Vector &vu8_result, HVX_Vector &vu32_sum0, HVX_Vector &vu32_sum1,
                                                 HVX_Vector &vu32_sum2, HVX_Vector &vu32_sum3)
{
    HVX_Vector vu16_sum0 = Q6_Vh_vpacke_VwVw(vu32_sum1, vu32_sum0);
    HVX_Vector vu16_sum1 = Q6_Vh_vpacke_VwVw(vu32_sum3, vu32_sum2);

    HVX_Vector vu8_sum = Q6_Vub_vasr_VuhVuhR_rnd_sat(vu16_sum1, vu16_sum0, 4);

    vu8_result = Q6_Vb_vdeal_Vb(vu8_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4SumCore(HVX_Vector &vs8_result, HVX_Vector &vs32_sum0, HVX_Vector &vs32_sum1,
                                                 HVX_Vector &vs32_sum2, HVX_Vector &vs32_sum3)
{
    HVX_Vector vs16_sum0 = Q6_Vh_vpacke_VwVw(vs32_sum1, vs32_sum0);
    HVX_Vector vs16_sum1 = Q6_Vh_vpacke_VwVw(vs32_sum3, vs32_sum2);

    HVX_Vector vs8_sum = Q6_Vb_vasr_VhVhR_rnd_sat(vs16_sum1, vs16_sum0, 4);

    vs8_result = Q6_Vb_vdeal_Vb(vs8_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4SumCore(HVX_Vector &vu16_result, HVX_Vector &vu32_sum0, HVX_Vector &vu32_sum1,
                                                 HVX_Vector &vu32_sum2, HVX_Vector &vu32_sum3)
{
    AURA_UNUSED(vu32_sum1);
    AURA_UNUSED(vu32_sum3);

    HVX_Vector vu16_sum = Q6_Vuh_vasr_VwVwR_rnd_sat(vu32_sum2, vu32_sum0, 4);
    vu16_result = Q6_Vh_vdeal_Vh(vu16_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4SumCore(HVX_Vector &vs16_result, HVX_Vector &vs32_sum0, HVX_Vector &vs32_sum1,
                                                 HVX_Vector &vs32_sum2, HVX_Vector &vs32_sum3)
{
    AURA_UNUSED(vs32_sum1);
    AURA_UNUSED(vs32_sum3);

    HVX_Vector vs16_sum = Q6_Vh_vasr_VwVwR_rnd_sat(vs32_sum2, vs32_sum0, 4);
    vs16_result = Q6_Vh_vdeal_Vh(vs16_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX2Y4Core(HVX_Vector &vu8_result, const HVX_Vector &vu8_c_x0_src, const HVX_Vector &vu8_c_x1_src, const HVX_Vector &vu8_n0_x0_src,
                                                const HVX_Vector &vu8_n0_x1_src, const HVX_Vector &vu8_n1_x0_src, const HVX_Vector &vu8_n1_x1_src,
                                                const HVX_Vector &vu8_n2_x0_src, const HVX_Vector &vu8_n2_x1_src)
{
    HVX_VectorPair wu16_c_x0_sum  = Q6_Wh_vadd_VubVub(vu8_c_x0_src, vu8_n0_x0_src);
    HVX_VectorPair wu16_c_x1_sum  = Q6_Wh_vadd_VubVub(vu8_c_x1_src, vu8_n0_x1_src);
    HVX_VectorPair wu16_n0_x0_sum = Q6_Wh_vadd_VubVub(vu8_n1_x0_src, vu8_n2_x0_src);
    HVX_VectorPair wu16_n0_x1_sum = Q6_Wh_vadd_VubVub(vu8_n1_x1_src, vu8_n2_x1_src);

    HVX_VectorPair wu16_sum0 = Q6_Wh_vadd_WhWh(wu16_c_x0_sum, wu16_n0_x0_sum);
    HVX_VectorPair wu16_sum1 = Q6_Wh_vadd_WhWh(wu16_c_x1_sum, wu16_n0_x1_sum);

    HVX_Vector vu16_sum0 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wu16_sum0), Q6_V_lo_W(wu16_sum0));
    HVX_Vector vu16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wu16_sum1), Q6_V_lo_W(wu16_sum1));

    HVX_Vector vu8_sum = Q6_Vub_vasr_VuhVuhR_rnd_sat(vu16_sum1, vu16_sum0, 3);

    vu8_result = Q6_Vb_vdeal_Vb(vu8_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX2Y4Core(HVX_Vector &vs8_result, const HVX_Vector &vs8_c_x0_src, const HVX_Vector &vs8_c_x1_src, const HVX_Vector &vs8_n0_x0_src,
                                                const HVX_Vector &vs8_n0_x1_src, const HVX_Vector &vs8_n1_x0_src, const HVX_Vector &vs8_n1_x1_src,
                                                const HVX_Vector &vs8_n2_x0_src, const HVX_Vector &vs8_n2_x1_src)
{
    HVX_VectorPair ws16_c_x0_src  = Q6_Wh_vsxt_Vb(vs8_c_x0_src);
    HVX_VectorPair ws16_c_x1_src  = Q6_Wh_vsxt_Vb(vs8_c_x1_src);
    HVX_VectorPair ws16_n0_x0_src = Q6_Wh_vsxt_Vb(vs8_n0_x0_src);
    HVX_VectorPair ws16_n0_x1_src = Q6_Wh_vsxt_Vb(vs8_n0_x1_src);
    HVX_VectorPair ws16_n1_x0_src = Q6_Wh_vsxt_Vb(vs8_n1_x0_src);
    HVX_VectorPair ws16_n1_x1_src = Q6_Wh_vsxt_Vb(vs8_n1_x1_src);
    HVX_VectorPair ws16_n2_x0_src = Q6_Wh_vsxt_Vb(vs8_n2_x0_src);
    HVX_VectorPair ws16_n2_x1_src = Q6_Wh_vsxt_Vb(vs8_n2_x1_src);

    HVX_VectorPair ws16_c_sum0  = Q6_Wh_vadd_WhWh(ws16_c_x0_src,  ws16_n0_x0_src);
    HVX_VectorPair ws16_c_sum1  = Q6_Wh_vadd_WhWh(ws16_c_x1_src,  ws16_n0_x1_src);
    HVX_VectorPair ws16_n0_sum0 = Q6_Wh_vadd_WhWh(ws16_n1_x0_src, ws16_n2_x0_src);
    HVX_VectorPair ws16_n0_sum1 = Q6_Wh_vadd_WhWh(ws16_n1_x1_src, ws16_n2_x1_src);

    HVX_VectorPair ws16_sum0 = Q6_Wh_vadd_WhWh(ws16_c_sum0, ws16_n0_sum0);
    HVX_VectorPair ws16_sum1 = Q6_Wh_vadd_WhWh(ws16_c_sum1, ws16_n0_sum1);

    HVX_Vector vs16_sum0 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_sum0), Q6_V_lo_W(ws16_sum0));
    HVX_Vector vs16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_sum1), Q6_V_lo_W(ws16_sum1));

    HVX_Vector vs8_sum = Q6_Vb_vasr_VhVhR_rnd_sat(vs16_sum1, vs16_sum0, 3);
    vs8_result = Q6_Vb_vdeal_Vb(vs8_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX2Y4Core(HVX_Vector &vu16_result, const HVX_Vector &vu16_c_x0_src, const HVX_Vector &vu16_c_x1_src, const HVX_Vector &vu16_n0_x0_src,
                                                const HVX_Vector &vu16_n0_x1_src, const HVX_Vector &vu16_n1_x0_src, const HVX_Vector &vu16_n1_x1_src,
                                                const HVX_Vector &vu16_n2_x0_src, const HVX_Vector &vu16_n2_x1_src)
{
    HVX_VectorPair wu32_c_x0_sum  = Q6_Ww_vadd_VuhVuh(vu16_c_x0_src, vu16_n0_x0_src);
    HVX_VectorPair wu32_c_x1_sum  = Q6_Ww_vadd_VuhVuh(vu16_c_x1_src, vu16_n0_x1_src);
    HVX_VectorPair wu32_n0_x0_sum = Q6_Ww_vadd_VuhVuh(vu16_n1_x0_src, vu16_n2_x0_src);
    HVX_VectorPair wu32_n0_x1_sum = Q6_Ww_vadd_VuhVuh(vu16_n1_x1_src, vu16_n2_x1_src);

    HVX_VectorPair wu32_sum0 = Q6_Ww_vadd_WwWw(wu32_c_x0_sum, wu32_n0_x0_sum);
    HVX_VectorPair wu32_sum1 = Q6_Ww_vadd_WwWw(wu32_c_x1_sum, wu32_n0_x1_sum);
    HVX_Vector vu32_sum0     = Q6_Vw_vadd_VwVw(Q6_V_hi_W(wu32_sum0), Q6_V_lo_W(wu32_sum0));
    HVX_Vector vu32_sum1     = Q6_Vw_vadd_VwVw(Q6_V_hi_W(wu32_sum1), Q6_V_lo_W(wu32_sum1));
    HVX_Vector vu16_sum      = Q6_Vuh_vasr_VwVwR_rnd_sat(vu32_sum1, vu32_sum0, 3);

    vu16_result = Q6_Vh_vdeal_Vh(vu16_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX2Y4Core(HVX_Vector &vs16_result, const HVX_Vector &vs16_c_x0_src, const HVX_Vector &vs16_c_x1_src, const HVX_Vector &vs16_n0_x0_src,
                                                const HVX_Vector &vs16_n0_x1_src, const HVX_Vector &vs16_n1_x0_src, const HVX_Vector &vs16_n1_x1_src,
                                                const HVX_Vector &vs16_n2_x0_src, const HVX_Vector &vs16_n2_x1_src)
{
    HVX_VectorPair ws32_c_x0_sum  = Q6_Ww_vadd_VhVh(vs16_c_x0_src, vs16_n0_x0_src);
    HVX_VectorPair ws32_c_x1_sum  = Q6_Ww_vadd_VhVh(vs16_c_x1_src, vs16_n0_x1_src);
    HVX_VectorPair ws32_n0_x0_sum = Q6_Ww_vadd_VhVh(vs16_n1_x0_src, vs16_n2_x0_src);
    HVX_VectorPair ws32_n0_x1_sum = Q6_Ww_vadd_VhVh(vs16_n1_x1_src, vs16_n2_x1_src);

    HVX_VectorPair ws32_sum0 = Q6_Ww_vadd_WwWw(ws32_c_x0_sum, ws32_n0_x0_sum);
    HVX_VectorPair ws32_sum1 = Q6_Ww_vadd_WwWw(ws32_c_x1_sum, ws32_n0_x1_sum);
    HVX_Vector vs32_sum0     = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_sum0), Q6_V_lo_W(ws32_sum0));
    HVX_Vector vs32_sum1     = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ws32_sum1), Q6_V_lo_W(ws32_sum1));
    HVX_Vector vs16_sum      = Q6_Vuh_vasr_VwVwR_rnd_sat(vs32_sum1, vs32_sum0, 3);

    vs16_result = Q6_Vh_vdeal_Vh(vs16_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4Y2Core(HVX_Vector &vu8_result, const HVX_Vector &vu8_c_x0_src, const HVX_Vector &vu8_c_x1_src, const HVX_Vector &vu8_c_x2_src,
                                                const HVX_Vector &vu8_c_x3_src, const HVX_Vector &vu8_n0_x0_src, const HVX_Vector &vu8_n0_x1_src,
                                                const HVX_Vector &vu8_n0_x2_src, const HVX_Vector &vu8_n0_x3_src)
{
    HVX_VectorPair wu16_sum0 = Q6_Wh_vadd_VubVub(vu8_c_x0_src, vu8_n0_x0_src);
    HVX_VectorPair wu16_sum1 = Q6_Wh_vadd_VubVub(vu8_c_x1_src, vu8_n0_x1_src);
    HVX_VectorPair wu16_sum2 = Q6_Wh_vadd_VubVub(vu8_c_x2_src, vu8_n0_x2_src);
    HVX_VectorPair wu16_sum3 = Q6_Wh_vadd_VubVub(vu8_c_x3_src, vu8_n0_x3_src);

    HVX_Vector vu16_sum0 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wu16_sum0), Q6_V_lo_W(wu16_sum0));
    HVX_Vector vu16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wu16_sum1), Q6_V_lo_W(wu16_sum1));
    HVX_Vector vu16_sum2 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wu16_sum2), Q6_V_lo_W(wu16_sum2));
    HVX_Vector vu16_sum3 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wu16_sum3), Q6_V_lo_W(wu16_sum3));

    HVX_Vector vu32_sum0 = Q6_Vw_vdmpy_VhRb(vu16_sum0, 0x01010101);
    HVX_Vector vu32_sum1 = Q6_Vw_vdmpy_VhRb(vu16_sum1, 0x01010101);
    HVX_Vector vu32_sum2 = Q6_Vw_vdmpy_VhRb(vu16_sum2, 0x01010101);
    HVX_Vector vu32_sum3 = Q6_Vw_vdmpy_VhRb(vu16_sum3, 0x01010101);

    HVX_Vector vu16_result0 = Q6_Vh_vpacke_VwVw(vu32_sum1, vu32_sum0);
    HVX_Vector vu16_result1 = Q6_Vh_vpacke_VwVw(vu32_sum3, vu32_sum2);

    vu8_result = Q6_Vb_vdeal_Vb(Q6_Vub_vasr_VuhVuhR_rnd_sat(vu16_result1, vu16_result0, 3));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4Y2Core(HVX_Vector &vs8_result, const HVX_Vector &vs8_c_x0_src, const HVX_Vector &vs8_c_x1_src, const HVX_Vector &vs8_c_x2_src,
                                                const HVX_Vector &vs8_c_x3_src, const HVX_Vector &vs8_n0_x0_src, const HVX_Vector &vs8_n0_x1_src,
                                                const HVX_Vector &vs8_n0_x2_src, const HVX_Vector &vs8_n0_x3_src)
{
    HVX_VectorPair ws16_c_x0_src = Q6_Wh_vsxt_Vb(vs8_c_x0_src);
    HVX_VectorPair ws16_c_x1_src = Q6_Wh_vsxt_Vb(vs8_c_x1_src);
    HVX_VectorPair ws16_c_x2_src = Q6_Wh_vsxt_Vb(vs8_c_x2_src);
    HVX_VectorPair ws16_c_x3_src = Q6_Wh_vsxt_Vb(vs8_c_x3_src);

    HVX_VectorPair ws16_n0_x0_src = Q6_Wh_vsxt_Vb(vs8_n0_x0_src);
    HVX_VectorPair ws16_n0_x1_src = Q6_Wh_vsxt_Vb(vs8_n0_x1_src);
    HVX_VectorPair ws16_n0_x2_src = Q6_Wh_vsxt_Vb(vs8_n0_x2_src);
    HVX_VectorPair ws16_n0_x3_src = Q6_Wh_vsxt_Vb(vs8_n0_x3_src);

    HVX_VectorPair ws16_x0_sum = Q6_Wh_vadd_WhWh(ws16_c_x0_src, ws16_n0_x0_src);
    HVX_VectorPair ws16_x1_sum = Q6_Wh_vadd_WhWh(ws16_c_x1_src, ws16_n0_x1_src);
    HVX_VectorPair ws16_x2_sum = Q6_Wh_vadd_WhWh(ws16_c_x2_src, ws16_n0_x2_src);
    HVX_VectorPair ws16_x3_sum = Q6_Wh_vadd_WhWh(ws16_c_x3_src, ws16_n0_x3_src);

    HVX_Vector vs16_sum0 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_x0_sum), Q6_V_lo_W(ws16_x0_sum));
    HVX_Vector vs16_sum1 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_x1_sum), Q6_V_lo_W(ws16_x1_sum));
    HVX_Vector vs16_sum2 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_x2_sum), Q6_V_lo_W(ws16_x2_sum));
    HVX_Vector vs16_sum3 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(ws16_x3_sum), Q6_V_lo_W(ws16_x3_sum));

    HVX_Vector vs32_sum0 = Q6_Vw_vdmpy_VhRb(vs16_sum0, 0x01010101);
    HVX_Vector vs32_sum1 = Q6_Vw_vdmpy_VhRb(vs16_sum1, 0x01010101);
    HVX_Vector vs32_sum2 = Q6_Vw_vdmpy_VhRb(vs16_sum2, 0x01010101);
    HVX_Vector vs32_sum3 = Q6_Vw_vdmpy_VhRb(vs16_sum3, 0x01010101);

    HVX_Vector vs16_result0 = Q6_Vh_vpacke_VwVw(vs32_sum1, vs32_sum0);
    HVX_Vector vs16_result1 = Q6_Vh_vpacke_VwVw(vs32_sum3, vs32_sum2);

    vs8_result = Q6_Vb_vdeal_Vb(Q6_Vub_vasr_VuhVuhR_rnd_sat(vs16_result1, vs16_result0, 3));
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4Y2Core(HVX_Vector &vu16_result, const HVX_Vector &vu16_c_x0_src, const HVX_Vector &vu16_c_x1_src, const HVX_Vector &vu16_c_x2_src,
                                                const HVX_Vector &vu16_c_x3_src, const HVX_Vector &vu16_n0_x0_src, const HVX_Vector &vu16_n0_x1_src,
                                                const HVX_Vector &vu16_n0_x2_src, const HVX_Vector &vu16_n0_x3_src)
{
    HVX_VectorPair wu32_x0_sum = Q6_Ww_vadd_VuhVuh(vu16_c_x0_src, vu16_n0_x0_src);
    HVX_VectorPair wu32_x1_sum = Q6_Ww_vadd_VuhVuh(vu16_c_x1_src, vu16_n0_x1_src);
    HVX_VectorPair wu32_x2_sum = Q6_Ww_vadd_VuhVuh(vu16_c_x2_src, vu16_n0_x2_src);
    HVX_VectorPair wu32_x3_sum = Q6_Ww_vadd_VuhVuh(vu16_c_x3_src, vu16_n0_x3_src);

    HVX_Vector vu32_x0_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_x0_sum), Q6_V_hi_W(wu32_x0_sum));
    HVX_Vector vu32_x1_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_x1_sum), Q6_V_hi_W(wu32_x1_sum));
    HVX_Vector vu32_x2_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_x2_sum), Q6_V_hi_W(wu32_x2_sum));
    HVX_Vector vu32_x3_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_x3_sum), Q6_V_hi_W(wu32_x3_sum));

    HVX_VectorPair wu32_sum0 = Q6_W_vdeal_VVR(vu32_x1_sum, vu32_x0_sum, -4);
    HVX_VectorPair wu32_sum1 = Q6_W_vdeal_VVR(vu32_x3_sum, vu32_x2_sum, -4);

    HVX_Vector vu32_result0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_sum0), Q6_V_hi_W(wu32_sum0));
    HVX_Vector vu32_result1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(wu32_sum1), Q6_V_hi_W(wu32_sum1));

    HVX_Vector vu16_sum = Q6_Vuh_vasr_VwVwR_rnd_sat(vu32_result1, vu32_result0, 3);
    vu16_result = Q6_Vh_vdeal_Vh(vu16_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnX4Y2Core(HVX_Vector &vs16_result, const HVX_Vector &vs16_c_x0_src, const HVX_Vector &vs16_c_x1_src, const HVX_Vector &vs16_c_x2_src,
                                                const HVX_Vector &vs16_c_x3_src, const HVX_Vector &vs16_n0_x0_src, const HVX_Vector &vs16_n0_x1_src,
                                                const HVX_Vector &vs16_n0_x2_src, const HVX_Vector &vs16_n0_x3_src)
{
    HVX_VectorPair ws32_x0_sum = Q6_Ww_vadd_VhVh(vs16_c_x0_src, vs16_n0_x0_src);
    HVX_VectorPair ws32_x1_sum = Q6_Ww_vadd_VhVh(vs16_c_x1_src, vs16_n0_x1_src);
    HVX_VectorPair ws32_x2_sum = Q6_Ww_vadd_VhVh(vs16_c_x2_src, vs16_n0_x2_src);
    HVX_VectorPair ws32_x3_sum = Q6_Ww_vadd_VhVh(vs16_c_x3_src, vs16_n0_x3_src);

    HVX_Vector vs32_x0_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_x0_sum), Q6_V_hi_W(ws32_x0_sum));
    HVX_Vector vs32_x1_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_x1_sum), Q6_V_hi_W(ws32_x1_sum));
    HVX_Vector vs32_x2_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_x2_sum), Q6_V_hi_W(ws32_x2_sum));
    HVX_Vector vs32_x3_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_x3_sum), Q6_V_hi_W(ws32_x3_sum));

    HVX_VectorPair ws32_sum0 = Q6_W_vdeal_VVR(vs32_x1_sum, vs32_x0_sum, -4);
    HVX_VectorPair ws32_sum1 = Q6_W_vdeal_VVR(vs32_x3_sum, vs32_x2_sum, -4);

    HVX_Vector vs32_result0 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_sum0), Q6_V_hi_W(ws32_sum0));
    HVX_Vector vs32_result1 = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ws32_sum1), Q6_V_hi_W(ws32_sum1));

    HVX_Vector vs16_sum = Q6_Vuh_vasr_VwVwR_rnd_sat(vs32_result1, vs32_result0, 3);
    vs16_result = Q6_Vh_vdeal_Vh(vs16_sum);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaUpX2Core(HVX_Vector &vd8_c_src, HVX_Vector &vd8_dst_l, HVX_Vector &vd8_dst_h)
{
    HVX_VectorPair w_c_dst;

    w_c_dst = Q6_Wuh_vunpack_Vub(vd8_c_src);
    w_c_dst = Q6_Wh_vunpackoor_WhVb(w_c_dst, vd8_c_src);
    vd8_dst_l = Q6_V_lo_W(w_c_dst);
    vd8_dst_h = Q6_V_hi_W(w_c_dst);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaUpX2Core(HVX_Vector &vd16_c_src, HVX_Vector &vd16_dst_l, HVX_Vector &vd16_dst_h)
{
    HVX_VectorPair w_c_dst;

    w_c_dst = Q6_Wuw_vunpack_Vuh(vd16_c_src);
    w_c_dst = Q6_Ww_vunpackoor_WwVh(w_c_dst, vd16_c_src);
    vd16_dst_l = Q6_V_lo_W(w_c_dst);
    vd16_dst_h = Q6_V_hi_W(w_c_dst);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaUpX4Core(HVX_Vector &vd8_c_src, HVX_Vector &vd8_c_dst, HVX_Vector &vd8_r0_dst,
                                              HVX_Vector &vd8_r1_dst, HVX_Vector &vd8_r2_dst)
{
    HVX_Vector v_c_l, v_c_h;
    HVX_VectorPair w_c_res, w_c_l, w_c_h;

    w_c_res = Q6_Wuh_vunpack_Vub(vd8_c_src);
    w_c_res = Q6_Wh_vunpackoor_WhVb(w_c_res, vd8_c_src);

    v_c_l = Q6_V_lo_W(w_c_res);
    v_c_h = Q6_V_hi_W(w_c_res);
    w_c_l = Q6_Wuh_vunpack_Vub(v_c_l);
    w_c_h = Q6_Wuh_vunpack_Vub(v_c_h);
    w_c_l = Q6_Wh_vunpackoor_WhVb(w_c_l, v_c_l);
    w_c_h = Q6_Wh_vunpackoor_WhVb(w_c_h, v_c_h);

    vd8_c_dst  = Q6_V_lo_W(w_c_l);
    vd8_r0_dst = Q6_V_hi_W(w_c_l);
    vd8_r1_dst = Q6_V_lo_W(w_c_h);
    vd8_r2_dst = Q6_V_hi_W(w_c_h);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaUpX4Core(HVX_Vector &vd16_c_src, HVX_Vector &vd16_c_dst, HVX_Vector &vd16_r0_dst,
                                              HVX_Vector &vd16_r1_dst, HVX_Vector &vd16_r2_dst)
{
    HVX_Vector v_c_l, v_c_h;
    HVX_VectorPair w_c_res, w_c_l, w_c_h;

    w_c_res = Q6_Wuw_vunpack_Vuh(vd16_c_src);
    w_c_res = Q6_Ww_vunpackoor_WwVh(w_c_res, vd16_c_src);

    v_c_l = Q6_V_lo_W(w_c_res);
    v_c_h = Q6_V_hi_W(w_c_res);
    w_c_l = Q6_Wuw_vunpack_Vuh(v_c_l);
    w_c_h = Q6_Wuw_vunpack_Vuh(v_c_h);
    w_c_l = Q6_Ww_vunpackoor_WwVh(w_c_l, v_c_l);
    w_c_h = Q6_Ww_vunpackoor_WwVh(w_c_h, v_c_h);

    vd16_c_dst  = Q6_V_lo_W(w_c_l);
    vd16_r0_dst = Q6_V_hi_W(w_c_l);
    vd16_r1_dst = Q6_V_lo_W(w_c_h);
    vd16_r2_dst = Q6_V_hi_W(w_c_h);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaFastDnVCore(HVX_VectorPair &wu16_src_sum, HVX_Vector &vu8_c_src)
{
    wu16_src_sum = Q6_Wh_vaddacc_WhVubVub(wu16_src_sum, vu8_c_src, Q6_V_vzero());
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaFastDnVCore(HVX_VectorPair &ws16_src_sum, HVX_Vector &vs8_c_src)
{
    HVX_VectorPair ws16_c_src = Q6_Wh_vmpy_VbVb(vs8_c_src, Q6_Vb_vsplat_R(1));
    ws16_src_sum = Q6_Wh_vadd_WhWh(ws16_src_sum, ws16_c_src);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaFastDnVCore(HVX_VectorPair &wu32_src_sum, HVX_Vector &vu16_c_src)
{
    wu32_src_sum = Q6_Ww_vaddacc_WwVuhVuh(wu32_src_sum, vu16_c_src, Q6_V_vzero());
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaFastDnVCore(HVX_VectorPair &ws32_src_sum, HVX_Vector &vs16_c_src)
{
    ws32_src_sum = Q6_Ww_vaddacc_WwVhVh(ws32_src_sum, vs16_c_src, Q6_V_vzero());
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaFastDnVCore(HVX_VectorPair &wd16_src_sum)
{
    wd16_src_sum = Q6_W_vshuff_VVR(Q6_V_hi_W(wd16_src_sum), Q6_V_lo_W(wd16_src_sum), -2);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaFastDnVCore(HVX_VectorPair &wd32_src_sum)
{
    wd32_src_sum = Q6_W_vshuff_VVR(Q6_V_hi_W(wd32_src_sum), Q6_V_lo_W(wd32_src_sum), -4);
}

template <typename Tp, typename Rt, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaFastDnHCore(HVX_Vector &vu8_result, HVX_Vector &vu16_offset0, HVX_Vector &vu16_offset1, Rt *src_buffer, HVX_Vector *vu16_gather0_ptr,
                                                 HVX_Vector *vu16_gather1_ptr, MI_S32 int_scale_x, MI_S32 scale_xy, MI_S32 ch, MI_S32 iwidth)
{
    HVX_VectorPair ws32_sum0 = Q6_W_vzero();
    HVX_VectorPair ws32_sum1 = Q6_W_vzero();
    HVX_Vector vu16_scale_xy = Q6_Vh_vsplat_R(scale_xy);
    HVX_VectorPair wu32_scale_add = Q6_Wuw_vmpy_VuhRuh(Q6_Vh_vsplat_R(scale_xy >> 1), 0x00010001);

    for (MI_S32 sx = 0; sx < int_scale_x; sx++)
    {
        HVX_Vector v_one  = Q6_Vh_vsplat_R(sx << 1);
        HVX_Vector v_idx0 = Q6_Vh_vadd_VhVh(vu16_offset0, v_one);
        HVX_Vector v_idx1 = Q6_Vh_vadd_VhVh(vu16_offset1, v_one);

        Q6_vgather_ARMVh(vu16_gather0_ptr, (MI_U32)(src_buffer + ch * iwidth), (iwidth << 1) -1, v_idx0);
        Q6_vgather_ARMVh(vu16_gather1_ptr, (MI_U32)(src_buffer + ch * iwidth), (iwidth << 1) -1, v_idx1);

        HVX_Vector vu16_gather0 = *vu16_gather0_ptr;
        HVX_Vector vu16_gather1 = *vu16_gather1_ptr;

        ws32_sum0 = Q6_Ww_vaddacc_WwVuhVuh(ws32_sum0, vu16_gather0, Q6_V_vzero());
        ws32_sum1 = Q6_Ww_vaddacc_WwVuhVuh(ws32_sum1, vu16_gather1, Q6_V_vzero());
    }

    HVX_Vector vu16_div_sum0 = Q6_Vuh_vdiv8_WuwVuh(Q6_Ww_vadd_WwWw(ws32_sum0, wu32_scale_add), vu16_scale_xy);
    HVX_Vector vu16_div_sum1 = Q6_Vuh_vdiv8_WuwVuh(Q6_Ww_vadd_WwWw(ws32_sum1, wu32_scale_add), vu16_scale_xy);
    vu8_result = Q6_Vb_vdeale_VbVb(vu16_div_sum1, vu16_div_sum0);
}

template <typename Tp, typename Rt, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaFastDnHCore(HVX_Vector &vs8_result, HVX_Vector &vu16_offset0, HVX_Vector &vu16_offset1, Rt *src_buffer, HVX_Vector *vs16_gather0_ptr,
                                                 HVX_Vector *vs16_gather1_ptr, MI_S32 int_scale_x, MI_S32 scale_xy, MI_S32 ch, MI_S32 iwidth)
{
    HVX_VectorPair ws32_sum0 = Q6_W_vzero();
    HVX_VectorPair ws32_sum1 = Q6_W_vzero();
    HVX_Vector vs16_scale_xy = Q6_Vh_vsplat_R(scale_xy);
    HVX_VectorPair ws32_scale_add = Q6_Ww_vmpy_VhRh(Q6_Vh_vsplat_R(scale_xy >> 1), 0x00010001);

    for (MI_S32 sx = 0; sx < int_scale_x; sx++)
    {
        HVX_Vector v_one  = Q6_Vh_vsplat_R(sx << 1);
        HVX_Vector v_idx0 = Q6_Vh_vadd_VhVh(vu16_offset0, v_one);
        HVX_Vector v_idx1 = Q6_Vh_vadd_VhVh(vu16_offset1, v_one);

        Q6_vgather_ARMVh(vs16_gather0_ptr, (MI_U32)(src_buffer + ch * iwidth), (iwidth << 1) -1, v_idx0);
        Q6_vgather_ARMVh(vs16_gather1_ptr, (MI_U32)(src_buffer + ch * iwidth), (iwidth << 1) -1, v_idx1);

        HVX_Vector vu16_gather0 = *vs16_gather0_ptr;
        HVX_Vector vu16_gather1 = *vs16_gather1_ptr;

        ws32_sum0 = Q6_Ww_vaddacc_WwVhVh(ws32_sum0, vu16_gather0, Q6_V_vzero());
        ws32_sum1 = Q6_Ww_vaddacc_WwVhVh(ws32_sum1, vu16_gather1, Q6_V_vzero());
    }

    HVX_Vector vs16_div_sum0 = Q6_Vh_vdiv8_WwVh(Q6_Ww_vadd_WwWw(ws32_sum0, ws32_scale_add), vs16_scale_xy);
    HVX_Vector vs16_div_sum1 = Q6_Vh_vdiv8_WwVh(Q6_Ww_vadd_WwWw(ws32_sum1, ws32_scale_add), vs16_scale_xy);
    vs8_result = Q6_Vb_vdeale_VbVb(vs16_div_sum1, vs16_div_sum0);
}

template <typename Tp, typename Rt, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaFastDnHCore(HVX_Vector &vu16_result, HVX_Vector &vu32_offset0, HVX_Vector &vu32_offset1, Rt *src_buffer, HVX_Vector *vu32_gather0_ptr,
                                                 HVX_Vector *vu32_gather1_ptr, MI_S32 int_scale_x, MI_S32 scale_xy, MI_S32 ch, MI_S32 iwidth)
{
    HVX_Vector v_scale_xy  = Q6_V_vsplat_R(scale_xy);
    HVX_Vector v_scale_add = Q6_V_vsplat_R(scale_xy >> 1);

    HVX_Vector vu32_sum0 = Q6_V_vzero();
    HVX_Vector vu32_sum1 = Q6_V_vzero();
    for (MI_S32 sx = 0; sx < int_scale_x; sx++)
    {
        HVX_Vector v_one  = Q6_V_vsplat_R(sx << 2);
        HVX_Vector v_idx0 = Q6_Vuw_vadd_VuwVuw_sat(vu32_offset0, v_one);
        HVX_Vector v_idx1 = Q6_Vuw_vadd_VuwVuw_sat(vu32_offset1, v_one);

        Q6_vgather_ARMVw(vu32_gather0_ptr, (unsigned int)(src_buffer + ch * iwidth), (iwidth << 2) -1, v_idx0);
        Q6_vgather_ARMVw(vu32_gather1_ptr, (unsigned int)(src_buffer + ch * iwidth), (iwidth << 2) -1, v_idx1);

        HVX_Vector vu32_gather0 = *vu32_gather0_ptr;
        HVX_Vector vu32_gather1 = *vu32_gather1_ptr;

        vu32_sum0 = Q6_Vuw_vadd_VuwVuw_sat(vu32_sum0, vu32_gather0);
        vu32_sum1 = Q6_Vuw_vadd_VuwVuw_sat(vu32_sum1, vu32_gather1);
    }

    HVX_Vector vu16_div_sum0 = Q6_Vuw_vdiv16_VuwVuw(Q6_Vuw_vadd_VuwVuw_sat(vu32_sum0, v_scale_add), v_scale_xy);
    HVX_Vector vu16_div_sum1 = Q6_Vuw_vdiv16_VuwVuw(Q6_Vuw_vadd_VuwVuw_sat(vu32_sum1, v_scale_add), v_scale_xy);
    vu16_result = Q6_Vh_vdeal_Vh(Q6_Vh_vshuffe_VhVh(vu16_div_sum1, vu16_div_sum0));
}

template <typename Tp, typename Rt, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaFastDnHCore(HVX_Vector &vs16_result, HVX_Vector &vu32_offset0, HVX_Vector &vu32_offset1, Rt *src_buffer, HVX_Vector *vs32_gather0_ptr,
                                                 HVX_Vector *vs32_gather1_ptr, MI_S32 int_scale_x, MI_S32 scale_xy, MI_S32 ch, MI_S32 iwidth)
{
    HVX_Vector v_scale_xy  = Q6_V_vsplat_R(scale_xy);
    HVX_Vector v_scale_add = Q6_V_vsplat_R(scale_xy >> 1);

    HVX_Vector vs32_sum0 = Q6_V_vzero();
    HVX_Vector vs32_sum1 = Q6_V_vzero();
    for (MI_S32 sx = 0; sx < int_scale_x; sx++)
    {
        HVX_Vector v_one  = Q6_V_vsplat_R(sx << 2);
        HVX_Vector v_idx0 = Q6_Vuw_vadd_VuwVuw_sat(vu32_offset0, v_one);
        HVX_Vector v_idx1 = Q6_Vuw_vadd_VuwVuw_sat(vu32_offset1, v_one);

        Q6_vgather_ARMVw(vs32_gather0_ptr, (unsigned int)(src_buffer + ch * iwidth), (iwidth << 2) -1, v_idx0);
        Q6_vgather_ARMVw(vs32_gather1_ptr, (unsigned int)(src_buffer + ch * iwidth), (iwidth << 2) -1, v_idx1);

        HVX_Vector vs32_gather0 = *vs32_gather0_ptr;
        HVX_Vector vs32_gather1 = *vs32_gather1_ptr;

        vs32_sum0 = Q6_Vw_vadd_VwVw_sat(vs32_sum0, vs32_gather0);
        vs32_sum1 = Q6_Vw_vadd_VwVw_sat(vs32_sum1, vs32_gather1);
    }

    HVX_Vector vs16_div_sum0 = Q6_Vw_vdiv16_VwVw(Q6_Vw_vadd_VwVw_sat(vs32_sum0, v_scale_add), v_scale_xy);
    HVX_Vector vs16_div_sum1 = Q6_Vw_vdiv16_VwVw(Q6_Vw_vadd_VwVw_sat(vs32_sum1, v_scale_add), v_scale_xy);
    vs16_result = Q6_Vh_vdeal_Vh(Q6_Vh_vshuffe_VhVh(vs16_div_sum1, vs16_div_sum0));
}

template<typename Tp, MI_S32 C>
static AURA_VOID ResizeAreaDnX2Row(const Tp *src_c, const Tp *src_n0, Tp *dst_row, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align = width & (-elem_counts);
    MI_S32 remain = width - width_align;
    MI_S32 i = 0;

    auto resize_area_downx2_func = [&]()
    {
        MVType mv_c_x0_src, mv_c_x1_src, mv_n0_x0_src, mv_n0_x1_src, mv_result;

        vload(src_c  + i * 2 * C, mv_c_x0_src);
        vload(src_c  + (i * 2 + 1 * elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + i * 2 * C, mv_n0_x0_src);
        vload(src_n0 + (i * 2 + 1 * elem_counts) * C, mv_n0_x1_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnX2Core<Tp>(mv_result.val[ch], mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch]);
        }

        vstore((dst_row + C * i), mv_result);
    };

    for (; i < width_align; i += elem_counts)
    {
        resize_area_downx2_func();
    }

    if (remain)
    {
        i = width - elem_counts;
        resize_area_downx2_func();
    }

    return;
}

template <typename Tp, MI_S32 C>
static AURA_VOID ResizeAreaDnX4Row(const Tp *src_c, const Tp *src_n0, const Tp *src_n1, const Tp *src_n2, Tp *dst_row, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    MI_S32 elem_counts   = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align64 = width & (-elem_counts);
    MI_S32 remain = width - width_align64;

    MI_S32 i = 0;

    auto resize_area_downx4_func = [&]()
    {
        MVType mv_c_x0_src,  mv_c_x1_src,  mv_c_x2_src,  mv_c_x3_src,
               mv_n0_x0_src, mv_n0_x1_src, mv_n0_x2_src, mv_n0_x3_src,
               mv_n1_x0_src, mv_n1_x1_src, mv_n1_x2_src, mv_n1_x3_src,
               mv_n2_x0_src, mv_n2_x1_src, mv_n2_x2_src, mv_n2_x3_src,
               mv_result;

        vload(src_c  + i * 4 * C, mv_c_x0_src);
        vload(src_c  + (i * 4 + 1 * elem_counts) * C, mv_c_x1_src);
        vload(src_c  + (i * 4 + 2 * elem_counts) * C, mv_c_x2_src);
        vload(src_c  + (i * 4 + 3 * elem_counts) * C, mv_c_x3_src);

        vload(src_n0 + i * 4 * C, mv_n0_x0_src);
        vload(src_n0 + (i * 4 + 1 * elem_counts) * C, mv_n0_x1_src);
        vload(src_n0 + (i * 4 + 2 * elem_counts) * C, mv_n0_x2_src);
        vload(src_n0 + (i * 4 + 3 * elem_counts) * C, mv_n0_x3_src);

        vload(src_n1 + i * 4 * C, mv_n1_x0_src);
        vload(src_n1 + (i * 4 + 1 * elem_counts) * C, mv_n1_x1_src);
        vload(src_n1 + (i * 4 + 2 * elem_counts) * C, mv_n1_x2_src);
        vload(src_n1 + (i * 4 + 3 * elem_counts) * C, mv_n1_x3_src);

        vload(src_n2 + i * 4 * C, mv_n2_x0_src);
        vload(src_n2 + (i * 4 + 1 * elem_counts) * C, mv_n2_x1_src);
        vload(src_n2 + (i * 4 + 2 * elem_counts) * C, mv_n2_x2_src);
        vload(src_n2 + (i * 4 + 3 * elem_counts) * C, mv_n2_x3_src);

        HVX_Vector v_x0_sum, v_x1_sum, v_x2_sum, v_x3_sum;

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnX4Core<Tp>(v_x0_sum, v_x1_sum, mv_c_x0_src.val[ch], mv_n0_x0_src.val[ch], mv_n1_x0_src.val[ch], mv_n2_x0_src.val[ch],
                                   mv_c_x1_src.val[ch], mv_n0_x1_src.val[ch], mv_n1_x1_src.val[ch], mv_n2_x1_src.val[ch]);
            ResizeAreaDnX4Core<Tp>(v_x2_sum, v_x3_sum, mv_c_x2_src.val[ch], mv_n0_x2_src.val[ch], mv_n1_x2_src.val[ch], mv_n2_x2_src.val[ch],
                                   mv_c_x3_src.val[ch], mv_n0_x3_src.val[ch], mv_n1_x3_src.val[ch], mv_n2_x3_src.val[ch]);
            ResizeAreaDnX4SumCore<Tp>(mv_result.val[ch], v_x0_sum, v_x1_sum, v_x2_sum, v_x3_sum);
        }

        vstore((dst_row + C * i), mv_result);
    };

    for (; i < width_align64; i += elem_counts)
    {
        resize_area_downx4_func();
    }

    if (remain)
    {
        i = width - elem_counts;
        resize_area_downx4_func();
    }

    return;
}

template <typename Tp, MI_S32 C>
static AURA_VOID ResizeAreaDownX2Y4Row(const Tp *src_c, const Tp *src_n0, const Tp *src_n1, const Tp *src_n2, Tp *dst_row, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align64 = width & (-elem_counts);
    MI_S32 remain = width - width_align64;

    MI_S32 i = 0;

    auto resize_area_downx2y4_func = [&]()
    {
        MVType mv_c_x0_src,  mv_c_x1_src,
               mv_n0_x0_src, mv_n0_x1_src,
               mv_n1_x0_src, mv_n1_x1_src,
               mv_n2_x0_src, mv_n2_x1_src,
               mv_result;

        vload(src_c  + i * 2 * C, mv_c_x0_src);
        vload(src_c  + (i * 2 + 1 * elem_counts) * C, mv_c_x1_src);

        vload(src_n0 + i * 2 * C, mv_n0_x0_src);
        vload(src_n0 + (i * 2 + 1 * elem_counts) * C, mv_n0_x1_src);

        vload(src_n1 + i * 2 * C, mv_n1_x0_src);
        vload(src_n1 + (i * 2 + 1 * elem_counts) * C, mv_n1_x1_src);

        vload(src_n2 + i * 2 * C, mv_n2_x0_src);
        vload(src_n2 + (i * 2 + 1 * elem_counts) * C, mv_n2_x1_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnX2Y4Core<Tp>(mv_result.val[ch], mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch],
                                     mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch]);
        }

        vstore((dst_row + C * i), mv_result);
    };

    for (; i < width_align64; i += elem_counts)
    {
        resize_area_downx2y4_func();
    }

    if (remain)
    {
        i = width - elem_counts;
        resize_area_downx2y4_func();
    }

    return;
}

template <typename Tp, MI_S32 C>
static AURA_VOID ResizeAreaDownX4Y2Row(const Tp *src_c, const Tp *src_n0, Tp *dst_row, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align64 = width & (-elem_counts);
    MI_S32 remain = width - width_align64;

    MI_S32 i = 0;

    auto resize_area_downx4y2_func = [&]()
    {
        MVType mv_c_x0_src,  mv_c_x1_src,  mv_c_x2_src,  mv_c_x3_src,
               mv_n0_x0_src, mv_n0_x1_src, mv_n0_x2_src, mv_n0_x3_src,
               mv_result;

        vload(src_c  + i * 4 * C, mv_c_x0_src);
        vload(src_c  + (i * 4 + 1 * elem_counts) * C, mv_c_x1_src);
        vload(src_c  + (i * 4 + 2 * elem_counts) * C, mv_c_x2_src);
        vload(src_c  + (i * 4 + 3 * elem_counts) * C, mv_c_x3_src);

        vload(src_n0 + i * 4 * C, mv_n0_x0_src);
        vload(src_n0 + (i * 4 + 1 * elem_counts) * C, mv_n0_x1_src);
        vload(src_n0 + (i * 4 + 2 * elem_counts) * C, mv_n0_x2_src);
        vload(src_n0 + (i * 4 + 3 * elem_counts) * C, mv_n0_x3_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnX4Y2Core<Tp>(mv_result.val[ch], mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], mv_c_x2_src.val[ch], mv_c_x3_src.val[ch],
                                     mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], mv_n0_x2_src.val[ch], mv_n0_x3_src.val[ch]);
        }

        vstore((dst_row + C * i), mv_result);
    };

    for (; i < width_align64; i += elem_counts)
    {
        resize_area_downx4y2_func();
    }

    if (remain)
    {
        i = width - elem_counts;
        resize_area_downx4y2_func();
    }

    return;
}

template <typename Tp, MI_S32 C>
static AURA_VOID ResizeAreaUpX2Row(const Tp *src_c, Tp *dst_c, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align = width & (-elem_counts);

    MVType mv_c_src, mv_c_dst, mv_r_dst;

    for (MI_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_c + x * C, mv_c_src);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaUpX2Core<Tp>(mv_c_src.val[ch], mv_c_dst.val[ch], mv_r_dst.val[ch]);
        }

        vstore(dst_c + x * 2 * C, mv_c_dst);
        vstore(dst_c + (x * 2 + elem_counts) * C, mv_r_dst);
    }

    if (width > width_align)
    {
        MI_S32 last_x = width - elem_counts;

        vload(src_c + last_x * C, mv_c_src);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaUpX2Core<Tp>(mv_c_src.val[ch], mv_c_dst.val[ch], mv_r_dst.val[ch]);
        }

        vstore(dst_c + last_x * 2 * C, mv_c_dst);
        vstore(dst_c + (last_x * 2 + elem_counts) * C, mv_r_dst);
    }

    return;
}

template <typename Tp, MI_S32 C>
static AURA_VOID ResizeAreaUpX4Row(const Tp *src_c, Tp *dst_c, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 width_align = width & (-elem_counts);

    MVType mv_c_src, mv_c_dst, mv_r0_dst, mv_r1_dst, mv_r2_dst;

    for (MI_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_c + x * C, mv_c_src);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaUpX4Core<Tp>(mv_c_src.val[ch], mv_c_dst.val[ch], mv_r0_dst.val[ch], mv_r1_dst.val[ch], mv_r2_dst.val[ch]);
        }

        vstore(dst_c + x * 4 * C,                        mv_c_dst);
        vstore(dst_c + (x * 4 + elem_counts)        * C, mv_r0_dst);
        vstore(dst_c + (x * 4 + (elem_counts << 1)) * C, mv_r1_dst);
        vstore(dst_c + (x * 4 + (elem_counts *  3)) * C, mv_r2_dst);
    };

    if (width > width_align)
    {
        MI_S32 last_x = width - elem_counts;

        vload(src_c + last_x * C, mv_c_src);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaUpX4Core<Tp>(mv_c_src.val[ch], mv_c_dst.val[ch], mv_r0_dst.val[ch], mv_r1_dst.val[ch], mv_r2_dst.val[ch]);
        }

        vstore(dst_c + last_x * 4 * C,                        mv_c_dst);
        vstore(dst_c + (last_x * 4 + elem_counts)        * C, mv_r0_dst);
        vstore(dst_c + (last_x * 4 + (elem_counts << 1)) * C, mv_r1_dst);
        vstore(dst_c + (last_x * 4 + (elem_counts *  3)) * C, mv_r2_dst);
    }

    return;
}

template <typename Tp, MI_S32 C, typename Rt>
static AURA_VOID ResizeAreaFastDnRow(Tp *src_row, Rt *src_buffer, Rt *xofs, Rt *gather_buffer, MI_S32 iwidth, MI_S32 istride,
                                   MI_S32 owidth, Tp *dst_c, MI_S32 int_scale_x, MI_S32 int_scale_y)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 iwidth_align = iwidth & (-elem_counts);
    MI_S32 owidth_align = owidth & (-elem_counts);
    MI_S32 iremain      = iwidth - iwidth_align;
    MI_S32 oremain      = owidth - owidth_align;
    MI_S32 half_elem    = elem_counts >> 1;
    MI_S32 istep        = istride / sizeof(Tp);
    MI_S32 scale_xy     = int_scale_x * int_scale_y;

    MVType mv_src;
    MVType mv_result;
    MWType mw_src_row_sum;
    HVX_Vector *xoffset_ptr = (HVX_Vector *)xofs;
    HVX_Vector *gather0_ptr = (HVX_Vector *)gather_buffer;
    HVX_Vector *gather1_ptr = (HVX_Vector *)(gather_buffer + half_elem);

    #pragma unroll(C)
    for (MI_S32 ch = 0; ch < C; ch++)
    {
        mw_src_row_sum.val[ch] = Q6_W_vzero();
    }

    MI_S32 i = 0;
    for (; i < iwidth_align; i += elem_counts)
    {
        Rt *src_buffer_ptr = (Rt *)src_buffer + i;

        for (MI_S32 sy = 0; sy < int_scale_y; sy++)
        {
            vload(src_row + i * C + sy * istep, mv_src);
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                ResizeAreaFastDnVCore<Tp>(mw_src_row_sum.val[ch], mv_src.val[ch]);
            }
        }

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaFastDnVCore<Tp>(mw_src_row_sum.val[ch]);
            vmemu(src_buffer_ptr + ch * iwidth)             = Q6_V_lo_W(mw_src_row_sum.val[ch]);
            vmemu(src_buffer_ptr + ch * iwidth + half_elem) = Q6_V_hi_W(mw_src_row_sum.val[ch]);
            mw_src_row_sum.val[ch] = Q6_W_vzero();
        }
    }

    if (iremain)
    {
        i = iwidth - elem_counts;
        Rt *src_buffer_ptr = (Rt *)src_buffer + i;
        for (MI_S32 sy = 0; sy < int_scale_y; sy++)
        {
            vload(src_row + i * C + sy * istep, mv_src);
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                ResizeAreaFastDnVCore<Tp>(mw_src_row_sum.val[ch], mv_src.val[ch]);
            }
        }

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaFastDnVCore<Tp>(mw_src_row_sum.val[ch]);
            vmemu(src_buffer_ptr + ch * iwidth) = Q6_V_lo_W(mw_src_row_sum.val[ch]);
            vmemu(src_buffer_ptr + ch * iwidth + half_elem) = Q6_V_hi_W(mw_src_row_sum.val[ch]);
            mw_src_row_sum.val[ch] = Q6_W_vzero();
        }
    }

    HVX_Vector v_offset0, v_offset1;
    MI_S32 j = 0;
    for (; j < owidth_align; j += elem_counts)
    {
        v_offset0 = *xoffset_ptr++;
        v_offset1 = *xoffset_ptr++;

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaFastDnHCore<Tp, Rt>(mv_result.val[ch], v_offset0, v_offset1, src_buffer, gather0_ptr, gather1_ptr, int_scale_x, scale_xy, ch, iwidth);
        }
        vstore((dst_c + C * j), mv_result);
    }

    if (oremain)
    {
        j = owidth - elem_counts;
        HVX_Vector v_offset0 = vmemu((xofs + j));
        HVX_Vector v_offset1 = vmemu((xofs + j + half_elem));

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaFastDnHCore<Tp, Rt>(mv_result.val[ch], v_offset0, v_offset1, src_buffer, gather0_ptr, gather1_ptr, int_scale_x, scale_xy, ch, iwidth);
        }
        vstore((dst_c + C * j), mv_result);
    }

    return;
}

template <typename Tp, MI_S32 C>
static Status ResizeAreaDnX2HvxImpl(const Mat &src, Mat &dst, MI_S32 start_height, MI_S32 end_height)
{
    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 height  = dst.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;

    MI_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 2, 0);
    for (MI_S32 y = start_height; y < end_height; y++)
    {
        if ((y + 2) < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y * 2 + 2)), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(y * 2);
        const Tp *src_n0 = src.Ptr<Tp>(y * 2 + 1);
        Tp *dst_row      = dst.Ptr<Tp>(y);

        ResizeAreaDnX2Row<Tp, C>(src_c, src_n0, dst_row, width);
    }

    return Status::OK;
}

template <typename Tp, MI_S32 C>
static Status ResizeAreaDnX4HvxImpl(const Mat &src, Mat &dst, MI_S32 start_height, MI_S32 end_height)
{
    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 height  = dst.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;

    MI_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 4, 0);
    for (MI_S32 y = start_height; y < end_height; y++)
    {
        if ((y + 4) < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y * 4 + 4)), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(y * 4);
        const Tp *src_n0 = src.Ptr<Tp>(y * 4 + 1);
        const Tp *src_n1 = src.Ptr<Tp>(y * 4 + 2);
        const Tp *src_n2 = src.Ptr<Tp>(y * 4 + 3);
        Tp *dst_row      = dst.Ptr<Tp>(y);

        ResizeAreaDnX4Row<Tp, C>(src_c, src_n0, src_n1, src_n2, dst_row, width);
    }

    return Status::OK;
}

template <typename Tp, MI_S32 C>
static Status ResizeAreaDnX2Y4HvxImpl(const Mat &src, Mat &dst, MI_S32 start_height, MI_S32 end_height)
{
    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 height  = dst.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;

    MI_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 4, 0);
    for (MI_S32 y = start_height; y < end_height; y++)
    {
        if ((y + 4) < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y * 4 + 4)), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(y * 4);
        const Tp *src_n0 = src.Ptr<Tp>(y * 4 + 1);
        const Tp *src_n1 = src.Ptr<Tp>(y * 4 + 2);
        const Tp *src_n2 = src.Ptr<Tp>(y * 4 + 3);
        Tp *dst_row      = dst.Ptr<Tp>(y);

        ResizeAreaDownX2Y4Row<Tp, C>(src_c, src_n0, src_n1, src_n2, dst_row, width);
    }

    return Status::OK;
}

template <typename Tp, MI_S32 C>
static Status ResizeAreaDnX4Y2HvxImpl(const Mat &src, Mat &dst, MI_S32 start_height, MI_S32 end_height)
{
    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 height  = dst.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;

    MI_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 2, 0);
    for (MI_S32 y = start_height; y < end_height; y++)
    {
        if ((y + 2) < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y * 2 + 2)), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(y * 2);
        const Tp *src_n0 = src.Ptr<Tp>(y * 2 + 1);
        Tp *dst_row      = dst.Ptr<Tp>(y);

        ResizeAreaDownX4Y2Row<Tp, C>(src_c, src_n0, dst_row, width);
    }

    return Status::OK;
}

template<typename Tp, MI_S32 C>
static Status ResizeAreaUpX2HvxImpl(const Mat &src, Mat &dst, MI_S32 start_height, MI_S32 end_height)
{
    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;
    MI_S32 ostride = dst.GetStrides().m_width;

    MI_U64 l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 1, 0);
    for (MI_S32 y = start_height; y < end_height; y++)
    {
        if ((y + 1) < iheight)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 1)), l2fetch_param);
        }

        const Tp *src_c = src.Ptr<Tp>(y);
        Tp *dst_c = dst.Ptr<Tp>(y * 2);
        Tp *dst_n = dst.Ptr<Tp>(y * 2 + 1);

        ResizeAreaUpX2Row<Tp, C>(src_c, dst_c, iwidth);
        AuraMemCopy(dst_n, dst_c, ostride);
    }

    return Status::OK;
}

template <typename Tp, MI_S32 C>
static Status ResizeAreaUpX4HvxImpl(const Mat &src, Mat &dst, MI_S32 start_height, MI_S32 end_height)
{
    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;
    MI_S32 ostride = dst.GetStrides().m_width;

    MI_U64 l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 1, 0);
    for (MI_S32 y = start_height; y < end_height; y++)
    {
        if ((y + 1) < iheight)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 1)), l2fetch_param);
        }

        const Tp *src_c = src.Ptr<Tp>(y);
        Tp *dst_c  = dst.Ptr<Tp>(y * 4);
        Tp *dst_n0 = dst.Ptr<Tp>(y * 4 + 1);
        Tp *dst_n1 = dst.Ptr<Tp>(y * 4 + 2);
        Tp *dst_n2 = dst.Ptr<Tp>(y * 4 + 3);

        ResizeAreaUpX4Row<Tp, C>(src_c, dst_c, iwidth);
        AuraMemCopy(dst_n0, dst_c, ostride);
        AuraMemCopy(dst_n1, dst_c, ostride);
        AuraMemCopy(dst_n2, dst_c, ostride);
    }

    return Status::OK;
}

template <typename Tp, MI_S32 C, typename Rt>
static Status ResizeAreaFastDnHvxImpl(const Mat &src, Mat &dst, ResizeAreaVtcmBuffer *vctm_buffer, MI_S32 thread_num, MI_S32 int_scale_x, MI_S32 int_scale_y, MI_S32 start_height, MI_S32 end_height)
{
    MI_S32 iwidth    = src.GetSizes().m_width;
    MI_S32 istride   = src.GetStrides().m_width;
    MI_S32 owidth    = dst.GetSizes().m_width;
    MI_S32 oheight   = dst.GetSizes().m_height;
    MI_S32 thread_id = SaturateCast<MI_S32>(static_cast<MI_F32>(start_height) * thread_num / oheight);

    Rt *xofs          = reinterpret_cast<Rt *>(vctm_buffer->xofs);
    Rt *yofs          = reinterpret_cast<Rt *>(vctm_buffer->yofs);
    Rt *src_buffer    = reinterpret_cast<Rt *>(vctm_buffer->src_buffer + C * vctm_buffer->src_buffer_pitch * thread_id);
    Rt *gather_buffer = reinterpret_cast<Rt *>(vctm_buffer->gather_buffer + C * (AURA_HVLEN << 2) * thread_id);

    MI_U64 l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, int_scale_y, 0);
    for (MI_S32 y = start_height; y < end_height; y++)
    {
        if ((y + int_scale_y) < oheight)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(yofs[y + 1])), l2fetch_param);
        }

        Tp *src_c = (Tp *)src.Ptr<Tp>(yofs[y]);
        Tp *dst_c = (Tp *)dst.Ptr<Tp>(y);

        ResizeAreaFastDnRow<Tp, C, Rt>(src_c, src_buffer, xofs, gather_buffer, iwidth, istride, owidth, dst_c, int_scale_x, int_scale_y);
    }

    return Status::OK;
}

template <typename Tp, MI_S32 C>
static Status ResizeAreaFastHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    Status ret         = Status::ERROR;
    MI_F32 scale_x     = static_cast<MI_F32>(src.GetSizes().m_width ) / dst.GetSizes().m_width;
    MI_F32 scale_y     = static_cast<MI_F32>(src.GetSizes().m_height) / dst.GetSizes().m_height;
    MI_S32 int_scale_x = src.GetSizes().m_width / dst.GetSizes().m_width;
    MI_S32 int_scale_y = src.GetSizes().m_height / dst.GetSizes().m_height;

    if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 2.f))
    {
        ret = wp->ParallelFor((MI_S32)0, dst.GetSizes().m_height, ResizeAreaDnX2HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaDnX2HvxImpl failed");
        }
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 4.f))
    {
        ret = wp->ParallelFor((MI_S32)0, dst.GetSizes().m_height, ResizeAreaDnX4HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaDnX4HvxImpl failed");
        }
    }
    else if (NearlyEqual(scale_x, 2.f) && NearlyEqual(scale_y, 4.f))
    {
        ret = wp->ParallelFor((MI_S32)0, dst.GetSizes().m_height, ResizeAreaDnX2Y4HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaDnX2Y4HvxImpl failed");
        }
    }
    else if (NearlyEqual(scale_x, 4.f) && NearlyEqual(scale_y, 2.f))
    {
        ret = wp->ParallelFor((MI_S32)0, dst.GetSizes().m_height, ResizeAreaDnX4Y2HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaDnX4Y2HvxImpl failed");
        }
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 0.5f))
    {
        ret = wp->ParallelFor((MI_S32)0, src.GetSizes().m_height, ResizeAreaUpX2HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaUpX2HvxImpl failed");
        }
    }
    else if (NearlyEqual(scale_x, scale_y) && NearlyEqual(scale_x, 0.25f))
    {
        ret = wp->ParallelFor((MI_S32)0, src.GetSizes().m_height, ResizeAreaUpX4HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaUpX4HvxImpl failed");
        }
    }
    else if (NearlyEqual(scale_x, int_scale_x) && NearlyEqual(scale_y, int_scale_y))
    {
        MI_S32 iwidth  = src.GetSizes().m_width;
        MI_S32 owidth  = dst.GetSizes().m_width;
        MI_S32 oheight = dst.GetSizes().m_height;
        MI_S32 channel = dst.GetSizes().m_channel;
        MI_S32 thread_num = wp->GetComputeThreadNum();

        using ResizeAreaPromoteType = typename ResizeAreaHvxTraits<Tp>::PromoteType;

        MI_S32 xofs_buffer_size    = AURA_ALIGN(owidth  * sizeof(ResizeAreaPromoteType), AURA_HVLEN);
        MI_S32 yofs_buffer_size    = AURA_ALIGN(oheight * sizeof(ResizeAreaPromoteType), AURA_HVLEN);
        MI_S32 row_buffer_sizes    = AURA_ALIGN(iwidth  * sizeof(ResizeAreaPromoteType), AURA_HVLEN) * thread_num * channel;
        MI_S32 gather_buffer_sizes = (AURA_HVLEN << 2) * channel * thread_num;
        MI_S32 total_buffer_sizes  = xofs_buffer_size + yofs_buffer_size + row_buffer_sizes + gather_buffer_sizes;

        MI_U8 *area_fast_buffer = static_cast<MI_U8*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_VTCM, total_buffer_sizes, AURA_HVLEN));
        if (MI_NULL == area_fast_buffer)
        {
            AURA_ADD_ERROR_STRING(ctx, "alloc vtcm memory failed");
            AURA_FREE(ctx, area_fast_buffer);
            return Status::ABORT;
        }

        struct ResizeAreaVtcmBuffer vctm_buffer;
        vctm_buffer.xofs             = (MI_U8 *)area_fast_buffer;
        vctm_buffer.yofs             = vctm_buffer.xofs + xofs_buffer_size;
        vctm_buffer.src_buffer       = vctm_buffer.yofs + yofs_buffer_size;
        vctm_buffer.src_buffer_pitch = row_buffer_sizes / (thread_num * channel);
        vctm_buffer.gather_buffer    = vctm_buffer.src_buffer  + row_buffer_sizes;

        ret = GetResizeAreaFastDnOffset<Tp, ResizeAreaPromoteType>(owidth, oheight, int_scale_x, int_scale_y, &vctm_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetResizeAreaFastDnOffset failed");
            AURA_FREE(ctx, area_fast_buffer);
            return ret;
        }

        ret = wp->ParallelFor((MI_S32)0, dst.GetSizes().m_height, ResizeAreaFastDnHvxImpl<Tp, C, ResizeAreaPromoteType>, std::cref(src), std::ref(dst), &vctm_buffer, thread_num, int_scale_x, int_scale_y);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastDnHvxImpl failed");
        }

        AURA_FREE(ctx, area_fast_buffer);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "not x2, x4 or int scale");
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status ResizeAreaFastHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    MI_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = ResizeAreaFastHvxHelper<Tp, 1>(ctx, src, dst);
            break;
        }

        case 2:
        {
            ret = ResizeAreaFastHvxHelper<Tp, 2>(ctx, src, dst);
            break;
        }

        case 3:
        {
            ret = ResizeAreaFastHvxHelper<Tp, 3>(ctx, src, dst);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "channel number is not supported.");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeAreaFastHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeAreaFastHvxHelper<MI_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastHvxHelper run failed, type: MI_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeAreaFastHvxHelper<MI_S8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastHvxHelper run failed, type: MI_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeAreaFastHvxHelper<MI_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastHvxHelper run failed, type: MI_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeAreaFastHvxHelper<MI_S16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaFastHvxHelper run failed, type: MI_S16");
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
