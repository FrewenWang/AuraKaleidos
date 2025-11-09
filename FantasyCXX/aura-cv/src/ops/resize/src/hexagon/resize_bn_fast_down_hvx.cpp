#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"
namespace aura
{

// Tp = DT_U8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value, Status>::type
ResizeBnDownX2Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));
    DT_S32 i = 0;

    auto resize_downx2_func = [&]()
    {
        MVType mv_result, mv_c_x0_src, mv_c_x1_src, mv_n0_x0_src, mv_n0_x1_src;
        vload(src_c  + i * 2, mv_c_x0_src);
        vload(src_n0 + i * 2, mv_n0_x0_src);
        vload(src_c  + i * 2 + elem_counts, mv_c_x1_src);
        vload(src_n0 + i * 2 + elem_counts, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair wu16_sum0 = Q6_Wh_vadd_VubVub(mv_c_x0_src.val[ch],  mv_n0_x0_src.val[ch]);
            HVX_VectorPair wu16_sum1 = Q6_Wh_vadd_VubVub(mv_c_x1_src.val[ch], mv_n0_x1_src.val[ch]);
            HVX_Vector vu16_sum0     = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wu16_sum0), Q6_V_lo_W(wu16_sum0));
            HVX_Vector vu16_sum1     = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wu16_sum1), Q6_V_lo_W(wu16_sum1));

            mv_result.val[ch] = Q6_Vb_vdeal_Vb(Q6_Vub_vasr_VuhVuhR_rnd_sat(vu16_sum1, vu16_sum0, 2));
        }

        vstore(dst_c + i, mv_result);
    };

    for (; i < width_align; i += elem_counts)
    {
        resize_downx2_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts;
        resize_downx2_func();
    }

    return Status::OK;
}

// Tp = DT_S8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S8, Tp>::value, Status>::type
ResizeBnDownX2Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));
    DT_S32 i = 0;

    auto resize_downx2_func = [&]()
    {
        MVType mv_result, mv_c_x0_src, mv_c_x1_src, mv_n0_x0_src, mv_n0_x1_src;
        vload(src_c  + i * 2, mv_c_x0_src);
        vload(src_n0 + i * 2, mv_n0_x0_src);
        vload(src_c  + i * 2 + elem_counts, mv_c_x1_src);
        vload(src_n0 + i * 2 + elem_counts, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair ws16_c_c_src   = Q6_Wh_vsxt_Vb(mv_c_x0_src.val[ch]);
            HVX_VectorPair ws16_c_r0_src  = Q6_Wh_vsxt_Vb(mv_c_x1_src.val[ch]);
            HVX_VectorPair ws16_n0_c_src  = Q6_Wh_vsxt_Vb(mv_n0_x0_src.val[ch]);
            HVX_VectorPair ws16_n0_r0_src = Q6_Wh_vsxt_Vb(mv_n0_x1_src.val[ch]);

            HVX_Vector vs16_c_c_sum   = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_c_c_src),   Q6_V_hi_W(ws16_c_c_src));
            HVX_Vector vs16_c_r0_sum  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_c_r0_src),  Q6_V_hi_W(ws16_c_r0_src));
            HVX_Vector vs16_n0_c_sum  = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_n0_c_src),  Q6_V_hi_W(ws16_n0_c_src));
            HVX_Vector vs16_n0_r0_sum = Q6_Vh_vadd_VhVh(Q6_V_lo_W(ws16_n0_r0_src), Q6_V_hi_W(ws16_n0_r0_src));
            HVX_Vector vs16_c_sum     = Q6_Vh_vadd_VhVh(vs16_c_c_sum, vs16_n0_c_sum);
            HVX_Vector vs16_r0_sum    = Q6_Vh_vadd_VhVh(vs16_c_r0_sum, vs16_n0_r0_sum);

            mv_result.val[ch] = Q6_Vb_vdeal_Vb(Q6_Vb_vasr_VhVhR_rnd_sat(vs16_r0_sum, vs16_c_sum, 2));
        }

        vstore(dst_c + i, mv_result);
    };

    for (; i < width_align; i += elem_counts)
    {
        resize_downx2_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts;
        resize_downx2_func();
    }

    return Status::OK;
}

// Tp = DT_U16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value, Status>::type
ResizeBnDownX2Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));
    DT_S32 i = 0;

    auto resize_downx2_func = [&]()
    {
        MVType mv_result, mv_c_x0_src, mv_c_x1_src, mv_n0_x0_src, mv_n0_x1_src;
        vload(src_c  + i * 2, mv_c_x0_src);
        vload(src_n0 + i * 2, mv_n0_x0_src);
        vload(src_c  + i * 2 + elem_counts, mv_c_x1_src);
        vload(src_n0 + i * 2 + elem_counts, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_result  = Q6_W_vdeal_VVR(mv_c_x1_src.val[ch],  mv_c_x0_src.val[ch],  -2);
            HVX_VectorPair w_n0_result = Q6_W_vdeal_VVR(mv_n0_x1_src.val[ch], mv_n0_x0_src.val[ch], -2);
            HVX_VectorPair wu32_c_sum  = Q6_Ww_vadd_VuhVuh(Q6_V_lo_W(w_c_result),  Q6_V_hi_W(w_c_result));
            HVX_VectorPair wu32_n0_sum = Q6_Ww_vadd_VuhVuh(Q6_V_lo_W(w_n0_result), Q6_V_hi_W(w_n0_result));

            w_c_result        = Q6_Ww_vadd_WwWw(wu32_c_sum, wu32_n0_sum);
            mv_result.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_c_result), Q6_V_lo_W(w_c_result), 2);
        }

        vstore(dst_c + i, mv_result);
    };

    for (; i < width_align; i += elem_counts)
    {
        resize_downx2_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts;
        resize_downx2_func();
    }

    return Status::OK;
}

// Tp = DT_S16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S16, Tp>::value, Status>::type
ResizeBnDownX2Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));
    DT_S32 i = 0;

    auto resize_downx2_func = [&]()
    {
        MVType mv_result, mv_c_x0_src, mv_c_x1_src, mv_n0_x0_src, mv_n0_x1_src;
        vload(src_c  + i * 2, mv_c_x0_src);
        vload(src_n0 + i * 2, mv_n0_x0_src);
        vload(src_c  + i * 2 + elem_counts, mv_c_x1_src);
        vload(src_n0 + i * 2 + elem_counts, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_result  = Q6_W_vdeal_VVR(mv_c_x1_src.val[ch],  mv_c_x0_src.val[ch],  -2);
            HVX_VectorPair w_n0_result = Q6_W_vdeal_VVR(mv_n0_x1_src.val[ch], mv_n0_x0_src.val[ch], -2);
            HVX_VectorPair ws32_sum0   = Q6_Ww_vadd_VhVh(Q6_V_lo_W(w_c_result),  Q6_V_hi_W(w_c_result));
            HVX_VectorPair ws32_sum1   = Q6_Ww_vadd_VhVh(Q6_V_lo_W(w_n0_result), Q6_V_hi_W(w_n0_result));

            w_c_result        = Q6_Ww_vadd_WwWw(ws32_sum0, ws32_sum1);
            mv_result.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_c_result), Q6_V_lo_W(w_c_result), 2);
        }

        vstore(dst_c + i, mv_result);
    };

    for (; i < width_align; i += elem_counts)
    {
        resize_downx2_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts;
        resize_downx2_func();
    }

    return Status::OK;
}

// Tp = DT_U8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value, Status>::type
ResizeBnDownX4Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));
    DT_S32 i = 0;

    auto resize_downx4_func = [&]()
    {
        MVType mv_c_x0_src, mv_c_x1_src, mv_n0_x0_src, mv_n0_x1_src;
        MWType mw_c_result, mw_n0_result;

        const Tp *src_c_row  = src_c  + i * 4;
        const Tp *src_n0_row = src_n0 + i * 4;

        vload(src_c_row, mv_c_x0_src);
        vload(src_c_row + elem_counts, mv_c_x1_src);
        vload(src_n0_row, mv_n0_x0_src);
        vload(src_n0_row + elem_counts, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mw_c_result.val[ch]  = Q6_W_vdeal_VVR(mv_c_x1_src.val[ch],  mv_c_x0_src.val[ch],  -1);
            mw_n0_result.val[ch] = Q6_W_vdeal_VVR(mv_n0_x1_src.val[ch], mv_n0_x0_src.val[ch], -1);
        }

        vload(src_c_row  + elem_counts * 2, mv_c_x0_src);
        vload(src_c_row  + elem_counts * 3, mv_c_x1_src);
        vload(src_n0_row + elem_counts * 2, mv_n0_x0_src);
        vload(src_n0_row + elem_counts * 3, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_result  = Q6_W_vdeal_VVR(mv_c_x1_src.val[ch],  mv_c_x0_src.val[ch],  -1);
            HVX_VectorPair w_n0_result = Q6_W_vdeal_VVR(mv_n0_x1_src.val[ch], mv_n0_x0_src.val[ch], -1);

            HVX_VectorPair w_c_result_l  = Q6_W_vdeal_VVR(Q6_V_lo_W(w_c_result),  Q6_V_lo_W(mw_c_result.val[ch]),  -1);
            HVX_VectorPair w_c_result_h  = Q6_W_vdeal_VVR(Q6_V_hi_W(w_c_result),  Q6_V_hi_W(mw_c_result.val[ch]),  -1);
            HVX_VectorPair w_n0_result_l = Q6_W_vdeal_VVR(Q6_V_lo_W(w_n0_result), Q6_V_lo_W(mw_n0_result.val[ch]), -1);
            HVX_VectorPair w_n0_result_h = Q6_W_vdeal_VVR(Q6_V_hi_W(w_n0_result), Q6_V_hi_W(mw_n0_result.val[ch]), -1);

            w_c_result  = Q6_Wh_vadd_VubVub(Q6_V_lo_W(w_c_result_h),  Q6_V_hi_W(w_c_result_l));
            w_n0_result = Q6_Wh_vadd_VubVub(Q6_V_lo_W(w_n0_result_h), Q6_V_hi_W(w_n0_result_l));
            w_c_result  = Q6_Wh_vadd_WhWh(w_c_result, w_n0_result);

            mv_n0_x1_src.val[ch] = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_c_result), Q6_V_lo_W(w_c_result), 2);
        }

        vstore(dst_c + i, mv_n0_x1_src);
    };

    for (; i < width_align; i += elem_counts)
    {
        resize_downx4_func();
    }

    if (width_align < width)
    {
        i = (width - elem_counts);
        resize_downx4_func();
    }

    return Status::OK;
}

// Tp = DT_S8
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S8, Tp>::value, Status>::type
ResizeBnDownX4Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));
    DT_S32 i = 0;

    auto resize_downx4_func = [&]()
    {
        MVType mv_c_x0_src, mv_c_x1_src, mv_n0_x0_src, mv_n0_x1_src;
        MWType mw_c_x01_src, mw_n0_x01_src;

        const Tp *src_c_row  = src_c  + i * 4;
        const Tp *src_n0_row = src_n0 + i * 4;

        vload(src_c_row, mv_c_x0_src);
        vload(src_c_row + elem_counts, mv_c_x1_src);
        vload(src_n0_row, mv_n0_x0_src);
        vload(src_n0_row + elem_counts, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mw_c_x01_src.val[ch]  = Q6_W_vdeal_VVR(mv_c_x1_src.val[ch],  mv_c_x0_src.val[ch],  -2);
            mw_n0_x01_src.val[ch] = Q6_W_vdeal_VVR(mv_n0_x1_src.val[ch], mv_n0_x0_src.val[ch], -2);
        }

        vload(src_c_row  + elem_counts * 2, mv_c_x0_src);
        vload(src_c_row  + elem_counts * 3, mv_c_x1_src);
        vload(src_n0_row + elem_counts * 2, mv_n0_x0_src);
        vload(src_n0_row + elem_counts * 3, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_x23_src  = Q6_W_vdeal_VVR(mv_c_x1_src.val[ch],  mv_c_x0_src.val[ch],  -2);
            HVX_VectorPair w_n0_x23_src = Q6_W_vdeal_VVR(mv_n0_x1_src.val[ch], mv_n0_x0_src.val[ch], -2);

            HVX_VectorPair w_c_result_l  = Q6_W_vdeal_VVR(Q6_V_lo_W(w_c_x23_src),  Q6_V_lo_W(mw_c_x01_src.val[ch]),  -1);
            HVX_VectorPair w_c_result_h  = Q6_W_vdeal_VVR(Q6_V_hi_W(w_c_x23_src),  Q6_V_hi_W(mw_c_x01_src.val[ch]),  -1);
            HVX_VectorPair w_n0_result_l = Q6_W_vdeal_VVR(Q6_V_lo_W(w_n0_x23_src), Q6_V_lo_W(mw_n0_x01_src.val[ch]), -1);
            HVX_VectorPair w_n0_result_h = Q6_W_vdeal_VVR(Q6_V_hi_W(w_n0_x23_src), Q6_V_hi_W(mw_n0_x01_src.val[ch]), -1);

            w_c_result_l = Q6_Wh_vadd_WhWh(Q6_Wh_vunpack_Vb(Q6_V_lo_W(w_c_result_h)),  Q6_Wh_vunpack_Vb(Q6_V_hi_W(w_c_result_l)));
            w_c_result_h = Q6_Wh_vadd_WhWh(Q6_Wh_vunpack_Vb(Q6_V_lo_W(w_n0_result_h)), Q6_Wh_vunpack_Vb(Q6_V_hi_W(w_n0_result_l)));
            w_c_result_l = Q6_Wh_vadd_WhWh(w_c_result_l, w_c_result_h);

            mv_c_x1_src.val[ch]  = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(w_c_result_l), Q6_V_lo_W(w_c_result_l), 2);
            mv_n0_x1_src.val[ch] = Q6_Vb_vdeal_Vb(mv_c_x1_src.val[ch]);
        }

        vstore(dst_c + i, mv_n0_x1_src);
    };

    for (; i < width_align; i += elem_counts)
    {
        resize_downx4_func();
    }

    if (width_align < width)
    {
        i = (width - elem_counts);
        resize_downx4_func();
    }

    return Status::OK;
}

// Tp = DT_U16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value, Status>::type
ResizeBnDownX4Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));
    DT_S32 i = 0;

    auto resize_downx4_func = [&]()
    {
        MVType mv_c_x0_src, mv_c_x1_src, mv_n0_x0_src, mv_n0_x1_src;
        MWType mw_c_result, mw_n0_result;

        const Tp *src_c_row  = src_c  + i * 4;
        const Tp *src_n0_row = src_n0 + i * 4;

        vload(src_c_row, mv_c_x0_src);
        vload(src_c_row + elem_counts, mv_c_x1_src);
        vload(src_n0_row, mv_n0_x0_src);
        vload(src_n0_row + elem_counts, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mw_c_result.val[ch]  = Q6_W_vdeal_VVR(mv_c_x1_src.val[ch],  mv_c_x0_src.val[ch],  -2);
            mw_n0_result.val[ch] = Q6_W_vdeal_VVR(mv_n0_x1_src.val[ch], mv_n0_x0_src.val[ch], -2);
        }

        vload(src_c_row  + elem_counts * 2, mv_c_x0_src);
        vload(src_c_row  + elem_counts * 3, mv_c_x1_src);
        vload(src_n0_row + elem_counts * 2, mv_n0_x0_src);
        vload(src_n0_row + elem_counts * 3, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_result  = Q6_W_vdeal_VVR(mv_c_x1_src.val[ch],  mv_c_x0_src.val[ch],  -2);
            HVX_VectorPair w_n0_result = Q6_W_vdeal_VVR(mv_n0_x1_src.val[ch], mv_n0_x0_src.val[ch], -2);

            HVX_VectorPair w_c_result_l  = Q6_W_vdeal_VVR(Q6_V_lo_W(w_c_result),  Q6_V_lo_W(mw_c_result.val[ch]),  -2);
            HVX_VectorPair w_c_result_h  = Q6_W_vdeal_VVR(Q6_V_hi_W(w_c_result),  Q6_V_hi_W(mw_c_result.val[ch]),  -2);
            HVX_VectorPair w_n0_result_l = Q6_W_vdeal_VVR(Q6_V_lo_W(w_n0_result), Q6_V_lo_W(mw_n0_result.val[ch]), -2);
            HVX_VectorPair w_n0_result_h = Q6_W_vdeal_VVR(Q6_V_hi_W(w_n0_result), Q6_V_hi_W(mw_n0_result.val[ch]), -2);

            w_c_result  = Q6_Ww_vadd_VuhVuh(Q6_V_lo_W(w_c_result_h),  Q6_V_hi_W(w_c_result_l));
            w_n0_result = Q6_Ww_vadd_VuhVuh(Q6_V_lo_W(w_n0_result_h), Q6_V_hi_W(w_n0_result_l));
            w_c_result  = Q6_Ww_vadd_WwWw(w_c_result, w_n0_result);

            mv_n0_x1_src.val[ch] = Q6_Vuh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_c_result), Q6_V_lo_W(w_c_result), 2);
        }

        vstore(dst_c + i, mv_n0_x1_src);
    };

    for (; i < width_align; i += elem_counts)
    {
        resize_downx4_func();
    }

    if (width_align < width)
    {
        i = (width - elem_counts);
        resize_downx4_func();
    }

    return Status::OK;
}

// Tp = DT_S16
template<typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S16, Tp>::value, Status>::type
ResizeBnDownX4Row(const Tp *src_c, const Tp *src_n0, Tp *dst_c, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    DT_S32 width_align = width & (~(elem_counts - 1));
    DT_S32 i = 0;

    auto resize_downx4_func = [&]()
    {
        MVType mv_c_x0_src, mv_c_x1_src, mv_n0_x0_src, mv_n0_x1_src;
        MWType mw_c_result, mw_n0_result;

        const Tp *src_c_row  = src_c  + i * 4;
        const Tp *src_n0_row = src_n0 + i * 4;

        vload(src_c_row, mv_c_x0_src);
        vload(src_c_row + elem_counts, mv_c_x1_src);
        vload(src_n0_row, mv_n0_x0_src);
        vload(src_n0_row + elem_counts, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mw_c_result.val[ch]  = Q6_W_vdeal_VVR(mv_c_x1_src.val[ch],  mv_c_x0_src.val[ch],  -2);
            mw_n0_result.val[ch] = Q6_W_vdeal_VVR(mv_n0_x1_src.val[ch], mv_n0_x0_src.val[ch], -2);
        }

        vload(src_c_row  + elem_counts * 2, mv_c_x0_src);
        vload(src_c_row  + elem_counts * 3, mv_c_x1_src);
        vload(src_n0_row + elem_counts * 2, mv_n0_x0_src);
        vload(src_n0_row + elem_counts * 3, mv_n0_x1_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_result  = Q6_W_vdeal_VVR(mv_c_x1_src.val[ch],  mv_c_x0_src.val[ch],  -2);
            HVX_VectorPair w_n0_result = Q6_W_vdeal_VVR(mv_n0_x1_src.val[ch], mv_n0_x0_src.val[ch], -2);

            HVX_VectorPair w_c_result_l  = Q6_W_vdeal_VVR(Q6_V_lo_W(w_c_result),  Q6_V_lo_W(mw_c_result.val[ch]),  -2);
            HVX_VectorPair w_c_result_h  = Q6_W_vdeal_VVR(Q6_V_hi_W(w_c_result),  Q6_V_hi_W(mw_c_result.val[ch]),  -2);
            HVX_VectorPair w_n0_result_l = Q6_W_vdeal_VVR(Q6_V_lo_W(w_n0_result), Q6_V_lo_W(mw_n0_result.val[ch]), -2);
            HVX_VectorPair w_n0_result_h = Q6_W_vdeal_VVR(Q6_V_hi_W(w_n0_result), Q6_V_hi_W(mw_n0_result.val[ch]), -2);

            w_c_result  = Q6_Ww_vadd_VhVh(Q6_V_lo_W(w_c_result_h),  Q6_V_hi_W(w_c_result_l));
            w_n0_result = Q6_Ww_vadd_VhVh(Q6_V_lo_W(w_n0_result_h), Q6_V_hi_W(w_n0_result_l));
            w_c_result  = Q6_Ww_vadd_WwWw(w_c_result, w_n0_result);

            mv_n0_x1_src.val[ch] = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(w_c_result), Q6_V_lo_W(w_c_result), 2);
        }

        vstore(dst_c + i, mv_n0_x1_src);
    };

    for (; i < width_align; i += elem_counts)
    {
        resize_downx4_func();
    }

    if (width_align < width)
    {
        i = width - elem_counts;
        resize_downx4_func();
    }

    return Status::OK;
}

template<typename Tp, DT_S32 C>
static Status ResizeBnDnX2HvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    DT_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 2, 0);
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if (y + 1 < oheight)
        {
            L2Fetch(reinterpret_cast<DT_U64>(src.Ptr<Tp>(y * 2 + 2)), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(y * 2);
        const Tp *src_n0 = src_c + istride / sizeof(Tp);
        Tp *dst_c        = dst.Ptr<Tp>(y);
        ResizeBnDownX2Row<Tp, C>(src_c, src_n0, dst_c, owidth * C);
    }

    return Status::OK;
}

template<typename Tp, DT_S32 C>
static Status ResizeBnDnX4HvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    DT_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 2, 0);
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if (y + 1 < oheight)
        {
            L2Fetch(reinterpret_cast<DT_U64>(src.Ptr<Tp>(y * 4 + 5)), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(y * 4 + 1);
        const Tp *src_n0 = src_c + istride / sizeof(Tp);
        Tp *dst_c        = dst.Ptr<Tp>(y);
        ResizeBnDownX4Row<Tp, C>(src_c, src_n0, dst_c, owidth * C);
    }

    return Status::OK;
}

template<typename Tp, DT_S32 C>
static Status ResizeBnDnX2Y4HvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    DT_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 2, 0);
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if (y + 1 < oheight)
        {
            L2Fetch(reinterpret_cast<DT_U64>(src.Ptr<Tp>(y * 4 + 5)), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(y * 4 + 1);
        const Tp *src_n0 = src_c + istride / sizeof(Tp);
        Tp *dst_c        = dst.Ptr<Tp>(y);
        ResizeBnDownX2Row<Tp, C>(src_c, src_n0, dst_c, owidth * C);
    }

    return Status::OK;
}

template<typename Tp, DT_S32 C>
static Status ResizeBnDnX4Y2HvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    DT_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 2, 0);
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if (y + 1 < oheight)
        {
            L2Fetch(reinterpret_cast<DT_U64>(src.Ptr<Tp>(y * 2 + 2)), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(y * 2);
        const Tp *src_n0 = src_c + istride / sizeof(Tp);
        Tp *dst_c        = dst.Ptr<Tp>(y);
        ResizeBnDownX4Row<Tp, C>(src_c, src_n0, dst_c, owidth * C);
    }

    return Status::OK;
}

template<typename Tp, DT_S32 C>
static Status ResizeBnFastDnHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool fail");
        return Status::ERROR;
    }

    Status ret     = Status::ERROR;
    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 oheight = dst.GetSizes().m_height;

    if ((iwidth == 2 * owidth) && (iheight == 2 * oheight))
    {
        ret = wp->ParallelFor((DT_S32)0, dst.GetSizes().m_height, ResizeBnDnX2HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
    }
    else if ((iwidth == 4 * owidth) && (iheight == 4 * oheight))
    {
        ret = wp->ParallelFor((DT_S32)0, dst.GetSizes().m_height, ResizeBnDnX4HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
    }
    else if ((iwidth == 2 * owidth) && (iheight == 4 * oheight))
    {
        ret = wp->ParallelFor((DT_S32)0, dst.GetSizes().m_height, ResizeBnDnX2Y4HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
    }
    else if ((iwidth == 4 * owidth) && (iheight == 2 * oheight))
    {
        ret = wp->ParallelFor((DT_S32)0, dst.GetSizes().m_height, ResizeBnDnX4Y2HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "only support scale_x 0.5, 0.25, 2, 4");
        return Status::ERROR;
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status ResizeBnFastDnHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret     = Status::ERROR;
    DT_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = ResizeBnFastDnHvxHelper<Tp, 1>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastDnHvxHelper failed for c1");
            }
            break;
        }

        case 2:
        {
            ret = ResizeBnFastDnHvxHelper<Tp, 2>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastDnHvxHelper failed for c2");
            }
            break;
        }

        case 3:
        {
            ret = ResizeBnFastDnHvxHelper<Tp, 3>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastDnHvxHelper failed for c3");
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

Status ResizeBnFastDnHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeBnFastDnHvxHelper<DT_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastDnHvxHelper run failed, type: DT_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeBnFastDnHvxHelper<DT_S8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastDnHvxHelper run failed, type: DT_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeBnFastDnHvxHelper<DT_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastDnHvxHelper run failed, type: DT_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeBnFastDnHvxHelper<DT_S16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnFastDnHvxHelper run failed, type: DT_S16");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type is not supported.");
            ret = Status::ERROR;
        }
    }

    return Status::OK;
}

} // namespace aura