#include "laplacian_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

// using St = MI_U8
template <typename St, typename std::enable_if<std::is_same<St, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Laplacian1x1Core(const HVX_Vector &vu8_src_p_x1, const HVX_Vector &vu8_src_c_x0,
                                            const HVX_Vector &vu8_src_c_x1, const HVX_Vector &vu8_src_c_x2,
                                            const HVX_Vector &vu8_src_n_x1, HVX_Vector &vs16_dst_lo,
                                            HVX_Vector &vs16_dst_hi)
{
    HVX_VectorPair wu16_sum_pos, wu16_sum_neg, ws16_sum, ws16_dst;

    wu16_sum_pos = Q6_Wh_vadd_VubVub(Q6_V_vlalign_VVR(vu8_src_c_x1, vu8_src_c_x0, 1), Q6_V_valign_VVR(vu8_src_c_x2, vu8_src_c_x1, 1));
    wu16_sum_pos = Q6_Wh_vaddacc_WhVubVub(wu16_sum_pos, vu8_src_p_x1, vu8_src_n_x1);
    wu16_sum_neg = Q6_Wh_vmpy_VubRb(vu8_src_c_x1, 0x04040404);
    ws16_sum     = Q6_Wh_vsub_WhWh(wu16_sum_pos, wu16_sum_neg);
    ws16_dst     = Q6_W_vshuff_VVR(Q6_V_hi_W(ws16_sum), Q6_V_lo_W(ws16_sum), -2);
    vs16_dst_lo  = Q6_V_lo_W(ws16_dst);
    vs16_dst_hi  = Q6_V_hi_W(ws16_dst);
}

// using St = MI_U16
template <typename St, typename std::enable_if<std::is_same<St, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Laplacian1x1Core(const HVX_Vector &vu16_src_p_x1, const HVX_Vector &vu16_src_c_x0,
                                            const HVX_Vector &vu16_src_c_x1, const HVX_Vector &vu16_src_c_x2,
                                            const HVX_Vector &vu16_src_n_x1, HVX_Vector &vu16_dst)
{
    HVX_VectorPair wu32_sum_pos, wu32_sum_neg, ws32_sum;

    wu32_sum_pos = Q6_Ww_vadd_VuhVuh(Q6_V_vlalign_VVR(vu16_src_c_x1, vu16_src_c_x0, 2), Q6_V_valign_VVR(vu16_src_c_x2, vu16_src_c_x1, 2));
    wu32_sum_pos = Q6_Ww_vaddacc_WwVuhVuh(wu32_sum_pos, vu16_src_p_x1, vu16_src_n_x1);
    wu32_sum_neg = Q6_Ww_vmpy_VhRh(vu16_src_c_x1, 0x00040004);
    ws32_sum     = Q6_Ww_vsub_WwWw(wu32_sum_pos, wu32_sum_neg);
    ws32_sum     = Q6_W_vshuff_VVR(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum), -4);
    vu16_dst     = Q6_Vuh_vpack_VwVw_sat(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum));
}

// using St = MI_S16
template <typename St, typename std::enable_if<std::is_same<St, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Laplacian1x1Core(const HVX_Vector &vs16_src_p_x1, const HVX_Vector &vs16_src_c_x0,
                                            const HVX_Vector &vs16_src_c_x1, const HVX_Vector &vs16_src_c_x2,
                                            const HVX_Vector &vs16_src_n_x1, HVX_Vector &vs16_dst)
{
    HVX_VectorPair ws32_sum_pos, ws32_sum_neg, ws32_sum;

    ws32_sum_pos = Q6_Ww_vadd_VhVh(Q6_V_vlalign_VVR(vs16_src_c_x1, vs16_src_c_x0, 2), Q6_V_valign_VVR(vs16_src_c_x2, vs16_src_c_x1, 2));
    ws32_sum_pos = Q6_Ww_vaddacc_WwVhVh(ws32_sum_pos, vs16_src_p_x1, vs16_src_n_x1);
    ws32_sum_neg = Q6_Ww_vmpy_VhRh(vs16_src_c_x1, 0x00040004);
    ws32_sum     = Q6_Ww_vsub_WwWw(ws32_sum_pos, ws32_sum_neg);
    ws32_sum     = Q6_W_vshuff_VVR(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum), -4);
    vs16_dst     = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum));
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C,
          typename std::enable_if<std::is_same<St, MI_U8>::value>::type* = MI_NULL>
static AURA_VOID Laplacian1x1Row(const St *src_p, const St *src_c, const St *src_n, Dt *dst_c,
                               const std::vector<St> &border_value, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    constexpr MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(St);
    MI_S32 back_offset = width - ELEM_COUNTS;

    MVType mvu8_src_p_x1, mvu8_src_p_x2;
    MVType mvu8_src_c_x0, mvu8_src_c_x1, mvu8_src_c_x2;
    MVType mvu8_src_n_x1, mvu8_src_n_x2;
    MVType mvs16_dst_lo,  mvs16_dst_hi;

    // left border
    {
        vload(src_p, mvu8_src_p_x1);
        vload(src_c, mvu8_src_c_x1);
        vload(src_n, mvu8_src_n_x1);

        #pragma unroll
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvu8_src_c_x0.val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvu8_src_c_x1.val[ch], src_c[ch], border_value[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (MI_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS)
        {
            vload(src_p + C * x, mvu8_src_p_x2);
            vload(src_c + C * x, mvu8_src_c_x2);
            vload(src_n + C * x, mvu8_src_n_x2);

            #pragma unroll
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Laplacian1x1Core<St>(mvu8_src_p_x1.val[ch], mvu8_src_c_x0.val[ch],
                                     mvu8_src_c_x1.val[ch], mvu8_src_c_x2.val[ch],
                                     mvu8_src_n_x1.val[ch], mvs16_dst_lo.val[ch],
                                     mvs16_dst_hi.val[ch]);
            }

            vstore(dst_c + C * (x - ELEM_COUNTS),        mvs16_dst_lo);
            vstore(dst_c + C * (x - (ELEM_COUNTS >> 1)), mvs16_dst_hi);

            mvu8_src_p_x1 = mvu8_src_p_x2;
            mvu8_src_c_x0 = mvu8_src_c_x1;
            mvu8_src_c_x1 = mvu8_src_c_x2;
            mvu8_src_n_x1 = mvu8_src_n_x2;
        }
    }

    // remain
    {
        MI_S32 last = (width - 1) * C;
        MI_S32 rest = width % ELEM_COUNTS;
        MVType mvs16_last_lo, mvs16_last_hi;

        vload(src_p + C * back_offset, mvu8_src_p_x2);
        vload(src_c + C * back_offset, mvu8_src_c_x2);
        vload(src_n + C * back_offset, mvu8_src_n_x2);

        #pragma unroll
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector vu8_border_c = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvu8_src_c_x2.val[ch], src_c[last + ch], border_value[ch]);

            HVX_Vector vu8_src_c_r = Q6_V_vlalign_VVR(vu8_border_c, mvu8_src_c_x2.val[ch], rest);
            HVX_Vector vu8_src_c_l = Q6_V_valign_VVR(mvu8_src_c_x1.val[ch], mvu8_src_c_x0.val[ch], rest);

            Laplacian1x1Core<St>(mvu8_src_p_x1.val[ch], mvu8_src_c_x0.val[ch],
                                 mvu8_src_c_x1.val[ch], vu8_src_c_r, mvu8_src_n_x1.val[ch],
                                 mvs16_dst_lo.val[ch],  mvs16_dst_hi.val[ch]);

            Laplacian1x1Core<St>(mvu8_src_p_x2.val[ch], vu8_src_c_l, mvu8_src_c_x2.val[ch],
                                 vu8_border_c,          mvu8_src_n_x2.val[ch],
                                 mvs16_last_lo.val[ch], mvs16_last_hi.val[ch]);
        }

        vstore(dst_c + C * (back_offset - rest),                      mvs16_dst_lo);
        vstore(dst_c + C * (back_offset - rest + (ELEM_COUNTS >> 1)), mvs16_dst_hi);
        vstore(dst_c + C * back_offset,                               mvs16_last_lo);
        vstore(dst_c + C * (back_offset + (ELEM_COUNTS >> 1)),        mvs16_last_hi);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C,
          typename std::enable_if<std::is_same<St, MI_U16>::value || std::is_same<St, MI_S16>::value>::type* = MI_NULL>
static AURA_VOID Laplacian1x1Row(const St *src_p, const St *src_c, const St *src_n, Dt *dst_c,
                               const std::vector<St> &border_value, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    constexpr MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(St);
    MI_S32 back_offset = width - ELEM_COUNTS;

    MVType mvd16_src_p_x1, mvd16_src_p_x2;
    MVType mvd16_src_c_x0, mvd16_src_c_x1, mvd16_src_c_x2;
    MVType mvd16_src_n_x1, mvd16_src_n_x2;
    MVType mvd16_dst;

    // left border
    {
        vload(src_p, mvd16_src_p_x1);
        vload(src_c, mvd16_src_c_x1);
        vload(src_n, mvd16_src_n_x1);

        #pragma unroll
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvd16_src_c_x0.val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mvd16_src_c_x1.val[ch], src_c[ch], border_value[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (MI_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS)
        {
            vload(src_p + C * x, mvd16_src_p_x2);
            vload(src_c + C * x, mvd16_src_c_x2);
            vload(src_n + C * x, mvd16_src_n_x2);

            #pragma unroll
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Laplacian1x1Core<St>(mvd16_src_p_x1.val[ch], mvd16_src_c_x0.val[ch],
                                     mvd16_src_c_x1.val[ch], mvd16_src_c_x2.val[ch],
                                     mvd16_src_n_x1.val[ch], mvd16_dst.val[ch]);
            }

            vstore(dst_c + C * (x - ELEM_COUNTS), mvd16_dst);

            mvd16_src_p_x1 = mvd16_src_p_x2;
            mvd16_src_c_x0 = mvd16_src_c_x1;
            mvd16_src_c_x1 = mvd16_src_c_x2;
            mvd16_src_n_x1 = mvd16_src_n_x2;
        }
    }

    // remain
    {
        MI_S32 last = (width - 1) * C;
        MI_S32 rest = width % ELEM_COUNTS;
        MVType mvd16_last;

        vload(src_p + C * back_offset, mvd16_src_p_x2);
        vload(src_c + C * back_offset, mvd16_src_c_x2);
        vload(src_n + C * back_offset, mvd16_src_n_x2);

        #pragma unroll
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector vd16_border_c = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mvd16_src_c_x2.val[ch], src_c[last + ch], border_value[ch]);

            HVX_Vector vd16_src_c_r = Q6_V_vlalign_VVR(vd16_border_c, mvd16_src_c_x2.val[ch], rest << 1);
            HVX_Vector vd16_src_c_l = Q6_V_valign_VVR(mvd16_src_c_x1.val[ch], mvd16_src_c_x0.val[ch], rest << 1);

            Laplacian1x1Core<St>(mvd16_src_p_x1.val[ch], mvd16_src_c_x0.val[ch],
                                 mvd16_src_c_x1.val[ch], vd16_src_c_r,
                                 mvd16_src_n_x1.val[ch], mvd16_dst.val[ch]);

            Laplacian1x1Core<St>(mvd16_src_p_x2.val[ch], vd16_src_c_l,
                                 mvd16_src_c_x2.val[ch], vd16_border_c,
                                 mvd16_src_n_x2.val[ch], mvd16_last.val[ch]);
        }

        vstore(dst_c + C * (back_offset - rest), mvd16_dst);
        vstore(dst_c + C * back_offset,          mvd16_last);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C>
static Status Laplacian1x1HvxImpl(const Mat &src, Mat &dst, const std::vector<St> &border_value,
                                  const St *border_buffer, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width   = src.GetSizes().m_width;
    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;

    const St *src_p = src.Ptr<St, BORDER_TYPE>(start_row - 1, border_buffer);
    const St *src_c = src.Ptr<St>(start_row);
    const St *src_n = src.Ptr<St, BORDER_TYPE>(start_row + 1, border_buffer);

    MI_U64 L2fetch_row_param = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y + 2 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<St>(y + 2)), L2fetch_row_param);
        }

        Dt *dst_c = dst.Ptr<Dt>(y);
        Laplacian1x1Row<St, Dt, BORDER_TYPE, C>(src_p, src_c, src_n, dst_c, border_value, width);

        src_p = src_c;
        src_c = src_n;
        src_n = src.Ptr<St, BORDER_TYPE>(y + 2, border_buffer);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE>
static Status Laplacian1x1HvxHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<St> &border_value, const St *border_buffer)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Laplacian1x1HvxImpl<St, Dt, BORDER_TYPE, 1>,
                                  std::cref(src), std::ref(dst), std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Laplacian1x1HvxImpl<St, Dt, BORDER_TYPE, 2>,
                                  std::cref(src), std::ref(dst), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Laplacian1x1HvxImpl<St, Dt, BORDER_TYPE, 3>,
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
static Status Laplacian1x1HvxHelper(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    St *border_buffer = MI_NULL;
    std::vector<St> vec_border_value = border_value.ToVector<St>();

    MI_S32 width   = src.GetSizes().m_width;
    MI_S32 channel = src.GetSizes().m_channel;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            border_buffer = CreateBorderBuffer(ctx, width, channel, vec_border_value);
            if (MI_NULL == border_buffer)
            {
                AURA_ADD_ERROR_STRING(ctx, "CreateBorderBuffer failed");
                return Status::ERROR;
            }

            ret = Laplacian1x1HvxHelper<St, Dt, BorderType::CONSTANT>(ctx, src, dst, vec_border_value, border_buffer);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Laplacian1x1HvxHelper<St, Dt, BorderType::REPLICATE>(ctx, src, dst, vec_border_value, border_buffer);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Laplacian1x1HvxHelper<St, Dt, BorderType::REFLECT_101>(ctx, src, dst, vec_border_value, border_buffer);
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

Status Laplacian1x1Hvx(Context *ctx, const Mat &src, Mat &dst, BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    MI_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = Laplacian1x1HvxHelper<MI_U8, MI_S16>(ctx, src, dst, border_type, border_value);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = Laplacian1x1HvxHelper<MI_U16, MI_U16>(ctx, src, dst, border_type, border_value);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = Laplacian1x1HvxHelper<MI_S16, MI_S16>(ctx, src, dst, border_type, border_value);
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
