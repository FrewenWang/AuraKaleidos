#include "sobel_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

const static MI_S8 g_kernel_tabel[2][3] =
{
    {-1,  0, 1},
    { 1, -2, 1}
};

// using St = MI_U8, Dt = S16
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U8>::value && std::is_same<Dt, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Sobel1x1Core(HVX_Vector &vu8_src_p0, HVX_Vector &vu8_src_c, HVX_Vector &vu8_src_n0,
                                        HVX_Vector &vs16_result_lo, HVX_Vector &vs16_result_hi, const MI_U8 *kernel)
{
    MI_S32 k0k0k0k0 = Q6_R_vsplatb_R(kernel[0]);
    MI_S32 k2k1k2k1 = (kernel[2] << 24) | (kernel[1] << 16) | (kernel[2] << 8) | kernel[1];

    HVX_VectorPair ws16_result;
    ws16_result = Q6_Wh_vmpy_VubRb(vu8_src_p0, k0k0k0k0);
    ws16_result = Q6_Wh_vmpaacc_WhWubRb(ws16_result, Q6_W_vcombine_VV(vu8_src_n0, vu8_src_c), k2k1k2k1);

    ws16_result    = Q6_W_vshuff_VVR(Q6_V_hi_W(ws16_result), Q6_V_lo_W(ws16_result), -2);
    vs16_result_lo = Q6_V_lo_W(ws16_result);
    vs16_result_hi = Q6_V_hi_W(ws16_result);
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U8>::value && std::is_same<Dt, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Sobel1x1HCore(HVX_Vector &vu8_src_x0, HVX_Vector &vu8_src_x1, HVX_Vector &vu8_src_x2,
                                         HVX_Vector &vs16_result_lo, HVX_Vector &vs16_result_hi, const MI_S8 *kernel)
{
    MI_S32 sum_elem_size = sizeof(St);

    HVX_Vector vu8_src_l = Q6_V_vlalign_VVR(vu8_src_x1, vu8_src_x0, sum_elem_size);
    HVX_Vector vu8_src_c = vu8_src_x1;
    HVX_Vector vu8_src_r = Q6_V_valign_VVR(vu8_src_x2, vu8_src_x1, sum_elem_size);

    Sobel1x1Core<St, Dt>(vu8_src_l, vu8_src_c, vu8_src_r, vs16_result_lo, vs16_result_hi, (MI_U8 *)kernel);
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U8>::value && std::is_same<Dt, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Sobel1x1HCore(HVX_Vector &vu8_src_x0, HVX_Vector &vu8_src_x1, HVX_Vector &vu8_src_x2, HVX_Vector &vu8_src_x3,
                                         HVX_Vector &vs16_result_x0_lo, HVX_Vector &vs16_result_x0_hi,
                                         HVX_Vector &vs16_result_x1_lo, HVX_Vector &vs16_result_x1_hi,
                                         const MI_S8 *kernel, MI_S32 rest)
{
    HVX_Vector vu8_src_r = Q6_V_vlalign_safe_VVR(vu8_src_x3, vu8_src_x2, rest * sizeof(St));
    HVX_Vector vu8_src_l = Q6_V_valign_safe_VVR(vu8_src_x1,  vu8_src_x0, rest * sizeof(St));

    Sobel1x1HCore<St, Dt>(vu8_src_x0, vu8_src_x1, vu8_src_r,  vs16_result_x0_lo, vs16_result_x0_hi, kernel);
    Sobel1x1HCore<St, Dt>(vu8_src_l,  vu8_src_x2, vu8_src_x3, vs16_result_x1_lo, vs16_result_x1_hi, kernel);
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C, MI_BOOL WITH_SCALE>
static AURA_VOID Sobel1x1DxRow(const St *src, Dt *dst, const MI_S8 *kernel, MI_F32 scale, const std::vector<St> &border_value, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    constexpr MI_S32 SRC_ELEM_COUNTS = AURA_HVLEN / sizeof(St);
    constexpr MI_S32 DST_ELEM_COUNTS = AURA_HVLEN / sizeof(Dt);

    const MI_S32 back_offset = width - SRC_ELEM_COUNTS;

    MVType mv_src_x0, mv_src_x1, mv_src_x2, mv_src_x3;
    MVType mv_result_lo, mv_result_hi;

    // left border
    {
        vload(src,  mv_src_x1);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_x0.val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mv_src_x1.val[ch], src[ch], border_value[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (MI_S32 x = SRC_ELEM_COUNTS; x <= back_offset; x += SRC_ELEM_COUNTS)
        {
            vload(src + C * x, mv_src_x2);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Sobel1x1HCore<St, Dt>(mv_src_x0.val[ch], mv_src_x1.val[ch], mv_src_x2.val[ch], mv_result_lo.val[ch], mv_result_hi.val[ch], kernel);
                if (WITH_SCALE)
                {
                    SobelPostProcess(mv_result_lo.val[ch], mv_result_hi.val[ch], scale);
                }
            }

            vstore(dst + C * (x - SRC_ELEM_COUNTS),                   mv_result_lo);
            vstore(dst + C * (x - SRC_ELEM_COUNTS + DST_ELEM_COUNTS), mv_result_hi);

            mv_src_x0 = mv_src_x1;
            mv_src_x1 = mv_src_x2;
        }
    }

    // remain
    {
        MI_S32 last = C * (width - 1);
        MI_S32 rest = width % SRC_ELEM_COUNTS;
        MVType mv_last_lo, mv_last_hi;

        vload(src + C * back_offset, mv_src_x2);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_x3.val[ch] = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mv_src_x2.val[ch], src[last + ch], border_value[ch]);

            Sobel1x1HCore<St, Dt>(mv_src_x0.val[ch],    mv_src_x1.val[ch],    mv_src_x2.val[ch],  mv_src_x3.val[ch],
                                  mv_result_lo.val[ch], mv_result_hi.val[ch], mv_last_lo.val[ch], mv_last_hi.val[ch], kernel, rest);
            if (WITH_SCALE)
            {
                SobelPostProcess(mv_result_lo.val[ch], mv_result_hi.val[ch], scale);
                SobelPostProcess(mv_last_lo.val[ch], mv_last_hi.val[ch], scale);
            }
        }

        vstore(dst + C * (back_offset - rest),                   mv_result_lo);
        vstore(dst + C * (back_offset - rest + DST_ELEM_COUNTS), mv_result_hi);
        vstore(dst + C * back_offset,                            mv_last_lo);
        vstore(dst + C * (back_offset        + DST_ELEM_COUNTS), mv_last_hi);
    }
}

template <typename St, typename Dt, MI_S32 C, MI_BOOL WITH_SCALE>
static AURA_VOID Sobel1x1DyRow(const St *src_p0, const St *src_c, const St *src_n0, Dt *dst, const MI_S8 *kernel, MI_F32 scale, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    constexpr MI_S32 SRC_ELEM_COUNTS = AURA_HVLEN / sizeof(St);
    constexpr MI_S32 DST_ELEM_COUNTS = AURA_HVLEN / sizeof(Dt);
    constexpr MI_S32 VOFFSET         = SRC_ELEM_COUNTS;

    const MI_S32 width_align = (width & (-SRC_ELEM_COUNTS));

    MVType mv_src_p0, mv_src_c, mv_src_n0;
    MVType mv_result_lo, mv_result_hi;

    // main(0 ~ n-1)
    {
        for (MI_S32 x = 0; x < width_align; x += VOFFSET)
        {
            vload(src_p0 + C * x, mv_src_p0);
            vload(src_c  + C * x, mv_src_c);
            vload(src_n0 + C * x, mv_src_n0);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Sobel1x1Core<St, Dt>(mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mv_result_lo.val[ch], mv_result_hi.val[ch], (MI_U8 *)kernel);
                if (WITH_SCALE)
                {
                    SobelPostProcess(mv_result_lo.val[ch], mv_result_hi.val[ch], scale);
                }
            }
            vstore(dst + C * x,                     mv_result_lo);
            vstore(dst + C * (x + DST_ELEM_COUNTS), mv_result_hi);
        }
    }

    // remain
    {
        if (width_align != width)
        {
            MI_S32 x = width - SRC_ELEM_COUNTS;

            vload(src_p0 + C * x, mv_src_p0);
            vload(src_c  + C * x, mv_src_c);
            vload(src_n0 + C * x, mv_src_n0);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Sobel1x1Core<St, Dt>(mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mv_result_lo.val[ch], mv_result_hi.val[ch], (MI_U8 *)kernel);
                if (WITH_SCALE)
                {
                    SobelPostProcess(mv_result_lo.val[ch], mv_result_hi.val[ch], scale);
                }
            }
            vstore(dst + C * x,                     mv_result_lo);
            vstore(dst + C * (x + DST_ELEM_COUNTS), mv_result_hi);
        }
    }
}

template <typename St, typename Dt, MI_S32 C, MI_BOOL WITH_SCALE, BorderType BORDER_TYPE>
static Status Sobel1x1DxHvxImpl(const Mat &src, Mat &dst, const MI_S8 *kernel, MI_F32 scale,
                                const std::vector<St> &border_value, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width  = src.GetSizes().m_width;
    MI_S32 stride = src.GetStrides().m_width;
    MI_S32 height = src.GetSizes().m_height;

    MI_U64 L2fetch_param = L2PfParam(stride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);

    // middle
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y + 1 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<St>(y + 1)), L2fetch_param);
        }

        const St *src_row  = src.Ptr<St>(y);
        Dt       *dst_row  = dst.Ptr<Dt>(y);
        Sobel1x1DxRow<St, Dt, BORDER_TYPE, C, WITH_SCALE>(src_row, dst_row, kernel, scale, border_value, width);
    }

    return Status::OK;
}

template <typename St, typename Dt, MI_S32 C, MI_BOOL WITH_SCALE, BorderType BORDER_TYPE>
static Status Sobel1x1DyHvxImpl(const Mat &src, Mat &dst, MI_S8 *kernel, MI_F32 scale,
                                const St *border_buffer, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width  = src.GetSizes().m_width;
    MI_S32 height = src.GetSizes().m_height;
    MI_S32 stride = src.GetStrides().m_width;

    const St *src_p0 = src.Ptr<St, BORDER_TYPE>(start_row - 1, border_buffer);
    const St *src_c  = src.Ptr<St>(start_row);
    const St *src_n0 = src.Ptr<St, BORDER_TYPE>(start_row + 1, border_buffer);

    MI_U64 L2fetch_param = L2PfParam(stride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y + 2 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<St>(y + 2)), L2fetch_param);
        }

        Dt *dst_row = dst.Ptr<Dt>(y);
        Sobel1x1DyRow<St, Dt, C, WITH_SCALE>(src_p0, src_c, src_n0, dst_row, kernel, scale, width);
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src.Ptr<St, BORDER_TYPE>(y + 2, border_buffer);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C, MI_BOOL WITH_SCALE>
static Status Sobel1x1HvxHelper(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                                const std::vector<St> &border_value, const St *border_buffer)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    MI_S8 *kernel = const_cast<MI_S8 *>(g_kernel_tabel[(dx | dy) - 1]);
    MI_S32 height = dst.GetSizes().m_height;

    if (0 == dy)
    {
        ret = wp->ParallelFor((MI_S32)0, height, Sobel1x1DxHvxImpl<St, Dt, C, WITH_SCALE, BORDER_TYPE>,
                              std::cref(src), std::ref(dst), kernel, scale, std::cref(border_value));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel1x1DxHvxImpl<BORDER_TYPE, St, Dt, C, WITH_SCALE> failed");
        }
    }
    else
    {
        ret = wp->ParallelFor((MI_S32)0, height, Sobel1x1DyHvxImpl<St, Dt, C, WITH_SCALE, BORDER_TYPE>,
                              std::cref(src), std::ref(dst), kernel, scale, border_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel1x1DyHvxImpl<BORDER_TYPE, St, Dt, C, WITH_SCALE> failed");
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C>
static Status Sobel1x1HvxHelper(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                                const std::vector<St> &border_value, const St *border_buffer)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if (NearlyEqual(scale, 1.f))
    {
        ret = Sobel1x1HvxHelper<St, Dt, BORDER_TYPE, C, MI_FALSE>(ctx, src, dst, dx, dy, scale, border_value, border_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel1x1HvxHelper<St, Dt, BORDER_TYPE, C, MI_FALSE> failed");
        }
    }
    else
    {
        ret = Sobel1x1HvxHelper<St, Dt, BORDER_TYPE, C, MI_TRUE>(ctx, src, dst, dx, dy, scale, border_value, border_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel1x1HvxHelper<St, Dt, BORDER_TYPE, C, MI_TRUE> failed");
        }
    }

    AURA_RETURN(ctx, ret);
}

template<typename St, typename Dt, BorderType BORDER_TYPE>
static Status Sobel1x1HvxHelper(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                                const std::vector<St> &border_value, const St *border_buffer)
{
    Status ret = Status::ERROR;

    switch (src.GetSizes().m_channel)
    {
        case 1:
        {
            ret = Sobel1x1HvxHelper<St, Dt, BORDER_TYPE, 1>(ctx, src, dst, dx, dy, scale, border_value, border_buffer);
            break;
        }

        case 2:
        {
            ret = Sobel1x1HvxHelper<St, Dt, BORDER_TYPE, 2>(ctx, src, dst, dx, dy, scale, border_value, border_buffer);
            break;
        }

        case 3:
        {
            ret = Sobel1x1HvxHelper<St, Dt, BORDER_TYPE, 3>(ctx, src, dst, dx, dy, scale, border_value, border_buffer);
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

template <typename St, typename Dt>
static Status Sobel1x1HvxHelper(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                                BorderType border_type, const Scalar &border_value)
{
    St *border_buffer = MI_NULL;
    std::vector<St> vec_border_value = border_value.ToVector<St>();

    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 channel = dst.GetSizes().m_channel;

    Status ret = Status::ERROR;

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

            ret = Sobel1x1HvxHelper<St, Dt, BorderType::CONSTANT>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Sobel1x1HvxHelper<St, Dt, BorderType::REPLICATE>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Sobel1x1HvxHelper<St, Dt, BorderType::REFLECT_101>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer);
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

Status Sobel1x1Hvx(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                   BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;
    MI_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = Sobel1x1HvxHelper<MI_U8, MI_S16>(ctx, src, dst, dx, dy, scale, border_type, border_value);
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
