#include "sobel_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

const static DT_S8 g_kernel_tabel[4][3] =
{
    { 1,  2, 1},
    {-1,  0, 1},
    { 1, -2, 1},
    { 3, 10, 3}
};

// using St = DT_U8, Dt = S16
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, DT_U8>::value && std::is_same<Dt, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Sobel3x3VCore(HVX_Vector &vu8_src_p0, HVX_Vector &vu8_src_c, HVX_Vector &vu8_src_n0,
                                         HVX_VectorPair &ws16_sum, const DT_U8 *kernel_y)
{
    DT_U32 k0k0k0k0 = Q6_R_vsplatb_R(kernel_y[0]);
    DT_S32 k2k1k2k1 = (kernel_y[2] << 24) | (kernel_y[1] << 16) | (kernel_y[2] << 8) | kernel_y[1];

    ws16_sum = Q6_Wh_vmpy_VubRb(vu8_src_p0, k0k0k0k0);
    ws16_sum = Q6_Wh_vmpaacc_WhWubRb(ws16_sum, Q6_W_vcombine_VV(vu8_src_n0, vu8_src_c), k2k1k2k1);
}

// using St = DT_U8, Dt = S16
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, DT_U8>::value && std::is_same<Dt, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Sobel3x3HCore(HVX_VectorPair &ws16_sum_x0, HVX_VectorPair &ws16_sum_x1, HVX_VectorPair &ws16_sum_x2,
                                         HVX_Vector &vs16_result_lo, HVX_Vector &vs16_result_hi, const DT_S8 *kernel_x)
{
    constexpr DT_S32 ALIGN     = sizeof(Dt);
    DT_U32 k0k0k0k0            = Q6_R_vsplatb_R(kernel_x[0]);
    DT_U32 k1k1k1k1            = Q6_R_vsplatb_R(kernel_x[1]);
    DT_U32 k2k2k2k2            = Q6_R_vsplatb_R(kernel_x[2]);

    HVX_Vector vs16_sum_x1_lo  = Q6_V_lo_W(ws16_sum_x1);
    HVX_Vector vs16_sum_x1_hi  = Q6_V_hi_W(ws16_sum_x1);
    HVX_Vector vs16_sum_l0_hi  = Q6_V_vlalign_VVR(vs16_sum_x1_hi, Q6_V_hi_W(ws16_sum_x0), ALIGN);
    HVX_Vector vs16_sum_c_lo   = vs16_sum_x1_lo;
    HVX_Vector vs16_sum_c_hi   = vs16_sum_x1_hi;
    HVX_Vector vs16_sum_r0_lo  = Q6_V_valign_VVR(Q6_V_lo_W(ws16_sum_x2), vs16_sum_x1_lo, ALIGN);

    HVX_Vector vs16_sum_lo     = Q6_Vh_vmpyi_VhRb(vs16_sum_l0_hi, k0k0k0k0);
    vs16_sum_lo                = Q6_Vh_vmpyiacc_VhVhRb(vs16_sum_lo, vs16_sum_c_lo, k1k1k1k1);
    vs16_sum_lo                = Q6_Vh_vmpyiacc_VhVhRb(vs16_sum_lo, vs16_sum_c_hi, k2k2k2k2);
    HVX_Vector vs16_sum_hi     = Q6_Vh_vmpyi_VhRb(vs16_sum_c_lo, k0k0k0k0);
    vs16_sum_hi                = Q6_Vh_vmpyiacc_VhVhRb(vs16_sum_hi, vs16_sum_c_hi, k1k1k1k1);
    vs16_sum_hi                = Q6_Vh_vmpyiacc_VhVhRb(vs16_sum_hi, vs16_sum_r0_lo, k2k2k2k2);

    HVX_VectorPair ws16_result = Q6_W_vshuff_VVR(vs16_sum_hi, vs16_sum_lo, -2);
    vs16_result_lo             = Q6_V_lo_W(ws16_result);
    vs16_result_hi             = Q6_V_hi_W(ws16_result);
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, DT_U8>::value && std::is_same<Dt, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Sobel3x3HCore(HVX_VectorPair &ws16_sum_x0, HVX_VectorPair &ws16_sum_x1, HVX_VectorPair &ws16_sum_x2, HVX_VectorPair &ws16_sum_x3,
                                         HVX_Vector &vs16_result_x0_lo, HVX_Vector &vs16_result_x0_hi, HVX_Vector &vs16_result_x1_lo, HVX_Vector &vs16_result_x1_hi,
                                         const DT_S8 *kernel_x, DT_S32 rest)
{
    HVX_Vector vs16_sum_x0_lo = Q6_V_lo_W(ws16_sum_x0);
    HVX_Vector vs16_sum_x0_hi = Q6_V_hi_W(ws16_sum_x0);
    HVX_Vector vs16_sum_x1_lo = Q6_V_lo_W(ws16_sum_x1);
    HVX_Vector vs16_sum_x1_hi = Q6_V_hi_W(ws16_sum_x1);
    HVX_Vector vs16_sum_x2_lo = Q6_V_lo_W(ws16_sum_x2);
    HVX_Vector vs16_sum_x2_hi = Q6_V_hi_W(ws16_sum_x2);
    HVX_Vector vs16_sum_x3_lo = Q6_V_lo_W(ws16_sum_x3);
    HVX_Vector vs16_sum_x3_hi = Q6_V_hi_W(ws16_sum_x3);

    HVX_Vector vs16_sum_l0_lo, vs16_sum_l0_hi, vs16_sum_r0_lo, vs16_sum_r0_hi;
    if (rest & 1)
    {
        DT_S32 align_size0 = (rest / 2) * sizeof(Dt);
        DT_S32 align_size1 = align_size0 + sizeof(Dt);
        vs16_sum_r0_lo = Q6_V_vlalign_safe_VVR(vs16_sum_x3_hi, vs16_sum_x2_hi, align_size1);
        vs16_sum_r0_hi = Q6_V_vlalign_safe_VVR(vs16_sum_x3_lo, vs16_sum_x2_lo, align_size0);
        vs16_sum_l0_lo = Q6_V_valign_safe_VVR(vs16_sum_x1_hi,  vs16_sum_x0_hi, align_size0);
        vs16_sum_l0_hi = Q6_V_valign_safe_VVR(vs16_sum_x1_lo,  vs16_sum_x0_lo, align_size1);
    }
    else
    {
        DT_S32 align_size = (rest / 2) * sizeof(Dt);
        vs16_sum_r0_lo = Q6_V_vlalign_safe_VVR(vs16_sum_x3_lo, vs16_sum_x2_lo, align_size);
        vs16_sum_r0_hi = Q6_V_vlalign_safe_VVR(vs16_sum_x3_hi, vs16_sum_x2_hi, align_size);
        vs16_sum_l0_lo = Q6_V_valign_safe_VVR(vs16_sum_x1_lo,  vs16_sum_x0_lo, align_size);
        vs16_sum_l0_hi = Q6_V_valign_safe_VVR(vs16_sum_x1_hi,  vs16_sum_x0_hi, align_size);
    }

    HVX_VectorPair ws16_sum_r0 = Q6_W_vcombine_VV(vs16_sum_r0_hi, vs16_sum_r0_lo);
    HVX_VectorPair ws16_sum_l0 = Q6_W_vcombine_VV(vs16_sum_l0_hi, vs16_sum_l0_lo);

    Sobel3x3HCore<St, Dt>(ws16_sum_x0, ws16_sum_x1, ws16_sum_r0, vs16_result_x0_lo, vs16_result_x0_hi, kernel_x);
    Sobel3x3HCore<St, Dt>(ws16_sum_l0, ws16_sum_x2, ws16_sum_x3, vs16_result_x1_lo, vs16_result_x1_hi, kernel_x);
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C, DT_BOOL WITH_SCALE>
static DT_VOID Sobel3x3Row(const St *src_p0, const St *src_c, const St *src_n0, Dt *dst, const DT_S8 *kernel_x,
                           const DT_S8 *kernel_y, DT_F32 scale, const std::vector<St> &border_value, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    constexpr DT_S32 SRC_ELEM_COUNTS = AURA_HVLEN / sizeof(St);
    constexpr DT_S32 DST_ELEM_COUNTS = AURA_HVLEN / sizeof(Dt);

    const DT_S32 back_offset = width - SRC_ELEM_COUNTS;

    MVType mv_src_p0, mv_src_c, mv_src_n0;
    MWType mw_sum_x0, mw_sum_x1, mw_sum_x2, mw_sum_x3;
    MVType mv_result_lo, mv_result_hi;

    // left border
    {
        vload(src_p0, mv_src_p0);
        vload(src_c,  mv_src_c);
        vload(src_n0, mv_src_n0);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_p0_border = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mv_src_p0.val[ch], src_p0[ch], border_value[ch]);
            HVX_Vector v_c_border  = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mv_src_c.val[ch],  src_c[ch],  border_value[ch]);
            HVX_Vector v_n0_border = GetBorderVector<St, BORDER_TYPE, BorderArea::LEFT>(mv_src_n0.val[ch], src_n0[ch], border_value[ch]);

            Sobel3x3VCore<St, Dt>(v_p0_border, v_c_border, v_n0_border, mw_sum_x0.val[ch], (DT_U8 *)kernel_y);
            Sobel3x3VCore<St, Dt>(mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mw_sum_x1.val[ch], (DT_U8 *)kernel_y);
        }
    }

    // main(0 ~ n-2)
    {
        for (DT_S32 x = SRC_ELEM_COUNTS; x <= back_offset; x += SRC_ELEM_COUNTS)
        {
            vload(src_p0 + C * x, mv_src_p0);
            vload(src_c  + C * x, mv_src_c);
            vload(src_n0 + C * x, mv_src_n0);

            #pragma unroll(C)
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Sobel3x3VCore<St, Dt>(mv_src_p0.val[ch], mv_src_c.val[ch],  mv_src_n0.val[ch], mw_sum_x2.val[ch],    (DT_U8 *)kernel_y);
                Sobel3x3HCore<St, Dt>(mw_sum_x0.val[ch], mw_sum_x1.val[ch], mw_sum_x2.val[ch], mv_result_lo.val[ch], mv_result_hi.val[ch], kernel_x);
                if (WITH_SCALE)
                {
                    SobelPostProcess(mv_result_lo.val[ch], mv_result_hi.val[ch], scale);
                }
            }
            vstore(dst + C * (x - SRC_ELEM_COUNTS),                   mv_result_lo);
            vstore(dst + C * (x - SRC_ELEM_COUNTS + DST_ELEM_COUNTS), mv_result_hi);

            mw_sum_x0 = mw_sum_x1;
            mw_sum_x1 = mw_sum_x2;
        }
    }

    // remain
    {
        DT_S32 last = C * (width - 1);
        DT_S32 rest = width % SRC_ELEM_COUNTS;
        MVType mv_last_lo, mv_last_hi;

        vload(src_p0 + C * back_offset, mv_src_p0);
        vload(src_c  + C * back_offset, mv_src_c);
        vload(src_n0 + C * back_offset, mv_src_n0);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_p0_border = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0.val[ch], src_p0[last + ch], border_value[ch]);
            HVX_Vector v_c_border  = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mv_src_c.val[ch],  src_c[last + ch],  border_value[ch]);
            HVX_Vector v_n0_border = GetBorderVector<St, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0.val[ch], src_n0[last + ch], border_value[ch]);

            Sobel3x3VCore<St, Dt>(mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mw_sum_x2.val[ch], (DT_U8 *)kernel_y);
            Sobel3x3VCore<St, Dt>(v_p0_border, v_c_border, v_n0_border, mw_sum_x3.val[ch], (DT_U8 *)kernel_y);

            Sobel3x3HCore<St, Dt>(mw_sum_x0.val[ch],    mw_sum_x1.val[ch],    mw_sum_x2.val[ch],  mw_sum_x3.val[ch],
                                  mv_result_lo.val[ch], mv_result_hi.val[ch], mv_last_lo.val[ch], mv_last_hi.val[ch], kernel_x, rest);
            if (WITH_SCALE)
            {
                SobelPostProcess(mv_result_lo.val[ch], mv_result_hi.val[ch], scale);
                SobelPostProcess(mv_last_lo.val[ch],   mv_last_hi.val[ch],   scale);
            }
        }

        vstore(dst + C * (back_offset - rest),                   mv_result_lo);
        vstore(dst + C * (back_offset - rest + DST_ELEM_COUNTS), mv_result_hi);
        vstore(dst + C * back_offset,                            mv_last_lo);
        vstore(dst + C * (back_offset + DST_ELEM_COUNTS),        mv_last_hi);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C, DT_BOOL WITH_SCALE>
static Status Sobel3x3HvxImpl(const Mat &src, Mat &dst, DT_S8 *kernel_x, DT_S8 *kernel_y, DT_F32 scale, const std::vector<St> &border_value,
                              const St *border_buffer, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width  = src.GetSizes().m_width;
    DT_S32 height = src.GetSizes().m_height;
    DT_S32 stride = src.GetStrides().m_width;

    const St *src_p0 = src.Ptr<St, BORDER_TYPE>(start_row - 1, border_buffer);
    const St *src_c  = src.Ptr<St>(start_row);
    const St *src_n0 = src.Ptr<St, BORDER_TYPE>(start_row + 1, border_buffer);

    DT_U64 L2fetch_param = L2PfParam(stride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if (y + 2 < height)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<St>(y + 2)), L2fetch_param);
        }

        Dt *dst_row  = dst.Ptr<Dt>(y);
        Sobel3x3Row<St, Dt, BORDER_TYPE, C, WITH_SCALE>(src_p0, src_c, src_n0, dst_row, kernel_x, kernel_y, scale, border_value, width);

        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src.Ptr<St, BORDER_TYPE>(y + 2, border_buffer);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C>
static Status Sobel3x3HvxHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 dx, DT_S32 dy, DT_F32 scale,
                                const std::vector<St> &border_value, const St *border_buffer)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    DT_S8 *kernel_x = const_cast<DT_S8 *>(g_kernel_tabel[dx]);
    DT_S8 *kernel_y = const_cast<DT_S8 *>(g_kernel_tabel[dy]);
    DT_S32 height = dst.GetSizes().m_height;

    if (NearlyEqual(scale, 1.f))
    {
        ret = wp->ParallelFor((DT_S32)0, height, Sobel3x3HvxImpl<St, Dt, BORDER_TYPE, C, DT_FALSE>,
                              std::cref(src), std::ref(dst), kernel_x, kernel_y, scale, std::cref(border_value), border_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel3x3HvxImpl<St, Dt, BORDER_TYPE, C, DT_FALSE> failed");
        }
    }
    else
    {
        ret = wp->ParallelFor((DT_S32)0, height, Sobel3x3HvxImpl<St, Dt, BORDER_TYPE, C, DT_TRUE>,
                              std::cref(src), std::ref(dst), kernel_x, kernel_y, scale, std::cref(border_value), border_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel3x3HvxImpl<St, Dt, BORDER_TYPE, C, DT_TRUE> failed");
        }
    }

    AURA_RETURN(ctx, ret);
}

template<typename St, typename Dt, BorderType BORDER_TYPE>
static Status Sobel3x3HvxHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 dx, DT_S32 dy, DT_F32 scale,
                                const std::vector<St> &border_value, const St *border_buffer)
{
    Status ret = Status::ERROR;

    switch (src.GetSizes().m_channel)
    {
        case 1:
        {
            ret = Sobel3x3HvxHelper<St, Dt, BORDER_TYPE, 1>(ctx, src, dst, dx, dy, scale, border_value, border_buffer);
            break;
        }

        case 2:
        {
            ret = Sobel3x3HvxHelper<St, Dt, BORDER_TYPE, 2>(ctx, src, dst, dx, dy, scale, border_value, border_buffer);
            break;
        }

        case 3:
        {
            ret = Sobel3x3HvxHelper<St, Dt, BORDER_TYPE, 3>(ctx, src, dst, dx, dy, scale, border_value, border_buffer);
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
static Status Sobel3x3HvxHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 dx, DT_S32 dy, DT_F32 scale,
                                BorderType border_type, const Scalar &border_value)
{
    St *border_buffer = DT_NULL;
    std::vector<St> vec_border_value = border_value.ToVector<St>();

    DT_S32 width   = dst.GetSizes().m_width;
    DT_S32 channel = dst.GetSizes().m_channel;

    Status ret = Status::ERROR;

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

            ret = Sobel3x3HvxHelper<St, Dt, BorderType::CONSTANT>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Sobel3x3HvxHelper<St, Dt, BorderType::REPLICATE>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Sobel3x3HvxHelper<St, Dt, BorderType::REFLECT_101>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer);
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

Status Sobel3x3Hvx(Context *ctx, const Mat &src, Mat &dst, DT_S32 dx, DT_S32 dy, DT_F32 scale,
                   BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;
    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = Sobel3x3HvxHelper<DT_U8, DT_S16>(ctx, src, dst, dx, dy, scale, border_type, border_value);
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
