#include "morph_impl.hpp"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp, MorphType MORPH_TYPE>
AURA_ALWAYS_INLINE DT_VOID Morph3x3VCore(HVX_Vector &v_src_p0, HVX_Vector &v_src_n0, HVX_Vector &v_result)
{
    MorphHvxMinMax<Tp, MORPH_TYPE>(v_src_p0, v_src_n0, v_result);
}

template <typename Tp, MorphType MORPH_TYPE>
AURA_ALWAYS_INLINE DT_VOID Morph3x3VCore(HVX_Vector &v_src_p0, HVX_Vector &v_src_c, HVX_Vector &v_src_n0, HVX_Vector &v_result)
{
    MorphHvxMinMax<Tp, MORPH_TYPE>(v_src_p0, v_src_c, v_src_n0, v_result);
}

template <typename Tp, MorphType MORPH_TYPE>
AURA_ALWAYS_INLINE DT_VOID Morph3x3HCore(HVX_Vector &v_vertical_px1, HVX_Vector &v_vertical_cx0, HVX_Vector &v_vertical_cx1,
                                         HVX_Vector &v_vertical_cx2, HVX_Vector &v_result)
{
    DT_S32 align_size = sizeof(Tp);
    HVX_Vector v_result_c;

    HVX_Vector v_vertical_cl0 = Q6_V_vlalign_VVR(v_vertical_cx1, v_vertical_cx0, align_size);
    HVX_Vector v_vertical_c   = v_vertical_cx1;
    HVX_Vector v_vertical_cr0 = Q6_V_valign_VVR(v_vertical_cx2, v_vertical_cx1, align_size);

    MorphHvxMinMax<Tp, MORPH_TYPE>(v_vertical_cl0, v_vertical_c, v_vertical_cr0, v_result_c);
    MorphHvxMinMax<Tp, MORPH_TYPE>(v_result_c, v_vertical_px1, v_result);
}

template <typename Tp, MorphType MORPH_TYPE>
AURA_ALWAYS_INLINE DT_VOID Morph3x3HCore(HVX_Vector &v_vertical_cx0, HVX_Vector &v_vertical_cx1, HVX_Vector &v_vertical_cx2, HVX_Vector &v_result)
{
    DT_S32 align_size = sizeof(Tp);

    HVX_Vector v_vertical_cl0 = Q6_V_vlalign_VVR(v_vertical_cx1, v_vertical_cx0, align_size);
    HVX_Vector v_vertical_c  = v_vertical_cx1;
    HVX_Vector v_vertical_cr0 = Q6_V_valign_VVR(v_vertical_cx2, v_vertical_cx1, align_size);

    MorphHvxMinMax<Tp, MORPH_TYPE>(v_vertical_cl0, v_vertical_c, v_vertical_cr0, v_result);
}

template <typename Tp, MorphType MORPH_TYPE>
AURA_ALWAYS_INLINE DT_VOID Morph3x3HCore(HVX_Vector &v_vertical_px1, HVX_Vector &v_vertical_px2, HVX_Vector &v_vertical_cx0, HVX_Vector &v_vertical_cx1,
                                         HVX_Vector &v_vertical_cx2, HVX_Vector &v_vertical_cx3,  HVX_Vector &v_result_x0, HVX_Vector &v_result_x1,
                                         DT_S32 rest)
{
    DT_S32 align_size = rest * sizeof(Tp);

    HVX_Vector v_vertical_cr0 = Q6_V_vlalign_VVR(v_vertical_cx3, v_vertical_cx2, align_size);
    HVX_Vector v_vertical_cl0 = Q6_V_valign_VVR(v_vertical_cx1, v_vertical_cx0, align_size);

    Morph3x3HCore<Tp, MORPH_TYPE>(v_vertical_px1, v_vertical_cx0, v_vertical_cx1, v_vertical_cr0, v_result_x0);
    Morph3x3HCore<Tp, MORPH_TYPE>(v_vertical_px2, v_vertical_cl0, v_vertical_cx2, v_vertical_cx3, v_result_x1);
}

template <typename Tp, MorphType MORPH_TYPE>
AURA_ALWAYS_INLINE DT_VOID Morph3x3HCore(HVX_Vector &v_vertical_cx0, HVX_Vector &v_vertical_cx1, HVX_Vector &v_vertical_cx2, HVX_Vector &v_vertical_cx3,
                                         HVX_Vector &v_result_x0, HVX_Vector &v_result_x1, DT_S32 rest)
{
    DT_S32 align_size = rest * sizeof(Tp);

    HVX_Vector v_vertical_cr0 = Q6_V_vlalign_VVR(v_vertical_cx3, v_vertical_cx2, align_size);
    HVX_Vector v_vertical_cl0 = Q6_V_valign_VVR(v_vertical_cx1, v_vertical_cx0, align_size);

    Morph3x3HCore<Tp, MORPH_TYPE>(v_vertical_cx0, v_vertical_cx1, v_vertical_cr0, v_result_x0);
    Morph3x3HCore<Tp, MORPH_TYPE>(v_vertical_cl0, v_vertical_cx2, v_vertical_cx3, v_result_x1);
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 C,
          typename std::enable_if<MORPH_SHAPE == MorphShape::RECT>::type* = DT_NULL>
DT_VOID Morph3x3Row(const Tp *src_p0, const Tp *src_c, const Tp *src_n0, Tp *dst, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    DT_S32 back_offset = width - elem_counts;

    MVType mv_src_p0, mv_src_c, mv_src_n0;
    MVType mv_vertical_cx0, mv_vertical_cx1, mv_vertical_cx2, mv_vertical_cx3;
    MVType mv_result;

    // left border
    {
        vload(src_p0, mv_src_p0);
        vload(src_c,  mv_src_c);
        vload(src_n0, mv_src_n0);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_p0.val[ch], src_p0[ch], src_p0[ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_c.val[ch],  src_c[ch],  src_c[ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_n0.val[ch], src_n0[ch], src_n0[ch]);

            Morph3x3VCore<Tp, MORPH_TYPE>(v_border_p0, v_border_c, v_border_n0, mv_vertical_cx0.val[ch]);
            Morph3x3VCore<Tp, MORPH_TYPE>(mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mv_vertical_cx1.val[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (DT_S32 x = elem_counts; x <= back_offset; x += elem_counts)
        {
            vload(src_p0 + C * x, mv_src_p0);
            vload(src_c  + C * x, mv_src_c);
            vload(src_n0 + C * x, mv_src_n0);

            #pragma unroll(C)
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Morph3x3VCore<Tp, MORPH_TYPE>(mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mv_vertical_cx2.val[ch]);
                Morph3x3HCore<Tp, MORPH_TYPE>(mv_vertical_cx0.val[ch], mv_vertical_cx1.val[ch], mv_vertical_cx2.val[ch], mv_result.val[ch]);
            }
            vstore(dst + C * (x - elem_counts), mv_result);

            mv_vertical_cx0 = mv_vertical_cx1;
            mv_vertical_cx1 = mv_vertical_cx2;
        }
    }

    // remain
    {
        DT_S32 last = C * (width - 1);
        DT_S32 rest = width % elem_counts;
        MVType mv_last;

        vload(src_p0 + C * back_offset, mv_src_p0);
        vload(src_c  + C * back_offset, mv_src_c);
        vload(src_n0 + C * back_offset, mv_src_n0);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_p0.val[ch], src_p0[last + ch], src_p0[last + ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_c.val[ch],  src_c[last + ch],  src_c[last + ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_n0.val[ch], src_n0[last + ch], src_n0[last + ch]);

            Morph3x3VCore<Tp, MORPH_TYPE>(mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mv_vertical_cx2.val[ch]);
            Morph3x3VCore<Tp, MORPH_TYPE>(v_border_p0, v_border_c, v_border_n0, mv_vertical_cx3.val[ch]);

            Morph3x3HCore<Tp, MORPH_TYPE>(mv_vertical_cx0.val[ch], mv_vertical_cx1.val[ch], mv_vertical_cx2.val[ch], 
                                          mv_vertical_cx3.val[ch], mv_result.val[ch], mv_last.val[ch], rest);
        }

        vstore(dst + C * (back_offset - rest), mv_result);
        vstore(dst + C * back_offset, mv_last);
    }
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 C,
          typename std::enable_if<(MORPH_SHAPE == MorphShape::CROSS) || (MORPH_SHAPE == MorphShape::ELLIPSE)>::type* = DT_NULL>
DT_VOID Morph3x3Row(const Tp *src_p0, const Tp *src_c, const Tp *src_n0, Tp *dst, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    DT_S32 back_offset = width - elem_counts;

    MVType mv_src_p0, mv_src_c, mv_src_n0;
    MVType mv_src_cx0, mv_src_cx1, mv_src_cx2;
    MVType mv_vertical_px0, mv_vertical_px1;
    MVType mv_result;

    // left border
    {
        vload(src_p0, mv_src_p0);
        vload(src_c,  mv_src_c);
        vload(src_n0, mv_src_n0);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_cx0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_c.val[ch], src_c[ch], src_c[ch]);
            mv_src_cx1.val[ch] = mv_src_c.val[ch];
            Morph3x3VCore<Tp, MORPH_TYPE>(mv_src_p0.val[ch], mv_src_n0.val[ch], mv_vertical_px0.val[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (DT_S32 x = elem_counts; x <= back_offset; x += elem_counts)
        {
            vload(src_p0 + C * x, mv_src_p0);
            vload(src_c  + C * x, mv_src_c);
            vload(src_n0 + C * x, mv_src_n0);

            #pragma unroll(C)
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mv_src_cx2.val[ch] = mv_src_c.val[ch];
                Morph3x3HCore<Tp, MORPH_TYPE>(mv_vertical_px0.val[ch], mv_src_cx0.val[ch], mv_src_cx1.val[ch], mv_src_cx2.val[ch], mv_result.val[ch]);
                Morph3x3VCore<Tp, MORPH_TYPE>(mv_src_p0.val[ch], mv_src_n0.val[ch], mv_vertical_px0.val[ch]);
            }
            vstore(dst + C * (x - elem_counts), mv_result);

            mv_src_cx0 = mv_src_cx1;
            mv_src_cx1 = mv_src_cx2;
        }
    }

    // remain
    {
        DT_S32 last = C * (width - 1);
        DT_S32 rest = width % elem_counts;
        MVType mv_last;

        vload(src_p0 + C * back_offset, mv_src_p0);
        vload(src_c  + C * back_offset, mv_src_c);
        vload(src_n0 + C * back_offset, mv_src_n0);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_cx2.val[ch] = mv_src_c.val[ch];
            HVX_Vector v_border_right = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_c.val[ch], src_c[last + ch], src_c[last + ch]);
            Morph3x3VCore<Tp, MORPH_TYPE>(mv_src_p0.val[ch], mv_src_n0.val[ch], mv_vertical_px1.val[ch]);

            Morph3x3HCore<Tp, MORPH_TYPE>(mv_vertical_px0.val[ch], mv_vertical_px1.val[ch], mv_src_cx0.val[ch], mv_src_cx1.val[ch], mv_src_cx2.val[ch],
                                          v_border_right, mv_result.val[ch], mv_last.val[ch], rest);
        }

        vstore(dst + C * (back_offset - rest), mv_result);
        vstore(dst + C * back_offset, mv_last);
    }
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 C>
static Status Morph3x3HvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width   = src.GetSizes().m_width;
    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    const Tp *src_p0 = src.Ptr<Tp, BorderType::REPLICATE>(start_row - 1, DT_NULL);
    const Tp *src_c  = src.Ptr<Tp>(start_row);
    const Tp *src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 1, DT_NULL);

    DT_U64 L2fetch_param = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if (y + 2 < height)
        {
            L2Fetch(reinterpret_cast<DT_U64>(src.Ptr<Tp>(y + 2)), L2fetch_param);
        }

        Tp *dst_row  = dst.Ptr<Tp>(y);
        Morph3x3Row<Tp, MORPH_SHAPE, MORPH_TYPE, C>(src_p0, src_c, src_n0, dst_row, width);

        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(y + 2, DT_NULL);
    }

    return Status::OK;
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE>
static Status Morph3x3HvxHelper(Context *ctx, const Mat &src, Mat &dst)
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
            ret = wp->ParallelFor((DT_S32)0, height, Morph3x3HvxImpl<Tp, MORPH_SHAPE, MORPH_TYPE, 1>, std::cref(src), std::ref(dst));
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Morph3x3HvxImpl<Tp, MORPH_SHAPE, MORPH_TYPE, 2>, std::cref(src), std::ref(dst));
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Morph3x3HvxImpl<Tp, MORPH_SHAPE, MORPH_TYPE, 3>, std::cref(src), std::ref(dst));
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

template <typename Tp, MorphShape MORPH_SHAPE>
static Status Morph3x3HvxHelper(Context *ctx, const Mat &src, Mat &dst, MorphType type)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case MorphType::ERODE:
        {
            ret = Morph3x3HvxHelper<Tp, MORPH_SHAPE, MorphType::ERODE>(ctx, src, dst);
            break;
        }

        case MorphType::DILATE:
        {
            ret = Morph3x3HvxHelper<Tp, MORPH_SHAPE, MorphType::DILATE>(ctx, src, dst);
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
static Status Morph3x3HvxHelper(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape)
{
    Status ret = Status::ERROR;

    switch(shape)
    {
        case MorphShape::RECT:
        {
            ret = Morph3x3HvxHelper<Tp, MorphShape::RECT>(ctx, src, dst, type);
            break;
        }

        case MorphShape::CROSS:
        {
            ret = Morph3x3HvxHelper<Tp, MorphShape::CROSS>(ctx, src, dst, type);
            break;
        }

        case MorphShape::ELLIPSE:
        {
            ret = Morph3x3HvxHelper<Tp, MorphShape::ELLIPSE>(ctx, src, dst, type);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported morph shape");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Morph3x3Hvx(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Morph3x3HvxHelper<DT_U8>(ctx, src, dst, type, shape);
            break;
        }

        case ElemType::U16:
        {
            ret = Morph3x3HvxHelper<DT_U16>(ctx, src, dst, type, shape);
            break;
        }

        case ElemType::S16:
        {
            ret = Morph3x3HvxHelper<DT_S16>(ctx, src, dst, type, shape);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported source format");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura