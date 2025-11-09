#include "median_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
AURA_ALWAYS_INLINE DT_VOID Median3x3Core(HVX_Vector &v_src_p0x0, HVX_Vector &v_src_p0x1, HVX_Vector &v_src_p0x2,
                                         HVX_Vector &v_src_c0x0, HVX_Vector &v_src_c0x1, HVX_Vector &v_src_c0x2,
                                         HVX_Vector &v_src_c1x0, HVX_Vector &v_src_c1x1, HVX_Vector &v_src_c1x2,
                                         HVX_Vector &v_src_n0x0, HVX_Vector &v_src_n0x1, HVX_Vector &v_src_n0x2,
                                         HVX_Vector &v_result0,  HVX_Vector &v_result1)
{
    HVX_Vector v_src_p0l = Q6_V_vlalign_VVR(v_src_p0x1, v_src_p0x0, sizeof(Tp));
    HVX_Vector v_src_p0c = v_src_p0x1;
    HVX_Vector v_src_p0r = Q6_V_valign_VVR(v_src_p0x2, v_src_p0x1, sizeof(Tp));
    HVX_Vector v_src_c0l = Q6_V_vlalign_VVR(v_src_c0x1, v_src_c0x0, sizeof(Tp));
    HVX_Vector v_src_c0c = v_src_c0x1;
    HVX_Vector v_src_c0r = Q6_V_valign_VVR(v_src_c0x2, v_src_c0x1, sizeof(Tp));
    HVX_Vector v_src_c1l = Q6_V_vlalign_VVR(v_src_c1x1, v_src_c1x0, sizeof(Tp));
    HVX_Vector v_src_c1c = v_src_c1x1;
    HVX_Vector v_src_c1r = Q6_V_valign_VVR(v_src_c1x2, v_src_c1x1, sizeof(Tp));
    HVX_Vector v_src_n0l = Q6_V_vlalign_VVR(v_src_n0x1, v_src_n0x0, sizeof(Tp));
    HVX_Vector v_src_n0c = v_src_n0x1;
    HVX_Vector v_src_n0r = Q6_V_valign_VVR(v_src_n0x2, v_src_n0x1, sizeof(Tp));

    // step1 Get minmax  from 14   delete 5 18
    VectorMinMax<Tp>(v_src_c0l, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r, v_src_c1l);
    VectorMinMax<Tp>(v_src_c1c, v_src_c1r);
    VectorMinMax<Tp>(v_src_c0l, v_src_c0r);
    VectorMinMax<Tp>(v_src_c0l, v_src_c1c);
    VectorMinMax<Tp>(v_src_c0c, v_src_c1l);
    VectorMinMax<Tp>(v_src_c1l, v_src_c1r);

    HVX_Vector v_reuse[4] = {v_src_c0c, v_src_c0r, v_src_c1l, v_src_c1c};

    //step3 up dst  Get minmax  from 6 + 1  delete 0 11
    VectorMinMax<Tp>(v_src_p0l, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r, v_src_c1l);
    VectorMinMax<Tp>(v_src_p0l, v_src_c0r);
    VectorMinMax<Tp>(v_src_p0l, v_src_c1c);
    VectorMinMax<Tp>(v_src_c0c, v_src_c1l);
    VectorMinMax<Tp>(v_src_c1l, v_src_c1c);

    //step3 up dst  Get minmax  from 5 + 1  delete 1 10
    VectorMinMax<Tp>(v_src_p0c, v_src_c0c);
    VectorMinMax<Tp>(v_src_c0r, v_src_c1l);
    VectorMinMax<Tp>(v_src_p0c, v_src_c0r);
    VectorMinMax<Tp>(v_src_c0c, v_src_c1l);

    //step3 up dst  Get minmax  from 5 + 1  delete 2 9
    VectorMinMax<Tp>(v_src_p0r, v_src_c0c);
    VectorMinMax<Tp>(v_src_p0r, v_src_c0r);
    VectorMinMax<Tp>(v_src_c0c, v_src_c0r);

    //part2
    //step3 up dst  Get minmax  from 6 + 1  delete 25 17
    VectorMinMax<Tp>(v_src_n0l,  v_reuse[0]);
    VectorMinMax<Tp>(v_reuse[1], v_reuse[2]);
    VectorMinMax<Tp>(v_src_n0l,  v_reuse[1]);
    VectorMinMax<Tp>(v_src_n0l,  v_reuse[3]);
    VectorMinMax<Tp>(v_reuse[0], v_reuse[2]);
    VectorMinMax<Tp>(v_reuse[2], v_reuse[3]);

    //step3 up dst  Get minmax  from 6 + 1  delete 26 16
    VectorMinMax<Tp>(v_src_n0c,  v_reuse[0]);
    VectorMinMax<Tp>(v_reuse[1], v_reuse[2]);
    VectorMinMax<Tp>(v_src_n0c,  v_reuse[1]);
    VectorMinMax<Tp>(v_reuse[0], v_reuse[2]);

    //step3 up dst  Get minmax  from 6 + 1  delete 27 15
    VectorMinMax<Tp>(v_src_n0r,  v_reuse[0]);
    VectorMinMax<Tp>(v_src_n0r,  v_reuse[1]);
    VectorMinMax<Tp>(v_reuse[0], v_reuse[1]);

    v_result0 = v_src_c0c;
    v_result1 = v_reuse[0];
}

template <typename Tp, DT_S32 C>
static DT_VOID Median3x3TwoRow(const Tp *src_p0, const Tp *src_c0, const Tp *src_c1,
                               const Tp *src_n0, Tp *dst_c0, Tp *dst_c1, DT_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    DT_S32 back_offset = width - elem_counts;

    MVType mv_src_p0x0, mv_src_p0x1, mv_src_p0x2;
    MVType mv_src_c0x0, mv_src_c0x1, mv_src_c0x2;
    MVType mv_src_c1x0, mv_src_c1x1, mv_src_c1x2;
    MVType mv_src_n0x0, mv_src_n0x1, mv_src_n0x2;

    MVType mv_result0, mv_result1;
    // left
    {
        vload(src_p0, mv_src_p0x1);
        vload(src_c0, mv_src_c0x1);
        vload(src_c1, mv_src_c1x1);
        vload(src_n0, mv_src_n0x1);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p0x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_p0x1.val[ch], src_p0[ch], 1);
            mv_src_c0x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_c0x1.val[ch], src_c0[ch], 1);
            mv_src_c1x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_c1x1.val[ch], src_c1[ch], 1);
            mv_src_n0x0.val[ch] = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::LEFT>(mv_src_n0x1.val[ch], src_n0[ch], 1);
        }
    }

    // middle
    for (DT_S32 x = elem_counts; x <= back_offset; x += elem_counts)
    {
        vload(src_p0 + C * x, mv_src_p0x2);
        vload(src_c0 + C * x, mv_src_c0x2);
        vload(src_c1 + C * x, mv_src_c1x2);
        vload(src_n0 + C * x, mv_src_n0x2);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            Median3x3Core<Tp>(mv_src_p0x0.val[ch], mv_src_p0x1.val[ch], mv_src_p0x2.val[ch],
                              mv_src_c0x0.val[ch], mv_src_c0x1.val[ch], mv_src_c0x2.val[ch],
                              mv_src_c1x0.val[ch], mv_src_c1x1.val[ch], mv_src_c1x2.val[ch],
                              mv_src_n0x0.val[ch], mv_src_n0x1.val[ch], mv_src_n0x2.val[ch],
                              mv_result0.val[ch],  mv_result1.val[ch]);
        }

        vstore(dst_c0 + C * (x - elem_counts), mv_result0);
        vstore(dst_c1 + C * (x - elem_counts), mv_result1);

        mv_src_p0x0 = mv_src_p0x1;
        mv_src_p0x1 = mv_src_p0x2;
        mv_src_c0x0 = mv_src_c0x1;
        mv_src_c0x1 = mv_src_c0x2;
        mv_src_c1x0 = mv_src_c1x1;
        mv_src_c1x1 = mv_src_c1x2;
        mv_src_n0x0 = mv_src_n0x1;
        mv_src_n0x1 = mv_src_n0x2;
    }

    // remain
    {
        DT_S32 last = (width - 1) * C;
        DT_S32 rest = width % elem_counts;
        MVType mv_last_result0, mv_last_result1;

        vload(src_p0 + C * back_offset, mv_src_p0x2);
        vload(src_c0 + C * back_offset, mv_src_c0x2);
        vload(src_c1 + C * back_offset, mv_src_c1x2);
        vload(src_n0 + C * back_offset, mv_src_n0x2);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_p0x2.val[ch], src_p0[last + ch], 1);
            HVX_Vector v_border_c0 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_c0x2.val[ch], src_c0[last + ch], 1);
            HVX_Vector v_border_c1 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_c1x2.val[ch], src_c1[last + ch], 1);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(mv_src_n0x2.val[ch], src_n0[last + ch], 1);

            HVX_Vector v_src_p0r = Q6_V_vlalign_VVR(v_border_p0, mv_src_p0x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_c0r = Q6_V_vlalign_VVR(v_border_c0, mv_src_c0x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_c1r = Q6_V_vlalign_VVR(v_border_c1, mv_src_c1x2.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_n0r = Q6_V_vlalign_VVR(v_border_n0, mv_src_n0x2.val[ch], rest * sizeof(Tp));

            Median3x3Core<Tp>(mv_src_p0x0.val[ch], mv_src_p0x1.val[ch], v_src_p0r,
                              mv_src_c0x0.val[ch], mv_src_c0x1.val[ch], v_src_c0r,
                              mv_src_c1x0.val[ch], mv_src_c1x1.val[ch], v_src_c1r,
                              mv_src_n0x0.val[ch], mv_src_n0x1.val[ch], v_src_n0r,
                              mv_result0.val[ch],  mv_result1.val[ch]);

            HVX_Vector v_src_p0l = Q6_V_valign_VVR(mv_src_p0x1.val[ch], mv_src_p0x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_c0l = Q6_V_valign_VVR(mv_src_c0x1.val[ch], mv_src_c0x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_c1l = Q6_V_valign_VVR(mv_src_c1x1.val[ch], mv_src_c1x0.val[ch], rest * sizeof(Tp));
            HVX_Vector v_src_n0l = Q6_V_valign_VVR(mv_src_n0x1.val[ch], mv_src_n0x0.val[ch], rest * sizeof(Tp));

            Median3x3Core<Tp>(v_src_p0l, mv_src_p0x2.val[ch], v_border_p0,
                              v_src_c0l, mv_src_c0x2.val[ch], v_border_c0,
                              v_src_c1l, mv_src_c1x2.val[ch], v_border_c1,
                              v_src_n0l, mv_src_n0x2.val[ch], v_border_n0,
                              mv_last_result0.val[ch], mv_last_result1.val[ch]);
        }
        vstore(dst_c0 + C * (back_offset - rest), mv_result0);
        vstore(dst_c0 + C * back_offset,          mv_last_result0);
        vstore(dst_c1 + C * (back_offset - rest), mv_result1);
        vstore(dst_c1 + C * back_offset,          mv_last_result1);
    }
}

template <typename Tp, DT_S32 C>
static Status Median3x3HvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 width   = src.GetSizes().m_width;
    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;

    const Tp *src_p0 = src.Ptr<Tp, BorderType::REPLICATE>(start_row - 1);
    const Tp *src_c0 = src.Ptr<Tp>(start_row);
    const Tp *src_c1 = src.Ptr<Tp>(start_row + 1);
    const Tp *src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 2);

    DT_U64 L2fetch_param = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 2, 0);
    DT_S32 y;

    for (y = start_row; y < end_row - 1; y += 2)
    {
        if (y + 2 < height)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<Tp>(y + 2)), L2fetch_param);
        }

        Tp *dst_c0 = dst.Ptr<Tp>(y);
        Tp *dst_c1 = dst.Ptr<Tp, BorderType::REPLICATE>(y + 1);

        Median3x3TwoRow<Tp, C>(src_p0, src_c0, src_c1, src_n0, dst_c0, dst_c1, width);

        src_p0 = src_c1;
        src_c0 = src_n0;
        src_c1 = src.Ptr<Tp, BorderType::REPLICATE>(y + 3);
        src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(y + 4);
    }

    if (y == end_row - 1)
    {
        src_p0 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 3);
        src_c0 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 2);
        src_c1 = src.Ptr<Tp, BorderType::REPLICATE>(end_row - 1);
        src_n0 = src.Ptr<Tp, BorderType::REPLICATE>(end_row);

        Tp *dst_c0 = dst.Ptr<Tp, BorderType::REPLICATE>(end_row - 2);
        Tp *dst_c1 = dst.Ptr<Tp, BorderType::REPLICATE>(end_row - 1);

        Median3x3TwoRow<Tp, C>(src_p0, src_c0, src_c1, src_n0, dst_c0, dst_c1, width);
    }

    return Status::OK;
}

template<typename Tp>
static Status Median3x3HvxHelper(Context *ctx, const Mat &src, Mat &dst)
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
            ret = wp->ParallelFor((DT_S32)0, height, Median3x3HvxImpl<Tp, 1>, src, dst);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Median3x3HvxImpl<Tp, 2>, src, dst);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Median3x3HvxImpl<Tp, 3>, src, dst);
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

Status Median3x3Hvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Median3x3HvxHelper<DT_U8>(ctx, src, dst);
            break;
        }

        case ElemType::S8:
        {
            ret = Median3x3HvxHelper<DT_S8>(ctx, src, dst);
            break;
        }

        case ElemType::U16:
        {
            ret = Median3x3HvxHelper<DT_U16>(ctx, src, dst);
            break;
        }

        case ElemType::S16:
        {
            ret = Median3x3HvxHelper<DT_S16>(ctx, src, dst);
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