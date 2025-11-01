#include "median_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp, typename VqType = typename neon::QVector<Tp>::VType>
AURA_ALWAYS_INLINE AURA_VOID Median3x3Core(VqType vq_src_p0l0, VqType vq_src_p0c, VqType vq_src_p0r0,
                                         VqType vq_src_c0l0, VqType vq_src_c0c, VqType vq_src_c0r0,
                                         VqType vq_src_c1l0, VqType vq_src_c1c, VqType vq_src_c1r0,
                                         VqType vq_src_n0l0, VqType vq_src_n0c, VqType vq_src_n0r0,
                                         VqType &vq_result0, VqType &vq_result1)
{
    MinMaxOp<VqType>(vq_src_c0c,  vq_src_c0r0);
    MinMaxOp<VqType>(vq_src_c1c,  vq_src_c1r0);
    MinMaxOp<VqType>(vq_src_c0l0, vq_src_c0c);
    MinMaxOp<VqType>(vq_src_c1l0, vq_src_c1c);
    MinMaxOp<VqType>(vq_src_c0c,  vq_src_c0r0);
    MinMaxOp<VqType>(vq_src_c1c,  vq_src_c1r0);
    MinMaxOp<VqType>(vq_src_c0c,  vq_src_c1c);
    vq_src_c0r0 = neon::vmin(vq_src_c0r0, vq_src_c1r0);

    MinMaxOp<VqType>(vq_src_p0c,  vq_src_p0r0);
    MinMaxOp<VqType>(vq_src_n0c,  vq_src_n0r0);
    MinMaxOp<VqType>(vq_src_p0l0, vq_src_p0c);
    MinMaxOp<VqType>(vq_src_n0l0, vq_src_n0c);
    MinMaxOp<VqType>(vq_src_p0c,  vq_src_p0r0);
    MinMaxOp<VqType>(vq_src_n0c,  vq_src_n0r0);

    vq_src_c1r0 = vq_src_c0l0;
    vq_src_c0l0 = neon::vmax(vq_src_p0l0, vq_src_c0l0);
    vq_src_p0l0 = vq_src_c1l0;

    vq_src_c1l0 = neon::vmax(vq_src_c0l0, vq_src_c1l0);
    vq_src_c1r0 = neon::vmax(vq_src_n0l0, vq_src_c1r0);
    vq_src_p0l0 = neon::vmax(vq_src_c1r0, vq_src_p0l0);

    vq_src_c1r0 = vq_src_c0r0;
    vq_src_p0r0 = neon::vmin(vq_src_p0r0, vq_src_c0r0);
    vq_src_n0r0 = neon::vmin(vq_src_n0r0, vq_src_c1r0);

    vq_src_c1r0 = vq_src_c0c;
    vq_src_c0c  = neon::vmax(vq_src_p0c, vq_src_c0c);
    vq_src_p0c  = vq_src_c1c;

    vq_src_c0c  = neon::vmin(vq_src_c0c,  vq_src_c1c);
    vq_src_c1r0 = neon::vmax(vq_src_n0c,  vq_src_c1r0);
    vq_src_c1r0 = neon::vmin(vq_src_c1r0, vq_src_p0c);

    MinMaxOp<VqType>(vq_src_c0c,  vq_src_p0r0);
    MinMaxOp<VqType>(vq_src_c1r0, vq_src_n0r0);

    vq_src_c0c  = neon::vmax(vq_src_c1l0, vq_src_c0c);
    vq_src_c0c  = neon::vmin(vq_src_c0c,  vq_src_p0r0);
    vq_src_c1r0 = neon::vmax(vq_src_p0l0, vq_src_c1r0);
    vq_src_c1r0 = neon::vmin(vq_src_c1r0, vq_src_n0r0);

    vq_result0 = vq_src_c0c;
    vq_result1 = vq_src_c1r0;
}

template <typename Tp, typename VqType = typename neon::QVector<Tp>::VType>
AURA_ALWAYS_INLINE AURA_VOID Median3x3Vector(VqType &vq_src_p0x0, VqType &vq_src_p0x1, VqType &vq_src_p0x2,
                                           VqType &vq_src_c0x0, VqType &vq_src_c0x1, VqType &vq_src_c0x2,
                                           VqType &vq_src_c1x0, VqType &vq_src_c1x1, VqType &vq_src_c1x2,
                                           VqType &vq_src_n0x0, VqType &vq_src_n0x1, VqType &vq_src_n0x2,
                                           VqType &vq_result0,  VqType &vq_result1)
{
    constexpr MI_S32 ELEM_COUNTS = static_cast<MI_S32>(16 / sizeof(Tp));

    VqType vq_src_p0l0 = neon::vext<ELEM_COUNTS - 1>(vq_src_p0x0, vq_src_p0x1);
    VqType vq_src_p0r0 = neon::vext<1>(vq_src_p0x1, vq_src_p0x2);

    VqType vq_src_c0l0 = neon::vext<ELEM_COUNTS - 1>(vq_src_c0x0, vq_src_c0x1);
    VqType vq_src_c0r0 = neon::vext<1>(vq_src_c0x1, vq_src_c0x2);

    VqType vq_src_c1l0 = neon::vext<ELEM_COUNTS - 1>(vq_src_c1x0, vq_src_c1x1);
    VqType vq_src_c1r0 = neon::vext<1>(vq_src_c1x1, vq_src_c1x2);

    VqType vq_src_n0l0 = neon::vext<ELEM_COUNTS - 1>(vq_src_n0x0, vq_src_n0x1);
    VqType vq_src_n0r0 = neon::vext<1>(vq_src_n0x1, vq_src_n0x2);

    Median3x3Core<Tp>(vq_src_p0l0, vq_src_p0x1, vq_src_p0r0,
                      vq_src_c0l0, vq_src_c0x1, vq_src_c0r0,
                      vq_src_c1l0, vq_src_c1x1, vq_src_c1r0,
                      vq_src_n0l0, vq_src_n0x1, vq_src_n0r0,
                      vq_result0,  vq_result1);

    vq_src_p0x0 = vq_src_p0x1;
    vq_src_c0x0 = vq_src_c0x1;
    vq_src_c1x0 = vq_src_c1x1;
    vq_src_n0x0 = vq_src_n0x1;

    vq_src_p0x1 = vq_src_p0x2;
    vq_src_c0x1 = vq_src_c0x2;
    vq_src_c1x1 = vq_src_c1x2;
    vq_src_n0x1 = vq_src_n0x2;
}

template <typename Tp, MI_S32 C>
static AURA_VOID Median3x3TwoRows(const Tp *src_p0, const Tp *src_c0, const Tp *src_c1, const Tp *src_n0,
                                Tp *dst_c0, Tp *dst_c1, MI_S32 width)
{
    using MVqType = typename neon::MQVector<Tp, C>::MVType;

    constexpr MI_S32 ELEM_COUNTS = static_cast<MI_S32>(16 / sizeof(Tp));
    constexpr MI_S32 VOFFSET = ELEM_COUNTS * C;
    const MI_S32 width_align = (width & -ELEM_COUNTS) * C;

    MVqType mvq_src_p0[3], mvq_src_c0[3], mvq_src_c1[3], mvq_src_n0[3];
    MVqType mvq_result0, mvq_result1;

    // left border
    {
        neon::vload(src_p0,           mvq_src_p0[1]);
        neon::vload(src_c0,           mvq_src_c0[1]);
        neon::vload(src_c1,           mvq_src_c1[1]);
        neon::vload(src_n0,           mvq_src_n0[1]);
        neon::vload(src_p0 + VOFFSET, mvq_src_p0[2]);
        neon::vload(src_c0 + VOFFSET, mvq_src_c0[2]);
        neon::vload(src_c1 + VOFFSET, mvq_src_c1[2]);
        neon::vload(src_n0 + VOFFSET, mvq_src_n0[2]);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p0[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_p0[1].val[ch], src_p0[ch], src_p0[ch]);
            mvq_src_c0[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_c0[1].val[ch], src_c0[ch], src_c0[ch]);
            mvq_src_c1[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_c1[1].val[ch], src_c1[ch], src_c1[ch]);
            mvq_src_n0[0].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::LEFT>(mvq_src_n0[1].val[ch], src_n0[ch], src_n0[ch]);

            Median3x3Vector<Tp>(mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                mvq_src_c0[0].val[ch], mvq_src_c0[1].val[ch], mvq_src_c0[2].val[ch],
                                mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                mvq_result0.val[ch], mvq_result1.val[ch]);
        }
        neon::vstore(dst_c0, mvq_result0);
        neon::vstore(dst_c1, mvq_result1);
    }

    // middle
    {
        for (MI_S32 x = VOFFSET; x < width_align - VOFFSET; x += VOFFSET)
        {
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c0 + x + VOFFSET, mvq_src_c0[2]);
            neon::vload(src_c1 + x + VOFFSET, mvq_src_c1[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Median3x3Vector<Tp>(mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                    mvq_src_c0[0].val[ch], mvq_src_c0[1].val[ch], mvq_src_c0[2].val[ch],
                                    mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                    mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                    mvq_result0.val[ch], mvq_result1.val[ch]);
            }
            neon::vstore(dst_c0 + x, mvq_result0);
            neon::vstore(dst_c1 + x, mvq_result1);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            MI_S32 x = (width - (ELEM_COUNTS << 1)) * C;
            neon::vload(src_p0 + x - VOFFSET, mvq_src_p0[0]);
            neon::vload(src_c0 + x - VOFFSET, mvq_src_c0[0]);
            neon::vload(src_c1 + x - VOFFSET, mvq_src_c1[0]);
            neon::vload(src_n0 + x - VOFFSET, mvq_src_n0[0]);
            neon::vload(src_p0 + x,           mvq_src_p0[1]);
            neon::vload(src_c0 + x,           mvq_src_c0[1]);
            neon::vload(src_c1 + x,           mvq_src_c1[1]);
            neon::vload(src_n0 + x,           mvq_src_n0[1]);
            neon::vload(src_p0 + x + VOFFSET, mvq_src_p0[2]);
            neon::vload(src_c0 + x + VOFFSET, mvq_src_c0[2]);
            neon::vload(src_c1 + x + VOFFSET, mvq_src_c1[2]);
            neon::vload(src_n0 + x + VOFFSET, mvq_src_n0[2]);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Median3x3Vector<Tp>(mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                    mvq_src_c0[0].val[ch], mvq_src_c0[1].val[ch], mvq_src_c0[2].val[ch],
                                    mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                    mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                    mvq_result0.val[ch], mvq_result1.val[ch]);
            }
            neon::vstore(dst_c0 + x, mvq_result0);
            neon::vstore(dst_c1 + x, mvq_result1);
        }
    }

    // right border
    {
        MI_S32 x    = (width - ELEM_COUNTS) * C;
        MI_S32 last = (width - 1) * C;

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mvq_src_p0[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_p0[1].val[ch], src_p0[last + ch], src_p0[last + ch]);
            mvq_src_c0[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_c0[1].val[ch], src_c0[last + ch], src_c0[last + ch]);
            mvq_src_c1[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_c1[1].val[ch], src_c1[last + ch], src_c1[last + ch]);
            mvq_src_n0[2].val[ch] = GetBorderVector<BorderType::REPLICATE, BorderArea::RIGHT>(mvq_src_n0[1].val[ch], src_n0[last + ch], src_n0[last + ch]);

            Median3x3Vector<Tp>(mvq_src_p0[0].val[ch], mvq_src_p0[1].val[ch], mvq_src_p0[2].val[ch],
                                mvq_src_c0[0].val[ch], mvq_src_c0[1].val[ch], mvq_src_c0[2].val[ch],
                                mvq_src_c1[0].val[ch], mvq_src_c1[1].val[ch], mvq_src_c1[2].val[ch],
                                mvq_src_n0[0].val[ch], mvq_src_n0[1].val[ch], mvq_src_n0[2].val[ch],
                                mvq_result0.val[ch], mvq_result1.val[ch]);
        }
        neon::vstore(dst_c0 + x, mvq_result0);
        neon::vstore(dst_c1 + x, mvq_result1);
    }
}

template <typename Tp, MI_S32 C>
static Status Median3x3NeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width = dst.GetSizes().m_width;
    MI_S32 y = start_row;

    const Tp *src_p  = src.Ptr<Tp, BorderType::REPLICATE>(y - 1);
    const Tp *src_c0 = src.Ptr<Tp>(y);
    const Tp *src_c1 = src.Ptr<Tp, BorderType::REPLICATE>(y + 1);
    const Tp *src_n  = src.Ptr<Tp, BorderType::REPLICATE>(y + 2);

    MI_S32 h_align2 = (end_row - start_row) & (-2);
    for (; y < start_row + h_align2; y += 2)
    {
        Tp *dst_c0 = dst.Ptr<Tp>(y);
        Tp *dst_c1 = dst.Ptr<Tp, BorderType::REPLICATE>(y + 1);
        Median3x3TwoRows<Tp, C>(src_p, src_c0, src_c1, src_n, dst_c0, dst_c1, width);

        src_p  = src_c1;
        src_c0 = src_n;
        src_c1 = src.Ptr<Tp, BorderType::REPLICATE>(y + 3);
        src_n  = src.Ptr<Tp, BorderType::REPLICATE>(y + 4);
    }

    if (y < end_row)
    {
        y--;
        src_p      = src.Ptr<Tp, BorderType::REPLICATE>(y - 1);
        src_c0     = src.Ptr<Tp, BorderType::REPLICATE>(y    );
        src_c1     = src.Ptr<Tp, BorderType::REPLICATE>(y + 1);
        src_n      = src.Ptr<Tp, BorderType::REPLICATE>(y + 2);
        Tp *dst_c0 = dst.Ptr<Tp, BorderType::REPLICATE>(y    );
        Tp *dst_c1 = dst.Ptr<Tp, BorderType::REPLICATE>(y + 1);

        Median3x3TwoRows<Tp, C>(src_p, src_c0, src_c1, src_n, dst_c0, dst_c1, width);
    }

    return Status::OK;
}

template <typename Tp>
static Status Median3x3NeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);

    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    MI_S32 height = dst.GetSizes().m_height;

    switch (dst.GetSizes().m_channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, Median3x3NeonImpl<Tp, 1>, std::cref(src), std::ref(dst));
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Median3x3NeonImpl<Tp, 2>, std::cref(src), std::ref(dst));
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Median3x3NeonImpl<Tp, 3>, std::cref(src), std::ref(dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported channel");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Median3x3Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Median3x3NeonHelper<MI_U8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median3x3NeonHelper<MI_U8> failed");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = Median3x3NeonHelper<MI_S8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median3x3NeonHelper<MI_S8> failed");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = Median3x3NeonHelper<MI_U16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median3x3NeonHelper<MI_U16> failed");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = Median3x3NeonHelper<MI_S16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median3x3NeonHelper<MI_S16> failed");
            }
            break;
        }

        case ElemType::U32:
        {
            ret = Median3x3NeonHelper<MI_U32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median3x3NeonHelper<MI_U32> failed");
            }
            break;
        }

        case ElemType::S32:
        {
            ret = Median3x3NeonHelper<MI_S32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median3x3NeonHelper<MI_S32> failed");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = Median3x3NeonHelper<MI_F16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median3x3NeonHelper<MI_F16> failed");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = Median3x3NeonHelper<MI_F32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Median3x3NeonHelper<MI_F32> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura