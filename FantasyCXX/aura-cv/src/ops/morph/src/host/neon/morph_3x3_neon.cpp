#include "morph_impl.hpp"
#include "aura/runtime/worker_pool.h"

namespace aura
{

#define QVECTOR_NUM         (3)

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 ELEM_COUNTS,
          typename VqTpye = typename neon::QVector<Tp>::VType,
          typename std::enable_if<MorphShape::RECT == MORPH_SHAPE, Tp>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Morph3x3Vector(VqTpye &vq_src_px0, VqTpye &vq_src_px1, VqTpye &vq_src_px2,
                                          VqTpye &vq_src_cx0, VqTpye &vq_src_cx1, VqTpye &vq_src_cx2,
                                          VqTpye &vq_src_nx0, VqTpye &vq_src_nx1, VqTpye &vq_src_nx2,
                                          VqTpye &vq_result,  VqTpye *vq_vertical_results)
{
    // vertical results
    vq_vertical_results[0] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_px0, vq_src_cx0, vq_src_nx0);
    vq_vertical_results[1] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_px1, vq_src_cx1, vq_src_nx1);
    vq_vertical_results[2] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_px2, vq_src_cx2, vq_src_nx2);

    // horizonal results
    VqTpye vq_vertical_l = neon::vext<ELEM_COUNTS - 1>(vq_vertical_results[0], vq_vertical_results[1]);
    VqTpye vq_vertical_r = neon::vext<1>(vq_vertical_results[1], vq_vertical_results[2]);

    vq_result = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_vertical_l, vq_vertical_results[1], vq_vertical_r);

    // slide results
    vq_vertical_results[0] = vq_vertical_results[1];
    vq_vertical_results[1] = vq_vertical_results[2];
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 ELEM_COUNTS,
          typename VqTpye = typename neon::QVector<Tp>::VType,
          typename std::enable_if<MorphShape::RECT == MORPH_SHAPE, Tp>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Morph3x3Vector(VqTpye &vq_src_px2, VqTpye &vq_src_cx0, VqTpye &vq_src_cx1, VqTpye &vq_src_cx2,
                                          VqTpye &vq_src_nx2, VqTpye &vq_result,  VqTpye *vq_vertical_results)
{
    AURA_UNUSED(vq_src_cx0);
    AURA_UNUSED(vq_src_cx1);

    // vertical results
    vq_vertical_results[2] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_px2, vq_src_cx2, vq_src_nx2);

    // horizonal results
    VqTpye vq_vertical_l = neon::vext<ELEM_COUNTS - 1>(vq_vertical_results[0], vq_vertical_results[1]);
    VqTpye vq_vertical_r = neon::vext<1>(vq_vertical_results[1], vq_vertical_results[2]);

    vq_result = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_vertical_l, vq_vertical_results[1], vq_vertical_r);

    // slide results
    vq_vertical_results[0] = vq_vertical_results[1];
    vq_vertical_results[1] = vq_vertical_results[2];
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 ELEM_COUNTS,
          typename VqTpye = typename neon::QVector<Tp>::VType,
          typename std::enable_if<MorphShape::CROSS == MORPH_SHAPE || MorphShape::ELLIPSE == MORPH_SHAPE, Tp>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Morph3x3Vector(VqTpye &vq_src_px0, VqTpye &vq_src_px1, VqTpye &vq_src_px2,
                                          VqTpye &vq_src_cx0, VqTpye &vq_src_cx1, VqTpye &vq_src_cx2,
                                          VqTpye &vq_src_nx0, VqTpye &vq_src_nx1, VqTpye &vq_src_nx2,
                                          VqTpye &vq_result,  VqTpye *vq_vertical_results)
{
    AURA_UNUSED(vq_src_px0);
    AURA_UNUSED(vq_src_nx0);

    // vertical results
    vq_vertical_results[1] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_px1, vq_src_cx1, vq_src_nx1);

    // horizonal results
    VqTpye vq_src_l = neon::vext<ELEM_COUNTS - 1>(vq_src_cx0, vq_src_cx1);
    VqTpye vq_src_r = neon::vext<1>(vq_src_cx1, vq_src_cx2);

    vq_result = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_l, vq_vertical_results[1], vq_src_r);

    // vertical results
    vq_vertical_results[1] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_px2, vq_src_cx2, vq_src_nx2);

    // slide src
    vq_src_cx0 = vq_src_cx1;
    vq_src_cx1 = vq_src_cx2;
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 ELEM_COUNTS,
          typename VqTpye = typename neon::QVector<Tp>::VType,
          typename std::enable_if<MorphShape::CROSS == MORPH_SHAPE || MorphShape::ELLIPSE == MORPH_SHAPE, Tp>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Morph3x3Vector(VqTpye &vq_src_px2, VqTpye &vq_src_cx0, VqTpye &vq_src_cx1, VqTpye &vq_src_cx2,
                                          VqTpye &vq_src_nx2, VqTpye &vq_result,  VqTpye *vq_vertical_results)
{
    // horizonal results
    VqTpye vq_src_l = neon::vext<ELEM_COUNTS - 1>(vq_src_cx0, vq_src_cx1);
    VqTpye vq_src_r = neon::vext<1>(vq_src_cx1, vq_src_cx2);

    vq_result = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_l, vq_vertical_results[1], vq_src_r);

    // vertical results
    vq_vertical_results[1] = MorphNeonMinMax<Tp, MORPH_TYPE>(vq_src_px2, vq_src_cx2, vq_src_nx2);

    // slide src
    vq_src_cx0 = vq_src_cx1;
    vq_src_cx1 = vq_src_cx2;
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 C>
static DT_VOID Morph3x3Row(const Tp *src_p, const Tp *src_c, const Tp *src_n, Tp *dst, DT_S32 width)
{
    using MVqTpye = typename neon::MQVector<Tp, C>::MVType;
    using VqTpye  = typename neon::QVector<Tp>::VType;

    constexpr DT_S32 elem_counts = static_cast<DT_S32>(16 / sizeof(Tp));
    constexpr DT_S32 voffset     = elem_counts * C;
    const DT_S32 width_align     = (width & -elem_counts) * C;

    MVqTpye mvq_src_p[3], mvq_src_c[3], mvq_src_n[3], mvq_result;
    VqTpye  vq_vertical_results[QVECTOR_NUM * C];

    // left
    {
        neon::vload(src_p,           mvq_src_p[1]);
        neon::vload(src_p + voffset, mvq_src_p[2]);
        neon::vload(src_c,           mvq_src_c[1]);
        neon::vload(src_c + voffset, mvq_src_c[2]);
        neon::vload(src_n,           mvq_src_n[1]);
        neon::vload(src_n + voffset, mvq_src_n[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            neon::vdup(mvq_src_p[0].val[ch], src_p[ch]);
            neon::vdup(mvq_src_c[0].val[ch], src_c[ch]);
            neon::vdup(mvq_src_n[0].val[ch], src_n[ch]);

            Morph3x3Vector<Tp, MORPH_SHAPE, MORPH_TYPE, elem_counts>(mvq_src_p[0].val[ch], mvq_src_p[1].val[ch], mvq_src_p[2].val[ch],
                                                                     mvq_src_c[0].val[ch], mvq_src_c[1].val[ch], mvq_src_c[2].val[ch],
                                                                     mvq_src_n[0].val[ch], mvq_src_n[1].val[ch], mvq_src_n[2].val[ch],
                                                                     mvq_result.val[ch], vq_vertical_results + ch * QVECTOR_NUM);
        }
        neon::vstore(dst, mvq_result);
    }

    // middle
    {
        for (DT_S32 x = voffset; x < width_align - voffset; x += voffset)
        {
            neon::vload(src_p + x + voffset, mvq_src_p[2]);
            neon::vload(src_c + x + voffset, mvq_src_c[2]);
            neon::vload(src_n + x + voffset, mvq_src_n[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Morph3x3Vector<Tp, MORPH_SHAPE, MORPH_TYPE, elem_counts>(mvq_src_p[2].val[ch], mvq_src_c[0].val[ch], mvq_src_c[1].val[ch],
                                                                         mvq_src_c[2].val[ch], mvq_src_n[2].val[ch],
                                                                         mvq_result.val[ch], vq_vertical_results + ch * QVECTOR_NUM);
            }
            neon::vstore(dst + x, mvq_result);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            DT_S32 x = (width - (elem_counts << 1)) * C;

            neon::vload(src_p + x - voffset, mvq_src_p[0]);
            neon::vload(src_p + x,           mvq_src_p[1]);
            neon::vload(src_p + x + voffset, mvq_src_p[2]);
            neon::vload(src_c + x - voffset, mvq_src_c[0]);
            neon::vload(src_c + x,           mvq_src_c[1]);
            neon::vload(src_c + x + voffset, mvq_src_c[2]);
            neon::vload(src_n + x - voffset, mvq_src_n[0]);
            neon::vload(src_n + x,           mvq_src_n[1]);
            neon::vload(src_n + x + voffset, mvq_src_n[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Morph3x3Vector<Tp, MORPH_SHAPE, MORPH_TYPE, elem_counts>(mvq_src_p[0].val[ch], mvq_src_p[1].val[ch], mvq_src_p[2].val[ch],
                                                                         mvq_src_c[0].val[ch], mvq_src_c[1].val[ch], mvq_src_c[2].val[ch],
                                                                         mvq_src_n[0].val[ch], mvq_src_n[1].val[ch], mvq_src_n[2].val[ch],
                                                                         mvq_result.val[ch], vq_vertical_results + ch * QVECTOR_NUM);
            }
            neon::vstore(dst + x, mvq_result);
        }
    }

    // right
    {
        DT_S32 x    = (width - elem_counts) * C;
        DT_S32 last = (width - 1) * C;

        mvq_src_p[1] = mvq_src_p[2];
        mvq_src_c[1] = mvq_src_c[2];
        mvq_src_n[1] = mvq_src_n[2];

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            neon::vdup(mvq_src_p[2].val[ch], src_p[last]);
            neon::vdup(mvq_src_c[2].val[ch], src_c[last]);
            neon::vdup(mvq_src_n[2].val[ch], src_n[last]);

            Morph3x3Vector<Tp, MORPH_SHAPE, MORPH_TYPE, elem_counts>(mvq_src_p[2].val[ch], mvq_src_c[0].val[ch], mvq_src_c[1].val[ch],
                                                                     mvq_src_c[2].val[ch], mvq_src_n[2].val[ch],
                                                                     mvq_result.val[ch], vq_vertical_results + ch * QVECTOR_NUM);
            last++;
        }
        neon::vstore(dst + x, mvq_result);
    }
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE, DT_S32 C>
static Status Morph3x3NeonImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    AURA_UNUSED(ctx);

    DT_S32 width = dst.GetSizes().m_width;

    DT_S32 y = start_row;

    const Tp *src_p = src.Ptr<Tp, BorderType::REPLICATE>(y - 1, DT_NULL);
    const Tp *src_c = src.Ptr<Tp>(y);
    const Tp *src_n = src.Ptr<Tp, BorderType::REPLICATE>(y + 1, DT_NULL);

    for (; y < end_row; y++)
    {
        Tp *dst_c = dst.Ptr<Tp>(y);
        Morph3x3Row<Tp, MORPH_SHAPE, MORPH_TYPE, C>(src_p, src_c, src_n, dst_c, width);

        src_p = src_c;
        src_c = src_n;
        src_n = src.Ptr<Tp, BorderType::REPLICATE>(y + 2, DT_NULL);
    }

    return Status::OK;
}

template <typename Tp, MorphShape MORPH_SHAPE, MorphType MORPH_TYPE>
static Status Morph3x3NeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    DT_S32 height = dst.GetSizes().m_height;

    switch (dst.GetSizes().m_channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, Morph3x3NeonImpl<Tp, MORPH_SHAPE, MORPH_TYPE, 1>, ctx, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonImpl<Tp, MORPH_SHAPE, 1> run failed!");
            }
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Morph3x3NeonImpl<Tp, MORPH_SHAPE, MORPH_TYPE, 2>, ctx, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonImpl<Tp, MORPH_SHAPE, 2> run failed!");
            }
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Morph3x3NeonImpl<Tp, MORPH_SHAPE, MORPH_TYPE, 3>, ctx, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonImpl<Tp, MORPH_SHAPE, 3> run failed!");
            }
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

template <typename Tp, MorphShape MORPH_SHAPE>
static Status Morph3x3NeonHelper(Context *ctx, const Mat &src, Mat &dst, MorphType type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case MorphType::ERODE:
        {
            ret = Morph3x3NeonHelper<Tp, MORPH_SHAPE, MorphType::ERODE>(ctx, src, dst, target);
            break;
        }

        case MorphType::DILATE:
        {
            ret = Morph3x3NeonHelper<Tp, MORPH_SHAPE, MorphType::DILATE>(ctx, src, dst, target);
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
static Status Morph3x3NeonHelper(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape,const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch(shape)
    {
        case MorphShape::RECT:
        {
            ret = Morph3x3NeonHelper<Tp, MorphShape::RECT>(ctx, src, dst, type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonHelper<Tp, MorphShape::RECT> run failed!");
            }
            break;
        }

        case MorphShape::CROSS:
        {
            ret = Morph3x3NeonHelper<Tp, MorphShape::CROSS>(ctx, src, dst, type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonHelper<Tp, MorphShape::CROSS> run failed!");
            }
            break;
        }

        case MorphShape::ELLIPSE:
        {
            ret = Morph3x3NeonHelper<Tp, MorphShape::ELLIPSE>(ctx, src, dst, type, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonHelper<Tp, MorphShape::ELLIPSE> run failed!");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported morph shape");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Morph3x3Neon(Context *ctx, const Mat &src, Mat &dst, MorphType type, MorphShape shape, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Morph3x3NeonHelper<DT_U8>(ctx, src, dst, type, shape, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonHelper<DT_U8> run failed!");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = Morph3x3NeonHelper<DT_U16>(ctx, src, dst, type, shape, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonHelper<DT_U16> run failed!");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = Morph3x3NeonHelper<DT_S16>(ctx, src, dst, type, shape, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonHelper<DT_S16> run failed!");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = Morph3x3NeonHelper<MI_F16>(ctx, src, dst, type, shape, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonHelper<MI_F16> run failed!");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = Morph3x3NeonHelper<DT_F32>(ctx, src, dst, type, shape, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Morph3x3NeonHelper<DT_F32> run failed!");
            }
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