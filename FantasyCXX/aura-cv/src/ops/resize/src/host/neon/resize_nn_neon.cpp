#include "resize_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

// Tp = MI_U8, MI_S8, MI_U16, MI_S16, MI_F16
template <typename Tp, typename MVType = typename std::conditional<1 == sizeof(Tp),
          typename neon::MDVector<Tp, 1>::MVType, typename neon::MQVector<Tp, 1>::MVType>::type>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value
                            || std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value
                            || std::is_same<MI_F16, Tp>::value, MVType>::type
ResizeNnNeonCore(const Tp *src_row, uint16x8_t &vqu16_sx1, MI_S32 channel,  MI_S32 c)
{
    MVType mv_dst;

    auto src_data = src_row[neon::vgetlane<0>(vqu16_sx1) * channel + c];
    mv_dst.val[0] = neon::vsetlane<0>(src_data, mv_dst.val[0]);
    src_data      = src_row[neon::vgetlane<1>(vqu16_sx1) * channel + c];
    mv_dst.val[0] = neon::vsetlane<1>(src_data, mv_dst.val[0]);
    src_data      = src_row[neon::vgetlane<2>(vqu16_sx1) * channel + c];
    mv_dst.val[0] = neon::vsetlane<2>(src_data, mv_dst.val[0]);
    src_data      = src_row[neon::vgetlane<3>(vqu16_sx1) * channel + c];
    mv_dst.val[0] = neon::vsetlane<3>(src_data, mv_dst.val[0]);
    src_data      = src_row[neon::vgetlane<4>(vqu16_sx1) * channel + c];
    mv_dst.val[0] = neon::vsetlane<4>(src_data, mv_dst.val[0]);
    src_data      = src_row[neon::vgetlane<5>(vqu16_sx1) * channel + c];
    mv_dst.val[0] = neon::vsetlane<5>(src_data, mv_dst.val[0]);
    src_data      = src_row[neon::vgetlane<6>(vqu16_sx1) * channel + c];
    mv_dst.val[0] = neon::vsetlane<6>(src_data, mv_dst.val[0]);
    src_data      = src_row[neon::vgetlane<7>(vqu16_sx1) * channel + c];
    mv_dst.val[0] = neon::vsetlane<7>(src_data, mv_dst.val[0]);

    return mv_dst;
}

// Tp = MI_U32, MI_S32
template <typename Tp>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<MI_U32, Tp>::value || std::is_same<MI_S32, Tp>::value
                            || std::is_same<MI_F32, Tp>::value, typename neon::MQVector<Tp, 1>::MVType>::type
ResizeNnNeonCore(const Tp *src_row, uint16x4_t &vdu16_sx1, MI_S32 channel,  MI_S32 c)
{
    using MVType = typename neon::MQVector<Tp, 1>::MVType;

    MVType mvq_dst;
    auto src_data  = src_row[neon::vgetlane<0>(vdu16_sx1) * channel + c];
    mvq_dst.val[0] = neon::vsetlane<0>(src_data, mvq_dst.val[0]);
    src_data       = src_row[neon::vgetlane<1>(vdu16_sx1) * channel + c];
    mvq_dst.val[0] = neon::vsetlane<1>(src_data, mvq_dst.val[0]);
    src_data       = src_row[neon::vgetlane<2>(vdu16_sx1) * channel + c];
    mvq_dst.val[0] = neon::vsetlane<2>(src_data, mvq_dst.val[0]);
    src_data       = src_row[neon::vgetlane<3>(vdu16_sx1) * channel + c];
    mvq_dst.val[0] = neon::vsetlane<3>(src_data, mvq_dst.val[0]);

    return mvq_dst;
}

// Tp = MI_U8, MI_S8, MI_U16, MI_S16, MI_F16
template <typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value
                || std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value
                || std::is_same<MI_F16, Tp>::value, Status>::type
ResizeNnNeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    using MVType = typename std::conditional<1 == sizeof(Tp), typename neon::MDVector<Tp, C>::MVType,
                   typename neon::MQVector<Tp, C>::MVType>::type;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 channel = dst.GetSizes().m_channel;
    MI_F32 scale_x = static_cast<MI_F64>(iwidth) / owidth;
    MI_F32 scale_y = static_cast<MI_F64>(iheight) / oheight;

    constexpr MI_S32 elem_counts = 8;
    MI_S32 width_align8 = owidth & (-elem_counts);
    MI_F32 tbl_l[4] = {0};
    MI_F32 tbl_h[4] = {0};

    for (MI_S32 i = 0; i < (elem_counts >> 1); i++)
    {
        tbl_l[i] = i * scale_x;
        tbl_h[i] = (i + 4) * scale_x;
    }

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        MI_S32 sy = Floor(y * scale_y);
        sy = Min(sy, iheight - 1);

        const Tp *src_row = src.Ptr<Tp>(sy);
        Tp *dst_row       = dst.Ptr<Tp>(y);

        MI_S32 x = 0;
        for (; x < width_align8; x += elem_counts)
        {
            MI_F32 sx = x * scale_x;

            float32x4_t vqf32_sx;
            neon::vdup(vqf32_sx, sx);
            float32x4_t vqf32_tbl_l = neon::vload1q(tbl_l);
            float32x4_t vqf32_tbl_h = neon::vload1q(tbl_h);
            float32x4_t vqf32_sx_l  = neon::vadd(vqf32_sx, vqf32_tbl_l);
            float32x4_t vqf32_sx_h  = neon::vadd(vqf32_sx, vqf32_tbl_h);

            uint32x4_t vqu32_sx_l        = neon::vcvt<MI_U32>(vqf32_sx_l);
            uint32x4_t vqu32_sx_h        = neon::vcvt<MI_U32>(vqf32_sx_h);
            uint16x8_t vqu16_sx          = neon::vcombine(neon::vmovn(vqu32_sx_l), neon::vmovn(vqu32_sx_h));
            uint16x8_t vqu16_width_bound;
            neon::vdup(vqu16_width_bound, static_cast<MI_U16>(iwidth - 1));
            uint16x8_t vqu16_sx1 = neon::vmin(vqu16_sx, vqu16_width_bound);

            MVType mv_dst;
            for (MI_S32 c = 0; c < channel; c++)
            {
                mv_dst.val[c] = ResizeNnNeonCore<Tp>(src_row, vqu16_sx1, channel, c).val[0];
            }
            neon::vstore(dst_row + x * channel, mv_dst);
        }

        for (; x < owidth; x++)
        {
            MI_S32 sx = Floor(x * scale_x);
            sx        = Min(sx, iwidth - 1);

            for (MI_S32 c = 0; c < channel; c++)
            {
                dst_row[x * channel + c] = src_row[sx * channel + c];
            }
        }
    }

    return Status::OK;
}

// Tp = MI_U32, MI_S32
template <typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U32, Tp>::value || std::is_same<MI_S32, Tp>::value
                || std::is_same<MI_F32, Tp>::value, Status>::type
ResizeNnNeonImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    using MVType    = typename neon::MQVector<Tp, C>::MVType;
    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 channel = dst.GetSizes().m_channel;
    MI_F32 scale_x = static_cast<MI_F64>(iwidth) / owidth;
    MI_F32 scale_y = static_cast<MI_F64>(iheight) / oheight;

    constexpr MI_S32 elem_counts = 4;
    MI_S32 width_align4 = owidth & (-elem_counts);
    MI_F32 tbl[4] = {0};

    for (MI_S32 i = 0; i < elem_counts; i++)
    {
        tbl[i] = i * scale_x;
    }

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        MI_S32 sy = Floor(y * scale_y);
        sy        = Min(sy, iheight - 1);

        const Tp *src_row = src.Ptr<Tp>(sy);
        Tp *dst_row       = dst.Ptr<Tp>(y);

        MI_S32 x = 0;
        for (; x < width_align4; x += elem_counts)
        {
            MI_F32 sx = x * scale_x;
            float32x4_t vqf32_sx;
            neon::vdup(vqf32_sx, sx);
            float32x4_t vqf32_tbl        = neon::vload1q(tbl);
            float32x4_t vqf32_sx1        = neon::vadd(vqf32_sx, vqf32_tbl);
            uint32x4_t vqu32_sx          = neon::vcvt<MI_U32>(vqf32_sx1);
            uint16x4_t vdu16_sx          = neon::vmovn(vqu32_sx);
            uint16x4_t vdu16_width_bound;
            neon::vdup(vdu16_width_bound, static_cast<MI_U16>(iwidth - 1));
            uint16x4_t vdu16_sx1         = neon::vmin(vdu16_sx, vdu16_width_bound);

            MVType mvq_dst;
            for (MI_S32 c = 0; c < channel; c++)
            {
                mvq_dst.val[c] = ResizeNnNeonCore<Tp>(src_row, vdu16_sx1, channel, c).val[0];
            }
            neon::vstore(dst_row + x * channel, mvq_dst);
        }

        for (; x < owidth; x++)
        {
            MI_S32 sx = Floor(x * scale_x);
            sx        = Min(sx, iwidth - 1);

            for (MI_S32 c = 0; c < channel; c++)
            {
                dst_row[x * channel + c] = src_row[sx * channel + c];
            }
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status ResizeNnNeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool fail");
        return Status::ERROR;
    }

    MI_S32 channel = dst.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeNnNeonImpl<Tp, 1>, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonImpl run fail, channel: 1");
            }
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeNnNeonImpl<Tp, 2>, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonImpl run fail, channel: 2");
            }
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeNnNeonImpl<Tp, 3>, std::cref(src), std::ref(dst));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonImpl run fail, channel: 3");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "do not surpport channel more than 3");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeNnNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeNnNeonHelper<MI_U8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonHelper run fail, type: MI_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeNnNeonHelper<MI_S8>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonHelper run fail, type: MI_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeNnNeonHelper<MI_U16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonHelper run fail, type: MI_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeNnNeonHelper<MI_S16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonHelper run fail, type: MI_S16");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = ResizeNnNeonHelper<MI_F16>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonHelper run fail, type: MI_F16");
            }
            break;
        }
#endif

        case ElemType::U32:
        {
            ret = ResizeNnNeonHelper<MI_U32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonHelper run fail, type: MI_U32");
            }
            break;
        }

        case ElemType::S32:
        {
            ret = ResizeNnNeonHelper<MI_S32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonHelper run fail, type: MI_S32");
            }
            break;
        }

        case ElemType::F32:
        {
            ret = ResizeNnNeonHelper<MI_F32>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeNnNeonHelper run fail, type: MI_F32");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "do not surpport elem type F64");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura