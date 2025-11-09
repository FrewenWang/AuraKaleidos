#include "integral_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

#include <vector>
namespace aura
{

template <typename Tp, typename SumType>
static Status IntegralBlockNoneImpl(const Mat &src, Mat &dst,
                                    DT_S32 block_h, DT_S32 block_w,
                                    DT_S32 idx_h,   DT_S32 idx_w)
{
    const DT_S32 height  = src.GetSizes().m_height;
    const DT_S32 width   = src.GetSizes().m_width;
    const DT_S32 channel = src.GetSizes().m_channel;

    const DT_S32 start_row = idx_h * block_h + 1;
    const DT_S32 start_col = idx_w * block_w;
    DT_S32 end_row = Min(start_row + block_h, height);
    DT_S32 end_col = Min(start_col + block_w, width);

    std::vector<SumType> sum_left(channel);

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const Tp  *src_c = src.Ptr<Tp>(y);

        SumType *dst_c = dst.Ptr<SumType>(y);
        SumType *dst_p = dst.Ptr<SumType>(y - 1);

        if (0 == start_col)
        {
            for (DT_S32 ch = 0; ch < channel; ch++)
            {
                sum_left[ch] = 0;
            }
        }
        else
        {
            for (DT_S32 ch = 0; ch < channel; ch++)
            {
                sum_left[ch] = dst_c[(start_col - 1) * channel + ch] -
                               dst_p[(start_col - 1) * channel + ch];
            }
        }

        for (DT_S32 x = start_col; x < end_col; ++x)
        {
            for (DT_S32 ch = 0; ch < channel; ch++)
            {
                sum_left[ch] += src_c[x * channel + ch];
                dst_c[x * channel + ch] = dst_p[x * channel + ch] + sum_left[ch];
            }
        }
    }

    return Status::OK;
}

template <typename Tp, typename SqSumType>
static Status IntegralSqBlockNoneImpl(const Mat &src, Mat &sqdst,
                                      DT_S32 block_h, DT_S32 block_w,
                                      DT_S32 idx_h,   DT_S32 idx_w)
{
    const DT_S32 height  = src.GetSizes().m_height;
    const DT_S32 width   = src.GetSizes().m_width;
    const DT_S32 channel = src.GetSizes().m_channel;

    const DT_S32 start_row = idx_h * block_h + 1;
    const DT_S32 start_col = idx_w * block_w;
    DT_S32 end_row = Min(start_row + block_h, height);
    DT_S32 end_col = Min(start_col + block_w, width);

    std::vector<SqSumType> sqsum_left(channel);

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const Tp  *src_c = src.Ptr<Tp>(y);

        SqSumType *sqdst_c = sqdst.Ptr<SqSumType>(y);
        SqSumType *sqdst_p = sqdst.Ptr<SqSumType>(y - 1);

        if (0 == start_col)
        {
            for (DT_S32 ch = 0; ch < channel; ch++)
            {
                sqsum_left[ch] = 0;
            }
        }
        else
        {
            for (DT_S32 ch = 0; ch < channel; ch++)
            {
                sqsum_left[ch] = sqdst_c[(start_col - 1) * channel + ch] -
                                 sqdst_p[(start_col - 1) * channel + ch];
            }
        }

        for (DT_S32 x = start_col; x < end_col; ++x)
        {
            for (DT_S32 ch = 0; ch < channel; ch++)
            {
                SqSumType src_val = static_cast<SqSumType>(src_c[x * channel + ch]);
                sqsum_left[ch] += src_val * src_val;
                sqdst_c[x * channel + ch] = sqdst_p[x * channel + ch] + sqsum_left[ch];
            }
        }
    }

    return Status::OK;
}

template <typename Tp, typename SumType>
static Status IntegralNoneImpl(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    const DT_S32 height       = src.GetSizes().m_height;
    const DT_S32 width        = src.GetSizes().m_width;
    const DT_S32 channel      = src.GetSizes().m_channel;
    const DT_S32 block_height = 256;
    const DT_S32 block_width  = 256;

    // first row
    const Tp *src_c = src.Ptr<Tp>(0);
    SumType  *dst_c = dst.Ptr<SumType>(0);

    for (DT_S32 ch = 0; ch < channel; ch++)
    {
        dst_c[ch] = src_c[ch];
    }

    for (DT_S32 x = 1; x < width; x++)
    {
        for (DT_S32 ch = 0; ch < channel; ch++)
        {
            dst_c[x * channel + ch] = dst_c[(x - 1) * channel + ch] + src_c[x * channel + ch];
        }
    }

    // remaining rows
    if (target.m_data.none.enable_mt)
    {
        ret = wp->WaveFront((height - 1 + block_height - 1) / block_height, (width + block_width - 1) / block_width,
                            IntegralBlockNoneImpl<Tp, SumType>, src, dst, block_height, block_width);
    }
    else
    {
        ret = IntegralBlockNoneImpl<Tp, SumType>(src, dst, height - 1, width, 0, 0);
    }

    return ret;
}

template <typename Tp, typename SqSumType>
static Status IntegralSqNoneImpl(Context *ctx, const Mat &src, Mat &dst_sq, const OpTarget &target)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    const DT_S32 height       = src.GetSizes().m_height;
    const DT_S32 width        = src.GetSizes().m_width;
    const DT_S32 channel      = src.GetSizes().m_channel;
    const DT_S32 block_height = 256;
    const DT_S32 block_width  = 256;

    // first row
    const Tp  *src_c = src.Ptr<Tp>(0);
    SqSumType *dst_c = dst_sq.Ptr<SqSumType>(0);

    for (DT_S32 ch = 0; ch < channel; ch++)
    {
        SqSumType src_val = static_cast<SqSumType>(src_c[ch]);
        dst_c[ch] = src_val * src_val;
    }

    for (DT_S32 x = 1; x < width; x++)
    {
        for (DT_S32 ch = 0; ch < channel; ch++)
        {
            SqSumType src_val = static_cast<SqSumType>(src_c[x * channel + ch]);
            dst_c[x * channel + ch] = dst_c[(x - 1) * channel + ch] + src_val * src_val;
        }
    }

    // remaining rows
    if (target.m_data.none.enable_mt)
    {
        ret = wp->WaveFront((height - 1 + block_height - 1) / block_height, (width + block_width - 1) / block_width,
                            IntegralSqBlockNoneImpl<Tp, SqSumType>, src, dst_sq, block_height, block_width);
    }
    else
    {
        ret = IntegralSqBlockNoneImpl<Tp, SqSumType>(src, dst_sq, height - 1, width, 0, 0);
    }

    return ret;
}

IntegralNone::IntegralNone(Context *ctx, const OpTarget &target) : IntegralImpl(ctx, target)
{}

Status IntegralNone::SetArgs(const Array *src, Array *dst, Array *dst_sq)
{
    if (IntegralImpl::SetArgs(src, dst, dst_sq) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "IntegralImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if (dst && dst->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst must be mat type");
        return Status::ERROR;
    }

    if (dst_sq && dst_sq->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst_sq must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status IntegralNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);
    Mat *dst_sq    = dynamic_cast<Mat*>(m_dst_sq);

    Status ret = Status::OK;

    if (dst && dst->IsValid())
    {
        switch (AURA_MAKE_PATTERN(src->GetElemType(), dst->GetElemType()))
        {
            case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U32):
            {
                ret = IntegralNoneImpl<DT_U8, DT_U32>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::U8, ElemType::F32):
            {
                ret = IntegralNoneImpl<DT_U8, DT_F32>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::U8, ElemType::F64):
            {
                ret = IntegralNoneImpl<DT_U8, DT_F64>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S32):
            {
                ret = IntegralNoneImpl<DT_S8, DT_S32>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S8, ElemType::F32):
            {
                ret = IntegralNoneImpl<DT_S8, DT_F32>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S8, ElemType::F64):
            {
                ret = IntegralNoneImpl<DT_S8, DT_F64>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U32):
            {
                ret = IntegralNoneImpl<DT_U16, DT_U32>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::U16, ElemType::F32):
            {
                ret = IntegralNoneImpl<DT_U16, DT_F32>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::U16, ElemType::F64):
            {
                ret = IntegralNoneImpl<DT_U16, DT_F64>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S32):
            {
                ret = IntegralNoneImpl<DT_S16, DT_S32>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S16, ElemType::F32):
            {
                ret = IntegralNoneImpl<DT_S16, DT_F32>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S16, ElemType::F64):
            {
                ret = IntegralNoneImpl<DT_S16, DT_F64>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
            {
                ret = IntegralNoneImpl<DT_F32, DT_F32>(m_ctx, *src, *dst, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F64):
            {
                ret = IntegralNoneImpl<DT_F32, DT_F64>(m_ctx, *src, *dst, m_target);
                break;
            }
            default:
            {
                AURA_ADD_ERROR_STRING(m_ctx, "elem type error..unsupport");
                ret = Status::ERROR;
            }
        }
    }

    if (dst_sq && dst_sq->IsValid())
    {
        switch (AURA_MAKE_PATTERN(src->GetElemType(), dst_sq->GetElemType()))
        {
            case AURA_MAKE_PATTERN(ElemType::U8, ElemType::F64):
            {
                ret = IntegralSqNoneImpl<DT_U8, DT_F64>(m_ctx, *src, *dst_sq, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U32):
            {
                ret = IntegralSqNoneImpl<DT_U8, DT_U32>(m_ctx, *src, *dst_sq, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S8, ElemType::F64):
            {
                ret = IntegralSqNoneImpl<DT_S8, DT_F64>(m_ctx, *src, *dst_sq, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S8, ElemType::U32):
            {
                // DT_S32 is not an error, here is for src data conversion, result is exactly the same as DT_U32
                ret = IntegralSqNoneImpl<DT_S8, DT_S32>(m_ctx, *src, *dst_sq, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::U16, ElemType::F64):
            {
                ret = IntegralSqNoneImpl<DT_U16, DT_F64>(m_ctx, *src, *dst_sq, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S16, ElemType::F64):
            {
                ret = IntegralSqNoneImpl<DT_S16, DT_F64>(m_ctx, *src, *dst_sq, m_target);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F64):
            {
                ret = IntegralSqNoneImpl<DT_F32, DT_F64>(m_ctx, *src, *dst_sq, m_target);
                break;
            }
            default:
            {
                AURA_ADD_ERROR_STRING(m_ctx, "elem type error..unsupport");
                ret = Status::ERROR;
            }
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura