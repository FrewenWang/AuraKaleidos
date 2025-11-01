#include "median_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
AURA_ALWAYS_INLINE Tp MedianFilterQuickSelect(std::vector<Tp> &arr, MI_S32 nums)
{
    MI_S32 low = 0;
    MI_S32 high = nums - 1;
    MI_S32 median = (low + high) / 2;

    for (;;)
    {
        if (high <= low)
        {
            return arr[median];
        }

        if (high == low + 1)
        {
            if (arr[low] > arr[high])
            {
                Swap(arr[low], arr[high]);
            }
            return arr[median];
        }

        MI_S32 middle = (low + high) / 2;
        if (arr[middle] > arr[high])
        {
            Swap(arr[middle], arr[high]);
        }

        if (arr[low] > arr[high])
        {
            Swap(arr[low], arr[high]);
        }

        if (arr[middle] > arr[low])
        {
            Swap(arr[middle], arr[low]);
        }

        Swap(arr[middle], arr[low + 1]);

        MI_S32 low_pointer = low + 1;
        MI_S32 high_pointer = high;
        for (;;)
        {
            do
            {
                low_pointer++;
            } while (arr[low] > arr[low_pointer]);

            do
            {
                high_pointer--;
            } while (arr[high_pointer] > arr[low]);

            if (high_pointer < low_pointer)
            {
                break;
            }
            Swap(arr[low_pointer], arr[high_pointer]);
        }

        Swap(arr[low], arr[high_pointer]);

        if (high_pointer <= median)
        {
            low = low_pointer;
        }

        if (high_pointer >= median)
        {
            high = high_pointer - 1;
        }
    }
}

template <typename Tp>
static Status MedianNoneImpl(const Mat &src, Mat &dst, MI_S32 ksize, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 ksh = ksize >> 1;
    MI_S32 y_index = 0, x_index = 0;

    std::vector<Tp> data;
    data.resize(ksize * ksize, 0);

    MI_S32 channel = dst.GetSizes().m_channel;
    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 height  = dst.GetSizes().m_height;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        Tp *dst_row = dst.Ptr<Tp>(y);
        for (MI_S32 x = 0; x < width; x++)
        {
            for (MI_S32 c = 0; c < channel; c++)
            {
                for (MI_S32 j = -ksh; j <= ksh; j++)
                {
                    y_index = Clamp(y + j, static_cast<MI_S32>(0), height - 1);
                    const Tp *src_row = src.Ptr<Tp>(y_index);

                    for (MI_S32 i = -ksh; i <= ksh; i++)
                    {
                        x_index = Clamp(x + i, static_cast<MI_S32>(0), width - 1);
                        data[(j + ksh) * ksize + (i + ksh)] = src_row[x_index * channel + c];
                    }
                }
                dst_row[x * channel + c] = MedianFilterQuickSelect(data, ksize * ksize);
            }
        }
    }
    return Status::OK;
}

MedianNone::MedianNone(Context *ctx, const OpTarget &target) : MedianImpl(ctx, target)
{}

Status MedianNone::SetArgs(const Array *src, Array *dst, MI_S32 ksize)
{
    if (MedianImpl::SetArgs(src, dst, ksize) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MedianImpl::Initialize failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MedianNone::Initialize()
{
    if (MedianImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MedianImpl::Prepare() failed");
        return Status::ERROR;
    }

    // Get border mat sizes
    MI_S32 ksh          = m_ksize >> 1;
    Sizes3 border_sizes = m_src->GetSizes() + Sizes3(ksh << 1, ksh << 1, 0);
    m_src_border        = Mat(m_ctx, m_src->GetElemType(), border_sizes);

    if (!m_src_border.IsValid())
    {
        return Status::ERROR;
    }

    return Status::OK;
}

template <typename Tp>
static Status MedianNoneHepler(Context *ctx, const Mat &src, Mat &dst, MI_S32 ksize, MI_S32 oheight, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return ret;
        }

        ret = wp->ParallelFor(static_cast<MI_S32>(0), oheight, MedianNoneImpl<Tp>, std::cref(src), std::ref(dst), ksize);
    }
    else
    {
        ret = MedianNoneImpl<Tp>(src, dst, ksize, 0, oheight);
    }

    return ret;
}

Status MedianNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);
    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    MI_S32 oheight = dst->GetSizes().m_height;
    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = MedianNoneHepler<MI_U8>(m_ctx, *src, *dst, m_ksize, oheight, m_target);
            break;
        }

        case ElemType::S8:
        {
            ret = MedianNoneHepler<MI_S8>(m_ctx, *src, *dst, m_ksize, oheight, m_target);
            break;
        }

        case ElemType::U16:
        {
            ret = MedianNoneHepler<MI_U16>(m_ctx, *src, *dst, m_ksize, oheight, m_target);
            break;
        }

        case ElemType::S16:
        {
            ret = MedianNoneHepler<MI_S16>(m_ctx, *src, *dst, m_ksize, oheight, m_target);
            break;
        }

        case ElemType::U32:
        {
            ret = MedianNoneHepler<MI_U32>(m_ctx, *src, *dst, m_ksize, oheight, m_target);
            break;
        }

        case ElemType::S32:
        {
            ret = MedianNoneHepler<MI_S32>(m_ctx, *src, *dst, m_ksize, oheight, m_target);
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = MedianNoneHepler<MI_F16>(m_ctx, *src, *dst, m_ksize, oheight, m_target);
            break;
        }

        case ElemType::F32:
        {
            ret = MedianNoneHepler<MI_F32>(m_ctx, *src, *dst, m_ksize, oheight, m_target);
            break;
        }
#endif

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

Status MedianNone::DeInitialize()
{
    m_src_border.Release();

    if (MedianImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MedianImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace aura
