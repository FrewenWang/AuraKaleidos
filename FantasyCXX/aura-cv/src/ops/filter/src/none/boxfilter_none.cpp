#include "boxfilter_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"

namespace aura
{

template <typename Tp>
struct BoxFilterTraits
{
    using InterType = typename Promote<Tp>::Type;
};

template <>
struct BoxFilterTraits<MI_U8>
{
    using InterType = MI_U32;
};

template <>
struct BoxFilterTraits<MI_S8>
{
    using InterType = MI_S32;
};

template <typename Tp>
static Status BoxFilterNoneImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, MI_S32 ksize, MI_S32 start_row, MI_S32 end_row)
{
    using InterType = typename BoxFilterTraits<Tp>::InterType;

    MI_S32 ksize_sq = ksize * ksize;

    const Sizes3 isizes  = src.GetSizes();
    const Sizes3 osizes  = dst.GetSizes();
    const MI_S32 iwidth  = isizes.m_width;
    const MI_S32 owidth  = osizes.m_width;
    const MI_S32 channel = isizes.m_channel;
    MI_S32 ksh = ksize >> 1;

    std::vector<const Tp*> src_rows(ksize);
    for (MI_S32 k = 0; k < ksize; k++)
    {
        src_rows[k] = src.Ptr<Tp>(start_row + k);
    }

    InterType *sum_row = thread_buffer.GetThreadData<InterType>();

    if (MI_NULL == sum_row)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        // calc vertical sum (iwidth = owidth + ksize/2 * 2)
        for (MI_S32 x = 0; x < iwidth; x++)
        {
            for (MI_S32 ch = 0; ch < channel; ch++)
            {
                InterType sum = 0;
                MI_S32 index  = x * channel + ch;

                for (MI_S32 k = 0; k < ksh; k++)
                {
                    Tp top = src_rows[k][index];
                    Tp bot = src_rows[ksize - k - 1][index];

                    sum += static_cast<InterType>(top) + static_cast<InterType>(bot);
                }
                sum += static_cast<InterType>(src_rows[ksh][index]);

                sum_row[index] = sum;
            }
        }

        Tp *dst_row = dst.Ptr<Tp>(y);

        // calc horizontal sum
        for (MI_S32 x = 0; x < owidth; x++)
        {
            for (MI_S32 ch = 0; ch < channel; ch++)
            {
                InterType sum_kernel = 0;

                for (MI_S32 k = 0; k < ksh; k++)
                {
                    sum_kernel += (sum_row[(x + k) * channel + ch] + sum_row[(x + ksize - k - 1) * channel + ch]);
                }
                sum_kernel += sum_row[(x + ksh) * channel + ch];

                Tp mean = static_cast<Tp>(sum_kernel / ksize_sq);

                dst_row[x * channel + ch] = mean;
            }
        }

        for (MI_S32 i = 0; i < ksize - 1; i++)
        {
            src_rows[i] = src_rows[i + 1];
        }
        src_rows[ksize - 1] = src.Ptr<Tp>(y + ksize);
    }

    return Status::OK;
}

BoxFilterNone::BoxFilterNone(Context *ctx, const OpTarget &target) : BoxFilterImpl(ctx, target)
{}

Status BoxFilterNone::SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                              BorderType border_type, const Scalar &border_value)
{
    if (BoxFilterImpl::SetArgs(src, dst, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BoxFilterImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BoxFilterNone::Initialize()
{
    if (BoxFilterImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BoxFilterImpl::Initialize() failed");
        return Status::ERROR;
    }

    // Get border mat sizes
    MI_S32 ksh          = m_ksize >> 1;
    Sizes3 border_sizes = m_src->GetSizes() + Sizes3(ksh << 1, ksh << 1, 0);
    m_src_border        = Mat(m_ctx, m_src->GetElemType(), border_sizes);

    if (!m_src_border.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_src_border is not valid");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BoxFilterNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    // Get border mat sizes
    MI_S32 ksh = m_ksize >> 1;
    if (IMakeBorder(m_ctx, *src, m_src_border, ksh, ksh, ksh, ksh, m_border_type, m_border_value, OpTarget::None()) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "make border fail..");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    MI_S32 oheight  = dst->GetSizes().m_height;
    MI_S32 iwidth   = m_src_border.GetSizes().m_width;
    MI_S32 ichannel = m_src_border.GetSizes().m_channel;

#define BOX_FILTER_NONE_IMPL(type)                                                                                      \
    if (m_target.m_data.none.enable_mt)                                                                                 \
    {                                                                                                                   \
        WorkerPool *wp = m_ctx->GetWorkerPool();                                                                        \
        if (MI_NULL == wp)                                                                                              \
        {                                                                                                               \
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");                                                       \
            return Status::ERROR;                                                                                       \
        }                                                                                                               \
                                                                                                                        \
        using InterType = typename BoxFilterTraits<type>::InterType;                                                    \
                                                                                                                        \
        MI_S32 buffer_size = iwidth * ichannel * sizeof(InterType);                                                     \
        ThreadBuffer thread_buffer(m_ctx, buffer_size);                                                                 \
                                                                                                                        \
        ret = wp->ParallelFor(static_cast<MI_S32>(0), oheight, BoxFilterNoneImpl<type>, m_ctx, std::cref(m_src_border), \
                              std::ref(*dst), std::ref(thread_buffer), m_ksize);                                        \
    }                                                                                                                   \
    else                                                                                                                \
    {                                                                                                                   \
        using InterType = typename BoxFilterTraits<type>::InterType;                                                    \
                                                                                                                        \
        MI_S32 buffer_size = iwidth * ichannel * sizeof(InterType);                                                     \
        ThreadBuffer thread_buffer(m_ctx, buffer_size);                                                                 \
                                                                                                                        \
        ret = BoxFilterNoneImpl<type>(m_ctx, m_src_border, *dst, thread_buffer, m_ksize, 0, oheight);                   \
    }                                                                                                                   \
    if (ret != Status::OK)                                                                                              \
    {                                                                                                                   \
        MI_CHAR error_msg[128];                                                                                         \
        std::snprintf(error_msg, sizeof(error_msg), "BoxFilterNoneImpl<%s> failed", #type);                             \
        AURA_ADD_ERROR_STRING(m_ctx, error_msg);                                                                        \
    }

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            BOX_FILTER_NONE_IMPL(MI_U8)
            break;
        }

        case ElemType::S8:
        {
            BOX_FILTER_NONE_IMPL(MI_S8)
            break;
        }

        case ElemType::U16:
        {
            BOX_FILTER_NONE_IMPL(MI_U16)
            break;
        }

        case ElemType::S16:
        {
            BOX_FILTER_NONE_IMPL(MI_S16)
            break;
        }

        case ElemType::U32:
        {
            BOX_FILTER_NONE_IMPL(MI_U32)
            break;
        }

        case ElemType::S32:
        {
            BOX_FILTER_NONE_IMPL(MI_S32)
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            BOX_FILTER_NONE_IMPL(MI_F16)
            break;
        }
#endif // AURA_BUILD_HOST

        case ElemType::F32:
        {
            BOX_FILTER_NONE_IMPL(MI_F32)
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

#undef BOX_FILTER_NONE_IMPL

    AURA_RETURN(m_ctx, ret);
}

Status BoxFilterNone::DeInitialize()
{
    m_src_border.Release();

    if (BoxFilterImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BoxFilterImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace aura
