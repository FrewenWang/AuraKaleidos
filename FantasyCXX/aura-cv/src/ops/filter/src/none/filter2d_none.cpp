#include "filter2d_impl.hpp"

#include "aura/ops/core.h"
#include "aura/ops/matrix.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
static Status Filter2dNoneImpl(Context *ctx, const Mat &src, Mat &dst, const std::vector<MI_F32> &kernel,
                               MI_S32 ksize, MI_S32 start_row, MI_S32 end_row)
{
    Status ret = Status::OK;

    Sizes3 sz      = dst.GetSizes();
    MI_S32 width   = sz.m_width;
    MI_S32 channel = sz.m_channel;
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        Tp *dst_row = dst.Ptr<Tp>(y);
        for (MI_S32 x = 0; x < width; x++)
        {
            for (MI_S32 c = 0; c < channel; c++)
            {
                // avoid ofast optimization, In order to be consistent with Opencv
                volatile MI_F32 result = 0.f;
                MI_S32 idx = x * channel + c;
                for (MI_S32 j = 0; j < ksize; j++)
                {
                    const Tp *src_row = src.Ptr<Tp>(y + j) + idx;
                    for (MI_S32 i = 0; i < ksize; i++)
                    {
                        result += src_row[i * channel] * kernel[j * ksize + i];
                    }
                }
                dst_row[idx] = SaturateCast<Tp>(result);
            }
        }
    }

    AURA_RETURN(ctx, ret);
}

Filter2dNone::Filter2dNone(Context *ctx, const OpTarget &target) : Filter2dImpl(ctx, target)
{}

Status Filter2dNone::SetArgs(const Array *src, Array *dst, const Array *kmat,
                             BorderType border_type, const Scalar &border_value)
{
    if (Filter2dImpl::SetArgs(src, dst, kmat, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Filter2dImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType()  != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT) ||
        (kmat->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status Filter2dNone::Initialize()
{
    if (Filter2dImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Filter2dImpl::Initialize() failed");
        return Status::ERROR;
    }

    MI_S32 radius      = m_ksize >> 1;
    Sizes3 border_size = m_src->GetSizes() + Sizes3(radius << 1, radius << 1, 0);
    m_src_border       = Mat(m_ctx, m_src->GetElemType(), border_size);

    if (!m_src_border.IsValid())
    {
        return Status::ERROR;
    }

    return Status::OK;
}

Status Filter2dNone::Run()
{
    const Mat *src  = dynamic_cast<const Mat*>(m_src);
    Mat *dst        = dynamic_cast<Mat*>(m_dst);
    const Mat *kmat = dynamic_cast<const Mat*>(m_kmat);

    if ((MI_NULL == src) || (MI_NULL == dst) || (MI_NULL == kmat))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst kmat is null");
        return Status::ERROR;
    }

    std::vector<MI_F32> kernel(m_ksize * m_ksize, 0.f);
    for (MI_S32 y = 0; y < m_ksize; y++)
    {
        const MI_F32 *k_row = kmat->Ptr<MI_F32>(y);
        for (MI_S32 x = 0; x < m_ksize; x++)
        {
            kernel[y * m_ksize + x] = k_row[x];
        }
    }

    // Get border mat sizes
    MI_S32 radius = m_ksize >> 1;
    if (IMakeBorder(m_ctx, *src, m_src_border, radius, radius, radius, radius, m_border_type, m_border_value, OpTarget::None()) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "make border fail..");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    MI_S32 oheight = dst->GetSizes().m_height;

    MI_S32 pattern = AURA_MAKE_PATTERN(src->GetElemType(), dst->GetElemType());

#define FILTER2D_NONE_IMPL(type)                                                                                            \
    if (m_target.m_data.none.enable_mt)                                                                                     \
    {                                                                                                                       \
        WorkerPool *wp = m_ctx->GetWorkerPool();                                                                            \
        if (MI_NULL == wp)                                                                                                  \
        {                                                                                                                   \
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");                                                           \
            return Status::ERROR;                                                                                           \
        }                                                                                                                   \
                                                                                                                            \
        ret = wp->ParallelFor(static_cast<MI_S32>(0), oheight, Filter2dNoneImpl<type>, m_ctx, std::cref(m_src_border),      \
                              std::ref(*dst), kernel, m_ksize);                                                             \
    }                                                                                                                       \
    else                                                                                                                    \
    {                                                                                                                       \
        ret = Filter2dNoneImpl<type>(m_ctx, m_src_border, *dst, kernel, m_ksize, 0, oheight);                               \
    }                                                                                                                       \
    if (ret != Status::OK)                                                                                                  \
    {                                                                                                                       \
        MI_CHAR error_msg[128];                                                                                             \
        std::snprintf(error_msg, sizeof(error_msg), "Filter2dNoneImpl<%s> failed", #type);                                  \
        AURA_ADD_ERROR_STRING(m_ctx, error_msg);                                                                            \
    }

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U8):
        {
            FILTER2D_NONE_IMPL(MI_U8)
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            FILTER2D_NONE_IMPL(MI_U16)
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            FILTER2D_NONE_IMPL(MI_S16)
            break;
        }

#if defined(AURA_BUILD_HOST)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            FILTER2D_NONE_IMPL(MI_F16)
            break;
        }
#endif // AURA_BUILD_HOST

        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            FILTER2D_NONE_IMPL(MI_F32)
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Unsupported combination of source format and destination format!");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

Status Filter2dNone::DeInitialize()
{
    m_src_border.Release();

    if (Filter2dImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Filter2dImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace aura
