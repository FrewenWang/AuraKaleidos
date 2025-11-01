#include "sobel_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

static Status ScharrKernelMat(Context *ctx, Mat &kx, Mat &ky, MI_S32 dx, MI_S32 dy)
{
    constexpr MI_S32 ksize = 3;
    Mat kx_tmp(ctx, ElemType::F32, Sizes3(1, ksize));
    Mat ky_tmp(ctx, ElemType::F32, Sizes3(1, ksize));
    if (!(kx_tmp.IsValid() && ky_tmp.IsValid()))
    {
        AURA_ADD_ERROR_STRING(ctx, "create kernel failed");
        return Status::ERROR;
    }

    MI_F32 *src = kx_tmp.Ptr<MI_F32>(0);
    if (0 == dx)
    {
        src[0] = 3, src[1] = 10, src[2] = 3;
    }
    else if (1 == dx)
    {
        src[0] = -1, src[1] = 0, src[2] = 1;
    }
    kx = kx_tmp;

    src = ky_tmp.Ptr<MI_F32>(0);
    if (0 == dy)
    {
        src[0] = 3, src[1] = 10, src[2] = 3;
    }
    else if (1 == dy)
    {
        src[0] = -1, src[1] = 0, src[2] = 1;
    }
    ky = ky_tmp;

    return Status::OK;
}

static Status SobelKernelMat(Context *ctx, Mat &kx, Mat &ky, MI_S32 dx, MI_S32 dy, MI_S32 ksize)
{
    if (ksize <= 0)
    {
        if (ScharrKernelMat(ctx, kx, ky, dx, dy) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ScharrKernelMat fail");
            return Status::ERROR;
        }

        return Status::OK;
    }

    const MI_S32 ksx = (1 == ksize && dx > 0) ? 3 : ksize;
    const MI_S32 ksy = (1 == ksize && dy > 0) ? 3 : ksize;

    Mat kx_tmp(ctx, ElemType::F32, Sizes3(1, ksx));
    Mat ky_tmp(ctx, ElemType::F32, Sizes3(1, ksy));
    Mat kernel(ctx, ElemType::F32, Sizes3(1, Max(ksx, ksy) + 1));
    if (!(kx_tmp.IsValid() && ky_tmp.IsValid() && kernel.IsValid()))
    {
        AURA_ADD_ERROR_STRING(ctx, "create kernel failed");
        return Status::ERROR;
    }
    MI_F32 *src = kernel.Ptr<MI_F32>(0);

    dx = Max(Min(dx, ksx - 1), static_cast<MI_S32>(0));
    dy = Max(Min(dy, ksy - 1), static_cast<MI_S32>(0));

    for (MI_S32 k = 0; k < 2; k++)
    {
        MI_F32 *dst  = 0 == k ? kx_tmp.Ptr<MI_F32>(0) : ky_tmp.Ptr<MI_F32>(0);
        MI_S32 order = 0 == k ? dx : dy;
        MI_S32 ks    = 0 == k ? ksx : ksy;

        if (1 == ks)
        {
            dst[0] = 1.f;
        }
        else if (3 == ks)
        {
            if (0 == order)
            {
                dst[0] = 1.f, dst[1] = 2.f, dst[2] = 1.f;
            }
            else if (1 == order)
            {
                dst[0] = -1.f, dst[1] = 0.f, dst[2] = 1.f;
            }
            else
            {
                dst[0] = 1.f, dst[1] = -2.f, dst[2] = 1.f;
            }
        }
        else
        {
            MI_F32 oldval, newval;
            src[0] = 1.f;
            for (MI_S32 i = 0; i < ks; i++)
            {
                src[i + 1] = 0.f;
            }

            for (MI_S32 j = 0; j < ks - order - 1; j++)
            {
                oldval = src[0];
                for (MI_S32 i = 1; i <= ks; i++)
                {
                    newval     = src[i] + src[i - 1];
                    src[i - 1] = oldval;
                    oldval     = newval;
                }
            }

            for (MI_S32 j = 0; j < order; j++)
            {
                oldval = -src[0];
                for (MI_S32 i = 1; i <= ks; i++)
                {
                    newval     = src[i - 1] - src[i];
                    src[i - 1] = oldval;
                    oldval     = newval;
                }
            }

            memcpy(dst, src, ks * sizeof(MI_F32));
        }
    }

    kx = kx_tmp;
    ky = ky_tmp;

    return Status::OK;
}

template <typename St, typename Dt>
static Status SobelNoneImpl(Context *ctx, const Mat &src, Mat &dst, const Mat &kx, const Mat &ky, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 ksx = kx.GetSizes().m_width;
    MI_S32 ksy = ky.GetSizes().m_width;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 channel = dst.GetSizes().m_channel;

    MI_F32 *sum_row = reinterpret_cast<MI_F32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, iwidth * channel * sizeof(MI_F32), 0));
    if (MI_NULL == sum_row)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
        return Status::ERROR;
    }

    std::vector<const St*> src_rows(ksy);
    for (MI_S32 k = 0; k < ksy; k++)
    {
        src_rows[k] = src.Ptr<St>(start_row + k);
    }

    const MI_F32 *kernel_x = kx.Ptr<MI_F32>(0);
    const MI_F32 *kernel_y = ky.Ptr<MI_F32>(0);
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        Dt *dst_row = dst.Ptr<Dt>(y);
        for (MI_S32 x = 0; x < iwidth; x++)
        {
            for (MI_S32 ch = 0; ch < channel; ch++)
            {
                MI_F32 sum = 0.f;
                for (MI_S32 k = 0; k < ksy; k++)
                {
                    sum += src_rows[k][x * channel + ch] * kernel_y[k];
                }
                sum_row[x * channel + ch] = sum;
            }
        }

        for (MI_S32 x = 0; x < owidth; x++)
        {
            for (MI_S32 ch = 0; ch < channel; ch++)
            {
                MI_F32 sum = 0.f;
                for (MI_S32 k = 0; k < ksx; k++)
                {
                    sum += sum_row[(x + k) * channel + ch] * kernel_x[k];
                }
                dst_row[x * channel + ch] = SaturateCast<Dt>(sum);
            }
        }

        for (MI_S32 i = 0; i < ksy - 1; i++)
        {
            src_rows[i] = src_rows[i + 1];
        }
        src_rows[ksy - 1] = src.Ptr<St>(y + ksy);
    }

    AURA_FREE(ctx, sum_row);

    return Status::OK;
}

SobelNone::SobelNone(Context *ctx, const OpTarget &target) : SobelImpl(ctx, target)
{}

Status SobelNone::SetArgs(const Array *src, Array *dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale,
                          BorderType border_type, const Scalar &border_value)
{
    if (SobelImpl::SetArgs(src, dst, dx, dy, ksize, scale, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SobelImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SobelNone::Initialize()
{
    if (SobelImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SobelImpl::Initialize() failed");
        return Status::ERROR;
    }

    if (SobelKernelMat(m_ctx, m_kx, m_ky, m_dx, m_dy, m_ksize) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SobelKernelMat failed");
        return Status::ERROR;
    }

    if (IMultiply(m_ctx, m_ky, m_scale, m_ky, m_target) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Multiply failed");
        return Status::ERROR;
    }

    // Get border mat sizes
    const MI_S32 kshx   = m_kx.GetSizes().m_width >> 1;
    const MI_S32 kshy   = m_ky.GetSizes().m_width >> 1;
    Sizes3 border_sizes = m_src->GetSizes() + Sizes3(kshy << 1, kshx << 1, 0);
    m_src_border        = Mat(m_ctx, m_src->GetElemType(), border_sizes);

    if (!m_src_border.IsValid())
    {
        return Status::ERROR;
    }

    return Status::OK;
}

Status SobelNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    const MI_S32 kshx = m_kx.GetSizes().m_width >> 1;
    const MI_S32 kshy = m_ky.GetSizes().m_width >> 1;
    if (IMakeBorder(m_ctx, *src, m_src_border, kshy, kshy, kshx, kshx, m_border_type, m_border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MakeBorder failed");
        return Status::ERROR;
    }

    Status ret     = Status::ERROR;
    MI_S32 pattern = AURA_MAKE_PATTERN(src->GetElemType(), dst->GetElemType());
    MI_S32 oheight = dst->GetSizes().m_height;

#define SOBEL_NONE_IMPL(type1, type2)                                                                                       \
    if (m_target.m_data.none.enable_mt)                                                                                     \
    {                                                                                                                       \
        WorkerPool *wp = m_ctx->GetWorkerPool();                                                                            \
        if (MI_NULL == wp)                                                                                                  \
        {                                                                                                                   \
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");                                                           \
            return Status::ERROR;                                                                                           \
        }                                                                                                                   \
                                                                                                                            \
        ret = wp->ParallelFor(static_cast<MI_S32>(0), oheight, SobelNoneImpl<type1, type2>, m_ctx, std::cref(m_src_border), \
                              std::ref(*dst), m_kx, m_ky);                                                                  \
    }                                                                                                                       \
    else                                                                                                                    \
    {                                                                                                                       \
        ret = SobelNoneImpl<type1, type2>(m_ctx, m_src_border, *dst, m_kx, m_ky, 0, oheight);                               \
    }                                                                                                                       \
    if (ret != Status::OK)                                                                                                  \
    {                                                                                                                       \
        MI_CHAR error_msg[128];                                                                                             \
        std::snprintf(error_msg, sizeof(error_msg), "SobelNoneImpl<%s, %s> failed", #type1, #type2);                        \
        AURA_ADD_ERROR_STRING(m_ctx, error_msg);                                                                            \
    }

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U8):
        {
            SOBEL_NONE_IMPL(MI_U8, MI_U8)
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U16):
        {
            SOBEL_NONE_IMPL(MI_U8, MI_U16)
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            SOBEL_NONE_IMPL(MI_U8, MI_S16)
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::F32):
        {
            SOBEL_NONE_IMPL(MI_U8, MI_F32)
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            SOBEL_NONE_IMPL(MI_U16, MI_U16)
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::F32):
        {
            SOBEL_NONE_IMPL(MI_U16, MI_F32)
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            SOBEL_NONE_IMPL(MI_S16, MI_S16)
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::F32):
        {
            SOBEL_NONE_IMPL(MI_S16, MI_F32)
            break;
        }

#if defined(AURA_BUILD_HOST)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            SOBEL_NONE_IMPL(MI_F16, MI_F16)
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F32):
        {
            SOBEL_NONE_IMPL(MI_F16, MI_F32)
            break;
        }
#endif // AURA_BUILD_HOST

        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            SOBEL_NONE_IMPL(MI_F32, MI_F32)
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported combination of source format and destination format");
            return Status::ERROR;
        }
    }

#undef SOBEL_NONE_IMPL

    AURA_RETURN(m_ctx, ret);
}

Status SobelNone::DeInitialize()
{
    m_src_border.Release();
    m_kx.Release();
    m_ky.Release();

    if (SobelImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SobelImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace aura