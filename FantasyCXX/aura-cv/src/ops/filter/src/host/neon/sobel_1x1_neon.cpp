#include "sobel_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

const static MI_S16 g_kernel_tabel[2][3] =
{
    {-1,  0, 1},
    { 1, -2, 1}
};

AURA_ALWAYS_INLINE int16x8_t Sobel1x1Core(uint8x8_t &vdu8_src_p, uint8x8_t &vdu8_src_c, uint8x8_t &vdu8_src_n, const MI_S16 *kernel)
{
    int16x8_t vqs16_sum_p = neon::vmul(neon::vreinterpret(neon::vmovl(vdu8_src_p)), kernel[0]);
    int16x8_t vqs16_sum_c = neon::vmul(neon::vreinterpret(neon::vmovl(vdu8_src_c)), kernel[1]);
    int16x8_t vqs16_sum_n = neon::vmul(neon::vreinterpret(neon::vmovl(vdu8_src_n)), kernel[2]);

    return neon::vadd(neon::vadd(vqs16_sum_p, vqs16_sum_n), vqs16_sum_c);
}

AURA_ALWAYS_INLINE float32x4_t Sobel1x1Core(float32x4_t &vqf32_src_p, float32x4_t &vqf32_src_c, float32x4_t &vqf32_src_n, const MI_S16 *kernel)
{
    float32x4_t vqf32_sum_p = neon::vmul(vqf32_src_p, static_cast<MI_F32>(kernel[0]));
    float32x4_t vqf32_sum_c = neon::vmul(vqf32_src_c, static_cast<MI_F32>(kernel[1]));
    float32x4_t vqf32_sum_n = neon::vmul(vqf32_src_n, static_cast<MI_F32>(kernel[2]));

    return neon::vadd(neon::vadd(vqf32_sum_p, vqf32_sum_n), vqf32_sum_c);
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C, MI_BOOL WITH_SCALE>
static AURA_VOID Sobel1x1DxRow(const St *src, Dt *dst, const MI_S16 *kernel, MI_F32 scale, const std::vector<St> &border_value, MI_S32 width)
{
    using MVSt      = typename std::conditional<std::is_same<St, MI_F32>::value, typename neon::MQVector<St, C>::MVType,
                                                typename neon::MDVector<St, C>::MVType>::type;
    using MVSumType = typename std::conditional<std::is_same<St, MI_F32>::value, typename neon::MQVector<St, C>::MVType,
                                                typename neon::MQVector<MI_S16, C>::MVType>::type;

    constexpr MI_S32 ELEM_COUNTS = static_cast<MI_S32>(sizeof(MVSt) / C / sizeof(St));
    constexpr MI_S32 VOFFSET     = ELEM_COUNTS * C;

    const MI_S32 width_align = (width & -ELEM_COUNTS) * C;

    MVSt mv_src_c[3];
    MVSt mv_src_l, mv_src_r;
    MVSumType mv_result;

    // left
    {
        neon::vload(src,           mv_src_c[1]);
        neon::vload(src + VOFFSET, mv_src_c[2]);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_c[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_c[1].val[ch], src[ch], border_value[ch]);
            mv_src_l.val[ch]    = neon::vext<ELEM_COUNTS - 1>(mv_src_c[0].val[ch], mv_src_c[1].val[ch]);
            mv_src_r.val[ch]    = neon::vext<1>(mv_src_c[1].val[ch], mv_src_c[2].val[ch]);
            mv_result.val[ch]   = Sobel1x1Core(mv_src_l.val[ch], mv_src_c[1].val[ch], mv_src_r.val[ch], kernel);
        }
        SobelPostProcess<MVSumType, WITH_SCALE>(mv_result, dst, scale);

        mv_src_c[0] = mv_src_c[1];
        mv_src_c[1] = mv_src_c[2];
    }

    // middle
    {
        for (MI_S32 x = VOFFSET; x < (width_align - VOFFSET); x += VOFFSET)
        {
            neon::vload(src + x + VOFFSET, mv_src_c[2]);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_src_l.val[ch]  = neon::vext<ELEM_COUNTS - 1>(mv_src_c[0].val[ch], mv_src_c[1].val[ch]);
                mv_src_r.val[ch]  = neon::vext<1>(mv_src_c[1].val[ch], mv_src_c[2].val[ch]);
                mv_result.val[ch] = Sobel1x1Core(mv_src_l.val[ch], mv_src_c[1].val[ch], mv_src_r.val[ch], kernel);
            }
            SobelPostProcess<MVSumType, WITH_SCALE>(mv_result, dst + x, scale);

            mv_src_c[0] = mv_src_c[1];
            mv_src_c[1] = mv_src_c[2];
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            MI_S32 x = (width - (ELEM_COUNTS << 1)) * C;

            neon::vload(src + x - VOFFSET, mv_src_c[0]);
            neon::vload(src + x,           mv_src_c[1]);
            neon::vload(src + x + VOFFSET, mv_src_c[2]);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_src_l.val[ch]  = neon::vext<ELEM_COUNTS - 1>(mv_src_c[0].val[ch], mv_src_c[1].val[ch]);
                mv_src_r.val[ch]  = neon::vext<1>(mv_src_c[1].val[ch], mv_src_c[2].val[ch]);
                mv_result.val[ch] = Sobel1x1Core(mv_src_l.val[ch], mv_src_c[1].val[ch], mv_src_r.val[ch], kernel);
            }
            SobelPostProcess<MVSumType, WITH_SCALE>(mv_result, dst + x, scale);

            mv_src_c[0] = mv_src_c[1];
            mv_src_c[1] = mv_src_c[2];
        }
    }

    // right
    {
        MI_S32 x    = (width - ELEM_COUNTS) * C;
        MI_S32 last = (width - 1) * C;

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_c[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_c[1].val[ch], src[last], border_value[ch]);
            mv_src_l.val[ch]    = neon::vext<ELEM_COUNTS - 1>(mv_src_c[0].val[ch], mv_src_c[1].val[ch]);
            mv_src_r.val[ch]    = neon::vext<1>(mv_src_c[1].val[ch], mv_src_c[2].val[ch]);
            mv_result.val[ch]   = Sobel1x1Core(mv_src_l.val[ch], mv_src_c[1].val[ch], mv_src_r.val[ch], kernel);
            last++;
        }
        SobelPostProcess<MVSumType, WITH_SCALE>(mv_result, dst + x, scale);
    }
}

template <typename St, typename Dt, MI_S32 C, MI_BOOL WITH_SCALE>
static AURA_VOID Sobel1x1DyRow(const St *src_p, const St *src_c, const St *src_n, Dt *dst, const MI_S16 *kernel, MI_F32 scale, MI_S32 width)
{
    using MVSt      = typename std::conditional<std::is_same<St, MI_F32>::value, typename neon::MQVector<St, C>::MVType,
                                                typename neon::MDVector<St, C>::MVType>::type;
    using MVSumType = typename std::conditional<std::is_same<St, MI_F32>::value, typename neon::MQVector<St, C>::MVType,
                                                typename neon::MQVector<MI_S16, C>::MVType>::type;

    constexpr MI_S32 ELEM_COUNTS = static_cast<MI_S32>(sizeof(MVSt) / C / sizeof(St));
    constexpr MI_S32 VOFFSET     = ELEM_COUNTS * C;

    const MI_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVSt mv_src_p, mv_src_c, mv_src_n;
    MVSumType mv_result;

    // left + middle
    {
        for (MI_S32 x = 0; x < width_align; x += VOFFSET)
        {
            neon::vload(src_p + x, mv_src_p);
            neon::vload(src_c + x, mv_src_c);
            neon::vload(src_n + x, mv_src_n);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_result.val[ch] = Sobel1x1Core(mv_src_p.val[ch], mv_src_c.val[ch], mv_src_n.val[ch], kernel);
            }
            SobelPostProcess<MVSumType, WITH_SCALE>(mv_result, dst + x, scale);
        }
    }

    // back + right
    {
        if (width_align != width * C)
        {
            MI_S32 x = (width - ELEM_COUNTS) * C;

            neon::vload(src_p + x, mv_src_p);
            neon::vload(src_c + x, mv_src_c);
            neon::vload(src_n + x, mv_src_n);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_result.val[ch] = Sobel1x1Core(mv_src_p.val[ch], mv_src_c.val[ch], mv_src_n.val[ch], kernel);
            }
            SobelPostProcess<MVSumType, WITH_SCALE>(mv_result, dst + x, scale);
        }
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C, MI_BOOL WITH_SCALE>
static Status Sobel1x1DxNeonImpl(const Mat &src, Mat &dst, const MI_S16 *kernel, MI_F32 scale,
                                 const std::vector<St> &border_value, MI_S32 start_row, MI_S32 end_row)
{
    const St *src_c = MI_NULL;
    Dt *dst_c = MI_NULL;

    MI_S32 width = dst.GetSizes().m_width;
    MI_S32 y = start_row;

    for (; y < end_row; y++)
    {
        src_c = src.Ptr<St>(y);
        dst_c = dst.Ptr<Dt>(y);
        Sobel1x1DxRow<St, Dt, BORDER_TYPE, C, WITH_SCALE>(src_c, dst_c, kernel, scale, border_value, width);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C, MI_BOOL WITH_SCALE>
static Status Sobel1x1DyNeonImpl(const Mat &src, Mat &dst, const MI_S16 *kernel, MI_F32 scale,
                                 const St *border_buffer, MI_S32 start_row, MI_S32 end_row)
{
    const St *src_p = MI_NULL, *src_c = MI_NULL, *src_n = MI_NULL;
    Dt *dst_c = MI_NULL;

    MI_S32 width = dst.GetSizes().m_width;
    MI_S32 y = start_row;

    src_p = src.Ptr<St, BORDER_TYPE>(y - 1, border_buffer);
    src_c = src.Ptr<St>(y);
    src_n = src.Ptr<St, BORDER_TYPE>(y + 1, border_buffer);

    for (; y < end_row; y++)
    {
        dst_c = dst.Ptr<Dt>(y);
        Sobel1x1DyRow<St, Dt, C, WITH_SCALE>(src_p, src_c, src_n, dst_c, kernel, scale, width);

        src_p = src_c;
        src_c = src_n;
        src_n = src.Ptr<St, BORDER_TYPE>(y + 2, border_buffer);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C, MI_BOOL WITH_SCALE>
static Status Sobel1x1NeonHelper(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                                 const std::vector<St> &border_value, const St *border_buffer, const OpTarget &target)
{
    AURA_UNUSED(target);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    const MI_S16 *kernel = g_kernel_tabel[(dx | dy) - 1];
    MI_S32 height   = dst.GetSizes().m_height;

    if (0 == dy)
    {
        ret = wp->ParallelFor(0, height, Sobel1x1DxNeonImpl<St, Dt, BORDER_TYPE, C, WITH_SCALE>,
                              std::cref(src), std::ref(dst), kernel, scale, std::cref(border_value));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel1x1DxNeonImpl<St, Dt, BORDER_TYPE, C, WITH_SCALE> failed");
        }
    }
    else
    {
        ret = wp->ParallelFor(0, height, Sobel1x1DyNeonImpl<St, Dt, BORDER_TYPE, C, WITH_SCALE>,
                              std::cref(src), std::ref(dst), kernel, scale, border_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel1x1DyNeonImpl<St, Dt, BORDER_TYPE, C, WITH_SCALE> failed");
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C>
static Status Sobel1x1NeonHelper(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                                 const std::vector<St> &border_value, const St *border_buffer, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if (NearlyEqual(scale, 1.f))
    {
        ret = Sobel1x1NeonHelper<St, Dt, BORDER_TYPE, C, MI_FALSE>(ctx, src, dst, dx, dy, scale, border_value, border_buffer, target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<St, Dt, BORDER_TYPE, C, MI_FALSE> failed");
        }
    }
    else
    {
        ret = Sobel1x1NeonHelper<St, Dt, BORDER_TYPE, C, MI_TRUE>(ctx, src, dst, dx, dy, scale, border_value, border_buffer, target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<St, Dt, BORDER_TYPE, C, MI_TRUE> failed");
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename St, typename Dt, BorderType BORDER_TYPE>
static Status Sobel1x1NeonHelper(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                                 const std::vector<St> &border_value, const St *border_buffer, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (dst.GetSizes().m_channel)
    {
        case 1:
        {
            ret = Sobel1x1NeonHelper<St, Dt, BORDER_TYPE, 1>(ctx, src, dst, dx, dy, scale, border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<St, Dt, BORDER_TYPE, 1> failed");
            }
            break;
        }

        case 2:
        {
            ret = Sobel1x1NeonHelper<St, Dt, BORDER_TYPE, 2>(ctx, src, dst, dx, dy, scale, border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<St, Dt, BORDER_TYPE, 2> failed");
            }
            break;
        }

        case 3:
        {
            ret = Sobel1x1NeonHelper<St, Dt, BORDER_TYPE, 3>(ctx, src, dst, dx, dy, scale, border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<St, Dt, BORDER_TYPE, 3> failed");
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

template <typename St, typename Dt>
static Status Sobel1x1NeonHelper(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                                 BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    St *border_buffer = MI_NULL;
    std::vector<St> vec_border_value = border_value.ToVector<St>();

    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 channel = dst.GetSizes().m_channel;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            border_buffer = CreateBorderBuffer(ctx, width, channel, vec_border_value);
            if (MI_NULL == border_buffer)
            {
                AURA_ADD_ERROR_STRING(ctx, "CreateBorderBuffer failed");
                return Status::ERROR;
            }

            ret = Sobel1x1NeonHelper<St, Dt, BorderType::CONSTANT>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<St, Dt, BorderType::CONSTANT> failed");
            }

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Sobel1x1NeonHelper<St, Dt, BorderType::REPLICATE>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<St, Dt, BorderType::REPLICATE> failed");
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Sobel1x1NeonHelper<St, Dt, BorderType::REFLECT_101>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<St, Dt, BorderType::REFLECT_101> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported border type");
            return ret;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Sobel1x1Neon(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_F32 scale,
                    BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;
    MI_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = Sobel1x1NeonHelper<MI_U8, MI_S16>(ctx, src, dst, dx, dy, scale, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<MI_U8, MI_S16> failed");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::F32):
        {
            ret = Sobel1x1NeonHelper<MI_U8, MI_F32>(ctx, src, dst, dx, dy, scale, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<MI_U8, MI_F32> failed");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = Sobel1x1NeonHelper<MI_F32, MI_F32>(ctx, src, dst, dx, dy, scale, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel1x1NeonHelper<MI_F32, MI_F32> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported combination of source format and destination format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
