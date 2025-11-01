#include "laplacian_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

// d16x4_t = uint16x4_t, int16x4_t
template <typename d16x4_t, typename d32x4_t = typename neon::WVectorBits<d16x4_t>::VType,
          typename std::enable_if<std::is_same<d16x4_t, uint16x4_t>::value || std::is_same<d16x4_t, int16x4_t>::value>::type * = MI_NULL>
AURA_ALWAYS_INLINE d32x4_t Laplacian3x3VCore(d16x4_t &vd16_src_p, d16x4_t &vd16_src_n)
{
    return neon::vaddl(vd16_src_p, vd16_src_n);
}

// d16x4_t = uint16x4_t, int16x4_t
template <typename d16x4_t, typename d32x4_t = typename neon::WVectorBits<d16x4_t>::VType,
          typename std::enable_if<std::is_same<d16x4_t, uint16x4_t>::value || std::is_same<d16x4_t, int16x4_t>::value>::type * = MI_NULL>
AURA_ALWAYS_INLINE d16x4_t Laplacian3x3HCore(d16x4_t &vd16_sum_x1, d32x4_t &vq32_sum_x0,
                                             d32x4_t &vq32_sum_x1, d32x4_t &vq32_sum_x2)
{
    d32x4_t vq32_sum_l  = neon::vext<3>(vq32_sum_x0, vq32_sum_x1);
    d32x4_t vq32_sum_r  = neon::vext<1>(vq32_sum_x1, vq32_sum_x2);
    d32x4_t vq32_sum_c  = neon::vshll_n<3>(vd16_sum_x1);

    d32x4_t vq32_result = neon::vadd(vq32_sum_l, vq32_sum_r);
    vq32_result         = neon::vadd(vq32_result, vq32_result);

    vq32_sum_x0 = vq32_sum_x1;
    vq32_sum_x1 = vq32_sum_x2;

    return neon::vqmovn(neon::vqsub(vq32_result, vq32_sum_c));
}

AURA_ALWAYS_INLINE int16x8_t Laplacian3x3VCore(uint8x8_t &vdu8_src_p, uint8x8_t &vdu8_src_n)
{
    return neon::vreinterpret(neon::vaddl(vdu8_src_p, vdu8_src_n));
}

AURA_ALWAYS_INLINE int16x8_t Laplacian3x3HCore(uint8x8_t &vdu8_sum_x1, int16x8_t &vqs16_sum_x0,
                                               int16x8_t &vqs16_sum_x1, int16x8_t &vqs16_sum_x2)
{
    int16x8_t vqs16_sum_l  = neon::vext<7>(vqs16_sum_x0, vqs16_sum_x1);
    int16x8_t vqs16_sum_r  = neon::vext<1>(vqs16_sum_x1, vqs16_sum_x2);
    int16x8_t vqs16_sum_c  = neon::vreinterpret(neon::vshll_n<2>(vdu8_sum_x1));

    int16x8_t vqs16_result = neon::vadd(vqs16_sum_l, vqs16_sum_r);
    vqs16_result           = neon::vsub(vqs16_result, vqs16_sum_c);

    vqs16_sum_x0 = vqs16_sum_x1;
    vqs16_sum_x1 = vqs16_sum_x2;

    return neon::vadd(vqs16_result, vqs16_result);
}

#if defined(AURA_ENABLE_NEON_FP16)
AURA_ALWAYS_INLINE float32x4_t Laplacian3x3VCore(float16x4_t &vdf16_src_p, float16x4_t &vdf16_src_n)
{
    return neon::vadd(neon::vcvt<MI_F32>(vdf16_src_p), neon::vcvt<MI_F32>(vdf16_src_n));
}

AURA_ALWAYS_INLINE float16x4_t Laplacian3x3HCore(float16x4_t &vdf16_src_x1, float32x4_t &vqf32_sum_x0,
                                                 float32x4_t &vqf32_sum_x1, float32x4_t &vqf32_sum_x2)
{
    float32x4_t vqf32_sum_l  = neon::vext<3>(vqf32_sum_x0, vqf32_sum_x1);
    float32x4_t vqf32_sum_r  = neon::vext<1>(vqf32_sum_x1, vqf32_sum_x2);
    float32x4_t vqf32_sum_c  = neon::vmul(neon::vcvt<MI_F32>(vdf16_src_x1), 4.f);

    float32x4_t vqf32_result = neon::vadd(vqf32_sum_l, vqf32_sum_r);
    vqf32_result             = neon::vsub(vqf32_result, vqf32_sum_c);

    vqf32_sum_x0 = vqf32_sum_x1;
    vqf32_sum_x1 = vqf32_sum_x2;

    return neon::vcvt<MI_F16>(neon::vadd(vqf32_result, vqf32_result));
}
#endif // AURA_ENABLE_NEON_FP16

AURA_ALWAYS_INLINE float32x4_t Laplacian3x3VCore(float32x4_t &vqf32_src_p, float32x4_t &vqf32_src_n)
{
    return neon::vadd(vqf32_src_p, vqf32_src_n);
}

AURA_ALWAYS_INLINE float32x4_t Laplacian3x3HCore(float32x4_t &vqf32_src_x1, float32x4_t &vqf32_sum_x0,
                                                 float32x4_t &vqf32_sum_x1, float32x4_t &vqf32_sum_x2)
{
    float32x4_t vqf32_sum_l  = neon::vext<3>(vqf32_sum_x0, vqf32_sum_x1);
    float32x4_t vqf32_sum_r  = neon::vext<1>(vqf32_sum_x1, vqf32_sum_x2);
    float32x4_t vqf32_sum_c  = neon::vmul(vqf32_src_x1, 4.f);

    float32x4_t vqf32_result = neon::vadd(vqf32_sum_l, vqf32_sum_r);
    vqf32_result             = neon::vsub(vqf32_result, vqf32_sum_c);

    vqf32_sum_x0 = vqf32_sum_x1;
    vqf32_sum_x1 = vqf32_sum_x2;

    return neon::vadd(vqf32_result, vqf32_result);
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C>
static AURA_VOID Laplacian3x3Row(const St *src_p, const St *src_c, const St *src_n, Dt *dst,
                               const std::vector<St> &border_value, MI_S32 width)
{
    using MVSt      = typename std::conditional<std::is_same<St, MI_F32>::value, typename neon::MQVector<St, C>::MVType,
                                                                                 typename neon::MDVector<St, C>::MVType>::type;
    using MVDt      = typename std::conditional<std::is_same<St, MI_U8>::value,  typename neon::MQVector<Dt, C>::MVType, MVSt>::type;
    using MVSumType = typename std::conditional<sizeof(St) == 2, typename neon::MQVector<typename Promote<St>::Type, C>::MVType, MVDt>::type;

    constexpr MI_S32 ELEM_COUNTS = static_cast<MI_S32>(sizeof(MVDt) / C / sizeof(Dt));
    constexpr MI_S32 VOFFSET     = ELEM_COUNTS * C;
    const MI_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVSt mv_src_p[3], mv_src_c, mv_src_n[3];
    MVDt mv_result;
    MVSumType mv_sum[3];

    // left
    {
        neon::vload(src_p,           mv_src_p[1]);
        neon::vload(src_p + VOFFSET, mv_src_p[2]);
        neon::vload(src_c,           mv_src_c);
        neon::vload(src_n,           mv_src_n[1]);
        neon::vload(src_n + VOFFSET, mv_src_n[2]);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_p[1].val[ch], src_p[ch], border_value[ch]);
            mv_src_n[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_n[1].val[ch], src_n[ch], border_value[ch]);
            mv_sum[0].val[ch]   = Laplacian3x3VCore(mv_src_p[0].val[ch], mv_src_n[0].val[ch]);
            mv_sum[1].val[ch]   = Laplacian3x3VCore(mv_src_p[1].val[ch], mv_src_n[1].val[ch]);
            mv_sum[2].val[ch]   = Laplacian3x3VCore(mv_src_p[2].val[ch], mv_src_n[2].val[ch]);
            mv_result.val[ch]   = Laplacian3x3HCore(mv_src_c.val[ch], mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch]);
        }
        neon::vstore(dst, mv_result);
    }

    // middle
    {
        for (MI_S32 x = VOFFSET; x < width_align - VOFFSET; x += VOFFSET)
        {
            neon::vload(src_p + x + VOFFSET, mv_src_p[2]);
            neon::vload(src_c + x,           mv_src_c);
            neon::vload(src_n + x + VOFFSET, mv_src_n[2]);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_sum[2].val[ch] = Laplacian3x3VCore(mv_src_p[2].val[ch], mv_src_n[2].val[ch]);
                mv_result.val[ch] = Laplacian3x3HCore(mv_src_c.val[ch], mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch]);
            }
            neon::vstore(dst + x, mv_result);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            MI_S32 x = (width - (ELEM_COUNTS << 1)) * C;

            neon::vload(src_p + x - VOFFSET, mv_src_p[0]);
            neon::vload(src_p + x,           mv_src_p[1]);
            neon::vload(src_p + x + VOFFSET, mv_src_p[2]);
            neon::vload(src_c + x,           mv_src_c);
            neon::vload(src_n + x - VOFFSET, mv_src_n[0]);
            neon::vload(src_n + x,           mv_src_n[1]);
            neon::vload(src_n + x + VOFFSET, mv_src_n[2]);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_sum[0].val[ch] = Laplacian3x3VCore(mv_src_p[0].val[ch], mv_src_n[0].val[ch]);
                mv_sum[1].val[ch] = Laplacian3x3VCore(mv_src_p[1].val[ch], mv_src_n[1].val[ch]);
                mv_sum[2].val[ch] = Laplacian3x3VCore(mv_src_p[2].val[ch], mv_src_n[2].val[ch]);
                mv_result.val[ch] = Laplacian3x3HCore(mv_src_c.val[ch], mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch]);
            }
            neon::vstore(dst + x, mv_result);
        }
    }

    // right
    {
        MI_S32 x    = (width - ELEM_COUNTS) * C;
        MI_S32 last = (width - 1) * C;

        neon::vload(src_p + x, mv_src_p[1]);
        neon::vload(src_c + x, mv_src_c);
        neon::vload(src_n + x, mv_src_n[1]);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_p[1].val[ch], src_p[last], border_value[ch]);
            mv_src_n[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_n[1].val[ch], src_n[last], border_value[ch]);
            mv_sum[2].val[ch]   = Laplacian3x3VCore(mv_src_p[2].val[ch], mv_src_n[2].val[ch]);
            mv_result.val[ch]   = Laplacian3x3HCore(mv_src_c.val[ch], mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch]);
            last++;
        }
        neon::vstore(dst + x, mv_result);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C>
static Status Laplacian3x3NeonImpl(const Mat &src, Mat &dst, const std::vector<St> &border_value,
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
        Laplacian3x3Row<St, Dt, BORDER_TYPE, C>(src_p, src_c, src_n, dst_c, border_value, width);

        src_p = src_c;
        src_c = src_n;
        src_n = src.Ptr<St, BORDER_TYPE>(y + 2, border_buffer);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE>
static Status Laplacian3x3NeonHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<St> &border_value,
                                     const St *border_buffer, const OpTarget &target)
{
    Status ret = Status::ERROR;

    AURA_UNUSED(target);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    MI_S32 height  = dst.GetSizes().m_height;
    MI_S32 channel = dst.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, height, Laplacian3x3NeonImpl<St, Dt, BORDER_TYPE, 1>, std::cref(src),
                                  std::ref(dst), std::cref(border_value), border_buffer);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonImpl<St, BORDER_TYPE, 1> failed");
            }
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Laplacian3x3NeonImpl<St, Dt, BORDER_TYPE, 2>, std::cref(src),
                                  std::ref(dst), std::cref(border_value), border_buffer);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonImpl<St, BORDER_TYPE, 2> failed");
            }
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Laplacian3x3NeonImpl<St, Dt, BORDER_TYPE, 3>, std::cref(src),
                                  std::ref(dst), std::cref(border_value), border_buffer);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonImpl<St, BORDER_TYPE, 3> failed");
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
static Status Laplacian3x3NeonHelper(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                                     const Scalar &border_value, const OpTarget &target)
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

            ret = Laplacian3x3NeonHelper<St, Dt, BorderType::CONSTANT>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonHelper<St, BorderType::CONSTANT> failed");
            }

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Laplacian3x3NeonHelper<St, Dt, BorderType::REPLICATE>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonHelper<St, BorderType::REPLICATE> failed");
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Laplacian3x3NeonHelper<St, Dt, BorderType::REFLECT_101>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonHelper<St, BorderType::REFLECT_101> failed");
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

Status Laplacian3x3Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;
    MI_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = Laplacian3x3NeonHelper<MI_U8, MI_S16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonHelper<MI_U8> failed");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = Laplacian3x3NeonHelper<MI_U16, MI_U16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonHelper<MI_U16> failed");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = Laplacian3x3NeonHelper<MI_S16, MI_S16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonHelper<MI_S16> failed");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            ret = Laplacian3x3NeonHelper<MI_F16, MI_F16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonHelper<MI_F16> failed");
            }
            break;
        }
#endif // AURA_ENABLE_NEON_FP16

        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = Laplacian3x3NeonHelper<MI_F32, MI_F32>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian3x3NeonHelper<MI_F32> failed");
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
