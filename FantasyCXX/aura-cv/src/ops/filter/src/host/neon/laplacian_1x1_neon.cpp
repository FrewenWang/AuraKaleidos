#include "laplacian_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

// Tp = uint16x4_t, int16x4_t
template <typename d16x4_t, typename std::enable_if<std::is_same<d16x4_t, uint16x4_t>::value ||
                                                    std::is_same<d16x4_t, int16x4_t>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE d16x4_t Laplacian1x1Core(d16x4_t &vd16_src_px1, d16x4_t &vd16_src_cx0, d16x4_t &vd16_src_cx1,
                                            d16x4_t &vd16_src_cx2, d16x4_t &vd16_src_nx1)
{
    using D32     = typename Promote<typename neon::Scalar<d16x4_t>::SType>::Type;
    using d32x4_t = typename neon::QVector<D32>::VType;

    d32x4_t vq32_sum_pn = neon::vaddl(vd16_src_px1, vd16_src_nx1);

    d32x4_t vq32_sum_pos_c = neon::vaddl(neon::vext<3>(vd16_src_cx0, vd16_src_cx1), neon::vext<1>(vd16_src_cx1, vd16_src_cx2));
    d32x4_t vq32_sum_neg_c = neon::vshll_n<2>(vd16_src_cx1);
    d32x4_t vq32_sum_pos   = neon::vadd(vq32_sum_pn, vq32_sum_pos_c);

    vd16_src_cx0 = vd16_src_cx1;
    vd16_src_cx1 = vd16_src_cx2;

    return neon::vqmovn(neon::vqsub(vq32_sum_pos, vq32_sum_neg_c));
}

AURA_ALWAYS_INLINE int16x8_t Laplacian1x1Core(uint8x8_t &vdu8_src_px1, uint8x8_t &vdu8_src_cx0,  uint8x8_t &vdu8_src_cx1,
                                              uint8x8_t &vdu8_src_cx2, uint8x8_t &vdu8_src_nx1)
{
    int16x8_t vqs16_sum_pn  = neon::vreinterpret(neon::vaddl(vdu8_src_px1, vdu8_src_nx1));

    uint16x8_t vqu16_sum_c_l0r0 = neon::vaddl(neon::vext<7>(vdu8_src_cx0, vdu8_src_cx1), neon::vext<1>(vdu8_src_cx1, vdu8_src_cx2));
    int16x8_t  vqu16_sum_c      = neon::vreinterpret(neon::vshll_n<2>(vdu8_src_cx1));
    int16x8_t  vqs16_sum_c_l0r0 = neon::vreinterpret(vqu16_sum_c_l0r0);
    int16x8_t  vqs16_sum_c      = neon::vsub(vqs16_sum_c_l0r0, vqu16_sum_c);

    vdu8_src_cx0 = vdu8_src_cx1;
    vdu8_src_cx1 = vdu8_src_cx2;

    return neon::vadd(vqs16_sum_pn, vqs16_sum_c);
}

#if defined(AURA_ENABLE_NEON_FP16)
AURA_ALWAYS_INLINE float16x4_t Laplacian1x1Core(float16x4_t &vdf16_src_px1, float16x4_t &vdf16_src_cx0, float16x4_t &vdf16_src_cx1,
                                                float16x4_t &vdf16_src_cx2, float16x4_t &vdf16_src_nx1)
{
    float32x4_t vqf32_sum_pn     = neon::vadd(neon::vcvt<MI_F32>(vdf16_src_px1), neon::vcvt<MI_F32>(vdf16_src_nx1));

    float32x4_t vqf32_src_cx0    = neon::vcvt<MI_F32>(vdf16_src_cx0);
    float32x4_t vqf32_src_cx1    = neon::vcvt<MI_F32>(vdf16_src_cx1);
    float32x4_t vqf32_src_cx2    = neon::vcvt<MI_F32>(vdf16_src_cx2);
    float32x4_t vqf32_sum_c_l0r0 = neon::vadd(neon::vext<3>(vqf32_src_cx0, vqf32_src_cx1), neon::vext<1>(vqf32_src_cx1, vqf32_src_cx2));
    float32x4_t vqf32_sum_c_c    = neon::vmul(vqf32_src_cx1, 4.f);
    float32x4_t vqf32_sum_c      = neon::vsub(vqf32_sum_c_l0r0, vqf32_sum_c_c);

    vdf16_src_cx0 = vdf16_src_cx1;
    vdf16_src_cx1 = vdf16_src_cx2;

    return neon::vcvt<MI_F16>(neon::vadd(vqf32_sum_pn, vqf32_sum_c));
}
#endif // AURA_ENABLE_NEON_FP16

AURA_ALWAYS_INLINE float32x4_t Laplacian1x1Core(float32x4_t &vqf32_src_px1, float32x4_t &vqf32_src_cx0, float32x4_t &vqf32_src_cx1,
                                                float32x4_t &vqf32_src_cx2, float32x4_t &vqf32_src_nx1)
{
    float32x4_t vqf32_sum_pn     = neon::vadd(vqf32_src_px1, vqf32_src_nx1);
    float32x4_t vqf32_sum_c_l0r0 = neon::vadd(neon::vext<3>(vqf32_src_cx0, vqf32_src_cx1), neon::vext<1>(vqf32_src_cx1, vqf32_src_cx2));
    float32x4_t vqf32_sum_c_c    = neon::vmul(vqf32_src_cx1, 4.f);
    float32x4_t vqf32_sum_c      = neon::vsub(vqf32_sum_c_l0r0, vqf32_sum_c_c);

    vqf32_src_cx0 = vqf32_src_cx1;
    vqf32_src_cx1 = vqf32_src_cx2;

    return neon::vadd(vqf32_sum_pn, vqf32_sum_c);
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C>
static AURA_VOID Laplacian1x1Row(const St *src_p, const St *src_c, const St *src_n, Dt *dst,
                               const std::vector<St> &border_value, MI_S32 width)
{
    using MVSt = typename std::conditional<std::is_same<St, MI_F32>::value, typename neon::MQVector<St, C>::MVType,
                                                                            typename neon::MDVector<St, C>::MVType>::type;
    using MVDt = typename std::conditional<std::is_same<St, MI_U8>::value,  typename neon::MQVector<Dt, C>::MVType, MVSt>::type;

    constexpr MI_S32 ELEM_COUNTS = static_cast<MI_S32>(sizeof(MVDt) / C / sizeof(Dt));
    constexpr MI_S32 VOFFSET     = ELEM_COUNTS * C;
    const MI_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVSt mv_src_p, mv_src_c[3], mv_src_n;
    MVDt mv_result;

    // left
    {
        neon::vload(src_p,           mv_src_p);
        neon::vload(src_c,           mv_src_c[1]);
        neon::vload(src_c + VOFFSET, mv_src_c[2]);
        neon::vload(src_n,           mv_src_n);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_c[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_c[1].val[ch], src_c[ch], border_value[ch]);
            mv_result.val[ch]   = Laplacian1x1Core(mv_src_p.val[ch], mv_src_c[0].val[ch], mv_src_c[1].val[ch],
                                                   mv_src_c[2].val[ch], mv_src_n.val[ch]);
        }
        neon::vstore(dst, mv_result);
    }

    // middle
    {
        for (MI_S32 x = VOFFSET; x < width_align - VOFFSET; x += VOFFSET)
        {
            neon::vload(src_p + x,           mv_src_p);
            neon::vload(src_c + x + VOFFSET, mv_src_c[2]);
            neon::vload(src_n + x,           mv_src_n);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_result.val[ch] = Laplacian1x1Core(mv_src_p.val[ch], mv_src_c[0].val[ch], mv_src_c[1].val[ch],
                                                     mv_src_c[2].val[ch], mv_src_n.val[ch]);
            }
            neon::vstore(dst + x, mv_result);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            MI_S32 x = (width - (ELEM_COUNTS << 1)) * C;

            neon::vload(src_p + x,           mv_src_p);
            neon::vload(src_c + x - VOFFSET, mv_src_c[0]);
            neon::vload(src_c + x,           mv_src_c[1]);
            neon::vload(src_c + x + VOFFSET, mv_src_c[2]);
            neon::vload(src_n + x,           mv_src_n);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_result.val[ch] = Laplacian1x1Core(mv_src_p.val[ch], mv_src_c[0].val[ch], mv_src_c[1].val[ch],
                                                     mv_src_c[2].val[ch], mv_src_n.val[ch]);
            }
            neon::vstore(dst + x, mv_result);
        }
    }

    // right
    {
        MI_S32 x    = (width - ELEM_COUNTS) * C;
        MI_S32 last = (width - 1) * C;

        neon::vload(src_p + x, mv_src_p);
        neon::vload(src_n + x, mv_src_n);

        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_c[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_c[1].val[ch], src_c[last], border_value[ch]);
            mv_result.val[ch]   = Laplacian1x1Core(mv_src_p.val[ch], mv_src_c[0].val[ch], mv_src_c[1].val[ch],
                                                   mv_src_c[2].val[ch], mv_src_n.val[ch]);
            last++;
        }
        neon::vstore(dst + x, mv_result);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, MI_S32 C>
static Status Laplacian1x1NeonImpl(const Mat &src, Mat &dst, const std::vector<St> &border_value,
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
        Laplacian1x1Row<St, Dt, BORDER_TYPE, C>(src_p, src_c, src_n, dst_c, border_value, width);

        src_p = src_c;
        src_c = src_n;
        src_n = src.Ptr<St, BORDER_TYPE>(y + 2, border_buffer);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE>
static Status Laplacian1x1NeonHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<St> &border_value,
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
            ret = wp->ParallelFor(0, height, Laplacian1x1NeonImpl<St, Dt, BORDER_TYPE, 1>, std::cref(src),
                                  std::ref(dst), std::cref(border_value), border_buffer);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonImpl<St, BORDER_TYPE, 1> failed");
            }
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor(0, height, Laplacian1x1NeonImpl<St, Dt, BORDER_TYPE, 2>, std::cref(src),
                                  std::ref(dst), std::cref(border_value), border_buffer);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonImpl<St, BORDER_TYPE, 2> failed");
            }
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor(0, height, Laplacian1x1NeonImpl<St, Dt, BORDER_TYPE, 3>, std::cref(src),
                                  std::ref(dst), std::cref(border_value), border_buffer);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonImpl<St, BORDER_TYPE, 3> failed");
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
static Status Laplacian1x1NeonHelper(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
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

            ret = Laplacian1x1NeonHelper<St, Dt, BorderType::CONSTANT>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonHelper<St, BorderType::CONSTANT> failed");
            }

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Laplacian1x1NeonHelper<St, Dt, BorderType::REPLICATE>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonHelper<St, BorderType::REPLICATE> failed");
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Laplacian1x1NeonHelper<St, Dt, BorderType::REFLECT_101>(ctx, src, dst, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonHelper<St, BorderType::REFLECT_101> failed");
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

Status Laplacian1x1Neon(Context *ctx, const Mat &src, Mat &dst, BorderType border_type,
                        const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;
    MI_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = Laplacian1x1NeonHelper<MI_U8, MI_S16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonHelper<MI_U8> failed");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = Laplacian1x1NeonHelper<MI_U16, MI_U16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonHelper<MI_U16> failed");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = Laplacian1x1NeonHelper<MI_S16, MI_S16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonHelper<MI_S16> failed");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            ret = Laplacian1x1NeonHelper<MI_F16, MI_F16>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonHelper<MI_F16> failed");
            }
            break;
        }
#endif // AURA_ENABLE_NEON_FP16

        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = Laplacian1x1NeonHelper<MI_F32, MI_F32>(ctx, src, dst, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Laplacian1x1NeonHelper<MI_F32> failed");
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
