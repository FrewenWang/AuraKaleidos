#include "sobel_impl.hpp"

#include "aura/runtime/worker_pool.h"

namespace aura
{

const static DT_S16 g_kernel_tabel[5][5] =
{
    { 1,  4,  6,  4, 1},
    {-1, -2,  0,  2, 1},
    { 1,  0, -2,  0, 1},
    {-1,  2,  0, -2, 1},
    { 1, -4,  6, -4, 1}
};

AURA_ALWAYS_INLINE int16x8_t Sobel5x5VCore(uint8x8_t &vdu8_src_p1, uint8x8_t &vdu8_src_p0, uint8x8_t &vdu8_src_c,
                                           uint8x8_t &vdu8_src_n0, uint8x8_t &vdu8_src_n1, const DT_S16 *kernel)
{
    int16x8_t vqs16_sum_p1   = neon::vmul(neon::vreinterpret(neon::vmovl(vdu8_src_p1)), kernel[0]);
    int16x8_t vqs16_sum_p0   = neon::vmul(neon::vreinterpret(neon::vmovl(vdu8_src_p0)), kernel[1]);
    int16x8_t vqs16_sum_c    = neon::vmul(neon::vreinterpret(neon::vmovl(vdu8_src_c)),  kernel[2]);
    int16x8_t vqs16_sum_n0   = neon::vmul(neon::vreinterpret(neon::vmovl(vdu8_src_n0)), kernel[3]);
    int16x8_t vqs16_sum_n1   = neon::vmul(neon::vreinterpret(neon::vmovl(vdu8_src_n1)), kernel[4]);

    int16x8_t vqs16_sum_p0n0 = neon::vadd(vqs16_sum_p0, vqs16_sum_n0);
    int16x8_t vqs16_sum_p1n1 = neon::vadd(vqs16_sum_p1, vqs16_sum_n1);

    return neon::vadd(neon::vadd(vqs16_sum_p0n0, vqs16_sum_p1n1), vqs16_sum_c);
}

AURA_ALWAYS_INLINE int16x8_t Sobel5x5HCore(int16x8_t &vqs16_sum_x0, int16x8_t &vqs16_sum_x1, int16x8_t &vqs16_sum_x2, const DT_S16 *kernel)
{
    int16x8_t vqs16_sum_l1     = neon::vext<6>(vqs16_sum_x0, vqs16_sum_x1);
    int16x8_t vqs16_sum_l0     = neon::vext<7>(vqs16_sum_x0, vqs16_sum_x1);
    int16x8_t vqs16_sum_r0     = neon::vext<1>(vqs16_sum_x1, vqs16_sum_x2);
    int16x8_t vqs16_sum_r1     = neon::vext<2>(vqs16_sum_x1, vqs16_sum_x2);

    int16x8_t vqs16_sum_mul_l1 = neon::vmul(vqs16_sum_l1, kernel[0]);
    int16x8_t vqs16_sum_mul_l0 = neon::vmul(vqs16_sum_l0, kernel[1]);
    int16x8_t vqs16_sum_mul_c  = neon::vmul(vqs16_sum_x1, kernel[2]);
    int16x8_t vqs16_sum_mul_r0 = neon::vmul(vqs16_sum_r0, kernel[3]);
    int16x8_t vqs16_sum_mul_r1 = neon::vmul(vqs16_sum_r1, kernel[4]);

    int16x8_t vqs16_sum_l1r1   = neon::vadd(vqs16_sum_mul_l1, vqs16_sum_mul_r1);
    int16x8_t vqs16_sum_l0r0   = neon::vadd(vqs16_sum_mul_l0, vqs16_sum_mul_r0);

    vqs16_sum_x0 = vqs16_sum_x1;
    vqs16_sum_x1 = vqs16_sum_x2;

    return neon::vadd(neon::vadd(vqs16_sum_l1r1, vqs16_sum_l0r0), vqs16_sum_mul_c);
}

AURA_ALWAYS_INLINE float32x4_t Sobel5x5VCore(float32x4_t &vqf32_src_p1, float32x4_t &vqf32_src_p0, float32x4_t &vqf32_src_c,
                                             float32x4_t &vqf32_src_n0, float32x4_t &vqf32_src_n1, const DT_S16 *kernel)
{
    float32x4_t vqf32_sum_p1   = neon::vmul(vqf32_src_p1, static_cast<DT_F32>(kernel[0]));
    float32x4_t vqf32_sum_p0   = neon::vmul(vqf32_src_p0, static_cast<DT_F32>(kernel[1]));
    float32x4_t vqf32_sum_c    = neon::vmul(vqf32_src_c,  static_cast<DT_F32>(kernel[2]));
    float32x4_t vqf32_sum_n0   = neon::vmul(vqf32_src_n0, static_cast<DT_F32>(kernel[3]));
    float32x4_t vqf32_sum_n1   = neon::vmul(vqf32_src_n1, static_cast<DT_F32>(kernel[4]));

    float32x4_t vqf32_sum_p0n0 = neon::vadd(vqf32_sum_p0, vqf32_sum_n0);
    float32x4_t vqf32_sum_p1n1 = neon::vadd(vqf32_sum_p1, vqf32_sum_n1);

    return neon::vadd(neon::vadd(vqf32_sum_p0n0, vqf32_sum_p1n1), vqf32_sum_c);
}

AURA_ALWAYS_INLINE float32x4_t Sobel5x5HCore(float32x4_t &vqf32_sum_x0, float32x4_t &vqf32_sum_x1, float32x4_t &vqf32_sum_x2, const DT_S16 *kernel)
{
    float32x4_t vqf32_sum_l1     = neon::vext<2>(vqf32_sum_x0, vqf32_sum_x1);
    float32x4_t vqf32_sum_l0     = neon::vext<3>(vqf32_sum_x0, vqf32_sum_x1);
    float32x4_t vqf32_sum_r0     = neon::vext<1>(vqf32_sum_x1, vqf32_sum_x2);
    float32x4_t vqf32_sum_r1     = neon::vext<2>(vqf32_sum_x1, vqf32_sum_x2);

    float32x4_t vqf32_sum_mul_l1 = neon::vmul(vqf32_sum_l1, static_cast<DT_F32>(kernel[0]));
    float32x4_t vqf32_sum_mul_l0 = neon::vmul(vqf32_sum_l0, static_cast<DT_F32>(kernel[1]));
    float32x4_t vqf32_sum_mul_c  = neon::vmul(vqf32_sum_x1, static_cast<DT_F32>(kernel[2]));
    float32x4_t vqf32_sum_mul_r0 = neon::vmul(vqf32_sum_r0, static_cast<DT_F32>(kernel[3]));
    float32x4_t vqf32_sum_mul_r1 = neon::vmul(vqf32_sum_r1, static_cast<DT_F32>(kernel[4]));

    float32x4_t vqf32_sum_l1r1   = neon::vadd(vqf32_sum_mul_l1, vqf32_sum_mul_r1);
    float32x4_t vqf32_sum_l0r0   = neon::vadd(vqf32_sum_mul_l0, vqf32_sum_mul_r0);

    vqf32_sum_x0 = vqf32_sum_x1;
    vqf32_sum_x1 = vqf32_sum_x2;

    return neon::vadd(neon::vadd(vqf32_sum_l1r1, vqf32_sum_l0r0), vqf32_sum_mul_c);
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C, DT_BOOL WITH_SCALE>
static DT_VOID Sobel5x5Row(const St *src_p1, const St *src_p0, const St *src_c, const St *src_n0, const St *src_n1, Dt *dst,
                           const DT_S16 *kernel_x, const DT_S16 *kernel_y, DT_F32 scale, const std::vector<St> &border_value, DT_S32 width)
{
    using MVSt      = typename std::conditional<std::is_same<St, DT_F32>::value, typename neon::MQVector<St, C>::MVType,
                                                typename neon::MDVector<St, C>::MVType>::type;
    using MVSumType = typename std::conditional<std::is_same<St, DT_F32>::value, typename neon::MQVector<St, C>::MVType,
                                                typename neon::MQVector<DT_S16, C>::MVType>::type;

    constexpr DT_S32 ELEM_COUNTS = static_cast<DT_S32>(sizeof(MVSt) / C / sizeof(St));
    constexpr DT_S32 VOFFSET     = ELEM_COUNTS * C;

    const DT_S32 width_align     = (width & -ELEM_COUNTS) * C;

    MVSt mv_src_p1[3], mv_src_p0[3], mv_src_c[3], mv_src_n0[3], mv_src_n1[3];
    MVSumType mv_sum[3], mv_result;

    // left
    {
        neon::vload(src_p1,           mv_src_p1[1]);
        neon::vload(src_p1 + VOFFSET, mv_src_p1[2]);
        neon::vload(src_p0,           mv_src_p0[1]);
        neon::vload(src_p0 + VOFFSET, mv_src_p0[2]);
        neon::vload(src_c,            mv_src_c[1]);
        neon::vload(src_c  + VOFFSET, mv_src_c[2]);
        neon::vload(src_n0,           mv_src_n0[1]);
        neon::vload(src_n0 + VOFFSET, mv_src_n0[2]);
        neon::vload(src_n1,           mv_src_n1[1]);
        neon::vload(src_n1 + VOFFSET, mv_src_n1[2]);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mv_src_p0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mv_src_c[0].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_c[1].val[ch],  src_c[ch],  border_value[ch]);
            mv_src_n0[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mv_src_n1[0].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::LEFT>(mv_src_n1[1].val[ch], src_n1[ch], border_value[ch]);
            mv_sum[0].val[ch]    = Sobel5x5VCore(mv_src_p1[0].val[ch], mv_src_p0[0].val[ch], mv_src_c[0].val[ch],
                                                 mv_src_n0[0].val[ch], mv_src_n1[0].val[ch], kernel_y);
            mv_sum[1].val[ch]    = Sobel5x5VCore(mv_src_p1[1].val[ch], mv_src_p0[1].val[ch], mv_src_c[1].val[ch],
                                                 mv_src_n0[1].val[ch], mv_src_n1[1].val[ch], kernel_y);
            mv_sum[2].val[ch]    = Sobel5x5VCore(mv_src_p1[2].val[ch], mv_src_p0[2].val[ch], mv_src_c[2].val[ch],
                                                 mv_src_n0[2].val[ch], mv_src_n1[2].val[ch], kernel_y);
            mv_result.val[ch]    = Sobel5x5HCore(mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch], kernel_x);
        }
        SobelPostProcess<MVSumType, WITH_SCALE>(mv_result, dst, scale);
    }

    // middle
    {
        for (DT_S32 x = VOFFSET; x < (width_align - VOFFSET); x += VOFFSET)
        {
            neon::vload(src_p1 + x + VOFFSET, mv_src_p1[2]);
            neon::vload(src_p0 + x + VOFFSET, mv_src_p0[2]);
            neon::vload(src_c  + x + VOFFSET, mv_src_c[2]);
            neon::vload(src_n0 + x + VOFFSET, mv_src_n0[2]);
            neon::vload(src_n1 + x + VOFFSET, mv_src_n1[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mv_sum[2].val[ch] = Sobel5x5VCore(mv_src_p1[2].val[ch], mv_src_p0[2].val[ch], mv_src_c[2].val[ch],
                                                  mv_src_n0[2].val[ch], mv_src_n1[2].val[ch], kernel_y);
                mv_result.val[ch] = Sobel5x5HCore(mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch], kernel_x);
            }
            SobelPostProcess<MVSumType, WITH_SCALE>(mv_result, dst + x, scale);
        }
    }

    // back
    {
        if (width_align != width * C)
        {
            DT_S32 x = (width - (ELEM_COUNTS << 1)) * C;

            neon::vload(src_p1 + x - VOFFSET, mv_src_p1[0]);
            neon::vload(src_p1 + x,           mv_src_p1[1]);
            neon::vload(src_p1 + x + VOFFSET, mv_src_p1[2]);
            neon::vload(src_p0 + x - VOFFSET, mv_src_p0[0]);
            neon::vload(src_p0 + x,           mv_src_p0[1]);
            neon::vload(src_p0 + x + VOFFSET, mv_src_p0[2]);
            neon::vload(src_c  + x - VOFFSET, mv_src_c[0]);
            neon::vload(src_c  + x,           mv_src_c[1]);
            neon::vload(src_c  + x + VOFFSET, mv_src_c[2]);
            neon::vload(src_n0 + x - VOFFSET, mv_src_n0[0]);
            neon::vload(src_n0 + x,           mv_src_n0[1]);
            neon::vload(src_n0 + x + VOFFSET, mv_src_n0[2]);
            neon::vload(src_n1 + x - VOFFSET, mv_src_n1[0]);
            neon::vload(src_n1 + x,           mv_src_n1[1]);
            neon::vload(src_n1 + x + VOFFSET, mv_src_n1[2]);

            for (DT_S32 ch = 0; ch < C; ch++)
            {
                mv_sum[0].val[ch] = Sobel5x5VCore(mv_src_p1[0].val[ch], mv_src_p0[0].val[ch], mv_src_c[0].val[ch],
                                                  mv_src_n0[0].val[ch], mv_src_n1[0].val[ch], kernel_y);
                mv_sum[1].val[ch] = Sobel5x5VCore(mv_src_p1[1].val[ch], mv_src_p0[1].val[ch], mv_src_c[1].val[ch],
                                                  mv_src_n0[1].val[ch], mv_src_n1[1].val[ch], kernel_y);
                mv_sum[2].val[ch] = Sobel5x5VCore(mv_src_p1[2].val[ch], mv_src_p0[2].val[ch], mv_src_c[2].val[ch],
                                                  mv_src_n0[2].val[ch], mv_src_n1[2].val[ch], kernel_y);
                mv_result.val[ch] = Sobel5x5HCore(mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch], kernel_x);
            }
            SobelPostProcess<MVSumType, WITH_SCALE>(mv_result, dst + x, scale);
        }
    }

    // right
    {
        DT_S32 x    = (width - ELEM_COUNTS) * C;
        DT_S32 last = (width - 1) * C;

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_p1[2].val[ch], src_p1[last], border_value[ch]);
            mv_src_p0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0[2].val[ch], src_p0[last], border_value[ch]);
            mv_src_c[2].val[ch]  = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_c[2].val[ch],  src_c[last],  border_value[ch]);
            mv_src_n0[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0[2].val[ch], src_n0[last], border_value[ch]);
            mv_src_n1[2].val[ch] = GetBorderVector<BORDER_TYPE, BorderArea::RIGHT>(mv_src_n1[2].val[ch], src_n1[last], border_value[ch]);
            mv_sum[2].val[ch]    = Sobel5x5VCore(mv_src_p1[2].val[ch], mv_src_p0[2].val[ch], mv_src_c[2].val[ch],
                                                 mv_src_n0[2].val[ch], mv_src_n1[2].val[ch], kernel_y);
            mv_result.val[ch]    = Sobel5x5HCore(mv_sum[0].val[ch], mv_sum[1].val[ch], mv_sum[2].val[ch], kernel_x);
            last++;
        }
        SobelPostProcess<MVSumType, WITH_SCALE>(mv_result, dst + x, scale);
    }
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C, DT_BOOL WITH_SCALE>
static Status Sobel5x5NeonImpl(const Mat &src, Mat &dst, const DT_S16 *kernel_x, const DT_S16 *kernel_y, DT_F32 scale,
                               const std::vector<St> &border_value, const St *border_buffer,
                               DT_S32 start_row, DT_S32 end_row)
{
    const St *src_p1 = DT_NULL, *src_p0 = DT_NULL, *src_c = DT_NULL, *src_n0 = DT_NULL, *src_n1 = DT_NULL;
    Dt *dst_c = DT_NULL;

    DT_S32 width = dst.GetSizes().m_width;
    DT_S32 y = start_row;

    src_p1 = src.Ptr<St, BORDER_TYPE>(y - 2, border_buffer);
    src_p0 = src.Ptr<St, BORDER_TYPE>(y - 1, border_buffer);
    src_c  = src.Ptr<St>(y);
    src_n0 = src.Ptr<St, BORDER_TYPE>(y + 1, border_buffer);
    src_n1 = src.Ptr<St, BORDER_TYPE>(y + 2, border_buffer);

    for (; y < end_row; y++)
    {
        dst_c = dst.Ptr<Dt>(y);
        Sobel5x5Row<St, Dt, BORDER_TYPE, C, WITH_SCALE>(src_p1, src_p0, src_c, src_n0, src_n1, dst_c,
                                                        kernel_x, kernel_y, scale, border_value, width);

        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src.Ptr<St, BORDER_TYPE>(y + 3, border_buffer);
    }

    return Status::OK;
}

template <typename St, typename Dt, BorderType BORDER_TYPE, DT_S32 C>
static Status Sobel5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 dx, DT_S32 dy, DT_F32 scale,
                                 const std::vector<St> &border_value, const St *border_buffer, const OpTarget &target)
{
    AURA_UNUSED(target);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    const DT_S16 *kernel_x = g_kernel_tabel[dx];
    const DT_S16 *kernel_y = g_kernel_tabel[dy];
    DT_S32 height    = dst.GetSizes().m_height;

    if (NearlyEqual(scale, 1.f))
    {
        ret = wp->ParallelFor(0, height, Sobel5x5NeonImpl<St, Dt, BORDER_TYPE, C, DT_FALSE>,
                              std::cref(src), std::ref(dst), kernel_x, kernel_y, scale, std::cref(border_value), border_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonImpl<St, Dt, BORDER_TYPE, C, DT_FALSE> failed");
        }
    }
    else
    {
        ret = wp->ParallelFor(0, height, Sobel5x5NeonImpl<St, Dt, BORDER_TYPE, C, DT_TRUE>,
                              std::cref(src), std::ref(dst), kernel_x, kernel_y, scale, std::cref(border_value), border_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonImpl<St, Dt, BORDER_TYPE, C, DT_TRUE> failed");
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename St, typename Dt, BorderType BORDER_TYPE>
static Status Sobel5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 dx, DT_S32 dy, DT_F32 scale,
                                 const std::vector<St> &border_value, const St *border_buffer, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (dst.GetSizes().m_channel)
    {
        case 1:
        {
            ret = Sobel5x5NeonHelper<St, Dt, BORDER_TYPE, 1>(ctx, src, dst, dx, dy, scale, border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonHelper<St, Dt, BORDER_TYPE, 1> failed");
            }
            break;
        }

        case 2:
        {
            ret = Sobel5x5NeonHelper<St, Dt, BORDER_TYPE, 2>(ctx, src, dst, dx, dy, scale, border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonHelper<St, Dt, BORDER_TYPE, 2> failed");
            }
            break;
        }

        case 3:
        {
            ret = Sobel5x5NeonHelper<St, Dt, BORDER_TYPE, 3>(ctx, src, dst, dx, dy, scale, border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonHelper<St, Dt, BORDER_TYPE, 3> failed");
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
static Status Sobel5x5NeonHelper(Context *ctx, const Mat &src, Mat &dst, DT_S32 dx, DT_S32 dy, DT_F32 scale,
                                 BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    St *border_buffer = DT_NULL;
    std::vector<St> vec_border_value = border_value.ToVector<St>();

    DT_S32 width   = dst.GetSizes().m_width;
    DT_S32 channel = dst.GetSizes().m_channel;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            border_buffer = CreateBorderBuffer(ctx, width, channel, vec_border_value);
            if (DT_NULL == border_buffer)
            {
                AURA_ADD_ERROR_STRING(ctx, "CreateBorderBuffer failed");
                return Status::ERROR;
            }

            ret = Sobel5x5NeonHelper<St, Dt, BorderType::CONSTANT>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonHelper<St, Dt, BorderType::CONSTANT> failed");
            }

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Sobel5x5NeonHelper<St, Dt, BorderType::REPLICATE>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonHelper<St, Dt, BorderType::REPLICATE> failed");
            }
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Sobel5x5NeonHelper<St, Dt, BorderType::REFLECT_101>(ctx, src, dst, dx, dy, scale, vec_border_value, border_buffer, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonHelper<St, Dt, BorderType::REFLECT_101> failed");
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

Status Sobel5x5Neon(Context *ctx, const Mat &src, Mat &dst, DT_S32 dx, DT_S32 dy, DT_F32 scale,
                    BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;
    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = Sobel5x5NeonHelper<DT_U8, DT_S16>(ctx, src, dst, dx, dy, scale, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonHelper<DT_U8, DT_S16> failed");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::F32):
        {
            ret = Sobel5x5NeonHelper<DT_U8, DT_F32>(ctx, src, dst, dx, dy, scale, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonHelper<DT_U8, DT_F32> failed");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = Sobel5x5NeonHelper<DT_F32, DT_F32>(ctx, src, dst, dx, dy, scale, border_type, border_value, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Sobel5x5NeonHelper<DT_F32, DT_F32> failed");
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
