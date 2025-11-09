#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

// Tp = DT_U8, DT_S8
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value, Status>::type
ResizeBnCommNeonImpl(Context *ctx, const Mat &src, Mat &dst,
                     const DT_S32 *buffer, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using BufType          = typename ResizeBnCuTraits<Tp>::BufType;
    using AlphaType        = typename ResizeBnCuTraits<Tp>::AlphaType;
    using MVType           = typename neon::MDVector<Tp, C>::MVType;
    using MVTypeD8         = typename neon::MDVector<Tp, 1>::MVType;
    using MVTypeD32        = typename neon::MDVector<BufType, C>::MVType;
    using VType            = typename neon::QVector<BufType>::VType;
    auto ResizeCastFunctor = typename ResizeBnCuTraits<Tp>::ResizeCastFunctor();

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    const DT_S32 *xofs     = buffer;
    const DT_S32 *yofs     = xofs + owidth;
    const AlphaType *alpha = reinterpret_cast<const AlphaType*>(yofs + oheight);
    const AlphaType *beta  = alpha + 2 * owidth + 2 * start_row;

    BufType *rows = thread_buffer.GetThreadData<BufType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    BufType *rows0 = rows;
    BufType *rows1 = rows + C * owidth;

    DT_F64 scale_x         = static_cast<DT_F64>(iwidth) / owidth;
    DT_S32 loop_end_owidth = Ceil(((iwidth - 8) + 0.5) / scale_x - 0.5);
    DT_S32 end_x           = loop_end_owidth & (-2);

    DT_S32 prev_sy1 = -1;
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy                  = yofs[y];
        const AlphaType *alpha_ptr = alpha;

        if (sy == prev_sy1)
        {
            BufType *rows0_tmp = rows0;
            rows0              = rows1;
            rows1              = rows0_tmp;
            BufType *rows1_tmp = rows1;

            const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

            DT_S32 x = 0;
            for (; x < end_x; x += 2)
            {
                DT_S32 sx_x0    = C * xofs[x];
                DT_S32 sx_x1    = C * xofs[x + 1];
                const Tp *r1_x0 = src_row1 + sx_x0;
                const Tp *r1_x1 = src_row1 + sx_x1;

                MVType mvd8_x0, mvd8_x1;
                neon::vload(r1_x0, mvd8_x0);
                neon::vload(r1_x1, mvd8_x1);
                auto vq32_a = neon::vmovl(neon::vload1(alpha_ptr));

                MVTypeD32 mvd32_result;
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    int32x2_t vds32_x0 = neon::vgetlow(neon::vmovl(neon::vgetlow(neon::vmovl(mvd8_x0.val[ch]))));
                    int32x2_t vds32_x1 = neon::vgetlow(neon::vmovl(neon::vgetlow(neon::vmovl(mvd8_x1.val[ch]))));

                    int32x4_t vqs32_result = neon::vcombine(vds32_x0, vds32_x1);
                    vqs32_result           = neon::vmul(vqs32_result, vq32_a);
                    mvd32_result.val[ch]   = neon::vpadd(neon::vgetlow(vqs32_result), neon::vgethigh(vqs32_result));
                }
                neon::vstore(rows1_tmp, mvd32_result);

                rows1_tmp += 2 * C;
                alpha_ptr += 4;
            }

            for (; x < owidth; x++)
            {
                DT_S32 sx    = C * xofs[x];
                const Tp *r1 = src_row1 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];

                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    rows1_tmp[ch] = r1[ch] * a0 + r1[ch + C] * a1;
                }

                rows1_tmp += C;
                alpha_ptr += 2;
            }
        }
        else if (sy > prev_sy1)
        {
            const Tp *src_row0 = src.Ptr<Tp>(sy);
            const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

            BufType *rows0_tmp = rows0;
            BufType *rows1_tmp = rows1;

            DT_S32 x = 0;
            for (; x < end_x; x += 2)
            {
                DT_S32 sx_x0    = C * xofs[x];
                DT_S32 sx_x1    = C * xofs[x + 1];
                const Tp *r0_x0 = src_row0 + sx_x0;
                const Tp *r0_x1 = src_row0 + sx_x1;
                const Tp *r1_x0 = src_row1 + sx_x0;
                const Tp *r1_x1 = src_row1 + sx_x1;

                MVType mvd8_x0, mvd8_x1;
                neon::vload(r1_x0, mvd8_x0);
                neon::vload(r1_x1, mvd8_x1);
                auto vq32_a = neon::vmovl(neon::vload1(alpha_ptr));

                MVTypeD32 mvd32_result;
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    int32x2_t vds32_x0 = neon::vgetlow(neon::vmovl(neon::vgetlow(neon::vmovl(mvd8_x0.val[ch]))));
                    int32x2_t vds32_x1 = neon::vgetlow(neon::vmovl(neon::vgetlow(neon::vmovl(mvd8_x1.val[ch]))));

                    int32x4_t vqs32_result = neon::vcombine(vds32_x0, vds32_x1);
                    vqs32_result           = neon::vmul(vqs32_result, vq32_a);
                    mvd32_result.val[ch]   = neon::vpadd(neon::vgetlow(vqs32_result), neon::vgethigh(vqs32_result));
                }
                neon::vstore(rows1_tmp, mvd32_result);

                neon::vload(r0_x0, mvd8_x0);
                neon::vload(r0_x1, mvd8_x1);
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    int32x2_t vsd32_x0 = neon::vgetlow(neon::vmovl(neon::vgetlow(neon::vmovl(mvd8_x0.val[ch]))));
                    int32x2_t vsd32_x1 = neon::vgetlow(neon::vmovl(neon::vgetlow(neon::vmovl(mvd8_x1.val[ch]))));

                    int32x4_t vqs32_result = neon::vcombine(vsd32_x0, vsd32_x1);
                    vqs32_result           = neon::vmul(vqs32_result, vq32_a);
                    mvd32_result.val[ch]   = neon::vpadd(neon::vgetlow(vqs32_result), neon::vgethigh(vqs32_result));
                }
                neon::vstore(rows0_tmp, mvd32_result);

                rows0_tmp += 2 * C;
                rows1_tmp += 2 * C;
                alpha_ptr += 4;
            }

            for (; x < owidth; x++)
            {
                DT_S32 sx    = C * xofs[x];
                const Tp *r0 = src_row0 + sx;
                const Tp *r1 = src_row1 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];

                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    rows0_tmp[ch] = r0[ch] * a0 + r0[ch + C] * a1;
                    rows1_tmp[ch] = r1[ch] * a0 + r1[ch + C] * a1;
                }

                rows0_tmp += C;
                rows1_tmp += C;
                alpha_ptr += 2;
            }
        }
        prev_sy1 = sy + 1;

        AlphaType b0 = beta[0];
        AlphaType b1 = beta[1];

        BufType *rows0_tmp = rows0;
        BufType *rows1_tmp = rows1;
        Tp *dst_c_ptr      = dst.Ptr<Tp>(y);

        VType vq32_b0, vq32_b1;
        neon::vdup(vq32_b0, static_cast<BufType>(b0));
        neon::vdup(vq32_b1, static_cast<BufType>(b1));

        DT_S32 owidth_xc        = owidth * C;
        DT_S32 owidth_xc_align8 = owidth_xc & (-8);
        DT_S32 x                = 0;
        for (; x < owidth_xc_align8; x += 8)
        {
            auto vq32_cx0  = neon::vload1q(rows0_tmp);
            auto vq32_n0x0 = neon::vload1q(rows1_tmp);
            auto vq32_cx1  = neon::vload1q(rows0_tmp + 4);
            auto vq32_n0x1 = neon::vload1q(rows1_tmp + 4);

            auto vq32_x0 = neon::vmul(vq32_cx0, vq32_b0);
            auto vq32_x1 = neon::vmul(vq32_cx1, vq32_b0);
            vq32_x0      = neon::vmla(vq32_x0, vq32_n0x0, vq32_b1);
            vq32_x1      = neon::vmla(vq32_x1, vq32_n0x1, vq32_b1);
            MVTypeD8 vd8_result;
            if (std::is_same<Tp, DT_U8>::value)
            {
                uint32x4_t vqu32_x0     = neon::vreinterpret(vq32_x0);
                uint32x4_t vqu32_x1     = neon::vreinterpret(vq32_x1);
                uint16x8_t vqu16_result = neon::vcombine(neon::vqshrn_n<14>(vqu32_x0), neon::vqshrn_n<14>(vqu32_x1));
                vd8_result.val[0]       = neon::vqrshrn_n<8>(vqu16_result);
            }
            else
            {
                int16x8_t vqs16_result = neon::vcombine(neon::vqshrn_n<14>(vq32_x0), neon::vqshrn_n<14>(vq32_x1));
                vd8_result.val[0]      = neon::vqrshrn_n<8>(vqs16_result);
            }

            neon::vstore(dst_c_ptr, vd8_result);

            dst_c_ptr += 8;
            rows0_tmp += 8;
            rows1_tmp += 8;
        }

        for (; x < owidth_xc; x++)
        {
            *dst_c_ptr++ = ResizeCastFunctor((*rows0_tmp++) * b0 + (*rows1_tmp++) * b1);
        }

        beta += 2;
    }

    AURA_RETURN(ctx, ret);
}

// Tp = DT_U16, DT_S16
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value || std::is_same<DT_S16, Tp>::value, Status>::type
ResizeBnCommNeonImpl(Context *ctx, const Mat &src, Mat &dst,
                     const DT_S32 *buffer, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using BufType   = typename ResizeBnCuTraits<Tp>::BufType;
    using AlphaType = typename ResizeBnCuTraits<Tp>::AlphaType;
    using MovlType  = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType    = typename neon::MDVector<Tp, C>::MVType;
    using MVTypeF32 = typename neon::MDVector<BufType, C>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    const DT_S32 *xofs     = buffer;
    const DT_S32 *yofs     = xofs + owidth;
    const AlphaType *alpha = reinterpret_cast<const AlphaType*>(yofs + oheight);
    const AlphaType *beta  = alpha + 2 * owidth + 2 * start_row;

    BufType *rows = thread_buffer.GetThreadData<BufType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    BufType *rows0 = rows;
    BufType *rows1 = rows + C * owidth;

    DT_F64 scale_x         = static_cast<DT_F64>(iwidth) / owidth;
    DT_S32 loop_end_owidth = Ceil(((iwidth - 8) + 0.5) / scale_x - 0.5);
    DT_S32 end_x           = loop_end_owidth & (-2);

    DT_S32 prev_sy1 = -1;
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy                  = yofs[y];
        const AlphaType *alpha_ptr = alpha;

        if (sy == prev_sy1)
        {
            BufType *rows0_tmp = rows0;
            rows0              = rows1;
            rows1              = rows0_tmp;
            BufType *rows1_tmp = rows1;

            const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

            DT_S32 x = 0;
            for (; x < end_x; x += 2)
            {
                DT_S32 sx_x0    = C * xofs[x];
                DT_S32 sx_x1    = C * xofs[x + 1];
                const Tp *r1_x0 = src_row1 + sx_x0;
                const Tp *r1_x1 = src_row1 + sx_x1;

                MVType mvd16_x0, mvd16_x1;
                neon::vload(r1_x0, mvd16_x0);
                neon::vload(r1_x1, mvd16_x1);
                float32x4_t vqf32_a = neon::vload1q(alpha_ptr);

                MVTypeF32 mvdf32_result;
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    auto vq32_result         = neon::vcombine(neon::vgetlow(neon::vmovl(mvd16_x0.val[ch])),
                                                              neon::vgetlow(neon::vmovl(mvd16_x1.val[ch])));
                    float32x4_t vqf32_result = neon::vcvt<BufType>(vq32_result);
                    vqf32_result             = neon::vmul(vqf32_result, vqf32_a);
                    mvdf32_result.val[ch]    = neon::vpadd(neon::vgetlow(vqf32_result), neon::vgethigh(vqf32_result));
                }
                neon::vstore(rows1_tmp, mvdf32_result);

                rows1_tmp += 2 * C;
                alpha_ptr += 4;
            }

            for (; x < owidth; x++)
            {
                DT_S32 sx    = C * xofs[x];
                const Tp *r1 = src_row1 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];

                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    rows1_tmp[ch] = r1[ch] * a0 + r1[ch + C] * a1;
                }

                rows1_tmp += C;
                alpha_ptr += 2;
            }
        }
        else if (sy > prev_sy1)
        {
            const Tp *src_row0 = src.Ptr<Tp>(sy);
            const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

            BufType *rows0_tmp = rows0;
            BufType *rows1_tmp = rows1;

            DT_S32 x = 0;
            for (; x < end_x; x += 2)
            {
                DT_S32 sx_x0    = C * xofs[x];
                DT_S32 sx_x1    = C * xofs[x + 1];
                const Tp *r0_x0 = src_row0 + sx_x0;
                const Tp *r1_x0 = src_row1 + sx_x0;
                const Tp *r0_x1 = src_row0 + sx_x1;
                const Tp *r1_x1 = src_row1 + sx_x1;

                MVType mvd16_x0, mvd16_x1;
                neon::vload(r1_x0, mvd16_x0);
                neon::vload(r1_x1, mvd16_x1);
                float32x4_t vqf32_a = neon::vload1q(alpha_ptr);

                MVTypeF32 mvdf32_result;
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    auto vq32_result         = neon::vcombine(neon::vgetlow(neon::vmovl(mvd16_x0.val[ch])),
                                                              neon::vgetlow(neon::vmovl(mvd16_x1.val[ch])));
                    float32x4_t vqf32_result = neon::vcvt<BufType>(vq32_result);
                    vqf32_result             = neon::vmul(vqf32_result, vqf32_a);
                    mvdf32_result.val[ch]    = neon::vpadd(neon::vgetlow(vqf32_result), neon::vgethigh(vqf32_result));
                }
                neon::vstore(rows1_tmp, mvdf32_result);

                neon::vload(r0_x0, mvd16_x0);
                neon::vload(r0_x1, mvd16_x1);
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    auto vq32_result         = neon::vcombine(neon::vgetlow(neon::vmovl(mvd16_x0.val[ch])),
                                                              neon::vgetlow(neon::vmovl(mvd16_x1.val[ch])));
                    float32x4_t vqf32_result = neon::vcvt<BufType>(vq32_result);
                    vqf32_result             = neon::vmul(vqf32_result, vqf32_a);
                    mvdf32_result.val[ch]    = neon::vpadd(neon::vgetlow(vqf32_result), neon::vgethigh(vqf32_result));
                }
                neon::vstore(rows0_tmp, mvdf32_result);

                rows0_tmp += 2 * C;
                rows1_tmp += 2 * C;
                alpha_ptr += 4;
            }

            for (; x < owidth; x++)
            {
                DT_S32 sx    = C * xofs[x];
                const Tp *r0 = src_row0 + sx;
                const Tp *r1 = src_row1 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    rows0_tmp[ch] = r0[ch] * a0 + r0[ch + C] * a1;
                    rows1_tmp[ch] = r1[ch] * a0 + r1[ch + C] * a1;
                }

                rows0_tmp += C;
                rows1_tmp += C;
                alpha_ptr += 2;
            }
        }
        prev_sy1 = sy + 1;

        AlphaType b0 = beta[0];
        AlphaType b1 = beta[1];

        BufType *rows0_tmp = rows0;
        BufType *rows1_tmp = rows1;
        Tp *dst_c_ptr      = dst.Ptr<Tp>(y);

        float32x4_t vqf32_b0, vqf32_b1;
        neon::vdup(vqf32_b0, b0);
        neon::vdup(vqf32_b1, b1);

        DT_S32 owidth_xc        = owidth * C;
        DT_S32 owidth_xc_align8 = owidth_xc & (-8);
        DT_S32 x                = 0;
        for (; x < owidth_xc_align8; x += 8)
        {
            float32x4_t vqf32_cx0  = neon::vload1q(rows0_tmp);
            float32x4_t vqf32_n0x0 = neon::vload1q(rows1_tmp);
            float32x4_t vqf32_cx1  = neon::vload1q(rows0_tmp + 4);
            float32x4_t vqf32_n0x1 = neon::vload1q(rows1_tmp + 4);

            float32x4_t vqf32_x0 = neon::vmul(vqf32_cx0, vqf32_b0);
            float32x4_t vqf32_x1 = neon::vmul(vqf32_cx1, vqf32_b0);

            vqf32_x0 = neon::vmla(vqf32_x0, vqf32_n0x0, vqf32_b1);
            vqf32_x1 = neon::vmla(vqf32_x1, vqf32_n0x1, vqf32_b1);

            auto vq32_x0 = neon::vcvt<MovlType>(vqf32_x0);
            auto vq32_x1 = neon::vcvt<MovlType>(vqf32_x1);

            auto vq16_result = neon::vcombine(neon::vmovn(vq32_x0), neon::vmovn(vq32_x1));
            neon::vstore(dst_c_ptr, vq16_result);

            dst_c_ptr += 8;
            rows0_tmp += 8;
            rows1_tmp += 8;
        }

        for (; x < owidth_xc; x++)
        {
            *dst_c_ptr++ = SaturateCast<Tp>((*rows0_tmp++) * b0 + (*rows1_tmp++) * b1);
        }

        beta += 2;
    }

    AURA_RETURN(ctx, ret);
}

// Tp = DT_F32
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_F32, Tp>::value, Status>::type
ResizeBnCommNeonImpl(Context *ctx, const Mat &src, Mat &dst,
                     const DT_S32 *buffer, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using BufType   = typename ResizeBnCuTraits<Tp>::BufType;
    using AlphaType = typename ResizeBnCuTraits<Tp>::AlphaType;
    using MVType    = typename neon::MDVector<Tp, C>::MVType;
    using MVTypeF32 = typename neon::MDVector<BufType, C>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    const DT_S32 *xofs     = buffer;
    const DT_S32 *yofs     = xofs + owidth;
    const AlphaType *alpha = reinterpret_cast<const AlphaType*>(yofs + oheight);
    const AlphaType *beta  = alpha + 2 * owidth + 2 * start_row;

    BufType *rows = thread_buffer.GetThreadData<BufType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    BufType *rows0 = rows;
    BufType *rows1 = rows + C * owidth;

    DT_F64 scale_x         = static_cast<DT_F64>(iwidth) / owidth;
    DT_S32 loop_end_owidth = Ceil(((iwidth - 8) + 0.5) / scale_x - 0.5);
    DT_S32 end_x           = loop_end_owidth & (-2);

    DT_S32 prev_sy1 = -1;
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy                  = yofs[y];
        const AlphaType *alpha_ptr = alpha;

        if (sy == prev_sy1)
        {
            BufType *rows0_tmp = rows0;
            rows0              = rows1;
            rows1              = rows0_tmp;
            BufType *rows1_tmp = rows1;

            const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

            DT_S32 x = 0;
            for (; x < end_x; x += 2)
            {
                DT_S32 sx_x0    = C * xofs[x];
                DT_S32 sx_x1    = C * xofs[x + 1];
                const Tp *r1_x0 = src_row1 + sx_x0;
                const Tp *r1_x1 = src_row1 + sx_x1;

                MVType mvdf32_x0, mvdf32_x1;
                neon::vload(r1_x0, mvdf32_x0);
                neon::vload(r1_x1, mvdf32_x1);
                float32x4_t vqf32_a = neon::vload1q(alpha_ptr);

                MVTypeF32 mvdf32_result;
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    float32x4_t vqf32_result = neon::vcombine(mvdf32_x0.val[ch], mvdf32_x1.val[ch]);
                    vqf32_result             = neon::vmul(vqf32_result, vqf32_a);
                    mvdf32_result.val[ch]    = neon::vpadd(neon::vgetlow(vqf32_result), neon::vgethigh(vqf32_result));
                }
                neon::vstore(rows1_tmp, mvdf32_result);

                rows1_tmp += 2 * C;
                alpha_ptr += 4;
            }

            for (; x < owidth; x++)
            {
                DT_S32 sx    = C * xofs[x];
                const Tp *r1 = src_row1 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];

                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    rows1_tmp[ch] = r1[ch] * a0 + r1[ch + C] * a1;
                }

                rows1_tmp += C;
                alpha_ptr += 2;
            }
        }
        else if (sy > prev_sy1)
        {
            const Tp *src_row0 = src.Ptr<Tp>(sy);
            const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

            BufType *rows0_tmp = rows0;
            BufType *rows1_tmp = rows1;

            DT_S32 x = 0;
            for (; x < end_x; x += 2)
            {
                DT_S32 sx_x0    = C * xofs[x];
                DT_S32 sx_x1    = C * xofs[x + 1];
                const Tp *r0_x0 = src_row0 + sx_x0;
                const Tp *r1_x0 = src_row1 + sx_x0;
                const Tp *r0_x1 = src_row0 + sx_x1;
                const Tp *r1_x1 = src_row1 + sx_x1;

                MVType mvdf32_x0, mvdf32_x1;
                neon::vload(r1_x0, mvdf32_x0);
                neon::vload(r1_x1, mvdf32_x1);
                float32x4_t vqf32_a = neon::vload1q(alpha_ptr);

                MVTypeF32 mvdf32_result;
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    float32x4_t vqf32_result = neon::vcombine(mvdf32_x0.val[ch], mvdf32_x1.val[ch]);
                    vqf32_result             = neon::vmul(vqf32_result, vqf32_a);
                    mvdf32_result.val[ch]    = neon::vpadd(neon::vgetlow(vqf32_result), neon::vgethigh(vqf32_result));
                }
                neon::vstore(rows1_tmp, mvdf32_result);

                neon::vload(r0_x0, mvdf32_x0);
                neon::vload(r0_x1, mvdf32_x1);
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    float32x4_t vqf32_result = neon::vcombine(mvdf32_x0.val[ch], mvdf32_x1.val[ch]);
                    vqf32_result             = neon::vmul(vqf32_result, vqf32_a);
                    mvdf32_result.val[ch]    = neon::vpadd(neon::vgetlow(vqf32_result), neon::vgethigh(vqf32_result));
                }
                neon::vstore(rows0_tmp, mvdf32_result);

                rows0_tmp += 2 * C;
                rows1_tmp += 2 * C;
                alpha_ptr += 4;
            }

            for (; x < owidth; x++)
            {
                DT_S32 sx    = C * xofs[x];
                const Tp *r0 = src_row0 + sx;
                const Tp *r1 = src_row1 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    rows0_tmp[ch] = r0[ch] * a0 + r0[ch + C] * a1;
                    rows1_tmp[ch] = r1[ch] * a0 + r1[ch + C] * a1;
                }

                rows0_tmp += C;
                rows1_tmp += C;
                alpha_ptr += 2;
            }
        }
        prev_sy1 = sy + 1;

        AlphaType b0 = beta[0];
        AlphaType b1 = beta[1];

        BufType *rows0_tmp = rows0;
        BufType *rows1_tmp = rows1;
        Tp *dst_c_ptr      = dst.Ptr<Tp>(y);

        float32x4_t vqf32_b0, vqf32_b1;
        neon::vdup(vqf32_b0, b0);
        neon::vdup(vqf32_b1, b1);

        DT_S32 owidth_xc        = owidth * C;
        DT_S32 owidth_xc_align4 = owidth_xc & (-4);
        DT_S32 x                = 0;
        for (; x < owidth_xc_align4; x += 4)
        {
            float32x4_t vqf32_x0 = neon::vload1q(rows0_tmp);
            float32x4_t vqf32_n0 = neon::vload1q(rows1_tmp);

            vqf32_x0                 = neon::vmul(vqf32_x0, vqf32_b0);
            float32x4_t vqf32_result = neon::vmla(vqf32_x0, vqf32_n0, vqf32_b1);

            neon::vstore(dst_c_ptr, vqf32_result);

            dst_c_ptr += 4;
            rows0_tmp += 4;
            rows1_tmp += 4;
        }

        for (; x < owidth_xc; x++)
        {
            *dst_c_ptr++ = (*rows0_tmp++) * b0 + (*rows1_tmp++) * b1;
        }

        beta += 2;
    }

    AURA_RETURN(ctx, ret);
}

// Tp = MI_F16
#if defined(AURA_ENABLE_NEON_FP16)
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<MI_F16, Tp>::value, Status>::type
ResizeBnCommNeonImpl(Context *ctx, const Mat &src, Mat &dst,
                     const DT_S32 *buffer, ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::OK;

    using BufType   = typename ResizeBnCuTraits<Tp>::BufType;
    using AlphaType = typename ResizeBnCuTraits<Tp>::AlphaType;
    using MovlType  = typename ResizeBnCuTraits<Tp>::MovlType;
    using MVType    = typename neon::MDVector<Tp, C>::MVType;
    using MVTypeF32 = typename neon::MDVector<BufType, C>::MVType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;

    const DT_S32 *xofs     = buffer;
    const DT_S32 *yofs     = xofs + owidth;
    const AlphaType *alpha = reinterpret_cast<const AlphaType*>(yofs + oheight);
    const AlphaType *beta  = alpha + 2 * owidth + 2 * start_row;

    BufType *rows = thread_buffer.GetThreadData<BufType>();

    if (!rows)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    BufType *rows0 = rows;
    BufType *rows1 = rows + C * owidth;

    DT_F64 scale_x         = static_cast<DT_F64>(iwidth) / owidth;
    DT_S32 loop_end_owidth = Ceil(((iwidth - 8) + 0.5) / scale_x - 0.5);
    DT_S32 end_x           = loop_end_owidth & (-2);

    DT_S32 prev_sy1 = -1;
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 sy                  = yofs[y];
        const AlphaType *alpha_ptr = alpha;

        if (sy == prev_sy1)
        {
            BufType *rows0_tmp = rows0;
            rows0              = rows1;
            rows1              = rows0_tmp;
            BufType *rows1_tmp = rows1;

            const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

            DT_S32 x = 0;
            for (; x < end_x; x += 2)
            {
                DT_S32 sx_x0    = C * xofs[x];
                DT_S32 sx_x1    = C * xofs[x + 1];
                const Tp *r1_x0 = src_row1 + sx_x0;
                const Tp *r1_x1 = src_row1 + sx_x1;

                MVType mvdf16_x0, mvdf16_x1;
                neon::vload(r1_x0, mvdf16_x0);
                neon::vload(r1_x1, mvdf16_x1);
                float32x4_t vqf32_a = neon::vload1q(alpha_ptr);

                MVTypeF32 mvdf32_result;
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    float32x4_t vqf32_result = neon::vcombine(neon::vgetlow(neon::vcvt<MovlType>(mvdf16_x0.val[ch])),
                                                              neon::vgetlow(neon::vcvt<MovlType>(mvdf16_x1.val[ch])));
                    vqf32_result             = neon::vmul(vqf32_result, vqf32_a);
                    mvdf32_result.val[ch]    = neon::vpadd(neon::vgetlow(vqf32_result), neon::vgethigh(vqf32_result));
                }
                neon::vstore(rows1_tmp, mvdf32_result);

                rows1_tmp += 2 * C;
                alpha_ptr += 4;
            }

            for (; x < owidth; x++)
            {
                DT_S32 sx    = C * xofs[x];
                const Tp *r1 = src_row1 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];

                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    rows1_tmp[ch] = r1[ch] * a0 + r1[ch + C] * a1;
                }

                rows1_tmp += C;
                alpha_ptr += 2;
            }
        }
        else if (sy > prev_sy1)
        {
            const Tp *src_row0 = src.Ptr<Tp>(sy);
            const Tp *src_row1 = src.Ptr<Tp>(sy + 1);

            BufType *rows0_tmp = rows0;
            BufType *rows1_tmp = rows1;

            DT_S32 x = 0;
            for (; x < end_x; x += 2)
            {
                DT_S32 sx_x0    = C * xofs[x];
                DT_S32 sx_x1    = C * xofs[x + 1];
                const Tp *r0_x0 = src_row0 + sx_x0;
                const Tp *r1_x0 = src_row1 + sx_x0;
                const Tp *r0_x1 = src_row0 + sx_x1;
                const Tp *r1_x1 = src_row1 + sx_x1;

                MVType mvdf16_x0, mvdf16_x1;
                neon::vload(r1_x0, mvdf16_x0);
                neon::vload(r1_x1, mvdf16_x1);
                float32x4_t vqf32_a = neon::vload1q(alpha_ptr);

                MVTypeF32 mvdf32_result;
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    float32x4_t vqf32_result = neon::vcombine(neon::vgetlow(neon::vcvt<MovlType>(mvdf16_x0.val[ch])),
                                                              neon::vgetlow(neon::vcvt<MovlType>(mvdf16_x1.val[ch])));
                    vqf32_result             = neon::vmul(vqf32_result, vqf32_a);
                    mvdf32_result.val[ch]    = neon::vpadd(neon::vgetlow(vqf32_result), neon::vgethigh(vqf32_result));
                }
                neon::vstore(rows1_tmp, mvdf32_result);

                neon::vload(r0_x0, mvdf16_x0);
                neon::vload(r0_x1, mvdf16_x1);
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    float32x4_t vqf32_result = neon::vcombine(neon::vgetlow(neon::vcvt<MovlType>(mvdf16_x0.val[ch])),
                                                              neon::vgetlow(neon::vcvt<MovlType>(mvdf16_x1.val[ch])));
                    vqf32_result             = neon::vmul(vqf32_result, vqf32_a);
                    mvdf32_result.val[ch]    = neon::vpadd(neon::vgetlow(vqf32_result), neon::vgethigh(vqf32_result));
                }
                neon::vstore(rows0_tmp, mvdf32_result);

                rows0_tmp += 2 * C;
                rows1_tmp += 2 * C;
                alpha_ptr += 4;
            }

            for (; x < owidth; x++)
            {
                DT_S32 sx    = C * xofs[x];
                const Tp *r0 = src_row0 + sx;
                const Tp *r1 = src_row1 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];
                for (DT_S32 ch = 0; ch < C; ++ch)
                {
                    rows0_tmp[ch] = r0[ch] * a0 + r0[ch + C] * a1;
                    rows1_tmp[ch] = r1[ch] * a0 + r1[ch + C] * a1;
                }

                rows0_tmp += C;
                rows1_tmp += C;
                alpha_ptr += 2;
            }
        }
        prev_sy1 = sy + 1;

        AlphaType b0 = beta[0];
        AlphaType b1 = beta[1];

        BufType *rows0_tmp = rows0;
        BufType *rows1_tmp = rows1;
        Tp *dst_c_ptr      = dst.Ptr<Tp>(y);

        float32x4_t vqf32_b0, vqf32_b1;
        neon::vdup(vqf32_b0, b0);
        neon::vdup(vqf32_b1, b1);

        DT_S32 owidth_xc        = owidth * C;
        DT_S32 owidth_xc_align8 = owidth_xc & (-8);
        DT_S32 x                = 0;
        for (; x < owidth_xc_align8; x += 8)
        {
            float32x4_t vqf32_cx0  = neon::vload1q(rows0_tmp);
            float32x4_t vqf32_n0x0 = neon::vload1q(rows1_tmp);
            float32x4_t vqf32_cx1  = neon::vload1q(rows0_tmp + 4);
            float32x4_t vqf32_n0x1 = neon::vload1q(rows1_tmp + 4);

            float32x4_t vqf32_x0 = neon::vmul(vqf32_cx0, vqf32_b0);
            float32x4_t vqf32_x1 = neon::vmul(vqf32_cx1, vqf32_b0);

            vqf32_x0 = neon::vmla(vqf32_x0, vqf32_n0x0, vqf32_b1);
            vqf32_x1 = neon::vmla(vqf32_x1, vqf32_n0x1, vqf32_b1);

            float16x4_t vdf16_x0 = neon::vcvt<Tp>(vqf32_x0);
            float16x4_t vdf16_x1 = neon::vcvt<Tp>(vqf32_x1);

            float16x8_t vqf16_result = neon::vcombine(vdf16_x0, vdf16_x1);
            neon::vstore(dst_c_ptr, vqf16_result);

            dst_c_ptr += 8;
            rows0_tmp += 8;
            rows1_tmp += 8;
        }

        for (; x < owidth_xc; x++)
        {
            *dst_c_ptr++ = SaturateCast<Tp>((*rows0_tmp++) * b0 + (*rows1_tmp++) * b1);
        }

        beta += 2;
    }

    AURA_RETURN(ctx, ret);
}
#endif

template <typename Tp>
static Status ResizeBnCommNeonHelper(Context *ctx, const Mat &src, Mat &dst, DT_BOOL is_area, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    using AlphaType = typename ResizeBnCuTraits<Tp>::AlphaType;

    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_S32 owidth  = dst.GetSizes().m_width;

    DT_S32 buffer_size = (owidth + oheight) * sizeof(DT_S32) + (owidth + oheight) * 2 * sizeof(AlphaType);
    DT_S32 *buffer     = reinterpret_cast<DT_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, buffer_size, 0));
    if (DT_NULL == buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
        return Status::ERROR;
    }

    GetBnOffset<Tp>(buffer, iwidth, owidth, iheight, oheight, is_area);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        return Status::ERROR;
    }

    switch (src.GetSizes().m_channel)
    {
        case 1:
        {
            ThreadBuffer thread_buffer(ctx, owidth * 2 * sizeof(DT_S32));

            ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeBnCommNeonImpl<Tp, 1>, ctx, std::cref(src),
                                  std::ref(dst), buffer, std::ref(thread_buffer));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommNeonImpl run failed, channel: 1");
            }
            break;
        }

        case 2:
        {
            ThreadBuffer thread_buffer(ctx, owidth * 2 * 2 * sizeof(DT_S32));
            ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeBnCommNeonImpl<Tp, 2>, ctx, std::cref(src),
                                  std::ref(dst), buffer, std::ref(thread_buffer));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommNeonImpl run failed, channel: 2");
            }
            break;
        }

        case 3:
        {
            ThreadBuffer thread_buffer(ctx, owidth * 2 * 3 * sizeof(DT_S32));

            ret = wp->ParallelFor(0, dst.GetSizes().m_height, ResizeBnCommNeonImpl<Tp, 3>, ctx, std::cref(src),
                                  std::ref(dst), buffer, std::ref(thread_buffer));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommNeonImpl run failed, channel: 3");
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

    AURA_FREE(ctx, buffer);

    AURA_RETURN(ctx, ret);
}

Status ResizeBnCommNeon(Context *ctx, const Mat &src, Mat &dst, DT_BOOL is_area, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeBnCommNeonHelper<DT_U8>(ctx, src, dst, is_area, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommNeonHelper run failed, type: DT_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeBnCommNeonHelper<DT_S8>(ctx, src, dst, is_area, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommNeonHelper run failed, type: DT_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeBnCommNeonHelper<DT_U16>(ctx, src, dst, is_area, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommNeonHelper run failed, type: DT_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeBnCommNeonHelper<DT_S16>(ctx, src, dst, is_area, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommNeonHelper run failed, type: DT_S16");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = ResizeBnCommNeonHelper<MI_F16>(ctx, src, dst, is_area, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommNeonHelper run failed, type: MI_F16");
            }
            break;
        }
#endif

        case ElemType::F32:
        {
            ret = ResizeBnCommNeonHelper<DT_F32>(ctx, src, dst, is_area, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommNeonHelper run failed, type: DT_F32");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "do not surpport elem type F64 or F16");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura