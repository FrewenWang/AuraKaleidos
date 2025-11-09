/** @brief      : resize impl for aura
 *  @file       : resize_impl.hpp
 *  @author     : xulei21@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : July. 13, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_RESIZE_RESIZE_IMPL_HPP__
#define AURA_OPS_RESIZE_RESIZE_IMPL_HPP__

#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#if defined(AURA_ENABLE_OPENCL)
#  include "aura/runtime/opencl.h"
#endif

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif

#define INTER_RESIZE_COEF_BITS              (11)
#define INTER_RESIZE_COEF_SCALE             (1 << INTER_RESIZE_COEF_BITS)
#define RESIZE_AREA_MIN_FLOAT               (1e-3)

namespace aura
{

template<typename St, typename Dt>
struct ResizeFloatPtCastFunctor
{
    Dt operator()(St val) const
    {
        return SaturateCast<Dt>(val);
    }
};

template<typename St, typename Dt, DT_S32 BITS>
struct ResizeFixedPtCastFunctor
{
    enum { SHIFT = BITS, DELTA = 1 << (BITS - 1) };

    Dt operator()(St val) const
    {
        return SaturateCast<Dt>((val + DELTA) >> SHIFT);
    }
};

template<typename St, typename Dt>
struct ResizeNoShiftFunctor
{
    St operator()(Dt val) const
    {
        return SaturateCast<St>(val * St(1));
    }
};

template<typename St, typename Dt, DT_S32 SCALE>
struct ResizeFixedPtShiftFunctor
{
    St operator()(Dt value) const
    {
        return SaturateCast<St>(value * SCALE);
    }
};

template <typename Tp>
struct ResizeAreaTraits { using FastSumType = DT_F32; };
template <> struct ResizeAreaTraits<DT_U8>  { using FastSumType = DT_S32; };
template <> struct ResizeAreaTraits<DT_S8>  { using FastSumType = DT_S32; };

template<typename Tp>
struct ResizeBnCuTraits
{
    using BufType            = DT_F32;
    using AlphaType          = DT_F32;
    using MovlType           = DT_F32;
    using ResizeShiftFunctor = ResizeNoShiftFunctor<DT_F32, DT_F32>;
    using ResizeCastFunctor  = ResizeFloatPtCastFunctor<DT_F32, Tp>;
};
template <> struct ResizeBnCuTraits<DT_U8>
{
    using BufType            = DT_S32;
    using AlphaType          = DT_S16;
    using MovlType           = DT_U16;
    using ResizeShiftFunctor = ResizeFixedPtShiftFunctor<DT_S16, DT_F32, INTER_RESIZE_COEF_SCALE>;
    using ResizeCastFunctor  = ResizeFixedPtCastFunctor<DT_S32, DT_U8, INTER_RESIZE_COEF_BITS * 2>;
};
template <> struct ResizeBnCuTraits<DT_S8>
{
    using BufType            = DT_S32;
    using AlphaType          = DT_S16;
    using MovlType           = DT_S16;
    using ResizeShiftFunctor = ResizeFixedPtShiftFunctor<DT_S16, DT_F32, INTER_RESIZE_COEF_SCALE>;
    using ResizeCastFunctor  = ResizeFixedPtCastFunctor<DT_S32, DT_S8, INTER_RESIZE_COEF_BITS * 2>;
};
template <> struct ResizeBnCuTraits<DT_U16>
{
    using BufType            = DT_F32;
    using AlphaType          = DT_F32;
    using MovlType           = DT_U32;
    using ResizeShiftFunctor = ResizeNoShiftFunctor<DT_F32, DT_F32>;
    using ResizeCastFunctor  = ResizeFloatPtCastFunctor<DT_F32, DT_U16>;
};

template <> struct ResizeBnCuTraits<DT_S16>
{
    using BufType            = DT_F32;
    using AlphaType          = DT_F32;
    using MovlType           = DT_S32;
    using ResizeShiftFunctor = ResizeNoShiftFunctor<DT_F32, DT_F32>;
    using ResizeCastFunctor  = ResizeFloatPtCastFunctor<DT_F32, DT_S16>;
};

struct AreaDecimateAlpha
{
    DT_S32 si;
    DT_S32 di;
    DT_F32 alpha;
};

struct ResizeCuFastVtcmBuffer
{
    DT_U8 *row0_ptr;
    DT_U8 *row1_ptr;
    DT_U8 *row2_ptr;
    DT_U8 *row3_ptr;
    DT_U8 *row_head;
};

template <typename Tp>
DT_VOID GetBnOffset(DT_S32 *buffer, DT_S32 iwidth, DT_S32 owidth, DT_S32 iheight, DT_S32 oheight, DT_BOOL is_area)
{
    using AlphaType = typename ResizeBnCuTraits<Tp>::AlphaType;
    auto  ResizeShiftFunctor = typename ResizeBnCuTraits<Tp>::ResizeShiftFunctor();

    DT_F32 scale_x = static_cast<DT_F32>(iwidth) / owidth;
    DT_F32 scale_y = static_cast<DT_F32>(iheight) / oheight;

    DT_S32 *xofs = buffer;
    DT_S32 *yofs = xofs + owidth;
    AlphaType *alpha = reinterpret_cast<AlphaType*>(yofs + oheight);
    AlphaType *beta  = alpha + 2 * owidth;

    for (DT_S32 x = 0; x < owidth; x++)
    {
        DT_F32 fx;
        DT_S32 sx;
        if (!is_area)
        {
            fx = static_cast<DT_F32>((x + 0.5) * scale_x - 0.5);
            sx = static_cast<DT_S32>(Floor(fx));
            fx -= sx;
        }
        else
        {
            sx = static_cast<DT_S32>(Floor(x * scale_x));
            fx = static_cast<DT_F32>((x + 1) - (sx + 1) / scale_x);
            fx = fx <= 0 ? 0.f : fx - Floor(fx);
        }

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx > iwidth - 2)
        {
            sx = iwidth - 2;
            fx = 1.f;
        }

        xofs[x] = sx;
        alpha[x * 2    ] = ResizeShiftFunctor(1.f - fx);
        alpha[x * 2 + 1] = ResizeShiftFunctor(fx);
    }

    for (DT_S32 y = 0; y < oheight; y++)
    {
        DT_F32 fy;
        DT_S32 sy;
        if (!is_area)
        {
            fy = static_cast<DT_F32>((y + 0.5) * scale_y - 0.5);
            sy = static_cast<DT_S32>(Floor(fy));
            fy -= sy;
        }
        else
        {
            sy = static_cast<DT_S32>(Floor(y * scale_y));
            fy = static_cast<DT_F32>((y + 1) - (sy + 1) / scale_y);
            fy = fy <= 0 ? 0.f : fy - Floor(fy);
        }

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy > iheight - 2)
        {
            sy = iheight - 2;
            fy = 1.f;
        }

        yofs[y] = sy;
        beta[y * 2    ] = ResizeShiftFunctor(1.f - fy);
        beta[y * 2 + 1] = ResizeShiftFunctor(fy);
    }
}

AURA_INLINE DT_F32 GetCuOffsetCore(DT_F32 x)
{
    const DT_F32 bicubic_a = -0.75f;
    const DT_F32 bicubic_lut[5] =
    {
        (bicubic_a + 3.f), (bicubic_a + 2.f), ((-4.f) * bicubic_a),
        (bicubic_a * 8.f), (bicubic_a * 5.f)
    };

    const DT_F32 xx       = x * x;
    const DT_F32 xxx      = xx * x;
    const DT_F32 *lut_ptr = bicubic_lut;

    return (x) <= (1.f) ? (1.0f - (lut_ptr[0] * xx) + (lut_ptr[1] * xxx))
                        : (lut_ptr[2] + (lut_ptr[3] * x) - (lut_ptr[4] * xx) + (bicubic_a * xxx));
}

template <typename Tp>
DT_VOID GetCuOffset(DT_S32 *buffer, DT_S32 iwidth, DT_S32 owidth, DT_S32 iheight, DT_S32 oheight)
{
    using AlphaType         = typename ResizeBnCuTraits<Tp>::AlphaType;
    auto ResizeShiftFunctor = typename ResizeBnCuTraits<Tp>::ResizeShiftFunctor();

    DT_F64 scale_x = static_cast<DT_F64>(iwidth) / owidth;
    DT_F64 scale_y = static_cast<DT_F64>(iheight) / oheight;

    DT_S32 *xofs     = buffer;
    DT_S32 *yofs     = xofs + owidth;
    AlphaType *alpha = reinterpret_cast<AlphaType*>(yofs + oheight);
    AlphaType *beta  = reinterpret_cast<AlphaType*>(alpha + (owidth * 4));

    DT_F32    fx[4], fy[4];
    DT_S32    sx, sy;
    AlphaType coe_x[4], coe_y[4];

    for (DT_S32 dx = 0; dx < owidth; dx++)
    {
        fx[0] = static_cast<DT_F32>(((dx + 0.5) * scale_x - 0.5));
        sx    = static_cast<DT_S32>(Floor(fx[0])) - 1;

        fx[0] -= sx;
        fx[1] = fx[0] - 1.0f;
        fx[2] = 2.0f - fx[0];
        fx[3] = 3.0f - fx[0];

        fx[0] = GetCuOffsetCore(fx[0]);
        fx[1] = GetCuOffsetCore(fx[1]);
        fx[2] = GetCuOffsetCore(fx[2]);

        coe_x[0] = ResizeShiftFunctor(fx[0]);
        coe_x[1] = ResizeShiftFunctor(fx[1]);
        coe_x[2] = ResizeShiftFunctor(fx[2]);
        coe_x[3] = ResizeShiftFunctor((1.0f - fx[0] - fx[1] - fx[2]));

        if (sx >= 0 && sx <= (iwidth - 4))
        {
            xofs[dx]             = sx;
            alpha[dx << 2]       = coe_x[0];
            alpha[(dx << 2) + 1] = coe_x[1];
            alpha[(dx << 2) + 2] = coe_x[2];
            alpha[(dx << 2) + 3] = coe_x[3];
        }
        else if ((-2) == sx)
        {
            xofs[dx]             = 0;
            alpha[dx << 2]       = coe_x[0] + coe_x[1] + coe_x[2];
            alpha[(dx << 2) + 1] = coe_x[3];
            alpha[(dx << 2) + 2] = 0;
            alpha[(dx << 2) + 3] = 0;
        }
        else if ((-1) == sx)
        {
            xofs[dx]             = 0;
            alpha[dx << 2]       = coe_x[0] + coe_x[1];
            alpha[(dx << 2) + 1] = coe_x[2];
            alpha[(dx << 2) + 2] = coe_x[3];
            alpha[(dx << 2) + 3] = 0;
        }
        else if ((iwidth - 3) == sx)
        {
            xofs[dx]             = iwidth - 4;
            alpha[dx << 2]       = 0;
            alpha[(dx << 2) + 1] = coe_x[0];
            alpha[(dx << 2) + 2] = coe_x[1];
            alpha[(dx << 2) + 3] = coe_x[2] + coe_x[3];
        }
        else if ((iwidth - 2) == sx)
        {
            xofs[dx]             = iwidth - 4;
            alpha[dx << 2]       = 0;
            alpha[(dx << 2) + 1] = 0;
            alpha[(dx << 2) + 2] = coe_x[0];
            alpha[(dx << 2) + 3] = coe_x[1] + coe_x[2] + coe_x[3];
        }
    }

    for (DT_S32 dy = 0; dy < oheight; dy++)
    {
        fy[0] = static_cast<DT_F32>(((dy + 0.5) * scale_y - 0.5));
        sy    = static_cast<DT_S32>(Floor(fy[0])) - 1;

        fy[0] -= sy;
        fy[1] = fy[0] - 1.0f;
        fy[2] = 2.0f - fy[0];
        fy[3] = 3.0f - fy[0];

        fy[0] = GetCuOffsetCore(fy[0]);
        fy[1] = GetCuOffsetCore(fy[1]);
        fy[2] = GetCuOffsetCore(fy[2]);

        coe_y[0] = ResizeShiftFunctor(fy[0]);
        coe_y[1] = ResizeShiftFunctor(fy[1]);
        coe_y[2] = ResizeShiftFunctor(fy[2]);
        coe_y[3] = ResizeShiftFunctor((1.0f - fy[0] - fy[1] - fy[2]));

        if (sy >= 0 && sy <= (iheight - 4))
        {
            yofs[dy]            = sy;
            beta[dy << 2]       = coe_y[0];
            beta[(dy << 2) + 1] = coe_y[1];
            beta[(dy << 2) + 2] = coe_y[2];
            beta[(dy << 2) + 3] = coe_y[3];
        }
        else if ((-2) == sy)
        {
            yofs[dy]            = 0;
            beta[dy << 2]       = coe_y[0] + coe_y[1] + coe_y[2];
            beta[(dy << 2) + 1] = coe_y[3];
            beta[(dy << 2) + 2] = 0;
            beta[(dy << 2) + 3] = 0;
        }
        else if ((-1) == sy)
        {
            yofs[dy]            = 0;
            beta[dy << 2]       = coe_y[0] + coe_y[1];
            beta[(dy << 2) + 1] = coe_y[2];
            beta[(dy << 2) + 2] = coe_y[3];
            beta[(dy << 2) + 3] = 0;
        }
        else if ((iheight - 3) == sy)
        {
            yofs[dy]            = iheight - 4;
            beta[dy << 2]       = 0;
            beta[(dy << 2) + 1] = coe_y[0];
            beta[(dy << 2) + 2] = coe_y[1];
            beta[(dy << 2) + 3] = coe_y[2] + coe_y[3];
        }
        else if ((iheight - 2) == sy)
        {
            yofs[dy]            = iheight - 4;
            beta[dy << 2]       = 0;
            beta[(dy << 2) + 1] = 0;
            beta[(dy << 2) + 2] = coe_y[0];
            beta[(dy << 2) + 3] = coe_y[1] + coe_y[2] + coe_y[3];
        }
    }
}

AURA_INLINE DT_S32 GetAreaOffset(DT_S32 isize, DT_S32 osize, DT_S32 channel, DT_F32 scale, AreaDecimateAlpha *tab)
{
    DT_S32 k = 0; /**< table size */

    for (DT_S32 dx = 0; dx < osize; dx++)
    {
        DT_F32 fsx1 = dx * scale;
        DT_F32 fsx2 = fsx1 + scale;
        DT_F32 cell_width = Min(scale, isize - fsx1);

        DT_S32 sx1 = Ceil(fsx1), sx2 = Floor(fsx2);

        sx2 = Min(sx2, isize - 1);
        sx1 = Min(sx1, sx2);

        if (sx1 - fsx1 > RESIZE_AREA_MIN_FLOAT)
        {
            tab[k].di = dx * channel;
            tab[k].si = (sx1 - 1) * channel;
            tab[k++].alpha = (sx1 - fsx1) / cell_width;
        }

        for (DT_S32 sx = sx1; sx < sx2; sx++)
        {
            tab[k].di = dx * channel;
            tab[k].si = sx * channel;
            tab[k++].alpha = 1.0f / cell_width;
        }

        if (fsx2 - sx2 > RESIZE_AREA_MIN_FLOAT)
        {
            tab[k].di = dx * channel;
            tab[k].si = sx2 * channel;
            tab[k++].alpha = Min(Min(fsx2 - sx2, 1.0f), cell_width) / cell_width;
        }
    }

    return k;
}

template <typename Tp>
Status ResizeBnNoneImpl(Context *ctx, const Mat &src, Mat &dst, DT_BOOL is_area = DT_FALSE)
{
    using BufType   = typename ResizeBnCuTraits<Tp>::BufType;
    using AlphaType = typename ResizeBnCuTraits<Tp>::AlphaType;
    auto  ResizeCastFunctor = typename ResizeBnCuTraits<Tp>::ResizeCastFunctor();

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_S32 channel = dst.GetSizes().m_channel;

    DT_S32 buffer_size = (owidth + oheight) * sizeof(DT_S32) + (owidth + oheight) * 2 * sizeof(AlphaType);
    DT_S32 *buffer = static_cast<DT_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, buffer_size, 0));
    if (DT_NULL == buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return Status::ERROR;
    }

    GetBnOffset<Tp>(buffer, iwidth, owidth, iheight, oheight, is_area);

    DT_S32 *xofs  = buffer;
    DT_S32 *yofs  = xofs + owidth;
    AlphaType *alpha = reinterpret_cast<AlphaType*>(yofs + oheight);
    AlphaType *beta  = reinterpret_cast<AlphaType*>(alpha + (owidth * 2));

    DT_S32 rows_size = owidth * channel * 2 * sizeof(BufType);
    BufType *rows = static_cast<BufType*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, rows_size, 0));
    if (DT_NULL == rows)
    {
        AURA_FREE(ctx, buffer);
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return Status::ERROR;
    }

    BufType *rows0 = rows;
    BufType *rows1 = rows + owidth * channel;

    DT_S32 prev_sy1 = -1;

    for (DT_S32 dy = 0; dy < oheight; dy++ )
    {
        DT_S32 sy = yofs[dy];

        if (sy == prev_sy1)
        {
            BufType *rows0_tmp = rows0;
            rows0 = rows1;
            rows1 = rows0_tmp;

            const Tp *src_c1 = src.Ptr<Tp>(sy + 1);
            const AlphaType *alpha_ptr = alpha;

            for (DT_S32 dx = 0; dx < owidth; dx++ )
            {
                DT_S32 sx = xofs[dx] * channel;
                DT_S32 x_id = dx * channel;
                const Tp *src_c1_ptr = src_c1 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];

                for (DT_S32 ch = 0; ch < channel; ch++)
                {
                    rows1[x_id + ch] = src_c1_ptr[ch] * a0 + src_c1_ptr[ch + channel] * a1;
                }
                alpha_ptr += 2;
            }
        }
        else if (sy > prev_sy1)
        {
            // hresize two rows
            const Tp *src_c0 = src.Ptr<Tp>(sy);
            const Tp *src_c1 = src.Ptr<Tp>(sy + 1);
            const AlphaType *alpha_ptr = alpha;

            for (DT_S32 dx = 0; dx < owidth; dx++ )
            {
                DT_S32 sx = xofs[dx] * channel;
                DT_S32 x_id = dx * channel;
                const Tp *src_c0_ptr = src_c0 + sx;
                const Tp *src_c1_ptr = src_c1 + sx;

                AlphaType a0 = alpha_ptr[0];
                AlphaType a1 = alpha_ptr[1];

                for (DT_S32 ch = 0; ch < channel; ch++)
                {
                    rows0[x_id + ch] = src_c0_ptr[ch] * a0 + src_c0_ptr[ch + channel] * a1;
                    rows1[x_id + ch] = src_c1_ptr[ch] * a0 + src_c1_ptr[ch + channel] * a1;
                }
                alpha_ptr += 2;
            }
        }

        prev_sy1 = sy + 1;

        AlphaType b0 = beta[0];
        AlphaType b1 = beta[1];
        Tp *dst_c_ptr = dst.Ptr<Tp>(dy);

        for (DT_S32 dx = 0; dx < owidth * channel; dx++)
        {
            dst_c_ptr[dx] = ResizeCastFunctor(static_cast<BufType>((rows0[dx] * b0 + rows1[dx] * b1)));
        }
        beta += 2;
    }

    AURA_FREE(ctx, buffer);
    AURA_FREE(ctx, rows);

    return Status::OK;
}

template <typename Tp>
Status ResizeAreaCommNoneImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, AreaDecimateAlpha *x_tab, DT_S32 x_tab_size,
                              AreaDecimateAlpha *y_tab, DT_S32 *tab_offset, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 channel     = src.GetSizes().m_channel;
    DT_S32 owidth      = dst.GetSizes().m_width;
    DT_S32 owidth_x_cn = owidth * channel;

    DT_F32 *buffer = thread_buffer.GetThreadData<DT_F32>();

    if (!buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    DT_F32 *sum = buffer + owidth_x_cn;
    DT_S32 start = tab_offset[start_row];
    DT_S32 end   = tab_offset[end_row];
    DT_S32 prev_dy = y_tab[start].di;

    for (DT_S32 dx = 0; dx < owidth_x_cn; dx++)
    {
        sum[dx] = static_cast<DT_F32>(0);
    }

    for (DT_S32 j = start; j < end; j++)
    {
        DT_F32 beta = y_tab[j].alpha;
        DT_S32 dy   = y_tab[j].di;
        DT_S32 sy   = y_tab[j].si;

        const Tp *src_c = src.Ptr<Tp>(sy);

        for (DT_S32 dx = 0; dx < owidth_x_cn; dx++)
        {
            buffer[dx] = static_cast<DT_F32>(0);
        }
        if (1 == channel)
        {
            for (DT_S32 k = 0; k < x_tab_size; k++)
            {
                DT_S32 dxn   = x_tab[k].di;
                DT_F32 alpha = x_tab[k].alpha;
                buffer[dxn] += src_c[x_tab[k].si] * alpha;
            }
        }
        else if (2 == channel)
        {
            for (DT_S32 k = 0; k < x_tab_size; k++)
            {
                DT_S32 sxn   = x_tab[k].si;
                DT_S32 dxn   = x_tab[k].di;
                DT_F32 alpha = x_tab[k].alpha;
                buffer[dxn]     += src_c[sxn]     * alpha;
                buffer[dxn + 1] += src_c[sxn + 1] * alpha;
            }
        }
        else if (3 == channel)
        {
            for (DT_S32 k = 0; k < x_tab_size; k++)
            {
                DT_S32 sxn   = x_tab[k].si;
                DT_S32 dxn   = x_tab[k].di;
                DT_F32 alpha = x_tab[k].alpha;
                buffer[dxn]     += src_c[sxn]     * alpha;
                buffer[dxn + 1] += src_c[sxn + 1] * alpha;
                buffer[dxn + 2] += src_c[sxn + 2] * alpha;
            }
        }
        else
        {
            for (DT_S32 k = 0; k < x_tab_size; k++)
            {
                DT_S32 sxn   = x_tab[k].si;
                DT_S32 dxn   = x_tab[k].di;
                DT_F32 alpha = x_tab[k].alpha;
                for (DT_S32 ch = 0; ch < channel; ch++)
                {
                    buffer[dxn + ch] += src_c[sxn + ch] * alpha;
                }
            }
        }

        if (dy != prev_dy)
        {
            Tp *dst_c = dst.Ptr<Tp>(prev_dy);

            for (DT_S32 dx = 0; dx < owidth_x_cn; dx++)
            {
                dst_c[dx] = SaturateCast<Tp>(sum[dx]);
                sum[dx] = beta * buffer[dx];
            }
            prev_dy = dy;
        }
        else
        {
            for (DT_S32 dx = 0; dx < owidth_x_cn; dx++)
            {
                sum[dx] += beta * buffer[dx];
            }
        }
    }

    Tp *dst_c = dst.Ptr<Tp>(prev_dy);

    for (DT_S32 dx = 0; dx < owidth_x_cn; dx++)
    {
        dst_c[dx] = SaturateCast<Tp>(sum[dx]);
    }

    return Status::OK;
}

class ResizeImpl : public OpImpl
{
public:
    ResizeImpl(Context *ctx, const OpTarget &target);

    virtual Status SetArgs(const Array *src, Array *dst, InterpType type);

    std::vector<const Array*> GetInputArrays() const override;

    std::vector<const Array*> GetOutputArrays() const override;

    std::string ToString() const override;

    DT_VOID Dump(const std::string &prefix) const override;

protected:
    const Array *m_src;
    Array       *m_dst;
    InterpType   m_type;
};

Status ResizeNnNone(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeCuNone(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeBnNone(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeAreaNone(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);

class ResizeNone : public ResizeImpl
{
public:
    ResizeNone(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, InterpType type) override;

    Status Run() override;
};

#if defined(AURA_ENABLE_NEON)
class ResizeNeon : public ResizeImpl
{
public:
    ResizeNeon(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, InterpType type) override;

    Status Run() override;
};

Status ResizeAreaFastNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeAreaCommNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeAreaNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);

Status ResizeBnFastC1Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeBnFastC2Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeBnFastC3Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeBnCommNeon(Context *ctx, const Mat &src, Mat &dst, DT_BOOL is_area, const OpTarget &target);
Status ResizeBnNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);

Status ResizeNnNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);

Status ResizeCuFastC1Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeCuFastC2Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeCuFastC3Neon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeCuCommNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
Status ResizeCuNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target);
#endif  // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
class ResizeCL : public ResizeImpl
{
public:
    ResizeCL(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, InterpType type) override;
    Status Initialize() override;
    Status DeInitialize() override;
    std::string ToString() const override;

protected:
    CLMem m_cl_src;
    CLMem m_cl_dst;
    std::string m_profiling_string;
};

class ResizeNnCL : public ResizeCL
{
public:
    ResizeNnCL(Context *ctx, const OpTarget &target);

    Status Initialize() override;
    Status Run() override;
    Status DeInitialize() override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel);

private:
    std::vector<CLKernel> m_cl_kernels;
};

class ResizeBnCL : public ResizeCL
{
public:
    ResizeBnCL(Context *ctx, const OpTarget &target);

    Status Initialize() override;
    Status Run() override;
    Status DeInitialize() override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel);

private:
    std::vector<CLKernel> m_cl_kernels;
};

class ResizeCuCL : public ResizeCL
{
public:
    ResizeCuCL(Context *ctx, const OpTarget &target);

    Status Initialize() override;
    Status Run() override;
    Status DeInitialize() override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 iwidth, DT_S32 iheight,
                                              DT_S32 owidth, DT_S32 oheight, DT_S32 channel);

private:
    DT_S32 m_border;
    std::vector<CLKernel> m_cl_kernels;
};

class ResizeAreaCL : public ResizeCL
{
public:
    ResizeAreaCL(Context *ctx, const OpTarget &target);

    Status Initialize() override;
    Status Run() override;
    Status DeInitialize() override;

    static std::vector<CLKernel> GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 iwidth, DT_S32 iheight,
                                              DT_S32 owidth, DT_S32 oheight, DT_S32 channel);

private:
    std::vector<CLKernel> m_cl_kernels;
    DT_S32 m_elem_x;
    DT_S32 m_elem_y;
};
#endif  // AURA_ENABLE_OPENCL

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

template <typename Tp> struct ResizeAreaHvxTraits;
template <> struct ResizeAreaHvxTraits<DT_U8>  { using PromoteType = DT_U16; };
template <> struct ResizeAreaHvxTraits<DT_S8>  { using PromoteType = DT_U16; };
template <> struct ResizeAreaHvxTraits<DT_U16> { using PromoteType = DT_U32; };
template <> struct ResizeAreaHvxTraits<DT_S16> { using PromoteType = DT_U32; };

class ResizeHvx : public ResizeImpl
{
public:
    ResizeHvx(Context *ctx, const OpTarget &target);

    Status SetArgs(const Array *src, Array *dst, InterpType type) override;

    Status Run() override;

    std::string ToString() const override;

private:
    std::string m_profiling_string;
};

#  if defined(AURA_BUILD_HEXAGON)
Status ResizeNnFastHvx(Context *ctx, const Mat &src, Mat &dst);
Status ResizeNnCommHvx(Context *ctx, const Mat &src, Mat &dst);
Status ResizeNnHvx(Context *ctx, const Mat &src, Mat &dst);

Status ResizeBnFastDnHvx(Context *ctx, const Mat &src, Mat &dst);
Status ResizeBnFastUpHvx(Context *ctx, const Mat &src, Mat &dst);
Status ResizeBnCommHvx(Context *ctx, const Mat &src, Mat &dst, DT_BOOL is_area);
Status ResizeBnHvx(Context *ctx, const Mat &src, Mat &dst);

Status ResizeCuCommHvx(Context *ctx, const Mat &src, Mat &dst);
Status ResizeCuFastUpHvx(Context *ctx, const Mat &src, Mat &dst);
Status ResizeCuFastDnHvx(Context *ctx, const Mat &src, Mat &dst);
Status ResizeCuHvx(Context *ctx, const Mat &src, Mat &dst);

Status ResizeAreaFastHvx(Context *ctx, const Mat &src, Mat &dst);
Status ResizeAreaCommonHvx(Context *ctx, const Mat &src, Mat &dst);
Status ResizeAreaHvx(Context *ctx, const Mat &src, Mat &dst);
#  endif// AURA_BUILD_HEXAGON

using ResizeInParam = HexagonRpcParamType<Mat, Mat, InterpType>;
#  define AURA_OPS_RESIZE_RESIZE_OP_NAME          "Resize"
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

} // namespace aura

#endif // AURA_OPS_RESIZE_RESIZE_IMPL_HPP__
