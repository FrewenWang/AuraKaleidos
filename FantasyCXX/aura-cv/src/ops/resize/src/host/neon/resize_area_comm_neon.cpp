#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetAreaOffsetAlign8X(Context *ctx, DT_S32 src_size, DT_S32 dst_size, DT_F32 scale, DT_S32 *x_idx, DT_F32 *x_coef)
{
    if (DT_NULL == x_idx || DT_NULL == x_coef)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    DT_S32 k = 0;
    for (DT_S32 dx = 0; dx < dst_size; dx++)
    {
        DT_S32 end_id = k + 8; // one pixel in x direction map into 8 pixels

        const DT_F32 fsx1 = dx * scale;
        const DT_F32 fsx2 = fsx1 + scale;
        const DT_F32 cell_width = Min(scale, src_size - fsx1);
        const DT_F32 inv_cell_width = 1.0 / cell_width;

        DT_S32 sx1 = Ceil(fsx1);
        DT_S32 sx2 = Floor(fsx2);

        sx2 = Min(sx2, src_size - 1);
        sx1 = Min(sx1, sx2);

        if (sx1 - fsx1 > RESIZE_AREA_MIN_FLOAT)
        {
            x_idx[k] = sx1 - 1;
            x_coef[k++] = (sx1 - fsx1) * inv_cell_width;
        }

        for (DT_S32 sx = sx1; sx < sx2; sx++)
        {
            x_idx[k] = sx;
            x_coef[k++] = inv_cell_width;
        }

        if (fsx2 - sx2 > RESIZE_AREA_MIN_FLOAT)
        {
            x_idx[k] = sx2;
            DT_F32 mod = fsx2 - sx2;
            mod = Min(mod, 1.0f);
            mod = Min(mod, cell_width);
            x_coef[k++] = mod * inv_cell_width;
        }

        // pad to 8 pixels
        for (; k < end_id; k++)
        {
            x_idx[k] = sx2;
            x_coef[k] = 0.f;
        }
    }

    return Status::OK;
}

static Status GetAreaOffsetAlign8Y(Context *ctx, DT_S32 src_size, DT_S32 dst_size, DT_F32 scale, DT_S32 *y_idx, DT_F32 *y_coef, DT_S32 *y_tab)
{
    if (DT_NULL == y_idx || DT_NULL == y_coef || DT_NULL == y_tab)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    DT_S32 k = 0;
    DT_U16 tab_idx = 0;
    for (DT_S32 dx = 0; dx < dst_size; dx++)
    {
        const DT_F32 fsx1 = dx * scale;
        const DT_F32 fsx2 = fsx1 + scale;
        const DT_F32 cell_width = Min(scale, src_size - fsx1);
        const DT_F32 inv_cell_width = 1.0f / cell_width;

        DT_S32 sx1 = Ceil(fsx1);
        DT_S32 sx2 = Floor(fsx2);

        sx2 = Min(sx2, src_size - 1);
        sx1 = Min(sx1, sx2);

        // record the start y_idx id
        y_tab[tab_idx] = k;
        tab_idx++;

        if (sx1 - fsx1 > RESIZE_AREA_MIN_FLOAT)
        {
            y_idx[k] = sx1 - 1;
            y_coef[k++] = (sx1 - fsx1) * inv_cell_width;
        }

        for (DT_S32 sx = sx1; sx < sx2; sx++)
        {
            y_idx[k] = sx;
            y_coef[k++] = inv_cell_width;
        }

        if (fsx2 - sx2 > RESIZE_AREA_MIN_FLOAT)
        {
            y_idx[k] = sx2;
            DT_F32 mod = (fsx2 - sx2);
            mod = Min(mod, 1.0f);
            mod = Min(mod, cell_width);
            y_coef[k++] = mod * inv_cell_width;
        }

        y_tab[tab_idx] = k;
        tab_idx++;
    }

    return Status::OK;
}

// SType = DT_U8, DT_S8
// VType = uint8x8_t, int8x8_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U8, SType>::value || std::is_same<DT_S8, SType>::value, DT_VOID>::type
ResizeAreaCommNeonCore0(DT_F32 *buffer, DT_S32 idx, float32x4_t &vqf32_alpha_l, float32x4_t &vqf32_alpha_h, VType &vd8_src,
                        float32x4_t &vqf32_beta)
{
    auto vq16_src = neon::vmovl(vd8_src);
    float32x4_t vqf32_sum_result = neon::vload1q(buffer + idx);

    float32x4_t vqf32_src_l = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src)));
    float32x4_t vqf32_src_h = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src)));

    float32x4_t vqf32_low_part  = neon::vmul(vqf32_alpha_l, vqf32_src_l);
    float32x4_t vqf32_high_part = neon::vmul(vqf32_alpha_h, vqf32_src_h);

    float32x4_t vqf32_result = neon::vadd(vqf32_low_part, vqf32_high_part);
    vqf32_sum_result = neon::vmla(vqf32_sum_result, vqf32_result, vqf32_beta);

    neon::vstore(buffer + idx, vqf32_sum_result);
}

// SType = DT_U16, DT_S16
// VType = uint16x8_t, int16x8_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_U16, SType>::value || std::is_same<DT_S16, SType>::value, DT_VOID>::type
ResizeAreaCommNeonCore0(DT_F32 *buffer, DT_S32 idx, float32x4_t &vqf32_alpha_l, float32x4_t &vqf32_alpha_h,
                        VType &vq16_src, float32x4_t &vqf32_beta)
{
    float32x4_t vqf32_sum_result = neon::vload1q(buffer + idx);

    float32x4_t vqf32_src_l = neon::vcvt<DT_F32>(neon::vmovl(neon::vgetlow(vq16_src)));
    float32x4_t vqf32_src_h = neon::vcvt<DT_F32>(neon::vmovl(neon::vgethigh(vq16_src)));

    float32x4_t vqf32_low_part  = neon::vmul(vqf32_alpha_l, vqf32_src_l);
    float32x4_t vqf32_high_part = neon::vmul(vqf32_alpha_h, vqf32_src_h);

    float32x4_t vqf32_result = neon::vadd(vqf32_low_part, vqf32_high_part);
    vqf32_sum_result = neon::vmla(vqf32_sum_result, vqf32_result, vqf32_beta);

    neon::vstore(buffer + idx, vqf32_sum_result);
}

// SType = DT_F32
// VType = float32x4_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<DT_F32, SType>::value, DT_VOID>::type
ResizeAreaCommNeonCore0(DT_F32 *buffer, DT_S32 idx, float32x4_t &vqf32_alpha_l, float32x4_t &vqf32_alpha_h, VType &vqf32_x0_src,
                        VType &vqf32_x1_src, float32x4_t &vqf32_beta)
{
    VType vqf32_sum_result = neon::vload1q(buffer + idx);

    VType vqf32_low_part  = neon::vmul(vqf32_alpha_l, vqf32_x0_src);
    VType vqf32_high_part = neon::vmul(vqf32_alpha_h, vqf32_x1_src);

    VType vqf32_result = neon::vadd(vqf32_low_part, vqf32_high_part);
    vqf32_sum_result = neon::vmla(vqf32_sum_result, vqf32_result, vqf32_beta);

    neon::vstore(buffer + idx, vqf32_sum_result);
}

#if defined(AURA_ENABLE_NEON_FP16)
// SType = MI_F16
// VType = float16x8_t
template <typename SType, typename VType>
AURA_ALWAYS_INLINE typename std::enable_if<std::is_same<MI_F16, SType>::value, DT_VOID>::type
ResizeAreaCommNeonCore0(DT_F32 *buffer, DT_S32 idx, float32x4_t &vqf32_alpha_l, float32x4_t &vqf32_alpha_h,
                        VType &vqf16_src, float32x4_t &vqf32_beta)
{
    float32x4_t vqf32_src_l = neon::vcvt<DT_F32>(neon::vgetlow(vqf16_src));
    float32x4_t vqf32_src_h = neon::vcvt<DT_F32>(neon::vgethigh(vqf16_src));

    float32x4_t vqf32_sum_result = neon::vload1q(buffer + idx);

    float32x4_t vqf32_low_part  = neon::vmul(vqf32_alpha_l, vqf32_src_l);
    float32x4_t vqf32_high_part = neon::vmul(vqf32_alpha_h, vqf32_src_h);

    float32x4_t vqf32_result = neon::vadd(vqf32_low_part, vqf32_high_part);
    vqf32_sum_result = neon::vmla(vqf32_sum_result, vqf32_result, vqf32_beta);

    neon::vstore(buffer + idx, vqf32_sum_result);
}
#endif

// Tp = DT_U8, DT_S8, DT_U16, DT_S16, DT_F32, MI_F16
template <typename Tp>
AURA_ALWAYS_INLINE DT_VOID ResizeAreaCommNeonCore1(DT_F32 *buffer, DT_S32 idx, Tp *dst_row)
{
    float32x4_t vqf32_sum_data = neon::vload1q(buffer + idx);

    float32x2_t vdf32_temp_data = neon::vadd(neon::vgetlow(vqf32_sum_data), neon::vgethigh(vqf32_sum_data));
    vdf32_temp_data = neon::vpadd(vdf32_temp_data, vdf32_temp_data);

    DT_F32 result = neon::vgetlane<0>(vdf32_temp_data);
    dst_row[0] = SaturateCast<Tp>(result);
}

// Tp = DT_U8, DT_S8, DT_U16, DT_S16, MI_F16
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value || std::is_same<DT_U16, Tp>::value ||
std::is_same<DT_S16, Tp>::value || std::is_same<MI_F16, Tp>::value, Status>::type
ResizeAreaCommNeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 *x_idx, DT_F32 *x_coef,
                       DT_S32 *y_idx, DT_F32 *y_coef, DT_S32 *y_table, DT_S32 start_row, DT_S32 end_row)
{
    if (DT_NULL == x_idx || DT_NULL == x_coef || DT_NULL == y_idx || DT_NULL == y_coef || DT_NULL == y_table)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    using VType  = typename std::conditional<1 == sizeof(Tp), typename neon::DVector<Tp>::VType,
                                             typename neon::QVector<Tp>::VType>::type;
    using MVType = typename std::conditional<1 == sizeof(Tp), typename neon::MDVector<Tp, C>::MVType,
                                             typename neon::MQVector<Tp, C>::MVType>::type;

    DT_S32 iwidth = src.GetSizes().m_width;
    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 owidth = dst.GetSizes().m_width;

    DT_F32 *buffer_data = thread_buffer.GetThreadData<DT_F32>();

    if (!buffer_data)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    DT_S32 max_src_height = iheight - 1;
    DT_S32 max_src_width = iwidth - 1;
    DT_S32 stop_x = iwidth - 8;

    float32x4_t vqf32_zeros = neon::vmovq(0.f);
    DT_S32 start_row_x2 = start_row << 1;
    DT_S32 end_row_x2   = end_row << 1;

    for (DT_S32 y = start_row_x2; y < end_row_x2; y += 2)
    {
        DT_S32 start_y_idx = y_table[y];
        DT_S32 end_y_idx = y_table[y + 1];

        // clear buffer
        for (DT_S32 x = 0; x < owidth * C; x++)
        {
            neon::vstore(buffer_data + (x << 2), vqf32_zeros);
        }

        for (DT_S32 j = start_y_idx; j < end_y_idx; j++)
        {
            DT_S32 sy = y_idx[j];
            const Tp *src_row = src.Ptr<Tp>(sy);
            float32x4_t vqf32_beta = neon::vmovq(y_coef[j]); // load y alpha

            if (sy == max_src_height)
            {
                for (DT_S32 x = 0; x < owidth; x++)
                {
                    DT_S32 tab_idx    = x << 3;
                    DT_S32 buffer_idx = x << 2;

                    DT_S32 start_x = x_idx[tab_idx];
                    if (start_x <= stop_x)
                    {
                        MVType mv_src;
                        neon::vload(src_row + C * start_x, mv_src);

                        float32x4_t vqf32_alpha_l = neon::vload1q(x_coef + tab_idx);
                        float32x4_t vqf32_alpha_h = neon::vload1q(x_coef + tab_idx + 4);

                        for (DT_S32 ch = 0; ch < C; ch++)
                        {
                            ResizeAreaCommNeonCore0<Tp, VType>(buffer_data, C * buffer_idx + 4 * ch, vqf32_alpha_l, vqf32_alpha_h,
                                                               mv_src.val[ch], vqf32_beta);
                        }
                    }
                    else
                    {
                        DT_S32 cof_idx = tab_idx;
                        DT_F32 y_beta = y_coef[j];

                        for (DT_S32 j = start_x; j <= max_src_width; j++)
                        {
                            for (DT_S32 ch = 0; ch < C; ch++)
                            {
                                buffer_data[C * buffer_idx + 4 * ch] += src_row[C * j + ch] * x_coef[cof_idx] * y_beta;
                            }
                            cof_idx++;
                        }
                    }
                }
            }
            else
            {
                for (DT_S32 x = 0; x < owidth; x++)
                {
                    DT_S32 tab_idx = x << 3;
                    DT_S32 buffer_idx = x << 2;

                    MVType mv_src;
                    neon::vload(src_row + C * x_idx[tab_idx], mv_src);

                    float32x4_t vqf32_alpha_l = neon::vload1q(x_coef + tab_idx);
                    float32x4_t vqf32_alpha_h = neon::vload1q(x_coef + tab_idx + 4);

                    for (DT_S32 ch = 0; ch < C; ch++)
                    {
                        ResizeAreaCommNeonCore0<Tp, VType>(buffer_data, C * buffer_idx + 4 * ch, vqf32_alpha_l, vqf32_alpha_h,
                                                           mv_src.val[ch], vqf32_beta);
                    }
                }
            }
        }

        // update dst_data and buffer
        {
            Tp *dst_row = dst.Ptr<Tp>(y >> 1);

            for (DT_S32 x = 0; x < owidth; x++)
            {
                DT_S32 buffer_idx = x << 2;

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    ResizeAreaCommNeonCore1<Tp>(buffer_data, C * buffer_idx + 4 * ch, dst_row + C * x + ch);
                }
            }
        }
    }

    return Status::OK;
}

// Tp = DT_F32
template <typename Tp, DT_S32 C>
static typename std::enable_if<std::is_same<DT_F32, Tp>::value, Status>::type
ResizeAreaCommNeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &thread_buffer, DT_S32 *x_idx, DT_F32 *x_coef,
                       DT_S32 *y_idx, DT_F32 *y_coef, DT_S32 *y_table, DT_S32 start_row, DT_S32 end_row)
{
    if (DT_NULL == x_idx || DT_NULL == x_coef || DT_NULL == y_idx || DT_NULL == y_coef || DT_NULL == y_table)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    using VType = typename neon::QVector<Tp>::VType;
    using MVType = typename neon::MQVector<Tp, C>::MVType;

    DT_S32 iwidth = src.GetSizes().m_width;
    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 owidth = dst.GetSizes().m_width;

    DT_F32 *buffer_data = thread_buffer.GetThreadData<DT_F32>();

    if (!buffer_data)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    DT_S32 max_src_height = iheight - 1;
    DT_S32 max_src_width = iwidth - 1;
    DT_S32 stop_x = iwidth - 8;

    float32x4_t vqf32_zeros = neon::vmovq(0.f);
    DT_S32 start_row_x2 = start_row << 1;
    DT_S32 end_row_x2   = end_row << 1;

    for (DT_S32 y = start_row_x2; y < end_row_x2; y += 2)
    {
        DT_S32 start_y_idx = y_table[y];
        DT_S32 end_y_idx = y_table[y + 1];

        // clear buffer
        for (DT_S32 x = 0; x < owidth * C; x++)
        {
            neon::vstore(buffer_data + (x << 2), vqf32_zeros);
        }

        for (DT_S32 j = start_y_idx; j < end_y_idx; j++)
        {
            DT_S32 sy = y_idx[j];
            const Tp *src_row = src.Ptr<Tp>(sy);
            float32x4_t vqf32_beta = neon::vmovq(y_coef[j]); // load y alpha

            if (sy == max_src_height)
            {
                for (DT_S32 x = 0; x < owidth; x++)
                {
                    DT_S32 tab_idx = x << 3;
                    DT_S32 buffer_idx = x << 2;

                    DT_S32 start_x = x_idx[tab_idx];
                    if (start_x <= stop_x)
                    {
                        MVType mvqf32_x0_src, mvqf32_x1_src;
                        neon::vload(src_row + C * start_x, mvqf32_x0_src);
                        neon::vload(src_row + C * start_x + 4 * C, mvqf32_x1_src);

                        float32x4_t vqf32_alpha_l = neon::vload1q(x_coef + tab_idx);
                        float32x4_t vqf32_alpha_h = neon::vload1q(x_coef + tab_idx + 4);

                        for (DT_S32 ch = 0; ch < C; ch++)
                        {
                            ResizeAreaCommNeonCore0<Tp, VType>(buffer_data, C * buffer_idx + 4 * ch, vqf32_alpha_l, vqf32_alpha_h,
                                                               mvqf32_x0_src.val[ch], mvqf32_x1_src.val[ch], vqf32_beta);
                        }
                    }
                    else
                    {
                        DT_S32 cof_idx = tab_idx;
                        DT_F32 y_beta = y_coef[j];

                        for (DT_S32 j = start_x; j <= max_src_width; j++)
                        {
                            for (DT_S32 ch = 0; ch < C; ch++)
                            {
                                buffer_data[C * buffer_idx + 4 * ch] += src_row[C * j + ch] * x_coef[cof_idx] * y_beta;
                            }
                            cof_idx++;
                        }
                    }
                }
            }
            else
            {
                for (DT_S32 x = 0; x < owidth; x++)
                {
                    DT_S32 tab_idx = x << 3;
                    DT_S32 buffer_idx = x << 2;

                    MVType mvqf32_x0_src, mvqf32_x1_src;
                    neon::vload(src_row + C * x_idx[tab_idx], mvqf32_x0_src);
                    neon::vload(src_row + C * x_idx[tab_idx] + 4 * C, mvqf32_x1_src);

                    float32x4_t vqf32_alpha_l = neon::vload1q(x_coef + tab_idx);
                    float32x4_t vqf32_alpha_h = neon::vload1q(x_coef + tab_idx + 4);

                    for (DT_S32 ch = 0; ch < C; ch++)
                    {
                        ResizeAreaCommNeonCore0<Tp, VType>(buffer_data, C * buffer_idx + 4 * ch, vqf32_alpha_l, vqf32_alpha_h,
                                                           mvqf32_x0_src.val[ch], mvqf32_x1_src.val[ch], vqf32_beta);
                    }
                }
            }
        }

        // update dst_data and buffer
        {
            Tp *dst_row = dst.Ptr<Tp>(y >> 1);

            for (DT_S32 x = 0; x < owidth; x++)
            {
                DT_S32 buffer_idx = x << 2;

                for (DT_S32 ch = 0; ch < C; ch++)
                {
                    ResizeAreaCommNeonCore1<Tp>(buffer_data, C * buffer_idx + 4 * ch, dst_row + C * x + ch);
                }
            }
        }
    }

    return Status::OK;
}

// Tp = DT_U8, DT_S8, DT_U16, DT_S16, DT_F32, MI_F16
template <typename Tp, DT_S32 C>
static Status ResizeAreaCommNeonHelper(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "null workerpool ptr");
        return Status::ERROR;
    }

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 oheight = dst.GetSizes().m_height;
    DT_F32 scale_x = static_cast<DT_F64>(iwidth) / owidth;
    DT_F32 scale_y = static_cast<DT_F64>(iheight) / oheight;

    Status ret = Status::ERROR;

    if (scale_x < 1.0 || scale_y < 1.0) /**> upscale using bilinear */
    {
        ret = ResizeBnCommNeon(ctx, src, dst, DT_TRUE, target);
    }
    else if (scale_x > 7.f)
    {
        AreaDecimateAlpha *x_tab = static_cast<AreaDecimateAlpha*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 2 * iwidth * sizeof(AreaDecimateAlpha), 0));
        AreaDecimateAlpha *y_tab = static_cast<AreaDecimateAlpha*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 2 * iheight * sizeof(AreaDecimateAlpha), 0));
        DT_S32 *tab_offset = static_cast<DT_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, (oheight + 1) * sizeof(DT_S32), 0));
        if ((DT_NULL == x_tab) || (DT_NULL == y_tab) || (DT_NULL == tab_offset))
        {
            AURA_FREE(ctx, x_tab);
            AURA_FREE(ctx, y_tab);
            AURA_FREE(ctx, tab_offset);
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
            return Status::ERROR;
        }

        DT_S32 x_tab_size = GetAreaOffset(iwidth, owidth, C, scale_x, x_tab);
        DT_S32 y_tab_size = GetAreaOffset(iheight, oheight, 1, scale_y, y_tab);
        DT_S32 dy = 0;
        for (DT_S32 k = 0; k < y_tab_size; k++)
        {
            if ((0 == k) || (y_tab[k].di != y_tab[k - 1].di))
            {
                tab_offset[dy++] = k;
            }
        }
        tab_offset[dy] = y_tab_size;

        ThreadBuffer thread_buffer(ctx, C * owidth * 2 * sizeof(DT_F32));

        ret = wp->ParallelFor(0, oheight, ResizeAreaCommNoneImpl<Tp>, ctx, std::cref(src), std::ref(dst),
                              std::ref(thread_buffer), x_tab, x_tab_size, y_tab, tab_offset);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNoneImpl<Tp> failed");
        }

        AURA_FREE(ctx, x_tab);
        AURA_FREE(ctx, y_tab);
        AURA_FREE(ctx, tab_offset);
    }
    else
    {
        const DT_S32 owidth_x8 = owidth << 3;

        DT_S32 table_size = sizeof(DT_S32) * ((owidth * 16) + (iheight * 4) + (oheight * 2));
        DT_S32 *table_data = static_cast<DT_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, table_size, 0));
        if (DT_NULL == table_data)
        {
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
            return Status::ERROR;
        }

        DT_S32 *x_idx   = table_data;
        DT_F32 *x_coef  = reinterpret_cast<DT_F32*>(x_idx + owidth_x8);
        DT_S32 *y_idx   = reinterpret_cast<DT_S32*>(x_coef + owidth_x8);
        DT_F32 *y_coef  = reinterpret_cast<DT_F32*>(y_idx + iheight * 2);
        DT_S32 *y_table = reinterpret_cast<DT_S32*>(y_coef + iheight * 2);

        if (GetAreaOffsetAlign8X(ctx, iwidth, owidth, scale_x, x_idx, x_coef) != Status::OK)
        {
            AURA_FREE(ctx, table_data);
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaGetOffsetCoefX failed");
            return Status::ERROR;
        }

        if (GetAreaOffsetAlign8Y(ctx, iheight, oheight, scale_y, y_idx, y_coef, y_table) != Status::OK)
        {
            AURA_FREE(ctx, table_data);
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaGetOffsetCoefY failed");
            return Status::ERROR;
        }

        ThreadBuffer thread_buffer(ctx, sizeof(DT_F32) * owidth * 4 * C);

        ret = wp->ParallelFor(0, oheight, ResizeAreaCommNeonImpl<Tp, C>, ctx, std::cref(src), std::ref(dst),
                              std::ref(thread_buffer), x_idx, x_coef, y_idx, y_coef, y_table);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonImpl<Tp, DT_F32, C> failed");
        }

        AURA_FREE(ctx, table_data);
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeAreaCommNeon(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    DT_S32 channel = src.GetSizes().m_channel;
    DT_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), channel);

    Status ret = Status::ERROR;

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, 1):
        {
            ret = ResizeAreaCommNeonHelper<DT_U8, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_U8, C1");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S8, 1):
        {
            ret = ResizeAreaCommNeonHelper<DT_S8, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_S8, C1");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, 1):
        {
            ret = ResizeAreaCommNeonHelper<DT_U16, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_U16, C1");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, 1):
        {
            ret = ResizeAreaCommNeonHelper<DT_S16, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_S16, C1");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, 1):
        {
            ret = ResizeAreaCommNeonHelper<MI_F16, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:MI_F16, C1");
            }
            break;
        }
#endif

        case AURA_MAKE_PATTERN(ElemType::F32, 1):
        {
            ret = ResizeAreaCommNeonHelper<DT_F32, 1>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_F32, C1");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U8, 2):
        {
            ret = ResizeAreaCommNeonHelper<DT_U8, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_U8, C2");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S8, 2):
        {
            ret = ResizeAreaCommNeonHelper<DT_S8, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_S8, C2");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, 2):
        {
            ret = ResizeAreaCommNeonHelper<DT_U16, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_U16, C2");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, 2):
        {
            ret = ResizeAreaCommNeonHelper<DT_S16, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_S16, C2");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, 2):
        {
            ret = ResizeAreaCommNeonHelper<MI_F16, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:MI_F16, C2");
            }
            break;
        }
#endif

        case AURA_MAKE_PATTERN(ElemType::F32, 2):
        {
            ret = ResizeAreaCommNeonHelper<DT_F32, 2>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_F32, C2");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U8, 3):
        {
            ret = ResizeAreaCommNeonHelper<DT_U8, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_U8, C3");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S8, 3):
        {
            ret = ResizeAreaCommNeonHelper<DT_S8, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_S8, C3");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, 3):
        {
            ret = ResizeAreaCommNeonHelper<DT_U16, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_U16, C3");
            }
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, 3):
        {
            ret = ResizeAreaCommNeonHelper<DT_S16, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_S16, C3");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, 3):
        {
            ret = ResizeAreaCommNeonHelper<MI_F16, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:MI_F16, C3");
            }
            break;
        }
#endif

        case AURA_MAKE_PATTERN(ElemType::F32, 3):
        {
            ret = ResizeAreaCommNeonHelper<DT_F32, 3>(ctx, src, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommNeonHelper failed, type:DT_F32, C3");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel number or data type");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura