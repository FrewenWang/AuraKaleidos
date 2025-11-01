#include "canny_impl.hpp"
#include "aura/ops/filter.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status CannyNeonImpl(Context *ctx, const Mat &dx, const Mat &dy, Mat &map, std::deque<MI_U8*> &border_peaks_parallel, ThreadBuffer &thread_buffer,
                            std::mutex &mutex, MI_S32 low_thresh, MI_S32 high_thresh, MI_BOOL l2_gradient, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 iwidth  = dx.GetSizes().m_width;
    MI_S32 iheight = dx.GetSizes().m_height;
    MI_S32 channel = dx.GetSizes().m_channel;
    MI_S32 map_w   = map.GetSizes().m_width;
    start_row      = start_row << 4;
    end_row        = Min(end_row << 4, iheight);

    MI_S32 start_row_idx = Max(0, (start_row - 1));
    MI_S32 end_row_idx   = Min(iheight, (end_row + 1));

    MI_S32 *buffer       = MI_NULL;
    const MI_S16 *dx_row = MI_NULL;
    const MI_S16 *dy_row = MI_NULL;
    MI_S16 *dx_c         = MI_NULL;
    MI_S16 *dy_c         = MI_NULL;
    MI_S16 *dx_n         = MI_NULL;
    MI_S16 *dy_n         = MI_NULL;
    MI_S16 *dx_max       = MI_NULL;
    MI_S16 *dy_max       = MI_NULL;

    if (channel > 1)
    {
        MI_U8 *rows = thread_buffer.GetThreadData<MI_U8>();

        if (!rows)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
            return Status::ERROR;
        }

        dx_max = reinterpret_cast<MI_S16*>(rows);
        dy_max = reinterpret_cast<MI_S16*>(rows + 2 * iwidth * sizeof(MI_S16));
        dx_c   = dx_max;
        dx_n   = dx_c + iwidth;
        dy_c   = dy_max;
        dy_n   = dy_c + iwidth;
        buffer = reinterpret_cast<MI_S32*>(rows + 4 * iwidth * sizeof(MI_S16));
    }
    else
    {
        MI_U8 *rows = thread_buffer.GetThreadData<MI_U8>();

        if (!rows)
        {
            AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
            return Status::ERROR;
        }

        buffer = reinterpret_cast<MI_S32*>(rows);
    }

    std::deque<MI_U8*> stack, border_peaks_local;;
    MI_S32 *mag_p = buffer + 1;
    MI_S32 *mag_c = mag_p + map_w * channel;
    MI_S32 *mag_n = mag_c + map_w * channel;

    if(start_row_idx == start_row)
    {
        memset(mag_n - 1, 0, map_w * sizeof(MI_S32));
    }
    else
    {
        mag_n[iwidth] = mag_n[-1] = 0;
    }

    mag_c[iwidth] = mag_c[-1] = mag_p[iwidth] = mag_p[-1] = 0;

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (MI_S32 y = start_row_idx; y <= end_row; ++y)
    {
        Swap(mag_n, mag_c);
        Swap(mag_n, mag_p);

        if (y < end_row_idx)
        {
            dx_row = dx.Ptr<MI_S16>(y);
            dy_row = dy.Ptr<MI_S16>(y);
            MI_S32 width = iwidth * channel;
            constexpr MI_S32 ELEM_COUNTS = 8;
            MI_S32 width_align8 = width & (-ELEM_COUNTS);
            MI_S32 x = 0;

            if (l2_gradient)
            {
                for (; x < width_align8; x += ELEM_COUNTS)
                {
                    int16x8_t vqs16_dx        = neon::vload1q(dx_row + x);
                    int16x8_t vqs16_dy        = neon::vload1q(dy_row + x);
                    int32x4_t vqs32_dx_lo     = neon::vmull(neon::vgetlow(vqs16_dx), neon::vgetlow(vqs16_dx));
                    int32x4_t vqs32_dx_hi     = neon::vmull(neon::vgethigh(vqs16_dx), neon::vgethigh(vqs16_dx));
                    int32x4_t vqs32_dy_lo     = neon::vmull(neon::vgetlow(vqs16_dy), neon::vgetlow(vqs16_dy));
                    int32x4_t vqs32_dy_hi     = neon::vmull(neon::vgethigh(vqs16_dy), neon::vgethigh(vqs16_dy));
                    int32x4_t vqs32_result_lo = neon::vadd(vqs32_dx_lo, vqs32_dy_lo);
                    int32x4_t vqs32_result_hi = neon::vadd(vqs32_dx_hi, vqs32_dy_hi);
                    neon::vstore(mag_n + x, vqs32_result_lo);
                    neon::vstore(mag_n + x + 4, vqs32_result_hi);
                }
                for (; x < width; ++x)
                {
                    mag_n[x] = static_cast<MI_S32>(dx_row[x]) * dx_row[x] + static_cast<MI_S32>(dy_row[x]) * dy_row[x];
                }
            }
            else
            {
                for (; x < width_align8; x += ELEM_COUNTS)
                {
                    int16x8_t vqs16_dx        = neon::vload1q(dx_row + x);
                    int16x8_t vqs16_dy        = neon::vload1q(dy_row + x);
                    int32x4_t vqs32_dx_lo     = neon::vabs(neon::vmovl(neon::vgetlow(vqs16_dx)));
                    int32x4_t vqs32_dx_hi     = neon::vabs(neon::vmovl(neon::vgethigh(vqs16_dx)));
                    int32x4_t vqs32_dy_lo     = neon::vabs(neon::vmovl(neon::vgetlow(vqs16_dy)));
                    int32x4_t vqs32_dy_hi     = neon::vabs(neon::vmovl(neon::vgethigh(vqs16_dy)));
                    int32x4_t vqs32_result_lo = neon::vadd(vqs32_dx_lo, vqs32_dy_lo);
                    int32x4_t vqs32_result_hi = neon::vadd(vqs32_dx_hi, vqs32_dy_hi);
                    neon::vstore(mag_n + x, vqs32_result_lo);
                    neon::vstore(mag_n + x + 4, vqs32_result_hi);
                }
                for (; x < width; ++x)
                {
                    mag_n[x] = Abs(static_cast<MI_S32>(dx_row[x])) + Abs(static_cast<MI_S32>(dy_row[x]));
                }
            }

            if (channel > 1)
            {
                Swap(dx_n, dx_c);
                Swap(dy_n, dy_c);

                for (MI_S32 x = 0, k = 0; x < iwidth; ++x, k += channel)
                {
                    MI_S32 max_idx = k;

                    for (MI_S32 c = 1; c < channel; ++c)
                    {
                        if (mag_n[k + c] > mag_n[max_idx])
                        {
                            max_idx = k + c;
                        }
                    }

                    mag_n[x] = mag_n[max_idx];
                    dx_n[x]  = dx_row[max_idx];
                    dy_n[x]  = dy_row[max_idx];
                }

                mag_n[iwidth] = 0;
            }

            if (y <= start_row)
            {
                continue;
            }
        }
        else
        {
            memset(mag_n - 1, 0, map_w * sizeof(MI_S32));

            if (channel > 1)
            {
                Swap(dx_n, dx_c);
                Swap(dy_n, dy_c);
            }
        }

        // From here actual src row is (y - 1)
        // Set left and right border to 1
        MI_U8 *map_row = map.Ptr<MI_U8>(y) + 1;
        map_row[-1] = 1;
        map_row[iwidth] = 1;

        if(1 == channel)
        {
            dx_row = dx.Ptr<MI_S16>(y - 1);
            dy_row = dy.Ptr<MI_S16>(y - 1);
        }
        else
        {
            dx_row = dx_c;
            dy_row = dy_c;
        }

        constexpr MI_S32 tg22 = 13573; // tan22.5*(1<<15)
        for (MI_S32 x = 0; x < iwidth; ++x)
        {
            MI_S32 m = mag_c[x];
            if (m > low_thresh)
            {
                MI_S16 xs = dx_row[x];
                MI_S16 ys = dy_row[x];
                MI_S32 xu = Abs(xs);
                MI_S32 yu = Abs(ys) << 15;
                MI_S32 tg22x = xu * tg22;

                if (yu < tg22x)
                {
                    if ((m > mag_c[x - 1]) && (m >= mag_c[x + 1]))
                    {
                        CannyCheck(m, high_thresh, (map_row + x), stack);
                        continue;
                    }
                }
                else
                {
                    MI_S32 tg67x = tg22x + (xu << 16); // tan67.5
                    if (yu > tg67x)
                    {
                        if ((m > mag_p[x]) && (m >= mag_n[x]))
                        {
                            CannyCheck(m, high_thresh, (map_row + x), stack);
                            continue;
                        }
                    }
                    else
                    {
                        MI_S32 s = (xs ^ ys) < 0 ? -1 : 1;

                        if ((m > mag_p[x - s]) && (m > mag_n[x + s]))
                        {
                            CannyCheck(m, high_thresh, (map_row + x), stack);
                            continue;
                        }
                    }
                }
            }

            map_row[x] = 1;
        }
    }

    // Not for first row of first slice or last row of last slice
    MI_U8 *map_start  = map.Ptr<MI_U8>(0);
    MI_U8 *map_lower  = (start_row_idx == 0) ? map_start : (map_start + (start_row + 2) * map_w);
    MI_U8 *data_limit = map_start + map.GetSizes().m_height * map_w;
    MI_U32 map_diff   = static_cast<MI_U32>(((end_row_idx == iheight) ? data_limit : (map_start + end_row * map_w)) - map_lower);

    while (!stack.empty())
    {
        MI_U8 *m = stack.back();
        stack.pop_back();

        if((unsigned)(m - map_lower) < map_diff)
        {
            if (!m[-map_w - 1])
            {
                CannyPush((m - map_w - 1), stack);
            }
            if (!m[-map_w])
            {
                CannyPush((m - map_w), stack);
            }
            if (!m[-map_w + 1])
            {
                CannyPush((m - map_w + 1), stack);
            }
            if (!m[-1])
            {
                CannyPush((m - 1), stack);
            }
            if (!m[1])
            {
                CannyPush((m + 1), stack);
            }
            if (!m[map_w - 1])
            {
                CannyPush((m + map_w - 1), stack);
            }
            if (!m[map_w])
            {
                CannyPush((m + map_w), stack);
            }
            if (!m[map_w + 1])
            {
                CannyPush((m + map_w + 1), stack);
            }
        }
        else
        {
            border_peaks_local.push_back(m);
            MI_S64 map_step2 = m < map_lower ? map_w : -map_w;

            if (!m[-1])
            {
                CannyPush((m-1), stack);
            }
            if (!m[1])
            {
                CannyPush((m+1), stack);
            }
            if (!m[map_step2 - 1])
            {
                CannyPush((m + map_step2 - 1), stack);
            }
            if (!m[map_step2])
            {
                CannyPush((m + map_step2), stack);
            }
            if (!m[map_step2 + 1])
            {
                CannyPush((m + map_step2 + 1), stack);
            }
        }
    }

    if(!border_peaks_local.empty())
    {
        std::lock_guard<std::mutex> guard(mutex);
        border_peaks_parallel.insert(border_peaks_parallel.end(), border_peaks_local.begin(), border_peaks_local.end());
    }

    return Status::OK;
}

static Status CannyFinalNeonImpl(Mat &map, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width = dst.GetSizes().m_width;
    constexpr MI_S32 ELEM_COUNTS = 16;
    MI_S32 width_align16 = width & (-ELEM_COUNTS);

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        MI_U8 *map_c = map.Ptr<MI_U8>(y + 1) + 1;
        MI_U8 *dst_c = dst.Ptr<MI_U8>(y);
        MI_S32 x = 0;

        for (; x < width_align16; x += ELEM_COUNTS)
        {
            uint8x16_t vqu8_map    = neon::vload1q(map_c + x);
            uint8x16_t vqu8_map0   = neon::vshr_n<1>(vqu8_map);
            uint8x16_t vqu8_result = neon::vmul(vqu8_map0, (MI_U8)(255));
            neon::vstore(dst_c + x, vqu8_result);
        }

        for (; x < width; ++x)
        {
            dst_c[x] = static_cast<MI_U8>(-(map_c[x] >> 1));
        }
    }

    return Status::OK;
}

CannyNeon::CannyNeon(Context *ctx, const OpTarget &target) : CannyImpl(ctx, target)
{}

Status CannyNeon::SetArgs(const Array *src, Array *dst, MI_F64 low_thresh, MI_F64 high_thresh,
                          MI_S32 aperture_size, MI_BOOL l2_gradient)
{
    if (CannyImpl::SetArgs(src, dst, low_thresh, high_thresh, aperture_size, l2_gradient) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CannyImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status CannyNeon::SetArgs(const Array *dx, const Array *dy, Array *dst, MI_F64 low_thresh,
                          MI_F64 high_thresh, MI_BOOL l2_gradient)
{
    if (CannyImpl::SetArgs(dx, dy, dst, low_thresh, high_thresh, l2_gradient) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CannyImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((dx->GetArrayType() != ArrayType::MAT) || (dy->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status CannyNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    const Mat *src_dx = dynamic_cast<const Mat*>(m_dx);
    const Mat *src_dy = dynamic_cast<const Mat*>(m_dy);
    Mat *dst = dynamic_cast<Mat *>(m_dst);

    if ((MI_NULL == src) && m_is_aperture)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    if (((MI_NULL == src_dx) || (MI_NULL == src_dy)) && !m_is_aperture)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_dx m_dy is null");
        return Status::ERROR;
    }

    if (MI_NULL == dst)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst is null");
        return Status::ERROR;
    }

    WorkerPool *wp = m_ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerPool fail");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if (m_is_aperture)
    {
        MI_F64 scale = 1.0;

        if (7 == m_aperture_size)
        {
            m_low_thresh  = m_low_thresh / 16.0;
            m_high_thresh = m_high_thresh / 16.0;
            scale         = 1 / 16.0;
        }

        if (m_low_thresh > m_high_thresh)
        {
            Swap(m_low_thresh, m_high_thresh);
        }

        if (m_l2_gradient)
        {
            m_low_thresh  = Min(32767.0, m_low_thresh);
            m_high_thresh = Min(32767.0, m_high_thresh);

            if (m_low_thresh > 0)
            {
                m_low_thresh *= m_low_thresh;
            }

            if (m_high_thresh > 0)
            {
                m_high_thresh *= m_high_thresh;
            }
        }

        MI_S32 low  = static_cast<MI_S32>(Floor(m_low_thresh));
        MI_S32 high = static_cast<MI_S32>(Floor(m_high_thresh));

        Sizes3 size = src->GetSizes();
        Mat dx(m_ctx, ElemType::S16, size);
        Mat dy(m_ctx, ElemType::S16, size);

        ret  = ISobel(m_ctx, (*(dynamic_cast<const Mat*>(m_src))), dx, 1, 0, m_aperture_size, scale, BorderType::REPLICATE, Scalar(), OpTarget::Neon());
        ret |= ISobel(m_ctx, (*(dynamic_cast<const Mat*>(m_src))), dy, 0, 1, m_aperture_size, scale, BorderType::REPLICATE, Scalar(), OpTarget::Neon());
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "sobel excute failed");
            return Status::ERROR;
        }

        MI_S32 iwidth  = src->GetSizes().m_width;
        MI_S32 iheight = src->GetSizes().m_height;
        MI_S32 channel = src->GetSizes().m_channel;
        MI_S32 map_w   = (iwidth + 2);
        MI_S32 map_h   = iheight + 2;

        Mat map(m_ctx, ElemType::U8, Sizes3(map_h, map_w, 1));
        if (!map.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "map is invalid");
            return Status::ERROR;
        }

        MI_U8 *map_t = map.Ptr<MI_U8>(0);
        MI_U8 *map_b = map.Ptr<MI_U8>(iheight + 1);

        for (MI_S32 x = 0; x < map_w; ++x)
        {
            map_t[x] = 1;
            map_b[x] = 1;
        }

        MI_S32 buffer_size = 0;
        if (channel > 1)
        {
            buffer_size = (4 * iwidth * sizeof(MI_S16)) + (3 * (map_w * channel) * sizeof(MI_S32));
        }
        else
        {
            buffer_size = 3 * (map_w * channel) * sizeof(MI_S32);
        }

        ThreadBuffer thread_buffer(m_ctx, buffer_size);

        std::deque<MI_U8*> border_peaks_parallel;
        std::mutex mutex;

        ret = wp->ParallelFor(0, AURA_ALIGN(src->GetSizes().m_height, 16) / 16, CannyNeonImpl, m_ctx, std::cref(dx), std::ref(dy), std::ref(map),
                              std::ref(border_peaks_parallel), std::ref(thread_buffer), std::ref(mutex), low, high, m_l2_gradient);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CannyNeonImpl excute failed");
            return Status::ERROR;
        }

        // process borderPeaksParallel
        while (!border_peaks_parallel.empty())
        {
            MI_U8* m = border_peaks_parallel.back();
            border_peaks_parallel.pop_back();

            if (!m[-map_w - 1])
            {
                CannyPush((m - map_w - 1), border_peaks_parallel);
            }

            if (!m[-map_w])
            {
                CannyPush((m - map_w), border_peaks_parallel);
            }

            if (!m[-map_w+1])
            {
                CannyPush((m - map_w + 1), border_peaks_parallel);
            }

            if (!m[-1])
            {
                CannyPush((m - 1), border_peaks_parallel);
            }

            if (!m[1])
            {
                CannyPush((m + 1), border_peaks_parallel);
            }

            if (!m[map_w - 1])
            {
                CannyPush((m + map_w - 1), border_peaks_parallel);
            }

            if (!m[map_w])
            {
                CannyPush((m + map_w), border_peaks_parallel);
            }

            if (!m[map_w + 1])
            {
                CannyPush((m + map_w + 1), border_peaks_parallel);
            }
        }

        ret = wp->ParallelFor(0, src->GetSizes().m_height, CannyFinalNeonImpl, std::ref(map), std::ref(*dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CannyFinalNeonImpl excute failed");
            return Status::ERROR;
        }
    }
    else
    {
        if (m_low_thresh > m_high_thresh)
        {
            Swap(m_low_thresh, m_high_thresh);
        }

        if (m_l2_gradient)
        {
            m_low_thresh  = Min(32767.0, m_low_thresh);
            m_high_thresh = Min(32767.0, m_high_thresh);

            if (m_low_thresh > 0)
            {
                m_low_thresh *= m_low_thresh;
            }

            if (m_high_thresh > 0)
            {
                m_high_thresh *= m_high_thresh;
            }
        }

        MI_S32 low  = static_cast<MI_S32>(Floor(m_low_thresh));
        MI_S32 high = static_cast<MI_S32>(Floor(m_high_thresh));

        MI_S32 iwidth  = src_dx->GetSizes().m_width;
        MI_S32 iheight = src_dx->GetSizes().m_height;
        MI_S32 channel = src_dx->GetSizes().m_channel;
        MI_S32 map_w   = (iwidth + 2);
        MI_S32 map_h   = iheight + 2;

        Mat map(m_ctx, ElemType::U8, Sizes3(map_h, map_w, 1));
        MI_U8 *map_t = map.Ptr<MI_U8>(0);
        MI_U8 *map_b = map.Ptr<MI_U8>(iheight + 1);

        for (MI_S32 x = 0; x < map_w; ++x)
        {
            map_t[x] = 1;
            map_b[x] = 1;
        }

        MI_S32 buffer_size = 0;
        if (channel > 1)
        {
            buffer_size = (4 * iwidth * sizeof(MI_S16)) + (3 * (map_w * channel) * sizeof(MI_S32));
        }
        else
        {
            buffer_size = 3 * (map_w * channel) * sizeof(MI_S32);
        }

        ThreadBuffer thread_buffer(m_ctx, buffer_size);

        std::deque<MI_U8*> border_peaks_parallel;
        std::mutex mutex;

        ret = wp->ParallelFor(0, AURA_ALIGN(src_dx->GetSizes().m_height, 16) / 16, CannyNeonImpl, m_ctx, std::cref(*src_dx), std::ref(*src_dy), std::ref(map),
                              std::ref(border_peaks_parallel), std::ref(thread_buffer), std::ref(mutex), low, high, m_l2_gradient);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CannyNeonImpl excute failed");
            return Status::ERROR;
        }

        // process borderPeaksParallel
        while (!border_peaks_parallel.empty())
        {
            MI_U8* m = border_peaks_parallel.back();
            border_peaks_parallel.pop_back();

            if (!m[-map_w - 1])
            {
                CannyPush((m - map_w - 1), border_peaks_parallel);
            }
            if (!m[-map_w])
            {
                CannyPush((m - map_w), border_peaks_parallel);
            }
            if (!m[-map_w+1])
            {
                CannyPush((m - map_w + 1), border_peaks_parallel);
            }
            if (!m[-1])
            {
                CannyPush((m - 1), border_peaks_parallel);
            }
            if (!m[1])
            {
                CannyPush((m + 1), border_peaks_parallel);
            }
            if (!m[map_w - 1])
            {
                CannyPush((m + map_w - 1), border_peaks_parallel);
            }
            if (!m[map_w])
            {
                CannyPush((m + map_w), border_peaks_parallel);
            }
            if (!m[map_w + 1])
            {
                CannyPush((m + map_w + 1), border_peaks_parallel);
            }
        }

        ret = wp->ParallelFor(0, src_dx->GetSizes().m_height, CannyFinalNeonImpl, std::ref(map), std::ref(*dst));
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CannyFinalNeonImpl excute failed");
            return Status::ERROR;
        }

    }

    return ret;
}

} // namespace aura