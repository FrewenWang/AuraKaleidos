#include "canny_impl.hpp"
#include "aura/ops/filter.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status CannyNoneImpl(Context *ctx, const Mat &dx, const Mat &dy, Mat &dst, MI_S32 low_thresh, MI_S32 high_thresh, MI_BOOL l2_gradient)
{
    MI_S32 iwidth  = dx.GetSizes().m_width;
    MI_S32 iheight = dx.GetSizes().m_height;
    MI_S32 channel = dx.GetSizes().m_channel;
    MI_S32 map_w   = iwidth + 2;
    MI_S32 map_h   = iheight + 2;

    Mat map(ctx, ElemType::U8, Sizes3(map_h, map_w, 1));
    if (!map.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "map is invalid");
        return Status::ERROR;
    }

    MI_U8 *map_t = map.Ptr<MI_U8>(0);
    MI_U8 *map_b = map.Ptr<MI_U8>(iheight + 1);

    for (MI_S32 x = 0; x < map_w; ++x)
    {
        map_t[x] = 1;
        map_b[x] = 1;
    }

    const MI_S16 *dx_row = MI_NULL;
    const MI_S16 *dy_row = MI_NULL;
    MI_S16 *dx_c   = MI_NULL;
    MI_S16 *dy_c   = MI_NULL;
    MI_S16 *dx_n   = MI_NULL;
    MI_S16 *dy_n   = MI_NULL;
    MI_S16 *dx_max = MI_NULL;
    MI_S16 *dy_max = MI_NULL;

    if (channel > 1)
    {
        dx_max = static_cast<MI_S16*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 2 * iwidth * sizeof(MI_S16), 0));
        dy_max = static_cast<MI_S16*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 2 * iwidth * sizeof(MI_S16), 0));
        if ((MI_NULL == dx_max) || (MI_NULL == dy_max))
        {
            AURA_FREE(ctx, dx_max);
            AURA_FREE(ctx, dy_max);
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
            return Status::ERROR;
        }

        dx_c = dx_max;
        dx_n = dx_c + iwidth;
        dy_c = dy_max;
        dy_n = dy_c + iwidth;
    }

    MI_S32 *buffer = static_cast<MI_S32*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, 3 * (map_w * channel) * sizeof(MI_S32), 0));
    if (MI_NULL == buffer)
    {

        AURA_FREE(ctx, dx_max);
        AURA_FREE(ctx, dy_max);
        AURA_FREE(ctx, buffer);
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return Status::ERROR;
    }

    std::deque<MI_U8*> stack;
    MI_S32 *mag_p = buffer + 1;
    MI_S32 *mag_c = mag_p + map_w * channel;
    MI_S32 *mag_n = mag_c + map_w * channel;

    memset(mag_n - 1, 0, map_w * sizeof(MI_S32));
    mag_c[iwidth] = mag_c[-1] = mag_p[iwidth] = mag_p[-1] = 0;

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (MI_S32 y = 0; y <= iheight; ++y)
    {
        Swap(mag_n, mag_c);
        Swap(mag_n, mag_p);

        if (y < iheight)
        {
            dx_row = dx.Ptr<MI_S16>(y);
            dy_row = dy.Ptr<MI_S16>(y);
            MI_S32 width = iwidth * channel;

            if (l2_gradient)
            {
                for (MI_S32 x = 0; x < width; ++x)
                {
                    mag_n[x] = static_cast<MI_S32>(dx_row[x]) * dx_row[x] + static_cast<MI_S32>(dy_row[x]) * dy_row[x];
                }
            }
            else
            {
                for (MI_S32 x = 0; x < width; ++x)
                {
                    mag_n[x] = Abs(static_cast<MI_S32>(dx_row[x])) + Abs(static_cast<MI_S32>(dy_row[x]));
                }
            }

            if (channel > 1)
            {
                Swap(dx_n, dx_c);
                Swap(dy_n, dy_c);

                for (MI_S32 x = 0, jn = 0; x < iwidth; ++x, jn += channel)
                {
                    MI_S32 max_idx = jn;

                    for (MI_S32 k = 1; k < channel; ++k)
                    {
                        if (mag_n[jn + k] > mag_n[max_idx])
                        {
                            max_idx = jn + k;
                        }
                    }

                    mag_n[x] = mag_n[max_idx];
                    dx_n[x]  = dx_row[max_idx];
                    dy_n[x]  = dy_row[max_idx];
                }

                mag_n[iwidth] = 0;
            }

            if (y <= 0)
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
        MI_U8 *map_ptr = map.Ptr<MI_U8>(y) + 1;
        map_ptr[-1] = 1;
        map_ptr[iwidth] = 1;

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

        constexpr MI_S32 TG22 = 13573; // tan22.5*(1<<15)
        for (MI_S32 x = 0; x < iwidth; ++x)
        {
            MI_S32 m = mag_c[x];
            if (m > low_thresh)
            {
                MI_S16 xs = dx_row[x];
                MI_S16 ys = dy_row[x];
                MI_S32 xu  = Abs(xs);
                MI_S32 yu  = Abs(ys) << 15;

                MI_S32 tg22x = xu * TG22;

                if (yu < tg22x)
                {
                    if ((m > mag_c[x - 1]) && (m >= mag_c[x + 1]))
                    {
                        CannyCheck(m, high_thresh, (map_ptr + x), stack);
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
                            CannyCheck(m, high_thresh, (map_ptr + x), stack);
                            continue;
                        }
                    }
                    else
                    {
                        MI_S32 s = (xs ^ ys) < 0 ? -1 : 1;

                        if ((m > mag_p[x - s]) && (m > mag_n[x + s]))
                        {
                            CannyCheck(m, high_thresh, (map_ptr + x), stack);
                            continue;
                        }
                    }
                }
            }

            map_ptr[x] = 1;
        }
    }

    while (!stack.empty())
    {
        MI_U8 *m = stack.back();
        stack.pop_back();

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

    for (MI_S32 y = 0; y < iheight; ++y)
    {
        MI_U8 *map_row = map.Ptr<MI_U8>(y + 1) + 1;
        MI_U8 *dst_row = dst.Ptr<MI_U8>(y);

        for (MI_S32 x = 0; x < iwidth; ++x)
        {
            dst_row[x] = static_cast<MI_U8>(-(map_row[x] >> 1));
        }
    }

    AURA_FREE(ctx, dx_max);
    AURA_FREE(ctx, dy_max);
    AURA_FREE(ctx, buffer);

    return Status::OK;
}

CannyNone::CannyNone(Context *ctx, const OpTarget &target) : CannyImpl(ctx, target)
{}

Status CannyNone::SetArgs(const Array *src, Array *dst, MI_F64 low_thresh, MI_F64 high_thresh,
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

Status CannyNone::SetArgs(const Array *dx, const Array *dy, Array *dst, MI_F64 low_thresh,
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

Status CannyNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    const Mat *src_dx = dynamic_cast<const Mat*>(m_dx);
    const Mat *src_dy = dynamic_cast<const Mat*>(m_dy);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src)  && m_is_aperture)
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

    Status ret = Status::ERROR;
    MI_S32 low, high;

    if (m_is_aperture)
    {
        MI_F64 scale = 1.0;

        if (7 == m_aperture_size)
        {
            m_low_thresh  = m_low_thresh / 16.0;
            m_high_thresh = m_high_thresh / 16.0;
            scale       = 1 / 16.0;
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

        low  = static_cast<MI_S32>(Floor(m_low_thresh));
        high = static_cast<MI_S32>(Floor(m_high_thresh));

        Sizes3 size = src->GetSizes();
        Mat dx(m_ctx, ElemType::S16, size);
        Mat dy(m_ctx, ElemType::S16, size);
        if (!dx.IsValid() || !dy.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dx or dy is invalid");
            return Status::ERROR;
        }

        ret =  ISobel(m_ctx, *src, dx, 1, 0, m_aperture_size, scale, BorderType::REPLICATE, Scalar(), OpTarget::None());
        ret |= ISobel(m_ctx, *src, dy, 0, 1, m_aperture_size, scale, BorderType::REPLICATE, Scalar(), OpTarget::None());
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "sobel excute failed");
            return Status::ERROR;
        }

        ret = CannyNoneImpl(m_ctx, dx, dy, *dst, low, high, m_l2_gradient);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CannyNoneImpl excute failed");
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

        low  = static_cast<MI_S32>(Floor(m_low_thresh));
        high = static_cast<MI_S32>(Floor(m_high_thresh));

        ret = CannyNoneImpl(m_ctx, *src_dx, *src_dy, *dst, low, high, m_l2_gradient);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CannyNoneImpl excute failed");
            return Status::ERROR;
        }
    }

    return ret;
}

} // namespace aura