#include "houghcircles_impl.hpp"
#include "aura/ops/filter.h"
#include "aura/ops/feature2d.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/logger.h"

namespace aura
{

struct HoughAccumParam
{
    DT_S32 index;
    DT_S32 value;
};

struct CircleParam
{
    CircleParam() : x(0), y(0), r(0), accum(0)
    {}

    CircleParam(DT_F32 x, DT_F32 y, DT_F32 r, DT_F32 accum) : x(x), y(y), r(r), accum(accum)
    {}

    DT_F32 x;
    DT_F32 y;
    DT_F32 r;
    DT_F32 accum;
};

static Status HoughCirclesAccum(Context *ctx, const Mat &edges, const Mat &dx, const Mat &dy, Mat &accum, Mat &nz, DT_S32 min_radius,
                                DT_S32 max_radius, DT_F32 idp, volatile DT_S32 *nz_size, ThreadBuffer &thread_buffer, std::mutex &mutex,
                                DT_S32 start_blk, DT_S32 end_blk)
{
    DT_S32 awidth  = accum.GetSizes().m_width;
    DT_S32 aheight = accum.GetSizes().m_height;

    DT_S32 start_row = start_blk << 4;
    DT_S32 end_row   = Min(end_blk << 4, edges.GetSizes().m_height);

    DT_S32 nz_size_tmp    = 0;
    DT_S32 iwidth         = edges.GetSizes().m_width;
    DT_S32 process_height = end_row - start_row;

    DT_S32 acols = awidth  - 2;
    DT_S32 arows = aheight - 2;
    DT_S32 astep = awidth;

    DT_U8 *data_buffer = thread_buffer.GetThreadData<DT_U8>();

    if (!data_buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    memset(data_buffer, 0, aheight * awidth * sizeof(DT_S32) + iwidth * process_height * sizeof(DT_U8));

    DT_S32 *accum_data = reinterpret_cast<DT_S32*>(data_buffer + iwidth * process_height);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const DT_U8 *edges_data = edges.Ptr<DT_U8>(y);
        const DT_S16 *dx_data = dx.Ptr<DT_S16>(y);
        const DT_S16 *dy_data = dy.Ptr<DT_S16>(y);
        DT_U8 *nz_data = data_buffer + (y - start_row) * iwidth;

        for (DT_S32 x = 0; x < iwidth; x++)
        {
            if (!edges_data[x])
            {
                continue;
            }

            DT_F32 vx = dx_data[x];
            DT_F32 vy = dy_data[x];
            if (0 == vx && 0 == vy)
            {
                continue;
            }

            DT_F64 mag = Sqrt(vx * vx + vy * vy);
            if (mag < 1.0f)
            {
                continue;
            }

            DT_S32 sx, sy, x0, y0, x1, y1;
            nz_data[x] = 1;
            ++nz_size_tmp;

            sx = Round((vx * idp) * 1024 / mag);
            sy = Round((vy * idp) * 1024 / mag);

            x0 = Round(x * idp * 1024);
            y0 = Round(y * idp * 1024);

            for (DT_S32 k = 0; k < 2; k++)
            {
                x1 = x0 + min_radius * sx;
                y1 = y0 + min_radius * sy;

                for (DT_S32 r = min_radius; r <= max_radius; x1 += sx, y1 += sy, r++)
                {
                    DT_S32 x2 = x1 >> 10, y2 = y1 >> 10;
                    if (static_cast<DT_U32>(x2) >= static_cast<DT_U32>(acols) ||
                        static_cast<DT_U32>(y2) >= static_cast<DT_U32>(arows))
                    {
                        break;
                    }

                    accum_data[y2 * astep + x2]++;
                }

                sx = -sx;
                sy = -sy;
            }
        }
    }

    std::lock_guard<std::mutex> guard(mutex);
    *nz_size += nz_size_tmp;
    std::memcpy(nz.Ptr<DT_U8>(start_row), data_buffer, process_height * iwidth);

    for (DT_S32 y = 0; y < aheight; y++)
    {
        DT_S32 *accum_data_tmp = accum_data + y * awidth;
        DT_S32 *accum_data_rst = accum.Ptr<DT_S32>(y);

        for (DT_S32 x = 0; x < awidth; x++)
        {
            accum_data_rst[x] += accum_data_tmp[x];
        }
    }

    return Status::OK;
}

static Status HoughCirclesFindCenters(Context *ctx, const Mat &accum, HoughAccumParam *centers, DT_S32 acc_thresh, DT_S32 &size)
{
    if (DT_NULL == centers)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    size = 0;
    DT_S32 iwidth = accum.GetSizes().m_width;
    DT_S32 iheight = accum.GetSizes().m_height;
    const DT_S32 *accum_data = accum.Ptr<DT_S32>(0);

    for (DT_S32 y = 1; y < iheight - 1; y++)
    {
        DT_S32 x = 1;
        DT_S32 base = y * iwidth + x;
        for (; x < iwidth - 1; x++, base++)
        {
            if (accum_data[base] > acc_thresh &&
                accum_data[base] > accum_data[base - 1] && accum_data[base] >= accum_data[base + 1] &&
                accum_data[base] > accum_data[base - iwidth] && accum_data[base] >= accum_data[base + iwidth])
            {
                centers[size].index = base;
                centers[size].value = accum_data[base];
                size++;
            }
        }
    }

    return Status::OK;
}

static DT_BOOL HoughAccumParamCmpGt(HoughAccumParam &a, HoughAccumParam &b)
{
    if (a.value > b.value || (a.value == b.value && a.index < b.index))
    {
        return DT_TRUE;
    }
    else
    {
        return DT_FALSE;
    }
}

static DT_BOOL CheckDistance(std::vector<CircleParam> &lst, DT_S32 end_idx, CircleParam &circle, DT_F32 min_dist2)
{
    for (DT_S32 i = 0; i < end_idx; ++i)
    {
        CircleParam pt = lst[i];
        DT_F32 dist_x = circle.x - pt.x;
        DT_F32 dist_y = circle.y - pt.y;
        if (dist_x * dist_x + dist_y * dist_y < min_dist2)
        {
            return DT_FALSE;
        }
    }

    return DT_TRUE;
}

static Status GetCircleCenters(Context *ctx, HoughAccumParam *centers, DT_S32 center_count, std::vector<CircleParam> &lst,
                               DT_S32 w, DT_F32 min_dist, DT_F32 dp)
{
    if (DT_NULL == centers)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    DT_F32 min_dist2 = min_dist * min_dist;

    for (DT_S32 i = 0; i < center_count; i++)
    {
        DT_S32 center = centers[i].index;
        DT_S32 y = center / w;
        DT_S32 x = center - y * w;
        CircleParam circle((x + 0.5f) * dp, (y + 0.5f) * dp, 0, static_cast<DT_F32>(center));

        DT_BOOL good_point = CheckDistance(lst, lst.size(), circle, min_dist2);
        if (good_point)
        {
            lst.emplace_back(circle);
        }
    }

    return Status::OK;
}

static DT_S32 FilterCirclesMat(Point2f &cur_center, DT_F32 *data, const Mat &nz, DT_S32 max_radius, DT_F32 min_radius2, DT_F32 max_radius2)
{
    DT_S32 nz_count = 0;

    const DT_S32 r_outer = max_radius + 1;
    DT_S32 x_start = Max(static_cast<DT_S32>(cur_center.m_x - r_outer), static_cast<DT_S32>(0));
    DT_S32 x_end   = Min(static_cast<DT_S32>(cur_center.m_x + r_outer), nz.GetSizes().m_width);
    DT_S32 y_start = Max(static_cast<DT_S32>(cur_center.m_y - r_outer), static_cast<DT_S32>(0));
    DT_S32 y_end   = Min(static_cast<DT_S32>(cur_center.m_y + r_outer), nz.GetSizes().m_height);

    for (DT_S32 y = y_start; y < y_end; y++)
    {
        const DT_U8 *nz_data = nz.Ptr<DT_U8>(y);
        DT_F32 dy = cur_center.m_y - y;
        DT_F32 dy2 = dy * dy;
        for (DT_S32 x = x_start; x < x_end; x++)
        {
            if (nz_data[x])
            {
                DT_F32 dx = cur_center.m_x - x;
                DT_F32 r2 = dx * dx + dy2;
                if (r2 >= min_radius2 && r2 <= max_radius2)
                {
                    data[nz_count] = r2;
                    nz_count++;
                }
            }
        }
    }

    return nz_count;
}

static Status HoughCirclesEstimateRadius(Context *ctx, const Mat &nz, DT_S32 nz_size, HoughAccumParam *centers,
                                         std::vector<CircleParam> &lst, DT_S32 width, DT_S32 acc_thresh, DT_S32 min_radius,
                                         DT_S32 max_radius, DT_F32 dp, ThreadBuffer &thread_buffer, std::mutex &mutex,
                                         DT_S32 start_count, DT_S32 end_count)
{
    if (DT_NULL == centers)
    {
        AURA_ADD_ERROR_STRING(ctx, "null ptr");
        return Status::ERROR;
    }

    DT_F32 min_radius2 = static_cast<DT_F32>(min_radius * min_radius);
    DT_F32 max_radius2 = static_cast<DT_F32>(max_radius * max_radius);

    const DT_S32 nbins_per_dr = 10;
    DT_S32 nbins = Round((max_radius - min_radius) / dp * nbins_per_dr);

    DT_S32 *buffer    = thread_buffer.GetThreadData<DT_S32>();

    if (!buffer)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    DT_S32 *bins = buffer;
    DT_F32 *dist = reinterpret_cast<DT_F32*>(buffer + nbins);
    DT_F32 *dist_sqrt = dist + nz_size;

    std::vector<CircleParam> lst_tmp;
    lst_tmp.reserve(128);

    for (DT_S32 i = start_count; i < end_count; i++)
    {
        DT_S32 offset = centers[i].index;
        DT_S32 y = offset / width;
        DT_S32 x = offset % width;

        Point2f cur_center = Point2f((x + 0.5f) * dp, (y + 0.5f) * dp);
        DT_S32 nz_count = FilterCirclesMat(cur_center, dist, nz, max_radius, min_radius2, max_radius2);

        DT_S32 max_count = 0;
        DT_F32 r_best = 0;
        if (nz_count)
        {
            for (DT_S32 j = 0; j < nz_count; ++j)
            {
                dist_sqrt[j] = Sqrt(dist[j]);
            }

            memset(bins, 0, nbins * sizeof(DT_S32));
            for (DT_S32 k = 0; k < nz_count; ++k)
            {
                DT_S32 bin = Max(static_cast<DT_S32>(0), Min(nbins - 1, Round((dist_sqrt[k] - min_radius) / dp * nbins_per_dr)));
                bins[bin]++;
            }

            for (DT_S32 j = nbins - 1; j > 0; j--)
            {
                if (bins[j])
                {
                    DT_S32 upbin = j;
                    DT_S32 cur_count = 0;
                    for (; j > upbin - nbins_per_dr && j >= 0; j--)
                    {
                        cur_count += bins[j];
                    }

                    DT_F32 r_cur = (upbin + j) / 2.f / nbins_per_dr * dp + min_radius;
                    if ((cur_count * r_best >= max_count * r_cur) || (r_best < FLT_EPSILON && cur_count >= max_count))
                    {
                        r_best = r_cur;
                        max_count = cur_count;
                    }
                }
            }
        }

        if (max_count > acc_thresh)
        {
            lst_tmp.emplace_back(cur_center.m_x, cur_center.m_y, r_best, static_cast<DT_F32>(max_count));
        }
    }

    std::lock_guard<std::mutex> guard(mutex);
    lst.insert(lst.end(), lst_tmp.begin(), lst_tmp.end());

    return Status::OK;
}

static DT_BOOL CmpAccum(const CircleParam &a, const CircleParam &b)
{
    // Compare everything so the order is completely deterministic
    // Larger accum first
    if (a.accum > b.accum)
    {
        return DT_TRUE;
    }
    else if (a.accum < b.accum)
    {
        return DT_FALSE;
    }
    // Larger radius first
    else if (a.r > b.r)
    {
        return DT_TRUE;
    }
    else if (a.r < b.r)
    {
        return DT_FALSE;
    }
    // Smaller X
    else if (a.x < b.x)
    {
        return DT_TRUE;
    }
    else if (a.x > b.x)
    {
        return DT_FALSE;
    }
    // Smaller Y
    else if (a.y < b.y)
    {
        return DT_TRUE;
    }
    else if (a.y > b.y)
    {
        return DT_FALSE;
    }
    // Identical - neither object is less than the other
    else
    {
        return DT_FALSE;
    }
}

static DT_VOID RemoveOverlaps(std::vector<CircleParam> &lst, DT_F32 min_dist)
{
    DT_F32 min_dist2 = min_dist * min_dist;
    DT_S32 end_idx = 1;

    for (DT_U64 i = 1; i < lst.size(); i++)
    {
        CircleParam circle = lst[i];
        if (CheckDistance(lst, end_idx, circle, min_dist2))
        {
            lst[end_idx++] = circle;
        }
    }

    lst.resize(end_idx);
}

static Status HoughCirclesGradient(Context *ctx, const Mat &dx, const Mat &dy, const Mat &edges, Mat &nz, Mat &accum,
                                   std::vector<Scalar> &circles, DT_F32 dp, DT_F32 min_dist, DT_S32 min_radius, DT_S32 max_radius,
                                   DT_S32 acc_thresh, DT_S32 centers_only, const OpTarget &target)
{
    Status ret = Status::ERROR;

    DT_F32 idp = 1.f / dp;
    DT_S32 nz_size = 0;
    DT_S32 centers_count = 0;

    DT_S32 sizes_h = accum.GetSizes().m_height;
    DT_S32 sizes_w = accum.GetSizes().m_width;
    DT_S32 iwidth  = edges.GetSizes().m_width;
    DT_S32 iheight = edges.GetSizes().m_height;
    HoughAccumParam *centers = static_cast<HoughAccumParam*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, (sizes_h - 2) * (sizes_w - 2) * sizeof(HoughAccumParam), 0));
    if (DT_NULL == centers)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
        return ret;
    }

    std::vector<CircleParam> lst;
    lst.reserve(256);

    std::mutex mutex;

    if (target.m_data.none.enable_mt)
    {
        WorkerPool *wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "null workerpool ptr");
            return ret;
        }

        ThreadBuffer thread_buffer(ctx, sizes_h * sizes_w * sizeof(DT_S32) + iwidth * 16 * sizeof(DT_U8));

        ret = wp->ParallelFor(static_cast<DT_S32>(0), AURA_ALIGN(iheight, 16) / 16, HoughCirclesAccum, ctx,
                              std::cref(edges), std::cref(dx), std::cref(dy), std::ref(accum), std::ref(nz),
                              min_radius, max_radius, idp, &nz_size, std::ref(thread_buffer), std::ref(mutex));
    }
    else
    {
        ThreadBuffer thread_buffer(ctx, sizes_h * sizes_w * sizeof(DT_S32) + iwidth * iheight * sizeof(DT_U8));

        ret = HoughCirclesAccum(ctx, edges, dx, dy, accum, nz, min_radius, max_radius, idp, &nz_size, thread_buffer, mutex,
                                0, AURA_ALIGN(iheight, 16) / 16);
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "HoughCirclesAccumNeon failed");
        goto EXIT;
    }

    if (nz_size <= 0)
    {
        AURA_LOGD(ctx, AURA_TAG, "no circles found");
        ret = Status::OK;
        goto EXIT;
    }

    ret = HoughCirclesFindCenters(ctx, accum, centers, acc_thresh, centers_count);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "HoughCirclesFindCenters failed");
        goto EXIT;
    }

    if (centers_count <= 0)
    {
        AURA_LOGD(ctx, AURA_TAG, "no circles found");
        ret = Status::OK;
        goto EXIT;
    }

    std::sort(centers, centers + centers_count, HoughAccumParamCmpGt);

    if (centers_only)
    {
        ret = GetCircleCenters(ctx, centers, centers_count, lst, sizes_w, min_dist, dp);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetCircleCenters failed");
            goto EXIT;
        }
    }
    else
    {
        const DT_S32 nbins_per_dr = 10;
        DT_S32 nbins = Round((max_radius - min_radius) * idp * nbins_per_dr);
        if (target.m_data.none.enable_mt)
        {
            WorkerPool *wp = ctx->GetWorkerPool();
            if (DT_NULL == wp)
            {
                AURA_ADD_ERROR_STRING(ctx, "null workerpool ptr");
                return ret;
            }

            ThreadBuffer thread_buffer_radius(ctx, nbins * sizeof(DT_S32) + nz_size * 2 * sizeof(DT_F32));

            ret = wp->ParallelFor(static_cast<DT_S32>(0), centers_count, HoughCirclesEstimateRadius, ctx, std::cref(nz), nz_size, centers, std::ref(lst),
                                  sizes_w, acc_thresh, min_radius, max_radius, dp, std::ref(thread_buffer_radius), std::ref(mutex));
        }
        else
        {
            ThreadBuffer thread_buffer_radius(ctx, nbins * sizeof(DT_S32) + nz_size * 2 * sizeof(DT_F32));

            ret = HoughCirclesEstimateRadius(ctx, nz, nz_size, centers, lst, sizes_w, acc_thresh, min_radius, max_radius, dp, thread_buffer_radius, mutex,
                                             static_cast<DT_S32>(0), centers_count);
        }
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "HoughCirclesEstimateRadiusNeon failed");
            goto EXIT;
        }

        std::sort(lst.begin(), lst.end(), CmpAccum);

        RemoveOverlaps(lst, min_dist);
    }

    for (DT_U64 idx = 0; idx < lst.size(); idx++)
    {
        circles.emplace_back(lst[idx].x, lst[idx].y, lst[idx].r, lst[idx].accum);
    }

    ret = Status::OK;

EXIT:
    AURA_FREE(ctx, centers);
    AURA_RETURN(ctx, ret);
}

HoughCirclesNone::HoughCirclesNone(Context *ctx, const OpTarget &target) : HoughCirclesImpl(ctx, target)
{}

Status HoughCirclesNone::SetArgs(const Array *src, std::vector<Scalar> &circles, HoughCirclesMethod method, DT_F64 dp,
                                 DT_F64 min_dist, DT_F64 canny_thresh, DT_F64 acc_thresh, DT_S32 min_radius, DT_S32 max_radius)
{
    if (HoughCirclesImpl::SetArgs(src, circles, method, dp, min_dist, canny_thresh, acc_thresh, min_radius, max_radius) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "HoughCirclesImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status HoughCirclesNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if ((DT_NULL == src))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    m_circles->clear();

    DT_F32 dp_f             = Max(static_cast<DT_F32>(m_dp), 1.f);
    DT_F32 idp              = 1.f / dp_f;
    DT_S32 canny_thresh_s32 = Round(m_canny_thresh);
    DT_S32 kernel_size      = 3;

    Sizes3 sz = src->GetSizes();
    Sizes3 sz_acc;
    sz_acc.m_channel = sz.m_channel;
    sz_acc.m_height  = Ceil(sz.m_height * idp) + 2;
    sz_acc.m_width   = Ceil(sz.m_width * idp) + 2;

    Mat dx(m_ctx, ElemType::S16, sz);
    Mat dy(m_ctx, ElemType::S16, sz);
    Mat edges(m_ctx, ElemType::U8, sz);
    Mat nz(m_ctx, ElemType::U8, sz);
    Mat accum(m_ctx, ElemType::S32, sz_acc);
    if (!(dx.IsValid() && dy.IsValid() && edges.IsValid() && nz.IsValid() && accum.IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat create failed");
        return Status::ERROR;
    }

    ret = ISobel(m_ctx, *src, dx, 1, 0, kernel_size, 1.f, BorderType::REPLICATE, {Scalar()}, m_target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "sobel failed");
        return Status::ERROR;
    }

    ret = ISobel(m_ctx, *src, dy, 0, 1, kernel_size, 1.f, BorderType::REPLICATE, {Scalar()}, m_target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "sobel failed");
        return Status::ERROR;
    }

    ret = ICanny(m_ctx, dx, dy, edges, Max((DT_S32)1, canny_thresh_s32 / 2), canny_thresh_s32, 0, m_target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Canny failed");
        return Status::ERROR;
    }

    DT_S32 width  = src->GetSizes().m_width;
    DT_S32 height = src->GetSizes().m_height;

    DT_S32 acc_thresh_s32 = Round(m_acc_thresh);
    m_min_radius = Max(m_min_radius, static_cast<DT_S32>(0));
    DT_S32 centers_only = (m_max_radius < 0);

    if (m_max_radius <= 0)
    {
        m_max_radius = Max(height, width);
    }
    else if (m_max_radius <= m_min_radius)
    {
        m_max_radius = m_min_radius + 2;
    }

    DT_U8 *nz_data = nz.Ptr<DT_U8>(0);
    DT_S32 *accum_data = accum.Ptr<DT_S32>(0);
    memset(nz_data, 0, nz.GetTotalBytes());
    memset(accum_data, 0, accum.GetTotalBytes());

    switch(m_method)
    {
        case HoughCirclesMethod::HOUGH_GRADIENT:
        {
            ret = HoughCirclesGradient(m_ctx, dx, dy, edges, nz, accum, *m_circles, dp_f, m_min_dist, m_min_radius, m_max_radius,
                                       acc_thresh_s32, centers_only, m_target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "hough circles method not supported");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura