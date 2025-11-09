#include "min_max_loc_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
static Status MinMaxLocNoneImpl(Context *ctx, const Mat &mat, std::vector<DT_F64> &min_val, std::vector<DT_F64> &max_val,
                                std::vector<Point3i> &min_pos, std::vector<Point3i> &max_pos, DT_S32 start_row, DT_S32 end_row)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    DT_S32 thread_idx = wp ? wp->GetComputeThreadIdx() : 0;

    Tp min_value = mat.At<Tp>(0, 0, 0);
    Tp max_value = mat.At<Tp>(0, 0, 0);

    Sizes3 sz = mat.GetSizes();

    DT_S32 idx_min_x = 0;
    DT_S32 idx_min_y = 0;
    DT_S32 idx_max_x = 0;
    DT_S32 idx_max_y = 0;

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const Tp *src_c = mat.Ptr<Tp>(y);

        for (DT_S32 x = 0; x < sz.m_width * sz.m_channel; ++x)
        {
            Tp value = src_c[x];

            if (value < min_value)
            {
                min_value = value;
                idx_min_x = x;
                idx_min_y = y;
            }
            else if (value > max_value)
            {
                max_value = value;
                idx_max_x = x;
                idx_max_y = y;
            }
        }
    }

    if (min_val[thread_idx] >  min_value)
    {
        min_pos[thread_idx] = Point3i(idx_min_x / sz.m_channel, idx_min_y, idx_min_x % sz.m_channel);
        min_val[thread_idx] = min_value;
    }

    if (max_val[thread_idx] < max_value)
    {
        max_val[thread_idx] = max_value;
        max_pos[thread_idx] = Point3i(idx_max_x / sz.m_channel, idx_max_y, idx_max_x % sz.m_channel);
    }

    return Status::OK;
}

MinMaxLocNone::MinMaxLocNone(Context *ctx, const OpTarget &target) : MinMaxLocImpl(ctx, target)
{}

Status MinMaxLocNone::SetArgs(const Array *src, DT_F64 *min_val, DT_F64 *max_val, Point3i *min_pos, Point3i *max_pos)
{
    if (MinMaxLocImpl::SetArgs(src, min_val, max_val, min_pos, max_pos) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MinMaxLocImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

#define MINMAXLOC_NONE_IMPL(type)                                                                                                       \
    if (m_target.m_data.none.enable_mt)                                                                                                 \
    {                                                                                                                                   \
        WorkerPool *wp = m_ctx->GetWorkerPool();                                                                                        \
        if (DT_NULL == wp)                                                                                                              \
        {                                                                                                                               \
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");                                                                       \
            return Status::ERROR;                                                                                                       \
        }                                                                                                                               \
        ret = wp->ParallelFor(static_cast<DT_S32>(0), height, MinMaxLocNoneImpl<type>, m_ctx, std::cref(*src), std::ref(min_val),       \
                              std::ref(max_val), std::ref(min_pos), std::ref(max_pos));                                                 \
    }                                                                                                                                   \
    else                                                                                                                                \
    {                                                                                                                                   \
        ret = MinMaxLocNoneImpl<type>(m_ctx, *src, min_val, max_val, min_pos, max_pos,                                                  \
                                      static_cast<DT_S32>(0), height);                                                                  \
    }                                                                                                                                   \
                                                                                                                                        \
    if (ret != Status::OK)                                                                                                              \
    {                                                                                                                                   \
        DT_CHAR error_msg[128];                                                                                                         \
        std::snprintf(error_msg, sizeof(error_msg), "MinMaxLocNoneImpl failed (type %s)", #type);                                       \
        AURA_ADD_ERROR_STRING(m_ctx, error_msg);                                                                                        \
        goto EXIT;                                                                                                                      \
    }

Status MinMaxLocNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    DT_S32 height     = src->GetSizes().m_height;
    DT_S32 thread_num = (m_target.m_data.none.enable_mt && m_ctx->GetWorkerPool()) ? 
                        m_ctx->GetWorkerPool()->GetComputeThreadNum() : 1;

    std::vector<DT_F64>  min_val(thread_num, std::numeric_limits<DT_F64>::max());
    std::vector<DT_F64>  max_val(thread_num, std::numeric_limits<DT_F64>::lowest());
    std::vector<Point3i> min_pos(thread_num, {0, 0, 0});
    std::vector<Point3i> max_pos(thread_num, {0, 0, 0});

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            MINMAXLOC_NONE_IMPL(DT_U8);
            break;
        }
        case ElemType::S8:
        {
            MINMAXLOC_NONE_IMPL(DT_S8);
            break;
        }
        case ElemType::U16:
        {
            MINMAXLOC_NONE_IMPL(DT_U16);
            break;
        }
        case ElemType::S16:
        {
            MINMAXLOC_NONE_IMPL(DT_S16);
            break;
        }
        case ElemType::U32:
        {
            MINMAXLOC_NONE_IMPL(DT_U32);
            break;
        }
        case ElemType::S32:
        {
            MINMAXLOC_NONE_IMPL(DT_S32);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            MINMAXLOC_NONE_IMPL(MI_F16);
            break;
        }
        case ElemType::F32:
        {
            MINMAXLOC_NONE_IMPL(DT_F32);
            break;
        }
#endif
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "input mat with invalid ElemType.");
            ret = Status::ERROR;
            break;
        }
    }

    *m_min_val = min_val[0];
    *m_min_pos = min_pos[0];
    *m_max_val = max_val[0];
    *m_max_pos = max_pos[0];

    for (DT_U32 i = 1; i < min_val.size(); i++)
    {
        if (*m_min_val > min_val[i])
        {
            *m_min_val = min_val[i];
            *m_min_pos = min_pos[i];
        }
        else if (*m_min_val == min_val[i])
        {
            if (min_pos[i].m_y < (*m_min_pos).m_y)
            {
                *m_min_pos = min_pos[i];
            }
            else if ((min_pos[i].m_y == (*m_min_pos).m_y) && (min_pos[i].m_x < (*m_min_pos).m_x))
            {
                *m_min_pos = min_pos[i];
            }
        }

        if (*m_max_val < max_val[i])
        {
            *m_max_val = max_val[i];
            *m_max_pos = max_pos[i];
        }
        else if (*m_max_val == max_val[i])
        {
            if (max_pos[i].m_y < (*m_max_pos).m_y)
            {
                *m_max_pos = max_pos[i];
            }
            else if ((max_pos[i].m_y == (*m_min_pos).m_y) && (min_pos[i].m_x < (*m_min_pos).m_x))
            {
                *m_max_pos = max_pos[i];
            }
        }
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

} // namespace aura