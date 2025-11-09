#include "min_max_loc_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

constexpr static DT_S32 g_block_size = 16384;

template <typename Tp>
struct MinMaxLocInfo
{
    Tp min;
    Tp max;
    DT_S32 idx_min_x;
    DT_S32 idx_min_y;
    DT_S32 idx_max_x;
    DT_S32 idx_max_y;
};

template <typename Tp>
static Status MinMaxLocNeonImpl(const Mat &mat, std::vector<MinMaxLocInfo<Tp>> &result, DT_S32 start_row, DT_S32 end_row)
{
    using VType = typename neon::QVector<Tp>::VType;  // vector type of Tp (128 bits)
    constexpr DT_S32 VEC_SIZE = 16 / sizeof(Tp);      // vector length, can be 16/8/4 according to different element type

    const DT_S32 width   = mat.GetSizes().m_width;
    const DT_S32 channel = mat.GetSizes().m_channel;
    const DT_S32 num_per_row = width * channel;

    Tp min = std::numeric_limits<Tp>::max();    // min value from start row to end row
    Tp max = std::numeric_limits<Tp>::lowest(); // max value from start row to end row
    DT_S32 idx_min_x = 0; // block idx in x-axis of min value
    DT_S32 idx_min_y = 0; // row idx for min value
    DT_S32 idx_max_x = 0; // block idx in x-axis of max value
    DT_S32 idx_max_y = 0; // row idx for max value

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = mat.Ptr<Tp>(y);

        for (DT_S32 x = 0; x < num_per_row; x += g_block_size)
        {
            // init
            VType vq_min;
            neon::vdup(vq_min, std::numeric_limits<Tp>::max()); // init min val vector
            VType vq_max;
            neon::vdup(vq_max, std::numeric_limits<Tp>::lowest()); // init max val vector

            const DT_S32 x_size = Min(num_per_row - x, g_block_size);
            const DT_S32 x_size_align = x_size & (-VEC_SIZE);
            DT_S32 ix = 0;
            for (; ix < x_size_align; ix += VEC_SIZE)
            {
                VType vq_cur = neon::vload1q(src_row + x + ix);
                vq_min = neon::vmin(vq_cur, vq_min);
                vq_max = neon::vmax(vq_cur, vq_max);
            } // ix loop

            // handle leftover
            if (x_size_align != x_size)
            {
                VType vq_cur = neon::vload1q(src_row + x_size - VEC_SIZE);
                vq_min = neon::vmin(vq_cur, vq_min);
                vq_max = neon::vmax(vq_cur, vq_max);
            }

            Tp arr_min_val[VEC_SIZE];
            Tp arr_max_val[VEC_SIZE];
            neon::vstore(arr_min_val, vq_min);
            neon::vstore(arr_max_val, vq_max);

            Tp min_block = arr_min_val[0];
            Tp max_block = arr_max_val[0];

            for (DT_S32 i = 1; i < VEC_SIZE; i++)
            {
                if (arr_min_val[i] < min_block)
                {
                    min_block = arr_min_val[i];
                }
            }
            for (DT_S32 i = 1; i < VEC_SIZE; i++)
            {
                if (arr_max_val[i] > max_block)
                {
                    max_block = arr_max_val[i];
                }
            }

            // Locate Min/Max
            {
                // update global min and max
                if (min_block < min)
                {
                    min = min_block;
                    idx_min_y = y;
                    idx_min_x = x / g_block_size;
                }

                if (max_block > max)
                {
                    max = max_block;
                    idx_max_y = y;
                    idx_max_x = x / g_block_size;
                }
            }

        } // x loop
    } // y loop

    // store result
    DT_S32 idx = start_row;
    result[idx].min = min;
    result[idx].max = max;
    result[idx].idx_min_x = idx_min_x;
    result[idx].idx_min_y = idx_min_y;
    result[idx].idx_max_x = idx_max_x;
    result[idx].idx_max_y = idx_max_y;

    return Status::OK;
}

template <typename Tp>
static Status MinMaxLocNeonHelper(Context *ctx, const Mat &mat, DT_F64 *min_val, DT_F64 *max_val,
                                  Point3i *min_pos, Point3i *max_pos, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    const DT_S32 height  = mat.GetSizes().m_height;
    const DT_S32 width   = mat.GetSizes().m_width;
    const DT_S32 channel = mat.GetSizes().m_channel;
    const DT_S32 num_per_row = width * channel;

    std::vector<MinMaxLocInfo<Tp>> result(height);

    Status ret = wp->ParallelFor(0, height, MinMaxLocNeonImpl<Tp>, std::cref(mat), std::ref(result));
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ParallelFor failed");
        return Status::ERROR;
    }

    // init
    Tp min = result[0].min;
    Tp max = result[0].max;
    DT_S32 idx_min_x = result[0].idx_min_x;
    DT_S32 idx_min_y = result[0].idx_min_y;
    DT_S32 idx_max_x = result[0].idx_max_x;
    DT_S32 idx_max_y = result[0].idx_max_y;

    for (DT_S32 i = 1; i < height; i++)
    {
        if (result[i].min < min)
        {
            min = result[i].min;
            idx_min_x = result[i].idx_min_x;
            idx_min_y = result[i].idx_min_y;
        }
        if (result[i].max > max)
        {
            max = result[i].max;
            idx_max_x = result[i].idx_max_x;
            idx_max_y = result[i].idx_max_y;
        }
    }

    // firstly locate min value
    const Tp *min_data_row = mat.Ptr<Tp>(idx_min_y);
    DT_S32 min_pos_x = -1;
    DT_S32 pos_min_end = Min((idx_min_x + 1) * g_block_size, num_per_row);
    for (DT_S32 i = idx_min_x * g_block_size; i < pos_min_end; i++)
    {
        if (min_data_row[i] == min)
        {
            min_pos_x = i;
            break;
        }
    }

    if (min_pos_x < 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "something is wrong in locating min value.");
        return Status::ERROR;
    }

    // secondly locate max value
    const Tp *max_data_row = mat.Ptr<Tp>(idx_max_y);
    DT_S32 max_pos_x = -1;
    DT_S32 pos_max_end = Min((idx_max_x + 1) * g_block_size, num_per_row);
    for (DT_S32 i = idx_max_x * g_block_size; i < pos_max_end; i++)
    {
        if (max_data_row[i] == max)
        {
            max_pos_x = i;
            break;
        }
    }

    if (max_pos_x < 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "something is wrong in locating max value");
        return Status::ERROR;
    }

    *min_val = static_cast<DT_F64>(min);
    *max_val = static_cast<DT_F64>(max);
    *min_pos = Point3i(min_pos_x / channel, idx_min_y, min_pos_x % channel);
    *max_pos = Point3i(max_pos_x / channel, idx_max_y, max_pos_x % channel);

    return Status::OK;
}

MinMaxLocNeon::MinMaxLocNeon(Context *ctx, const OpTarget &target) : MinMaxLocImpl(ctx, target)
{}

Status MinMaxLocNeon::SetArgs(const Array *src, DT_F64 *min_val, DT_F64 *max_val, Point3i *min_pos, Point3i *max_pos)
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

Status MinMaxLocNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::OK;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = MinMaxLocNeonHelper<DT_U8>(m_ctx, *src, m_min_val, m_max_val, m_min_pos, m_max_pos, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MinMaxLocNeonHelper<DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = MinMaxLocNeonHelper<DT_S8>(m_ctx, *src, m_min_val, m_max_val, m_min_pos, m_max_pos, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MinMaxLocNeonHelper<DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = MinMaxLocNeonHelper<DT_U16>(m_ctx, *src, m_min_val, m_max_val, m_min_pos, m_max_pos, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MinMaxLocNeonHelper<DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = MinMaxLocNeonHelper<DT_S16>(m_ctx, *src, m_min_val, m_max_val, m_min_pos, m_max_pos, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MinMaxLocNeonHelper<DT_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = MinMaxLocNeonHelper<DT_U32>(m_ctx, *src, m_min_val, m_max_val, m_min_pos, m_max_pos, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MinMaxLocNeonHelper<DT_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = MinMaxLocNeonHelper<DT_S32>(m_ctx, *src, m_min_val, m_max_val, m_min_pos, m_max_pos, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MinMaxLocNeonHelper<DT_S32> failed.");
            }
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = MinMaxLocNeonHelper<MI_F16>(m_ctx, *src, m_min_val, m_max_val, m_min_pos, m_max_pos, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MinMaxLocNeonHelper<MI_F16> failed.");
            }
            break;
        }
#endif
        case ElemType::F32:
        {
            ret = MinMaxLocNeonHelper<DT_F32>(m_ctx, *src, m_min_val, m_max_val, m_min_pos, m_max_pos, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "MinMaxLocNeonHelper<DT_F32> failed.");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "input mat with invalid ElemType.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura