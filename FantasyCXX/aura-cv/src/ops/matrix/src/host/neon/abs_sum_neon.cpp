#include "norm_impl.hpp"
#include "aura/ops/matrix/sum.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

#define SUM_BLK   (8)

namespace aura
{

static Status AbsSumS8NeonImpl(Context *ctx, const Mat &mat, std::vector<MI_F64> &task_result, MI_S32 start_blk, MI_S32 end_blk)
{
    AURA_UNUSED(ctx);
    constexpr MI_S32 block_size = (1 << 8);
    constexpr MI_S32 block_step = block_size * 16;

    Sizes3 sz             = mat.GetSizes();
    MI_S32 width          = sz.m_width;
    MI_S32 channel        = sz.m_channel;
    MI_S32 row_elem_count = width * channel;
    MI_S32 start_row      = start_blk * SUM_BLK;
    MI_S32 end_row        = Min(end_blk * SUM_BLK, sz.m_height);

    MI_F64 result = 0.0;
    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const MI_S8 *src_row = mat.Ptr<MI_S8>(y);
        uint32x4_t vqu32_row_sum;
        neon::vdup(vqu32_row_sum, 0);
        MI_S32 bx = 0;
        for (MI_S32 x = 0; x < row_elem_count; x += block_step)
        {
            MI_S32 blk_len = Min(row_elem_count - x, block_step);
            MI_S32 blk_len_align16 = blk_len & (-16);
            uint16x8_t vqu16_block_sum;
            neon::vdup(vqu16_block_sum, 0);
            for (bx = x; bx < x + blk_len_align16; bx += 16)
            {
                int8x16_t vqs8_src_data;
                neon::vload(src_row + bx, vqs8_src_data);
                uint8x16_t vqu8_vec_abs = neon::vreinterpret(neon::vabs(vqs8_src_data));
                vqu16_block_sum = neon::vpadal(vqu16_block_sum, vqu8_vec_abs);
            }
            vqu32_row_sum = neon::vpadal(vqu32_row_sum, vqu16_block_sum);
        }

        for (; bx < row_elem_count; bx++)
        {
            result += Abs(static_cast<MI_F64>(src_row[bx]));
        }
        result += neon::vgetlane<0>(vqu32_row_sum);
        result += neon::vgetlane<1>(vqu32_row_sum);
        result += neon::vgetlane<2>(vqu32_row_sum);
        result += neon::vgetlane<3>(vqu32_row_sum);
    }

    task_result[start_blk] = result;

    return Status::OK;
}

static Status AbsSumS16NeonImpl(Context *ctx, const Mat &mat, std::vector<MI_F64> &task_result, MI_S32 start_blk, MI_S32 end_blk)
{
    AURA_UNUSED(ctx);
    constexpr MI_S32 block_size = (1 << 16);
    constexpr MI_S32 block_step = block_size * 8;

    Sizes3 sz             = mat.GetSizes();
    MI_S32 width          = sz.m_width;
    MI_S32 channel        = sz.m_channel;
    MI_S32 row_elem_count = width * channel;
    MI_S32 start_row      = start_blk * SUM_BLK;
    MI_S32 end_row        = Min(end_blk * SUM_BLK, sz.m_height);

    MI_F64 result = 0.0;
    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const MI_S16 *src_row = mat.Ptr<MI_S16>(y);
        uint64x2_t vqu64_row_sum;
        neon::vdup(vqu64_row_sum, 0);
        MI_S32 bx = 0;
        for (MI_S32 x = 0; x < row_elem_count; x += block_step)
        {
            MI_S32 blk_len = Min(row_elem_count - x, block_step);
            MI_S32 blk_len_align8 = blk_len & (-8);
            uint32x4_t vqu32_block_sum;
            neon::vdup(vqu32_block_sum, 0);
            for (bx = x; bx < x + blk_len_align8; bx += 8)
            {
                int16x8_t vqs16_src_data;
                neon::vload(src_row + bx, vqs16_src_data);
                uint16x8_t vqu16_vec_abs = neon::vreinterpret(neon::vabs(vqs16_src_data));
                vqu32_block_sum = neon::vpadal(vqu32_block_sum, vqu16_vec_abs);
            }
            vqu64_row_sum = neon::vpadal(vqu64_row_sum, vqu32_block_sum);
            for (; bx < row_elem_count; bx++)
            {
                result += Abs(static_cast<MI_F64>(src_row[bx]));
            }
            result += static_cast<MI_F64>(neon::vgetlane<0>(vqu64_row_sum));
            result += static_cast<MI_F64>(neon::vgetlane<1>(vqu64_row_sum));
        }
    }

    task_result[start_blk] = result;

    return Status::OK;
}

#if defined(AURA_ENABLE_NEON_FP16)
static Status AbsSumF16NeonImpl(Context *ctx, const Mat &mat, std::vector<MI_F64> &task_result, MI_S32 start_blk, MI_S32 end_blk)
{
    AURA_UNUSED(ctx);
    Sizes3 sz                = mat.GetSizes();
    MI_S32 width             = sz.m_width;
    MI_S32 channel           = sz.m_channel;
    MI_S32 row_elem_count    = width * channel;
    MI_S32 elem_count_align8 = row_elem_count & (-8);
    MI_S32 start_row         = start_blk * SUM_BLK;
    MI_S32 end_row           = Min(end_blk * SUM_BLK, sz.m_height);

    MI_F64 result = 0.0;
    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const MI_F16 *src_row = mat.Ptr<MI_F16>(y);
        MI_S32 x = 0;
        float32x4_t vqf32_row_sum;
        neon::vdup(vqf32_row_sum, 0);
        for (; x < elem_count_align8; x += 8)
        {
            float16x8_t vqf16_src_data;
            neon::vload(src_row + x, vqf16_src_data);
            float32x4_t vqf32_hi_abs = neon::vabs(neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_data)));
            float32x4_t vqf32_lo_abs = neon::vabs(neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_data)));
            vqf32_row_sum = neon::vadd(vqf32_row_sum, vqf32_hi_abs);
            vqf32_row_sum = neon::vadd(vqf32_row_sum, vqf32_lo_abs);
        }
        result += neon::vgetlane<0>(vqf32_row_sum);
        result += neon::vgetlane<1>(vqf32_row_sum);
        result += neon::vgetlane<2>(vqf32_row_sum);
        result += neon::vgetlane<3>(vqf32_row_sum);

        for (; x < row_elem_count; ++x)
        {
            result += Abs(static_cast<MI_F64>(src_row[x]));
        }
    }

    task_result[start_blk] = result;

    return Status::OK;
}
#endif

static Status AbsSumF32NeonImpl(Context *ctx, const Mat &mat, std::vector<MI_F64> &task_result, MI_S32 start_blk, MI_S32 end_blk)
{
    AURA_UNUSED(ctx);
    Sizes3 sz                = mat.GetSizes();
    MI_S32 width             = sz.m_width;
    MI_S32 channel           = sz.m_channel;
    MI_S32 row_elem_count    = width * channel;
    MI_S32 elem_count_align4 = row_elem_count & (-4);
    MI_S32 start_row         = start_blk * SUM_BLK;
    MI_S32 end_row           = Min(end_blk * SUM_BLK, sz.m_height);

    MI_F64 result = 0.0;
    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const MI_F32 *src_row = mat.Ptr<MI_F32>(y);
        MI_S32 x = 0;
        float32x4_t vqf32_row_sum;
        neon::vdup(vqf32_row_sum, 0);
        for (; x < elem_count_align4; x += 4)
        {
            float32x4_t vqf32_src_data;
            neon::vload(src_row + x, vqf32_src_data);
            vqf32_row_sum = neon::vadd(vqf32_row_sum, neon::vabs(vqf32_src_data));
        }
        result += neon::vgetlane<0>(vqf32_row_sum);
        result += neon::vgetlane<1>(vqf32_row_sum);
        result += neon::vgetlane<2>(vqf32_row_sum);
        result += neon::vgetlane<3>(vqf32_row_sum);

        for (; x < row_elem_count; ++x)
        {
            result += Abs(src_row[x]);
        }
    }

    task_result[start_blk] = result;

    return Status::OK;
}

static Status AbsSumNeonHelper(Context *ctx, const Mat &mat, MI_F64 &result, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::ERROR;

    Sizes3 sz        = mat.GetSizes();
    MI_S32 height    = sz.m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get worker_pool failed.");
        return Status::ERROR;
    }

    MI_S32 task_nums = (height + SUM_BLK - 1) / SUM_BLK;
    std::vector<MI_F64> task_result(task_nums, 0.0);

    switch (mat.GetElemType())
    {
        case ElemType::S8:
        {
            ret = wp->ParallelFor(0, task_nums, AbsSumS8NeonImpl, ctx, std::ref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor AbsSumS8NeonImpl failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = wp->ParallelFor(0, task_nums, AbsSumS16NeonImpl, ctx, std::ref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor AbsSumS16NeonImpl failed.");
            }
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = wp->ParallelFor(0, task_nums, AbsSumF16NeonImpl, ctx, std::ref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor AbsSumF16NeonImpl failed.");
            }
            break;
        }
#endif
        case ElemType::F32:
        {
            ret = wp->ParallelFor(0, task_nums, AbsSumF32NeonImpl, ctx, std::ref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor AbsSumF32NeonImpl failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported elem_type.");
            ret = Status::ERROR;
        }
    }

    result = 0.0;
    for (const auto val : task_result)
    {
        result += val;
    }

    AURA_RETURN(ctx, ret);
}

Status AbsSumNeon(Context *ctx, const Mat &mat, MI_F64 &result, const OpTarget &target)
{
    Status ret = Status::ERROR;

    ElemType type = mat.GetElemType();

    if (ElemType::U8 == type || ElemType::U16 == type)
    {
        Scalar scalar_result = Scalar::All(0.0);
        ret = ISum(ctx, mat, scalar_result, target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "AbsSumNeon call SumNeon failed.");
            return Status::ERROR;
        }
        result = scalar_result.m_val[0] +  scalar_result.m_val[1] + scalar_result.m_val[2] + scalar_result.m_val[3];
    }
    else
    {
        ret = AbsSumNeonHelper(ctx, mat, result, target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "AbsSumNeonHelper failed.");
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
