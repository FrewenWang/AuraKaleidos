#include "norm_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

#define SUM_BLK   (8)

namespace aura
{

template <typename Tp0, typename Tp1, typename Tp2, MI_S32 C, typename = typename std::enable_if<C <= 3, Tp0>::type>
static AURA_VOID SqSumBorderImpl(const Tp0 *src_row, Tp1 *row_result, MI_S32 cur_x, MI_S32 row_elem_count, Scalar &result)
{
    for (; cur_x < row_elem_count; cur_x += C)
    {
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            row_result[ch] += static_cast<Tp2>(src_row[cur_x + ch]) * src_row[cur_x + ch];
        }
    }

    if (1 == C)
    {
        result.m_val[0] += row_result[0] + row_result[1] + row_result[2] + row_result[3];
    }
    else if (2 == C)
    {
        result.m_val[0] += row_result[0] + row_result[2];
        result.m_val[1] += row_result[1] + row_result[3];
    }
    else
    {
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            result.m_val[ch] += row_result[ch];
        }
    }
}

template <typename Tp, MI_S32 C>
struct SqSumNeonFunctor
{
    Status operator()(Context *ctx, const Mat &mat, std::vector<Scalar> &task_result, MI_S32 start_blk, MI_S32 end_blk)
    {
        AURA_UNUSED(ctx);
        using PromType0 = typename Promote<Tp>::Type;
        using PromType1 = typename Promote<PromType0>::Type;

        constexpr MI_S32 load_size  = 16 / sizeof(Tp);
        constexpr MI_S32 block_size = 1 << 15;
        constexpr MI_S32 block_step = load_size * block_size;
        constexpr MI_S32 block_vec_size = 4 / sizeof(Tp);

        using VType   = typename neon::QVector<Tp>::VType;
        using WVType  = typename neon::MQVector<PromType0, 2>::MVType;
        using SumType = typename neon::MQVector<PromType1, 2>::MVType;

        Sizes3 sz             = mat.GetSizes();
        MI_S32 width          = sz.m_width;
        MI_S32 row_elem_count = width * C;
        MI_S32 start_row      = start_blk * SUM_BLK;
        MI_S32 end_row        = Min(end_blk * SUM_BLK, sz.m_height);

        Scalar result = Scalar::All(0.0);
        for (MI_S32 y = start_row; y < end_row; ++y)
        {
            auto src_row = mat.Ptr<Tp>(y);

            MI_U64 row_result[4] = {0};
            MI_S32 x = 0;
            MI_S32 bx = 0;
            for (; x < row_elem_count; x += block_step)
            {
                MI_S32 blk_len = Min(row_elem_count - x, block_step);
                MI_S32 blk_len_align = blk_len & (-load_size);
                SumType block_sum;
                neon::vdup(block_sum.val[0], 0);
                neon::vdup(block_sum.val[1], 0);
                for (bx = x; bx < x + blk_len_align; bx += load_size)
                {
                    VType vq_src_data;
                    neon::vload(src_row + bx, vq_src_data);
                    WVType v2q_src_widen;
                    v2q_src_widen.val[0] = neon::vmull(neon::vgetlow(vq_src_data), neon::vgetlow(vq_src_data));
                    v2q_src_widen.val[1] = neon::vmull(neon::vgethigh(vq_src_data), neon::vgethigh(vq_src_data));
                    block_sum.val[0] = neon::vaddw(block_sum.val[0], neon::vgetlow(v2q_src_widen.val[0]));
                    block_sum.val[1] = neon::vaddw(block_sum.val[1], neon::vgethigh(v2q_src_widen.val[0]));
                    block_sum.val[0] = neon::vaddw(block_sum.val[0], neon::vgetlow(v2q_src_widen.val[1]));
                    block_sum.val[1] = neon::vaddw(block_sum.val[1], neon::vgethigh(v2q_src_widen.val[1]));
                }
                {
                    PromType1 block_result[block_vec_size * 2] = {0};
                    neon::vstore(block_result, block_sum.val[0]);
                    neon::vstore(block_result + block_vec_size, block_sum.val[1]);

                    for (MI_S32 i = 0; i < block_vec_size * 2; i += 4)
                    {
                        row_result[0] += block_result[i];
                        row_result[1] += block_result[i + 1];
                        row_result[2] += block_result[i + 2];
                        row_result[3] += block_result[i + 3];
                    }
                }
            }
            SqSumBorderImpl<Tp, MI_U64, PromType0, C>(src_row, row_result, bx, row_elem_count, result);
        }

        task_result[start_blk] = result;
        return Status::OK;
    }
};

template <typename Tp>
struct SqSumNeonFunctor<Tp, 3>
{
    Status operator()(Context *ctx, const Mat &mat, std::vector<Scalar> &task_result, MI_S32 start_blk, MI_S32 end_blk)
    {
        AURA_UNUSED(ctx);
        using PromType0 = typename Promote<Tp>::Type;
        using PromType1 = typename Promote<PromType0>::Type;

        constexpr MI_S32 load_size = 16 / sizeof(Tp) * 3;
        constexpr MI_S32 block_size = (1 << 11);
        constexpr MI_S32 block_step = load_size * block_size;
        constexpr MI_S32 block_vec_size = 4 / sizeof(Tp);

        using VType   = typename neon::MQVector<Tp, 3>::MVType;
        using WVType  = typename neon::MQVector<PromType0, 2>::MVType;
        using SumType = typename neon::MQVector<PromType1, 2>::MVType;

        Sizes3 sz             = mat.GetSizes();
        MI_S32 width          = sz.m_width;
        MI_S32 channel        = sz.m_channel;
        MI_S32 row_elem_count = width * channel;
        MI_S32 start_row      = start_blk * SUM_BLK;
        MI_S32 end_row        = Min(end_blk * SUM_BLK, sz.m_height);

        Scalar result = Scalar::All(0.0);
        for (MI_S32 y = start_row; y < end_row; ++y)
        {
            auto src_row = mat.Ptr<Tp>(y);

            MI_U64 row_result[3] = {0};
            SumType vzero;
            neon::vdup(vzero.val[0], 0);
            neon::vdup(vzero.val[1], 0);

            MI_S32 x = 0;
            MI_S32 bx = 0;
            for (; x < row_elem_count; x += block_step)
            {
                MI_S32 blk_len = Min(width - x, block_step);
                MI_S32 blk_len_align = blk_len & (-load_size);
                SumType block_sum[3] = {vzero, vzero, vzero};
                for (bx = x; bx < x + blk_len_align; bx += load_size)
                {
                    VType v3q_src_data;
                    neon::vload(src_row + bx, v3q_src_data);
                    for (MI_S32 i = 0; i < 3; ++i)
                    {
                        WVType v2q_src_widen;
                        v2q_src_widen.val[0] = neon::vmull(neon::vgetlow(v3q_src_data.val[i]), neon::vgetlow(v3q_src_data.val[i]));
                        v2q_src_widen.val[1] = neon::vmull(neon::vgethigh(v3q_src_data.val[i]), neon::vgethigh(v3q_src_data.val[i]));

                        block_sum[i].val[0] = neon::vaddw(block_sum[i].val[0], neon::vgetlow(v2q_src_widen.val[0]));
                        block_sum[i].val[0] = neon::vaddw(block_sum[i].val[0], neon::vgethigh(v2q_src_widen.val[0]));
                        block_sum[i].val[1] = neon::vaddw(block_sum[i].val[1], neon::vgetlow(v2q_src_widen.val[1]));
                        block_sum[i].val[1] = neon::vaddw(block_sum[i].val[1], neon::vgethigh(v2q_src_widen.val[1]));
                    }
                }
                for (MI_S32 i = 0; i < 3; ++i)
                {
                    PromType1 block_result[block_vec_size * 2] = {0};
                    neon::vstore(block_result, block_sum[i].val[0]);
                    neon::vstore(block_result + block_vec_size, block_sum[i].val[1]);

                    for (MI_S32 j = 0; j < block_vec_size * 2; j++)
                    {
                        row_result[i] += block_result[j];
                    }
                }
            }
            SqSumBorderImpl<Tp, MI_U64, PromType1, 3>(src_row, row_result, bx, row_elem_count, result);
        }

        task_result[start_blk] = result;
        return Status::OK;
    }
};

template <MI_S32 C>
struct SqSumNeonFunctor<MI_F32, C>
{
    Status operator()(Context *ctx, const Mat &mat, std::vector<Scalar> &task_result, MI_S32 start_blk, MI_S32 end_blk)
    {
        AURA_UNUSED(ctx);
        Sizes3 sz                = mat.GetSizes();
        MI_S32 width             = sz.m_width;
        MI_S32 row_elem_count    = width * C;
        MI_S32 elem_count_align4 = row_elem_count & (-4);
        MI_S32 start_row         = start_blk * SUM_BLK;
        MI_S32 end_row           = Min(end_blk * SUM_BLK, sz.m_height);

        Scalar result = Scalar::All(0.0);
        for (MI_S32 y = start_row; y < end_row; ++y)
        {
            const MI_F32 *src_row = mat.Ptr<MI_F32>(y);
            MI_S32 x = 0;
            float32x4_t vqf32_row_sum;
            neon::vdup(vqf32_row_sum, 0);
            MI_F32 row_result[4] = {0.0f};
            for (; x < elem_count_align4; x +=4)
            {
                float32x4_t vqf32_src_data;
                neon::vload(src_row + x, vqf32_src_data);
                vqf32_row_sum = neon::vmla(vqf32_row_sum, vqf32_src_data, vqf32_src_data);
            }
            neon::vstore(row_result, vqf32_row_sum);
            SqSumBorderImpl<MI_F32, MI_F32, MI_F32, C>(src_row, row_result, x, row_elem_count, result);
        }

        task_result[start_blk] = result;

        return Status::OK;
    }
};

template <>
struct SqSumNeonFunctor<MI_F32, 3>
{
    Status operator()(Context *ctx, const Mat &mat, std::vector<Scalar> &task_result, MI_S32 start_blk, MI_S32 end_blk)
    {
        AURA_UNUSED(ctx);
        Sizes3 sz                 = mat.GetSizes();
        MI_S32 width              = sz.m_width;
        MI_S32 width_align4       = width & (-4);
        MI_S32 row_elem_count     = width * 3;
        MI_S32 elem_count_align12 = width_align4 * 3;
        MI_S32 start_row          = start_blk * SUM_BLK;
        MI_S32 end_row            = Min(end_blk * SUM_BLK, sz.m_height);

        Scalar result = Scalar::All(0.0);
        for (MI_S32 y = start_row; y < end_row; ++y)
        {
            const MI_F32 *src_row = mat.Ptr<MI_F32>(y);
            MI_S32 x = 0;
            MI_F32 row_result[3] = {0.0f};
            float32x4x3_t v3qf32_row_sum;
            neon::vdup(v3qf32_row_sum.val[0], 0.0f);
            neon::vdup(v3qf32_row_sum.val[1], 0.0f);
            neon::vdup(v3qf32_row_sum.val[2], 0.0f);
            for (; x < elem_count_align12; x += 12)
            {
                float32x4x3_t v3qf32_src_data;
                neon::vload(src_row + x, v3qf32_src_data);
                for (MI_S32 i = 0; i < 3; ++i)
                {
                    v3qf32_row_sum.val[i] = neon::vmla(v3qf32_row_sum.val[i], v3qf32_src_data.val[i], v3qf32_src_data.val[i]);
                }
            }
            for (MI_S32 i = 0; i < 3; ++i)
            {
                row_result[i] += neon::vgetlane<0>(v3qf32_row_sum.val[i]) + neon::vgetlane<1>(v3qf32_row_sum.val[i]) +
                              neon::vgetlane<2>(v3qf32_row_sum.val[i]) + neon::vgetlane<3>(v3qf32_row_sum.val[i]);
            }
            SqSumBorderImpl<MI_F32, MI_F32, MI_F32, 3>(src_row, row_result, x, row_elem_count, result);
        }

        task_result[start_blk] = result;
        return Status::OK;
    }
};

#if defined(AURA_ENABLE_NEON_FP16)
template <MI_S32 C>
struct SqSumNeonFunctor<MI_F16, C>
{
    Status operator()(Context *ctx, const Mat &mat, std::vector<Scalar> &task_result, MI_S32 start_blk, MI_S32 end_blk)
    {
        AURA_UNUSED(ctx);
        Sizes3 sz                = mat.GetSizes();
        MI_S32 width             = sz.m_width;
        MI_S32 row_elem_count    = width * C;
        MI_S32 elem_count_align8 = row_elem_count & (-8);
        MI_S32 start_row         = start_blk * SUM_BLK;
        MI_S32 end_row           = Min(end_blk * SUM_BLK, sz.m_height);

        Scalar result = Scalar::All(0.0);
        for (MI_S32 y = start_row; y < end_row; ++y)
        {
            const MI_F16 *src_row = mat.Ptr<MI_F16>(y);
            MI_S32 x = 0;
            float32x4_t vqf32_row_sum;
            neon::vdup(vqf32_row_sum, 0);
            MI_F32 row_result[4] = {0.0f};
            for (; x < elem_count_align8; x += 8)
            {
                float16x8_t vqf16_src_data;
                neon::vload(src_row + x, vqf16_src_data);
                vqf32_row_sum = neon::vmla(vqf32_row_sum, neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_data)), neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_data)));
                vqf32_row_sum = neon::vmla(vqf32_row_sum, neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_data)), neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_data)));
            }
            neon::vstore(row_result, vqf32_row_sum);
            SqSumBorderImpl<MI_F16, MI_F32, MI_F32, C>(src_row, row_result, x, row_elem_count, result);
        }

        task_result[start_blk] = result;
        return Status::OK;
    }
};

template <>
struct SqSumNeonFunctor<MI_F16, 3>
{
    Status operator()(Context *ctx, const Mat &mat, std::vector<Scalar> &task_result, MI_S32 start_blk, MI_S32 end_blk)
    {
        AURA_UNUSED(ctx);
        Sizes3 sz                 = mat.GetSizes();
        MI_S32 width              = sz.m_width;
        MI_S32 width_align8       = width & (-8);
        MI_S32 row_elem_count     = width * 3;
        MI_S32 elem_count_align24 = width_align8 * 3;
        MI_S32 start_row          = start_blk * SUM_BLK;
        MI_S32 end_row            = Min(end_blk * SUM_BLK, sz.m_height);

        Scalar result = Scalar::All(0.0);
        for (MI_S32 y = start_row; y < end_row; ++y)
        {
            const MI_F16 *src_row = mat.Ptr<MI_F16>(y);
            MI_S32 x = 0;
            MI_F32 row_result[3] = {0.0f};
            float32x4x3_t v3qf32_row_sum;
            neon::vdup(v3qf32_row_sum.val[0], 0.0f);
            neon::vdup(v3qf32_row_sum.val[1], 0.0f);
            neon::vdup(v3qf32_row_sum.val[2], 0.0f);
            for (; x < elem_count_align24; x += 24)
            {
                float16x8x3_t v3qf16_src_data;
                neon::vload(src_row + x, v3qf16_src_data);
                for (MI_S32 i = 0; i < 3; ++i)
                {
                    v3qf32_row_sum.val[i] = neon::vmla(v3qf32_row_sum.val[i], neon::vcvt<MI_F32>(neon::vgethigh(v3qf16_src_data.val[i])), neon::vcvt<MI_F32>(neon::vgethigh(v3qf16_src_data.val[i])));
                    v3qf32_row_sum.val[i] = neon::vmla(v3qf32_row_sum.val[i], neon::vcvt<MI_F32>(neon::vgetlow(v3qf16_src_data.val[i])), neon::vcvt<MI_F32>(neon::vgetlow(v3qf16_src_data.val[i])));
                }
            }
            for (MI_S32 i = 0; i < 3; ++i)
            {
                row_result[i] += neon::vgetlane<0>(v3qf32_row_sum.val[i]) + neon::vgetlane<1>(v3qf32_row_sum.val[i]) +
                              neon::vgetlane<2>(v3qf32_row_sum.val[i]) + neon::vgetlane<3>(v3qf32_row_sum.val[i]);
            }
            SqSumBorderImpl<MI_F16, MI_F32, MI_F32, 3>(src_row, row_result, x, row_elem_count, result);
        }

        task_result[start_blk] = result;
        return Status::OK;
    }
};
#endif

template <typename Tp>
static Status SqSumNeonHelper(Context *ctx, const Mat &mat, Scalar &result, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::OK;

    Sizes3 sz        = mat.GetSizes();
    MI_S32 height    = sz.m_height;
    MI_S32 channel   = sz.m_channel;
    MI_S32 task_nums = (height + SUM_BLK - 1) / SUM_BLK;
    std::vector<Scalar> task_result(task_nums, Scalar::All(0.0));

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get worker_pool failed.");
        return Status::ERROR;
    }

    switch (channel)
    {
        case 1:
        {
            SqSumNeonFunctor<Tp, 1> op;
            ret = wp->ParallelFor(0, task_nums, op, ctx, std::cref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SqSumNeonFunctor call ParallelFor (channel 1) error.");
            }
            break;
        }
        case 2:
        {
            SqSumNeonFunctor<Tp, 2> op;
            ret = wp->ParallelFor(0, task_nums, op, ctx, std::cref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SqSumNeonFunctor call ParallelFor (channel 2) error.");
            }
            break;
        }
        case 3:
        {
            SqSumNeonFunctor<Tp, 3> op;
            ret = wp->ParallelFor(0, task_nums, op, ctx, std::cref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SqSumNeonFunctor call ParallelFor (channel 3) error.");
            }
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "channel should <= 3.");
            break;
        }
    }

    result = Scalar::All(0.0);
    for (const auto &val : task_result)
    {
        result += val;
    }

    AURA_RETURN(ctx, ret);
}

Status SqSumNeon(Context *ctx, const Mat &mat, Scalar &result, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (mat.GetElemType())
    {
        case ElemType::U8:
        {
            ret = SqSumNeonHelper<MI_U8>(ctx, mat, result, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SqSumNeonHelper<MI_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = SqSumNeonHelper<MI_S8>(ctx, mat, result, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SqSumNeonHelper<MI_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = SqSumNeonHelper<MI_U16>(ctx, mat, result, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SqSumNeonHelper<MI_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = SqSumNeonHelper<MI_S16>(ctx, mat, result, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SqSumNeonHelper<MI_S16> failed.");
            }
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = SqSumNeonHelper<MI_F16>(ctx, mat, result, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SqSumNeonHelper<MI_F16> failed.");
            }
            break;
        }
#endif
        case ElemType::F32:
        {
            ret = SqSumNeonHelper<MI_F32>(ctx, mat, result, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SqSumNeonHelper<MI_F32> failed.");
            }
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupported elem_type.");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
