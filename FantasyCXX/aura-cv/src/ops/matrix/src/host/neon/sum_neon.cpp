#include "sum_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

#define SUM_BLK   (8)

namespace aura
{
template <typename Tp, typename Tv, typename Ts, MI_S32 VEC_SIZE, MI_S32 C>
struct SumBorderFunctor
{
    AURA_VOID operator()(const Tp *src, Tv vq_row_sum, MI_S32 cur_x, MI_S32 row_elem_count, Scalar &result)
    {
        Ts row_result[2 * VEC_SIZE] = {0};
        neon::vstore(row_result, vq_row_sum.val[0]);
        neon::vstore(row_result + VEC_SIZE, vq_row_sum.val[1]);
        for (; cur_x < row_elem_count; cur_x += C)
        {
            for (MI_S32 ch = 0; ch < C; ++ch)
            {
                row_result[ch] += src[cur_x + ch];
            }
        }
        for (MI_S32 i = 0; i < 2 * VEC_SIZE; i += C)
        {
            for (MI_S32 ch = 0; ch < C; ++ch)
            {
                result.m_val[ch] += row_result[i + ch];
            }
        }
    }
};

template <typename Tp, typename Tv, typename Ts, MI_S32 VEC_SIZE>
struct SumBorderFunctor<Tp, Tv, Ts, VEC_SIZE, 3>
{
    AURA_VOID operator()(const Tp *src, Tv v3q_row_sum, MI_S32 cur_x, MI_S32 row_elem_count, Scalar &result)
    {
        Ts row_result[3 * VEC_SIZE] = {0};
        neon::vstore(row_result, v3q_row_sum.val[0]);
        neon::vstore(row_result + VEC_SIZE, v3q_row_sum.val[1]);
        neon::vstore(row_result + 2 * VEC_SIZE, v3q_row_sum.val[2]);

        for (MI_S32 ch = 0; ch < 3; ch++)
        {
            Ts *cur_ch = row_result + ch * VEC_SIZE;
            for (MI_S32 i = 0; i < VEC_SIZE; ++i)
            {
                result.m_val[ch] += cur_ch[i];
            }
        }

        for (; cur_x < row_elem_count; cur_x += 3)
        {
            result.m_val[0] += src[cur_x];
            result.m_val[1] += src[cur_x + 1];
            result.m_val[2] += src[cur_x + 2];
        }
    }
};

template <typename Tp, MI_S32 C>
struct SumNeonFunctor
{
    static_assert(is_integral<Tp>::value && (1 == C || 2 == C), "support integer type with channel 1 2 only.");

    Status operator()(Context *ctx, const Mat &mat, std::vector<Scalar> &task_result, MI_S32 start_blk, MI_S32 end_blk)
    {
        AURA_UNUSED(ctx);

        using PromType0 = typename Promote<Tp>::Type;
        using PromType1 = typename Promote<PromType0>::Type;
        using LoadType  = typename neon::QVector<Tp>::VType;
        using BlockType = typename neon::MQVector<PromType0, 2>::MVType;
        using SumType   = typename neon::MQVector<PromType1, 2>::MVType;

        constexpr MI_S32 load_size  = 16 / sizeof(Tp);
        constexpr MI_S32 block_size = 1 << ((sizeof(PromType0) - sizeof(Tp)) * 8);
        constexpr MI_S32 block_step = block_size * load_size;

        Sizes3 sz             = mat.GetSizes();
        MI_S32 width          = sz.m_width;
        MI_S32 row_elem_count = width * C;
        MI_S32 start_row      = start_blk * SUM_BLK;
        MI_S32 end_row        = Min(end_blk * SUM_BLK, sz.m_height);

        Scalar result = Scalar::All(0.0);
        for (MI_S32 y = start_row; y < end_row; ++y)
        {
            const Tp *src = mat.Ptr<Tp>(y);
            SumType v2q_row_sum;
            neon::vdup(v2q_row_sum.val[0], 0);
            neon::vdup(v2q_row_sum.val[1], 0);

            MI_S32 bx = 0;
            for (MI_S32 x = 0; x < row_elem_count; x += block_step)
            {
                MI_S32 blk_len = Min(row_elem_count - x, block_step);
                MI_S32 blk_len_align = blk_len & (-load_size);
                BlockType v2q_block_sum;
                neon::vdup(v2q_block_sum.val[0], 0);
                neon::vdup(v2q_block_sum.val[1], 0);

                for (bx = x; bx < x + blk_len_align; bx += load_size)
                {
                    LoadType vq_src_data;
                    neon::vload(src + bx, vq_src_data);
                    v2q_block_sum.val[0] = neon::vaddw(v2q_block_sum.val[0], neon::vgetlow(vq_src_data));
                    v2q_block_sum.val[1] = neon::vaddw(v2q_block_sum.val[1], neon::vgethigh(vq_src_data));
                }

                v2q_row_sum.val[0] = neon::vaddw(v2q_row_sum.val[0], neon::vgetlow(v2q_block_sum.val[0]));
                v2q_row_sum.val[0] = neon::vaddw(v2q_row_sum.val[0], neon::vgetlow(v2q_block_sum.val[1]));
                v2q_row_sum.val[1] = neon::vaddw(v2q_row_sum.val[1], neon::vgethigh(v2q_block_sum.val[0]));
                v2q_row_sum.val[1] = neon::vaddw(v2q_row_sum.val[1], neon::vgethigh(v2q_block_sum.val[1]));
            }

            SumBorderFunctor<Tp, SumType, PromType1, 16 / sizeof(PromType1), C>()(src, v2q_row_sum, bx, row_elem_count, result);
        }

        task_result[start_blk] = result;

        return Status::OK;
    }
};

template <typename Tp>
struct SumNeonFunctor<Tp, 3>
{
    static_assert(is_integral<Tp>::value, "support integer type with channel 3 only.");

    Status operator()(Context *ctx, const Mat &mat, std::vector<Scalar> &task_result, MI_S32 start_blk, MI_S32 end_blk)
    {
        AURA_UNUSED(ctx);

        using PromType0 = typename Promote<Tp>::Type;
        using PromType1 = typename Promote<PromType0>::Type;
        using LoadType  = typename neon::MQVector<Tp, 3>::MVType;
        using BlockType = typename neon::MQVector<PromType0, 3>::MVType;
        using SumType   = typename neon::MQVector<PromType1, 3>::MVType;

        constexpr MI_S32 load_size  = 16 / sizeof(Tp) * 3;
        constexpr MI_S32 block_size = (1 << ((sizeof(PromType0) - sizeof(Tp)) * 8)) / 2;
        constexpr MI_S32 block_step = block_size * load_size;

        Sizes3 sz             = mat.GetSizes();
        MI_S32 width          = sz.m_width;
        MI_S32 row_elem_count = width * 3;
        MI_S32 start_row      = start_blk * SUM_BLK;
        MI_S32 end_row        = Min(end_blk * SUM_BLK, sz.m_height);

        Scalar result = Scalar::All(0.0);
        for (MI_S32 y = start_row; y < end_row; ++y)
        {
            const Tp *src = mat.Ptr<Tp>(y);

            SumType v3q_row_sum;
            neon::vdup(v3q_row_sum.val[0], 0);
            neon::vdup(v3q_row_sum.val[1], 0);
            neon::vdup(v3q_row_sum.val[2], 0);

            MI_S32 bx = 0;
            for (MI_S32 x = 0; x < row_elem_count; x+= block_step)
            {
                MI_S32 blk_len = Min(row_elem_count - x, block_step);
                MI_S32 blk_len_align = blk_len & (-load_size);
                BlockType v3q_block_sum;
                neon::vdup(v3q_block_sum.val[0], 0);
                neon::vdup(v3q_block_sum.val[1], 0);
                neon::vdup(v3q_block_sum.val[2], 0);
                for (bx = x; bx < x + blk_len_align; bx += load_size)
                {
                    LoadType v3q_src_data;
                    neon::vload(src + bx, v3q_src_data);
                    v3q_block_sum.val[0] = neon::vpadal(v3q_block_sum.val[0], v3q_src_data.val[0]);
                    v3q_block_sum.val[1] = neon::vpadal(v3q_block_sum.val[1], v3q_src_data.val[1]);
                    v3q_block_sum.val[2] = neon::vpadal(v3q_block_sum.val[2], v3q_src_data.val[2]);
                }
                v3q_row_sum.val[0] = neon::vpadal(v3q_row_sum.val[0], v3q_block_sum.val[0]);
                v3q_row_sum.val[1] = neon::vpadal(v3q_row_sum.val[1], v3q_block_sum.val[1]);
                v3q_row_sum.val[2] = neon::vpadal(v3q_row_sum.val[2], v3q_block_sum.val[2]);
            }

            // Border Scalar Process
            SumBorderFunctor<Tp, SumType, PromType1, 16 / sizeof(PromType1), 3>()(src, v3q_row_sum, bx, row_elem_count, result);
        }

        task_result[start_blk] = result;

        return Status::OK;
    }
};

template <MI_S32 C>
struct SumNeonFunctor<MI_F32, C>
{
    static_assert(1 == C || 2 == C, "support for float with channel 1 2 only.");

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
            const MI_F32 *src = mat.Ptr<MI_F32>(y);
            float32x4_t vqf32_row_sum;
            neon::vdup(vqf32_row_sum, 0);
            MI_S32 x = 0;
            for (; x < elem_count_align4; x += 4)
            {
                vqf32_row_sum = neon::vadd(vqf32_row_sum, neon::vload1q(src + x));
            }
            float32x4x2_t row_sum_temp;
            row_sum_temp.val[0] = vqf32_row_sum;
            neon::vdup(row_sum_temp.val[1], 0);
            SumBorderFunctor<MI_F32, float32x4x2_t, MI_F32, 4, C>()(src, row_sum_temp, x, row_elem_count, result);
        }

        task_result[start_blk] = result;

        return Status::OK;
    }
};

template <>
struct SumNeonFunctor<MI_F32, 3>
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
            float32x4x3_t v3qf32_row_sum;
            neon::vdup(v3qf32_row_sum.val[0], 0);
            neon::vdup(v3qf32_row_sum.val[1], 0);
            neon::vdup(v3qf32_row_sum.val[2], 0);

            const MI_F32 *src = mat.Ptr<MI_F32>(y);
            MI_S32 x = 0;
            for (; x < elem_count_align12; x += 12)
            {
                float32x4x3_t v3qf32_src_data = neon::vload3q(src + x);
                v3qf32_row_sum.val[0] = neon::vadd(v3qf32_row_sum.val[0], v3qf32_src_data.val[0]);
                v3qf32_row_sum.val[1] = neon::vadd(v3qf32_row_sum.val[1], v3qf32_src_data.val[1]);
                v3qf32_row_sum.val[2] = neon::vadd(v3qf32_row_sum.val[2], v3qf32_src_data.val[2]);
            }
            SumBorderFunctor<MI_F32, float32x4x3_t, MI_F32, 4, 3>()(src, v3qf32_row_sum, x, row_elem_count, result);
        }

        task_result[start_blk] = result;

        return Status::OK;
    }
};

#if defined(AURA_ENABLE_NEON_FP16)
template <MI_S32 C>
struct SumNeonFunctor<MI_F16, C>
{
    static_assert(1 == C || 2 == C, "support for float with channel 1 2 only.");

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
            const MI_F16 *src = mat.Ptr<MI_F16>(y);
            float32x4_t vqf32_row_sum;
            neon::vdup(vqf32_row_sum, 0);
            MI_S32 x = 0;
            for (; x < elem_count_align8; x += 8)
            {
                float16x8_t vqf16_src_data = neon::vload1q(src + x);
                vqf32_row_sum = neon::vadd(vqf32_row_sum, neon::vcvt<MI_F32>(neon::vgethigh(vqf16_src_data)));
                vqf32_row_sum = neon::vadd(vqf32_row_sum, neon::vcvt<MI_F32>(neon::vgetlow(vqf16_src_data)));
            }
            float32x4x2_t row_sum_temp;
            row_sum_temp.val[0] = vqf32_row_sum;
            neon::vdup(row_sum_temp.val[1], 0);
            SumBorderFunctor<MI_F16, float32x4x2_t, MI_F32, 4, C>()(src, row_sum_temp, x, row_elem_count, result);
        }

        task_result[start_blk] = result;

        return Status::OK;
    }
};

template <>
struct SumNeonFunctor<MI_F16, 3>
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
            float32x4x3_t v3qf32_row_sum;
            neon::vdup(v3qf32_row_sum.val[0], 0);
            neon::vdup(v3qf32_row_sum.val[1], 0);
            neon::vdup(v3qf32_row_sum.val[2], 0);

            const MI_F16 *src = mat.Ptr<MI_F16>(y);
            MI_S32 x = 0;
            for (; x < elem_count_align24; x += 24)
            {
                float16x8x3_t v3qf16_src_data = neon::vload3q(src + x);
                v3qf32_row_sum.val[0] = neon::vadd(v3qf32_row_sum.val[0], neon::vcvt<MI_F32>(neon::vgethigh(v3qf16_src_data.val[0])));
                v3qf32_row_sum.val[0] = neon::vadd(v3qf32_row_sum.val[0], neon::vcvt<MI_F32>(neon::vgetlow(v3qf16_src_data.val[0])));
                v3qf32_row_sum.val[1] = neon::vadd(v3qf32_row_sum.val[1], neon::vcvt<MI_F32>(neon::vgethigh(v3qf16_src_data.val[1])));
                v3qf32_row_sum.val[1] = neon::vadd(v3qf32_row_sum.val[1], neon::vcvt<MI_F32>(neon::vgetlow(v3qf16_src_data.val[1])));
                v3qf32_row_sum.val[2] = neon::vadd(v3qf32_row_sum.val[2], neon::vcvt<MI_F32>(neon::vgethigh(v3qf16_src_data.val[2])));
                v3qf32_row_sum.val[2] = neon::vadd(v3qf32_row_sum.val[2], neon::vcvt<MI_F32>(neon::vgetlow(v3qf16_src_data.val[2])));
            }
            SumBorderFunctor<MI_F16, float32x4x3_t, MI_F32, 4, 3>()(src, v3qf32_row_sum, x, row_elem_count, result);
        }

        task_result[start_blk] = result;

        return Status::OK;
    }
};
#endif

template <typename Tp>
Status SumNeonHelper(Context *ctx, const Mat &mat, Scalar &result, const OpTarget &target)
{
    AURA_UNUSED(target);
    Status ret = Status::OK;

    Sizes3 sz      = mat.GetSizes();
    MI_S32 height  = sz.m_height;
    MI_S32 channel = sz.m_channel;


    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get worker_pool failed.");
        return Status::ERROR;
    }

    MI_S32 task_nums = (height + SUM_BLK - 1) / SUM_BLK;
    std::vector<Scalar> task_result(task_nums, Scalar::All(0.0));

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor(0, task_nums, SumNeonFunctor<Tp, 1>(), ctx, std::cref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SumNeonFunctor<Tp, 1> failed.");
            }
            break;
        }
        case 2:
        {
            ret = wp->ParallelFor(0, task_nums, SumNeonFunctor<Tp, 2>(), ctx, std::cref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SumNeonFunctor<Tp, 2> failed.");
            }
            break;
        }
        case 3:
        {
            ret = wp->ParallelFor(0, task_nums, SumNeonFunctor<Tp, 3>(), ctx, std::cref(mat), std::ref(task_result));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "SumNeonFunctor<Tp, 3> failed.");
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

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "SumNeonHelper failed.");
        return Status::ERROR;
    }

    for (const auto &val : task_result)
    {
        result += val;
    }

    return Status::OK;
}

SumNeon::SumNeon(Context *ctx, const OpTarget &target) : SumImpl(ctx, target)
{}

Status SumNeon::SetArgs(const Array *src, Scalar *result)
{
    if (SumImpl::SetArgs(src, result) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SumNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = SumNeonHelper<MI_U8>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNeonHelper<MI_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = SumNeonHelper<MI_S8>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNeonHelper<MI_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = SumNeonHelper<MI_U16>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNeonHelper<MI_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = SumNeonHelper<MI_S16>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNeonHelper<MI_S16> failed.");
            }
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = SumNeonHelper<MI_F16>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNeonHelper<MI_F16> failed.");
            }
            break;
        }
#endif
        case ElemType::F32:
        {
            ret = SumNeonHelper<MI_F32>(m_ctx, *src, *m_result, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SumNeonHelper<MI_F32> failed.");
            }
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported elem_type.");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

MeanNeon::MeanNeon(Context *ctx, const OpTarget &target) : SumNeon(ctx, target)
{}

Status MeanNeon::Run()
{
    Status ret = SumNeon::Run();
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumNeon run failed.");
        return Status::ERROR;
    }

    const MI_S32 height = m_src->GetSizes().m_height;
    const MI_S32 width  = m_src->GetSizes().m_width;
    *m_result           = (*m_result) / static_cast<MI_F64>(height * width);

    return Status::OK;
}

} // namespace aura