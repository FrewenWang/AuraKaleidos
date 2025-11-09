#include "mul_spectrums_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

static Status MulSpectrumsNeonImpl(const Mat &src0, const Mat &src1, Mat &dst, DT_BOOL conj_src1, DT_S32 start_row, DT_S32 end_row)
{
    Sizes3 sz = src0.GetSizes();
    DT_S32 width  = sz.m_width;
    DT_S32 elem_count = width * sz.m_channel;
    DT_S32 elem_count_align8 = elem_count & (-8);

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const DT_F32 *src0_c = src0.Ptr<DT_F32>(y);
        const DT_F32 *src1_c = src1.Ptr<DT_F32>(y);
        DT_F32 *dst_c = dst.Ptr<DT_F32>(y);

        DT_S32 x = 0;
        if (conj_src1)
        {
            for (; x < elem_count_align8; x += 8)
            {
                float32x4x2_t v2qf32_result;
                float32x4x2_t v2qf32_x0  = neon::vload2q(src0_c + x);
                float32x4x2_t v2qf32_x1  = neon::vload2q(src1_c + x);
                float32x4_t vqf32_x0rx1r = neon::vmul(v2qf32_x0.val[0], v2qf32_x1.val[0]);
                float32x4_t vqf32_x0ix1r = neon::vmul(v2qf32_x0.val[1], v2qf32_x1.val[0]);

                v2qf32_result.val[0] = neon::vmla(vqf32_x0rx1r, v2qf32_x0.val[1], v2qf32_x1.val[1]);
                v2qf32_result.val[1] = neon::vmls(vqf32_x0ix1r, v2qf32_x0.val[0], v2qf32_x1.val[1]);
                neon::vstore(dst_c + x, v2qf32_result);
            }
        }
        else
        {
            for (; x < elem_count_align8; x += 8)
            {
                float32x4x2_t v2qf32_result;
                float32x4x2_t v2qf32_x0  = neon::vload2q(src0_c + x);
                float32x4x2_t v2qf32_x1  = neon::vload2q(src1_c + x);
                float32x4_t vqf32_x0rx1r = neon::vmul(v2qf32_x0.val[0], v2qf32_x1.val[0]);
                float32x4_t vqf32_x0rx1i = neon::vmul(v2qf32_x0.val[0], v2qf32_x1.val[1]);

                v2qf32_result.val[0] = neon::vmls(vqf32_x0rx1r, v2qf32_x0.val[1], v2qf32_x1.val[1]);
                v2qf32_result.val[1] = neon::vmla(vqf32_x0rx1i, v2qf32_x0.val[1], v2qf32_x1.val[0]);
                neon::vstore(dst_c + x, v2qf32_result);
            }
        }

        for (; x < elem_count; x += 2)
        {
            DT_F32 real0 = src0_c[x];
            DT_F32 imag0 = src0_c[x + 1];

            DT_F32 real1 = src1_c[x];
            DT_F32 imag1 = conj_src1 ? -src1_c[x + 1] : src1_c[x + 1];

            dst_c[x] = real0 * real1 - imag0 * imag1;
            dst_c[x + 1] = real0 * imag1 + real1 * imag0;
        }
    }

    return Status::OK;
}

MulSpectrumsNeon::MulSpectrumsNeon(Context *ctx, const OpTarget &target) : MulSpectrumsImpl(ctx, target)
{}

Status MulSpectrumsNeon::SetArgs(const Array *src0, const Array *src1, Array *dst, DT_BOOL conj_src1)
{
    if (MulSpectrumsImpl::SetArgs(src0, src1, dst, conj_src1) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MulSpectrumsImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src0->GetArrayType() != ArrayType::MAT) || (src1->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MulSpectrumsNeon::Run()
{
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);
    Mat *dst        = dynamic_cast<Mat*>(m_dst);

    if ((NULL == src0) || (NULL == src1) || (NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    WorkerPool *wp = m_ctx->GetWorkerPool();

    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    Sizes3 sz     = src0->GetSizes();
    DT_S32 height = sz.m_height;

    Status ret = wp->ParallelFor(0, height, MulSpectrumsNeonImpl, std::cref(*src0), std::cref(*src1), std::ref(*dst), m_conj_src1);

    AURA_RETURN(m_ctx, ret);
}

} // aura