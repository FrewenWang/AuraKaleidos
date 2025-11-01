#include "mul_spectrums_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

static Status MulSpectrumsNoneImpl(const Mat &src0, const Mat &src1, Mat &dst, MI_BOOL conj_src1, MI_S32 start_row, MI_S32 end_row)
{
    Sizes3 sz              = src0.GetSizes();
    MI_S32 width           = sz.m_width;
    MI_S32 row_elem_counts = width * sz.m_channel;

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const MI_F32 *src0_row = src0.Ptr<MI_F32>(y);
        const MI_F32 *src1_row = src1.Ptr<MI_F32>(y);
        MI_F32 *dst_row        = dst.Ptr<MI_F32>(y);

        for (MI_S32 x = 0; x < row_elem_counts; x += 2)
        {
            MI_F32 real0 = src0_row[x];
            MI_F32 imag0 = src0_row[x + 1];

            MI_F32 real1 = src1_row[x];
            MI_F32 imag1 = conj_src1 ? -src1_row[x + 1] : src1_row[x + 1];

            dst_row[x]     = real0 * real1 - imag0 * imag1;
            dst_row[x + 1] = real0 * imag1 + real1 * imag0;
        }
    }

    return Status::OK;
}


MulSpectrumsNone::MulSpectrumsNone(Context *ctx, const OpTarget &target) : MulSpectrumsImpl(ctx, target)
{}

Status MulSpectrumsNone::SetArgs(const Array *src0, const Array *src1, Array *dst, MI_BOOL conj_src1)
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

Status MulSpectrumsNone::Run()
{
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);
    Mat *dst        = dynamic_cast<Mat*>(m_dst);

    if ((NULL == src0) || (NULL == src1) || (NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    MI_S32 height = dst->GetSizes().m_height;

    if (m_target.m_data.none.enable_mt)
    {
        WorkerPool *wp = m_ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor(static_cast<MI_S32>(0), height, MulSpectrumsNoneImpl, std::cref(*src0), std::cref(*src1),
                              std::ref(*dst), m_conj_src1);
    }
    else
    {
        ret = MulSpectrumsNoneImpl(*src0, *src1, *dst, m_conj_src1, static_cast<MI_S32>(0), height);
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura