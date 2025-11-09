#include "guidefilter_impl.hpp"
#include "aura/ops/filter.h"
#include "aura/ops/matrix.h"
#include "aura/ops/resize.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"

namespace aura
{

template<typename Tp>
struct GuideFilterTraits
{
    using SumType     = DT_F32;
    using SqSumType   = DT_F32;
};

template<>
struct GuideFilterTraits<DT_U8>
{
    using SumType   = DT_U32;
    using SqSumType = DT_U32;
};

template<>
struct GuideFilterTraits<DT_S8>
{
    using SumType   = DT_S32;
    using SqSumType = DT_S32;
};

template<>
struct GuideFilterTraits<DT_U16>
{
    using SumType   = DT_U32;
    using SqSumType = DT_U64;
};

template<>
struct GuideFilterTraits<DT_S16>
{
    using SumType   = DT_S32;
    using SqSumType = DT_S64;
};

template <typename Tp>
static Status GuideFilterCalcABNoneImpl(Context *ctx, const Mat &src_a, const Mat &src_b, Mat &dst_a, Mat &dst_b,
                                        ThreadBuffer &thread_buffer_a, ThreadBuffer &thread_buffer_b, ThreadBuffer &thread_buffer_asq,
                                        ThreadBuffer &thread_buffer_bsq, DT_S32 ksize, DT_F32 eps, DT_S32 start_row, DT_S32 end_row)
{
    Status ret = Status::ERROR;

    using SumType   = typename GuideFilterTraits<Tp>::SumType;
    using SqSumType = typename GuideFilterTraits<Tp>::SqSumType;

    const Sizes3 isizes  = src_a.GetSizes();
    const Sizes3 osizes  = dst_a.GetSizes();
    const DT_S32 iwidth  = isizes.m_width;
    const DT_S32 owidth  = osizes.m_width;
    const DT_S32 channel = isizes.m_channel;
    DT_S32 ksh = ksize >> 1;
    DT_S32 ksq = ksize * ksize;

    std::vector<const Tp*> src_rows_a(ksize);
    std::vector<const Tp*> src_rows_b(ksize);
    for (DT_S32 k = 0; k < ksize; k++)
    {
        src_rows_a[k] = src_a.Ptr<Tp>(start_row + k);
        src_rows_b[k] = src_b.Ptr<Tp>(start_row + k);
    }

    SumType   *sum_a_row    = thread_buffer_a.GetThreadData<SumType>();
    SumType   *sum_b_row    = thread_buffer_b.GetThreadData<SumType>();
    SqSumType *sqsum_aa_row = thread_buffer_asq.GetThreadData<SqSumType>();
    SqSumType *sqsum_ab_row = thread_buffer_bsq.GetThreadData<SqSumType>();

    if (DT_NULL == sum_a_row || DT_NULL == sum_b_row || DT_NULL == sqsum_aa_row || DT_NULL == sqsum_ab_row)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
        goto EXIT;
    }

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        // calc vertical sum (iwidth = owidth + ksize/2 * 2)
        for (DT_S32 x = 0; x < iwidth; x++)
        {
            for (DT_S32 ch = 0; ch < channel; ch++)
            {
                SumType   sum_a    = 0;
                SumType   sum_b    = 0;
                SqSumType sqsum_aa = 0;
                SqSumType sqsum_ab = 0;

                DT_S32 index = x * channel + ch;
                for (DT_S32 k = 0; k < ksh; k++)
                {
                    Tp a_top = src_rows_a[k][index];
                    Tp b_top = src_rows_b[k][index];
                    Tp a_bot = src_rows_a[ksize - k - 1][index];
                    Tp b_bot = src_rows_b[ksize - k - 1][index];

                    sum_a    += static_cast<SumType>(a_top)   + static_cast<SumType>(a_bot);
                    sum_b    += static_cast<SumType>(b_top)   + static_cast<SumType>(b_bot);
                    sqsum_aa += static_cast<SqSumType>(a_top) * static_cast<SqSumType>(a_top) +
                                static_cast<SqSumType>(a_bot) * static_cast<SqSumType>(a_bot);
                    sqsum_ab += static_cast<SqSumType>(a_top) * static_cast<SqSumType>(b_top) +
                                static_cast<SqSumType>(a_bot) * static_cast<SqSumType>(b_bot);
                }

                sum_a    += static_cast<SumType  >(src_rows_a[ksh][index]);
                sum_b    += static_cast<SumType  >(src_rows_b[ksh][index]);
                sqsum_aa += static_cast<SqSumType>(src_rows_a[ksh][index]) * static_cast<SqSumType>(src_rows_a[ksh][index]);
                sqsum_ab += static_cast<SqSumType>(src_rows_a[ksh][index]) * static_cast<SqSumType>(src_rows_b[ksh][index]);

                sum_a_row[index]    = sum_a;
                sum_b_row[index]    = sum_b;
                sqsum_aa_row[index] = sqsum_aa;
                sqsum_ab_row[index] = sqsum_ab;
            }
        }

        DT_F32 *dst_row_a = dst_a.Ptr<DT_F32>(y);
        DT_F32 *dst_row_b = dst_b.Ptr<DT_F32>(y);

        // calc horizontal sum
        for (DT_S32 x = 0; x < owidth; x++)
        {
            for (DT_S32 ch = 0; ch < channel; ch++)
            {
                SumType   sum_kernel_a    = 0;
                SumType   sum_kernel_b    = 0;
                SqSumType sqsum_kernel_aa = 0;
                SqSumType sqsum_kernel_ab = 0;

                for (DT_S32 k = 0; k < ksh; k++)
                {
                    sum_kernel_a    += (   sum_a_row[(x + k) * channel + ch] +    sum_a_row[(x + ksize - k - 1) * channel + ch]);
                    sum_kernel_b    += (   sum_b_row[(x + k) * channel + ch] +    sum_b_row[(x + ksize - k - 1) * channel + ch]);
                    sqsum_kernel_aa += (sqsum_aa_row[(x + k) * channel + ch] + sqsum_aa_row[(x + ksize - k - 1) * channel + ch]);
                    sqsum_kernel_ab += (sqsum_ab_row[(x + k) * channel + ch] + sqsum_ab_row[(x + ksize - k - 1) * channel + ch]);
                }
                sum_kernel_a    +=    sum_a_row[(x + ksh) * channel + ch];
                sum_kernel_b    +=    sum_b_row[(x + ksh) * channel + ch];
                sqsum_kernel_aa += sqsum_aa_row[(x + ksh) * channel + ch];
                sqsum_kernel_ab += sqsum_ab_row[(x + ksh) * channel + ch];

                DT_F32 mean_i  = static_cast<DT_F32>(sum_kernel_a) / ksq;
                DT_F32 mean_p  = static_cast<DT_F32>(sum_kernel_b) / ksq;
                DT_F32 var     = static_cast<DT_F32>(sqsum_kernel_aa) / ksq - mean_i * mean_i;
                DT_F32 cov     = static_cast<DT_F32>(sqsum_kernel_ab) / ksq - mean_i * mean_p;
                DT_F32 var_eps = (Abs(var + eps) < GUIDE_FILTER_EPS) ? GUIDE_FILTER_EPS : (var + eps);

                dst_row_a[x * channel + ch] = cov / var_eps;
                dst_row_b[x * channel + ch] = mean_p - mean_i * dst_row_a[x * channel + ch];
            }
        }

        for (DT_S32 i = 0; i < ksize - 1; i++)
        {
            src_rows_a[i] = src_rows_a[i + 1];
            src_rows_b[i] = src_rows_b[i + 1];
        }

        src_rows_a[ksize - 1] = src_a.Ptr<Tp>(y + ksize);
        src_rows_b[ksize - 1] = src_b.Ptr<Tp>(y + ksize);
    }

    ret = Status::OK;

EXIT:

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status GuideFilterLinearTransImpl(const Mat &src, const Mat &mean_a, const Mat &mean_b, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    const Sizes3 sz = mean_a.GetSizes();

    for (DT_S32 y = start_row; y < end_row; ++y)
    {
        const Tp     *src_row    = src.Ptr<Tp>(y);
        const DT_F32 *mean_a_row = mean_a.Ptr<DT_F32>(y);
        const DT_F32 *mean_b_row = mean_b.Ptr<DT_F32>(y);

        Tp *dst_row = dst.Ptr<Tp>(y);

        for (DT_S32 x = 0; x < sz.m_width * sz.m_channel; ++x)
        {
            dst_row[x] = SaturateCast<Tp>(static_cast<DT_F32>(src_row[x]) * mean_a_row[x] + mean_b_row[x]);
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status GuideFilteNormalNoneImpl(Context *ctx, const Mat &src0, const Mat &border_src0, const Mat &border_src1, Mat &dst,
                                       DT_S32 ksize, DT_F32 eps, BorderType border_type, const Scalar &border_value, OpTarget &target)
{
    Status ret = Status::ERROR;

    Mat a = Mat(ctx, ElemType::F32, src0.GetSizes());
    Mat b = Mat(ctx, ElemType::F32, src0.GetSizes());
    if (!a.IsValid() || !b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat a or b is invalid");
        return Status::ERROR;
    }

    using SumType   = typename GuideFilterTraits<Tp>::SumType;
    using SqSumType = typename GuideFilterTraits<Tp>::SqSumType;
    DT_S32 iwidth   = border_src0.GetSizes().m_width;
    DT_S32 ichannel = border_src0.GetSizes().m_channel;

    DT_S32 ab_buffer_size = iwidth * ichannel * sizeof(SumType);
    DT_S32 sq_buffer_size = iwidth * ichannel * sizeof(SqSumType);

    WorkerPool *wp = DT_NULL;
    if (target.m_data.none.enable_mt)
    {
        wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ThreadBuffer thread_buffer_a(ctx,   ab_buffer_size);
        ThreadBuffer thread_buffer_b(ctx,   ab_buffer_size);
        ThreadBuffer thread_buffer_asq(ctx, sq_buffer_size);
        ThreadBuffer thread_buffer_bsq(ctx, sq_buffer_size);

        ret = wp->ParallelFor(static_cast<DT_S32>(0), a.GetSizes().m_height, GuideFilterCalcABNoneImpl<Tp>, ctx, std::cref(border_src0), std::cref(border_src1),
                              std::ref(a), std::ref(b), std::ref(thread_buffer_a), std::ref(thread_buffer_b), std::ref(thread_buffer_asq),
                              std::ref(thread_buffer_bsq), ksize, eps);
    }
    else
    {
        ThreadBuffer thread_buffer_a(ctx,   ab_buffer_size);
        ThreadBuffer thread_buffer_b(ctx,   ab_buffer_size);
        ThreadBuffer thread_buffer_asq(ctx, sq_buffer_size);
        ThreadBuffer thread_buffer_bsq(ctx, sq_buffer_size);

        ret = GuideFilterCalcABNoneImpl<Tp>(ctx, border_src0, border_src1, a, b, thread_buffer_a, thread_buffer_b, thread_buffer_asq, thread_buffer_bsq,
                                            ksize, eps, 0, a.GetSizes().m_height);
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GuideFilterCalcABNoneImpl failed");
        return Status::ERROR;
    }

    Mat mean_a = Mat(ctx, ElemType::F32, src0.GetSizes());
    Mat mean_b = Mat(ctx, ElemType::F32, src0.GetSizes());
    if (!mean_a.IsValid() || !mean_b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat mean_a or mean_b is invalid");
        return Status::ERROR;
    }

    ret |= IBoxfilter(ctx, a, mean_a, ksize, border_type, border_value, target);
    ret |= IBoxfilter(ctx, b, mean_b, ksize, border_type, border_value, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "BoxFilterNone failed");
        return Status::ERROR;
    }

    if (target.m_data.none.enable_mt)
    {
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor(static_cast<DT_S32>(0), mean_a.GetSizes().m_height, GuideFilterLinearTransImpl<Tp>,
                              std::cref(src0), std::cref(mean_a), std::cref(mean_b), std::ref(dst));
    }
    else
    {
        ret = GuideFilterLinearTransImpl<Tp>(src0, mean_a, mean_b, dst, 0, dst.GetSizes().m_height);
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GuideFilterLinearTransImpl failed");
        return Status::ERROR;
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status GuideFilterFastNoneImpl(Context *ctx, const Mat &src0, Mat &sub_src0, Mat &border_src0, Mat &border_src1, Mat &dst,
                                      DT_S32 ksize, DT_F32 eps, BorderType border_type, const Scalar &border_value, OpTarget &target)
{
    Status ret = Status::ERROR;

    DT_S32 ksize_fast = GetFastKsize(ksize);

    Mat sub_a = Mat(ctx, ElemType::F32, sub_src0.GetSizes());
    Mat sub_b = Mat(ctx, ElemType::F32, sub_src0.GetSizes());
    if (!sub_a.IsValid() || !sub_b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat sub_a or sub_b is invalid");
        return Status::ERROR;
    }

    using SumType   = typename GuideFilterTraits<Tp>::SumType;
    using SqSumType = typename GuideFilterTraits<Tp>::SqSumType;
    DT_S32 iwidth   = border_src0.GetSizes().m_width;
    DT_S32 ichannel = border_src0.GetSizes().m_channel;

    DT_S32 ab_buffer_size = iwidth * ichannel * sizeof(SumType);
    DT_S32 sq_buffer_size = iwidth * ichannel * sizeof(SqSumType);

    WorkerPool *wp = DT_NULL;
    if (target.m_data.none.enable_mt)
    {
        wp = ctx->GetWorkerPool();
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ThreadBuffer thread_buffer_a(ctx,   ab_buffer_size);
        ThreadBuffer thread_buffer_b(ctx,   ab_buffer_size);
        ThreadBuffer thread_buffer_asq(ctx, sq_buffer_size);
        ThreadBuffer thread_buffer_bsq(ctx, sq_buffer_size);

        ret = wp->ParallelFor(static_cast<DT_S32>(0), sub_a.GetSizes().m_height, GuideFilterCalcABNoneImpl<Tp>, ctx, std::cref(border_src0), std::cref(border_src1),
                              std::ref(sub_a), std::ref(sub_b), std::ref(thread_buffer_a), std::ref(thread_buffer_b), std::ref(thread_buffer_asq),
                              std::ref(thread_buffer_bsq), ksize_fast, eps);
    }
    else
    {
        ThreadBuffer thread_buffer_a(ctx,   ab_buffer_size);
        ThreadBuffer thread_buffer_b(ctx,   ab_buffer_size);
        ThreadBuffer thread_buffer_asq(ctx, sq_buffer_size);
        ThreadBuffer thread_buffer_bsq(ctx, sq_buffer_size);

        ret = GuideFilterCalcABNoneImpl<Tp>(ctx, border_src0, border_src1, sub_a, sub_b, thread_buffer_a, thread_buffer_b, thread_buffer_asq, thread_buffer_bsq,
                                            ksize_fast, eps, 0, sub_a.GetSizes().m_height);
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GuideFilterCalcABNoneImpl failed");
        return Status::ERROR;
    }

    Mat mean_a_sub = Mat(ctx, ElemType::F32, sub_src0.GetSizes());
    Mat mean_b_sub = Mat(ctx, ElemType::F32, sub_src0.GetSizes());
    if (!mean_a_sub.IsValid() || !mean_b_sub.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat mean_a_sub or mean_b_sub is invalid");
        return Status::ERROR;
    }

    ret |= IBoxfilter(ctx, sub_a, mean_a_sub, ksize_fast, border_type, border_value, target);
    ret |= IBoxfilter(ctx, sub_b, mean_b_sub, ksize_fast, border_type, border_value, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "BoxFilterNone failed");
        return Status::ERROR;
    }

    Mat mean_a = Mat(ctx, ElemType::F32, src0.GetSizes());
    Mat mean_b = Mat(ctx, ElemType::F32, src0.GetSizes());
    if (!mean_a.IsValid() || !mean_b.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "Mat mean_a_full or mean_b_full is invalid");
        return Status::ERROR;
    }

    ret |= IResize(ctx, mean_a_sub, mean_a, InterpType::LINEAR, target);
    ret |= IResize(ctx, mean_b_sub, mean_b, InterpType::LINEAR, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "IResize failed");
        return Status::ERROR;
    }

    if (target.m_data.none.enable_mt)
    {
        if (DT_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor(static_cast<DT_S32>(0), mean_a.GetSizes().m_height, GuideFilterLinearTransImpl<Tp>,
                              std::cref(src0), std::cref(mean_a), std::cref(mean_b), std::ref(dst));
    }
    else
    {
        ret = GuideFilterLinearTransImpl<Tp>(src0, mean_a, mean_b, dst, 0, dst.GetSizes().m_height);
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GuideFilterLinearTransImpl failed");
        return Status::ERROR;
    }

    AURA_RETURN(ctx, ret);
}

GuideFilterNone::GuideFilterNone(Context *ctx, const OpTarget &target) : GuideFilterImpl(ctx, target)
{}

Status GuideFilterNone::SetArgs(const Array *src0, const Array *src1, Array *dst, DT_S32 ksize, DT_F32 eps,
                                GuideFilterType type, BorderType border_type, const Scalar &border_value)
{
    if (GuideFilterImpl::SetArgs(src0, src1, dst, ksize, eps, type, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GuideFilterImpl::Initialize failed");
        return Status::ERROR;
    }

    if ((src0->GetArrayType() != ArrayType::MAT) || (src1->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src0 src1 dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GuideFilterNone::Initialize()
{
    if (GuideFilterImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GuideFilterImpl::Prepare() failed");
        return Status::ERROR;
    }

    // Get border mat sizes
    Sizes3 border_sizes;
    if (GuideFilterType::FAST == m_type)
    {
        DT_S32 radius = GetFastKsize(m_ksize) >> 1;
        border_sizes  = {(m_src0->GetSizes().m_height >> 1) + (radius << 1),
                         (m_src0->GetSizes().m_width  >> 1) + (radius << 1),
                          m_src0->GetSizes().m_channel};
    }
    else
    {
        DT_S32 radius = m_ksize >> 1;
        border_sizes  = m_src0->GetSizes() + Sizes3(radius << 1, radius << 1, 0);
    }

    m_src_border0 = Mat(m_ctx, m_src0->GetElemType(), border_sizes);
    m_src_border1 = Mat(m_ctx, m_src1->GetElemType(), border_sizes);
    if (!m_src_border0.IsValid() || !m_src_border1.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_src_border0 or m_src_border1 is null");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GuideFilterNone::Run()
{
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);
    Mat       *dst  = dynamic_cast<Mat*>(m_dst);

    Status ret = Status::ERROR;

    if ((DT_NULL == src0) || (DT_NULL == src1) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src0 src1 dst is null");
        return Status::ERROR;
    }

    if (GuideFilterType::FAST == m_type)
    {
        // Resize dn src0 & src1
        Sizes3 sub_sizes   = {src0->GetSizes().m_height >> 1, src0->GetSizes().m_width >> 1, src0->GetSizes().m_channel};
        Sizes  sub_strides = {sub_sizes.m_height, sub_sizes.m_width * sub_sizes.m_channel * ElemTypeSize(src0->GetElemType())};
        Mat    sub_src0    = Mat(m_ctx, src0->GetElemType(), sub_sizes, AURA_MEM_DEFAULT, sub_strides);
        Mat    sub_src1    = Mat(m_ctx, src1->GetElemType(), sub_sizes, AURA_MEM_DEFAULT, sub_strides);

        if (IResize(m_ctx, *src0, sub_src0, InterpType::LINEAR, OpTarget::None()) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "resize fail..");
            return Status::ERROR;
        }

        if (IResize(m_ctx, *src1, sub_src1, InterpType::LINEAR, OpTarget::None()) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "resize fail..");
            return Status::ERROR;
        }

        // Get border mat sizes
        DT_S32 radius = GetFastKsize(m_ksize) >> 1;
        if (IMakeBorder(m_ctx, sub_src0, m_src_border0, radius, radius, radius, radius, m_border_type, m_border_value, OpTarget::None()) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "make border fail..");
            return Status::ERROR;
        }

        if (IMakeBorder(m_ctx, sub_src1, m_src_border1, radius, radius, radius, radius, m_border_type, m_border_value, OpTarget::None()) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "make border fail..");
            return Status::ERROR;
        }

        switch (src0->GetElemType())
        {
            case ElemType::U8:
            {
                ret = GuideFilterFastNoneImpl<DT_U8>(m_ctx, *src0, sub_src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                     m_border_type, m_border_value, m_target);
                break;
            }
            case ElemType::S8:
            {
                ret = GuideFilterFastNoneImpl<DT_S8>(m_ctx, *src0, sub_src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                     m_border_type, m_border_value, m_target);
                break;
            }
            case ElemType::U16:
            {
                ret = GuideFilterFastNoneImpl<DT_U16>(m_ctx, *src0, sub_src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                      m_border_type, m_border_value, m_target);
                break;
            }
            case ElemType::S16:
            {
                ret = GuideFilterFastNoneImpl<DT_S16>(m_ctx, *src0, sub_src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                      m_border_type, m_border_value, m_target);
                break;
            }
#if defined (AURA_BUILD_HOST)
            case ElemType::F16:
            {
                ret = GuideFilterFastNoneImpl<MI_F16>(m_ctx, *src0, sub_src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                      m_border_type, m_border_value, m_target);
                break;
            }
            case ElemType::F32:
            {
                ret = GuideFilterFastNoneImpl<DT_F32>(m_ctx, *src0, sub_src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                      m_border_type, m_border_value, m_target);
                break;
            }
#endif // AURA_BUILD_HOST
            default:
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Unsupported data type!");
                return Status::ERROR;
            }
        }
    }
    else // GuideFilterType::NORMAL
    {
        // Get border mat sizes
        DT_S32 radius = m_ksize >> 1;
        if (IMakeBorder(m_ctx, *src0, m_src_border0, radius, radius, radius, radius, m_border_type, m_border_value, OpTarget::None()) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "make border fail..");
            return Status::ERROR;
        }

        if (IMakeBorder(m_ctx, *src1, m_src_border1, radius, radius, radius, radius, m_border_type, m_border_value, OpTarget::None()) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "make border fail..");
            return Status::ERROR;
        }

        switch (src0->GetElemType())
        {
            case ElemType::U8:
            {
                ret = GuideFilteNormalNoneImpl<DT_U8>(m_ctx, *src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                      m_border_type, m_border_value, m_target);
                break;
            }
            case ElemType::S8:
            {
                ret = GuideFilteNormalNoneImpl<DT_S8>(m_ctx, *src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                      m_border_type, m_border_value, m_target);
                break;
            }
            case ElemType::U16:
            {
                ret = GuideFilteNormalNoneImpl<DT_U16>(m_ctx, *src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                       m_border_type, m_border_value, m_target);
                break;
            }
            case ElemType::S16:
            {
                ret = GuideFilteNormalNoneImpl<DT_S16>(m_ctx, *src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                       m_border_type, m_border_value, m_target);
                break;
            }
#if defined (AURA_BUILD_HOST)
            case ElemType::F16:
            {
                ret = GuideFilteNormalNoneImpl<MI_F16>(m_ctx, *src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                       m_border_type, m_border_value, m_target);
                break;
            }
            case ElemType::F32:
            {
                ret = GuideFilteNormalNoneImpl<DT_F32>(m_ctx, *src0, m_src_border0, m_src_border1, *dst, m_ksize, m_eps,
                                                       m_border_type, m_border_value, m_target);
                break;
            }
#endif // AURA_BUILD_HOST
            default:
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Unsupported data type!");
                return Status::ERROR;
            }
        }
    }

    AURA_RETURN(m_ctx, ret);
}

Status GuideFilterNone::DeInitialize()
{
    m_src_border0.Release();
    m_src_border1.Release();

    if (GuideFilterImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GuideFilterImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace aura
