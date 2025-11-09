#include "boxfilter_impl.hpp"
#include "make_border_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/thread_buffer.h"

namespace aura
{

template <typename Tp>
struct BoxFilterTraits
{
    using RowSumType    = typename Promote<Tp>::Type;
    using KernelSumType = typename std::conditional<sizeof(Tp) == 1, typename Promote<RowSumType>::Type, typename Promote<Tp>::Type>::type;
};

template <typename Tp, typename RowSumType>
AURA_ALWAYS_INLINE DT_VOID BoxFilterKxKAddTwoRowCore(Tp *src0, Tp *src1, RowSumType *dst)
{
    static_assert(std::is_integral<Tp>::value && std::is_integral<RowSumType>::value && (sizeof(RowSumType) == 2 * sizeof(Tp)),
                  "Tp and RowSumType must be integral type and RowSumType size must be twice of Tp size.");

    using VqType = typename neon::QVector<RowSumType>::VType;

    VqType vq_src_sum = neon::vaddl(neon::vload1(src0), neon::vload1(src1));
    neon::vstore(dst, neon::vadd(neon::vload1q(dst), vq_src_sum));
}

#if defined(AURA_ENABLE_NEON_FP16)
template <>
AURA_ALWAYS_INLINE DT_VOID BoxFilterKxKAddTwoRowCore(MI_F16 *src0, MI_F16 *src1, DT_F32 *dst)
{
    float32x4_t vqf32_src0    = neon::vcvt<DT_F32>(neon::vload1(src0));
    float32x4_t vqf32_src1    = neon::vcvt<DT_F32>(neon::vload1(src1));
    float32x4_t vqf32_src_sum = neon::vadd(vqf32_src0, vqf32_src1);
    neon::vstore(dst, neon::vadd(neon::vload1q(dst), vqf32_src_sum));
}
#endif // AURA_ENABLE_NEON_FP16

template <>
AURA_ALWAYS_INLINE DT_VOID BoxFilterKxKAddTwoRowCore(DT_F32 *src0, DT_F32 *src1, DT_F32 *dst)
{
    float32x4_t vqf32_src_sum = neon::vadd(neon::vload1q(src0), neon::vload1q(src1));
    neon::vstore(dst, neon::vadd(neon::vload1q(dst), vqf32_src_sum));
}

template<typename Tp, typename RowSumType>
AURA_ALWAYS_INLINE DT_VOID BoxFilterKxKAddTwoRow(Tp *src0, Tp *src1, RowSumType *dst, const DT_S32 width)
{
    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(RowSumType);

    DT_S32 x = 0;
    for (; x < width - ELEM_COUNTS; x += ELEM_COUNTS)
    {
        BoxFilterKxKAddTwoRowCore<Tp, RowSumType>(src0 + x, src1 + x, dst + x);
    }

    for (; x < width; x++)
    {
        dst[x] += (src0[x] + src1[x]);
    }
}

template <typename Tp, typename RowSumType>
AURA_ALWAYS_INLINE DT_VOID BoxFilterKxKSubAddRowCore(Tp *src0, Tp *src1, RowSumType *dst)
{
    static_assert(std::is_integral<Tp>::value && std::is_integral<RowSumType>::value && (sizeof(RowSumType) == 2 * sizeof(Tp)),
                  "Tp and RowSumType must be integral type and RowSumType size must be twice of Tp size.");

    using VqType = typename neon::QVector<RowSumType>::VType;

    VqType vq_dst = neon::vload1q(dst);
    VqType vq_sub = neon::vsubw(vq_dst, neon::vload1(src0));
    VqType vq_add = neon::vaddw(vq_sub, neon::vload1(src1));
    neon::vstore(dst, vq_add);
}

#if defined(AURA_ENABLE_NEON_FP16)
template <>
AURA_ALWAYS_INLINE DT_VOID BoxFilterKxKSubAddRowCore(MI_F16 *src0, MI_F16 *src1, DT_F32 *dst)
{
    float32x4_t vqf32_dst = neon::vload1q(dst);
    float32x4_t vqf32_sub = neon::vsub(vqf32_dst, neon::vcvt<DT_F32>(neon::vload1(src0)));
    float32x4_t vqf32_add = neon::vadd(vqf32_sub, neon::vcvt<DT_F32>(neon::vload1(src1)));
    neon::vstore(dst, vqf32_add);
}
#endif // AURA_ENABLE_NEON_FP16

template <>
AURA_ALWAYS_INLINE DT_VOID BoxFilterKxKSubAddRowCore(DT_F32 *src0, DT_F32 *src1, DT_F32 *dst)
{
    float32x4_t vqf32_dst = neon::vload1q(dst);
    float32x4_t vqf32_sub = neon::vsub(vqf32_dst, neon::vload1q(src0));
    float32x4_t vqf32_add = neon::vadd(vqf32_sub, neon::vload1q(src1));
    neon::vstore(dst, vqf32_add);
}

template<typename Tp, typename RowSumType>
AURA_ALWAYS_INLINE DT_VOID BoxFilterKxKSubAddRow(Tp *src_sub, Tp *src_add, RowSumType *dst, const DT_S32 width)
{
    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(RowSumType);

    DT_S32 x = 0;
    for (; x < width - ELEM_COUNTS; x += ELEM_COUNTS)
    {
        BoxFilterKxKSubAddRowCore<Tp, RowSumType>(src_sub + x , src_add + x, dst + x);
    }

    for (; x < width; x++)
    {
        dst[x] = dst[x] - src_sub[x] + src_add[x];
    }
}

template<typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static Status BoxFilterKxKNeonImpl(Context *ctx, const Mat &src, Mat &dst, ThreadBuffer &src_border_thread_buffer, ThreadBuffer &row_sum_thread_buffer,
                                   DT_S32 ksize, DT_S32 block_size, const Scalar &border_value, DT_S32 start_blk, DT_S32 end_blk)
{
    using RowSumType    = typename BoxFilterTraits<Tp>::RowSumType;
    using KernelSumType = typename BoxFilterTraits<Tp>::KernelSumType;

    const DT_S32 ksize_sq     = ksize * ksize;
    const DT_S32 ksh          = ksize >> 1;
    const DT_S32 width        = src.GetSizes().m_width;
    const DT_S32 height       = dst.GetSizes().m_height;
    const DT_S32 border_width = (width + 2 * ksh) * C;
    const DT_S32 start_row    = start_blk * block_size;
    const DT_S32 end_row      = Min(end_blk * block_size, height);

    Tp *src_border_data = src_border_thread_buffer.GetThreadData<Tp>();
    RowSumType *row_sum = row_sum_thread_buffer.GetThreadData<RowSumType>();
    if ((DT_NULL == src_border_data) || (DT_NULL == row_sum))
    {
        AURA_ADD_ERROR_STRING(ctx, "malloc failed");
        return Status::ERROR;
    }

    std::vector<Tp*> src_border(ksize + 1);
    for (DT_S32 i = 0; i < (ksize + 1); i++)
    {
        src_border[i] = src_border_data + i * border_width;
    }

    // init row_sum and head row as 0
    memset(row_sum,       0, border_width * sizeof(RowSumType));
    memset(src_border[0], 0, border_width * sizeof(Tp));

    for (DT_S32 k = 0; k < ksize - 1; k += 2)
    {
        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src, start_row + k - ksh,     width, ksize, src_border[k + 1], border_value);
        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src, start_row + k - ksh + 1, width, ksize, src_border[k + 2], border_value);

        BoxFilterKxKAddTwoRow<Tp, RowSumType>(src_border[k + 1], src_border[k + 2], row_sum, border_width);
    }

    // 2.loop rows
    DT_S32 idx_head = 0;
    DT_S32 idx_tail = ksize;
    KernelSumType kernel_sum[C];

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        Tp *dst_row = dst.Ptr<Tp>(y);

        MakeBorderOneRow<Tp, BORDER_TYPE, C>(src, y + ksh, width, ksize, src_border[idx_tail], border_value);
        BoxFilterKxKSubAddRow<Tp, RowSumType>(src_border[idx_head], src_border[idx_tail], row_sum, border_width);

        // update idx
        idx_head = (idx_head + 1) % (ksize + 1);
        idx_tail = (idx_tail + 1) % (ksize + 1);

        // calc first C pixel
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            kernel_sum[ch] = 0;
        }

        for (DT_S32 k = 0; k < ksize; k++)
        {
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                kernel_sum[ch] += row_sum[k * C + ch];
            }
        }

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            dst_row[ch] = kernel_sum[ch] / ksize_sq;
        }

        // slide window
        for (DT_S32 lx = 1, rx = ksize; lx < width; lx++, rx++)
        {
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                kernel_sum[ch]       = kernel_sum[ch] - row_sum[(lx - 1) * C + ch] + row_sum[rx * C + ch];
                dst_row[lx * C + ch] = static_cast<Tp>(kernel_sum[ch] / ksize_sq);
            }
        }
    }

    return Status::OK;
}

template<typename Tp, BorderType BORDER_TYPE>
static Status BoxFilterKxKNeonHelper(Context *ctx, const Mat &src, Mat &dst, const DT_S32 ksize, const Scalar &border_value)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 width   = src.GetSizes().m_width;
    DT_S32 channel = src.GetSizes().m_channel;

    using RowSumType = typename BoxFilterTraits<Tp>::RowSumType;
    DT_S32 ksh = ksize >> 1;
    DT_S32 src_border_buffer_size = (width + ksh * 2) * channel * sizeof(Tp) * (ksize + 1);
    DT_S32 row_sum_buffer_size    = (width + ksh * 2) * channel * sizeof(RowSumType);

    ThreadBuffer src_border_thread_buffer(ctx, src_border_buffer_size);
    ThreadBuffer row_sum_thread_buffer(ctx, row_sum_buffer_size);

    DT_S32 block_size = Max<DT_S32>(2 * ksize,  65536 / width / sizeof(Tp));
    DT_S32 num_blocks = AURA_ALIGN(height, block_size) / block_size;

    switch(channel)
    {
        case 1:
        {
            ret  = wp->ParallelFor(0, num_blocks, BoxFilterKxKNeonImpl<Tp, BORDER_TYPE, 1>, ctx, std::cref(src), std::ref(dst),
                                   std::ref(src_border_thread_buffer), std::ref(row_sum_thread_buffer), ksize, block_size, border_value);
            break;
        }

        case 2:
        {
            ret  = wp->ParallelFor(0, num_blocks, BoxFilterKxKNeonImpl<Tp, BORDER_TYPE, 2>, ctx, std::cref(src), std::ref(dst),
                                   std::ref(src_border_thread_buffer), std::ref(row_sum_thread_buffer), ksize, block_size, border_value);
            break;
        }

        case 3:
        {
            ret  = wp->ParallelFor(0, num_blocks, BoxFilterKxKNeonImpl<Tp, BORDER_TYPE, 3>, ctx, std::cref(src), std::ref(dst),
                                   std::ref(src_border_thread_buffer), std::ref(row_sum_thread_buffer), ksize, block_size, border_value);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported channel");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

template<typename Tp>
static Status BoxFilterKxKNeonHelper(Context *ctx, const Mat &src, Mat &dst, const DT_S32 ksize, const BorderType border_type,
                                        const Scalar &border_value)
{
    Status ret = Status::ERROR;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            ret = BoxFilterKxKNeonHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, ksize, border_value);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = BoxFilterKxKNeonHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, ksize, border_value);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = BoxFilterKxKNeonHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, ksize, border_value);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "border_type is not supported");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status BoxFilterKxKNeon(Context *ctx, const Mat &src, Mat &dst, const DT_S32 ksize, BorderType border_type,
                           const Scalar &border_value, const OpTarget &target)
{
    Status ret = Status::ERROR;

    AURA_UNUSED(target);

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = BoxFilterKxKNeonHelper<DT_U8>(ctx, src, dst, ksize, border_type, border_value);
            break;
        }

        case ElemType::S8:
        {
            ret = BoxFilterKxKNeonHelper<DT_U8>(ctx, src, dst, ksize, border_type, border_value);
            break;
        }

        case ElemType::U16:
        {
            ret = BoxFilterKxKNeonHelper<DT_U16>(ctx, src, dst, ksize, border_type, border_value);
            break;
        }

        case ElemType::S16:
        {
            ret = BoxFilterKxKNeonHelper<DT_S16>(ctx, src, dst, ksize, border_type, border_value);
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = BoxFilterKxKNeonHelper<MI_F16>(ctx, src, dst, ksize, border_type, border_value);
            break;
        }
#endif // AURA_ENABLE_NEON_FP16

        case ElemType::F32:
        {
            ret = BoxFilterKxKNeonHelper<DT_F32>(ctx, src, dst, ksize, border_type, border_value);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura