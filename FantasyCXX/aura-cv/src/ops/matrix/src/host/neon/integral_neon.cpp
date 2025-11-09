#include "integral_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_F32>::value>::type* = DT_NULL>
AURA_INLINE float32x4_t CvtIntegralVector(const int32x4_t &vqs32_src)
{
    return neon::vcvt<DT_F32>(vqs32_src);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_F32>::value>::type* = DT_NULL>
AURA_INLINE float32x4_t CvtIntegralVector(const uint32x4_t &vqu32_src)
{
    return neon::vcvt<DT_F32>(vqu32_src);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_S32>::value>::type* = DT_NULL>
AURA_INLINE int32x4_t CvtIntegralVector(const int32x4_t &vqs32_src)
{
    return vqs32_src;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U32>::value>::type* = DT_NULL>
AURA_INLINE uint32x4_t CvtIntegralVector(const uint32x4_t &vqu32_src)
{
    return vqu32_src;
}

template <typename Tp, typename SumType, 
          typename VTp      = typename neon::QVector<Tp>::VType,
          typename VSumType = typename neon::QVector<SumType>::VType>
AURA_INLINE DT_VOID IntegralVector(const VTp &vq_src, VSumType &vq_left, VSumType &vq_dst_lo, VSumType &vq_dst_hi)
{
    VSumType vqu32_zero = neon::vmovq((DT_U32)0);

    vq_dst_lo = CvtIntegralVector<SumType>(neon::vmovl(neon::vgetlow(vq_src)));
    vq_dst_hi = CvtIntegralVector<SumType>(neon::vmovl(neon::vgethigh(vq_src)));

    // shift 4, and add up
    vq_dst_hi = neon::vadd(vq_dst_hi, vq_dst_lo);

    // shift 1, and add up
    VSumType vq_ext_lo = neon::vext<3>(vqu32_zero,   vq_dst_lo);
    VSumType vq_ext_hi = neon::vext<3>(vq_dst_lo, vq_dst_hi);
    vq_dst_lo          = neon::vadd(vq_dst_lo, vq_ext_lo);
    vq_dst_hi          = neon::vadd(vq_dst_hi, vq_ext_hi);

    // shift 2, and add up
    vq_ext_lo = neon::vext<2>(vqu32_zero,   vq_dst_lo);
    vq_ext_hi = neon::vext<2>(vq_dst_lo, vq_dst_hi);
    vq_dst_lo = neon::vadd(vq_dst_lo, vq_ext_lo);
    vq_dst_hi = neon::vadd(vq_dst_hi, vq_ext_hi);

    // add left vector and update left vector
    vq_dst_lo = neon::vadd(vq_dst_lo, vq_left);
    vq_dst_hi = neon::vadd(vq_dst_hi, vq_left);
    vq_left   = neon::vduplaneq<1>(neon::vgethigh(vq_dst_hi));
}

template <typename Tp, typename SumType>
static Status IntegralBlockNeonImpl(const Mat &src, Mat &dst, DT_S32 block_h, DT_S32 block_w, DT_S32 idx_h, DT_S32 idx_w)
{
    using VSrc        = typename neon::DVector<Tp>::VType;
    using VDst        = typename neon::QVector<SumType>::VType;
    using PromoteType = typename Promote<Tp>::Type;

    constexpr DT_S32 simd_width = 8;
    const DT_S32 height = src.GetSizes().m_height;
    const DT_S32 width  = src.GetSizes().m_width;

    DT_S32 start_row = idx_h * block_h + 1;
    DT_S32 start_col = idx_w * block_w;
    DT_S32 end_row   = Min(start_row + block_h, height);
    DT_S32 end_col   = Min(start_col + block_w, width);

    VSrc vd_src;
    VDst vq_dst_lo, vq_dst_hi;

    // first row
    if (0 == idx_h && 0 == idx_w)
    {
        const Tp *src_c = src.Ptr<Tp>(0);
        SumType  *dst_c = dst.Ptr<SumType>(0);

        VDst vq_dst_left = neon::vmovq(0);

        DT_S32 x = 0;
        for (; x <= width - simd_width; x += simd_width)
        {
            neon::vload(src_c + x, vd_src);

            auto vq_src_sq = neon::vmovl(vd_src);
            IntegralVector<PromoteType, SumType>(vq_src_sq, vq_dst_left, vq_dst_lo, vq_dst_hi);

            neon::vstore(dst_c + x,     vq_dst_lo);
            neon::vstore(dst_c + x + 4, vq_dst_hi);
        }

        if (x < width)
        {
            SumType row_sum = neon::vgetlane<3>(vq_dst_left);
            for (; x < width; x++)
            {
                row_sum += src_c[x];
                dst_c[x] = row_sum;
            }
        }
    }

    VDst vq_prev_lo, vq_prev_hi;
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c = src.Ptr<Tp>(y);
        SumType  *dst_c = dst.Ptr<SumType>(y);
        SumType  *dst_p = dst.Ptr<SumType>(y - 1);

        DT_S32 x = start_col;
        VDst vq_dst_left = start_col > 0 ? neon::vmovq(dst_c[start_col - 1] - dst_p[start_col - 1]) :
                                           neon::vmovq(0);

        for (; x <= end_col - simd_width; x += simd_width)
        {
            neon::vload(dst_p + x,     vq_prev_lo);
            neon::vload(dst_p + x + 4, vq_prev_hi);
            neon::vload(src_c + x,     vd_src);

            auto vq_src_sq = neon::vmovl(vd_src);
            IntegralVector<PromoteType, SumType>(vq_src_sq, vq_dst_left, vq_dst_lo, vq_dst_hi);

            neon::vstore(dst_c + x,     neon::vadd(vq_dst_lo, vq_prev_lo));
            neon::vstore(dst_c + x + 4, neon::vadd(vq_dst_hi, vq_prev_hi));
        }

        if (x < end_col)
        {
            SumType row_sum = neon::vgetlane<3>(vq_dst_left);
            for (; x < end_col; x++)
            {
                row_sum += (SumType)src_c[x];
                dst_c[x] = row_sum + dst_p[x];
            }
        }
    }

    return Status::OK;
}

template <typename Tp, typename SumType>
static Status IntegralNeonImpl(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    DT_S32 height       = src.GetSizes().m_height;
    DT_S32 width        = src.GetSizes().m_width;
    DT_S32 block_height = 256;
    DT_S32 block_width  = 256;

    Status ret = wp->WaveFront((height - 1 + block_height - 1) / block_height, (width + block_width - 1) / block_width,
                               IntegralBlockNeonImpl<Tp, SumType>, src, dst, block_height, block_width);

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status IntegralSqBlockNeonImpl(const Mat &src, Mat &dst, DT_S32 block_h, DT_S32 block_w, DT_S32 idx_h, DT_S32 idx_w)
{
    using DVec = typename neon::DVector<Tp>::VType;

    constexpr DT_S32 simd_width = 8;
    const DT_S32 height = src.GetSizes().m_height;
    const DT_S32 width  = src.GetSizes().m_width;

    DT_S32 start_row = idx_h * block_h + 1;
    DT_S32 start_col = idx_w * block_w;
    DT_S32 end_row   = Min(start_row + block_h, height);
    DT_S32 end_col   = Min(start_col + block_w, width);

    DVec vd_src;
    uint32x4_t vqu32_dst_lo, vqu32_dst_hi;

    // first row
    if (0 == idx_h && 0 == idx_w)
    {
        const Tp *src_c = src.Ptr<Tp>(0);
        DT_U32   *dst_c = dst.Ptr<DT_U32>(0);

        uint32x4_t vqu32_left = neon::vmovq(0);

        DT_S32 x = 0;
        for (; x <= width - simd_width; x += simd_width)
        {
            neon::vload(src_c + x,     vd_src);

            uint16x8_t vqu16_src_sq = neon::vmull(vd_src, vd_src);
            IntegralVector<DT_U16, DT_U32>(vqu16_src_sq, vqu32_left, vqu32_dst_lo, vqu32_dst_hi);

            neon::vstore(dst_c + x,     vqu32_dst_lo);
            neon::vstore(dst_c + x + 4, vqu32_dst_hi);
        }

        if (x < width)
        {
            DT_U32 row_sum = neon::vgetlane<3>(vqu32_left);
            for (; x < width; x++)
            {
                row_sum += (DT_U32)((DT_S32)src_c[x] * src_c[x]);
                dst_c[x] = row_sum;
            }
        }
    }

    uint32x4_t vqu32_prev_lo, vqu32_prev_hi;
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_c = src.Ptr<Tp>(y);
        DT_U32   *dst_c = dst.Ptr<DT_U32>(y);
        DT_U32   *dst_p = dst.Ptr<DT_U32>(y - 1);

        DT_S32 x = start_col;
        uint32x4_t vqu32_left = start_col > 0 ? neon::vmovq(dst_c[start_col - 1] - dst_p[start_col - 1]) :
                                                neon::vmovq(0);

        for (; x <= end_col - simd_width; x += simd_width)
        {
            neon::vload(dst_p + x,     vqu32_prev_lo);
            neon::vload(dst_p + x + 4, vqu32_prev_hi);
            neon::vload(src_c + x,     vd_src);

            uint16x8_t vqu16_src_sq = neon::vmull(vd_src, vd_src);
            IntegralVector<DT_U16, DT_U32>(vqu16_src_sq, vqu32_left, vqu32_dst_lo, vqu32_dst_hi);

            neon::vstore(dst_c + x,     neon::vadd(vqu32_dst_lo, vqu32_prev_lo));
            neon::vstore(dst_c + x + 4, neon::vadd(vqu32_dst_hi, vqu32_prev_hi));
        }

        if (x < end_col)
        {
            DT_U32 row_sum = neon::vgetlane<3>(vqu32_left);
            for (; x < end_col; x++)
            {
                row_sum += (DT_U32)((DT_S32)src_c[x] * src_c[x]);
                dst_c[x] = row_sum + dst_p[x];
            }
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status IntegralSqNeonImpl(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    const DT_S32 height       = src.GetSizes().m_height;
    const DT_S32 width        = src.GetSizes().m_width;
    const DT_S32 block_height = 256;
    const DT_S32 block_width  = 256;

    Status ret = wp->WaveFront((height - 1 + block_height - 1) / block_height, (width + block_width - 1) / block_width,
                               IntegralSqBlockNeonImpl<Tp>, src, dst, block_height, block_width);

    AURA_RETURN(ctx, ret);
}

IntegralNeon::IntegralNeon(Context *ctx, const OpTarget &target) : IntegralImpl(ctx, target)
{}

Status IntegralNeon::SetArgs(const Array *src, Array *dst, Array *dst_sq)
{
    if (IntegralImpl::SetArgs(src, dst, dst_sq) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "IntegralImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be single channel");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    if (dst && dst->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst must be mat type");
        return Status::ERROR;
    }

    if (dst_sq && dst_sq->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst_sq must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status IntegralNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);
    Mat *dst_sq    = dynamic_cast<Mat*>(m_dst_sq);

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if (dst && dst->IsValid())
    {
        switch (AURA_MAKE_PATTERN(src->GetElemType(), dst->GetElemType()))
        {
            case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U32):
            {
                ret = IntegralNeonImpl<DT_U8, DT_U32>(m_ctx, *src, *dst);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::U8, ElemType::F32):
            {
                ret = IntegralNeonImpl<DT_U8, DT_F32>(m_ctx, *src, *dst);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S32):
            {
                ret = IntegralNeonImpl<DT_S8, DT_S32>(m_ctx, *src, *dst);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S8, ElemType::F32):
            {
                ret = IntegralNeonImpl<DT_S8, DT_F32>(m_ctx, *src, *dst);
                break;
            }
            default:
            {
                AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
                ret = Status::ERROR;
                break;
            }
        }
    }

    if (dst_sq && dst_sq->IsValid())
    {
        switch (AURA_MAKE_PATTERN(src->GetElemType(), dst_sq->GetElemType()))
        {
            case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U32):
            {
                ret = IntegralSqNeonImpl<DT_U8>(m_ctx, *src, *dst_sq);
                break;
            }
            case AURA_MAKE_PATTERN(ElemType::S8, ElemType::U32):
            {
                ret = IntegralSqNeonImpl<DT_S8>(m_ctx, *src, *dst_sq);
                break;
            }
            default:
            {
                AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
                ret = Status::ERROR;
                break;
            }
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura