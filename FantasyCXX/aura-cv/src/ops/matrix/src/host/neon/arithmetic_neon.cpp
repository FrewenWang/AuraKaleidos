#include "arithmetic_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

enum class ArithmNeonOpType
{
    NEON_OP_VADD = 0,
    NEON_OP_VADDL,
    NEON_OP_VQADD,
    NEON_OP_VSUB,
    NEON_OP_VSUBL,
    NEON_OP_VQSUB,
    NEON_OP_VMUL,
    NEON_OP_VMULL,
};

template <typename Tp, ArithmNeonOpType Type, typename VqType = typename neon::QVector<Tp>::VType,
          typename std::enable_if<(ArithmNeonOpType::NEON_OP_VADD == Type)>::type* = DT_NULL>
AURA_ALWAYS_INLINE VqType NeonOpFunc(VqType vq_src0, VqType vq_src1)
{
    return neon::vadd(vq_src0, vq_src1);
}

template <typename Tp, ArithmNeonOpType Type, typename VdType = typename neon::DVector<Tp>::VType,
          typename VqType = typename neon::WVectorBits<VdType>::VType,
          typename std::enable_if<(ArithmNeonOpType::NEON_OP_VADDL == Type)>::type* = DT_NULL>
AURA_ALWAYS_INLINE VqType NeonOpFunc(VdType vq_src0, VdType vq_src1)
{
    return neon::vaddl(vq_src0, vq_src1);
}

template <typename Tp, ArithmNeonOpType Type, typename VqType = typename neon::QVector<Tp>::VType,
          typename std::enable_if<(ArithmNeonOpType::NEON_OP_VQADD == Type)>::type* = DT_NULL>
AURA_ALWAYS_INLINE VqType NeonOpFunc(VqType vq_src0, VqType vq_src1)
{
    return neon::vqadd(vq_src0, vq_src1);
}

template <typename Tp, ArithmNeonOpType Type, typename VqType = typename neon::QVector<Tp>::VType,
          typename std::enable_if<(ArithmNeonOpType::NEON_OP_VSUB == Type)>::type* = DT_NULL>
AURA_ALWAYS_INLINE VqType NeonOpFunc(VqType vq_src0, VqType vq_src1)
{
    return neon::vsub(vq_src0, vq_src1);
}

template <typename Tp, ArithmNeonOpType Type, typename VdType = typename neon::DVector<Tp>::VType,
          typename VqType = typename neon::WVectorBits<VdType>::VType,
          typename std::enable_if<(ArithmNeonOpType::NEON_OP_VSUBL == Type)>::type* = DT_NULL>
AURA_ALWAYS_INLINE VqType NeonOpFunc(VdType vq_src0, VdType vq_src1)
{
    return neon::vsubl(vq_src0, vq_src1);
}

template <typename Tp, ArithmNeonOpType Type, typename VqType = typename neon::QVector<Tp>::VType,
          typename std::enable_if<(ArithmNeonOpType::NEON_OP_VQSUB == Type)>::type* = DT_NULL>
AURA_ALWAYS_INLINE VqType NeonOpFunc(VqType vq_src0, VqType vq_src1)
{
    return neon::vqsub(vq_src0, vq_src1);
}

template <typename Tp, ArithmNeonOpType Type, typename VqType = typename neon::QVector<Tp>::VType,
          typename std::enable_if<(ArithmNeonOpType::NEON_OP_VMUL == Type)>::type* = DT_NULL>
AURA_ALWAYS_INLINE VqType NeonOpFunc(VqType vq_src0, VqType vq_src1)
{
    return neon::vmul(vq_src0, vq_src1);
}

template <typename Tp, ArithmNeonOpType Type, typename VdType = typename neon::DVector<Tp>::VType,
          typename VqType = typename neon::WVectorBits<VdType>::VType,
          typename std::enable_if<(ArithmNeonOpType::NEON_OP_VMULL == Type)>::type* = DT_NULL>
AURA_ALWAYS_INLINE VqType NeonOpFunc(VdType vq_src0, VdType vq_src1)
{
    return neon::vmull(vq_src0, vq_src1);
}

template <typename Tp, ArithmNeonOpType Type>
static Status ArithmNeonImpl(const Mat &src0, const Mat &src1, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    using VqType = typename neon::QVector<Tp>::VType;
    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);

    DT_S32 width       = dst.GetSizes().m_width * dst.GetSizes().m_channel;
    DT_S32 width_align = width & (-ELEM_COUNTS);

    VqType vq_src0, vq_src1, vq_dst;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src0_row = src0.Ptr<Tp>(y);
        const Tp *src1_row = src1.Ptr<Tp>(y);
        Tp       *dst_row  = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += ELEM_COUNTS)
        {
LOOP_BODY:
            neon::vload(src0_row + x, vq_src0);
            neon::vload(src1_row + x, vq_src1);

            vq_dst = NeonOpFunc<Tp, Type>(vq_src0, vq_src1);
            neon::vstore(dst_row + x, vq_dst);
        }

        if (x < width)
        {
            x = width - ELEM_COUNTS;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

template <typename Tp, ArithmNeonOpType Type>
static Status ArithmPromoteNeonImpl(const Mat &src0, const Mat &src1, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    using Dt        = typename Promote<Tp>::Type;
    using VqType    = typename neon::QVector<Tp>::VType;
    using VdType    = typename neon::DVector<Tp>::VType;
    using VqSumType = typename neon::WVectorBits<VdType>::VType;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);

    DT_S32 width       = dst.GetSizes().m_width * dst.GetSizes().m_channel;
    DT_S32 width_align = width & (-ELEM_COUNTS);

    VqType    vq_src0, vq_src1;
    VqSumType vq_dst_lo, vq_dst_hi;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src0_row = src0.Ptr<Tp>(y);
        const Tp *src1_row = src1.Ptr<Tp>(y);
        Dt       *dst_row  = dst.Ptr<Dt>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += ELEM_COUNTS)
        {
LOOP_BODY:
            neon::vload(src0_row + x, vq_src0);
            neon::vload(src1_row + x, vq_src1);

            vq_dst_lo = NeonOpFunc<Tp, Type>(neon::vgetlow(vq_src0),  neon::vgetlow(vq_src1));
            vq_dst_hi = NeonOpFunc<Tp, Type>(neon::vgethigh(vq_src0), neon::vgethigh(vq_src1));

            neon::vstore(dst_row + x, vq_dst_lo);
            neon::vstore(dst_row + x + (ELEM_COUNTS >> 1), vq_dst_hi);
        }

        if (x < width)
        {
            x = width - ELEM_COUNTS;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

template <typename St, typename Dt>
static Status ArithmUnsignedSubPromoteNeonImpl(const Mat &src0, const Mat &src1, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    using VqStType  = typename neon::QVector<St>::VType;
    using VdStType  = typename neon::DVector<St>::VType;
    using VqStLType = typename neon::WVectorBits<VdStType>::VType;
    using VqDtType  = typename neon::QVector<Dt>::VType;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(St);

    DT_S32 width       = dst.GetSizes().m_width * dst.GetSizes().m_channel;
    DT_S32 width_align = width & (-ELEM_COUNTS);

    VqStType  vq_src0, vq_src1;
    VqStLType vq_src0_lo, vq_src0_hi, vq_src1_lo, vq_src1_hi;
    VqDtType  vq_dst_lo, vq_dst_hi;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const St *src0_row = src0.Ptr<St>(y);
        const St *src1_row = src1.Ptr<St>(y);
        Dt       *dst_row  = dst.Ptr<Dt>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += ELEM_COUNTS)
        {
LOOP_BODY:
            neon::vload(src0_row + x, vq_src0);
            neon::vload(src1_row + x, vq_src1);

            vq_src0_lo = neon::vmovl(neon::vgetlow(vq_src0));
            vq_src0_hi = neon::vmovl(neon::vgethigh(vq_src0));
            vq_src1_lo = neon::vmovl(neon::vgetlow(vq_src1));
            vq_src1_hi = neon::vmovl(neon::vgethigh(vq_src1));

            vq_dst_lo = neon::vsub(neon::vreinterpret(vq_src0_lo), neon::vreinterpret(vq_src1_lo));
            vq_dst_hi = neon::vsub(neon::vreinterpret(vq_src0_hi), neon::vreinterpret(vq_src1_hi));

            neon::vstore(dst_row + x, vq_dst_lo);
            neon::vstore(dst_row + x + (ELEM_COUNTS >> 1), vq_dst_hi);
        }

        if (x < width)
        {
            x = width - ELEM_COUNTS;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status ArithmIntegerMulNeonImpl(const Mat &src0, const Mat &src1, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    using VqType    = typename neon::QVector<Tp>::VType;
    using VdType    = typename neon::DVector<Tp>::VType;
    using VqSumType = typename neon::WVectorBits<VdType>::VType;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);

    DT_S32 width       = dst.GetSizes().m_width * dst.GetSizes().m_channel;
    DT_S32 width_align = width & (-ELEM_COUNTS);

    VqType    vq_src0, vq_src1, vq_dst;
    VqSumType vq_dst_lo, vq_dst_hi;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src0_row = src0.Ptr<Tp>(y);
        const Tp *src1_row = src1.Ptr<Tp>(y);
        Tp       *dst_row  = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += ELEM_COUNTS)
        {
LOOP_BODY:
            neon::vload(src0_row + x, vq_src0);
            neon::vload(src1_row + x, vq_src1);

            vq_dst_lo = neon::vmull(neon::vgetlow(vq_src0),  neon::vgetlow(vq_src1));
            vq_dst_hi = neon::vmull(neon::vgethigh(vq_src0), neon::vgethigh(vq_src1));

            vq_dst = neon::vcombine(neon::vqmovn(vq_dst_lo), neon::vqmovn(vq_dst_hi));
            neon::vstore(dst_row + x, vq_dst);
        }

        if (x < width)
        {
            x = width - ELEM_COUNTS;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status ArithmFloatDivNeonImpl(const Mat &src0, const Mat &src1, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    using VqType = typename neon::QVector<Tp>::VType;
    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);

    DT_S32 width       = dst.GetSizes().m_width * dst.GetSizes().m_channel;
    DT_S32 width_align = width & (-ELEM_COUNTS);

    VqType vq_zero, vq_src0, vq_src1, vq_dst;
    neon::vdup(vq_zero, static_cast<Tp>(0));

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src0_row = src0.Ptr<Tp>(y);
        const Tp *src1_row = src1.Ptr<Tp>(y);
        Tp       *dst_row  = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += ELEM_COUNTS)
        {
LOOP_BODY:
            neon::vload(src0_row + x, vq_src0);
            neon::vload(src1_row + x, vq_src1);

            vq_dst = neon::vdiv(vq_src0, vq_src1);
            vq_dst = neon::vbsl(neon::vceq(vq_src1, vq_zero), vq_zero, vq_dst);
            neon::vstore(dst_row + x, vq_dst);
        }

        if (x < width)
        {
            x = width - ELEM_COUNTS;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

static Status ArithmAddNeonHelper(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst)
{
    Status ret = Status::OK;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    DT_S32 height  = dst.GetSizes().m_height;
    DT_S32 pattern = AURA_MAKE_PATTERN(src0.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U8):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_U8, ArithmNeonOpType::NEON_OP_VQADD>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S8):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_S8, ArithmNeonOpType::NEON_OP_VQADD>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_U16, ArithmNeonOpType::NEON_OP_VQADD>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_S16, ArithmNeonOpType::NEON_OP_VQADD>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U32, ElemType::U32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_U32, ArithmNeonOpType::NEON_OP_VQADD>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S32, ElemType::S32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_S32, ArithmNeonOpType::NEON_OP_VQADD>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmPromoteNeonImpl<DT_U8, ArithmNeonOpType::NEON_OP_VADDL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmPromoteNeonImpl<DT_S8, ArithmNeonOpType::NEON_OP_VADDL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmPromoteNeonImpl<DT_U16, ArithmNeonOpType::NEON_OP_VADDL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmPromoteNeonImpl<DT_S16, ArithmNeonOpType::NEON_OP_VADDL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<MI_F16, ArithmNeonOpType::NEON_OP_VADD>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
#endif // AURA_ENABLE_NEON_FP16
        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_F32, ArithmNeonOpType::NEON_OP_VADD>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported elem type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

static Status ArithmSubNeonHelper(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst)
{
    Status ret = Status::OK;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    DT_S32 height  = dst.GetSizes().m_height;
    DT_S32 pattern = AURA_MAKE_PATTERN(src0.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U8):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_U8, ArithmNeonOpType::NEON_OP_VQSUB>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S8):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_S8, ArithmNeonOpType::NEON_OP_VQSUB>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_U16, ArithmNeonOpType::NEON_OP_VQSUB>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_S16, ArithmNeonOpType::NEON_OP_VQSUB>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U32, ElemType::U32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_U32, ArithmNeonOpType::NEON_OP_VQSUB>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S32, ElemType::S32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_S32, ArithmNeonOpType::NEON_OP_VQSUB>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::S16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmUnsignedSubPromoteNeonImpl<DT_U8, DT_S16>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmPromoteNeonImpl<DT_S8, ArithmNeonOpType::NEON_OP_VSUBL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::S32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmUnsignedSubPromoteNeonImpl<DT_U16, DT_S32>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmPromoteNeonImpl<DT_S16, ArithmNeonOpType::NEON_OP_VSUBL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<MI_F16, ArithmNeonOpType::NEON_OP_VSUB>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
#endif // AURA_ENABLE_NEON_FP16
        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_F32, ArithmNeonOpType::NEON_OP_VSUB>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported elem type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

static Status ArithmMulNeonHelper(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst)
{
    Status ret = Status::OK;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    DT_S32 height  = dst.GetSizes().m_height;
    DT_S32 pattern = AURA_MAKE_PATTERN(src0.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U8):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmIntegerMulNeonImpl<DT_U8>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S8):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmIntegerMulNeonImpl<DT_S8>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmIntegerMulNeonImpl<DT_U16>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmIntegerMulNeonImpl<DT_S16>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U32, ElemType::U32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmIntegerMulNeonImpl<DT_U32>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S32, ElemType::S32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmIntegerMulNeonImpl<DT_S32>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmPromoteNeonImpl<DT_U8, ArithmNeonOpType::NEON_OP_VMULL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmPromoteNeonImpl<DT_S8, ArithmNeonOpType::NEON_OP_VMULL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmPromoteNeonImpl<DT_U16, ArithmNeonOpType::NEON_OP_VMULL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmPromoteNeonImpl<DT_S16, ArithmNeonOpType::NEON_OP_VMULL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<MI_F16, ArithmNeonOpType::NEON_OP_VMUL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
#endif // AURA_ENABLE_NEON_FP16
        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNeonImpl<DT_F32, ArithmNeonOpType::NEON_OP_VMUL>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported elem type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

static Status ArithmDivNeonHelper(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst)
{
    Status ret = Status::OK;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    DT_S32 height  = dst.GetSizes().m_height;
    DT_S32 pattern = AURA_MAKE_PATTERN(src0.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
#if defined(AURA_ENABLE_NEON_FP16)
        case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmFloatDivNeonImpl<MI_F16>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }
#endif // AURA_ENABLE_NEON_FP16
        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmFloatDivNeonImpl<DT_F32>,
                                  std::cref(src0), std::cref(src1), std::ref(dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported elem type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

ArithmeticNeon::ArithmeticNeon(Context *ctx, const OpTarget &target) : ArithmeticImpl(ctx, target)
{}

Status ArithmeticNeon::SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op)
{
    if (ArithmeticImpl::SetArgs(src0, src1, dst, op) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ArithmeticImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src0->GetArrayType() != ArrayType::MAT) || (src1->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ArithmeticNeon::Run()
{
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);
    Mat       *dst  = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src0) || (DT_NULL == src1) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_op_type)
    {
        case ArithmOpType::ADD:
        {
            ret = ArithmAddNeonHelper(m_ctx, *src0, *src1, *dst);
            break;
        }
        case ArithmOpType::SUB:
        {
            ret = ArithmSubNeonHelper(m_ctx, *src0, *src1, *dst);
            break;
        }
        case ArithmOpType::MUL:
        {
            ret = ArithmMulNeonHelper(m_ctx, *src0, *src1, *dst);
            break;
        }
        case ArithmOpType::DIV:
        {
            ret = ArithmDivNeonHelper(m_ctx, *src0, *src1, *dst);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported operator type");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

template <typename Tp>
static Status ScalarDivideMatNeonImpl(DT_F32 scalar, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    using VqType = typename neon::QVector<Tp>::VType;
    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);

    DT_S32 width       = dst.GetSizes().m_width * dst.GetSizes().m_channel;
    DT_S32 width_align = width & (-ELEM_COUNTS);

    VqType vq_zero, vq_scalar, vq_src, vq_dst;
    neon::vdup(vq_zero, static_cast<Tp>(0));
    neon::vdup(vq_scalar, static_cast<Tp>(scalar));

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        Tp       *dst_row = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += ELEM_COUNTS)
        {
LOOP_BODY:
            neon::vload(src_row + x, vq_src);

            vq_dst = neon::vdiv(vq_scalar, vq_src);
            vq_dst = neon::vbsl(neon::vceq(vq_src, vq_zero), vq_zero, vq_dst);
            neon::vstore(dst_row + x, vq_dst);
        }

        if (x < width)
        {
            x = width - ELEM_COUNTS;
            goto LOOP_BODY;
        }
    }

    return Status::OK;
}

ScalarDivideNeon::ScalarDivideNeon(Context *ctx, const OpTarget &target) : ScalarDivideImpl(ctx, target)
{}

Status ScalarDivideNeon::SetArgs(DT_F32 scalar, const Array *src, Array *dst)
{
    if (ScalarDivideImpl::SetArgs(scalar, src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ScalarDivideImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ScalarDivideNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    WorkerPool *wp = m_ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    DT_S32 height  = dst->GetSizes().m_height;
    DT_S32 pattern = AURA_MAKE_PATTERN(src->GetElemType(), dst->GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
        {
            ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ScalarDivideMatNeonImpl<DT_F32>,
                                  m_scalar, std::cref(*src), std::ref(*dst));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported elem type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura