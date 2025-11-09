#include "arithmetic_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

/****************************************************************************************\
*                         Add/Subtract/Multiply/Divide                                   *
\****************************************************************************************/
template <typename Tp = DT_VOID>
struct IntDivideFunctor
{
    constexpr Tp operator()(const Tp& left, const Tp& right) const
    {
        return (0 == right) ? 0 : (left / right);
    }
};

template <typename St, typename Dt, typename Wt, typename Functor>
static Status ArithmNoneImpl(const Mat &src0, const Mat &src1, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    Functor op;

    const DT_S32 width   = src0.GetSizes().m_width;
    const DT_S32 channel = src0.GetSizes().m_channel;
    const DT_S32 num_per_row = width * channel;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const St *src0_row = src0.Ptr<St>(y);
        const St *src1_row = src1.Ptr<St>(y);
        Dt *dst_row = dst.Ptr<Dt>(y);

        for (DT_S32 x = 0; x < num_per_row; x++)
        {
            Wt result = op(SaturateCast<Wt>(src0_row[x]), SaturateCast<Wt>(src1_row[x]));
            dst_row[x] = SaturateCast<Dt>(result);
        }
    }

    return Status::OK;
}

template <typename St, typename Dt>
static Status ArithmIntegerNoneHelper(Context *ctx, ArithmOpType op, const Mat &src0, const Mat &src1, Mat &dst, OpTarget &target)
{
    Status ret = Status::OK;
    DT_S32 height = dst.GetSizes().m_height;

#define ARITHMETIC_NONE_IMPL(type)                                                                      \
    if(target.m_data.none.enable_mt)                                                                    \
    {                                                                                                   \
        WorkerPool *wp = ctx->GetWorkerPool();                                                          \
        if (DT_NULL == wp)                                                                              \
        {                                                                                               \
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");                                         \
            return Status::ERROR;                                                                       \
        }                                                                                               \
        ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNoneImpl<St, Dt, WType, type>,      \
                              std::cref(src0), std::cref(src1), std::ref(dst));                         \
    }                                                                                                   \
    else                                                                                                \
    {                                                                                                   \
        ret = ArithmNoneImpl<St, Dt, WType, type>(src0, src1, dst, static_cast<DT_S32>(0), height);     \
    }                                                                                                   \
                                                                                                        \
    if (ret != Status::OK)                                                                              \
    {                                                                                                   \
        DT_CHAR error_msg[128];                                                                         \
        std::snprintf(error_msg, sizeof(error_msg), "ArithmNoneImpl<%s> failed", #type);                \
        AURA_ADD_ERROR_STRING(ctx, error_msg);                                                          \
    }

    switch (op)
    {
        case ArithmOpType::ADD:
        {
            using WType = typename ArithmIntegerTraits<St, ArithmOpType::ADD>::WType;
            ARITHMETIC_NONE_IMPL(std::plus<WType>);
            break;
        }
        case ArithmOpType::SUB:
        {
            using WType = typename ArithmIntegerTraits<St, ArithmOpType::SUB>::WType;
            ARITHMETIC_NONE_IMPL(std::minus<WType>);
            break;
        }
        case ArithmOpType::MUL:
        {
            using WType = typename ArithmIntegerTraits<St, ArithmOpType::MUL>::WType;
            ARITHMETIC_NONE_IMPL(std::multiplies<WType>);
            break;
        }
        case ArithmOpType::DIV:
        {
            using WType = typename ArithmIntegerTraits<St, ArithmOpType::DIV>::WType;
            ARITHMETIC_NONE_IMPL(IntDivideFunctor<WType>);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported operator type");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename St, typename Dt>
static Status ArithmFloatNoneHelper(Context *ctx, ArithmOpType op, const Mat &src0, const Mat &src1, Mat &dst, OpTarget &target)
{
    Status ret = Status::OK;
    DT_S32 height = dst.GetSizes().m_height;

#define ARITHMETIC_FLOAT_NONE_IMPL(type)                                                                \
    if(target.m_data.none.enable_mt)                                                                    \
    {                                                                                                   \
        WorkerPool *wp = ctx->GetWorkerPool();                                                          \
        if (DT_NULL == wp)                                                                              \
        {                                                                                               \
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");                                         \
            return Status::ERROR;                                                                       \
        }                                                                                               \
        ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ArithmNoneImpl<St, Dt, DT_F32, type> ,    \
                              std::cref(src0), std::cref(src1), std::ref(dst));                         \
    }                                                                                                   \
    else                                                                                                \
    {                                                                                                   \
        ret = ArithmNoneImpl<St, Dt, DT_F32, type>(src0, src1, dst, static_cast<DT_S32>(0), height);    \
    }                                                                                                   \
                                                                                                        \
    if (ret != Status::OK)                                                                              \
    {                                                                                                   \
        DT_CHAR error_msg[128];                                                                         \
        std::snprintf(error_msg, sizeof(error_msg), "ArithmNoneImpl<%s> failed", #type);                \
        AURA_ADD_ERROR_STRING(ctx, error_msg);                                                          \
    }

    switch (op)
    {
        case ArithmOpType::ADD:
        {
            ARITHMETIC_FLOAT_NONE_IMPL(std::plus<DT_F32>);
            break;
        }
        case ArithmOpType::SUB:
        {
            ARITHMETIC_FLOAT_NONE_IMPL(std::minus<DT_F32>);
            break;
        }
        case ArithmOpType::MUL:
        {
            ARITHMETIC_FLOAT_NONE_IMPL(std::multiplies<DT_F32>);
            break;
        }
        case ArithmOpType::DIV:
        {
            ARITHMETIC_FLOAT_NONE_IMPL(std::divides<DT_F32>);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported operator type");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename St, typename Dt, typename std::enable_if<is_integral<St>::value && is_integral<Dt>::value, St>::type* = DT_NULL>
static Status ArithmNoneHelper(Context *ctx, ArithmOpType op, const Mat &src0, const Mat &src1, Mat &dst, OpTarget &target)
{
    Status ret = ArithmIntegerNoneHelper<St, Dt>(ctx, op, src0, src1, dst, target);
    AURA_RETURN(ctx, ret);
}

template <typename St, typename Dt, typename std::enable_if<!(is_integral<St>::value && is_integral<Dt>::value), St>::type* = DT_NULL>
static Status ArithmNoneHelper(Context *ctx, ArithmOpType op, const Mat &src0, const Mat &src1, Mat &dst, OpTarget &target)
{
    Status ret = ArithmFloatNoneHelper<St, Dt>(ctx, op, src0, src1, dst, target);
    AURA_RETURN(ctx, ret);
}

template <typename St>
static Status ArithmNoneHelper(Context *ctx, ArithmOpType op, const Mat &src0, const Mat &src1, Mat &dst, OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (dst.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ArithmNoneHelper<St, DT_U8>(ctx, op, src0, src1, dst, target);
            if (ret !=  Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ArithmNoneHelper<St, DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = ArithmNoneHelper<St, DT_S8>(ctx, op, src0, src1, dst, target);
            if (ret !=  Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ArithmNoneHelper<St, DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = ArithmNoneHelper<St, DT_U16>(ctx, op, src0, src1, dst, target);
            if (ret !=  Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ArithmNoneHelper<St, DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = ArithmNoneHelper<St, DT_S16>(ctx, op, src0, src1, dst, target);
            if (ret !=  Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ArithmNoneHelper<St, DT_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = ArithmNoneHelper<St, DT_U32>(ctx, op, src0, src1, dst, target);
            if (ret !=  Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ArithmNoneHelper<St, DT_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = ArithmNoneHelper<St, DT_S32>(ctx, op, src0, src1, dst, target);
            if (ret !=  Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ArithmNoneHelper<St, DT_S32> failed.");
            }
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = ArithmNoneHelper<St, MI_F16>(ctx, op, src0, src1, dst, target);
            if (ret !=  Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ArithmNoneHelper<St, MI_F16> failed.");
            }
            break;
        }
        case ElemType::F32:
        {
            ret = ArithmNoneHelper<St, DT_F32>(ctx, op, src0, src1, dst, target);
            if (ret !=  Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ArithmNoneHelper<St, DT_F32> failed.");
            }
            break;
        }
#endif
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported element type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

ArithmeticNone::ArithmeticNone(Context *ctx, const OpTarget &target) : ArithmeticImpl(ctx, target)
{}

Status ArithmeticNone::SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op)
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

Status ArithmeticNone::Run()
{
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);
    Mat *dst        = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src0) || (DT_NULL == src1) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src0->GetElemType())
    {
        case ElemType::U8:
        {
            ret = ArithmNoneHelper<DT_U8>(m_ctx, m_op_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ArithmNoneHelper<DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = ArithmNoneHelper<DT_S8>(m_ctx, m_op_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ArithmNoneHelper<DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = ArithmNoneHelper<DT_U16>(m_ctx, m_op_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ArithmNoneHelper<DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = ArithmNoneHelper<DT_S16>(m_ctx, m_op_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ArithmNoneHelper<DT_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = ArithmNoneHelper<DT_U32>(m_ctx, m_op_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ArithmNoneHelper<DT_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = ArithmNoneHelper<DT_S32>(m_ctx, m_op_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ArithmNoneHelper<DT_S32> failed.");
            }
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = ArithmNoneHelper<MI_F16>(m_ctx, m_op_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ArithmNoneHelper<MI_F16> failed.");
            }
            break;
        }
        case ElemType::F32:
        {
            ret = ArithmNoneHelper<DT_F32>(m_ctx, m_op_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ArithmNoneHelper<DT_F32> failed.");
            }
            break;
        }
#endif

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

template <typename Tp0, typename Tp1>
static Status ScalarDivideMatNoneImpl(DT_F32 scalar, const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    const DT_S32 width   = src.GetSizes().m_width;
    const DT_S32 channel = src.GetSizes().m_channel;
    const DT_S32 num_per_row = width * channel;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp0 *src_c = src.Ptr<Tp0>(y);
        Tp1 *dst_row = dst.Ptr<Tp1>(y);

        for (DT_S32 x = 0; x < num_per_row; x++)
        {
            dst_row[x] = SaturateCast<Tp1>(scalar / src_c[x]);
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status ScalarDivideMatNoneHelper(Context *ctx, DT_F32 scalar, const Mat &src, Mat &dst, OpTarget &target)
{
    DT_S32 height = dst.GetSizes().m_height;
    Status ret    = Status::OK;

#define SCALAR_DIVIDE_MAT(type)                                                                     \
    if (target.m_data.none.enable_mt)                                                               \
    {                                                                                               \
        WorkerPool *wp = ctx->GetWorkerPool();                                                      \
        if (DT_NULL == wp)                                                                          \
        {                                                                                           \
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");                                     \
            return Status::ERROR;                                                                   \
        }                                                                                           \
        ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ScalarDivideMatNoneImpl<Tp, type>,    \
                              scalar, std::cref(src), std::ref(dst));                               \
    }                                                                                               \
    else                                                                                            \
    {                                                                                               \
        ret = ScalarDivideMatNoneImpl<Tp, type>(scalar, src, dst, static_cast<DT_S32>(0), height);  \
    }                                                                                               \
    if (ret != Status::OK)                                                                          \
    {                                                                                               \
        DT_CHAR error_msg[128];                                                                     \
        std::snprintf(error_msg, sizeof(error_msg), "ScalarDivideMatNoneImpl<%s> failed", #type);   \
        AURA_ADD_ERROR_STRING(ctx, error_msg);                                                      \
    }

    switch (dst.GetElemType())
    {
        case ElemType::U8:
        {
            SCALAR_DIVIDE_MAT(DT_U8);
            break;
        }
        case ElemType::S8:
        {
            SCALAR_DIVIDE_MAT(DT_S8);
            break;
        }
        case ElemType::U16:
        {
            SCALAR_DIVIDE_MAT(DT_U16);
            break;
        }
        case ElemType::S16:
        {
            SCALAR_DIVIDE_MAT(DT_S16);
            break;
        }
        case ElemType::U32:
        {
            SCALAR_DIVIDE_MAT(DT_U32);
            break;
        }
        case ElemType::S32:
        {
            SCALAR_DIVIDE_MAT(DT_S32);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F32:
        {
            SCALAR_DIVIDE_MAT(DT_F32);
            break;
        }
#endif

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported element type");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

ScalarDivideNone::ScalarDivideNone(Context *ctx, const OpTarget &target) : ScalarDivideImpl(ctx, target)
{}

Status ScalarDivideNone::SetArgs(DT_F32 scalar, const Array *src, Array *dst)
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

Status ScalarDivideNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = ScalarDivideMatNoneHelper<DT_U8>(m_ctx, m_scalar, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ScalarDivideMatNoneHelper<DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = ScalarDivideMatNoneHelper<DT_S8>(m_ctx, m_scalar, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ScalarDivideMatNoneHelper<DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = ScalarDivideMatNoneHelper<DT_U16>(m_ctx, m_scalar, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ScalarDivideMatNoneHelper<DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = ScalarDivideMatNoneHelper<DT_S16>(m_ctx, m_scalar, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ScalarDivideMatNoneHelper<DT_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = ScalarDivideMatNoneHelper<DT_U32>(m_ctx, m_scalar, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ScalarDivideMatNoneHelper<DT_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = ScalarDivideMatNoneHelper<DT_S32>(m_ctx, m_scalar, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ScalarDivideMatNoneHelper<DT_S32> failed.");
            }
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F32:
        {
            ret = ScalarDivideMatNoneHelper<DT_F32>(m_ctx, m_scalar, *src, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ScalarDivideMatNoneHelper<DT_F32> failed.");
            }
            break;
        }
#endif

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura