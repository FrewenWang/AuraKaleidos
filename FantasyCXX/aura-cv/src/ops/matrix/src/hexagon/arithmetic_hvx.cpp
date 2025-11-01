#include "matrix_comm.hpp"
#include "arithmetic_impl.hpp"
#include "aura/ops/matrix/arithmetic.hpp"
#include "aura/runtime/worker_pool.h"

namespace aura
{
/****************************************************************************************\
*                                    arithmetic Add                                      *
\****************************************************************************************/
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U8>::value && std::is_same<Dt, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticAddCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vub_vadd_VubVub_sat(v_src0, v_src1);
    return v_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U8>::value && std::is_same<Dt, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticAddCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Wuh_vadd_WuhWuh_sat(Q6_Wuh_vunpack_Vub(v_src1), Q6_Wuh_vunpack_Vub(v_src0));
    return w_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S8>::value && std::is_same<Dt, MI_S8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticAddCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vb_vadd_VbVb_sat(v_src0, v_src1);
    return v_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S8>::value && std::is_same<Dt, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticAddCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Wh_vadd_WhWh_sat(Q6_Wh_vunpack_Vb(v_src1), Q6_Wh_vunpack_Vb(v_src0));
    return w_result;
}

// U/S16 ===> U/S16  U/S32
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U16>::value && std::is_same<Dt, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticAddCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vuh_vadd_VuhVuh_sat(v_src0, v_src1);
    return v_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U16>::value && std::is_same<Dt, MI_U32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticAddCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Wuw_vadd_WuwWuw_sat(Q6_Wuw_vunpack_Vuh(v_src0), Q6_Wuw_vunpack_Vuh(v_src1));
    return w_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S16>::value && std::is_same<Dt, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticAddCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vh_vadd_VhVh_sat(v_src1, v_src0);
    return v_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S16>::value && std::is_same<Dt, MI_S32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticAddCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Ww_vadd_WwWw_sat(Q6_Ww_vunpack_Vh(v_src0), Q6_Ww_vunpack_Vh(v_src1));
    return w_result;
}

// U/S32 ===> U/S32
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U32>::value && std::is_same<Dt, MI_U32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticAddCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vuw_vadd_VuwVuw_sat(v_src1, v_src0);
    return v_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S32>::value && std::is_same<Dt, MI_S32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticAddCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vw_vadd_VwVw_sat(v_src1, v_src0);
    return v_result;
}

/****************************************************************************************\
*                                    arithmetic Sub                                      *
\****************************************************************************************/
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U8>::value && std::is_same<Dt, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticSubCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vub_vsub_VubVub_sat(v_src0, v_src1);
    return v_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S8>::value && std::is_same<Dt, MI_S8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticSubCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vb_vsub_VbVb_sat(v_src0, v_src1);
    return v_result;
}

// U/S16 ===> U/S16  U/S32
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U16>::value && std::is_same<Dt, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticSubCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vuh_vsub_VuhVuh_sat(v_src0, v_src1);
    return v_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S16>::value && std::is_same<Dt, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticSubCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vh_vsub_VhVh_sat(v_src0, v_src1);
    return v_result;
}

// U/S32 ===> U/S32
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U32>::value && std::is_same<Dt, MI_U32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticSubCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vuw_vsub_VuwVuw_sat(v_src0, v_src1);
    return v_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S32>::value && std::is_same<Dt, MI_S32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector ArithmeticSubCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_Vector v_result;
    v_result = Q6_Vw_vsub_VwVw_sat(v_src0, v_src1);
    return v_result;
}

// U8 - U8 ===> s16
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U8>::value && std::is_same<Dt, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticSubCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Wh_vsub_WhWh_sat(Q6_Wuh_vunpack_Vub(v_src0), Q6_Wuh_vunpack_Vub(v_src1));
    return w_result;
}

// S8 - S8 ===> S16
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S8>::value && std::is_same<Dt, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticSubCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Wh_vsub_WhWh_sat(Q6_Wh_vunpack_Vb(v_src0), Q6_Wh_vunpack_Vb(v_src1));
    return w_result;
}

// U16 - U16 ===> S32
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U16>::value && std::is_same<Dt, MI_S32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticSubCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Ww_vsub_WwWw_sat(Q6_Wuw_vunpack_Vuh(v_src0), Q6_Wuw_vunpack_Vuh(v_src1));
    return w_result;
}

// S16 - S16 ===> S32
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S16>::value && std::is_same<Dt, MI_S32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticSubCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Ww_vsub_WwWw_sat(Q6_Ww_vunpack_Vh(v_src0), Q6_Ww_vunpack_Vh(v_src1));
    return w_result;
}

/****************************************************************************************\
*                                    arithmetic Mul                                      *
\****************************************************************************************/
template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U8>::value && std::is_same<Dt, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticMulCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Wuh_vmpy_VubVub(v_src0, v_src1);
    w_result = Q6_W_vshuff_VVR(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), -2);
    return w_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S8>::value && std::is_same<Dt, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticMulCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Wh_vmpy_VbVb(v_src0, v_src1);
    w_result = Q6_W_vshuff_VVR(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), -2);
    return w_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_U16>::value && std::is_same<Dt, MI_U32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticMulCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Wuw_vmpy_VuhVuh(v_src0, v_src1);
    w_result = Q6_W_vshuff_VVR(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), -4);
    return w_result;
}

template <typename St, typename Dt, typename std::enable_if<std::is_same<St, MI_S16>::value && std::is_same<Dt, MI_S32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_VectorPair ArithmeticMulCore(HVX_Vector &v_src0, HVX_Vector &v_src1)
{
    HVX_VectorPair w_result;
    w_result = Q6_Ww_vmpy_VhVh(v_src0, v_src1);
    w_result = Q6_W_vshuff_VVR(Q6_V_hi_W(w_result), Q6_V_lo_W(w_result), -4);
    return w_result;
}

template <typename St, typename Dt, ArithmOpType OP_TYPE>
struct ArithmFunctor;

template <typename St, typename Dt>
struct ArithmFunctor<St, Dt, ArithmOpType::ADD>
{
    using RetType = typename std::conditional<std::is_same<St, Dt>::value, HVX_Vector, HVX_VectorPair>::type;
    RetType operator()(HVX_Vector &v_src0, HVX_Vector &v_src1)
    {
        return ArithmeticAddCore<St, Dt>(v_src0, v_src1);
    }
};

template <typename St, typename Dt>
struct ArithmFunctor<St, Dt, ArithmOpType::SUB>
{
    using RetType = typename std::conditional<std::is_same<St, Dt>::value, HVX_Vector, HVX_VectorPair>::type;
    RetType operator()(HVX_Vector &v_src0, HVX_Vector &v_src1)
    {
        return ArithmeticSubCore<St, Dt>(v_src0, v_src1);
    }
};

template <typename St, typename Dt>
struct ArithmFunctor<St, Dt, ArithmOpType::MUL>
{
    using RetType = typename std::conditional<std::is_same<St, Dt>::value, HVX_Vector, HVX_VectorPair>::type;
    RetType operator()(HVX_Vector &v_src0, HVX_Vector &v_src1)
    {
        return ArithmeticMulCore<St, Dt>(v_src0, v_src1);
    }
};

template <typename St, typename Dt, ArithmOpType OP_TYPE>
static Status ArithmeticHvxImpl(const Mat &src0, const Mat &src1, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    constexpr MI_S32  elem_counts = AURA_HVLEN / sizeof(St);

    MI_S32 width  = src0.GetSizes().m_width;
    MI_S32 height = src0.GetSizes().m_height;
    MI_S32 ch     = src0.GetSizes().m_channel;

    ArithmFunctor<St, Dt, OP_TYPE> func;
    HVX_Vector v_src0;
    HVX_Vector v_src1;

    MI_S32 src0_stride = src0.GetStrides().m_width;
    MI_S32 src1_stride = src1.GetStrides().m_width;
    MI_S32 width_total = width * ch;

    MI_S32 width_align = width_total & (-elem_counts);
    MI_S32 width_rest  = width_total - width_align;

    MI_U64 L2fetch_param0 = L2PfParam(src0_stride, width * ch * ElemTypeSize(src0.GetElemType()), 1, 0);
    MI_U64 L2fetch_param1 = L2PfParam(src1_stride, width * ch * ElemTypeSize(src1.GetElemType()), 1, 0);

    for (MI_S32 i = start_row; i < end_row; i++)
    {
        if (i + 1 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src0.Ptr<St>(i + 1)), L2fetch_param0);
            L2Fetch(reinterpret_cast<MI_U32>(src1.Ptr<St>(i + 1)), L2fetch_param1);
        }

        const St *src0_row = src0.Ptr<St>(i);
        const St *src1_row = src1.Ptr<St>(i);
        Dt *dst_row = dst.Ptr<Dt>(i);

        for (MI_S32 x = 0; x < width_align; x += elem_counts)
        {
            vload(src0_row + x, v_src0);
            vload(src1_row + x, v_src1);
            auto v_dst = func(v_src0, v_src1);
            vstore(dst_row + x, v_dst);
        }

        if (width_rest)
        {
            MI_S32 back_offset = width_total - elem_counts;
            vload(src0_row + back_offset, v_src0);
            vload(src1_row + back_offset, v_src1);
            auto v_dst = func(v_src0, v_src1);
            vstore(dst_row + back_offset, v_dst);
        }
    }

    return Status::OK;
}

template <typename St, typename Dt, ArithmOpType OP_TYPE>
static Status ArithmeticHvxHelper(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool fail");
        return Status::ERROR;
    }

    MI_S32 height = src0.GetSizes().m_height;
    Status ret = wp->ParallelFor((MI_S32)0, height, ArithmeticHvxImpl<St, Dt, OP_TYPE>, std::cref(src0), std::cref(src1), std::ref(dst));

    AURA_RETURN(ctx, ret);
}

ArithmeticHvx::ArithmeticHvx(Context *ctx, const OpTarget &target) : ArithmeticImpl(ctx, target)
{}

Status ArithmeticHvx::SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op)
{
    if (ArithmeticImpl::SetArgs(src0, src1, dst, op) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ArithmeticImpl::Initialize failed");
        return Status::ERROR;
    }

    MI_S32 pattern = AURA_MAKE_PATTERN(op, src0->GetElemType(), dst->GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::U8,  ElemType::U8):
        case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::U8,  ElemType::U16):
        case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::S8,  ElemType::S8):
        case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::S8,  ElemType::S16):
        case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::U16, ElemType::U16):
        case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::U16, ElemType::U32):
        case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::S16, ElemType::S16):
        case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::S16, ElemType::S32):
        case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::U32, ElemType::U32):
        case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::S32, ElemType::S32):
        case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::U8,  ElemType::U8):
        case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::U8,  ElemType::S16):
        case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::S8,  ElemType::S8):
        case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::S8,  ElemType::S16):
        case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::U16, ElemType::U16):
        case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::U16, ElemType::S32):
        case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::S16, ElemType::S16):
        case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::S16, ElemType::S32):
        case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::U32, ElemType::U32):
        case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::S32, ElemType::S32):
        case AURA_MAKE_PATTERN(ArithmOpType::MUL, ElemType::U8,  ElemType::U16):
        case AURA_MAKE_PATTERN(ArithmOpType::MUL, ElemType::S8,  ElemType::S16):
        case AURA_MAKE_PATTERN(ArithmOpType::MUL, ElemType::U16, ElemType::U32):
        case AURA_MAKE_PATTERN(ArithmOpType::MUL, ElemType::S16, ElemType::S32):
        {
            return Status::OK;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "data type not supported");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status ArithmeticHvx::Run()
{
    const Mat *src0 = dynamic_cast<const Mat *>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat *>(m_src1);
    Mat *dst = dynamic_cast<Mat *>(m_dst);

    if ((NULL == src0) || (NULL == src1) || (NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    ElemType src_elem_type = src0->GetElemType();
    ElemType dst_elem_type = dst->GetElemType();

    MI_S32 pattern = AURA_MAKE_PATTERN(src_elem_type, dst_elem_type, m_op_type);

    switch (pattern)
    {
        // add
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U8, ArithmOpType::ADD):
        {
            ret = ArithmeticHvxHelper<MI_U8, MI_U8, ArithmOpType::ADD>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U16, ArithmOpType::ADD):
        {
            ret = ArithmeticHvxHelper<MI_U8, MI_U16, ArithmOpType::ADD>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S8, ArithmOpType::ADD):
        {
            ret = ArithmeticHvxHelper<MI_S8, MI_S8, ArithmOpType::ADD>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8, ElemType::S16, ArithmOpType::ADD):
        {
            ret = ArithmeticHvxHelper<MI_S8, MI_S16, ArithmOpType::ADD>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16, ArithmOpType::ADD):
        {
            ret = ArithmeticHvxHelper<MI_U16, MI_U16, ArithmOpType::ADD>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U32, ArithmOpType::ADD):
        {
            ret = ArithmeticHvxHelper<MI_U16, MI_U32, ArithmOpType::ADD>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16, ArithmOpType::ADD):
        {
            ret = ArithmeticHvxHelper<MI_S16, MI_S16, ArithmOpType::ADD>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S32, ArithmOpType::ADD):
        {
            ret = ArithmeticHvxHelper<MI_S16, MI_S32, ArithmOpType::ADD>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U32, ElemType::U32, ArithmOpType::ADD):
        {
            ret = ArithmeticHvxHelper<MI_U32, MI_U32, ArithmOpType::ADD>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S32, ElemType::S32, ArithmOpType::ADD):
        {
            ret = ArithmeticHvxHelper<MI_S32, MI_S32, ArithmOpType::ADD>(m_ctx, *src0, *src1, *dst);
            break;
        }

        // sub
        case AURA_MAKE_PATTERN(ElemType::U8,  ElemType::U8,  ArithmOpType::SUB):
        {
            ret = ArithmeticHvxHelper<MI_U8,  MI_U8,  ArithmOpType::SUB>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U8,  ElemType::S16, ArithmOpType::SUB):
        {
            ret = ArithmeticHvxHelper<MI_U8,  MI_S16, ArithmOpType::SUB>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8,  ElemType::S8,  ArithmOpType::SUB):
        {
            ret = ArithmeticHvxHelper<MI_S8,  MI_S8,  ArithmOpType::SUB>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8,  ElemType::S16, ArithmOpType::SUB):
        {
            ret = ArithmeticHvxHelper<MI_S8,  MI_S16, ArithmOpType::SUB>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16, ArithmOpType::SUB):
        {
            ret = ArithmeticHvxHelper<MI_U16, MI_U16, ArithmOpType::SUB>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::S32, ArithmOpType::SUB):
        {
            ret = ArithmeticHvxHelper<MI_U16, MI_S32, ArithmOpType::SUB>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16, ArithmOpType::SUB):
        {
            ret = ArithmeticHvxHelper<MI_S16, MI_S16, ArithmOpType::SUB>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S32, ArithmOpType::SUB):
        {
            ret = ArithmeticHvxHelper<MI_S16, MI_S32, ArithmOpType::SUB>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U32, ElemType::U32, ArithmOpType::SUB):
        {
            ret = ArithmeticHvxHelper<MI_U32, MI_U32, ArithmOpType::SUB>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S32, ElemType::S32, ArithmOpType::SUB):
        {
            ret = ArithmeticHvxHelper<MI_S32, MI_S32, ArithmOpType::SUB>(m_ctx, *src0, *src1, *dst);
            break;
        }

        // mul
        case AURA_MAKE_PATTERN(ElemType::U8,  ElemType::U16, ArithmOpType::MUL):
        {
            ret = ArithmeticHvxHelper<MI_U8,  MI_U16, ArithmOpType::MUL>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S8,  ElemType::S16, ArithmOpType::MUL):
        {
            ret = ArithmeticHvxHelper<MI_S8,  MI_S16, ArithmOpType::MUL>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U32, ArithmOpType::MUL):
        {
            ret = ArithmeticHvxHelper<MI_U16, MI_U32, ArithmOpType::MUL>(m_ctx, *src0, *src1, *dst);
            break;
        }
        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S32, ArithmOpType::MUL):
        {
            ret = ArithmeticHvxHelper<MI_S16, MI_S32, ArithmOpType::MUL>(m_ctx, *src0, *src1, *dst);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported op type or data type");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string ArithmeticHvx::ToString() const
{
    return ArithmeticImpl::ToString();
}

Status ArithmeticRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src0, src1, dst;
    ArithmOpType op_type;

    ArithmeticInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src0, src1, dst, op_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Rpc transmission param failed");
        return Status::ERROR;
    }

    Arithmetic arithmetic(ctx, OpTarget::Hvx());

    return OpCall(ctx, arithmetic, &src0, &src1, &dst, op_type);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_MATRIX_PACKAGE_NAME, AURA_OPS_MATRIX_ARITHMETIC_OP_NAME, ArithmeticRpc);

} // namespace aura