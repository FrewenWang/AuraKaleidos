#include "aura/ops/misc/threshold.hpp"
#include "threshold_impl.hpp"
#include "misc_comm.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

// AURA_THRESH_BINARY, U8
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_BINARY == THRESH_TYPE &&
          std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vu8_src, Tp thresh, Tp max_val, HVX_Vector &vu8_result)
{
    HVX_VectorPred qu8_sign = Q6_Q_vcmp_gt_VubVub(vu8_src, Q6_Vb_vsplat_R(thresh));
    vu8_result = Q6_V_vmux_QVV(qu8_sign, Q6_Vb_vsplat_R(max_val), Q6_V_vzero());
}

// AURA_THRESH_BINARY, U16
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_BINARY == THRESH_TYPE &&
          std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vu16_src, Tp thresh, Tp max_val, HVX_Vector &vu16_result)
{
    HVX_VectorPred qu16_sign = Q6_Q_vcmp_gt_VuhVuh(vu16_src, Q6_Vh_vsplat_R(thresh));
    vu16_result = Q6_V_vmux_QVV(qu16_sign, Q6_Vh_vsplat_R(max_val), Q6_V_vzero());
}

// AURA_THRESH_BINARY, S16
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_BINARY == THRESH_TYPE &&
          std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vs16_src, Tp thresh, Tp max_val, HVX_Vector &vs16_result)
{
    HVX_VectorPred qs16_sign = Q6_Q_vcmp_gt_VhVh(vs16_src, Q6_Vh_vsplat_R(thresh));
    vs16_result = Q6_V_vmux_QVV(qs16_sign, Q6_Vh_vsplat_R(max_val), Q6_V_vzero());
}

// AURA_THRESH_BINARY_INV, U8
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_BINARY_INV == THRESH_TYPE &&
          std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vu8_src, Tp thresh, Tp max_val, HVX_Vector &vu8_result)
{
    HVX_VectorPred qu8_sign = Q6_Q_vcmp_gt_VubVub(vu8_src, Q6_Vb_vsplat_R(thresh));
    vu8_result = Q6_V_vmux_QVV(qu8_sign, Q6_V_vzero(), Q6_Vb_vsplat_R(max_val));
}

// AURA_THRESH_BINARY_INV, U16
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_BINARY_INV == THRESH_TYPE &&
          std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vu16_src, Tp thresh, Tp max_val, HVX_Vector &vu16_result)
{
    HVX_VectorPred qu16_sign = Q6_Q_vcmp_gt_VuhVuh(vu16_src, Q6_Vh_vsplat_R(thresh));
    vu16_result = Q6_V_vmux_QVV(qu16_sign, Q6_V_vzero(), Q6_Vh_vsplat_R(max_val));
}

// AURA_THRESH_BINARY_INV, S16
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_BINARY_INV == THRESH_TYPE &&
          std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vs16_src, Tp thresh, Tp max_val, HVX_Vector &vs16_result)
{
    HVX_VectorPred qs16_sign = Q6_Q_vcmp_gt_VhVh(vs16_src, Q6_Vh_vsplat_R(thresh));
    vs16_result = Q6_V_vmux_QVV(qs16_sign, Q6_V_vzero(), Q6_Vh_vsplat_R(max_val));
}

// AURA_THRESH_TRUNC, U8
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_TRUNC == THRESH_TYPE &&
          std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vu8_src, Tp thresh, Tp max_val, HVX_Vector &vu8_result)
{
    AURA_UNUSED(max_val);
    HVX_VectorPred qu8_sign = Q6_Q_vcmp_gt_VubVub(vu8_src, Q6_Vb_vsplat_R(thresh));
    vu8_result = Q6_V_vmux_QVV(qu8_sign, Q6_Vb_vsplat_R(thresh), vu8_src);
}

// AURA_THRESH_TRUNC, U16
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_TRUNC == THRESH_TYPE &&
          std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vu16_src, Tp thresh, Tp max_val, HVX_Vector &vu16_result)
{
    AURA_UNUSED(max_val);
    HVX_VectorPred qu16_sign = Q6_Q_vcmp_gt_VuhVuh(vu16_src, Q6_Vh_vsplat_R(thresh));
    vu16_result = Q6_V_vmux_QVV(qu16_sign, Q6_Vh_vsplat_R(thresh), vu16_src);
}

// AURA_THRESH_TRUNC, S16
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_TRUNC == THRESH_TYPE &&
          std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vs16_src, Tp thresh, Tp max_val, HVX_Vector &vs16_result)
{
    AURA_UNUSED(max_val);
    HVX_VectorPred qs16_sign = Q6_Q_vcmp_gt_VhVh(vs16_src, Q6_Vh_vsplat_R(thresh));
    vs16_result = Q6_V_vmux_QVV(qs16_sign, Q6_Vh_vsplat_R(thresh), vs16_src);
}

// AURA_THRESH_TOZERO, U8
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_TOZERO == THRESH_TYPE &&
          std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vu8_src, Tp thresh, Tp max_val, HVX_Vector &vu8_result)
{
    AURA_UNUSED(max_val);
    HVX_VectorPred qu8_sign = Q6_Q_vcmp_gt_VubVub(vu8_src, Q6_Vb_vsplat_R(thresh));
    vu8_result = Q6_V_vmux_QVV(qu8_sign, vu8_src, Q6_V_vzero());
}

// AURA_THRESH_TOZERO, U16
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_TOZERO == THRESH_TYPE &&
          std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vu16_src, Tp thresh, Tp max_val, HVX_Vector &vu16_result)
{
    AURA_UNUSED(max_val);
    HVX_VectorPred qu16_sign = Q6_Q_vcmp_gt_VuhVuh(vu16_src, Q6_Vh_vsplat_R(thresh));
    vu16_result = Q6_V_vmux_QVV(qu16_sign, vu16_src, Q6_V_vzero());
}

// AURA_THRESH_TOZERO, S16
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_TOZERO == THRESH_TYPE &&
          std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vs16_src, Tp thresh, Tp max_val, HVX_Vector &vs16_result)
{
    AURA_UNUSED(max_val);
    HVX_VectorPred qs16_sign = Q6_Q_vcmp_gt_VhVh(vs16_src, Q6_Vh_vsplat_R(thresh));
    vs16_result = Q6_V_vmux_QVV(qs16_sign, vs16_src, Q6_V_vzero());
}

// AURA_THRESH_TOZERO_INV, U8
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_TOZERO_INV == THRESH_TYPE &&
          std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vu8_src, Tp thresh, Tp max_val, HVX_Vector &vu8_result)
{
    AURA_UNUSED(max_val);
    HVX_VectorPred qu8_sign = Q6_Q_vcmp_gt_VubVub(vu8_src, Q6_Vb_vsplat_R(thresh));
    vu8_result = Q6_V_vmux_QVV(qu8_sign, Q6_V_vzero(), vu8_src);
}

// AURA_THRESH_TOZERO_INV, U16
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_TOZERO_INV == THRESH_TYPE &&
          std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vu16_src, Tp thresh, Tp max_val, HVX_Vector &vu16_result)
{
    AURA_UNUSED(max_val);
    HVX_VectorPred qu16_sign = Q6_Q_vcmp_gt_VuhVuh(vu16_src, Q6_Vh_vsplat_R(thresh));
    vu16_result = Q6_V_vmux_QVV(qu16_sign, Q6_V_vzero(), vu16_src);
}

// AURA_THRESH_TOZERO_INV, S16
template <typename Tp, MI_S32 THRESH_TYPE, typename std::enable_if<AURA_THRESH_TOZERO_INV == THRESH_TYPE &&
          std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
static AURA_VOID ThresholdRowCore(HVX_Vector &vs16_src, Tp thresh, Tp max_val, HVX_Vector &vs16_result)
{
    AURA_UNUSED(max_val);
    HVX_VectorPred qs16_sign = Q6_Q_vcmp_gt_VhVh(vs16_src, Q6_Vh_vsplat_R(thresh));
    vs16_result = Q6_V_vmux_QVV(qs16_sign, Q6_V_vzero(), vs16_src);
}

template <typename Tp, MI_S32 C, MI_S32 THRESH_TYPE>
static AURA_VOID ThresholdRow(const Tp *src, Tp *dst, Tp thresh, Tp max_val, MI_S32 width)
{
    using MVType = typename MVHvxVector<C>::Type;

    constexpr MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);

    MVType mv_src, mv_result;

    // main
    {
        for (MI_S32 x = 0; x <= (width - ELEM_COUNTS); x += ELEM_COUNTS)
        {
            vload(src + C * x, mv_src);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                ThresholdRowCore<Tp, THRESH_TYPE>(mv_src.val[ch], thresh, max_val, mv_result.val[ch]);
            }

            vstore(dst + C * x, mv_result);
        }
    }

    // remain
    if (width % ELEM_COUNTS)
    {
        const Tp *src_data = src + C * (width - ELEM_COUNTS);
        Tp       *dst_data = dst + C * (width - ELEM_COUNTS);
        vload(src_data, mv_src);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ThresholdRowCore<Tp, THRESH_TYPE>(mv_src.val[ch], thresh, max_val, mv_result.val[ch]);
        }

        vstore(dst_data, mv_result);
    }
}

template <typename Tp, MI_S32 C, MI_S32 THRESH_TYPE>
static Status ThresholdHvxImpl(const Mat &src, Mat &dst, Tp thresh, Tp max_val, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width  = src.GetSizes().m_width;
    MI_S32 height = src.GetSizes().m_height;
    MI_S32 stride = src.GetStrides().m_width;

    MI_U64 L2fetch_param = L2PfParam(stride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y + 1 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 1)), L2fetch_param);
        }

        const Tp *src_row  = src.Ptr<Tp>(y);
        Tp *dst_row  = dst.Ptr<Tp>(y);
        ThresholdRow<Tp, C, THRESH_TYPE>(src_row, dst_row, thresh, max_val, width);
    }

    return Status::OK;
}

template<typename Tp, MI_S32 THRESH_TYPE>
static Status ThresholdHvxHelper(Context *ctx, const Mat &src, Mat &dst, Tp thresh, Tp max_val)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((MI_S32)0, height, ThresholdHvxImpl<Tp, 1, THRESH_TYPE>, std::cref(src), std::ref(dst), thresh, max_val);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((MI_S32)0, height, ThresholdHvxImpl<Tp, 2, THRESH_TYPE >, std::cref(src), std::ref(dst), thresh, max_val);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((MI_S32)0, height, ThresholdHvxImpl<Tp, 3, THRESH_TYPE>, std::cref(src), std::ref(dst), thresh, max_val);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status ThresholdHvxHelper(Context *ctx, const Mat &src, Mat &dst, Tp thresh, Tp max_val, MI_S32 type)
{
    Status ret = Status::ERROR;

    switch(type & AURA_THRESH_MASK_LOW)
    {
        case AURA_THRESH_BINARY:
        {
            ret = ThresholdHvxHelper<Tp, AURA_THRESH_BINARY>(ctx, src, dst, thresh, max_val);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ThresholdHvxHelper<Tp, AURA_THRESH_BINARY> failed");
            }
            break;
        }
        
        case AURA_THRESH_BINARY_INV:
        {
            ret = ThresholdHvxHelper<Tp, AURA_THRESH_BINARY_INV>(ctx, src, dst, thresh, max_val);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ThresholdHvxHelper<Tp, AURA_THRESH_BINARY_INV> failed");
            }
            break;
        }

        case AURA_THRESH_TRUNC:
        {
            ret = ThresholdHvxHelper<Tp, AURA_THRESH_TRUNC>(ctx, src, dst, thresh, max_val);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ThresholdHvxHelper<Tp, AURA_THRESH_TRUNC> failed");
            }
            break;
        }

        case AURA_THRESH_TOZERO:
        {
            ret = ThresholdHvxHelper<Tp, AURA_THRESH_TOZERO>(ctx, src, dst, thresh, max_val);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ThresholdHvxHelper<Tp, AURA_THRESH_TOZERO> failed");
            }
            break;
        }

        case AURA_THRESH_TOZERO_INV:
        {
            ret = ThresholdHvxHelper<Tp, AURA_THRESH_TOZERO_INV>(ctx, src, dst, thresh, max_val);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ThresholdHvxHelper<Tp, AURA_THRESH_TOZERO_INV> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "threshold method not supported");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

ThresholdHvx::ThresholdHvx(Context *ctx, const OpTarget &target) : ThresholdImpl(ctx, target)
{}

Status ThresholdHvx::SetArgs(const Array *src, Array *dst, MI_F32 thresh, MI_F32 max_val, MI_S32 type)
{
    if (ThresholdImpl::SetArgs(src, dst, thresh, max_val, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ThresholdImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8 && elem_type != ElemType::U16 && elem_type != ElemType::S16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8/u16/s16");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ThresholdHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            MI_U8 max_val = SaturateCast<MI_U8>(m_max_val);
            MI_U8 thresh  = SaturateCast<MI_U8>(Floor(m_thresh));

            ret = ThresholdHvxHelper<MI_U8>(m_ctx, *src, *dst, thresh, max_val, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdHvxHelper<MI_U8> failed");
            }
            break;
        }
        
        case ElemType::U16:
        {
            MI_U16 max_val = SaturateCast<MI_U16>(m_max_val);
            MI_U16 thresh  = SaturateCast<MI_U16>(Floor(m_thresh));

            ret = ThresholdHvxHelper<MI_U16>(m_ctx, *src, *dst, thresh, max_val, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdHvxHelper<MI_U16> failed");
            }
            break;
        }

        case ElemType::S16:
        {
            MI_S16 max_val = SaturateCast<MI_S16>(m_max_val);
            MI_S16 thresh  = SaturateCast<MI_S16>(Floor(m_thresh));

            ret = ThresholdHvxHelper<MI_S16>(m_ctx, *src, *dst, thresh, max_val, m_type);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdHvxHelper<MI_S16> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string ThresholdHvx::ToString() const
{
    return ThresholdImpl::ToString() + m_profiling_string;
}

Status ThresholdRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    MI_F32 thresh;
    MI_F32 max_val;
    MI_S32 type;

    ThresholdInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, thresh, max_val, type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    Threshold threshold(ctx, OpTarget::Hvx());

    return OpCall(ctx, threshold, &src, &dst, thresh, max_val, type);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_MISC_PACKAGE_NAME, AURA_OPS_MISC_THRESHOLD_OP_NAME, ThresholdRpc);

}