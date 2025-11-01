#include "aura/ops/matrix/sum.hpp"
#include "sum_impl.hpp"
#include "matrix_comm.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp> struct MatrixSumTraits;// Get accumulation dst Type
template <> struct MatrixSumTraits<MI_U8>  { using PartSumType = MI_U32; using SumType = MI_U64; };
template <> struct MatrixSumTraits<MI_S8>  { using PartSumType = MI_S32; using SumType = MI_S64; };
template <> struct MatrixSumTraits<MI_U16> { using PartSumType = MI_U32; using SumType = MI_U64; };
template <> struct MatrixSumTraits<MI_S16> { using PartSumType = MI_S32; using SumType = MI_S64; };

// using St = MI_U8
template <typename St, typename std::enable_if<std::is_same<St, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CalSumRow2Core(HVX_Vector &vu8_src_c, HVX_Vector &vu8_src_n, HVX_VectorPair &wu32_dst)
{
    HVX_VectorPair wu16_sum = Q6_Wh_vadd_VubVub(vu8_src_c, vu8_src_n);
    wu32_dst = Q6_Ww_vaddacc_WwVuhVuh(wu32_dst, Q6_V_lo_W(wu16_sum), Q6_V_hi_W(wu16_sum));
}

// using St = MI_U8
template <typename St, typename std::enable_if<std::is_same<St, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CalSumRow1Core(HVX_Vector &vu8_src_c, HVX_VectorPair &wu32_dst)
{
    HVX_VectorPair wu16_sum = Q6_Wuh_vzxt_Vub(vu8_src_c);
    wu32_dst = Q6_Ww_vaddacc_WwVuhVuh(wu32_dst, Q6_V_lo_W(wu16_sum), Q6_V_hi_W(wu16_sum));
}

// using St = MI_S8
template <typename St, typename std::enable_if<std::is_same<St, MI_S8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CalSumRow2Core(HVX_Vector &vs8_src_c, HVX_Vector &vs8_src_n, HVX_VectorPair &ws32_dst)
{
    HVX_VectorPair ws16_src_c = Q6_Wh_vsxt_Vb(vs8_src_c);
    HVX_VectorPair ws16_src_n = Q6_Wh_vsxt_Vb(vs8_src_n);
    HVX_VectorPair ws16_sum = Q6_Wh_vadd_WhWh(ws16_src_c, ws16_src_n);
    ws32_dst = Q6_Ww_vaddacc_WwVhVh(ws32_dst, Q6_V_lo_W(ws16_sum), Q6_V_hi_W(ws16_sum));
}

// using St = MI_S8
template <typename St, typename std::enable_if<std::is_same<St, MI_S8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CalSumRow1Core(HVX_Vector &vs8_src_c, HVX_VectorPair &ws32_dst)
{
    HVX_VectorPair ws16_sum = Q6_Wh_vsxt_Vb(vs8_src_c);
    ws32_dst = Q6_Ww_vaddacc_WwVhVh(ws32_dst, Q6_V_lo_W(ws16_sum), Q6_V_hi_W(ws16_sum));
}

// using St = MI_U16
template <typename St, typename std::enable_if<std::is_same<St, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CalSumRow2Core(HVX_Vector &vu16_src_c, HVX_Vector &vu16_src_n, HVX_VectorPair &wu32_dst)
{
    wu32_dst = Q6_Ww_vaddacc_WwVuhVuh(wu32_dst, vu16_src_c, vu16_src_n);
}

// using St = MI_U16
template <typename St, typename std::enable_if<std::is_same<St, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CalSumRow1Core(HVX_Vector &vu16_src_c, HVX_VectorPair &wu32_dst)
{
    HVX_VectorPair wu32_sum = Q6_Wuw_vzxt_Vuh(vu16_src_c);
    wu32_dst = Q6_Wuw_vadd_WuwWuw_sat(wu32_dst, wu32_sum);
}

// using St = MI_S16
template <typename St, typename std::enable_if<std::is_same<St, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CalSumRow2Core(HVX_Vector &vs16_src_c, HVX_Vector &vs16_src_n, HVX_VectorPair &ws32_dst)
{
    ws32_dst = Q6_Ww_vaddacc_WwVhVh(ws32_dst, vs16_src_c, vs16_src_n);
}

// using St = MI_S16
template <typename St, typename std::enable_if<std::is_same<St, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID CalSumRow1Core(HVX_Vector &vs16_src_c, HVX_VectorPair &ws32_dst)
{
    HVX_VectorPair ws32_sum = Q6_Ww_vsxt_Vh(vs16_src_c);
    ws32_dst = Q6_Ww_vadd_WwWw_sat(ws32_dst, ws32_sum);
}

template <typename St, typename MVType, typename MWType, MI_S32 C>
AURA_ALWAYS_INLINE AURA_VOID CalSumRow2(const St *src_c, const St *src_n, MWType &mwd32_sum, HVX_VectorPred &q, MI_S32 back_offset,
                                      MI_S32 elem_counts, MI_S32 width, MI_S32 rest)
{
    MVType mv_src_c, mv_src_n;

    for (MI_S32 x = 0; x <= back_offset; x += elem_counts)
    {
        vload(src_c + x * C, mv_src_c);
        vload(src_n + x * C, mv_src_n);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            CalSumRow2Core<St>(mv_src_c.val[ch], mv_src_n.val[ch], mwd32_sum.val[ch]);
        }
    }

    if (rest)
    {
        vload(src_c + (width - elem_counts) * C, mv_src_c);
        vload(src_n + (width - elem_counts) * C, mv_src_n);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_c.val[ch] = Q6_V_vmux_QVV(q, Q6_V_vzero(), mv_src_c.val[ch]);
            mv_src_n.val[ch] = Q6_V_vmux_QVV(q, Q6_V_vzero(), mv_src_n.val[ch]);
            CalSumRow2Core<St>(mv_src_c.val[ch], mv_src_n.val[ch], mwd32_sum.val[ch]);
        }
    }
}

template <typename St, typename MVType, typename MWType, MI_S32 C>
AURA_ALWAYS_INLINE AURA_VOID CalSumRow1(const St *src_c, MWType &mwd32_sum, HVX_VectorPred &q, MI_S32 back_offset,
                                      MI_S32 elem_counts, MI_S32 width, MI_S32 rest)
{
    MVType mv_src_c;

    for (MI_S32 x = 0; x <= back_offset; x += elem_counts)
    {
        vload(src_c + x * C, mv_src_c);
        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            CalSumRow1Core<St>(mv_src_c.val[ch], mwd32_sum.val[ch]);
        }
    }

    if (rest)
    {
        vload(src_c + (width - elem_counts) * C, mv_src_c);
        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_c.val[ch] = Q6_V_vmux_QVV(q, Q6_V_vzero(), mv_src_c.val[ch]);
            CalSumRow1Core<St>(mv_src_c.val[ch], mwd32_sum.val[ch]);
        }
    }
}

template <typename PartSumType, typename SumType, typename MWType, MI_S32 C>
AURA_NO_INLINE AURA_VOID ProcessPartSum(SumType *final_sum, MI_U32 *vec_buf, MWType &mwd32_sum, MI_S32 len)
{
    for (MI_S32 ch = 0; ch < C; ch++)
    {
        HVX_Vector vd32_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(mwd32_sum.val[ch]), Q6_V_hi_W(mwd32_sum.val[ch]));
        *(HVX_Vector*)vec_buf = vd32_sum;

        for (MI_S32 i = 0; i < len; i++)
        {
            final_sum[ch] += static_cast<PartSumType>(vec_buf[i]);
        }
    }
}

template <typename St, typename SumType, MI_S32 C>
static Status SumHvxImpl(Context *ctx, const Mat &src, std::vector<SumType> &dst_buffer, MI_S32 start_row, MI_S32 end_row)
{
    using MVType      = typename MVHvxVector<C>::Type;
    using MWType      = typename MWHvxVector<C>::Type;
    using PartSumType = typename MatrixSumTraits<St>::PartSumType;

    constexpr MI_S32 VEC_LEN = AURA_HVLEN / sizeof(PartSumType);

    MI_S32 width  = src.GetSizes().m_width;
    MI_S32 height = end_row - start_row;

    MI_S32 num_vec_per_row = (width + VEC_LEN - 1) / VEC_LEN;
    MI_S32 height_block    = ((1 << ((sizeof(PartSumType) - sizeof(St)) * 8)) / num_vec_per_row) & (-2); // height_block = 2 minimum
    if (height_block < 1)
    {
        AURA_ADD_ERROR_STRING(ctx, "height_block too small");
        return Status::ERROR;
    }

    MI_S32 stride = src.GetStrides().m_width;
    MI_U32 vec_buf[32] __attribute__((aligned(AURA_HVLEN)));
    SumType final_sum[C] = {0};

    const St *src_c = NULL;
    const St *src_n = NULL;
    MI_U64 L2fetch_param = L2PfParam(stride, width * C * ElemTypeSize(src.GetElemType()), 2, 0);
    L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<St>(start_row)), L2fetch_param);

    MI_S32 elem_counts = AURA_HVLEN / sizeof(St);
    MI_S32 back_offset = width - elem_counts;
    MI_S32 rest        = (width % elem_counts) * sizeof(St);
    HVX_VectorPred q   = Q6_Q_vsetq_R(AURA_HVLEN - rest);
    MWType mwd32_sum;

    MI_S32 y = 0;
    for (; y <= (height - height_block); y += height_block)
    {
        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mwd32_sum.val[ch] = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        }

        for (MI_S32 z = y; z < y + height_block; z += 2)
        {
            MI_S32 r = Min(z + 2, height - 2);
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<St>(start_row + r)), L2fetch_param);
            src_c = src.Ptr<St>(start_row + z);
            src_n = src.Ptr<St>(start_row + z + 1);

            CalSumRow2<St, MVType, MWType, C>(src_c, src_n, mwd32_sum, q, back_offset, elem_counts, width, rest);
        }

        ProcessPartSum<PartSumType, SumType, MWType, C>(final_sum, vec_buf, mwd32_sum, VEC_LEN);
    }

    if (y < height)
    {
        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mwd32_sum.val[ch] = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        }

        MI_S32 height_align = height & (-2);

        for (; y < height_align; y += 2)
        {
            MI_S32 r = Min(y + 2, height - 2);
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<St>(start_row + r)), L2fetch_param);
            src_c = src.Ptr<St>(start_row + y);
            src_n = src.Ptr<St>(start_row + y + 1);

            CalSumRow2<St, MVType, MWType, C>(src_c, src_n, mwd32_sum, q, back_offset, elem_counts, width, rest);
        }

        for (; y < height; y++)
        {
            src_c = src.Ptr<St>(start_row + y);
            CalSumRow1<St, MVType, MWType, C>(src_c, mwd32_sum, q, back_offset, elem_counts, width, rest);
        }

        ProcessPartSum<PartSumType, SumType, MWType, C>(final_sum, vec_buf, mwd32_sum, VEC_LEN);
    }

    MI_S32 thread_id = ctx->GetWorkerPool()->GetComputeThreadIdx();
    #pragma unroll(C)
    for (MI_S32 ch = 0; ch < C; ch++)
    {
        dst_buffer[thread_id * C + ch] += static_cast<SumType>(final_sum[ch]);
    }

    return Status::OK;
}

template<typename St>
static Status SumHvxHelper(Context *ctx, const Mat &src, Scalar &result)
{
    using SumType = typename MatrixSumTraits<St>::SumType;

    Status ret = Status::ERROR;
    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 channel = src.GetSizes().m_channel;

    result = Scalar(); // clear result
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return Status::ERROR;
    }

    MI_S32 task_nums = wp->GetComputeThreadNum();
    std::vector<SumType> row_buffer(channel * task_nums, 0);

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((MI_S32)0, height, SumHvxImpl<St, SumType, 1>, ctx, std::cref(src), std::ref(row_buffer));
            for (MI_S32 i = 0; i < task_nums; i++)
            {
                result.m_val[0] += static_cast<MI_F64>(row_buffer[i]);
            }
            break;
        }
        case 2:
        {
            ret = wp->ParallelFor((MI_S32)0, height, SumHvxImpl<St, SumType, 2>, ctx, std::cref(src), std::ref(row_buffer));
            for (MI_S32 i = 0; i < task_nums; i++)
            {
                result.m_val[0] += static_cast<MI_F64>(row_buffer[2 * i]);
                result.m_val[1] += static_cast<MI_F64>(row_buffer[2 * i + 1]);
            }
            break;
        }
        case 3:
        {
            ret = wp->ParallelFor((MI_S32)0, height, SumHvxImpl<St, SumType, 3>, ctx, std::cref(src), std::ref(row_buffer));
            for (MI_S32 i = 0; i < task_nums; i++)
            {
                result.m_val[0] += static_cast<MI_F64>(row_buffer[3 * i]);
                result.m_val[1] += static_cast<MI_F64>(row_buffer[3 * i + 1]);
                result.m_val[2] += static_cast<MI_F64>(row_buffer[3 * i + 2]);
            }
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

SumHvx::SumHvx(Context *ctx, const OpTarget &target) : SumImpl(ctx, target)
{}

Status SumHvx::SetArgs(const Array *src, Scalar *result)
{
    if (SumImpl::SetArgs(src, result) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumImpl::Initialize failed");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    MI_S32 width = src->GetSizes().m_width;
    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8 && elem_type != ElemType::S8 && elem_type != ElemType::U16 && elem_type != ElemType::S16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8/s8/u16/s16");
        return Status::ERROR;
    }

    MI_S32 shift = (elem_type == ElemType::U8 || elem_type == ElemType::S8) ? 24 : 16;
    if (width > ((1 << shift) * 16)) //at least accumulate 2 rows
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat width too large");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SumHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = SumHvxHelper<MI_U8>(m_ctx, *src, *m_result);
            break;
        }

        case ElemType::S8:
        {
            ret = SumHvxHelper<MI_S8>(m_ctx, *src, *m_result);
            break;
        }

        case ElemType::U16:
        {
            ret = SumHvxHelper<MI_U16>(m_ctx, *src, *m_result);
            break;
        }

        case ElemType::S16:
        {
            ret = SumHvxHelper<MI_S16>(m_ctx, *src, *m_result);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported source format");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string SumHvx::ToString() const
{
    return SumImpl::ToString();
}

Status SumRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Scalar result;

    SumInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    Sum sum(ctx, OpTarget::Hvx());

    ret = OpCall(ctx, sum, &src, result);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "OpCall failed");
        return Status::ERROR;
    }

    SumOutParam out_param(ctx, rpc_param);
    ret = out_param.Set(result);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    return Status::OK;
}

MeanHvx::MeanHvx(Context *ctx, const OpTarget &target) : SumHvx(ctx, target)
{}

Status MeanHvx::Run()
{
    Status ret = SumHvx::Run();
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumHvx run failed.");
        return Status::ERROR;
    }

    const MI_S32 height = m_src->GetSizes().m_height;
    const MI_S32 width  = m_src->GetSizes().m_width;
    *m_result           = (*m_result) / static_cast<MI_F64>(height * width);

    AURA_RETURN(m_ctx, ret);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_MATRIX_PACKAGE_NAME, AURA_OPS_MATRIX_SUM_OP_NAME, SumRpc);

} // namespace aura