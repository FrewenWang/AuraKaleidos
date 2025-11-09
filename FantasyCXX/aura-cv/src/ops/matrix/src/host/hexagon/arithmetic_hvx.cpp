#include "matrix_comm.hpp"
#include "arithmetic_impl.hpp"
#include "aura/ops/matrix/arithmetic.hpp"

namespace aura
{

ArithmeticHvx::ArithmeticHvx(Context *ctx, const OpTarget &target) : ArithmeticImpl(ctx, target)
{}

Status ArithmeticHvx::SetArgs(const Array *src0, const Array *src1, Array *dst, ArithmOpType op)
{
    if (ArithmeticImpl::SetArgs(src0, src1, dst, op) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ArithmeticImpl::Initialize failed");
        return Status::ERROR;
    }

    if ((src0->GetMemType() != AURA_MEM_DMA_BUF_HEAP) || (src1->GetMemType() != AURA_MEM_DMA_BUF_HEAP) || (dst->GetMemType() != AURA_MEM_DMA_BUF_HEAP))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat memory must be AURA_MEM_DMA_BUF_HEAP type");
        return Status::ERROR;
    }

    DT_S32 pattern = AURA_MAKE_PATTERN(op, src0->GetElemType(), dst->GetElemType());

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
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src0) || (DT_NULL == src1) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    HexagonRpcParam rpc_param(m_ctx);
    ArithmeticInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src0, *src1, *dst, m_op_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_MATRIX_PACKAGE_NAME, AURA_OPS_MATRIX_ARITHMETIC_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && DT_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string ArithmeticHvx::ToString() const
{
    return ArithmeticImpl::ToString() + m_profiling_string;
}

} // namespace aura