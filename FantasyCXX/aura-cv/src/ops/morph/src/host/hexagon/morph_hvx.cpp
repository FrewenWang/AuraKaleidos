#include "morph_impl.hpp"
#include "morph_comm.hpp"

namespace aura
{

MorphHvx::MorphHvx(Context *ctx, MorphType type, const OpTarget &target) : MorphImpl(ctx, type, target)
{}

Status MorphHvx::SetArgs(const Array *src, Array *dst, MI_S32 ksize, MorphShape shape, MI_S32 iterations)
{
    if (MorphImpl::SetArgs(src, dst, ksize, shape, iterations) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if ((src->GetMemType() != AURA_MEM_DMA_BUF_HEAP) || (dst->GetMemType() != AURA_MEM_DMA_BUF_HEAP))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat memory must be AURA_MEM_DMA_BUF_HEAP type");
        return Status::ERROR;
    }

    if (ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 3/5/7");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
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

Status MorphHvx::Initialize()
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (MorphImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::Initialize() failed");
        return Status::ERROR;
    }

    if (m_iterations > 1)
    {
        m_iter_buffer = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(m_ctx, AURA_MEM_DMA_BUF_HEAP,
                                                       m_dst->GetStrides().m_width * m_dst->GetStrides().m_height, 0));
        if (!m_iter_buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetBuffer failed");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status MorphHvx::DeInitialize()
{
    Status ret = Status::OK;

    if (m_iterations > 1)
    {
        if (m_iter_buffer.IsValid())
        {
            if (AURA_FREE(m_ctx, m_iter_buffer.m_origin) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "free failed");
                ret = Status::ERROR;
            }
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "buff is invalid");
            ret = Status::ERROR;
        }
    }

    if (MorphImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    AURA_RETURN(m_ctx, ret);
}

Status MorphHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    HexagonRpcParam rpc_param(m_ctx);
    MorphInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, *dst, m_iter_buffer, m_type, m_ksize, m_shape, m_iterations);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_MORPH_PACKAGE_NAME, AURA_OPS_MORPH_MORPH_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && MI_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string MorphHvx::ToString() const
{
    return MorphImpl::ToString() + m_profiling_string;
}

} // namespace aura