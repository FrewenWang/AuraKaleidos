#include "morph_impl.hpp"
#include "morph_comm.hpp"

namespace aura
{

MorphHvx::MorphHvx(Context *ctx, MorphType type, const OpTarget &target) : MorphImpl(ctx, type, target), m_buffer_name("morph_buffer")
{}

Status MorphHvx::SetArgs(const Array *src, Array *dst, DT_S32 ksize, MorphShape shape, DT_S32 iterations)
{
    if (MorphImpl::SetArgs(src, dst, ksize, shape, iterations) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 3/5/7");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;
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
    if (MorphImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::Initialize() failed");
        return Status::ERROR;
    }

    if (m_iterations > 1)
    {
        m_iter_buffer = m_ctx->GetShareBuffer(m_buffer_name);
        if (!m_iter_buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_iter_buffer is invalid");
            return Status::ERROR;
        }

        m_iter_mat = Mat(m_ctx, m_dst->GetElemType(), m_dst->GetSizes(), m_iter_buffer, m_dst->GetStrides());
        if (!m_iter_mat.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "iter_mat is invalid");
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
        m_iter_mat = Mat();
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

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    const Mat *iter_src = src;
    Mat *iter_dst = ((m_iterations & 1) == 1) ? dst : &m_iter_mat;

    for (DT_S32 i = 0; i < m_iterations; i++)
    {
        switch (m_ksize)
        {
            case 3:
            {
                ret = Morph3x3Hvx(m_ctx, *iter_src, *iter_dst, m_type, m_shape);
                break;
            }

            case 5:
            {
                ret = Morph5x5Hvx(m_ctx, *iter_src, *iter_dst, m_type, m_shape);
                break;
            }

            case 7:
            {
                ret = Morph7x7Hvx(m_ctx, *iter_src, *iter_dst, m_type, m_shape);
                break;
            }

            default:
            {
                AURA_ADD_ERROR_STRING(m_ctx, "unsupported kernel size");
                return Status::ERROR;
            }
        }

        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Morph" + std::to_string(m_ksize) + "x" + std::to_string(m_ksize) + "Hvx failed").c_str());
            AURA_RETURN(m_ctx, ret);
        }

        iter_src = (((i + m_iterations) & 1) == 1) ? dst : &m_iter_mat;
        iter_dst = (((i + m_iterations) & 1) == 1) ? &m_iter_mat : dst;
    }

    AURA_RETURN(m_ctx, ret);
}

std::string MorphHvx::ToString() const
{
    return MorphImpl::ToString();
}

Status MorphRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    Buffer iter_buffer;
    MorphType type;
    MorphShape shape;
    DT_S32 ksize;
    DT_S32 iterations;

    MorphInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, iter_buffer, type, ksize, shape, iterations);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return ret;
    }

    const std::string morph_buffer_name = "morph_buffer";

    if (iter_buffer.IsValid())
    {
        ret = ctx->AddShareBuffer(morph_buffer_name, iter_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "AddShareBuffer failed");
            return ret;
        }
    }

    if (type == MorphType::DILATE)
    {
        Dilate dilate(ctx, OpTarget::Hvx());
        ret = OpCall(ctx, dilate, &src, &dst, ksize, shape, iterations);
    }
    else if (type == MorphType::ERODE)
    {
        Erode erode(ctx, OpTarget::Hvx());
        ret = OpCall(ctx, erode, &src, &dst, ksize, shape, iterations);
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "OpCall failed");
    }

    if (iter_buffer.IsValid())
    {
        if (ctx->RemoveShareBuffer(morph_buffer_name) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "RemoveShareBuffer Buffer failed");
            ret |= Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_MORPH_PACKAGE_NAME, AURA_OPS_MORPH_MORPH_OP_NAME, MorphRpc);

} // namespace aura