#include "gaussian_impl.hpp"
#include "filter_comm.hpp"

namespace aura
{

GaussianHvx::GaussianHvx(Context *ctx, const OpTarget &target) : GaussianImpl(ctx, target)
{}

Status GaussianHvx::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                            BorderType border_type, const Scalar &border_value)
{
    if (GaussianImpl::SetArgs(src, dst, ksize, sigma, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }
    /// 在高通的HVM里面数据。输入数据和输出数据必须使用DMA buffer
    if ((src->GetMemType() != AURA_MEM_DMA_BUF_HEAP) || (dst->GetMemType() != AURA_MEM_DMA_BUF_HEAP))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat memory must be AURA_MEM_DMA_BUF_HEAP type");
        return Status::ERROR;
    }

    /// HVX 的高斯核只支持3 5 7 9
    if (ksize != 3 && ksize != 5 && ksize != 7 && ksize != 9)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 3/5/7/9");
        return Status::ERROR;
    }

    // 数据的通道只支持1 2 3
    DT_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }
    // TODO 数据类型为什么还支持signed int 32
    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8 && elem_type != ElemType::U16 && elem_type != ElemType::S16 &&
        elem_type != ElemType::U32 && elem_type != ElemType::S32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8/u16/s16/u32/s32");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    /// 初始化HexagonRpcParam RPC通信的param
    HexagonRpcParam rpc_param(m_ctx);
    /// 获取高斯滤波的HVX的输入参数。 需要将我们的rpc_param传入
    GaussianInParamHvx in_param(m_ctx, rpc_param);
    /// 同时将对应的入参传入
    ret = in_param.Set(*src, *dst, m_ksize, m_sigma, m_border_type, m_border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    /// 获取对应的HexagonEngine
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    /// 同时传入对应的包名、和对应的OP名称，方便我们对应对应的后端对应函数
    ret = engine->Run(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_GAUSSIAN_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && DT_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string GaussianHvx::ToString() const
{
    return GaussianImpl::ToString() + m_profiling_string;
}

} // namespace aura