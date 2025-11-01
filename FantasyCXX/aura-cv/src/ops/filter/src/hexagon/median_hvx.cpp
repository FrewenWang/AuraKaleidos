#include "aura/ops/filter/median.hpp"
#include "median_impl.hpp"
#include "filter_comm.hpp"

namespace aura
{

MedianHvx::MedianHvx(Context *ctx, const OpTarget &target) : MedianImpl(ctx, target)
{}

Status MedianHvx::SetArgs(const Array *src, Array *dst, MI_S32 ksize)
{
    if (MedianImpl::SetArgs(src, dst, ksize) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MedianImpl::Initialize failed");
        return Status::ERROR;
    }

    if (ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 3/5/7");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::S8 && elem_type != ElemType::U8 && elem_type != ElemType::S16 && elem_type != ElemType::U16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support s8/u8/u16/s16");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MedianHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 3:
        {
            ret = Median3x3Hvx(m_ctx, *src, *dst);
            break;
        }

        case 5:
        {
            ret = Median5x5Hvx(m_ctx, *src, *dst);
            break;
        }

        case 7:
        {
            ret = Median7x7Hvx(m_ctx, *src, *dst);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported ksize");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string MedianHvx::ToString() const
{
    return MedianImpl::ToString();
}

Status MedianRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    MI_S32 ksize;

    MedianInParam in_param(ctx, rpc_param);

    Status ret = in_param.Get(src, dst, ksize);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    Median median(ctx, OpTarget::Hvx());

    return OpCall(ctx, median, &src, &dst, ksize);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_MEDIAN_OP_NAME, MedianRpc);

} // namespace aura