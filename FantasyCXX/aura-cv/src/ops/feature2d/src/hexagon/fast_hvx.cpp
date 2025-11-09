#include "fast_impl.hpp"
#include "feature2d_comm.hpp"
#include "aura/ops/feature2d/fast.hpp"

namespace aura
{

FastHvx::FastHvx(Context *ctx, const OpTarget &target) : FastImpl(ctx, target)
{}

Status FastHvx::SetArgs(const Array *src, std::vector<KeyPoint> &key_points, DT_S32 threshold,
                        DT_BOOL nonmax_suppression, FastDetectorType type, DT_U32 max_num_corners)
{
    if (FastImpl::SetArgs(src, key_points, threshold, nonmax_suppression, type, max_num_corners) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "FastImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    DT_S32 height = src->GetSizes().m_height;
    if (height < 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src height only support > 7");
        return Status::ERROR;
    }

    if (type != FastDetectorType::FAST_9_16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "fast feature hvx only support fast9");
        return Status::ERROR;
    }

    if (((DT_UPTR_T)(dynamic_cast<const Mat*>(src)->GetData()) & 127) && (src->GetStrides().m_width & 127))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src mat strides need align to 128");
        return Status::ERROR;
    }

    return Status::OK;
}

Status FastHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_detector_type)
    {
        case FastDetectorType::FAST_9_16:
        {
            ret = Fast9Hvx(m_ctx, *src, m_key_points, m_threshold, m_nonmax_suppression, m_max_num_corners);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Fast9Hvx failed");
                return Status::ERROR;
            }

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

std::string FastHvx::ToString() const
{
    return FastImpl::ToString();
}

Status FastRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    std::vector<KeyPoint> dst_points;
    DT_S32 threshold;
    DT_U32 max_num_corners;
    DT_BOOL nonmax_suppression;
    FastDetectorType type;

    FastInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, threshold, nonmax_suppression, type, max_num_corners);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    dst_points.reserve(max_num_corners);
    Fast fast(ctx, OpTarget::Hvx());

    ret = OpCall(ctx, fast, &src, dst_points, threshold, nonmax_suppression, type, max_num_corners);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return Status::ERROR;
    }

    FastOutParam out_param(ctx, rpc_param);
    ret |= out_param.Set(dst_points);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return Status::ERROR;
    }

    return Status::OK;
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_FEATURE2D_PACKAGE_NAME, AURA_OPS_FEATURE2D_FAST_OP_NAME, FastRpc);

} // namespace aura