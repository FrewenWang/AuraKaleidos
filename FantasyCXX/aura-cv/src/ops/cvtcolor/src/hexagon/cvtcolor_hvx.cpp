#include "cvtcolor_impl.hpp"
#include "cvtcolor_comm.hpp"

namespace aura
{

static Status CvtColorHvxImpl(Context *ctx, const std::vector<const Mat*> &src, std::vector<Mat*> &dst, CvtColorType type)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        // RGB <-> BGRA
        case CvtColorType::BGR2BGRA:
        case CvtColorType::BGRA2BGR:
        case CvtColorType::BGR2RGB:
        {
            ret = CvtBgr2BgraHvx(ctx, *(src[0]), *(dst[0]), SwapBlue(type));
            break;
        }

        case CvtColorType::BGR2GRAY:
        case CvtColorType::BGRA2GRAY:
        case CvtColorType::RGB2GRAY:
        case CvtColorType::RGBA2GRAY:
        {
            ret = CvtBgr2GrayHvx(ctx, *(src[0]), *(dst[0]), SwapBlue(type));
            break;
        }

        case CvtColorType::GRAY2BGR:
        case CvtColorType::GRAY2BGRA:
        {
            ret = CvtGray2BgrHvx(ctx, *(src[0]), *(dst[0]));
            break;
        }

        // YUV -> RGB
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        {
            ret = CvtNv2RgbHvx(ctx, *(src[0]), *(src[1]), *(dst[0]), SwapUv(type), type);
            break;
        }
        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        {
            ret = CvtY4202RgbHvx(ctx, *(src[0]), *(src[1]), *(src[2]), *(dst[0]), SwapUv(type), type);
            break;
        }
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        {
            DT_BOOL swapy = (CvtColorType::YUV2RGB_Y422 == type) || (CvtColorType::YUV2RGB_Y422_601 == type);
            ret = CvtY4222RgbHvx(ctx, *(src[0]), *(dst[0]), SwapUv(type), swapy, type);
            break;
        }
        case CvtColorType::YUV2RGB_Y444:
        case CvtColorType::YUV2RGB_Y444_601:
        {
            ret = CvtY4442RgbHvx(ctx, *(src[0]), *(src[1]), *(src[2]), *(dst[0]), type);
            break;
        }

        // RGB -> YUV
        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        {
            ret = CvtRgb2NvHvx(ctx, *(src[0]), *(dst[0]), *(dst[1]), SwapUv(type), type);
            break;
        }
        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        {
            ret = CvtRgb2Y420Hvx(ctx, *(src[0]), *(dst[0]), *(dst[1]), *(dst[2]), SwapUv(type), type);
            break;
        }
        case CvtColorType::RGB2YUV_Y444:
        case CvtColorType::RGB2YUV_Y444_601:
        {
            ret = CvtRgb2Y444Hvx(ctx, *(src[0]), *(dst[0]), *(dst[1]), *(dst[2]), type);
            break;
        }
        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {

            ret = CvtRgb2NvP010Hvx(ctx, *(src[0]), *(dst[0]), *(dst[1]), SwapUv(type));
            break;
        }

        // BAYER -> BGR
        case CvtColorType::BAYERBG2BGR:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            ret = CvtBayer2BgrHvx(ctx, *(src[0]), *(dst[0]), SwapBlue(type), SwapGreen(type));
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

CvtColorHvx::CvtColorHvx(Context *ctx, const OpTarget &target) : CvtColorImpl(ctx, target)
{}

Status CvtColorHvx::SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type)
{
    if (CvtColorImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CvtColorImpl::SetArgs failed(Hvx)");
        return Status::ERROR;
    }

    for (DT_U32 i = 0; i < src.size(); i++)
    {
        if (src[i]->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
            return Status::ERROR;
        }

        if (src[i]->GetMemType() != AURA_MEM_DMA_BUF_HEAP)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "mat memory must be AURA_MEM_DMA_BUF_HEAP type");
            return Status::ERROR;
        }
    }

    for (DT_U32 i = 0; i < dst.size(); i++)
    {
        if (dst[i]->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst must be mat type");
            return Status::ERROR;
        }

        if (dst[i]->GetMemType() != AURA_MEM_DMA_BUF_HEAP)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "mat memory must be AURA_MEM_DMA_BUF_HEAP type");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status CvtColorHvx::Run()
{
    Status ret = Status::OK;
    std::vector<const Mat*> src;
    std::vector<Mat*> dst;

    for (DT_U32 i = 0; i < m_src.size(); i++)
    {
        const Mat *mat = dynamic_cast<const Mat*>(m_src[i]);
        src.push_back(mat);
    }

    for (DT_U32 i = 0; i < m_dst.size(); i++)
    {
        Mat *mat = dynamic_cast<Mat*>(m_dst[i]);
        dst.push_back(mat);
    }


    if (src.empty() || dst.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    ret = CvtColorHvxImpl(m_ctx, src, dst, m_type);

    AURA_RETURN(m_ctx, ret);
}

std::string CvtColorHvx::ToString() const
{
    return CvtColorImpl::ToString();
}

Status CvtColorRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    std::vector<Mat> src;
    std::vector<Mat> dst;
    CvtColorType     type = CvtColorType::INVALID;

    CvtColorInParamHvx in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    std::vector<const Array*> vec_src;
    std::vector<Array*> vec_dst;

    for (DT_U32 i = 0; i < src.size(); i++)
    {
        const Array *p_src = &src[i];
        vec_src.push_back(p_src);
    }

    for (DT_U32 i = 0; i < dst.size(); i++)
    {
        Array *p_dst = &dst[i];
        vec_dst.push_back(p_dst);
    }

    CvtColor cvtcolor(ctx, OpTarget::Hvx());

    return OpCall(ctx, cvtcolor, vec_src, vec_dst, type);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_CVTCOLOR_PACKAGE_NAME, AURA_OPS_CVTCOLOR_OP_NAME, CvtColorRpc);

} // namespace aura