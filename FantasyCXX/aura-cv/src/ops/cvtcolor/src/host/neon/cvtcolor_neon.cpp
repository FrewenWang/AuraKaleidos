#include "cvtcolor_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status CvtColorNeonImpl(Context *ctx, const std::vector<const Mat*> &src, std::vector<Mat*> &dst, CvtColorType type, const OpTarget &target)
{
    Status ret = Status::ERROR;
    switch (type)
    {
        // RGB <-> BGRA
        case CvtColorType::BGR2BGRA:
        {
            ret = CvtBgr2BgraNeon(ctx, *(src[0]), *(dst[0]), target);
            break;
        }

        case CvtColorType::BGRA2BGR:
        {
            ret = CvtBgra2BgrNeon(ctx, *(src[0]), *(dst[0]), target);
            break;
        }

        case CvtColorType::BGR2RGB:
        {
            ret = CvtBgr2RgbNeon(ctx, *(src[0]), *(dst[0]), target);
            break;
        }

        case CvtColorType::BGR2GRAY:
        case CvtColorType::BGRA2GRAY:
        case CvtColorType::RGB2GRAY:
        case CvtColorType::RGBA2GRAY:
        {
            ret = CvtBgr2GrayNeon(ctx, *(src[0]), *(dst[0]), SwapBlue(type), target);
            break;
        }

        case CvtColorType::GRAY2BGR:
        case CvtColorType::GRAY2BGRA:
        {
            ret = CvtGray2BgrNeon(ctx, *(src[0]), *(dst[0]), target);
            break;
        }

        // YUV -> RGB
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        {
            /// NV12或者NV21进行转化成为RGB。使用NEON的做法
            ret = CvtNv2RgbNeon(ctx, *(src[0]), *(src[1]), *(dst[0]), SwapUv(type), type, target);
            break;
        }
        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        {
            ret = CvtY4202RgbNeon(ctx, *(src[0]), *(src[1]), *(src[2]), *(dst[0]), SwapUv(type), type, target);
            break;
        }
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        {
            MI_BOOL swapy = (CvtColorType::YUV2RGB_Y422 == type) || (CvtColorType::YUV2RGB_Y422_601 == type);
            ret           = CvtY4222RgbNeon(ctx, *(src[0]), *(dst[0]), SwapUv(type), swapy, type, target);
            break;
        }
        case CvtColorType::YUV2RGB_Y444:
        case CvtColorType::YUV2RGB_Y444_601:
        {
            ret = CvtY4442RgbNeon(ctx, *(src[0]), *(src[1]), *(src[2]), *(dst[0]), type, target);
            break;
        }

        // RGB -> YUV
        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        {
            ret = CvtRgb2NvNeon(ctx, *(src[0]), *(dst[0]), *(dst[1]), SwapUv(type), type, target);
            break;
        }
        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        {
            ret = CvtRgb2Y420Neon(ctx, *(src[0]), *(dst[0]), *(dst[1]), *(dst[2]), SwapUv(type), type, target);
            break;
        }
        case CvtColorType::RGB2YUV_Y444:
        case CvtColorType::RGB2YUV_Y444_601:
        {
            ret = CvtRgb2Y444Neon(ctx, *(src[0]), *(dst[0]), *(dst[1]), *(dst[2]), type, target);
            break;
        }
        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            ret = CvtRgb2NvP010Neon(ctx, *(src[0]), *(dst[0]), *(dst[1]), SwapUv(type), target);
            break;
        }

        // BAYER -> BGR
        case CvtColorType::BAYERBG2BGR:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            ret = CvtBayer2BgrNeon(ctx, *(src[0]), *(dst[0]), SwapBlue(type), SwapGreen(type), target);
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

CvtColorNeon::CvtColorNeon(Context *ctx, const OpTarget &target) : CvtColorImpl(ctx, target)
{}

Status CvtColorNeon::SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type)
{
    if (CvtColorImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CvtColorImpl::SetArgs failed(neon)");
        return Status::ERROR;
    }

    for(MI_U32 i = 0; i < src.size(); i++)
    {
        if (src[i]->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
            return Status::ERROR;
        }
    }

    for(MI_U32 i = 0; i < dst.size(); i++)
    {
        if (dst[i]->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst must be mat type");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status CvtColorNeon::Run()
{
    Status ret = Status::OK;
    std::vector<const Mat*> src;
    std::vector<Mat*> dst;

    for (MI_U32 i = 0; i < m_src.size(); i++)
    {
        const Mat *mat = dynamic_cast<const Mat*>(m_src[i]);
        src.push_back(mat);
    }

    for (MI_U32 i = 0; i < m_dst.size(); i++)
    {
        Mat *mat = dynamic_cast<Mat*>(m_dst[i]);
        dst.push_back(mat);
    }

    if (src.empty() || dst.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    ret = CvtColorNeonImpl(m_ctx, src, dst, m_type, OpTarget::Neon());

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura