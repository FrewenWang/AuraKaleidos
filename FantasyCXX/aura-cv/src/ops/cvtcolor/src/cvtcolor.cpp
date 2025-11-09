#include "cvtcolor_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
static Status CheckHvxSupport(Context *ctx, const Array &array, CvtColorType type)
{
    Status ret = Status::ERROR;

    DT_S32 width = array.GetSizes().m_width;
    switch (type)
    {
        case CvtColorType::BGR2BGRA:
        case CvtColorType::BGRA2BGR:
        case CvtColorType::BGR2RGB:
        case CvtColorType::BGR2GRAY:
        case CvtColorType::BGRA2GRAY:
        case CvtColorType::RGB2GRAY:
        case CvtColorType::RGBA2GRAY:
        case CvtColorType::GRAY2BGR:
        case CvtColorType::GRAY2BGRA:
        case CvtColorType::YUV2RGB_Y444:
        case CvtColorType::RGB2YUV_Y444:
        case CvtColorType::YUV2RGB_Y444_601:
        case CvtColorType::RGB2YUV_Y444_601:
        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            ret = (width >= 128) ? Status::OK : Status::ERROR;
            break;
        }

        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        {
            ret = (width >= 256) ? Status::OK : Status::ERROR;
            break;
        }

        case CvtColorType::BAYERBG2BGR:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            ret = (width >= 258) ? Status::OK : Status::ERROR;
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
#endif // defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)

static Status CheckArrayElemType(Context *ctx, const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type)
{
    Status ret = Status::OK;

    const DT_S32 src_len = src.size();
    const DT_S32 dst_len = dst.size();

    /// 判断每个存放数据是否是非法的
    for (DT_S32 i = 0; i < src_len; i++)
    {
        if (!(src[i]->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "invalid src");
            return Status::ERROR;
        }
    }
    /// 判断每个输出数据是否是非法的
    for (DT_S32 i = 0; i < dst_len; i++)
    {
        if (!(dst[i]->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "invalid dst");
            return Status::ERROR;
        }
    }

    switch (type)
    {
        case CvtColorType::BGR2BGRA:
        case CvtColorType::BGRA2BGR:
        case CvtColorType::BGR2RGB:
        case CvtColorType::BGR2GRAY:
        case CvtColorType::RGB2GRAY:
        case CvtColorType::GRAY2BGR:
        case CvtColorType::GRAY2BGRA:
        case CvtColorType::BGRA2GRAY:
        case CvtColorType::RGBA2GRAY:
        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        case CvtColorType::RGB2YUV_Y444:
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        case CvtColorType::RGB2YUV_Y444_601:
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        case CvtColorType::YUV2RGB_Y444:
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        case CvtColorType::YUV2RGB_Y444_601:
        {
            /// 判断所有的数据是否是U8
            for (DT_S32 i = 0; i < src_len; i++)
            {
                if (src[i]->GetElemType() != ElemType::U8)
                {
                    AURA_ADD_ERROR_STRING(ctx, "src elem type should be u8");
                    return Status::ERROR;
                }
            }
            for (DT_S32 i = 0; i < dst_len; i++)
            {
                if (dst[i]->GetElemType() != ElemType::U8)
                {
                    AURA_ADD_ERROR_STRING(ctx, "dst elem type should be u8");
                    return Status::ERROR;
                }
            }
            break;
        }

        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            /// 判断所有的数据是否是U16
            for (DT_S32 i = 0; i < src_len; i++)
            {
                if (src[i]->GetElemType() != ElemType::U16)
                {
                    AURA_ADD_ERROR_STRING(ctx, "src elem type should be u16");
                    return Status::ERROR;
                }
            }
            for (DT_S32 i = 0; i < dst_len; i++)
            {
                if (dst[i]->GetElemType() != ElemType::U16)
                {
                    AURA_ADD_ERROR_STRING(ctx, "dst elem type should be u16");
                    return Status::ERROR;
                }
            }
            break;
        }
        /// 下面这两个我们先不管
        case CvtColorType::BAYERBG2BGR:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            for (DT_S32 i = 0; i < src_len; i++)
            {
                if (src[i]->GetElemType() != ElemType::U8 && src[i]->GetElemType() != ElemType::U16)
                {
                    AURA_ADD_ERROR_STRING(ctx, "src elem type should be u8|u16");
                    ret = Status::ERROR;
                }
            }
            for (DT_S32 i = 0; i < dst_len; i++)
            {
                if (dst[i]->GetElemType() != ElemType::U8 && dst[i]->GetElemType() != ElemType::U16)
                {
                    AURA_ADD_ERROR_STRING(ctx, "dst elem type should be u8|u16");
                    ret = Status::ERROR;
                }
            }
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

static Status CheckArrayNum(Context *ctx, const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type)
{
    Status ret = Status::ERROR;
    // 但是输入数据的大小
    const DT_S32 src_len = src.size();
    const DT_S32 dst_len = dst.size();

    switch (type)
    {
        case CvtColorType::BGR2BGRA:
        case CvtColorType::BGRA2BGR:
        case CvtColorType::BGR2RGB:
        case CvtColorType::BGR2GRAY:
        case CvtColorType::BGRA2GRAY:
        case CvtColorType::RGB2GRAY:
        case CvtColorType::RGBA2GRAY:
        case CvtColorType::GRAY2BGR:
        case CvtColorType::GRAY2BGRA:
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        case CvtColorType::BAYERBG2BGR:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            /// 上面所有RGB 和 灰度  和RGR先关的转换。其实就是单通道数据转单单通道数据
            /// 所以输入输出都是的size都是一。
            ret = (1 == src_len && 1 == dst_len) ? Status::OK : Status::ERROR;
            break;
        }

        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        {
            /// 上面NV21和NV12，因为是UV数据重叠存放，所以
            ret = (2 == src_len && 1 == dst_len) ? Status::OK : Status::ERROR;
            break;
        }

        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        case CvtColorType::YUV2RGB_Y444:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        case CvtColorType::YUV2RGB_Y444_601:
        {
            /// 上面的YU12和YV12因为数据UV数据是分开存放的。所以输入数据的size是
            ret = (3 == src_len && 1 == dst_len) ? Status::OK : Status::ERROR;
            break;
        }

        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            ret = (1 == src_len && 2 == dst_len) ? Status::OK : Status::ERROR;
            break;
        }

        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        case CvtColorType::RGB2YUV_Y444:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        case CvtColorType::RGB2YUV_Y444_601:
        {
            ret = (1 == src_len && 3 == dst_len) ? Status::OK : Status::ERROR;
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "the number of src mat or dst mat is not match to type");
    }

    AURA_RETURN(ctx, ret);
}

static std::shared_ptr<CvtColorImpl> CreateCvtColorImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<CvtColorImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new CvtColorNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new CvtColorNeon(ctx, target));
#endif // defined(AURA_ENABLE_NEON)
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new CvtColorCL(ctx, target));
#endif // defined(AURA_ENABLE_OPENCL)
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new CvtColorHvx(ctx, target));
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            break;
        }

        case TargetType::VDSP:
        {
#if (defined(AURA_ENABLE_XTENSA) || defined(AURA_BUILD_XTENSA))
            impl.reset(new CvtColorVdsp(ctx, target));
#endif // (defined(AURA_ENABLE_XTENSA) || defined(AURA_BUILD_XTENSA))
            break;
        }

        default:
        {
            break;
        }
    }

    return impl;
}

CvtColor::CvtColor(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status CvtColor::SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((0 == src.size()) || (0 == dst.size()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst size is zero");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // check impl type
    switch (m_target.m_type)
    {
        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            if (CheckNeonWidth(*(src[0])) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            if (CheckHvxSupport(m_ctx, *(src[0]), type) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            break;
        }

        default:
        {
            break;
        }
    }

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateCvtColorImpl(m_ctx, impl_target);
    }

    CvtColorImpl *cvtcolor_impl = dynamic_cast<CvtColorImpl*>(m_impl.get());
    if (DT_NULL == cvtcolor_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cvtcolor_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = cvtcolor_impl->SetArgs(src, dst, type);

    AURA_RETURN(m_ctx, ret);
}

Status CvtColor::CLPrecompile(Context *ctx, ElemType elem_type, CvtColorType cvtcolor_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = CvtColorCL::GetCLKernels(ctx, elem_type, cvtcolor_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Gaussian CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(cvtcolor_type);
#endif
    return Status::OK;
}

AURA_EXPORTS Status ICvtColor(Context *ctx, const std::vector<Mat> &src, std::vector<Mat> &dst,
                              CvtColorType type, const OpTarget &target)
{
    CvtColor cvtcolor(ctx, target);

    std::vector<const Array*> src_arrays;
    std::vector<Array*> dst_arrays;
    src_arrays.reserve(src.size());
    dst_arrays.reserve(dst.size());

    for (auto &src_mat : src)
    {
        src_arrays.push_back(&src_mat);
    }

    for (auto &dst_mat : dst)
    {
        dst_arrays.push_back(&dst_mat);
    }

    return OpCall(ctx, cvtcolor, src_arrays, dst_arrays, type);
}

CvtColorImpl::CvtColorImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, AURA_OPS_CVTCOLOR_OP_NAME, target), m_type(CvtColorType::INVALID)
{}

Status CvtColorImpl::SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }
    /// 设置数据参数。判断输入输出数据的通道数，存放的size
    if (CheckArrayNum(m_ctx, src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "array num is not match");
        return Status::ERROR;
    }

    if (CheckArrayElemType(m_ctx, src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "array elem type is not match");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;

    m_type = type;
    return Status::OK;
}

std::vector<const Array*> CvtColorImpl::GetInputArrays() const
{
    return m_src;
}

std::vector<const Array*> CvtColorImpl::GetOutputArrays() const
{
    std::vector<const Array*> dst_out;
    for (DT_U32 i = 0; i < m_dst.size(); i++)
    {
        dst_out.push_back(m_dst[i]);
    }

    return dst_out;
}

std::string CvtColorImpl::ToString() const
{
    std::string str;

    str = "op(CvtColor)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(m_type:" + CvtColorTypeToString(m_type) + ")\n";

    return str;
}

DT_VOID CvtColorImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    if (json_wrapper.SetArray("dst", m_dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray dst failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_type);
}

} // namespace aura
