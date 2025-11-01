#include "aura/ops/filter/median.hpp"
#include "median_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<MedianImpl> CreateMedianImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<MedianImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new MedianNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new MedianNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new MedianCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new MedianHvx(ctx, target));
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

Median::Median(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Median::SetArgs(const Array *src, Array *dst, MI_S32 ksize)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // check impl type
    switch (m_target.m_type)
    {
        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            if (CheckNeonWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            if (CheckCLWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            if (CheckHvxWidth(*src) != Status::OK)
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
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateMedianImpl(m_ctx, impl_target);
    }

    // run initialize
    MedianImpl *median_impl = dynamic_cast<MedianImpl *>(m_impl.get());
    if (MI_NULL == median_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "median_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = median_impl->SetArgs(src, dst, ksize);

    AURA_RETURN(m_ctx, ret);
}

Status Median::CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = MedianCL::GetCLKernels(ctx, elem_type, channel, ksize);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Median CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(channel);
    AURA_UNUSED(ksize);
#endif
    return Status::OK;
}

AURA_EXPORTS Status IMedian(Context *ctx, const Mat &src, Mat &dst, MI_S32 ksize, const OpTarget &target)
{
    Median median(ctx, target);

    return OpCall(ctx, median, &src, &dst, ksize);
}

MedianImpl::MedianImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Median", target),
                                                               m_ksize(0), m_src(MI_NULL), m_dst(MI_NULL)
{}

Status MedianImpl::SetArgs(const Array *src, Array *dst, MI_S32 ksize)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!(src->IsValid() && dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return Status::ERROR;
    }

    if (!src->IsEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same size");
        return Status::ERROR;
    }

    if ((ksize & 1) != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize must be odd");
        return Status::ERROR;
    }

    if ((src->GetSizes().m_height < ksize) || (src->GetSizes().m_width < ksize))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "height/width must be bigger than ksize");
        return Status::ERROR;
    }

    m_src   = src;
    m_dst   = dst;
    m_ksize = ksize;

    return Status::OK;
}

std::vector<const Array*> MedianImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> MedianImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string MedianImpl::ToString() const
{
    std::string str;

    str = "op(Median)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" "| ksize:" + std::to_string(m_ksize) + ")\n";

    return str;
}

AURA_VOID MedianImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_ksize);
}

} // namespace aura