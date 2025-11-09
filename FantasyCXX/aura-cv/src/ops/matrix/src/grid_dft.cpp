#include "aura/ops/matrix/grid_dft.hpp"
#include "grid_dft_impl.hpp"
#include "dft_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{
#  if defined(AURA_ENABLE_OPENCL)
static Status CheckCLSupport(const Array *src)
{
    Sizes3 sz = src->GetSizes();
    if (sz.m_height < 32 || sz.m_width < 32)
    {
        return Status::ERROR;
    }

    return Status::OK;
}
#endif // AURA_ENABLE_OPENCL

static std::shared_ptr<GridDftImpl> CreateGridDftImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<GridDftImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new GridDftNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new GridDftNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new GridDftCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

GridDft::GridDft(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status GridDft::SetArgs(const Array *src, Array *dst, DT_S32 grid_len)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst))
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
#  if defined(AURA_ENABLE_OPENCL)
            if (CheckCLSupport(src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_OPENCL
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
        m_impl = CreateGridDftImpl(m_ctx, impl_target);
    }

    // run initialize
    GridDftImpl *grid_dft_impl = dynamic_cast<GridDftImpl*>(m_impl.get());
    if (DT_NULL == grid_dft_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "grid_dft_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = grid_dft_impl->SetArgs(src, dst, grid_len);

    AURA_RETURN(m_ctx, ret);
}

Status GridDft::CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 grid_len)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = GridDftCL::GetCLKernels(ctx, elem_type, grid_len);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GridDft CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(grid_len);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}

AURA_EXPORTS Status IGridDft(Context *ctx, const Mat &src, Mat &dst, DT_S32 grid_len, const OpTarget &target)
{
    GridDft grid_dft(ctx, target);

    return OpCall(ctx, grid_dft, &src, &dst, grid_len);
}

GridDftImpl::GridDftImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "GridDft", target),
                                                                 m_src(DT_NULL), m_dst(DT_NULL),
                                                                 m_grid_len(0)
{}

Status GridDftImpl::SetArgs(const Array *src, Array *dst, DT_S32 grid_len)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (DT_NULL == src || DT_NULL == dst)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    if (!src->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    Sizes3 src_sz = src->GetSizes();
    Sizes3 dst_sz = dst->GetSizes();

    if (!IsPowOf2(grid_len))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "grid length must be pow of 2.");
        return Status::ERROR;
    }

    if (src_sz.m_width != dst_sz.m_width || src_sz.m_height != dst_sz.m_height)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst shape doesn't match.");
        return Status::ERROR;
    }

    if (src_sz.m_channel != 1 || dst_sz.m_channel != 2)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridDft only support real input with ch = 1 and complex output with ch = 2 ");
        return Status::ERROR;
    }

    if (ElemType::F32 != dst->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "currently dst does only support DT_F32 type.");
        return Status::ERROR;
    }

    if ((src_sz.m_width % grid_len != 0) || (src_sz.m_height % grid_len != 0))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "height and width must be multiple of grid_len.");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;
    m_grid_len = grid_len;

    return Status::OK;
}

std::vector<const Array*> GridDftImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> GridDftImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string GridDftImpl::ToString() const
{
    std::string str;

    str = "op(GridDft)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID GridDftImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_grid_len);
}

static std::shared_ptr<GridIDftImpl> CreateGridIDftImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<GridIDftImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new GridIDftNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new GridIDftNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new GridIDftCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

GridIDft::GridIDft(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status GridIDft::SetArgs(const Array *src, Array *dst, DT_S32 grid_len, DT_BOOL with_scale)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst))
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
#  if defined(AURA_ENABLE_OPENCL)
            if (CheckCLSupport(src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_OPENCL
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
        m_impl = CreateGridIDftImpl(m_ctx, impl_target);
    }

    // run initialize
    GridIDftImpl *grididft_impl = dynamic_cast<GridIDftImpl*>(m_impl.get());
    if (DT_NULL == grididft_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "grididft_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = grididft_impl->SetArgs(src, dst, grid_len, with_scale);

    AURA_RETURN(m_ctx, ret);
}

Status GridIDft::CLPrecompile(Context *ctx, ElemType elem_type, DT_S32 grid_len, DT_S32 with_scale, DT_BOOL save_real_only)
{
#if defined(AURA_ENABLE_OPENCL)
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = GridIDftCL::GetCLKernels(ctx, elem_type, grid_len, with_scale, save_real_only);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GridIDft CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(grid_len);
    AURA_UNUSED(with_scale);
    AURA_UNUSED(save_real_only);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}
AURA_EXPORTS Status IGridIDft(Context *ctx, const Mat &src, Mat &dst, DT_S32 grid_len, DT_BOOL with_scale, const OpTarget &target)
{
    GridIDft grid_idft(ctx, target);

    return OpCall(ctx, grid_idft, &src, &dst, grid_len, with_scale);
}

GridIDftImpl::GridIDftImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "GridIDft", target),
                                                                   m_src(DT_NULL), m_dst(DT_NULL),
                                                                   m_grid_len(0), m_with_scale(DT_FALSE)
{}

Status GridIDftImpl::SetArgs(const Array *src, Array *dst, DT_S32 grid_len, DT_BOOL with_scale)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (DT_NULL == src || DT_NULL == dst)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    if (!src->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    Sizes3 src_sz = src->GetSizes();
    Sizes3 dst_sz = dst->GetSizes();

    if (!IsPowOf2(grid_len))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "grid length must be pow of 2.");
        return Status::ERROR;
    }

    if (src_sz.m_width != dst_sz.m_width || src_sz.m_height != dst_sz.m_height)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst shape doesn't match.");
        return Status::ERROR;
    }

    if (src_sz.m_channel != 2)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridIDft only support complex input with ch = 2");
        return Status::ERROR;
    }

    if (src->GetElemType() != ElemType::F32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "currently src only supports DT_F32 type.");
        return Status::ERROR;
    }

    if (dst_sz.m_channel != 1 && dst_sz.m_channel != 2)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridIDft only support real output with ch = 1 and complex output with ch = 2 ");
        return Status::ERROR;
    }

    if ((src_sz.m_width % grid_len != 0) || (src_sz.m_height % grid_len != 0))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "height and width must be multiple of grid_len.");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;
    m_grid_len = grid_len;
    m_with_scale = with_scale;

    return Status::OK;
}

std::vector<const Array*> GridIDftImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> GridIDftImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string GridIDftImpl::ToString() const
{
    std::string str;

    str = "op(GridIDft)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID GridIDftImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_grid_len, m_with_scale);
}

} // namespace aura
