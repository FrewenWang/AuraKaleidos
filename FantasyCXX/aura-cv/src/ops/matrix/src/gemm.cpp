#include "aura/ops/matrix/gemm.hpp"
#include "gemm_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<GemmImpl> CreateGemmImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<GemmImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new GemmNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new GemmNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            GpuType m_gpu_type = ctx->GetCLEngine()->GetCLRuntime()->GetGpuInfo().m_type;

            if (GpuType::ADRENO == m_gpu_type)
            {
                impl.reset(new GemmAdrenoCL(ctx, target));
            }
            else if (GpuType::MALI == m_gpu_type)
            {
                impl.reset(new GemmMaliCL(ctx, target));
            }
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

#if defined(AURA_ENABLE_NEON)
AURA_INLINE Status CheckGemmNeonParam(const Array *mat)
{
    MI_S32 width  = mat->GetSizes().m_width;
    MI_S32 height = mat->GetSizes().m_height;
    if ((width < 4) || (height < 4))
    {
        return Status::ERROR;
    }
    return Status::OK;
}
#endif // AURA_ENABLE_NEON

#if defined(AURA_ENABLE_OPENCL)
static Status CheckCLSupport(const Array *src0, const Array *src1)
{
    MI_S32 m = src0->GetSizes().m_height;
    MI_S32 k = src0->GetSizes().m_width;
    MI_S32 n = src1->GetSizes().m_width;

    if (m < 64 || n < 64 || k < 8)
    {
        return Status::ERROR;
    }

    return Status::OK;
}
#endif // AURA_ENABLE_OPENCL

Gemm::Gemm(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Gemm::SetArgs(const Array *src0, const Array *src1, Array *dst)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src0) || (MI_NULL == src1) || (MI_NULL == dst))
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
            if (CheckGemmNeonParam(src0) != Status::OK || CheckGemmNeonParam(src1) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            if (CheckCLSupport(src0, src1) != Status::OK)
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
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateGemmImpl(m_ctx, impl_target);
    }

    // run initialize
    GemmImpl *gemm_impl = dynamic_cast<GemmImpl*>(m_impl.get());
    if (MI_NULL == gemm_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "gemm_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = gemm_impl->SetArgs(src0, src1, dst);

    AURA_RETURN(m_ctx, ret);
}

Status Gemm::CLPrecompile(Context *ctx)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    GpuType m_gpu_type = ctx->GetCLEngine()->GetCLRuntime()->GetGpuInfo().m_type;

    std::vector<CLKernel> cl_kernels;

    if (GpuType::ADRENO == m_gpu_type)
    {
        MI_S32 elem_counts = 8;
        MI_S32 bm = 64;
        MI_S32 bn = 64;
        MI_S32 bk = 8;
        MI_S32 load_size   = bm * bk / (2 * (bn / 8) * (bm / 8));

        cl_kernels = GemmAdrenoCL::GetCLKernels(ctx, elem_counts, load_size);
    }
    else if (GpuType::MALI == m_gpu_type)
    {
        cl_kernels = GemmMaliCL::GetCLKernels(ctx);
    }

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Gemm CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
#endif // AURA_ENABLE_OPENCL

    return Status::OK;
}

AURA_EXPORTS Status IGemm(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target)
{
    Gemm gemm(ctx, target);

    return OpCall(ctx, gemm, &src0, &src1, &dst);
}

GemmImpl::GemmImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Gemm", target),
                                                           m_src0(MI_NULL), m_src1(MI_NULL),
                                                           m_dst(MI_NULL)
{}

Status GemmImpl::SetArgs(const Array *src0, const Array *src1, Array *dst)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src0) || (MI_NULL == src1) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    if (!src0->IsValid() || !src1->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    if (src0->GetElemType() != ElemType::F32 ||
        src1->GetElemType() != ElemType::F32 ||
         dst->GetElemType() != ElemType::F32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the elemtype should be float");
        return Status::ERROR;
    }

    Sizes3 src0_sz = src0->GetSizes();
    Sizes3 src1_sz = src1->GetSizes();
    Sizes3 dst_sz  = dst->GetSizes();

    if (src0_sz.m_width != src1_sz.m_height || src1_sz.m_width != dst_sz.m_width || src0_sz.m_height != dst_sz.m_height)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GemmCheckParam failed, src and dst shape doesn't match.");
        return Status::ERROR;
    }

    m_src0 = src0;
    m_src1 = src1;
    m_dst  = dst;
    return Status::OK;
}

std::vector<const Array*> GemmImpl::GetInputArrays() const
{
    return {m_src0, m_src1};
}

std::vector<const Array*> GemmImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string GemmImpl::ToString() const
{
    std::string str;

    str = "op(Gemm)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

AURA_VOID GemmImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src_0", "src_1", "dst"};
    std::vector<const Array*> arrays = {m_src0, m_src1, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src0, m_src1, m_dst);
}

} // namespace aura