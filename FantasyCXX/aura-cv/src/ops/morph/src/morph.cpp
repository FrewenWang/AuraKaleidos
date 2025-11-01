#include "morph_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/tools/json.h"

namespace aura
{

#if defined(AURA_ENABLE_OPENCL)
AURA_INLINE Status CheckCLSupport(Context *ctx, const Array &array)
{
    if ((array.GetSizes().m_width < 64) || (array.GetSizes().m_height < 6))
    {
        AURA_ADD_ERROR_STRING(ctx, "opencl impl cannot support the array img size");
        return Status::ERROR;
    }

    return Status::OK;
}
#endif

static std::shared_ptr<MorphImpl> CreateMorphImpl(Context *ctx, MorphType type, const OpTarget &target)
{
    std::shared_ptr<MorphImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new MorphNone(ctx, type, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new MorphNeon(ctx, type, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new MorphCL(ctx, type, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new MorphHvx(ctx, type, target));
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

Dilate::Dilate(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Dilate::SetArgs(const Array *src, Array *dst, MI_S32 ksize, MorphShape shape, MI_S32 iterations)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

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

// FIXME: Opencl impl is buggy, resolve it later
//         case TargetType::OPENCL:
//         {
// #if defined(AURA_ENABLE_OPENCL)
//             if (CheckCLSupport(m_ctx, *src) != Status::OK)
//             {
//                 impl_target = OpTarget::None();
//             }
// #endif // defined(AURA_ENABLE_OPENCL)
//             break;
//         }

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
        m_impl = CreateMorphImpl(m_ctx, MorphType::DILATE, impl_target);
    }

    // run initialize
    MorphImpl *morph_impl = dynamic_cast<MorphImpl*>(m_impl.get());
    if (MI_NULL == morph_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "morph_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = morph_impl->SetArgs(src, dst, ksize, shape, iterations);

    AURA_RETURN(m_ctx, ret);
}

Status Dilate::CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, MorphShape shape)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = MorphCL::GetCLKernels(ctx, elem_type, channel, ksize, shape, MorphType::DILATE);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Dilate CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(channel);
    AURA_UNUSED(ksize);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(shape);
#endif

    return Status::OK;
}

Erode::Erode(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Erode::SetArgs(const Array *src, Array *dst, MI_S32 ksize, MorphShape shape, MI_S32 iterations)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

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
            if (CheckCLSupport(m_ctx, *src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // defined(AURA_ENABLE_OPENCL)
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
        m_impl = CreateMorphImpl(m_ctx, MorphType::ERODE, impl_target);
    }

    // run initialize
    MorphImpl *morph_impl = dynamic_cast<MorphImpl*>(m_impl.get());
    if (MI_NULL == morph_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "morph_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = morph_impl->SetArgs(src, dst, ksize, shape, iterations);

    AURA_RETURN(m_ctx, ret);
}

Status Erode::CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, MorphShape shape)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = MorphCL::GetCLKernels(ctx, elem_type, channel, ksize, shape, MorphType::ERODE);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Erode CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(channel);
    AURA_UNUSED(ksize);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(shape);
#endif

    return Status::OK;
}

Status IDilate(Context *ctx, const Mat &src, Mat &dst, MI_S32 ksize, MorphShape shape,
               MI_S32 iterations, const OpTarget &target)
{
    Dilate dilate(ctx, target);

    return OpCall(ctx, dilate, &src, &dst, ksize, shape, iterations);
}

Status IErode(Context *ctx, const Mat &src, Mat &dst, MI_S32 ksize, MorphShape shape,
              MI_S32 iterations, const OpTarget &target)
{
    Erode erode(ctx, target);

    return OpCall(ctx, erode, &src, &dst, ksize, shape, iterations);
}

AURA_EXPORTS Status IMorphologyEx(Context *ctx, const Mat &src, Mat &dst, MorphType type, MI_S32 ksize,
                                  MorphShape shape, MI_S32 iterations, const OpTarget &target)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (type)
    {
        case MorphType::ERODE:
        {
            ret = IErode(ctx, src, dst, ksize, shape, iterations, target);
            break;
        }

        case MorphType::DILATE:
        {
            ret = IDilate(ctx, src, dst, ksize, shape, iterations, target);
            break;
        }

        case MorphType::OPEN:
        {
            Mat mid_mat(ctx, dst.GetElemType(), dst.GetSizes());
            if (!mid_mat.IsValid())
            {
                AURA_ADD_ERROR_STRING(ctx, "invalid mid_mat");
                return Status::ERROR;
            }

            if (IErode(ctx, src, mid_mat, ksize, shape, iterations, target) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Open operation call erode failed");
                return Status::ERROR;
            }
            ret = IDilate(ctx, mid_mat, dst, ksize, shape, iterations, target);
            break;
        }

        case MorphType::CLOSE:
        {
            Mat mid_mat(ctx, dst.GetElemType(), dst.GetSizes());
            if (!mid_mat.IsValid())
            {
                AURA_ADD_ERROR_STRING(ctx, "invalid mid_mat");
                return Status::ERROR;
            }

            if (IDilate(ctx, src, mid_mat, ksize, shape, iterations, target) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Close operation call dilate failed");
                return Status::ERROR;
            }
            ret = IErode(ctx, mid_mat, dst, ksize, shape, iterations, target);
            break;
        }

        case MorphType::GRADIENT:
        {
            Mat mid_mat1(ctx, dst.GetElemType(), dst.GetSizes());
            Mat mid_mat2(ctx, dst.GetElemType(), dst.GetSizes());
            if ((!mid_mat1.IsValid()) || !mid_mat2.IsValid())
            {
                AURA_ADD_ERROR_STRING(ctx, "invalid mid_mat1 or mid_mat2");
                return Status::ERROR;
            }

            if (IDilate(ctx, src, mid_mat1, ksize, shape, iterations, target) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Gradient operation call dilate failed");
                return Status::ERROR;
            }
            if (IErode(ctx, src, mid_mat2, ksize, shape, iterations, target) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Gradient operation call erode failed");
                return Status::ERROR;
            }
            ret = ISubtract(ctx, mid_mat1, mid_mat2, dst, target);
            break;
        }

        case MorphType::TOPHAT:
        {
            Mat mid_mat(ctx, dst.GetElemType(), dst.GetSizes());
            if (!mid_mat.IsValid())
            {
                AURA_ADD_ERROR_STRING(ctx, "invalid mid_mat");
                return Status::ERROR;
            }

            if (IErode(ctx, src, mid_mat, ksize, shape, iterations, target) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Tophat operation call erode failed");
                return Status::ERROR;
            }
            if (IDilate(ctx, mid_mat, dst, ksize, shape, iterations, target) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Tophat operation call dilate failed");
                return Status::ERROR;
            }
            ret = ISubtract(ctx, src, dst, dst, target);
            break;
        }

        case MorphType::BLACKHAT:
        {
            Mat mid_mat(ctx, dst.GetElemType(), dst.GetSizes());
            if (!mid_mat.IsValid())
            {
                AURA_ADD_ERROR_STRING(ctx, "invalid mid_mat");
                return Status::ERROR;
            }

            if (IDilate(ctx, src, mid_mat, ksize, shape, iterations, target) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Blackhat operation call dilate failed");
                return Status::ERROR;
            }
            if (IErode(ctx, mid_mat, dst, ksize, shape, iterations, target) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "Blackhat operation call erode failed");
                return Status::ERROR;
            }
            ret = ISubtract(ctx, dst, src, dst, target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unsupported morph type");
            return Status::ERROR;
        }
    }
    AURA_RETURN(ctx, ret);
}

MorphImpl::MorphImpl(Context *ctx, MorphType type, const OpTarget &target) : OpImpl(ctx, "Morph", target),
                                                                             m_ksize(0), m_type(type), m_shape(MorphShape::RECT),
                                                                             m_iterations(1), m_src(MI_NULL), m_dst(MI_NULL)
{}

Status MorphImpl::SetArgs(const Array *src, Array *dst, MI_S32 ksize, MorphShape shape, MI_S32 iterations)
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
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst not equal");
        return Status::ERROR;
    }

    if (((ksize & 1) == 0) || (ksize <= 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid ksize number");
        return Status::ERROR;
    }

    if (iterations <= 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "iterations must be a positive number");
        return Status::ERROR;
    }

    m_src        = src;
    m_dst        = dst;
    m_ksize      = ksize;
    m_shape      = shape;
    m_iterations = iterations;

    return Status::OK;
}

std::vector<const Array*> MorphImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> MorphImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string MorphImpl::ToString() const
{
    std::string str;

    str = "op(Morph)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + MorphTypeToString(m_type) + " | " +
            "ksize:" + std::to_string(m_ksize) + " | " + MorphShapeToString(m_shape) + " | " +
            "iterations:" + std::to_string(m_iterations) + ")\n";

    return str;
}

AURA_VOID MorphImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_ksize, m_type, m_shape, m_iterations);
}

} // namespace aura
