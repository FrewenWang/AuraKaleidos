#include "aura/ops/filter/bilateral.hpp"
#include "bilateral_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/ops/matrix.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<BilateralImpl> CreateBilateralImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<BilateralImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new BilateralNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new BilateralNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new BilateralCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
// #if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
//             impl.reset(new BilateralHvx(ctx, target));
// #endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            break;
        }

        default:
        {
            break;
        }
    }

    return impl;
}

Bilateral::Bilateral(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Bilateral::SetArgs(const Array *src, Array *dst, MI_F32 sigma_color, MI_F32 sigma_space,
                          MI_S32 ksize, BorderType border_type, const Scalar &border_value)
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
            // if (CheckHvxWidth(*src) != Status::OK)
            // {
            //     impl_target = OpTarget::None();
            // }
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
        m_impl = CreateBilateralImpl(m_ctx, impl_target);
    }

    // run SetArgs
    BilateralImpl *bilateral_impl = dynamic_cast<BilateralImpl *>(m_impl.get());
    if (MI_NULL == bilateral_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "bilateral_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = bilateral_impl->SetArgs(src, dst, sigma_color, sigma_space, ksize, border_type, border_value);

    AURA_RETURN(m_ctx, ret);
}

Status Bilateral::CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    MI_S32 ksh        = ksize >> 1;
    MI_S32 ksh_square = ksh * ksh;
    MI_S32 valid_num  = 0;

    for (MI_S32 i = -ksh; i <= ksh; i++)
    {
        for (MI_S32 j = -ksh; j <= ksh; j++)
        {
            MI_S32 r_square = i * i + j * j;
            if (r_square > ksh_square)
            {
                continue;
            }
            valid_num++;
        }
    }

    std::vector<CLKernel> cl_kernels = BilateralCL::GetCLKernels(ctx, elem_type, channel, ksize, border_type, valid_num);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Bilateral CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(channel);
    AURA_UNUSED(ksize);
    AURA_UNUSED(border_type);
#endif
    return Status::OK;
}

AURA_EXPORTS Status IBilateral(Context *ctx, const Mat &src, Mat &dst, MI_F32 sigma_color, MI_F32 sigma_space,
                               MI_S32 ksize, BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Bilateral bilateral(ctx, target);

    return OpCall(ctx, bilateral, &src, &dst, sigma_color, sigma_space, ksize, border_type, border_value);
}

BilateralImpl::BilateralImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Bilateral", target),
                                                                     m_ksize(0), m_border_type(BorderType::REFLECT_101),
                                                                     m_sigma_color(0.f), m_sigma_space(0.f),
                                                                     m_valid_num(0), m_scale_index(0.f),
                                                                     m_src(MI_NULL), m_dst(MI_NULL)
{}

Status BilateralImpl::SetArgs(const Array *src, Array *dst, MI_F32 sigma_color, MI_F32 sigma_space,
                                 MI_S32 ksize, BorderType border_type, const Scalar &border_value)
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
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same size and the same data type");
        return Status::ERROR;
    }

    if (ksize > src->GetSizes().m_width || ksize > src->GetSizes().m_height)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize must be less or equal src.GetSizes().m_width and src.GetSizes().m_height");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel != 1 && src->GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "bad channel number for BilateralFilter");
        return Status::ERROR;
    }

    m_src          = src;
    m_dst          = dst;
    m_sigma_color  = sigma_color;
    m_sigma_space  = sigma_space;
    m_ksize        = ksize;
    m_border_type  = border_type;
    m_border_value = border_value;

    return Status::OK;
}

Status BilateralImpl::Initialize()
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    Status ret = Status::OK;
    MI_S32 ksh = 0;

    m_sigma_space = (m_sigma_space <= 0) ? 1.f : m_sigma_space;
    ksh           = m_ksize <= 0 ? Round(m_sigma_space * 1.5) : m_ksize / 2;
    ksh           = Max(ksh, static_cast<MI_S32>(1));
    m_ksize       = ksh * 2 + 1;

    ret = PrepareSpaceMat();
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetSpaceMat fail");
        return ret;
    }

    ret = PrepareColorMat();
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetColorMat fail");
        return ret;
    }

    return ret;
}

Status BilateralImpl::DeInitialize()
{
    m_space_ofs.Release();
    m_space_weight.Release();
    m_color_weight.Release();

   return Status::OK;
}

std::vector<const Array*> BilateralImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> BilateralImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string BilateralImpl::ToString() const
{
    std::string str;

    MI_CHAR sigma_color_str[20];
    snprintf(sigma_color_str, sizeof(sigma_color_str), "%.2f", m_sigma_color);
    MI_CHAR sigma_space_str[20];
    snprintf(sigma_space_str, sizeof(sigma_space_str), "%.2f", m_sigma_space);

    str = "op(Bilateral)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + BorderTypeToString(m_border_type) + " | " + "ksize:" + std::to_string(m_ksize) + " | " +
            "sigma_color:" + sigma_color_str + " | " + "sigma_space:" + sigma_space_str + " | "
            "border_value:" + m_border_value.ToString() + ")\n";

    return str;
}

AURA_VOID BilateralImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_ksize, m_border_type,
                        m_border_value, m_sigma_color, m_sigma_space, m_valid_num, m_scale_index);
}

Sizes BilateralImpl::GetColorMatStride(Sizes3 color_size)
{
    AURA_UNUSED(color_size);

    return Sizes(1, 1);
}

Status BilateralImpl::PrepareSpaceMat()
{
    Status ret = Status::OK;

    MI_S32 ksh        = m_ksize >> 1;
    Sizes3 space_size = Sizes3(1, m_ksize * m_ksize, 1);

    // space
    m_space_weight = Mat(m_ctx, ElemType::F32, space_size);
    if (!m_space_weight.IsValid())
    {
        ret = Status::ERROR;
        AURA_ADD_ERROR_STRING(m_ctx, "space_weight mat invalid");
        return ret;
    }

    m_space_ofs = Mat(m_ctx, ElemType::S32, space_size);
    if (!m_space_ofs.IsValid())
    {
        ret = Status::ERROR;
        AURA_ADD_ERROR_STRING(m_ctx, "space_ofs mat invalid");
        return ret;
    }

    MI_F32 *sp_weight_data = (MI_F32*)m_space_weight.Ptr<MI_F32>(0);
    MI_S32 *sp_ofs_data    = (MI_S32*)m_space_ofs.Ptr<MI_S32>(0);

    MI_S32 ksh_square     = ksh * ksh;
    MI_F64 gauss_sp_coeff = -0.5 / (m_sigma_space * m_sigma_space);
    const MI_S32 ch       = m_src->GetSizes().m_channel;

    // Add (ksh<<1) because the input data calculated by the none code is after makeborder, neon/opencl don't need to use istep/space_ofs
    MI_S32 step  = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 istep = (m_src->GetSizes().m_width + (ksh << 1)) * ch;
    step         = (istep > step) ? istep : step;
    m_valid_num  = 0;

    for (MI_S32 i = -ksh; i <= ksh; i++)
    {
        for (MI_S32 j = -ksh; j <= ksh; j++)
        {
            MI_S32 r_square = i * i + j * j;
            if (r_square > ksh_square)
            {
                continue;
            }

            sp_weight_data[m_valid_num]   = static_cast<MI_F32>(Exp(r_square * gauss_sp_coeff));
            sp_ofs_data[m_valid_num++] = static_cast<MI_S32>(i * step + j * ch);
        }
    }

    AURA_RETURN(m_ctx, ret);
}

Status BilateralImpl::PrepareColorMat()
{
    Status ret = Status::OK;

    const MI_S32 ch            = m_src->GetSizes().m_channel;
    const MI_S32 bin_scale     = (m_src->GetElemType() == ElemType::U8) ? 256 : (1 << 12);
    const MI_S32 total_bin_num = bin_scale * ch;

    m_sigma_color            = (m_sigma_color <= 0) ? 1.f : m_sigma_color;
    MI_F64 gauss_color_coeff = -0.5 / (m_sigma_color * m_sigma_color);
    m_scale_index            = 1.0f;

    Sizes3 color_size  = Sizes3(1, total_bin_num + 2, 1);
    Sizes color_stride = GetColorMatStride(color_size);

    m_color_weight = Mat(m_ctx, ElemType::F32, color_size, AURA_MEM_DEFAULT, color_stride);
    if (!m_color_weight.IsValid())
    {
        ret = Status::ERROR;
        AURA_ADD_ERROR_STRING(m_ctx, "color_weight mat invalid");
        return ret;
    }

    MI_F32 *color_weight_data = (MI_F32*)m_color_weight.Ptr<MI_F32>(0);

    if (ElemType::U8 == m_src->GetElemType())
    {
        for (MI_S32 i = 0; i < 256 * ch; i++)
        {
            color_weight_data[i] = static_cast<MI_F32>(Exp(i * i * gauss_color_coeff));
        }
    }
    else
    {
        Point3i min_pos;
        Point3i max_pos;
        MI_F64 min_val, max_val;

        OpTarget cur_impl = m_target;
        cur_impl.m_type = (cur_impl.m_type != TargetType::NONE && cur_impl.m_type != TargetType::NEON) ? TargetType::NEON : cur_impl.m_type;
        ret = IMinMaxLoc(m_ctx, *(dynamic_cast<const Mat*>(m_src)), &min_val, &max_val, &min_pos, &max_pos, cur_impl);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "MinMaxLoc failed");
            return ret;
        }

        m_scale_index = bin_scale / (max_val - min_val);
        MI_F32 last_exp_val = 1.f;
        for (MI_S32 i = 0; i < total_bin_num + 2; i++)
        {
            if (last_exp_val > 0.f)
            {
                MI_F32 val           = i / m_scale_index;
                color_weight_data[i] = static_cast<MI_F32>(Exp(val * val * gauss_color_coeff));
                last_exp_val         = color_weight_data[i];
            }
            else
            {
                color_weight_data[i] = 0.f;
            }
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura