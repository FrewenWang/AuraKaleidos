#include "cvtcolor_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetCLName(CvtColorType type, std::vector<std::string> &program_name, std::vector<std::string> &kernel_name)
{
    switch (type)
    {
        case CvtColorType::BGR2GRAY:
        case CvtColorType::RGB2GRAY:
        {
            kernel_name.emplace_back("CvtBgr2Gray");
            program_name.emplace_back("aura_cvt_bgr2gray");
            break;
        }
        case CvtColorType::BGRA2GRAY:
        case CvtColorType::RGBA2GRAY:
        {
            kernel_name.emplace_back("CvtBgra2Gray");
            program_name.emplace_back("aura_cvt_bgra2gray");
            break;
        }

        // BAYER -> BGR
        case CvtColorType::BAYERBG2BGR:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            kernel_name.emplace_back("CvtBayer2bgrMain");
            kernel_name.emplace_back("CvtBayer2bgrRemain");
            program_name.emplace_back("aura_cvt_bayer2bgr_main");
            program_name.emplace_back("aura_cvt_bayer2bgr_remain");
            break;
        }

        // YUV -> RGB
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        {
            kernel_name.emplace_back("CvtNv2Rgb");
            program_name.emplace_back("aura_cvt_nv2rgb");
            break;
        }
        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        {
            kernel_name.emplace_back("CvtY4202Rgb");
            program_name.emplace_back("aura_cvt_y4202rgb");
            break;
        }
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        {
            kernel_name.emplace_back("CvtY4222Rgb");
            program_name.emplace_back("aura_cvt_y4222rgb");
            break;
        }
        case CvtColorType::YUV2RGB_Y444:
        case CvtColorType::YUV2RGB_Y444_601:
        {
            kernel_name.emplace_back("CvtY4442Rgb");
            program_name.emplace_back("aura_cvt_y4442rgb");
            break;
        }

        // RGB -> YUV
        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        {
            kernel_name.emplace_back("CvtRgb2Nv");
            program_name.emplace_back("aura_cvt_rgb2nv");
            break;
        }
        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        {
            kernel_name.emplace_back("CvtRgb2Y420");
            program_name.emplace_back("aura_cvt_rgb2y420");
            break;
        }
        case CvtColorType::RGB2YUV_Y444:
        case CvtColorType::RGB2YUV_Y444_601:
        {
            kernel_name.emplace_back("CvtRgb2Y444");
            program_name.emplace_back("aura_cvt_rgb2y444");
            break;
        }
        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            kernel_name.emplace_back("CvtRgb2NvP010");
            program_name.emplace_back("aura_cvt_rgb2nv_p010");
            break;
        }

        default:
        {
            return Status::ERROR;
        }
    }

    return Status::OK;
}

static std::string GetCLBuildOptions(Context *ctx, ElemType elem_type, CvtColorType type)
{
    CLBuildOptions cl_build_opt(ctx);

    switch (type)
    {
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        case CvtColorType::YUV2RGB_Y444_601:
        {
            cl_build_opt.AddOption("CVTCOLOR_YUV2RGB_601", "1");
            break;
        }

        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        case CvtColorType::RGB2YUV_Y444_601:
        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            cl_build_opt.AddOption("CVTCOLOR_RGB2YUV_601", "1");
            break;
        }

        case CvtColorType::BAYERBG2BGR:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            cl_build_opt.AddOption("Tp", CLTypeString(elem_type));
            break;
        }

        default:
        {
            break;
        }
    }

    return cl_build_opt.ToString();
}

CvtColorCL::CvtColorCL(Context *ctx, const OpTarget &target) : CvtColorImpl(ctx, target)
{}

Status CvtColorCL::SetArgs(const std::vector<const Array *> &src, const std::vector<Array *> &dst, CvtColorType type)
{
    if (CvtColorImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CvtColorImpl::SetArgs failed(none)");
        return Status::ERROR;
    }

    if ((CvtColorType::BGR2BGRA == type) || (CvtColorType::BGRA2BGR == type) || (CvtColorType::BGR2RGB == type) ||
        (CvtColorType::GRAY2BGR == type) || (CvtColorType::GRAY2BGRA == type))
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("opencl target not support cvtcolor type: " + CvtColorTypeToString(type)).c_str());
        return Status::ERROR;
    }

    return Status::OK;
}

Status CvtColorCL::Initialize()
{
    if (CvtColorImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CvtColorImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 1. init cl_mem
    m_cl_src.clear();
    for (MI_U32 i = 0; i < m_src.size(); i++)
    {
        m_cl_src.push_back(CLMem::FromArray(m_ctx, *(m_src[i]), CLMemParam(CL_MEM_READ_ONLY)));
        if (!m_cl_src[i].IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
            return Status::ERROR;
        }
    }

    m_cl_dst.clear();
    for (MI_U32 i = 0; i < m_dst.size(); i++)
    {
        m_cl_dst.push_back(CLMem::FromArray(m_ctx, *(m_dst[i]), CLMemParam(CL_MEM_WRITE_ONLY)));
        if (!m_cl_dst[i].IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
            return Status::ERROR;
        }
    }

    // 2. init kernel
    m_cl_kernels = CvtColorCL::GetCLKernels(m_ctx, m_dst[0]->GetElemType(), m_type);

    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync start
    for (MI_U32 i = 0; i < m_src.size(); i++)
    {
        if (m_cl_src[i].Sync(CLMemSyncType::WRITE) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src Sync start failed");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status CvtColorCL::DeInitialize()
{
    for (MI_U32 i = 0; i < m_cl_src.size(); i++)
    {
        m_cl_src[i].Release();
    }

    for (MI_U32 i = 0; i < m_cl_dst.size(); i++)
    {
        m_cl_dst[i].Release();
    }

    for (MI_U32 i = 0; i < m_cl_kernels.size(); i++)
    {
        m_cl_kernels[i].DeInitialize();
    }

    if (CvtColorImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CvtColorImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status CvtColorCL::Run()
{
    Status ret = Status::ERROR;

    switch (m_type)
    {
        case CvtColorType::BGR2GRAY:
        case CvtColorType::BGRA2GRAY:
        case CvtColorType::RGB2GRAY:
        case CvtColorType::RGBA2GRAY:
        {
            ret = CvtBgr2GrayCLImpl();
            break;
        }

        // YUV -> RGB
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        {
            ret = CvtNv2RgbCLImpl();
            break;
        }

        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        {
            ret = CvtY4202RgbCLImpl();
            break;
        }

        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        {
            ret = CvtY4222RgbCLImpl();
            break;
        }

        case CvtColorType::YUV2RGB_Y444:
        case CvtColorType::YUV2RGB_Y444_601:
        {
            ret = CvtY4442RgbCLImpl();
            break;
        }

        // RGB -> YUV
        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            ret = CvtRgb2NvCLImpl();
            break;
        }

        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        {
            ret = CvtRgb2Y420CLImpl();
            break;
        }

        case CvtColorType::RGB2YUV_Y444:
        case CvtColorType::RGB2YUV_Y444_601:
        {
            ret = CvtRgb2Y444CLImpl();
            break;
        }

        // BAYER -> BGR
        case CvtColorType::BAYERBG2BGR:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            ret = CvtBayer2BgrCLImpl();
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string CvtColorCL::ToString() const
{
    return CvtColorImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> CvtColorCL::GetCLKernels(Context *ctx, ElemType elem_type, CvtColorType cvtcolor_type)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, elem_type, cvtcolor_type);

    std::vector<std::string> kernel_name, program_name;
    if (GetCLName(cvtcolor_type, program_name, kernel_name) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName failed");
        return cl_kernels;
    }

    for (size_t i = 0; i < kernel_name.size(); i++)
    {
        cl_kernels.push_back({ctx, program_name[i], kernel_name[i], "", build_opt});
    }

    return cl_kernels;
}

Status CvtColorCL::CvtBgr2GrayCLImpl()
{
    if ((m_cl_src[0].GetSizes() != m_cl_dst[0].GetSizes() * Sizes3(1, 1, 3) &&
         m_cl_src[0].GetSizes() != m_cl_dst[0].GetSizes() * Sizes3(1, 1, 4)) ||
         m_cl_dst[0].GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the sizes of src and dst do not match");
        return Status::ERROR;
    }

    MI_S32 height = m_cl_src[0].GetSizes().m_height;
    MI_S32 width  = m_cl_src[0].GetSizes().m_width;
    MI_S32 istep  = m_cl_src[0].GetRowPitch() / ElemTypeSize(m_cl_src[0].GetElemType());
    MI_S32 ostep  = m_cl_dst[0].GetRowPitch() / ElemTypeSize(m_cl_dst[0].GetElemType());

    MI_S32 shift   = 15;
    MI_S32 b_coeff = Bgr2GrayParam::BC;
    MI_S32 g_coeff = Bgr2GrayParam::GC;
    MI_S32 r_coeff = Bgr2GrayParam::RC;

    if (SwapBlue(m_type))
    {
        Swap(b_coeff, r_coeff);
    }

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size = cl::NDRange((width + 3) >> 2, height);

    // 2. opencl run
    cl::Event cl_event;
    cl_int    cl_ret = CL_SUCCESS;
    Status    ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32>(
                                 m_cl_src[0].GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst[0].GetCLMemRef<cl::Buffer>(), ostep,
                                 width,
                                 (MI_S32)(cl_global_size.get()[1]),
                                 (MI_S32)(cl_global_size.get()[0]),
                                 b_coeff, g_coeff, r_coeff, shift,
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &cl_event);
    if (cl_ret != CL_SUCCESS)
    {
        std::cout << m_ctx->GetLogger()->GetErrorString() << std::endl;
        AURA_ADD_ERROR_STRING(m_ctx, ("Run kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst[0]->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    // 8. sync end
    ret = m_cl_dst[0].Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

Status CvtColorCL::CvtNv2RgbCLImpl()
{
    if ((m_cl_src[0].GetSizes().m_width & 1) || (m_cl_src[0].GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "width and height must be even");
        return Status::ERROR;
    }

    if (m_cl_src[0].GetSizes() * Sizes3(1, 1, 2) != m_cl_src[1].GetSizes() * Sizes3(2, 2, 1) ||
        m_cl_src[0].GetSizes() * Sizes3(1, 1, 3) != m_cl_dst[0].GetSizes() ||
        m_cl_src[0].GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the sizes of src and dst do not match");
        return Status::ERROR;
    }
    ///
    MI_S32 height = m_cl_src[0].GetSizes().m_height;
    MI_S32 width  = m_cl_src[0].GetSizes().m_width;
    MI_S32 istep0 = m_cl_src[0].GetStrides().m_width / ElemTypeSize(m_cl_src[0].GetElemType());
    MI_S32 istep1 = m_cl_src[1].GetStrides().m_width / ElemTypeSize(m_cl_src[1].GetElemType());
    MI_S32 ostep  = m_cl_dst[0].GetStrides().m_width / ElemTypeSize(m_cl_dst[0].GetElemType());

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    // 4096 同时处理8个数据。那么我们
    cl::NDRange cl_global_size = {cl::NDRange((width + 7) >> 3, (height >> 1))};

    // 2. opencl run
    cl::Event cl_event;
    cl_int    cl_ret = CL_SUCCESS;
    Status    ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_U8>(
                                 m_cl_src[0].GetCLMemRef<cl::Buffer>(), istep0,
                                 m_cl_src[1].GetCLMemRef<cl::Buffer>(), istep1,
                                 m_cl_dst[0].GetCLMemRef<cl::Buffer>(), ostep,
                                 width,
                                 (MI_S32)(cl_global_size.get()[1]),
                                 (MI_S32)(cl_global_size.get()[0]),
                                 static_cast<MI_U8>(SwapUv(m_type)),
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &cl_event);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst[0]->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    ret = m_cl_dst[0].Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

Status CvtColorCL::CvtY4202RgbCLImpl()
{
    if ((m_cl_src[0].GetSizes().m_width & 1) || (m_cl_src[0].GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "width and height must be even");
        return Status::ERROR;
    }

    if (m_cl_src[0].GetSizes() != m_cl_src[1].GetSizes() * Sizes3(2, 2, 1) ||
        m_cl_src[0].GetSizes() != m_cl_src[2].GetSizes() * Sizes3(2, 2, 1) ||
        m_cl_src[0].GetSizes() * Sizes3(1, 1, 3) != m_cl_dst[0].GetSizes() ||
        m_cl_src[0].GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the sizes of src and dst do not match");
        return Status::ERROR;
    }

    MI_S32 height = m_cl_src[0].GetSizes().m_height;
    MI_S32 width  = m_cl_src[0].GetSizes().m_width;
    MI_S32 istep0 = m_cl_src[0].GetStrides().m_width / ElemTypeSize(m_cl_src[0].GetElemType());
    MI_S32 istep1 = m_cl_src[1].GetStrides().m_width / ElemTypeSize(m_cl_src[1].GetElemType());
    MI_S32 istep2 = m_cl_src[2].GetStrides().m_width / ElemTypeSize(m_cl_src[2].GetElemType());
    MI_S32 ostep  = m_cl_dst[0].GetStrides().m_width / ElemTypeSize(m_cl_dst[0].GetElemType());

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size = {cl::NDRange((width + 7) >> 3, (height >> 1))};

    // 2. opencl run
    cl::Event cl_event;
    cl_int    cl_ret = CL_SUCCESS;
    Status    ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_U8>(
                                 m_cl_src[0].GetCLMemRef<cl::Buffer>(), istep0,
                                 m_cl_src[1].GetCLMemRef<cl::Buffer>(), istep1,
                                 m_cl_src[2].GetCLMemRef<cl::Buffer>(), istep2,
                                 m_cl_dst[0].GetCLMemRef<cl::Buffer>(), ostep,
                                 width,
                                 (MI_S32)(cl_global_size.get()[1]),
                                 (MI_S32)(cl_global_size.get()[0]),
                                 static_cast<MI_U8>(SwapUv(m_type)),
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &cl_event);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst[0]->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    ret = m_cl_dst[0].Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

Status CvtColorCL::CvtY4222RgbCLImpl()
{
    if (m_cl_src[0].GetSizes().m_width & 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "width must be even");
        return Status::ERROR;
    }

    if (m_cl_src[0].GetSizes() * Sizes3(1, 1, 3) != m_cl_dst[0].GetSizes() * Sizes3(1, 1, 2) ||
        m_cl_src[0].GetSizes().m_channel != 2)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the sizes of src and dst do not match");
        return Status::ERROR;
    }

    MI_S32  height = m_cl_src[0].GetSizes().m_height;
    MI_S32  width  = m_cl_src[0].GetSizes().m_width;
    MI_S32  istep  = m_cl_src[0].GetStrides().m_width / ElemTypeSize(m_cl_src[0].GetElemType());
    MI_S32  ostep  = m_cl_dst[0].GetStrides().m_width / ElemTypeSize(m_cl_dst[0].GetElemType());
    MI_BOOL swapy  = (CvtColorType::YUV2RGB_Y422 == m_type) || (CvtColorType::YUV2RGB_Y422_601 == m_type);

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size = {cl::NDRange((width + 7) >> 3, height)};

    // 2. opencl run
    cl::Event cl_event;
    cl_int    cl_ret = CL_SUCCESS;
    Status    ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_U8, MI_U8>(
                                 m_cl_src[0].GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst[0].GetCLMemRef<cl::Buffer>(), ostep,
                                 width,
                                 (MI_S32)(cl_global_size.get()[1]),
                                 (MI_S32)(cl_global_size.get()[0]),
                                 static_cast<MI_U8>(SwapUv(m_type)), static_cast<MI_U8>(swapy),
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &cl_event);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst[0]->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    ret = m_cl_dst[0].Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

Status CvtColorCL::CvtY4442RgbCLImpl()
{
    if (m_cl_src[0].GetSizes() != m_cl_src[1].GetSizes() ||
        m_cl_src[0].GetSizes() != m_cl_src[2].GetSizes() ||
        m_cl_src[0].GetSizes() * Sizes3(1, 1, 3) != m_cl_dst[0].GetSizes() ||
        m_cl_src[0].GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the sizes of src and dst do not match");
        return Status::ERROR;
    }

    MI_S32 height = m_cl_src[0].GetSizes().m_height;
    MI_S32 width  = m_cl_src[0].GetSizes().m_width;
    MI_S32 istep0 = m_cl_src[0].GetStrides().m_width / ElemTypeSize(m_cl_src[0].GetElemType());
    MI_S32 istep1 = m_cl_src[1].GetStrides().m_width / ElemTypeSize(m_cl_src[1].GetElemType());
    MI_S32 istep2 = m_cl_src[2].GetStrides().m_width / ElemTypeSize(m_cl_src[2].GetElemType());
    MI_S32 ostep  = m_cl_dst[0].GetStrides().m_width / ElemTypeSize(m_cl_dst[0].GetElemType());

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size = {cl::NDRange((width + 7) >> 3, height)};

    // 2. opencl run
    cl::Event cl_event;
    cl_int    cl_ret = CL_SUCCESS;
    Status    ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32>(
                                 m_cl_src[0].GetCLMemRef<cl::Buffer>(), istep0,
                                 m_cl_src[1].GetCLMemRef<cl::Buffer>(), istep1,
                                 m_cl_src[2].GetCLMemRef<cl::Buffer>(), istep2,
                                 m_cl_dst[0].GetCLMemRef<cl::Buffer>(), ostep,
                                 width,
                                 (MI_S32)(cl_global_size.get()[1]),
                                 (MI_S32)(cl_global_size.get()[0]),
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &cl_event);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst[0]->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    ret = m_cl_dst[0].Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

Status CvtColorCL::CvtRgb2NvCLImpl()
{
    if ((m_cl_src[0].GetSizes().m_width & 1) || (m_cl_src[0].GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "width and height must be even");
        return Status::ERROR;
    }

    if (m_cl_dst[0].GetSizes() * Sizes3(1, 1, 2) != m_cl_dst[1].GetSizes() * Sizes3(2, 2, 1) ||
        m_cl_dst[0].GetSizes() * Sizes3(1, 1, 3) != m_cl_src[0].GetSizes() ||
        m_cl_dst[0].GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the sizes of src and dst do not match");
        return Status::ERROR;
    }

    MI_S32 height   = m_cl_src[0].GetSizes().m_height;
    MI_S32 width    = m_cl_src[0].GetSizes().m_width;
    MI_S32 istep    = m_cl_src[0].GetRowPitch() / ElemTypeSize(m_cl_src[0].GetElemType());
    MI_S32 ostep0   = m_cl_dst[0].GetRowPitch() / ElemTypeSize(m_cl_dst[0].GetElemType());
    MI_S32 ostep1   = m_cl_dst[1].GetRowPitch() / ElemTypeSize(m_cl_dst[1].GetElemType());
    MI_S32 uv_const = (CvtColorType::RGB2YUV_NV12_P010 == m_type || CvtColorType::RGB2YUV_NV21_P010 == m_type)
                      ? (512 * (1 << CVTCOLOR_COEF_BITS)) : (128 * (1 << CVTCOLOR_COEF_BITS));

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size = cl::NDRange((width + 7) >> 3, height >> 1);

    // 2. opencl run
    cl::Event cl_event;
    cl_int    cl_ret = CL_SUCCESS;
    Status    ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32>(
                                 m_cl_src[0].GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst[0].GetCLMemRef<cl::Buffer>(), ostep0,
                                 m_cl_dst[1].GetCLMemRef<cl::Buffer>(), ostep1,
                                 uv_const, width,
                                 (MI_S32)(cl_global_size.get()[1]),
                                 (MI_S32)(cl_global_size.get()[0]),
                                 (MI_S32)(SwapUv(m_type)),
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &cl_event);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run m_cl_kernels[0] failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst[0]->GetArrayType() != ArrayType::CL_MEMORY) || (m_dst[1]->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait m_cl_kernels[0] failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    for (MI_U32 i = 0; i < m_cl_dst.size(); i++)
    {
        ret = m_cl_dst[i].Sync(CLMemSyncType::READ);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
            goto EXIT;
        }
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

Status CvtColorCL::CvtRgb2Y420CLImpl()
{
    if ((m_cl_src[0].GetSizes().m_width & 1) || (m_cl_src[0].GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "width and height must be even");
        return Status::ERROR;
    }

    if (m_cl_dst[0].GetSizes() != m_cl_dst[1].GetSizes() * Sizes3(2, 2, 1) ||
        m_cl_dst[0].GetSizes() != m_cl_dst[2].GetSizes() * Sizes3(2, 2, 1) ||
        m_cl_dst[0].GetSizes() * Sizes3(1, 1, 3) != m_cl_src[0].GetSizes() ||
        m_cl_dst[0].GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the sizes of src and dst do not match");
        return Status::ERROR;
    }

    MI_S32 height   = m_cl_src[0].GetSizes().m_height;
    MI_S32 width    = m_cl_src[0].GetSizes().m_width;
    MI_S32 istep    = m_cl_src[0].GetRowPitch() / ElemTypeSize(m_cl_src[0].GetElemType());
    MI_S32 ostep0   = m_cl_dst[0].GetRowPitch() / ElemTypeSize(m_cl_dst[0].GetElemType());
    MI_S32 ostep1   = m_cl_dst[1].GetRowPitch() / ElemTypeSize(m_cl_dst[1].GetElemType());
    MI_S32 ostep2   = m_cl_dst[2].GetRowPitch() / ElemTypeSize(m_cl_dst[2].GetElemType());
    MI_S32 uv_const = 128 * (1 << CVTCOLOR_COEF_BITS);

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size = cl::NDRange((width + 7) >> 3, height >> 1);

    // 2. opencl run
    cl::Event cl_event;
    cl_int    cl_ret = CL_SUCCESS;
    Status    ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32>(
                                 m_cl_src[0].GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst[0].GetCLMemRef<cl::Buffer>(), ostep0,
                                 m_cl_dst[1].GetCLMemRef<cl::Buffer>(), ostep1,
                                 m_cl_dst[2].GetCLMemRef<cl::Buffer>(), ostep2,
                                 uv_const, width,
                                 (MI_S32)(cl_global_size.get()[1]),
                                 (MI_S32)(cl_global_size.get()[0]),
                                 (MI_S32)(SwapUv(m_type)),
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &cl_event);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst[0]->GetArrayType() != ArrayType::CL_MEMORY) || (m_dst[1]->GetArrayType() != ArrayType::CL_MEMORY) || (m_dst[2]->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    for (MI_U32 i = 0; i < m_cl_dst.size(); i++)
    {
        ret = m_cl_dst[i].Sync(CLMemSyncType::READ);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
            goto EXIT;
        }
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

Status CvtColorCL::CvtRgb2Y444CLImpl()
{
    if (m_cl_dst[0].GetSizes() != m_cl_dst[1].GetSizes() ||
        m_cl_dst[0].GetSizes() != m_cl_dst[2].GetSizes() ||
        m_cl_dst[0].GetSizes() * Sizes3(1, 1, 3) != m_cl_src[0].GetSizes() ||
        m_cl_dst[0].GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the sizes of src and dst do not match");
        return Status::ERROR;
    }

    MI_S32 height   = m_cl_src[0].GetSizes().m_height;
    MI_S32 width    = m_cl_src[0].GetSizes().m_width;
    MI_S32 istep    = m_cl_src[0].GetRowPitch() / ElemTypeSize(m_cl_src[0].GetElemType());
    MI_S32 ostep0   = m_cl_dst[0].GetRowPitch() / ElemTypeSize(m_cl_dst[0].GetElemType());
    MI_S32 ostep1   = m_cl_dst[1].GetRowPitch() / ElemTypeSize(m_cl_dst[1].GetElemType());
    MI_S32 ostep2   = m_cl_dst[2].GetRowPitch() / ElemTypeSize(m_cl_dst[2].GetElemType());
    MI_S32 uv_const = 128 * (1 << CVTCOLOR_COEF_BITS);

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size = cl::NDRange(((width + 7) >> 3), height >> 1);

    // 2. opencl run
    cl::Event cl_event;
    cl_int    cl_ret = CL_SUCCESS;
    Status    ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32>(
                                 m_cl_src[0].GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst[0].GetCLMemRef<cl::Buffer>(), ostep0,
                                 m_cl_dst[1].GetCLMemRef<cl::Buffer>(), ostep1,
                                 m_cl_dst[2].GetCLMemRef<cl::Buffer>(), ostep2,
                                 uv_const, width,
                                 (MI_S32)(cl_global_size.get()[1]),
                                 (MI_S32)(cl_global_size.get()[0]),
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &cl_event);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst[0]->GetArrayType() != ArrayType::CL_MEMORY) ||
        (m_dst[1]->GetArrayType() != ArrayType::CL_MEMORY) || (m_dst[2]->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    for (MI_U32 i = 0; i < m_cl_dst.size(); i++)
    {
        ret = m_cl_dst[i].Sync(CLMemSyncType::READ);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
            goto EXIT;
        }
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

Status CvtColorCL::CvtBayer2BgrCLImpl()
{
    if ((m_cl_src[0].GetSizes().m_width & 1) || (m_cl_src[0].GetSizes().m_height & 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "width and height must be even");
        return Status::ERROR;
    }

    if (m_cl_src[0].GetSizes() * Sizes3(1, 1, 3) != m_cl_dst[0].GetSizes() ||
        m_cl_src[0].GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the sizes of src and dst do not match");
        return Status::ERROR;
    }

    MI_S32 height = m_cl_src[0].GetSizes().m_height;
    MI_S32 width  = m_cl_src[0].GetSizes().m_width;
    MI_S32 istep  = m_cl_src[0].GetRowPitch() / ElemTypeSize(m_cl_src[0].GetElemType());
    MI_S32 ostep  = m_cl_dst[0].GetRowPitch() / ElemTypeSize(m_cl_dst[0].GetElemType());

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size[2];
    MI_U8       back_flag     = 0;
    MI_S32      remain_offset = width + ((height - 2) << 1);

    MI_S32 vec_width   = width - 2;
    MI_S32 elem_counts = 4;
    back_flag          = ((vec_width % elem_counts) != 0);

    cl_global_size[0] = cl::NDRange(width >> 2, (height - 2) >> 1); // main global size
    cl_global_size[1] = cl::NDRange((vec_width + height) << 1, 1);  // remain global size

    // 2. opencl run
    cl::Event cl_event[2];
    cl_int    cl_ret = CL_SUCCESS;
    Status    ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_U8, MI_U8, MI_U8>(
                                 m_cl_src[0].GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst[0].GetCLMemRef<cl::Buffer>(), ostep,
                                 width,
                                 (MI_S32)(cl_global_size[0].get()[1]),
                                 (MI_S32)(cl_global_size[0].get()[0]),
                                 back_flag,
                                 static_cast<MI_U8>(SwapBlue(m_type)), static_cast<MI_U8>(SwapGreen(m_type)),
                                 cl_global_size[0], cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size[0]),
                                 &cl_event[0]);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret = m_cl_kernels[1].Run<cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32>(
                                 m_cl_dst[0].GetCLMemRef<cl::Buffer>(), ostep,
                                 height, width,
                                 (MI_S32)(cl_global_size[1].get()[1]),
                                 (MI_S32)(cl_global_size[1].get()[0]),
                                 remain_offset,
                                 cl_global_size[1], cl_rt->GetCLDefaultLocalSize(m_cl_kernels[1].GetMaxGroupSize(), cl_global_size[1]),
                                 &(cl_event[1]), {cl_event[0]});
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel remain failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst[0]->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event[1].wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event[0]) +
                                       GetCLProfilingInfo(m_cl_kernels[1].GetKernelName(), cl_event[1]);
        }
    }

    ret = m_cl_dst[0].Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

} // namespace aura