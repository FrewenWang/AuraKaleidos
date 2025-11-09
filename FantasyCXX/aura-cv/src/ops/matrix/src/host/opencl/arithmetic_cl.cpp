#include "arithmetic_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static std::string GetCLBuildOptions(Context *ctx, ArithmOpType op, const ElemType src_type, const ElemType dst_type)
{
    std::vector<std::string> tbl;
    tbl.reserve(5);
    tbl.push_back("   U8,    S8,   U16,   S16,   U32,   S32,   F32,   F16");// elem   type
    tbl.push_back("St");
    tbl.push_back("uchar, char, ushort, short,  uint,   int, float, half");// source type
    tbl.push_back("InterType");

    if ((ElemType::F32 == dst_type) || (ElemType::F16 == dst_type))
    {
        tbl.push_back("float, float, float, float, float, float, float, float");// inter type
    }
    else if (ArithmOpType::SUB == op)
    {
        tbl.push_back("short, short,   int,   int,   int,   int, float, float");// inter type
    }
    else
    {
        tbl.push_back("ushort, short, uint,   int,  uint,   int, float, float");// inter type
    }

    CLBuildOptions cl_build_opt(ctx, tbl);
    DT_S32 elem_counts = 16 / ElemTypeSize(src_type);
    cl_build_opt.AddOption("Dt", CLTypeString(dst_type));
    cl_build_opt.AddOption("ARITHM_TYPE", ArithmeOpTypeToString(op));
    cl_build_opt.AddOption("ELEM_COUNTS", std::to_string(elem_counts));

    return cl_build_opt.ToString(src_type);
}

static DT_VOID GetCLGlobalSize(DT_S32 height, DT_S32 width, DT_S32 channel, DT_S32 elem_counts, cl::NDRange &cl_range)
{
    cl_range = cl::NDRange((width * channel + elem_counts - 1) / elem_counts, height);
}

ArithmeticCL::ArithmeticCL(Context *ctx, const OpTarget &target) : ArithmeticImpl(ctx, target), m_elem_counts(0), m_profiling_string()
{}

Status ArithmeticCL::Initialize()
{
    if (ArithmeticImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ArithmeticImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 1. init cl_mem
    m_cl_src0 = CLMem::FromArray(m_ctx, *m_src0, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src0.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src0 is invalid");
        return Status::ERROR;
    }

    m_cl_src1 = CLMem::FromArray(m_ctx, *m_src1, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src1.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src1 is invalid");
        return Status::ERROR;
    }

    m_cl_dst = CLMem::FromArray(m_ctx, *m_dst, CLMemParam(CL_MEM_WRITE_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
        return Status::ERROR;
    }

    // 2. init kernel
    m_cl_kernels  = ArithmeticCL::GetCLKernels(m_ctx, m_src0->GetElemType(), m_dst->GetElemType(), m_op_type);
    m_elem_counts = 16 / ElemTypeSize(m_src0->GetElemType());

    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync start
    if (m_cl_src0.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src0 sync start failed");
        return Status::ERROR;
    }

    if (m_cl_src1.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src1 sync start failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ArithmeticCL::DeInitialize()
{
    m_cl_src0.Release();
    m_cl_src1.Release();
    m_cl_dst.Release();

    m_cl_kernels[0].DeInitialize();

    if (ArithmeticImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ArithmeticImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ArithmeticCL::Run()
{
    DT_S32 istep0  = m_src0->GetRowPitch() / ElemTypeSize(m_src0->GetElemType());
    DT_S32 istep1  = m_src1->GetRowPitch() / ElemTypeSize(m_src1->GetElemType());
    DT_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize( m_dst->GetElemType());
    DT_S32 height  = m_src0->GetSizes().m_height;
    DT_S32 width   = m_src0->GetSizes().m_width;
    DT_S32 channel = m_src0->GetSizes().m_channel;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size;
    GetCLGlobalSize(height, width, channel, m_elem_counts, cl_global_size);
    cl_int cl_ret = CL_SUCCESS;
    cl::Event cl_event;
    Status ret = Status::ERROR;

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32>(
                                 m_cl_src0.GetCLMemRef<cl::Buffer>(), istep0,
                                 m_cl_src1.GetCLMemRef<cl::Buffer>(), istep1,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 width * channel, height,
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &cl_event);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("cl_kernel run failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((DT_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (DT_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    ret = m_cl_dst.Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string ArithmeticCL::ToString() const
{
    return ArithmeticImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> ArithmeticCL::GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, ArithmOpType op_type)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, op_type, src_elem_type, dst_elem_type);

    std::string program_name = "aura_arithmetic";
    std::string kernel_name  = "Arithmetic";

    cl_kernels.push_back({ctx, program_name, kernel_name, "", build_opt});

    return cl_kernels;
}

} // namespace aura
