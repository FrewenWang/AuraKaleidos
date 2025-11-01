#include "aura/runtime/opencl/cl_kernel.hpp"
#include "aura/runtime/opencl/cl_engine.hpp"
#include "cl_runtime_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

CLKernel::CLKernel() : m_ctx(MI_NULL), m_max_group_size(0), m_kernel_name()
{}

CLKernel::CLKernel(Context *ctx,
                   const std::string &program_name,
                   const std::string &kernel_name,
                   const std::string &source,
                   const std::string &build_options)
                   : m_ctx(ctx), m_cl_queue(), m_cl_kernel(), 
                     m_max_group_size(0), m_kernel_name(kernel_name)
{
    if (m_ctx)
    {
        CLEngine *cl_engine = ctx->GetCLEngine();
        if (cl_engine)
        {
            std::shared_ptr<CLRuntime> cl_rt = cl_engine->GetCLRuntime();
            if (cl_rt)
            {
                std::shared_ptr<cl::Device> cl_device = cl_rt->GetDevice();
                std::shared_ptr<cl::Program> m_cl_program;
                if (source.empty())
                {
                    std::string program_string = GetClProgramString(program_name);
                    if (!program_string.empty())
                    {
                        m_cl_program = cl_rt->GetCLProgram(program_name, program_string, build_options);
                    }
                    else
                    {
                        std::string info = "find program_name failed, error: cl_kernel name: " + kernel_name + " program name: " + program_name;
                        AURA_ADD_ERROR_STRING(ctx, info.c_str());
                    }
                }
                else
                {
                    m_cl_program = cl_rt->GetCLProgram(program_name, source, build_options);
                }

                if (m_cl_program)
                {
                    cl_int cl_err = CL_SUCCESS;

                    m_cl_kernel = std::make_shared<cl::Kernel>(*m_cl_program, kernel_name.c_str(), &cl_err);
                    if (cl_err != CL_SUCCESS)
                    {
                        m_max_group_size = 0;
                        std::string info = "create cl_kernel failed, error: " + GetCLErrorInfo(cl_err) + " cl_kernel name: " + kernel_name + " program name: " + program_name;
                        AURA_ADD_ERROR_STRING(ctx, info.c_str());
                    }
                    else
                    {
                        size_t max_group_size = 0;
                        m_cl_kernel->getWorkGroupInfo(*cl_device, CL_KERNEL_WORK_GROUP_SIZE, &max_group_size);
                        m_max_group_size = static_cast<MI_S32>(max_group_size);
                    }
                    m_cl_queue = cl_rt->GetCommandQueue();
                }
            }
        }
    }
}

std::shared_ptr<cl::Kernel> CLKernel::GetClKernel() const
{
    return m_cl_kernel;
}

MI_U32 CLKernel::GetMaxGroupSize() const
{
    return m_max_group_size;
}

AURA_VOID CLKernel::DeInitialize()
{
    m_ctx = MI_NULL;
    m_max_group_size = 0;
    m_cl_kernel.reset();
    m_cl_queue.reset();
}

CLKernel::~CLKernel()
{
    DeInitialize();
}

Status CLKernel::CheckKenrelWorkSize(cl::NDRange &cl_global, cl::NDRange &cl_local)
{
    Status ret = Status::OK;

    CLEngine *cl_engine = m_ctx->GetCLEngine();

    if (cl_engine)
    {
        std::shared_ptr<cl::Device> cl_device = cl_engine->GetCLRuntime()->GetDevice();

        if (m_cl_kernel && cl_device && m_ctx)
        {
            size_t *lws = cl_local.get();
            MI_S32 cur_local_size = 0;

            if (1 == cl_global.dimensions())
            {
                cur_local_size = lws[0];
            }
            else if (2 == cl_global.dimensions())
            {
                cur_local_size = lws[0] * lws[1];
            }
            else if (3 == cl_global.dimensions())
            {
                cur_local_size = lws[0] * lws[1] * lws[2];
            }

            if (cur_local_size < 0)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "cl_local size is less 0");
                return Status::ERROR;
            }

            if (cur_local_size > m_max_group_size)
            {
                std::string info = "cl_local size is bigger kenrel max group sizes, max_group_sizes is " + std::to_string(m_max_group_size);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            CLEngine *cl_engine = m_ctx->GetCLEngine();
            if (cl_engine)
            {
                std::shared_ptr<CLRuntime> cl_rt = cl_engine->GetCLRuntime();
                if (cl_rt && (cl_global.dimensions() == cl_local.dimensions()) && (cl_global.dimensions() > 0))
                {
                    if (MI_FALSE == cl_rt->IsNonUniformWorkgroupsSupported())
                    {
                        size_t *cl_gws = cl_global.get();
                        for (MI_S32 i = 0; i < static_cast<MI_S32>(cl_global.dimensions()); i++)
                        {
                            cl_gws[i] = (cl_gws[i] + lws[i] - 1) / lws[i] * lws[i];
                        }
                    }
                }
            }
        }
    }

    return ret;
}

MI_BOOL CLKernel::IsValid() const 
{
    return m_cl_kernel != MI_NULL;
}

std::string CLKernel::GetKernelName() const
{
    return m_kernel_name;
}

} // namespace aura