#include "aura/runtime/opencl/cl_runtime.hpp"
#include "cl_program.hpp"
#include "aura/runtime/logger.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <algorithm>

namespace aura
{

CLProgram::CLProgram(Context *ctx,
                     std::shared_ptr<cl::Device> &cl_device,
                     std::shared_ptr<cl::Context> &cl_context,
                     const std::string &name,
                     const std::string &source,
                     const std::string &build_options,
                     CLProgramType cl_type)
                     : m_name(name), m_build_options(build_options),
                       m_cl_program(), m_cl_type(cl_type),
                       m_crc_val(0),
                       m_cl_build_status(CLProgramBuildStatus::NO_BUILD),
                       m_ctx(ctx), m_cl_device(cl_device)
{
    if (!source.empty() && cl_context)
    {
        cl_int cl_err = CL_SUCCESS;
        cl::Program::Sources sources;
        sources.emplace_back(source);
        m_cl_program = std::make_shared<cl::Program>(*cl_context, sources, &cl_err);

        if (cl_err != CL_SUCCESS)
        {
            std::string info = "clCreateProgramWithSource failed, Error: " + GetCLErrorInfo(cl_err);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        }
    }
}

CLProgram::CLProgram(Context *ctx,
                     std::shared_ptr<cl::Device> &cl_device,
                     std::shared_ptr<cl::Context> &cl_context,
                     const std::string &name,
                     std::vector<std::vector<MI_UCHAR> > binaries_vecs,
                     const std::string &build_options,
                     MI_U32 crc_val,
                     CLProgramType cl_type)
                     : m_name(name), m_build_options(build_options),
                       m_cl_program(), m_cl_type(cl_type), m_crc_val(crc_val),
                       m_cl_build_status(CLProgramBuildStatus::NO_BUILD),
                       m_ctx(ctx), m_cl_device(cl_device)
{
    if (cl_device && cl_context && binaries_vecs.size() > 0)
    {
        cl::Program cl_program(*cl_context, {*cl_device}, binaries_vecs);
        m_cl_program = std::make_shared<cl::Program>(cl_program);
    }
}

Status CLProgram::Build()
{
    Status ret = Status::ERROR;

    if (m_cl_program && m_cl_device)
    {
        cl_int cl_err = m_cl_program->build(*m_cl_device, m_build_options.c_str());

        if (cl_err != CL_SUCCESS)
        {
            std::string build_info = "build kernel failed:" + GetCLErrorInfo(cl_err) + "  build info: " +
                                      m_cl_program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*m_cl_device);
            AURA_ADD_ERROR_STRING(m_ctx, build_info.c_str());
        }
        else
        {
            ret = Status::OK;
        }
    }

    return ret;
}

CLProgram::~CLProgram()
{
    m_build_options   = std::string();
    m_cl_type         = CLProgramType::INVALID;
    m_crc_val         = 0;
    m_cl_build_status = CLProgramBuildStatus::NO_BUILD;
}

} // namespace aura