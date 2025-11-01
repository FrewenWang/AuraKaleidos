#ifndef AURA_RUNTIME_OPENCL_CL_PROGRAM_HPP__
#define AURA_RUNTIME_OPENCL_CL_PROGRAM_HPP__

#include "aura/runtime/opencl/cl_runtime.hpp"

#include <string>
#include <memory>

namespace aura
{

enum class CLProgramType
{
    INVALID        = 0,
    CACHED,
    PRECOMPILED,
    NEW,
};

enum class CLProgramBuildStatus
{
    NO_BUILD        = 0,
    BINARY_BUILD,
    SOURCE_BUILD,
};

inline std::string GetCLProgramTypeToString(CLProgramType cl_type)
{
    std::string cl_program_type = "INVALID";

    switch (cl_type)
    {
        case CLProgramType::CACHED:
        {
            cl_program_type = "CACHED";
            break;
        }
        case CLProgramType::PRECOMPILED:
        {
            cl_program_type = "PRECOMPILED";
            break;
        }
        case CLProgramType::NEW:
        {
            cl_program_type = "NEW";
            break;
        }
        default:
        {
            break;
        }
    }

    return cl_program_type;
}

class CLProgram
{
public:
    CLProgram(Context *ctx,
              std::shared_ptr<cl::Device> &cl_device,
              std::shared_ptr<cl::Context> &cl_context,
              const std::string &name,
              const std::string &source,
              const std::string &build_options,
              CLProgramType cl_type);

    CLProgram(Context *ctx,
              std::shared_ptr<cl::Device> &cl_device,
              std::shared_ptr<cl::Context> &cl_context,
              const std::string &name,
              std::vector<std::vector<MI_UCHAR> > binaries_vecs,
              const std::string &build_options,
              MI_U32 crc_val,
              CLProgramType cl_type);

    ~CLProgram();

    Status Build();

    AURA_DISABLE_COPY_AND_ASSIGN(CLProgram);

public:
    std::string                  m_name;
    std::string                  m_build_options;
    std::shared_ptr<cl::Program> m_cl_program;
    CLProgramType                m_cl_type;
    MI_U32                       m_crc_val;
    CLProgramBuildStatus         m_cl_build_status;

private:
    Context                      *m_ctx;
    std::shared_ptr<cl::Device>  m_cl_device;

};

} // namespace aura

#endif // AURA_RUNTIME_OPENCL_CL_PROGRAM_HPP__