#ifndef AURA_RUNTIME_OPENCL_CL_PROGRAM_CONTAINER_HPP__
#define AURA_RUNTIME_OPENCL_CL_PROGRAM_CONTAINER_HPP__

#include "aura/runtime/opencl/cl_runtime.hpp"
#include "cl_program.hpp"

#include <unordered_map>

namespace aura
{

#define AURA_CL_PROG_MAGIC      (0xA5F3C010)

enum class EncodeProgramsType
{
    INVALID            = 0,
    CACHE_APPEND,
    CACHE_ALL,
    PRECOMMPILED_ALL,
};

class CLProgramContainer
{
public:
    CLProgramContainer(Context *ctx,
                       std::shared_ptr<cl::Device> &cl_device,
                       std::shared_ptr<cl::Context> &cl_context,
                       const std::string &cl_driver_version,
                       const std::string &aura_version,
                       const std::string &external_version,
                       std::shared_ptr<CLEngineConfig> &cl_conf);

    ~CLProgramContainer();

    std::shared_ptr<cl::Program> GetCLProgram(const std::string &name,
                                              const std::string &source = std::string(),
                                              const std::string &build_options = std::string());

    Status CreatePrecompiledCLProgram(const std::string &file_path, const std::string &prefix);

private:
    Context                         *m_ctx;
    std::shared_ptr<cl::Device>     m_cl_device;
    std::shared_ptr<cl::Context>    m_cl_context;
    std::shared_ptr<CLEngineConfig> m_cl_conf;
    std::string                     m_cache_bin_fname;
    std::string                     m_cl_driver_version;
    std::string                     m_aura_version;
    std::string                     m_external_version;

    std::unordered_map<DT_U32, std::shared_ptr<CLProgram>> m_cl_programs_list;
    DT_BOOL    m_cache_header_valid;
    DT_BOOL    m_is_update_cache;
    std::mutex m_container_mutex;

private:
    std::shared_ptr<cl::Program> BuildProgramFromSource(const std::string &name,
                                                        const std::string &source,
                                                        const CLProgramType type,
                                                        DT_U32 hash_value,
                                                        DT_U32 crc_value,
                                                        const std::string &build_options = std::string());
    Status Initialize();

    Status LoadBinaryCLProgram(const std::string &fname, CLProgramType type);
    Status CreateCacheCLProgram(const std::string &fname);

    Status DecodeBinaryCLProgram(const std::string &str, CLProgramType type = CLProgramType::CACHED);

    Status EncodeBinaryCLProgram(std::string &output, EncodeProgramsType encode_type);

    Status WriteBinaryFile(const std::string &fname, const std::string &str, DT_BOOL is_append_flag);
    Status WriteHppFile(const std::string &fname, const std::string &str);

    DT_U32 GetHash(const std::string &str);
};

} // namespace aura

#endif // AURA_RUNTIME_OPENCL_CL_PROGRAM_CONTAINER_HPP__