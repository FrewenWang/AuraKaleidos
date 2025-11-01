#include "cl_program_container.hpp"
#include "aura/runtime/logger.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <algorithm>

namespace aura
{

template<typename Tp> 
AURA_INLINE AURA_VOID Serialize(std::stringstream &ss, const Tp &val)
{
    ss.write((MI_CHAR *)(&val), sizeof(Tp));
}

template<> 
AURA_NO_STATIC_INLINE AURA_VOID Serialize<std::string>(std::stringstream &ss, const std::string &val)
{
    Serialize<MI_U32>(ss, val.size());
    ss.write(val.c_str(), val.size());
}

template<>
AURA_NO_STATIC_INLINE AURA_VOID Serialize<std::vector<std::vector<MI_UCHAR>>>(std::stringstream &ss, const std::vector<std::vector<MI_UCHAR>> &val)
{
    MI_U32 nums = val.size();
    Serialize<MI_U32>(ss, nums);

    for (MI_U32 i = 0; i < nums; ++i)
    {
        Serialize<std::string>(ss, std::string(val[i].begin(), val[i].end()));
    }
}

template<typename Tp> 
AURA_INLINE Tp Deserialize(std::stringstream &str)
{
    Tp num = 0;
    str.read((MI_CHAR *)(&num), sizeof(Tp));

    return num;
}

template<> 
AURA_NO_STATIC_INLINE std::string Deserialize<std::string>(std::stringstream &str)
{
    MI_U32 num = Deserialize<MI_U32>(str);

    std::string str_sub;
    str_sub.resize(num);

    str.read((MI_CHAR *)(str_sub.c_str()), num);

    return str_sub;
}

template<> 
AURA_NO_STATIC_INLINE std::vector<std::vector<MI_UCHAR>> Deserialize<std::vector<std::vector<MI_UCHAR>>>(std::stringstream &str)
{
    MI_U32 nums = Deserialize<MI_U32>(str);

    std::vector<std::vector<MI_UCHAR> > vecs;
    vecs.reserve(nums);

    for (MI_U32 i = 0; i < nums; ++i)
    {
        std::string s = Deserialize<std::string>(str);
        vecs.push_back(std::vector<MI_UCHAR>(s.begin(), s.end()));
    }

    return vecs;
}

CLProgramContainer::CLProgramContainer(Context *ctx,
                                       std::shared_ptr<cl::Device> &cl_device,
                                       std::shared_ptr<cl::Context> &cl_context,
                                       const std::string &cl_driver_version,
                                       const std::string &aura_version,
                                       const std::string &external_version,
                                       std::shared_ptr<CLEngineConfig> &cl_conf)
                                       : m_ctx(ctx), m_cl_device(cl_device), m_cl_context(cl_context),
                                         m_cl_conf(cl_conf), m_cache_bin_fname(),
                                         m_cl_driver_version(cl_driver_version),
                                         m_aura_version(aura_version),
                                         m_external_version(external_version),
                                         m_cl_programs_list(),
                                         m_cache_header_valid(MI_FALSE),
                                         m_is_update_cache(MI_FALSE)
{
    m_cache_bin_fname = m_cl_conf->m_cache_path;

    Initialize();
}

Status CLProgramContainer::Initialize()
{
    Status ret = Status::ERROR;
    Status ret_precompiled = Status::ERROR;

    if (m_cl_conf)
    {
        if (CLPrecompiledType::PATH == m_cl_conf->m_cl_precompiled_type)
        {
            if (!m_cl_conf->m_precompiled_sources.empty())
            {
                ret_precompiled = LoadBinaryCLProgram(m_cl_conf->m_precompiled_sources, CLProgramType::PRECOMPILED);
                if (ret_precompiled != Status::OK)
                {
                    AURA_LOGD(m_ctx, AURA_TAG, "precompiled sorces file %s load failed\n", m_cl_conf->m_precompiled_sources.c_str());
                }
            }
        }
        else if (CLPrecompiledType::STRING == m_cl_conf->m_cl_precompiled_type)
        {
            if (!m_cl_conf->m_precompiled_sources.empty())
            {
                ret_precompiled = DecodeBinaryCLProgram(m_cl_conf->m_precompiled_sources, CLProgramType::PRECOMPILED);
                if (ret_precompiled != Status::OK)
                {
                    AURA_LOGD(m_ctx, AURA_TAG, "precompiled sorces string DecodeBinaryCLProgram failed\n");
                }
            }
        }

        if ((!m_cache_bin_fname.empty()) && (!m_cl_conf->m_cache_prefix.empty()))
        {
            // remove last "/" or "\\"
            MI_S32 last_pos = m_cache_bin_fname.size() - 1;
            const MI_CHAR last_char = m_cache_bin_fname[last_pos];
            last_pos = (last_char == '/' || last_char == '\\') ? (last_pos) : (last_pos + 1);

            m_cache_bin_fname = m_cache_bin_fname.substr(0, last_pos) + "/" + m_cl_conf->m_cache_prefix + std::string(".cache");
            LoadBinaryCLProgram(m_cache_bin_fname, CLProgramType::CACHED);
        }
        else
        {
            AURA_LOGD(m_ctx, AURA_TAG, "cach_bin_path or cache_bin_prefix is invalid\n");
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CLEngineConfig is invalid");
    }

    return ret;
}

CLProgramContainer::~CLProgramContainer()
{
    if (!m_cache_bin_fname.empty())
    {
        if (CreateCacheCLProgram(m_cache_bin_fname) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CL program binary file update failed");
        }
    }
    else
    {
        AURA_LOGD(m_ctx, AURA_TAG, "cache bin path is invalid\n", MI_TRUE);
    }
}

std::shared_ptr<cl::Program> CLProgramContainer::GetCLProgram(const std::string &name,
                                                              const std::string &source,
                                                              const std::string &build_options)
{
    std::string build_options_str = "-cl-mad-enable -Werror -cl-fast-relaxed-math " + build_options;

#if (CL_HPP_TARGET_OPENCL_VERSION == 120)
    if (build_options_str.find("-cl-std=CL1.2") == build_options_str.npos)
    {
        build_options_str = build_options_str + std::string(" -cl-std=CL1.2");
    }
#elif ((CL_HPP_TARGET_OPENCL_VERSION >= 200) && (CL_HPP_TARGET_OPENCL_VERSION < 300))
    if (build_options_str.find("-cl-std=CL2.0") == build_options_str.npos)
    {
        build_options_str = build_options_str + std::string(" -cl-std=CL2.0");
    }
#elif (CL_HPP_TARGET_OPENCL_VERSION >= 300)
    if (build_options_str.find("-cl-std=CL3.0") == build_options_str.npos)
    {
        build_options_str = build_options_str + std::string(" -cl-std=CL3.0");
    }
#endif

    Status ret = Status::OK;
    std::shared_ptr<cl::Program> cl_program;

    std::string program_key_str = name + build_options_str;
    MI_U32 hash_value = GetHash(program_key_str);
    MI_U32 crc_val = GetHash(source);

    std::lock_guard<std::mutex> guard(m_container_mutex);

    auto it = m_cl_programs_list.find(hash_value);
    if (it != m_cl_programs_list.end())
    {
        if (it->second->m_cl_build_status != CLProgramBuildStatus::NO_BUILD)
        {
            AURA_LOGD(m_ctx, AURA_TAG, "program name: %s binary build\n", name.c_str());
            return it->second->m_cl_program;
        }
        else
        {
            if (crc_val == it->second->m_crc_val)
            {
                ret = it->second->Build();
                if (ret != Status::OK)
                {
                    cl_program = BuildProgramFromSource(name, source, it->second->m_cl_type, hash_value, crc_val, build_options_str);
                    AURA_LOGD(m_ctx, AURA_TAG, "program name: %s source build\n", name.c_str());
                }
                else
                {
                    AURA_LOGD(m_ctx, AURA_TAG, "program name: %s binary build\n", name.c_str());
                    it->second->m_cl_build_status = CLProgramBuildStatus::BINARY_BUILD;
                    cl_program = it->second->m_cl_program;
                }
            }
            else
            {
                cl_program = BuildProgramFromSource(name, source, it->second->m_cl_type, hash_value, crc_val, build_options_str);
                AURA_LOGD(m_ctx, AURA_TAG, "program name: %s source build\n", name.c_str());
            }
        }
    }
    else
    {
        cl_program = BuildProgramFromSource(name, source, CLProgramType::NEW, hash_value, crc_val, build_options_str);
        AURA_LOGD(m_ctx, AURA_TAG, "program name: %s source build\n", name.c_str());
    }

    return cl_program;
}

std::shared_ptr<cl::Program> CLProgramContainer::BuildProgramFromSource(const std::string &name,
                                                                        const std::string &source,
                                                                        const CLProgramType cl_type,
                                                                        MI_U32 hash_value,
                                                                        MI_U32 crc_value,
                                                                        const std::string &build_options)
{
    Status ret = Status::OK;

    std::shared_ptr<CLProgram> cl_program = std::make_shared<CLProgram>(m_ctx, m_cl_device, m_cl_context, name, source, build_options, cl_type);
    ret = cl_program->Build();

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "source build failed");
        return MI_NULL;
    }
    else
    {
        cl_program->m_crc_val = crc_value;
        cl_program->m_cl_build_status = CLProgramBuildStatus::SOURCE_BUILD;

        if (m_cl_programs_list.count(hash_value))
        {
            m_cl_programs_list[hash_value] = cl_program;
        }
        else
        {
            m_cl_programs_list.insert(std::make_pair(hash_value, cl_program));
        }

        if ((CLProgramType::CACHED == cl_type) || (CLProgramType::PRECOMPILED == cl_type))
        {
            m_cache_header_valid = MI_FALSE;
        }

        m_is_update_cache = MI_TRUE;

        return cl_program->m_cl_program;
    }

    return MI_NULL;
}

Status CLProgramContainer::LoadBinaryCLProgram(const std::string &fname, CLProgramType cl_type)
{
    Status ret = Status::ERROR;

    if (fname.empty())
    {
        return ret;
    }

    // load cache bin
    FILE *fp = fopen(fname.c_str(), "rb");
    if (MI_NULL == fp)
    {
        return ret;
    }

    fseek(fp, 0, SEEK_END);
    MI_S32 file_length = ftell(fp);

    fseek(fp, 0, SEEK_SET);

    std::string cl_program_str;
    cl_program_str.resize(file_length);

    size_t bytes = fread((AURA_VOID*)(cl_program_str.data()), 1, file_length, fp);
    if (static_cast<MI_S32>(bytes) != file_length)
    {
        goto EXIT;
    }

    // decode cache bin
    ret = DecodeBinaryCLProgram(cl_program_str, cl_type);
    if (ret != Status::OK)
    {
        AURA_LOGD(m_ctx, AURA_TAG, "DecodeBinaryCLProgram failed, error : %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
        goto EXIT;
    }

EXIT:
    if (fp)
    {
        fclose(fp);
    }

    return ret;
}

Status CLProgramContainer::CreateCacheCLProgram(const std::string &fname)
{
    Status ret = Status::ERROR;

    if (!fname.empty())
    {
        std::string str;

        if (m_is_update_cache)
        {
            MI_BOOL is_append_flag = m_cache_header_valid ? MI_TRUE : MI_FALSE;

            EncodeProgramsType encode_type = m_cache_header_valid ? EncodeProgramsType::CACHE_APPEND : EncodeProgramsType::CACHE_ALL;
            ret = EncodeBinaryCLProgram(str, encode_type);

            if (Status::OK == ret)
            {
                ret = WriteBinaryFile(fname, str, is_append_flag);
            }
            else
            {
                AURA_ADD_ERROR_STRING(m_ctx, "EncodeBinaryCLProgram failed");
            }
        }
    }
    return ret;
}

Status CLProgramContainer::WriteBinaryFile(const std::string &fname, const std::string &str, MI_BOOL is_append_flag)
{
    Status ret = Status::ERROR;

    if (!fname.empty() && !str.empty())
    {
        std::string mode_flag = is_append_flag ? "a+" : "wb";
        FILE *fp = fopen(fname.c_str(), mode_flag.c_str());
        if (MI_NULL == fp)
        {
            AURA_LOGE(m_ctx, AURA_TAG, "open file %s failed\n", fname.c_str());
            return ret;
        }

        size_t bytes = fwrite(str.data(), 1, str.size(), fp);
        fclose(fp);

        if (bytes != str.size())
        {
            AURA_LOGE(m_ctx, AURA_TAG, "fwrite size(%ld, %ld) not matched\n", bytes, str.size());
            return ret;
        }

        ret = Status::OK;
    }

    return ret;
}

Status CLProgramContainer::WriteHppFile(const std::string &fname, const std::string &str)
{
    Status ret = Status::ERROR;

    if (!fname.empty() && !str.empty())
    {
        std::string str_start, str_end;
        str_start = "#ifndef AURA_CL_PRECOMPILED_BIN_H__\n";
        str_start += "#define AURA_CL_PRECOMPILED_BIN_H__\n\n";

        str_start += "// This is a generated file. DO NOT EDIT!\n\n";

        str_start += "static char g_cl_precompiled_str[] = \n";

        str_start += "{\n";

        str_end = "\n};\n";

        std::ofstream ofs(fname, std::ios::out | std::ios::trunc);
        if (ofs.is_open())
        {
            ofs << str_start;
            ofs << "    ";
            for (MI_U32 i = 0; i < (MI_U32)(str.size()); i++)
            {
                ofs << "0x"<<std::setbase(16) << std::setw(2) << std::setfill('0') << static_cast<MI_U16>(str[i]) << ", ";
                if ((i + 1) % 16 == 0)
                {
                    ofs << "\n";
                    ofs << "    ";
                }
            }

            ofs << str_end;
            ofs << "#endif";

            ofs.close();
            ret = Status::OK;
        }
        else
        {
            std::string info = "precompile filename: " + fname + " open failed";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            ret = Status::ERROR;
        }
    }

    return ret;
}

Status CLProgramContainer::CreatePrecompiledCLProgram(const std::string &file_path, const std::string &prefix)
{
    Status ret = Status::ERROR;

    if (!file_path.empty())
    {
        // remove last "/" or "\\"
        MI_S32 last_pos = file_path.size() - 1;
        const MI_CHAR last_char = file_path[last_pos];
        last_pos = (last_char == '/' || last_char == '\\') ? (last_pos) : (last_pos + 1);

        std::string bin_file = file_path.substr(0, last_pos) + "/" + prefix + std::string("_precompiled.bin");

        std::string str;
        ret = EncodeBinaryCLProgram(str, EncodeProgramsType::PRECOMMPILED_ALL);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "EncodeBinaryCLProgram failed");
            return ret;
        }

        ret = WriteBinaryFile(bin_file, str, MI_FALSE);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "WriteBinaryFile failed");
            return ret;
        }

        std::string hpp_file = file_path.substr(0, last_pos) + "/" + prefix + "_precompiled_cl_string.h";

        ret = WriteHppFile(hpp_file, str);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "WriteHppFile failed");
            return ret;
        }
    }

    return ret;
}

Status CLProgramContainer::DecodeBinaryCLProgram(const std::string &bin_str, CLProgramType cl_type)
{
    if (bin_str.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CL cache bin is empty");
        return Status::ERROR;
    }

    std::stringstream ss;
    ss.write(bin_str.c_str(), bin_str.size());

    // decode header
    MI_U32 magic_num = Deserialize<MI_U32>(ss);
    if (magic_num != AURA_CL_PROG_MAGIC)
    {
        MI_CHAR info[256];
        snprintf(info, sizeof(info), "program cl_type is : %s, magic_num(%x, %x) is not matched\n", 
                 GetCLProgramTypeToString(cl_type).c_str(), magic_num, AURA_CL_PROG_MAGIC);
        AURA_ADD_ERROR_STRING(m_ctx, info);
        return Status::ERROR;
    }

    std::string driver_version = Deserialize<std::string>(ss);
    if (driver_version != m_cl_driver_version)
    {
        std::string info = "program cl_type is :" + GetCLProgramTypeToString(cl_type) + ",file driver version:" + 
                            driver_version + ", aura driver version:" + m_cl_driver_version + ", is not matched";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return Status::ERROR;
    }

    //// 获取aura的版本号
    std::string aura_version = Deserialize<std::string>(ss);
    if (aura_version != m_aura_version)
    {
        std::string info = "program cl_type is :" + GetCLProgramTypeToString(cl_type) + ", file aura version:" +
                            aura_version + ", aura version:" + m_aura_version + ", is not matched";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return Status::ERROR;
    }

    std::string external_version = Deserialize<std::string>(ss);
    if (external_version != m_external_version)
    {
        std::string info = "program cl_type is :" + GetCLProgramTypeToString(cl_type) + ",file external version:" + 
                            external_version + ", aura external version:" + m_external_version + ", is not matched";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return Status::ERROR;
    }

    std::lock_guard<std::mutex> guard(m_container_mutex);
    MI_U32 hash_value = 0;
    // decode program
    while (1)
    {
        hash_value = Deserialize<MI_U32>(ss);
        if (!ss.good())
        {
            break;
        }

        // program name
        std::string name = Deserialize<std::string>(ss);
        if (!ss.good())
        {
            break;
        }

        // program build options
        std::string build_opt = Deserialize<std::string>(ss);
        if (!ss.good())
        {
            break;
        }

        // program crc
        MI_U32 crc_val = Deserialize<MI_U32>(ss);
        if (!ss.good())
        {
            break;
        }

        // program binaries
        std::vector<std::vector<MI_UCHAR> > binaries_vecs = Deserialize<std::vector<std::vector<MI_UCHAR>>>(ss);
        if (!ss.good())
        {
            break;
        }

        std::shared_ptr<CLProgram> cl_program = std::make_shared<CLProgram>(m_ctx, m_cl_device, m_cl_context, name, binaries_vecs, build_opt, crc_val, cl_type);

        // add to program set
        if (cl_program)
        {
            if (m_cl_programs_list.count(hash_value))
            {
                m_cl_programs_list[hash_value] = cl_program;
            }
            else
            {
                m_cl_programs_list.insert(std::make_pair(hash_value, cl_program));
            }
        }
    }

    if (CLProgramType::CACHED == cl_type)
    {
        m_cache_header_valid = MI_TRUE;
    }

    return Status::OK;
}

Status CLProgramContainer::EncodeBinaryCLProgram(std::string &output, EncodeProgramsType encode_type)
{
    Status ret = Status::OK;

    output.clear();

    std::lock_guard<std::mutex> guard(m_container_mutex);

    if (m_cl_programs_list.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CL progoram set is empty");
        return Status::ERROR;
    }

    std::stringstream stream;

    if ((EncodeProgramsType::CACHE_ALL == encode_type) ||
        (EncodeProgramsType::PRECOMMPILED_ALL == encode_type)) //write cache header infomation
    {
        // magic num
        Serialize<MI_U32>(stream, AURA_CL_PROG_MAGIC);

        // driver version
        Serialize<std::string>(stream, m_cl_driver_version);

        // aura version
        Serialize<std::string>(stream, m_aura_version);

        // external vesion
        Serialize<std::string>(stream, m_external_version);

        output += stream.str();
    }

    // program
    for (auto &bprog : m_cl_programs_list)
    {
        std::stringstream prog_stream;

        MI_BOOL ecode_flag = MI_FALSE;

        if (EncodeProgramsType::CACHE_ALL == encode_type)
        {
            ecode_flag = (CLProgramBuildStatus::SOURCE_BUILD == bprog.second->m_cl_build_status) || (CLProgramType::CACHED == bprog.second->m_cl_type);
        }
        else if (EncodeProgramsType::CACHE_APPEND == encode_type)
        {
            ecode_flag = (CLProgramBuildStatus::SOURCE_BUILD == bprog.second->m_cl_build_status);
        }
        else if (EncodeProgramsType::PRECOMMPILED_ALL == encode_type)
        {
            ecode_flag = MI_TRUE;
        }

        if (ecode_flag && bprog.second->m_cl_program)
        {
            //name+build options hash
            Serialize<MI_U32>(prog_stream, bprog.first);

            // name
            Serialize<std::string>(prog_stream, bprog.second->m_name);

            // build options
            Serialize<std::string>(prog_stream, bprog.second->m_build_options);

            // crc
            Serialize<MI_U32>(prog_stream, bprog.second->m_crc_val);

            std::vector<std::vector<MI_UCHAR>> binaries_vec;
            cl_int cl_err = bprog.second->m_cl_program->getInfo(CL_PROGRAM_BINARIES, &binaries_vec);

            if (cl_err != CL_SUCCESS)
            {
                std::string info = "Get program binary failed: " + GetCLErrorInfo(cl_err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                ret = Status::ERROR;
                break;
            }

            Serialize<std::vector<std::vector<MI_UCHAR>>>(prog_stream, binaries_vec);

            // stream << prog_stream.str();
            output += prog_stream.str();
        }
    }

    return ret;
}

MI_U32 CLProgramContainer::GetHash(const std::string &str)
{
    MI_U32 hash = 0;

    const MI_S32 length = str.length();
    const MI_U8 *ptr    = reinterpret_cast<const MI_U8*>(str.c_str());

    for (MI_S32 i = 0; i < length; i++)
    {
        hash += (MI_U32)(ptr[i]);
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    return hash;
}

} // namespace aura