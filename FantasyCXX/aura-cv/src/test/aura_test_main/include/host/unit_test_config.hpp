
#ifndef AURA_TEST_CONFIG_HPP
#define AURA_TEST_CONFIG_HPP

#include "aura/runtime/core.h"
#include "aura/runtime/context.h"
#include "aura/tools/json.h"
#include "aura/tools/unit_test.h"

#include <fstream>

namespace aura
{

AURA_INLINE std::ostream& operator<<(std::ostream &os, const LogLevel &log_level)
{
    switch (log_level)
    {
        case LogLevel::DEBUG:
        {
            os << "DEBUG";
            break;
        }

        case LogLevel::ERROR:
        {
            os << "ERROR";
            break;
        }

        case LogLevel::INFO:
        {
            os << "INFO";
            break;
        }

        default:
        {
            os << "undefined log level";
            break;
        }
    }

    return os;
}

AURA_INLINE std::ostream& operator<<(std::ostream &os, const LogOutput &log_output)
{
    switch (log_output)
    {
        case LogOutput::FARF:
        {
            os << "FARF";
            break;
        }

        case LogOutput::FILE:
        {
            os << "FILE";
            break;
        }

        case LogOutput::LOGCAT:
        {
            os << "LOGCAT";
            break;
        }

        case LogOutput::STDOUT:
        {
            os << "STDOUT";
            break;
        }

        default:
        {
            os << "undefined log output type";
            break;
        }
    }

    return os;
}

AURA_INLINE std::ostream& operator<<(std::ostream &os, const CpuAffinity &affinity)
{
    switch (affinity)
    {
        case CpuAffinity::ALL:
        {
            os << "ALL";
            break;
        }

        case CpuAffinity::BIG:
        {
            os << "BIG";
            break;
        }

        case CpuAffinity::LITTLE:
        {
            os << "LITTLE";
            break;
        }
        default:
        {
            os << "undefined affinity type";
            break;
        }
    }

    return os;
}

AURA_INLINE LogLevel StringToLogLevelType(const std::string &type_str)
{
    std::unordered_map<std::string, const LogLevel> loglevel_maps
    {
        {"DEBUG", LogLevel::DEBUG},
        {"ERROR", LogLevel::ERROR},
        {"INFO",  LogLevel::INFO },
    };

    std::string key = type_str;
    StringToUpper(key);

    if (loglevel_maps.count(key) > 0)
    {
        return loglevel_maps.at(key);
    }
    else
    {
        return LogLevel::DEBUG;
    }
}

AURA_INLINE LogOutput StringToLogOutputType(const std::string &type_str)
{
    std::unordered_map<std::string, const LogOutput> logoutput_maps
    {
        {"STDOUT", LogOutput::STDOUT},
        {"LOGCAT", LogOutput::LOGCAT},
        {"FILE",   LogOutput::FILE  },
        {"FARF",   LogOutput::FARF  },
    };

    std::string key = type_str;
    StringToUpper(key);

    if (logoutput_maps.count(key) > 0)
    {
        return logoutput_maps.at(key);
    }
    else
    {
        return LogOutput::STDOUT;
    }
}

AURA_INLINE CpuAffinity StringToAffinityType(const std::string &type_str)
{
    std::unordered_map<std::string, const CpuAffinity> affinity_maps
    {
        {"ALL",    CpuAffinity::ALL   },
        {"BIG",    CpuAffinity::BIG   },
        {"LITTLE", CpuAffinity::LITTLE},
    };

    std::string key = type_str;
    StringToUpper(key);

    if (affinity_maps.count(key) > 0)
    {
        return affinity_maps.at(key);
    }
    else
    {
        return CpuAffinity::ALL;
    }
}

class AURA_EXPORTS UnitTestConfig
{
public:
    UnitTestConfig()
    {
        m_log_level            = LogLevel::DEBUG;
        m_log_output           = LogOutput::STDOUT;
        m_log_file_name        = std::string("log");
        m_compute_affinity     = CpuAffinity::BIG;
        m_async_affinity       = CpuAffinity::LITTLE;
        m_cache_bin_path       = std::string("/data/local/tmp");
        m_cache_bin_prefix     = std::string("aura_unit_test");
        m_report_type          = std::string("txt");
        m_report_name          = std::string("auto_test");
        m_stress_count         = 0;
        m_data_path            = std::string("./data/");
        m_pil_path             = std::string("./pil/");
        m_enable_mem_profiling = 0;
        m_dump_path            = std::string("./");
    };

    ~UnitTestConfig() = default;

    DT_VOID PrintInfo() const
    {
        std::stringstream sstream;
        sstream << "============ Current UnitTest Config ===========" << std::endl;
        sstream << "log_level              = " << this->m_log_level << std::endl;
        sstream << "log_output             = " << this->m_log_output << std::endl;
        sstream << "log_file_name          = " << this->m_log_file_name << std::endl;
        sstream << "compute_affinity       = " << this->m_compute_affinity << std::endl;
        sstream << "async_affinity         = " << this->m_async_affinity << std::endl;
        sstream << "cache_bin_path         = " << this->m_cache_bin_path << std::endl;
        sstream << "cache_bin_prefix       = " << this->m_cache_bin_prefix << std::endl;
        sstream << "report_type            = " << this->m_report_type << std::endl;
        sstream << "report_name            = " << this->m_report_name << std::endl;
        sstream << "stress_count           = " << this->m_stress_count << std::endl;
        sstream << "data_path              = " << this->m_data_path << std::endl;
        sstream << "pil_path               = " << this->m_pil_path << std::endl;
        sstream << "enable_mem_profiling   = " << this->m_enable_mem_profiling << std::endl;
        sstream << "dump_path              = " << this->m_dump_path << std::endl;
        sstream << "================================================" << std::endl;

        std::cout << sstream.str() << std::endl;
    }

    Status Load(const std::string &filename)
    {
        if (GetFileSuffixStr(filename) != "json")
        {
            return Status::ERROR;
        }

        std::ifstream ifs(filename);
        if (!ifs.is_open())
        {
            return Status::ERROR;
        }

        aura_json::json json_obj;
        ifs >> json_obj;

        if (json_obj.count("log_level") > 0)
        {
            this->m_log_level = StringToLogLevelType(json_obj["log_level"]);
        }

        if (json_obj.count("log_output") > 0)
        {
            this->m_log_output = StringToLogOutputType(json_obj["log_output"]);
        }

        if (json_obj.count("log_file_name") > 0)
        {
            this->m_log_file_name = json_obj["log_file_name"];
        }

        if (json_obj.count("compute_affinity") > 0)
        {
            this->m_compute_affinity = StringToAffinityType(json_obj["compute_affinity"]);
        }

        if (json_obj.count("async_affinity") > 0)
        {
            this->m_async_affinity = StringToAffinityType(json_obj["async_affinity"]);
        }

        if (json_obj.count("cache_bin_path") > 0)
        {
            this->m_cache_bin_path = json_obj["cache_bin_path"];
        }

        if (json_obj.count("cache_bin_prefix") > 0)
        {
            this->m_cache_bin_prefix = json_obj["cache_bin_prefix"];
        }

        if (json_obj.count("report_type") > 0)
        {
            this->m_report_type = json_obj["report_type"];
        }

        if (json_obj.count("report_name") > 0)
        {
            this->m_report_name = json_obj["report_name"];
        }

        if (json_obj.count("stress_count") > 0)
        {
            this->m_stress_count = json_obj["stress_count"];
            this->m_stress_count = Max(this->m_stress_count, (DT_S32)0);
        }

        if (json_obj.count("data_path") > 0)
        {
            this->m_data_path = json_obj["data_path"];
        }

        if (json_obj.count("pil_path") > 0)
        {
            this->m_pil_path = json_obj["pil_path"];
        }

        if (json_obj.count("enable_mem_profiling") > 0)
        {
            this->m_enable_mem_profiling = json_obj["enable_mem_profiling"];
        }

        if (json_obj.count("dump_path") > 0)
        {
            this->m_dump_path = json_obj["dump_path"];
        }

        ifs.close();

        return Status::OK;
    }

    Status Save(const std::string &filename) const
    {
        Context *ctx = UnitTest::GetInstance()->GetContext();

        if (GetFileSuffixStr(filename) != "json")
        {
            AURA_LOGI(ctx, AURA_TAG, "UnitTestConfig Save only support .json filename.");
            return Status::ERROR;
        }

        std::ofstream ofs(filename);

        if (!ofs.is_open())
        {
            return Status::ERROR;
        }

        aura_json::json    json_obj;
        std::stringstream sstream;
        std::string       str_buffer;

        sstream << this->m_log_level << "\n" << this->m_log_output << "\n" << this->m_report_type << "\n"
                << this->m_compute_affinity << "\n" << this->m_async_affinity << "\n";

        sstream >> str_buffer;
        json_obj["log_level"] = str_buffer;

        sstream >> str_buffer;
        json_obj["log_output"] = str_buffer;

        sstream >> str_buffer;
        json_obj["report_type"] = str_buffer;

        sstream >> str_buffer;
        json_obj["compute_affinity"] = str_buffer;

        sstream >> str_buffer;
        json_obj["async_affinity"] = str_buffer;

        json_obj["log_file_name"]       = this->m_log_file_name;
        json_obj["cache_bin_path"]      = this->m_cache_bin_path;
        json_obj["cache_bin_prefix"]    = this->m_cache_bin_prefix;

        json_obj["report_type"]          = this->m_report_type;
        json_obj["report_name"]          = this->m_report_name;
        json_obj["stress_count"]         = this->m_stress_count;
        json_obj["data_path"]            = this->m_data_path;
        json_obj["pil_path"]             = this->m_pil_path;
        json_obj["enable_mem_profiling"] = this->m_enable_mem_profiling;
        json_obj["dump_path"]            = this->m_dump_path;

        ofs << json_obj.dump(4) << std::endl;
        ofs.close();

        return Status::OK;
    }

public:
    LogLevel    m_log_level;
    LogOutput   m_log_output;
    std::string m_log_file_name;
    CpuAffinity m_compute_affinity;
    CpuAffinity m_async_affinity;
    std::string m_cache_bin_path;
    std::string m_cache_bin_prefix;
    std::string m_report_type;
    std::string m_report_name;
    DT_S32      m_stress_count;
    std::string m_data_path;
    std::string m_pil_path;
    DT_S32      m_enable_mem_profiling;
    std::string m_dump_path;
};

} // namespace aura
#endif //AURA_TEST_CONFIG_HPP
