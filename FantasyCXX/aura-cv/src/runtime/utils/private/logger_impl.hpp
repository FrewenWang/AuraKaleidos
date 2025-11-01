#ifndef AURA_RUNTIME_UTILS_LOGGER_IMPL_HPP__
#define AURA_RUNTIME_UTILS_LOGGER_IMPL_HPP__

#include "aura/runtime/utils/logger.hpp"

#include <list>
#include <mutex>

namespace aura
{

class ErrorString
{
public:
    ErrorString(const MI_CHAR *file, const MI_CHAR *func, MI_S32 line, const MI_CHAR *info)
               : m_file(file), m_func(func), m_line(line), m_info(info)
    {
        m_time_stamp = TimeStamp::Now();
    }

    ~ErrorString() = default;

    friend std::ostream& operator<<(std::ostream &os, const ErrorString &err_str)
    {
        os << "  "   << err_str.m_time_stamp
           << " "    << err_str.m_func
           << "["    << err_str.m_file
           << ":"    << err_str.m_line
           << "] : " << err_str.m_info;
        return os;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << (*this);
        return ss.str();
    }

    TimeStamp   m_time_stamp;
    std::string m_file;
    std::string m_func;
    MI_S32      m_line;
    std::string m_info;
};

class Logger::Impl
{
public:
    Impl(LogOutput output, LogLevel level, const std::string &file, MI_S32 max_num_error_string = 128);

    ~Impl();

    AURA_VOID Log(const MI_CHAR *tag, LogLevel level, const MI_CHAR *buffer);
    AURA_VOID AddErrorString(ErrorString &error_string);
    std::string GetErrorString();

private:
    LogOutput   m_output;
    LogLevel    m_level;
    std::string m_file;
    std::mutex  m_mutex;

    MI_S32 m_max_num_error_string;
    std::list<ErrorString> m_error_string_list;
    FILE *m_fp;
};

} // namespace aura

#endif // AURA_RUNTIME_UTILS_LOGGER_IMPL_HPP__