#include "logger_impl.hpp"

#include <sstream>
#include <thread>


#if defined(AURA_BUILD_WINDOWS)
#define LOGER_PRINT_RESET
#define LOGER_PRINT_RED
#define LOGER_PRINT_BLUE
#define LOGER_PRINT_GREEN
#else
#define LOGER_PRINT_RESET "\033[1;0m"
#define LOGER_PRINT_RED "\033[1;31m"
#define LOGER_PRINT_BLUE "\033[1;34m"
#define LOGER_PRINT_GREEN "\033[1;32m"
#endif

#define GET_FORMAT_STR(buffer, format)                                                                                 \
    va_list args;                                                                                                      \
    va_start(args, format);                                                                                            \
    va_list args_copy;                                                                                                 \
    va_copy(args_copy, args);                                                                                          \
    AURA_S32 length = std::vsnprintf(AURA_NULL, 0, format, args_copy);                                                 \
    va_end(args_copy);                                                                                                 \
    std::string buffer(length, 0);                                                                                     \
    std::vsnprintf(&buffer[0], length + 1, format, args);                                                              \
    va_end(args);

namespace aura::utils
{

static std::string GetExtInfo()
{
    std::stringstream ss;
    ss << TimeStamp::Now() << " ";
    return ss.str();
}

static AURA_VOID StdoutPrintImpl(LogLevel level, const std::string &info)
{
    switch (level)
    {
        case LogLevel::ERROR:
        {
            fprintf(stdout, LOGER_PRINT_RED "%s" LOGER_PRINT_RESET, info.c_str());
            break;
        }

        case LogLevel::INFO:
        {
            fprintf(stdout, LOGER_PRINT_BLUE "%s" LOGER_PRINT_RESET, info.c_str());
            break;
        }

        case LogLevel::DEBUG:
        {
            fprintf(stdout, "%s", info.c_str());
            break;
        }

        default:
        {
            fprintf(stdout, "%s", info.c_str());
        }
    }

    return;
}

static AURA_VOID FilePrintImpl(FILE *fp, const std::string &info)
{
    if (AURA_NULL == fp || info.empty())
    {
        return;
    }

    fprintf(fp, "%s", info.c_str());
}

#if (defined(AURA_BUILD_ANDROID) || defined(AURA_BUILD_HEXAGON))
static std::vector<std::string> SplitString(const std::string &str, const AURA_CHAR *delim)
{
    std::vector<std::string> sub_strings;
    size_t last  = 0;
    size_t index = str.find_first_of(delim, last);

    while (index != std::string::npos)
    {
        sub_strings.emplace_back(str.substr(last, index - last));
        last  = index + 1;
        index = str.find_first_of(delim, last);
    }

    if (last < str.size())
    {
        sub_strings.emplace_back(str.substr(last));
    }

    return sub_strings;
}
#endif // AURA_BUILD_ANDROID || AURA_BUILD_HEXAGON

static AURA_VOID LogcatPrintImpl(const AURA_CHAR *tag, LogLevel level, const std::string &info)
{
    AURA_UNUSED(tag);
    AURA_UNUSED(level);
    AURA_UNUSED(info);
    return;
}

AURA_EXPORTS AURA_VOID StdoutPrint(const AURA_CHAR *tag, LogLevel level, const AURA_CHAR *format, ...)
{
    GET_FORMAT_STR(info, format);
    std::string output = std::string("[") + tag + "] " + GetExtInfo() + info;
    StdoutPrintImpl(level, output);
}

AURA_EXPORTS AURA_VOID FilePrint(FILE *fp, const AURA_CHAR *tag, const AURA_CHAR *format, ...)
{
    GET_FORMAT_STR(info, format);
    std::string output = std::string("[") + tag + "] " + GetExtInfo() + info;

    FilePrintImpl(fp, output);
}

AURA_EXPORTS AURA_VOID LogcatPrint(const AURA_CHAR *tag, LogLevel level, const AURA_CHAR *format, ...)
{
    GET_FORMAT_STR(info, format);
    LogcatPrintImpl(tag, level, info);
}

AURA_EXPORTS AURA_VOID Print(const AURA_CHAR *tag, LogLevel level, const AURA_CHAR *format, ...)
{
    GET_FORMAT_STR(info, format);

#if defined(AURA_BUILD_ANDROID)
    LogcatPrintImpl(tag, level, info);
#elif defined(AURA_BUILD_HEXAGON)
    AURA_UNUSED(tag);
    FarfPrintImpl(level, info);
#else
    std::string output = std::string("[") + tag + "] " + GetExtInfo() + info;
    StdoutPrintImpl(level, output);
#endif
}

Logger::Impl::Impl(LogOutput output, LogLevel level, const std::string &file, AURA_S32 max_num_error_string) :
    m_output(output), m_level(level), m_file(file), m_mutex(), m_max_num_error_string(max_num_error_string),
    m_error_string_list(), m_fp(AURA_NULL)
{
    if (LogOutput::FILE == m_output && !m_file.empty())
    {
        m_fp = fopen(m_file.c_str(), "a+");
    }
}

Logger::Impl::~Impl()
{
    if (m_fp != AURA_NULL)
    {
        fclose(m_fp);
        m_fp = AURA_NULL;
    }
}

AURA_VOID Logger::Impl::Log(const AURA_CHAR *tag, LogLevel level, const AURA_CHAR *info)
{
    if (static_cast<AURA_S32>(m_level) < static_cast<AURA_S32>(level))
    {
        return;
    }

    std::string output = std::string("[") + tag + "] ";

    std::lock_guard<std::mutex> guard(m_mutex);
    switch (m_output)
    {
        case LogOutput::LOGCAT:
        {
            output = info;
            LogcatPrintImpl(tag, level, output);
            break;
        }
        case LogOutput::FILE:
        {
            output = output + GetExtInfo() + info;
            FilePrintImpl(m_fp, output);
            break;
        }
        default:
        {
            output = output + GetExtInfo() + info;
            StdoutPrintImpl(level, output);
            break;
        }
    }

    return;
}

AURA_VOID Logger::Impl::AddErrorString(ErrorString &error_string)
{
    std::lock_guard<std::mutex> guard(m_mutex);

    while (static_cast<AURA_S32>(m_error_string_list.size()) >= m_max_num_error_string)
    {
        m_error_string_list.pop_front();
    }

    m_error_string_list.push_back(error_string);
}

std::string Logger::Impl::GetErrorString()
{
    std::lock_guard<std::mutex> guard(m_mutex);
    std::string err_str;
#if defined(AURA_BUILD_HOST)
    err_str = "mage error backtrace : \n";
#elif defined(AURA_BUILD_HEXAGON)
    err_str = "\n";
#endif

    for (auto it = m_error_string_list.begin(); it != m_error_string_list.end(); it++)
    {
#if defined(AURA_BUILD_HOST)
        err_str += it->ToString() + "\n";
#elif defined(AURA_BUILD_HEXAGON)
        err_str += "    [hexagon]" + it->ToString() + "\n";
#endif
    }

    m_error_string_list.clear();
    return err_str;
}

Logger::Logger(LogOutput output, LogLevel level, const std::string &file)
{
    m_impl.reset(new Logger::Impl(output, level, file));
}

Logger::~Logger()
{
}

AURA_VOID Logger::Log(const AURA_CHAR *tag, LogLevel level, const AURA_CHAR *format, ...)
{
    GET_FORMAT_STR(info, format);

    if (m_impl)
    {
        m_impl->Log(tag, level, info.c_str());
    }
}

AURA_VOID Logger::AddErrorString(const AURA_CHAR *file, const AURA_CHAR *func, AURA_S32 line, const AURA_CHAR *info)
{
    ErrorString error_string(file, func, line, info);
    m_impl->AddErrorString(error_string);
}

std::string Logger::GetErrorString()
{
    return m_impl->GetErrorString();
}

} // namespace aura::utils
