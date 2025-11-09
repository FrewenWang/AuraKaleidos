#include "aura/runtime/utils/host/systrace.hpp"

#define TRACE_MESSAGE_MAX_LEN                               (256)

namespace aura
{

Systrace::Systrace(DT_BOOL enable_trace) : m_handle(-1)
{
    if (DT_TRUE == enable_trace)
    {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        if (-1 == m_handle)
        {
            m_handle = open("/sys/kernel/tracing/trace_marker", O_WRONLY);
        }
    }
}

Systrace::~Systrace()
{
    if (m_handle != -1)
    {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        close(m_handle);
        m_handle = -1;
    }
}

DT_VOID Systrace::Begin(const DT_CHAR *tag)
{
    if (m_handle != -1)
    {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        DT_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        DT_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "B|%d|%s", getpid(), tag);
        write(m_handle, buffer, len);
    }
}

DT_VOID Systrace::End(const DT_CHAR *tag)
{
    if (m_handle != -1)
    {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        DT_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        DT_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "E|%d|%s", getpid(), tag);
        write(m_handle, buffer, len);
    }
}

DT_VOID Systrace::AsyncBegin(const DT_CHAR *tag, DT_S32 cookie)
{
    if (m_handle != -1)
    {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        DT_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        DT_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "S|%d|%s|%i", getpid(), tag, cookie);
        write(m_handle, buffer, len);
    }
}

DT_VOID Systrace::AsyncEnd(const DT_CHAR *tag, DT_S32 cookie)
{
    if (m_handle != -1)
    {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        DT_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        DT_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "F|%d|%s|%i", getpid(), tag, cookie);
        write(m_handle, buffer, len);
    }
}

} // namespace aura