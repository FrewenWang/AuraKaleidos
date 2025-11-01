#include "aura/runtime/utils/host/systrace.hpp"

#define TRACE_MESSAGE_MAX_LEN                               (256)

namespace aura
{

Systrace::Systrace(MI_BOOL enable_trace) : m_handle(-1)
{
    if (MI_TRUE == enable_trace)
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

AURA_VOID Systrace::Begin(const MI_CHAR *tag)
{
    if (m_handle != -1)
    {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        MI_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        MI_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "B|%d|%s", getpid(), tag);
        write(m_handle, buffer, len);
    }
}

AURA_VOID Systrace::End(const MI_CHAR *tag)
{
    if (m_handle != -1)
    {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        MI_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        MI_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "E|%d|%s", getpid(), tag);
        write(m_handle, buffer, len);
    }
}

AURA_VOID Systrace::AsyncBegin(const MI_CHAR *tag, MI_S32 cookie)
{
    if (m_handle != -1)
    {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        MI_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        MI_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "S|%d|%s|%i", getpid(), tag, cookie);
        write(m_handle, buffer, len);
    }
}

AURA_VOID Systrace::AsyncEnd(const MI_CHAR *tag, MI_S32 cookie)
{
    if (m_handle != -1)
    {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        MI_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        MI_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "F|%d|%s|%i", getpid(), tag, cookie);
        write(m_handle, buffer, len);
    }
}

} // namespace aura