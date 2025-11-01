
#include "systrace.hpp"
#include "aura/utils/core.h"

#include <mutex>
#include <unistd.h>
#include <sys/fcntl.h>

#define TRACE_MESSAGE_MAX_LEN                               (256)

namespace aura::utils
{
Systrace::Systrace(AURA_BOOL enable_trace) : m_handle(-1) {
    if (AURA_TRUE == enable_trace) {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        if (-1 == m_handle) {
            m_handle = open("/sys/kernel/tracing/trace_marker", O_WRONLY);
        }
    }
}

Systrace::~Systrace() {
    if (m_handle != -1) {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        close(m_handle);
        m_handle = -1;
    }
}

AURA_VOID Systrace::Begin(const AURA_CHAR *tag) {
    if (m_handle != -1) {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        AURA_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        AURA_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "B|%d|%s", getpid(), tag);
        write(m_handle, buffer, len);
    }
}

::AURA_VOID Systrace::End(const AURA_CHAR *tag) {
    if (m_handle != -1) {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        AURA_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        AURA_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "E|%d|%s", getpid(), tag);
        write(m_handle, buffer, len);
    }
}

AURA_VOID Systrace::AsyncBegin(const AURA_CHAR *tag, AURA_S32 cookie) {
    if (m_handle != -1) {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        AURA_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        AURA_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "S|%d|%s|%i", getpid(), tag, cookie);
        write(m_handle, buffer, len);
    }
}

AURA_VOID Systrace::AsyncEnd(const AURA_CHAR *tag, AURA_S32 cookie) {
    if (m_handle != -1) {
        std::lock_guard<std::mutex> guard(m_handle_lock);
        AURA_CHAR buffer[TRACE_MESSAGE_MAX_LEN] = {0};
        AURA_S32 len = snprintf(buffer, TRACE_MESSAGE_MAX_LEN, "F|%d|%s|%i", getpid(), tag, cookie);
        write(m_handle, buffer, len);
    }
}
} // namespace mage
