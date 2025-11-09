#include "host/hexagon_library.hpp"

#include "aura/runtime/context.h"
#include "aura/runtime/logger.h"

#include <dlfcn.h>

namespace aura
{

HexagonLibrary& HexagonLibrary::Get()
{
    static HexagonLibrary library;
    return library;
}

HexagonLibrary::HexagonLibrary() : m_handle(DT_NULL)
{
    const std::vector<std::string> default_hexagon_library_paths =
    {
        "libcdsprpc.so",
        "/vendor/lib64/libcdsprpc.so"
    };

    for (auto &path : default_hexagon_library_paths)
    {
        DT_VOID *handle = LoadSymbols(path);
        if (handle)
        {
            m_handle = handle;
            break;
        }
    }
}

HexagonLibrary::~HexagonLibrary()
{
    if (m_handle)
    {
        dlclose(m_handle);
        m_handle = DT_NULL;
    }
}

DT_VOID* HexagonLibrary::LoadSymbols(const std::string &path)
{
    Status ret = Status::ERROR;

    dlerror();

    DT_VOID *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (DT_NULL == handle)
    {
        std::string info = "dlopen " + path + " failed, err : " + std::string(dlerror());
        AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
        return handle;
    }

    do
    {
        AURA_DLSYM_API(handle, remote_handle64_open)
        AURA_DLSYM_API(handle, remote_handle64_invoke)
        AURA_DLSYM_API(handle, remote_handle64_close)
        AURA_DLSYM_API(handle, remote_handle_control)
        AURA_DLSYM_API(handle, remote_session_control)
        AURA_DLSYM_API(handle, remote_register_buf_attr)

        ret = Status::OK;
    } while (0);

    if (ret != Status::OK)
    {
        dlclose(handle);
        handle = DT_NULL;
    }

    return handle;
}

} // namespace aura

int remote_handle64_open(const char *name, remote_handle64 *ph)
{
    auto func = aura::HexagonLibrary::Get().remote_handle64_open;

    if (func)
    {
        return func(name, ph);
    }

    return -1;
}

int remote_handle64_invoke(remote_handle64 h, uint32_t dw_scalars, remote_arg *pra)
{
    auto func = aura::HexagonLibrary::Get().remote_handle64_invoke;

    if (func)
    {
        return func(h, dw_scalars, pra);
    }

    return -1;
}

int remote_handle64_close(remote_handle64 h)
{
    auto func = aura::HexagonLibrary::Get().remote_handle64_close;

    if (func)
    {
        return func(h);
    }

    return -1;
}

int remote_handle_control(uint32_t req, void *data, uint32_t datalen)
{
    auto func = aura::HexagonLibrary::Get().remote_handle_control;

    if (func)
    {
        return func(req, data, datalen);
    }

    return -1;
}

int remote_session_control(uint32_t req, void *data, uint32_t datalen)
{
    auto func = aura::HexagonLibrary::Get().remote_session_control;

    if (func)
    {
        return func(req, data, datalen);
    }

    return -1;
}

void remote_register_buf_attr(void *buf, int size, int fd, int attr)
{
    auto func = aura::HexagonLibrary::Get().remote_register_buf_attr;

    if (func)
    {
        func(buf, size, fd, attr);
    }
}