#include "host/xtensa_library.hpp"

#if !defined(AURA_BUILD_XPLORER)

#include "aura/runtime/logger.h"
#include "vdsp_interface_c.h"

#include <dlfcn.h>
#include <iostream>

namespace aura
{

XtensaLibrary& XtensaLibrary::Get()
{
    static XtensaLibrary library;
    return library;
}

XtensaLibrary::XtensaLibrary() : m_handle(MI_NULL)
{
    const std::vector<std::string> g_default_xtensa_library_paths =
    {
        "/vendor/lib64/libVDSPRuntime.so"
    };

    for (auto &path : g_default_xtensa_library_paths)
    {
        AURA_VOID *handle = LoadSymbols(path);
        if (handle)
        {
            m_handle = handle;
            break;
        }
    }
}

XtensaLibrary::~XtensaLibrary()
{
    if (m_handle)
    {
        dlclose(m_handle);
        m_handle = MI_NULL;
    }
}

AURA_VOID* XtensaLibrary::LoadSymbols(const std::string &path)
{
    Status ret = Status::ERROR;

    dlerror();

    AURA_VOID *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (MI_NULL == handle)
    {
        std::string info = "dlopen " + path + " failed, err : " + std::string(dlerror());
        AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
        return handle;
    }

    do
    {
        AURA_DLSYM_API(handle, vdsp_init)
        AURA_DLSYM_API(handle, vdsp_create_buffer)
        AURA_DLSYM_API(handle, vdsp_free_buffer)
        AURA_DLSYM_API(handle, vdsp_set_power)
        AURA_DLSYM_API(handle, vdsp_cache_start)
        AURA_DLSYM_API(handle, vdsp_cache_end)
        AURA_DLSYM_API(handle, vdsp_map_buffer)
        AURA_DLSYM_API(handle, vdsp_unmap_buffer)
        AURA_DLSYM_API(handle, vdsp_run_node)
        AURA_DLSYM_API(handle, vdsp_release)

        ret = Status::OK;
    } while (0);

    if (ret != Status::OK)
    {
        dlclose(handle);
        handle = MI_NULL;
    }

    return handle;
}

} // namespace aura

int vdsp_init(void **handle, vdsp_init_para *param, vdsp_init_response *ids)
{
    auto func = aura::XtensaLibrary::Get().vdsp_init;

    if (func)
    {
        return func(handle, param, ids);
    }

    return -1;
}

int vdsp_create_buffer(void *handle, uint32_t mem_size, bool to_dsp, uint32_t *mem_handle, uint8_t **cpu_addr, uint32_t* dsp_addr)
{
    auto func = aura::XtensaLibrary::Get().vdsp_create_buffer;

    if (func)
    {
        return func(handle, mem_size, to_dsp, mem_handle, cpu_addr, dsp_addr);
    }

    return -1;
}

int vdsp_free_buffer(void *handle, uint32_t mem_handle)
{
    auto func = aura::XtensaLibrary::Get().vdsp_free_buffer;

    if (func)
    {
        return func(handle, mem_handle);
    }

    return -1;
}

int vdsp_set_power(void *handle, uint32_t *level)
{
    auto func = aura::XtensaLibrary::Get().vdsp_set_power;

    if (func)
    {
        return func(handle, level);
    }

    return -1;
}

int vdsp_cache_start(void *handle, uint32_t mem_handle)
{
    auto func = aura::XtensaLibrary::Get().vdsp_cache_start;

    if (func)
    {
        return func(handle, mem_handle);
    }

    return -1;
}

int vdsp_cache_end(void *handle, uint32_t mem_handle)
{
    auto func = aura::XtensaLibrary::Get().vdsp_cache_end;

    if (func)
    {
        return func(handle, mem_handle);
    }

    return -1;
}

int vdsp_map_buffer(void *handle, uint32_t mem_handle, uint32_t mem_size, bool to_vdsp, uint32_t *dsp_va)
{
    auto func = aura::XtensaLibrary::Get().vdsp_map_buffer;

    if (func)
    {
        return func(handle, mem_handle, mem_size, to_vdsp, dsp_va);
    }

    return -1;
}

int vdsp_unmap_buffer(void *handle, uint32_t mem_handle)
{
    auto func = aura::XtensaLibrary::Get().vdsp_unmap_buffer;

    if (func)
    {
        return func(handle, mem_handle);
    }

    return -1;
}

int vdsp_run_node(void *handle, uint32_t op_id, uint32_t cycle, void* in_data, uint32_t in_size, void* out_data, uint32_t out_size)
{
    auto func = aura::XtensaLibrary::Get().vdsp_run_node;

    if (func)
    {
        return func(handle, op_id, cycle, in_data, in_size, out_data, out_size);
    }

    return -1;
}

int vdsp_release(void *handle)
{
    auto func = aura::XtensaLibrary::Get().vdsp_release;

    if (func)
    {
        return func(handle);
    }

    return -1;
}
#endif // AURA_BUILD_XPLORER