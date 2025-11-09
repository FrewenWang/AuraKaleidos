#ifndef AURA_RUNTIME_XTENSA_HOST_XTENSA_LIBRARY_HPP__
#define AURA_RUNTIME_XTENSA_HOST_XTENSA_LIBRARY_HPP__

#include "aura/runtime/core.h"
#include "vdsp_interface_c.h"

#if !defined(AURA_BUILD_XPLORER)

namespace aura
{
class XtensaLibrary final
{
public:
    static XtensaLibrary& Get();

    AURA_API_DEF(vdsp_init) = int (*)(void*, vdsp_init_para*, vdsp_init_response*);
    AURA_API_PTR(vdsp_init);

    AURA_API_DEF(vdsp_create_buffer) = int (*)(void*, uint32_t, bool, uint32_t*, uint8_t**, uint32_t*);
    AURA_API_PTR(vdsp_create_buffer);

    AURA_API_DEF(vdsp_free_buffer) = int (*)(void*, uint32_t);
    AURA_API_PTR(vdsp_free_buffer);

    AURA_API_DEF(vdsp_set_power) = int (*)(void*, uint32_t*);
    AURA_API_PTR(vdsp_set_power);

    AURA_API_DEF(vdsp_wait) = int (*)(void*, uint32_t);
    AURA_API_PTR(vdsp_wait);

    AURA_API_DEF(vdsp_cache_start) = int (*)(void*, uint32_t);
    AURA_API_PTR(vdsp_cache_start);

    AURA_API_DEF(vdsp_cache_end) = int (*)(void*, uint32_t);
    AURA_API_PTR(vdsp_cache_end);

    AURA_API_DEF(vdsp_map_buffer) = int (*)(void*, uint32_t, uint32_t, bool, uint32_t*);
    AURA_API_PTR(vdsp_map_buffer);

    AURA_API_DEF(vdsp_unmap_buffer) = int (*)(void*, uint32_t);
    AURA_API_PTR(vdsp_unmap_buffer);

    AURA_API_DEF(vdsp_run_node) = int (*)(void*, uint32_t, uint32_t, void*, uint32_t, void*, uint32_t);
    AURA_API_PTR(vdsp_run_node);

    AURA_API_DEF(vdsp_release) = int (*)(void*);
    AURA_API_PTR(vdsp_release);

private:
    XtensaLibrary();

    ~XtensaLibrary();

    AURA_DISABLE_COPY_AND_ASSIGN(XtensaLibrary);

    DT_VOID* LoadSymbols(const std::string &path);

private:
    DT_VOID    *m_handle;
};

} // namespace aura

#endif // AURA_BUILD_XPLORER
#endif // AURA_RUNTIME_XTENSA_HOST_XTENSA_LIBRARY_HPP__