#ifndef AURA_RUNTIME_HEXAGON_HEXAGON_LIBRARY_HPP__
#define AURA_RUNTIME_HEXAGON_HEXAGON_LIBRARY_HPP__

#include "aura/runtime/core.h"

#include "remote.h"

namespace aura
{

class HexagonLibrary final
{
public:
    static HexagonLibrary& Get();

    // AURA_API_DEF(remote_handle_open) = int (*)(const char*, remote_handle*);
    // AURA_API_PTR(remote_handle_open);
    AURA_API_DEF(remote_handle64_open) = int (*)(const char*, remote_handle64*);
    AURA_API_PTR(remote_handle64_open);

    // AURA_API_DEF(remote_handle_invoke) = int (*)(remote_handle, uint32_t, remote_arg*);
    // AURA_API_PTR(remote_handle_invoke);
    AURA_API_DEF(remote_handle64_invoke) = int (*)(remote_handle64, uint32_t, remote_arg*);
    AURA_API_PTR(remote_handle64_invoke);

    // AURA_API_DEF(remote_handle_close) = int (*)(remote_handle);
    // AURA_API_PTR(remote_handle_close);
    AURA_API_DEF(remote_handle64_close) = int (*)(remote_handle64);
    AURA_API_PTR(remote_handle64_close);

    AURA_API_DEF(remote_handle_control) = int (*)(uint32_t, void*, uint32_t);
    AURA_API_PTR(remote_handle_control);
    // AURA_API_DEF(remote_handle64_control) = int (*)(remote_handle64, uint32_t, void*, uint32_t);
    // AURA_API_PTR(remote_handle64_control);

    AURA_API_DEF(remote_session_control) = int (*)(uint32_t, void*, uint32_t);
    AURA_API_PTR(remote_session_control);

    // AURA_API_DEF(remote_handle_invoke_async) = int (*)(remote_handle, fastrpc_async_descriptor_t*, uint32_t, remote_arg*);
    // AURA_API_PTR(remote_handle_invoke_async);
    // AURA_API_DEF(remote_handle64_invoke_async) = int (*)(remote_handle64, fastrpc_async_descriptor_t*, uint32_t, remote_arg*);
    // AURA_API_PTR(remote_handle64_invoke_async);

    // AURA_API_DEF(fastrpc_async_get_status) = int (*)(fastrpc_async_jobid, int, int*);
    // AURA_API_PTR(fastrpc_async_get_status);

    // AURA_API_DEF(fastrpc_release_async_job) = int (*)(fastrpc_async_jobid);
    // AURA_API_PTR(fastrpc_release_async_job);

    // AURA_API_DEF(remote_mmap) = int (*)(int, uint32_t, uint32_t, int, uint32_t*);
    // AURA_API_PTR(remote_mmap);
    // AURA_API_DEF(remote_munmap) = int (*)(uint32_t, int);
    // AURA_API_PTR(remote_munmap);

    // AURA_API_DEF(remote_mem_map) = int (*)(int, int, int, uint64_t, size_t, uint64_t*);
    // AURA_API_PTR(remote_mem_map);
    // AURA_API_DEF(remote_mem_unmap) = int (*)(int, uint64_t, size_t);
    // AURA_API_PTR(remote_mem_unmap);

    // AURA_API_DEF(remote_mmap64) = int (*)(int, uint32_t, uint64_t, int16x4_t, uint64_t*);
    // AURA_API_PTR(remote_mmap64);
    // AURA_API_DEF(remote_munmap64) = int (*)(uint64_t, int16x4_t);
    // AURA_API_PTR(remote_munmap64);

    // AURA_API_DEF(fastrpc_mmap) = int (*)(int, int, void*, int, size_t, enum fastrpc_map_flags);
    // AURA_API_PTR(fastrpc_mmap);
    // AURA_API_DEF(fastrpc_munmap) = int (*)(int, int, void*, size_t);
    // AURA_API_PTR(fastrpc_munmap);

    // AURA_API_DEF(remote_register_buf) = void (*)(void*, int, int);
    // AURA_API_PTR(remote_register_buf);
    AURA_API_DEF(remote_register_buf_attr) = void (*)(void*, int, int, int);
    AURA_API_PTR(remote_register_buf_attr);

    // AURA_API_DEF(remote_register_dma_handle) = int (*)(int, uint32_t);
    // AURA_API_PTR(remote_register_dma_handle);
    // AURA_API_DEF(remote_register_dma_handle_attr) = int (*)(int, uint32_t, uint32_t);
    // AURA_API_PTR(remote_register_dma_handle_attr);

    // AURA_API_DEF(remote_set_mode) = int (*)(uint32_t);
    // AURA_API_PTR(remote_set_mode);

    // AURA_API_DEF(remote_register_fd) = void* (*)(int, int);
    // AURA_API_PTR(remote_register_fd)

private:
    HexagonLibrary();

    ~HexagonLibrary();

    AURA_DISABLE_COPY_AND_ASSIGN(HexagonLibrary);

    DT_VOID* LoadSymbols(const std::string &path);

private:
    DT_VOID    *m_handle;
};

} // namespace aura

#endif // AURA_RUNTIME_HEXAGON_HEXAGON_LIBRARY_HPP__