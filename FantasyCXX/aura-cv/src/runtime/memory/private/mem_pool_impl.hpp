 #ifndef AURA_RUNTIME_MEMORY_MEM_POOL_IMPL_HPP__
#define AURA_RUNTIME_MEMORY_MEM_POOL_IMPL_HPP__

#include "aura/runtime/memory/mem_pool.hpp"

#include <unordered_map>
#include <deque>
#include <mutex>
#include <atomic>

namespace aura
{
class MemPool::Impl
{
public:
~Impl();

    AURA_VOID* Allocate(MI_S32 type, MI_S64 size, MI_S32 align,
        const MI_CHAR *file, const MI_CHAR *func, MI_S32 line);
        Impl(Context *ctx);

    Status Free(AURA_VOID *ptr);

    Status Map(const Buffer &buffer);

    Status Unmap(const Buffer &buffer);

    Buffer GetBuffer(AURA_VOID *ptr);

    Status RegisterAllocator(MI_S32 type, Allocator *allocator);

    Status UnregisterAllocator(MI_S32 type);

    Allocator* GetAllocator(MI_S32 type);

    AURA_VOID MemTraceSet(MI_BOOL enable);

    AURA_VOID MemTraceBegin(const std::string &tag);

    AURA_VOID MemTraceEnd(const std::string &tag);

    AURA_VOID MemTraceClear();

    std::string MemTraceReport();

private:
    struct MemTraceData
    {
        MemTraceData(MI_S32 l, size_t cnt, MI_S64 cur, MI_S64 max)
                   : level(l), alloc_cnt(cnt), cur_size(cur), max_size(max)
        {}

        size_t level;
        size_t alloc_cnt;
        MI_S64 cur_size;
        MI_S64 max_size;
    };

    struct MemBlkData
    {
        MemBlkData(const MI_CHAR *file_name, const MI_CHAR *func_name, MI_S32 line_num,
                   Buffer &buffer, MI_BOOL mapped) 
                 : file(file_name), func(func_name), line(line_num),
                   buffer(buffer), is_mapped(mapped)
        {}

        std::string file;
        std::string func;
        MI_S32      line;

        Buffer  buffer;
        MI_BOOL is_mapped;
    };

    Context   *m_ctx;
    std::mutex m_lock;

    std::deque<std::pair<std::string, MemTraceData>> m_mtrace_running;
    std::deque<std::pair<std::string, MemTraceData>> m_mtrace_completed;
    std::atomic<MI_BOOL> m_mtrace_enable;
    std::atomic<MI_BOOL> m_mtrace_match;
    std::string m_mtrace_dismatch_string;
    std::mutex m_mtrace_lock;

    MI_S64 m_peak_mem_size;
    MI_S64 m_total_mem_size;

    std::unordered_map<MI_UPTR_T, MemBlkData> m_mblk_map;
    std::unordered_map<MI_S32, Allocator*> m_allocators;
};

} // namespace aura

#endif // AURA_RUNTIME_MEMORY_MEM_POOL_IMPL_HPP__
