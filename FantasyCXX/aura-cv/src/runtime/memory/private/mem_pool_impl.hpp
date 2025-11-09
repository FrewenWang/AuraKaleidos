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

    DT_VOID* Allocate(DT_S32 type, DT_S64 size, DT_S32 align,
        const DT_CHAR *file, const DT_CHAR *func, DT_S32 line);
        Impl(Context *ctx);

    Status Free(DT_VOID *ptr);

    Status Map(const Buffer &buffer);

    Status Unmap(const Buffer &buffer);

    Buffer GetBuffer(DT_VOID *ptr);

    Status RegisterAllocator(DT_S32 type, Allocator *allocator);

    Status UnregisterAllocator(DT_S32 type);

    Allocator* GetAllocator(DT_S32 type);

    DT_VOID MemTraceSet(DT_BOOL enable);

    DT_VOID MemTraceBegin(const std::string &tag);

    DT_VOID MemTraceEnd(const std::string &tag);

    DT_VOID MemTraceClear();

    std::string MemTraceReport();

private:
    struct MemTraceData
    {
        MemTraceData(DT_S32 l, size_t cnt, DT_S64 cur, DT_S64 max)
                   : level(l), alloc_cnt(cnt), cur_size(cur), max_size(max)
        {}

        size_t level;
        size_t alloc_cnt;
        DT_S64 cur_size;
        DT_S64 max_size;
    };

    struct MemBlkData
    {
        MemBlkData(const DT_CHAR *file_name, const DT_CHAR *func_name, DT_S32 line_num,
                   Buffer &buffer, DT_BOOL mapped) 
                 : file(file_name), func(func_name), line(line_num),
                   buffer(buffer), is_mapped(mapped)
        {}

        std::string file;
        std::string func;
        DT_S32      line;

        Buffer  buffer;
        DT_BOOL is_mapped;
    };

    Context   *m_ctx;
    std::mutex m_lock;

    std::deque<std::pair<std::string, MemTraceData>> m_mtrace_running;
    std::deque<std::pair<std::string, MemTraceData>> m_mtrace_completed;
    std::atomic<DT_BOOL> m_mtrace_enable;
    std::atomic<DT_BOOL> m_mtrace_match;
    std::string m_mtrace_dismatch_string;
    std::mutex m_mtrace_lock;

    DT_S64 m_peak_mem_size;
    DT_S64 m_total_mem_size;

    std::unordered_map<DT_UPTR_T, MemBlkData> m_mblk_map;
    std::unordered_map<DT_S32, Allocator*> m_allocators;
};

} // namespace aura

#endif // AURA_RUNTIME_MEMORY_MEM_POOL_IMPL_HPP__
