#include "mem_pool_impl.hpp"
#include "allocator_heap.hpp"

#if defined(AURA_BUILD_HEXAGON)
#  include "hexagon/allocator_vtcm.hpp"
#elif defined(AURA_BUILD_ANDROID)
#  include "host/android/allocator_dma_buf_heap.hpp"
#endif
#include "aura/runtime/logger.h"

namespace aura
{

MemPool::Impl::Impl(Context *ctx) : m_ctx(ctx), m_lock(),
                    m_mtrace_enable(DT_FALSE), m_peak_mem_size(0),
                    m_total_mem_size(0), m_mblk_map(), m_allocators()
{
    std::lock_guard<std::mutex> guard(m_lock);
    // 在内存池对象进行初始化的时候，我们初始化各种内存池对应的内存分配器对象
    // 默认分配器是： 普通堆内存分配器
    m_allocators.emplace(AURA_MEM_HEAP, new AllocatorHeap());

#if defined(AURA_BUILD_HEXAGON)
    /// 如果是构建的Hexagon,则对应的分配器是AURA_MEM_VTCM（向量紧耦合内存）
    m_allocators.emplace(AURA_MEM_VTCM, new AllocatorVtcm());
#elif defined(AURA_BUILD_ANDROID)
    //// 如果是构建android系统，我们同样还需要构建DMA buffer的内存分配器
    {
        Allocator *allocator = CreateDMABufHeapAllocator();
        if (allocator)
        {
            m_allocators.emplace(AURA_MEM_DMA_BUF_HEAP, allocator);
        }
    }
#endif
}

MemPool::Impl::~Impl()
{
    std::lock_guard<std::mutex> guard(m_lock);

    AURA_LOGD(m_ctx, AURA_TAG, "***********************************************\n");

    AURA_LOGD(m_ctx, AURA_TAG, "* Peak  Mem size: %.2f KB (%.4f MB)\n",
              m_peak_mem_size / 1024.f, m_peak_mem_size / 1048576.f);

    AURA_LOGD(m_ctx, AURA_TAG, "* Total Mem size: %.2f KB (%.4f MB)\n",
              m_total_mem_size / 1024.f, m_total_mem_size / 1048576.f);

    if (!m_mblk_map.empty())
    {
        AURA_LOGD(m_ctx, AURA_TAG, "****************** Mem Leak *******************\n");
        AURA_LOGD(m_ctx, AURA_TAG, "****************** Blk info *******************\n");

        DT_S32 counter = 0;
        DT_S32 leak_mem_size = 0;

        for (auto iter = m_mblk_map.begin(); iter != m_mblk_map.end(); ++iter)
        {
            DT_S32 type = iter->second.buffer.m_type;
            DT_S64 size = iter->second.buffer.m_size;

            AURA_LOGD(m_ctx, AURA_TAG, "* blk [%zu] - %p\n", counter, reinterpret_cast<DT_VOID*>(iter->first));
            AURA_LOGD(m_ctx, AURA_TAG, "*   type: %s\n", m_allocators[type]->GetName().c_str());
            AURA_LOGD(m_ctx, AURA_TAG, "*   size: %zu byte\n", size);
            AURA_LOGD(m_ctx, AURA_TAG, "*   file: %s\n",  iter->second.file.c_str());
            AURA_LOGD(m_ctx, AURA_TAG, "*   func: %s - %zu\n", iter->second.func.c_str(), iter->second.line);
            AURA_LOGD(m_ctx, AURA_TAG, "*\n");

            counter++;
            leak_mem_size += size;

            if (m_allocators.count(type))
            {
                m_allocators[type]->Free(iter->second.buffer);
            }
        }

        AURA_LOGD(m_ctx, AURA_TAG, "***********************************************\n");

        AURA_LOGD(m_ctx, AURA_TAG, "* total leak mem size: %.2f KB (%.4f MB)\n",
                  leak_mem_size / 1024.f, leak_mem_size / 1048576.f);

        m_mblk_map.clear();
    }

    if (!m_allocators.empty())
    {
        for (auto iter = m_allocators.begin(); iter != m_allocators.end(); ++iter)
        {
            if (iter->second)
            {
                delete iter->second;
                iter->second = DT_NULL;
            }
        }

        m_allocators.clear();
    }
}

DT_VOID* MemPool::Impl::Allocate(DT_S32 type, DT_S64 size, DT_S32 align,
                                 const DT_CHAR *file, const DT_CHAR *func, DT_S32 line)
{
    std::lock_guard<std::mutex> guard(m_lock);
    /// 我们进行内存分配的时候，会进行判断对应类型的内存分配器是否已经初始化完成，如果没有则直接返回NULL
    ///
    if (m_allocators.count(type))
    {
        /// 借助对应类型的内存分配器进行分配内存
        Buffer buffer = m_allocators[type]->Allocate(size, align);
        if (buffer.m_type != AURA_MEM_INVALID)
        {
            buffer.m_type = type;
            m_total_mem_size += buffer.m_size;
            m_peak_mem_size  = Max(m_peak_mem_size, m_total_mem_size);
            m_mblk_map.emplace(reinterpret_cast<DT_UPTR_T>(buffer.m_origin), MemBlkData(file, func, line, buffer, DT_TRUE));

            if (m_mtrace_enable && m_mtrace_match)
            {
                std::lock_guard<std::mutex> stat_guard(m_mtrace_lock);

                for (auto it = m_mtrace_running.begin(); it != m_mtrace_running.end(); ++it)
                {
                    {
                        MemTraceData &data = it->second;
                        data.alloc_cnt++;
                        data.cur_size += buffer.m_size;
                        data.max_size = Max(data.max_size, data.cur_size);
                    }
                }
            }
        }

        return buffer.m_data;
    }
    else
    {
        return DT_NULL;
    }
}

Status MemPool::Impl::Free(DT_VOID *ptr)
{
    std::lock_guard<std::mutex> guard(m_lock);

    Status status = Status::OK;

    DT_UPTR_T addr = reinterpret_cast<DT_UPTR_T>(ptr);

    if (m_mblk_map.count(addr))
    {
        Buffer &buffer = m_mblk_map.at(addr).buffer;

        if (m_allocators.count(buffer.m_type))
        {
            m_total_mem_size -= buffer.m_size;

            if (m_mtrace_enable && m_mtrace_match)
            {
                std::lock_guard<std::mutex> stat_guard(m_mtrace_lock);
                for (auto it = m_mtrace_running.begin(); it != m_mtrace_running.end(); ++it)
                {
                    MemTraceData &data = it->second;
                    data.cur_size -= buffer.m_size;
                }
            }

            m_allocators[buffer.m_type]->Free(buffer);

            if (buffer.IsValid())
            {
                AURA_ADD_ERROR_STRING(m_ctx, "buffer free failed");
            }
            else
            {
                m_mblk_map.erase(addr);
            }
        }
        else
        {
            status = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "Unsupport Mem type for allocator");
        }
    }
    else
    {
        status = Status::ERROR;
        AURA_ADD_ERROR_STRING(m_ctx, "Unrecorded memory information");
    }

    return status;
}

Status MemPool::Impl::Map(const Buffer& buffer)
{
    std::lock_guard<std::mutex> guard(m_lock);

    Status status = Status::OK;
    if (m_allocators.count(buffer.m_type))
    {
        DT_UPTR_T addr = reinterpret_cast<DT_UPTR_T>(buffer.m_origin);

        if (m_mblk_map.count(addr))
        {
            m_allocators[buffer.m_type]->Map(buffer);
            m_mblk_map.at(addr).is_mapped = DT_TRUE;
        }
        else
        {
            status = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "Unrecorded memory information");
        }
    }
    else
    {
        status = Status::ERROR;
        AURA_ADD_ERROR_STRING(m_ctx, "Unsupport Mem type for allocator");
    }

    return status;
}

Status MemPool::Impl::Unmap(const Buffer& buffer)
{
    std::lock_guard<std::mutex> guard(m_lock);

    Status status = Status::OK;

    if (m_allocators.count(buffer.m_type))
    {
        DT_UPTR_T addr = reinterpret_cast<DT_UPTR_T>(buffer.m_origin);

        if (m_mblk_map.count(addr))
        {
            m_allocators[buffer.m_type]->Unmap(buffer);
            m_mblk_map.at(addr).is_mapped = DT_FALSE;
        }
        else
        {
            status = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "Unrecorded memory information");
        }
    }
    else
    {
        status = Status::ERROR;
        AURA_ADD_ERROR_STRING(m_ctx, "Unsupport Mem type for allocator");
    }

    return status;
}

Buffer MemPool::Impl::GetBuffer(DT_VOID *ptr)
{
    std::lock_guard<std::mutex> guard(m_lock);

    if (ptr)
    {
        DT_UPTR_T addr = reinterpret_cast<DT_UPTR_T>(ptr);
        if (m_mblk_map.count(addr))
        {
            Buffer &buffer = m_mblk_map.at(addr).buffer;
            return buffer;
        }
        else
        {
            return Buffer();
        }
    }
    else
    {
        return Buffer();
    }
}

Status MemPool::Impl::RegisterAllocator(DT_S32 type, Allocator *allocator)
{
    Status status = Status::ERROR;
    if (DT_NULL == allocator)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Allocator create failed");
        return status;
    }

    std::lock_guard<std::mutex> guard(m_lock);

    if (m_allocators.count(type))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Allocator type already exists");
    }
    else
    {
        m_allocators.emplace(type, allocator);
        status = Status::OK;
    }

    return status;
}

Status MemPool::Impl::UnregisterAllocator(DT_S32 type)
{
    std::lock_guard<std::mutex> guard(m_lock);

    Status status = Status::ERROR;

    auto iter = m_allocators.find(type);
    if(iter != m_allocators.end())
    {
        // AURA_LOGI(m_ctx, AURA_TAG, "Allocator (%s) destroy\n", iter->second->m_name.c_str());
        if (iter->second)
        {
            delete iter->second;
        }
        iter = m_allocators.erase(iter);
        status = Status::OK;
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Allocator type not exists");
    }

    return status;
}

Allocator* MemPool::Impl::GetAllocator(DT_S32 type)
{
    std::lock_guard<std::mutex> guard(m_lock);

    auto iter = m_allocators.find(type);
    if(iter != m_allocators.end())
    {
        // AURA_LOGI(m_ctx, AURA_TAG, "Allocator (%s) destroy\n", iter->second->m_name.c_str());
        if (iter->second)
        {
            return iter->second;
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Allocator type not exists");
    }

    return DT_NULL;
}

DT_VOID MemPool::Impl::MemTraceSet(DT_BOOL flag)
{
    if (m_mtrace_enable == flag)
    {
        return;
    }

    MemTraceClear();
    m_mtrace_enable = flag;     // set enable flag
}

DT_VOID MemPool::Impl::MemTraceClear()
{
    std::lock_guard<std::mutex> guard(m_mtrace_lock);

    std::deque<std::pair<std::string, MemTraceData>>().swap(m_mtrace_running);   // empty running   data
    std::deque<std::pair<std::string, MemTraceData>>().swap(m_mtrace_completed); // empty completed data
    m_mtrace_dismatch_string.clear(); // clear dismatch info
    m_mtrace_match  = DT_TRUE; // reset match flag
}

DT_VOID MemPool::Impl::MemTraceBegin(const std::string &tag)
{
    if (m_mtrace_enable && m_mtrace_match)
    {
        std::lock_guard<std::mutex> guard(m_mtrace_lock);

        MemTraceData data(m_mtrace_running.size(), 0, m_total_mem_size, m_total_mem_size);
        m_mtrace_running.emplace_back(std::pair<std::string, MemTraceData>(tag, data));
    }
}

DT_VOID MemPool::Impl::MemTraceEnd(const std::string &tag)
{
    if (m_mtrace_enable && m_mtrace_match)
    {
        std::lock_guard<std::mutex> guard(m_mtrace_lock);

        // if queue is empty
        if (m_mtrace_running.empty())
        {
            m_mtrace_dismatch_string = "begin tag miss: " + tag;
            m_mtrace_match = DT_FALSE;
            return;
        }

        // if tag doesn't match
        if (m_mtrace_running.back().first != tag)
        {
            m_mtrace_dismatch_string = "tag dismatch: begin tag: " + m_mtrace_running.back().first + ", end tag: " + tag;
            m_mtrace_match = DT_FALSE;
            return;
        }

        // if tag match
        m_mtrace_completed.emplace_back(m_mtrace_running.back());
        m_mtrace_running.pop_back();
    }
}

std::string MemPool::Impl::MemTraceReport()
{
    if (!m_mtrace_match)
    {
        return m_mtrace_dismatch_string;
    }

    if (m_mtrace_completed.empty())
    {
        return std::string("Nothing recorded");
    }

    // resort by level
    for (auto it = m_mtrace_completed.end() - 1; it != m_mtrace_completed.begin();)
    {
        auto it_inner = it;
        for (; it_inner != m_mtrace_completed.begin(); --it_inner)
        {
            // if current level if smaller than previous level, swap
            if (it_inner->second.level < (it_inner - 1)->second.level)
            {
                std::swap(*it_inner, *(it_inner - 1));
            }
            else
            {
                break; // find the right position
            }
        }

        // if nothing happened, --it
        if (it == it_inner)
        {
            --it;
        }
    }

    std::stringstream sstream;
    sstream << "******************************** Mem Statistic Report ********************************\n";
    for (auto it = m_mtrace_completed.begin(); it != m_mtrace_completed.end(); ++it)
    {
        std::string  &tag  = it->first;
        MemTraceData &data = it->second;

        DT_CHAR tag_buffer[256]     = {0};
        DT_CHAR mem_str_buffer[512] = {0};
        sprintf(tag_buffer, "%s[%s]", std::string(data.level * 4, ' ').c_str(), tag.c_str());
        sprintf(mem_str_buffer, "%-50s count: %-3zu peak: %.2f KB(%.4f MB)", tag_buffer, data.alloc_cnt,
                                                                             data.max_size / 1024.0f,
                                                                             data.max_size / 1048576.0f);

        sstream << mem_str_buffer << "\n";
    }
    sstream << "**************************************************************************************";

    return sstream.str();
}

MemPool::MemPool(Context *ctx) : m_impl(new MemPool::Impl(ctx))
{}

MemPool::~MemPool()
{
    if (m_impl)
    {
        delete m_impl;
        m_impl = DT_NULL;
    }
}

DT_VOID* MemPool::Allocate(DT_S32 type, DT_S64 size, DT_S32 align,
                           const DT_CHAR *file, const DT_CHAR *func, DT_S32 line)
{
    /// 根据对应类型。来进行内存分配。
    return m_impl->Allocate(type, size, align, file, func, line);
}

Status MemPool::Free(DT_VOID *ptr)
{
    return m_impl->Free(ptr);
}

Status MemPool::Map(const Buffer& buffer)
{
    return m_impl->Map(buffer);
}

Status MemPool::Unmap(const Buffer& buffer)
{
    return m_impl->Unmap(buffer);
}

Buffer MemPool::GetBuffer(DT_VOID *ptr)
{
    return m_impl->GetBuffer(ptr);
}

Status MemPool::RegisterAllocator(DT_S32 type, Allocator *allocator)
{
    return m_impl->RegisterAllocator(type, allocator);
}

Status MemPool::UnregisterAllocator(DT_S32 type)
{
    return m_impl->UnregisterAllocator(type);
}

Allocator* MemPool::GetAllocator(DT_S32 type)
{
    return m_impl->GetAllocator(type);
}

DT_VOID MemPool::MemTraceSet(DT_BOOL enable)
{
    m_impl->MemTraceSet(enable);
}

DT_VOID MemPool::MemTraceBegin(const std::string &tag)
{
    m_impl->MemTraceBegin(tag);
}

DT_VOID MemPool::MemTraceEnd(const std::string &tag)
{
    m_impl->MemTraceEnd(tag);
}

DT_VOID MemPool::MemTraceClear()
{
    m_impl->MemTraceClear();
}

std::string MemPool::MemTraceReport()
{
    return m_impl->MemTraceReport();
}

} // namespace aura