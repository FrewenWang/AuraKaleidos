#include "host/android/allocator_dma_buf_heap.hpp"

#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "IonAllocator/mtk/libion_mtk/ion.h"
#include "IonAllocator/mtk/linux/ion_drv.h"
#include "IonAllocator/qualcomm/msm_ion.h"

#define DMA_BUF_HEAP_ADDR_ALIGN      (4096)

namespace aura
{

Allocator* CreateDMABufHeapAllocator()
{
    // Create flow: (dma buf heap) -> (mtk ion) -> (qualcomm ion)

    // create DMABufHeap
    {
        AllocatorDMABufHeap *allocator = new AllocatorDMABufHeap();
        if (allocator && allocator->IsValid())
        {
            return allocator;
        }
        delete allocator;
    }

    // create mtk ion
    {
        AllocatorIonMTK *allocator = new AllocatorIonMTK();
        if (allocator && allocator->IsValid())
        {
            return allocator;
        }
        delete allocator;
    }

    // create qualcomm ion
    {
        AllocatorIonQualcomm *allocator = new AllocatorIonQualcomm();
        if (allocator && allocator->IsValid())
        {
            return allocator;
        }
        delete allocator;
    }

    return MI_NULL;
}

/////////////////////////////////////////AllocatorDMABufHeap//////////////////////////////////////////////////

AllocatorDMABufHeap::AllocatorDMABufHeap()
                                         : Allocator(AURA_MEM_DMA_BUF_HEAP, "DMABufHeap"),
                                           m_is_valid(MI_FALSE),
                                           m_dl_handle(MI_NULL),
                                           m_buffer_allocator(MI_NULL),
                                           m_create_func(MI_NULL),
                                           m_free_func(MI_NULL),
                                           m_alloc_func(MI_NULL),
                                           m_map2ion_func(MI_NULL)
{
    m_dl_handle = dlopen("libdmabufheap.so", RTLD_LAZY | RTLD_LOCAL);
    dlerror();     /* Clear any existing error */

    if (m_dl_handle != MI_NULL)
    {
        m_create_func     = (InitFunc     )(dlsym(m_dl_handle, "CreateDmabufHeapBufferAllocator"));
        m_free_func       = (DeinitFunc   )(dlsym(m_dl_handle, "FreeDmabufHeapBufferAllocator"));
        m_alloc_func      = (AllocFunc    )(dlsym(m_dl_handle, "DmabufHeapAlloc"));
        m_map2ion_func    = (Map2IonFunc  )(dlsym(m_dl_handle, "MapDmabufHeapNameToIonHeap"));
        m_sync_start_func = (SyncStartFunc)(dlsym(m_dl_handle, "DmabufHeapCpuSyncStart"));
        m_sync_end_func   = (SyncEndFunc  )(dlsym(m_dl_handle, "DmabufHeapCpuSyncEnd"));

        if (dlerror() == MI_NULL)
        {
            m_buffer_allocator = m_create_func();
            if (m_buffer_allocator != MI_NULL)
            {
                if (m_map2ion_func(m_buffer_allocator, "system", "", 0, ~0, 0) < 0)
                {
                    m_free_func(m_buffer_allocator);
                    m_buffer_allocator = MI_NULL;
                }
                else
                {
                    m_is_valid = MI_TRUE;
                }
            }
        }
    }
}

AllocatorDMABufHeap::~AllocatorDMABufHeap()
{
    m_is_valid = MI_FALSE;

    if (m_buffer_allocator != MI_NULL)
    {
        m_free_func(m_buffer_allocator);
        m_buffer_allocator = MI_NULL;
    }

    if (m_dl_handle != MI_NULL)
    {
        dlclose(m_dl_handle);
        m_dl_handle = MI_NULL;
    }
}

MI_BOOL AllocatorDMABufHeap::IsValid() const
{
    return m_is_valid;
}

Buffer AllocatorDMABufHeap::Allocate(MI_S64 size, MI_S32 align)
{
    if (IsValid())
    {
        align = (align <= 0) ? DMA_BUF_HEAP_ADDR_ALIGN : align;
        MI_S32 handle = m_alloc_func(m_buffer_allocator, "system", size, 1, align);
        AURA_VOID *host_addr = mmap(MI_NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, handle, 0);
        if (MAP_FAILED == host_addr)
        {
            close(handle);
            return Buffer();
        }
        return Buffer(AURA_MEM_DMA_BUF_HEAP, size, size, host_addr, host_addr, handle);
    }
    return Buffer();
}

AURA_VOID AllocatorDMABufHeap::Free(Buffer &buffer)
{
    if (AURA_MEM_DMA_BUF_HEAP == buffer.m_type && buffer.m_origin != MI_NULL)
    {
        munmap(buffer.m_origin, buffer.m_capacity);
        close(buffer.m_property);
        buffer.Clear();
    }
}

Status AllocatorDMABufHeap::Map(const Buffer &buffer)
{
    if (AURA_MEM_DMA_BUF_HEAP == buffer.m_type && buffer.m_origin != MI_NULL)
    {
        MI_S32 ret = m_sync_start_func(m_buffer_allocator, buffer.m_property, kSyncReadWrite, MI_NULL, MI_NULL);
        return (0 == ret) ? Status::OK : Status::ERROR;
    }
    return Status::ERROR;
}

Status AllocatorDMABufHeap::Unmap(const Buffer &buffer)
{
    if (AURA_MEM_DMA_BUF_HEAP == buffer.m_type && buffer.m_origin != MI_NULL)
    {
        MI_S32 ret = m_sync_end_func(m_buffer_allocator, buffer.m_property, kSyncReadWrite, MI_NULL, MI_NULL);
        return (0 == ret) ? Status::OK : Status::ERROR;
    }
    return Status::ERROR;
}

/////////////////////////////////////////AllocatorIonMTK//////////////////////////////////////////////////

AllocatorIonMTK::AllocatorIonMTK()
                                 : Allocator(AURA_MEM_DMA_BUF_HEAP, "Mtk Ion"),
                                   m_is_valid(MI_FALSE), m_dl_handle(MI_NULL),
                                   m_mtk_dl_handle(MI_NULL), m_ion_fd(-1),
                                   m_mt_ion_open_func(MI_NULL), m_ion_alloc_mm_func(MI_NULL),
                                   m_ion_mmap_func(MI_NULL), m_ion_import_func(MI_NULL),
                                   m_ion_munmap_func(MI_NULL), m_ion_share_close_func(MI_NULL),
                                   m_ion_custom_ioctl_func(MI_NULL), m_ion_close_func(MI_NULL),
                                   m_ion_shared_func(MI_NULL), m_ion_free_func(MI_NULL)
{
    MI_BOOL loaded_mt_ion  = MI_FALSE;
    MI_BOOL loaded_libion  = MI_FALSE;

    // load symbol from libion_mtk.so
    m_mtk_dl_handle = dlopen("libion_mtk.so", RTLD_LAZY | RTLD_LOCAL);
    dlerror();     /* Clear any existing error */

    if (m_mtk_dl_handle != MI_NULL)
    {
        m_mt_ion_open_func      = (MtIonOpenFunc     )(dlsym(m_mtk_dl_handle, "mt_ion_open"));
        m_ion_alloc_mm_func     = (IonAllocMmFunc    )(dlsym(m_mtk_dl_handle, "ion_alloc_mm"));
        m_ion_mmap_func         = (IonMmapFunc       )(dlsym(m_mtk_dl_handle, "ion_mmap"));
        m_ion_import_func       = (IonImportFunc     )(dlsym(m_mtk_dl_handle, "ion_import"));
        m_ion_munmap_func       = (IonMunmapFunc     )(dlsym(m_mtk_dl_handle, "ion_munmap"));
        m_ion_share_close_func  = (IonShareCloseFunc )(dlsym(m_mtk_dl_handle, "ion_share_close"));
        m_ion_custom_ioctl_func = (IonCustomIoctlFunc)(dlsym(m_mtk_dl_handle, "ion_custom_ioctl"));

        loaded_mt_ion = (dlerror() == MI_NULL);
    }

    // load symbol from libion.so
    if (loaded_mt_ion)
    {
        m_dl_handle = dlopen("libion.so", RTLD_LAZY | RTLD_LOCAL);
        dlerror();     /* Clear any existing error */

        if (m_dl_handle != MI_NULL)
        {
            m_ion_close_func  = (IonCloseFunc)(dlsym(m_dl_handle, "ion_close"));
            m_ion_shared_func = (IonShareFunc)(dlsym(m_dl_handle, "ion_share"));
            m_ion_free_func   = (IonFreeFunc )(dlsym(m_dl_handle, "ion_free"));
        }

        loaded_libion = (dlerror() == MI_NULL);
    }

    if (loaded_mt_ion && loaded_libion)
    {
        m_ion_fd = m_mt_ion_open_func(__FUNCTION__);
        m_is_valid = (m_ion_fd >= 0);
    }
}

AllocatorIonMTK::~AllocatorIonMTK()
{
    m_is_valid = MI_FALSE;

    if (m_ion_fd >= 0)
    {
        m_ion_close_func(m_ion_fd);
        m_ion_fd = -1;
    }

    if (m_dl_handle != MI_NULL)
    {
        dlclose(m_dl_handle);
        m_dl_handle = MI_NULL;
    }

    if (m_mtk_dl_handle != MI_NULL)
    {
        dlclose(m_mtk_dl_handle);
        m_mtk_dl_handle = MI_NULL;
    }
}

MI_BOOL AllocatorIonMTK::IsValid() const
{
    return m_is_valid;
}

Buffer AllocatorIonMTK::Allocate(MI_S64 size, MI_S32 align)
{
    if (IsValid())
    {
        align = (align <= 0) ? DMA_BUF_HEAP_ADDR_ALIGN : align;

        MI_S32 buffer_handle = 0;
        if (m_ion_alloc_mm_func(m_ion_fd, size, align, ION_FLAG_CACHED | ION_FLAG_CACHED_NEEDS_SYNC, &buffer_handle))
        {
            return Buffer();
        }

        MI_S32 handle = 0;
        if (m_ion_shared_func(m_ion_fd, buffer_handle, &handle))
        {
            m_ion_free_func(m_ion_fd, buffer_handle);
            return Buffer();
        }

        AURA_VOID *host_addr = m_ion_mmap_func(m_ion_fd, MI_NULL, size,
                                             PROT_READ | PROT_WRITE,
                                             MAP_SHARED, handle, 0);
        if (MAP_FAILED == host_addr)
        {
            if (m_ion_share_close_func(m_ion_fd, handle) == 0)
            {
                m_ion_free_func(m_ion_fd, buffer_handle);
            }
            return Buffer();
        }
        return Buffer(AURA_MEM_DMA_BUF_HEAP, size, size, host_addr, host_addr, handle);
    }
    return Buffer();
}

AURA_VOID AllocatorIonMTK::Free(Buffer &buffer)
{
    if (AURA_MEM_DMA_BUF_HEAP == buffer.m_type && buffer.m_origin != MI_NULL)
    {
        MI_BOOL free_success = MI_TRUE;
        ion_user_handle_t buffer_handle = 0;
        {
            // 1. Get buffer_handle by buf_share_fd
            if (m_ion_import_func(m_ion_fd, buffer.m_property, &buffer_handle))
            {
                free_success = MI_FALSE;
            }

            // 2. free import ref
            if (free_success && m_ion_free_func(m_ion_fd, buffer_handle))
            {
                free_success = MI_FALSE;
            }
        }

        // 1. unmap
        if (free_success && m_ion_munmap_func(m_ion_fd, buffer.m_origin, buffer.m_capacity))
        {
            free_success = MI_FALSE;
        }

        // 2. close share buffer fd
        if (free_success && m_ion_share_close_func(m_ion_fd, buffer.m_property))
        {
            free_success = MI_FALSE;
        }

        // 3. free ion mm
        if (free_success && m_ion_free_func(m_ion_fd, buffer_handle))
        {
            free_success = MI_FALSE;
        }

        if (free_success)
        {
            buffer.Clear();
        }
    }
}

Status AllocatorIonMTK::Map(const Buffer &buffer)
{
    if (AURA_MEM_DMA_BUF_HEAP == buffer.m_type && buffer.m_origin != MI_NULL)
    {
        struct ion_sys_data sys_data;
        ion_user_handle_t ion_user_handle;
        MI_S32 ret = m_ion_import_func(m_ion_fd, buffer.m_property, &ion_user_handle);

        if (0 == ret)
        {
            sys_data.sys_cmd = ION_SYS_CACHE_SYNC;
            sys_data.cache_sync_param.handle = ion_user_handle;
            sys_data.cache_sync_param.sync_type = ION_CACHE_INVALID_BY_RANGE;
            sys_data.cache_sync_param.va = buffer.m_data;
            sys_data.cache_sync_param.size = buffer.m_size;

            ret = m_ion_custom_ioctl_func(m_ion_fd, ION_CMD_SYSTEM, &sys_data);
        }

        return (0 == ret) ? Status::OK : Status::ERROR;
    }
    return Status::ERROR;
}

Status AllocatorIonMTK::Unmap(const Buffer &buffer)
{
    if (AURA_MEM_DMA_BUF_HEAP == buffer.m_type && buffer.m_origin != MI_NULL)
    {
        struct ion_sys_data sys_data;
        ion_user_handle_t ion_user_handle;
        MI_S32 ret = m_ion_import_func(m_ion_fd, buffer.m_property, &ion_user_handle);

        if (0 == ret)
        {
            sys_data.sys_cmd = ION_SYS_CACHE_SYNC;
            sys_data.cache_sync_param.handle = ion_user_handle;
            sys_data.cache_sync_param.sync_type = ION_CACHE_FLUSH_BY_RANGE;
            sys_data.cache_sync_param.va = buffer.m_data;
            sys_data.cache_sync_param.size = buffer.m_size;

            ret = m_ion_custom_ioctl_func(m_ion_fd, ION_CMD_SYSTEM, &sys_data);
        }

        return (0 == ret) ? Status::OK : Status::ERROR;
    }
    return Status::ERROR;
}

/////////////////////////////////////////AllocatorIonQualcomm//////////////////////////////////////////////////

AllocatorIonQualcomm::AllocatorIonQualcomm()
                                           : Allocator(AURA_MEM_DMA_BUF_HEAP, "Qcom Ion"),
                                             m_is_valid(MI_FALSE), m_dl_handle(MI_NULL),
                                             m_ion_fd(-1), m_ion_open_func(MI_NULL),
                                             m_ion_close_func(MI_NULL), m_ion_alloc_func(MI_NULL)
{
    m_dl_handle = dlopen("libion.so", RTLD_LAZY | RTLD_LOCAL);
    dlerror();     /* Clear any existing error */

    if (m_dl_handle != MI_NULL)
    {
        m_ion_open_func  = (IonOpenFunc )(dlsym(m_dl_handle, "ion_open"));
        m_ion_close_func = (IonCloseFunc)(dlsym(m_dl_handle, "ion_close"));
        m_ion_alloc_func = (IonAllocFunc)(dlsym(m_dl_handle, "ion_alloc_fd"));

        if (dlerror() == MI_NULL)
        {
            m_ion_fd   = m_ion_open_func();
            m_is_valid = (m_ion_fd >= 0);
        }
    }
}

AllocatorIonQualcomm::~AllocatorIonQualcomm()
{
    m_is_valid = MI_FALSE;

    if (m_ion_fd >= 0)
    {
        m_ion_close_func(m_ion_fd);
        m_ion_fd = -1;
    }

    if (m_dl_handle != MI_NULL)
    {
        dlclose(m_dl_handle);
        m_dl_handle = MI_NULL;
    }
}

MI_BOOL AllocatorIonQualcomm::IsValid() const
{
    return m_is_valid;
}

Buffer AllocatorIonQualcomm::Allocate(MI_S64 size, MI_S32 align)
{
    if (IsValid())
    {
        MI_S32 handle = 0;
        align = (align <= 0) ? DMA_BUF_HEAP_ADDR_ALIGN : align;

        if (m_ion_alloc_func(m_ion_fd, size, align, ION_HEAP(ION_SYSTEM_HEAP_ID), 1, &handle))
        {
            return Buffer();
        }
        AURA_VOID *host_addr = mmap(MI_NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, handle, 0);
        if (MAP_FAILED == host_addr)
        {
            close(handle);
            return Buffer();
        }
        return Buffer(AURA_MEM_DMA_BUF_HEAP, size, size, host_addr, host_addr, handle);
    }
    return Buffer();
}

AURA_VOID AllocatorIonQualcomm::Free(Buffer &buffer)
{
    if (AURA_MEM_DMA_BUF_HEAP == buffer.m_type && buffer.m_origin != MI_NULL)
    {
        munmap(buffer.m_origin, buffer.m_capacity);
        close(buffer.m_property);
        buffer.Clear();
    }
}

Status AllocatorIonQualcomm::Map(const Buffer &buffer)
{
    if (AURA_MEM_DMA_BUF_HEAP == buffer.m_type && buffer.m_origin != MI_NULL)
    {
        return Status::OK;
    }
    return Status::ERROR;
}

Status AllocatorIonQualcomm::Unmap(const Buffer &buffer)
{
    if (AURA_MEM_DMA_BUF_HEAP == buffer.m_type && buffer.m_origin != MI_NULL)
    {
        return Status::OK;
    }
    return Status::ERROR;
}

} // namespace aura