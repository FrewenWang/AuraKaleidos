#ifndef AURA_RUNTIME_MEMORY_ALLOCATOR_DMA_BUF_HEAP_HPP__
#define AURA_RUNTIME_MEMORY_ALLOCATOR_DMA_BUF_HEAP_HPP__

#include "aura/runtime/memory/allocator.hpp"

#include "BufferAllocator/BufferAllocatorWrapper.h"

namespace aura
{

Allocator* CreateDMABufHeapAllocator();

class AllocatorDMABufHeap : public Allocator
{
public:
    AllocatorDMABufHeap();
    ~AllocatorDMABufHeap();

    Buffer Allocate(DT_S64 size, DT_S32 align = 0) override;
    DT_VOID Free(Buffer &buffer) override;

    Status Map(const Buffer &buffer) override;
    Status Unmap(const Buffer &buffer) override;

    DT_BOOL IsValid() const;

private:
    using InitFunc      = BufferAllocator* (*)();
    using DeinitFunc    = void (*)(BufferAllocator*);
    using AllocFunc     = int (*)(BufferAllocator*, const char*, size_t, unsigned int, size_t);
    using Map2IonFunc   = int (*)(BufferAllocator*, const char*, const char*, unsigned int, unsigned int, unsigned);
    using SyncStartFunc = int (*)(BufferAllocator*, unsigned int, SyncType, int (*legacy_ion_cpu_sync)(int, int, void *), void *);
    using SyncEndFunc   = int (*)(BufferAllocator*, unsigned int, SyncType, int (*legacy_ion_cpu_sync)(int, int, void *), void *);

    DT_BOOL         m_is_valid;
    DT_VOID         *m_dl_handle;
    BufferAllocator *m_buffer_allocator;

    InitFunc      m_create_func;
    DeinitFunc    m_free_func;
    AllocFunc     m_alloc_func;
    Map2IonFunc   m_map2ion_func;
    SyncStartFunc m_sync_start_func;
    SyncEndFunc   m_sync_end_func;
};

class AllocatorIonQualcomm : public Allocator
{
public:
    AllocatorIonQualcomm();
    ~AllocatorIonQualcomm();

    Buffer Allocate(DT_S64 size, DT_S32 align = 0) override;
    DT_VOID Free(Buffer &buffer) override;

    Status Map(const Buffer &buffer) override;
    Status Unmap(const Buffer &buffer) override;

    DT_BOOL IsValid() const;

private:
    using IonOpenFunc  = int (*)();
    using IonCloseFunc = int (*)(int);
    using IonAllocFunc = int (*)(int, size_t, size_t, unsigned int, unsigned int, int*);

    DT_BOOL      m_is_valid;
    DT_VOID      *m_dl_handle;
    DT_S32       m_ion_fd;

    IonOpenFunc  m_ion_open_func;
    IonCloseFunc m_ion_close_func;
    IonAllocFunc m_ion_alloc_func;
};

class AllocatorIonMTK : public Allocator
{
public:
    AllocatorIonMTK();
    ~AllocatorIonMTK();

    Buffer Allocate(DT_S64 size, DT_S32 align = 0) override;
    DT_VOID Free(Buffer &buffer) override;

    Status Map(const Buffer &buffer) override;
    Status Unmap(const Buffer &buffer) override;

    DT_BOOL IsValid() const;

private:
    // libion_mtk/ion.h
    typedef int ion_user_handle_t;

    // Function from libion_mtk.so
    using MtIonOpenFunc      = int (*)(const char *);
    using IonAllocMmFunc     = int (*)(int fd, size_t len, size_t align, unsigned int flags, ion_user_handle_t *handle);
    using IonMmapFunc        = void* (*)(int fd, void *addr, size_t length, int prot, int flags, int share_fd, off_t offset);
    using IonImportFunc      = int (*)(int fd, int share_fd, ion_user_handle_t *handle);
    using IonMunmapFunc      = int (*)(int fd, void *addr, size_t length);
    using IonShareCloseFunc  = int (*)(int fd, int share_fd);
    using IonCustomIoctlFunc = int (*)(int fd, unsigned int cmd, void *arg);

    // Function from libion.so
    using IonCloseFunc = int (*)(int fd);
    using IonShareFunc = int (*)(int fd, ion_user_handle_t handle, int *share_fd);
    using IonFreeFunc  = int (*)(int fd, ion_user_handle_t handle);

    DT_BOOL            m_is_valid;
    DT_VOID            *m_dl_handle;
    DT_VOID            *m_mtk_dl_handle;
    DT_S32             m_ion_fd;

    MtIonOpenFunc      m_mt_ion_open_func;
    IonAllocMmFunc     m_ion_alloc_mm_func;
    IonMmapFunc        m_ion_mmap_func;
    IonImportFunc      m_ion_import_func;
    IonMunmapFunc      m_ion_munmap_func;
    IonShareCloseFunc  m_ion_share_close_func;
    IonCustomIoctlFunc m_ion_custom_ioctl_func;

    IonCloseFunc       m_ion_close_func;
    IonShareFunc       m_ion_shared_func;
    IonFreeFunc        m_ion_free_func;
};

} // namespace aura

#endif // AURA_RUNTIME_MEMORY_ALLOCATOR_DMA_BUF_HEAP_HPP__