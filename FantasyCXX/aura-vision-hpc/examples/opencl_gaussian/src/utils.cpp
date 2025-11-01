
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dlfcn.h>
#include "stdio.h"
#include "utils.h"

#include <cstdlib>

#ifdef ANDROID

#ifdef MTK
#include "mtk_ion.h"
#else
#include "msm_ion.h"
#endif

#include "BufferAllocator/BufferAllocatorWrapper.h"
#include "ion.h"

#define MIALGO_MEM_CACHED                   (1)     /*!< cached memory flag */

// dma_buf_heap function pointer
typedef BufferAllocator* (*PfCreateDmabufFunc)();

typedef void (*PfFreeDmabufFunc)(BufferAllocator*);

typedef int (*PfDmabufAllocFunc)(BufferAllocator*, const char*, size_t, unsigned int, size_t);

typedef int (*PfDmabufMapHeapNameFunc)(BufferAllocator* , const char* , const char*, unsigned int, unsigned int, unsigned int);

// ion function pointer
typedef int (*PfIonOpenFunc)();

typedef int (*PfIonCloseFunc)(int);

typedef int (*PfIonAllocFdFunc)(int, size_t, size_t, unsigned int, unsigned int, int*);

typedef struct 
{
    void *so_handle;
    int  ion_type;

    PfCreateDmabufFunc     func_dma_buf_create;
    PfFreeDmabufFunc   func_dma_buf_free;
    PfDmabufAllocFunc         func_dma_buf_alloc;
    PfDmabufMapHeapNameFunc   func_dma_buf_map;

    PfIonOpenFunc       func_ion_open;
    PfIonCloseFunc      func_ion_close;
    PfIonAllocFdFunc    func_ion_alloc_fd;
} MemIonLibFunc;

#define MEM_ION_TYPE     (0)
#define MEM_DMA_BUF_TYPE (1)
#define MEM_INVALID_TYPE (-1)

static MemIonLibFunc g_lib_funcs;

static int InitLibIonSo()
{
    int ret = 0;
    if (NULL != (g_lib_funcs.so_handle = dlopen("libdmabufheap.so", RTLD_LAZY)))
    {
        g_lib_funcs.ion_type        = MEM_DMA_BUF_TYPE;
        g_lib_funcs.func_dma_buf_create = (PfCreateDmabufFunc) dlsym(g_lib_funcs.so_handle, "CreateDmabufHeapBufferAllocator");
        g_lib_funcs.func_dma_buf_free   = (PfFreeDmabufFunc) dlsym(g_lib_funcs.so_handle, "FreeDmabufHeapBufferAllocator");
        g_lib_funcs.func_dma_buf_map    = (PfDmabufMapHeapNameFunc) dlsym(g_lib_funcs.so_handle, "MapDmabufHeapNameToIonHeap");
        g_lib_funcs.func_dma_buf_alloc  = (PfDmabufAllocFunc) dlsym(g_lib_funcs.so_handle, "DmabufHeapAlloc");

        if ((NULL == g_lib_funcs.func_dma_buf_create) || (NULL == g_lib_funcs.func_dma_buf_free) || 
            (NULL == g_lib_funcs.func_dma_buf_map) || (NULL == g_lib_funcs.func_dma_buf_alloc))
        {
            printf("cannot dlsym libdmabufheap functions\n");
            ret = -1;
        }

        return ret;
    }

    if (NULL != (g_lib_funcs.so_handle = dlopen("libion.so", RTLD_LAZY)))
    {
        g_lib_funcs.ion_type      = MEM_ION_TYPE;
        g_lib_funcs.func_ion_open     = (PfIonOpenFunc) dlsym(g_lib_funcs.so_handle, "ion_open");
        g_lib_funcs.func_ion_close    = (PfIonCloseFunc) dlsym(g_lib_funcs.so_handle, "ion_close");
        g_lib_funcs.func_ion_alloc_fd = (PfIonAllocFdFunc) dlsym(g_lib_funcs.so_handle, "ion_alloc_fd");

        if ((NULL == g_lib_funcs.func_ion_open) || (NULL == g_lib_funcs.func_ion_close) || (NULL == g_lib_funcs.func_ion_alloc_fd))
        {
            printf("cannot dlsym libion functions\n");
            ret = -1;
        }
    }
    else
    {
        g_lib_funcs.ion_type = -1;
        ret = -1;
    }

    return ret;
}

#endif

int UtilsInitIon(long long *ion_dev)
{
    #ifdef ANDROID
    if (0 != InitLibIonSo())
    {
        printf("can init libion/libdmabufheap\n");
        return -1;
    }

    if (MEM_ION_TYPE == g_lib_funcs.ion_type)
    {
        int ion_dev_fd = -1;

        if ((ion_dev_fd = g_lib_funcs.func_ion_open()) < 0)
        {
            printf("ion_open fail\n");
            return -1;
        }

        *ion_dev = (long long)ion_dev_fd;
        return MEM_ION_TYPE;
    }
    else if (MEM_DMA_BUF_TYPE == g_lib_funcs.ion_type)
    {
        BufferAllocator *bufferAllocator = NULL;

        if (NULL == (bufferAllocator = g_lib_funcs.func_dma_buf_create()))
        {
            printf("CreateDmabufHeapBufferAllocator fail\n");
            return -1;
        }

        if (g_lib_funcs.func_dma_buf_map(bufferAllocator, "system", "", 0, ~0, 0) < 0)
        {
            printf("MapDmabufHeapNameToIonHeap fail\n");
            return -1;
        }

        *ion_dev = (long long)bufferAllocator;
   
        return MEM_DMA_BUF_TYPE;
    }

    return -1;
    #else

    *ion_dev = 1;
    return 0;
    #endif
}

int UtilsUnitIon(long long ion_dev_fd)
{
    #ifdef ANDROID
    if (MEM_ION_TYPE == g_lib_funcs.ion_type)
    {
        if (ion_dev_fd >= 0)
        {
            g_lib_funcs.func_ion_close((int)ion_dev_fd);
        }
    }
    else if (MEM_DMA_BUF_TYPE == g_lib_funcs.ion_type)
    {
        if (NULL != (BufferAllocator *)ion_dev_fd)
        {
            g_lib_funcs.func_dma_buf_free((BufferAllocator *)ion_dev_fd);
        }
    }

    if (NULL != g_lib_funcs.so_handle)
    {
        dlclose(g_lib_funcs.so_handle);
    }

    #else
    (void)(ion_dev_fd);
    #endif

    return 0;
}

int AllocIonBuffer(long long ion_dev_fd, int mem_size, Meminfo *mem_buffer)
{
    #ifdef ANDROID
    if ((NULL == mem_buffer) ||
        ((MEM_DMA_BUF_TYPE == g_lib_funcs.ion_type) && (NULL == (BufferAllocator *)ion_dev_fd)) ||
        ((MEM_ION_TYPE == g_lib_funcs.ion_type) && (ion_dev_fd < 0)))
    {
        printf("input param is invalid...\n");
        return -1;
    }

    int mem_fd = 0;
    unsigned char *host_addr = NULL;
    int align = 4096;

    if (MEM_DMA_BUF_TYPE == g_lib_funcs.ion_type)
    {
        mem_fd = g_lib_funcs.func_dma_buf_alloc((BufferAllocator *)ion_dev_fd, "system", mem_size, MIALGO_MEM_CACHED, align);
        if (mem_fd < 0)
        {
            printf("%s %d dma_buf_alloc_fd fail\n", __FILE__, __LINE__);
            return -1;
        }
    }
    else if (MEM_ION_TYPE == g_lib_funcs.ion_type)
    {
#ifdef QCOM
        if (g_lib_funcs.func_ion_alloc_fd((int)ion_dev_fd, mem_size, align, ION_HEAP(ION_SYSTEM_HEAP_ID), MIALGO_MEM_CACHED, &mem_fd))
#else
        if (g_lib_funcs.func_ion_alloc_fd((int)ion_dev_fd, mem_size, align, ION_HEAP_MULTIMEDIA_MASK, MIALGO_MEM_CACHED, &mem_fd))
#endif
        {
            printf("%s %d ion_alloc_fd fail\n", __FILE__, __LINE__);
            return -1;
        }
    }

    host_addr = (unsigned char *)mmap(NULL, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0);
    if (MAP_FAILED == host_addr)
    {
        goto ERROR;
    }

    mem_buffer->addr = host_addr;
    mem_buffer->size = mem_size;
    mem_buffer->fd = mem_fd;

    printf("\033[1;32mAlloc Mem addr=%p size=%d fd=%d\n\033[0m", host_addr, mem_size, mem_fd);

    return 0;
ERROR:
    if (mem_fd > 0)
    {
        close(mem_fd);
    }

    return -1;

    #else
    (void)(ion_dev_fd);
    mem_buffer->addr = (unsigned char *)calloc(mem_size, 1);
    mem_buffer->fd = 1;

    if (NULL == mem_buffer->addr)
    {
        printf("%s %d calloc fail\n", __FILE__, __LINE__);
        return -1;
    }
    return 0;
    #endif 
}

int AllocBuffer(long long ion_dev_fd, int mem_size, Meminfo *mem_buffer)
{
    if (NULL == mem_buffer)
    {
        printf("input param is invalid...\n");
        return -1;
    }

    return AllocIonBuffer(ion_dev_fd, mem_size, mem_buffer);
}

int DeleteBuffer(Meminfo *mem_buffer)
{
    if (NULL == mem_buffer)
        return 0;
#ifdef ANDROID
    munmap(mem_buffer->addr, mem_buffer->size);
    close(mem_buffer->fd);
#else
    free(mem_buffer->addr);
#endif

    printf("\033[1;32mDelete Mem addr=%p size=%d fd=%d\033[0m\n", mem_buffer->addr, mem_buffer->size, mem_buffer->fd);
    return 0;
}
