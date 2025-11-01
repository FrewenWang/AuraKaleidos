//--------------------------------------------------------------------------------------
// File: dmabuf_buffer_and_image.cpp
// Desc: Standalone sample showcasing how to use dmabuf-backed buffers and images.
//
// Author:      QUALCOMM
//
//               Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <string>

// DMABUF
#include <linux/dma-buf.h>
#include <BufferAllocator/BufferAllocator.h>
#include <BufferAllocator/BufferAllocatorWrapper.h>

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_ext_qcom.h>

#define CL_ASSERT_ERRCODE(errcode)                                                                  \
    if ((errcode) != CL_SUCCESS)                                                                    \
    {                                                                                               \
        std::cerr << "OpenCL API error code " << (errcode) << " on line " << __LINE__ << "\n";      \
        goto cleanup;                                                                               \
    }

static const char *PROGRAM_SOURCE = R"(
    __kernel void copybuf(__global char *src,
                       __global char *dst
                       )
    {
        uint wid_x = get_global_id(0);
        dst[wid_x] = src[wid_x];
    }
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    __kernel void copyimg(__read_only  image2d_t src,
                          __write_only image2d_t dst)
    {
        uint         wid_x = get_global_id(0);
        uint         wid_y = get_global_id(1);
        const float4 pixel = read_imagef(src, sampler, (int2)(wid_x, wid_y));
        write_imagef(dst, (int2)(wid_x, wid_y), pixel);
    }
)";

#define BUFFER_SIZE 1024
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 256

bool confirm_extension_support(cl_device_id  device, const char *extnStr)
{
    cl_int      errcode = CL_SUCCESS;
    char        extensions[2048] = {0};

    errcode = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), &extensions, nullptr);

    if(errcode != CL_SUCCESS)
    {
        std::cerr << "Error " << errcode << " with clGetDeviceInfo for extensions." << "\n";
        std::exit(errcode);
    }
    return (strstr(extensions, extnStr) != nullptr);
}

cl_mem_dmabuf_host_ptr make_dmabuf_buffer(BufferAllocator *allocator, cl_device_id device, size_t size, unsigned int dmabuf_allocation_flags, cl_uint host_cache_policy)
{
    int fd = DmabufHeapAlloc(allocator, "qcom,system", size, 0, 0);
    if(fd < 0)
    {
        std::cerr << "Error alocating dmabuf memory\n";
        std::exit(EXIT_FAILURE);
    }

    void *host_addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (MAP_FAILED == host_addr)
    {
        close(fd);
        std::cerr << "Error " << errno << " mmapping fd to pointer: " << strerror(errno) << "\n";
        std::exit(errno);
    }

    cl_mem_dmabuf_host_ptr dmabuf_mem;
    dmabuf_mem.ext_host_ptr.allocation_type   = CL_MEM_DMABUF_HOST_PTR_QCOM;
    dmabuf_mem.ext_host_ptr.host_cache_policy = host_cache_policy;
    dmabuf_mem.dmabuf_filedesc                = fd;
    dmabuf_mem.dmabuf_hostptr                 = host_addr;

    return dmabuf_mem;
}

int dmabuf_buffer()
{
    cl_int                 errcode           = 0;
    cl_device_id           device            = nullptr;
    cl_platform_id         platform          = nullptr;
    cl_context             context           = nullptr;
    cl_command_queue       queue             = nullptr;
    cl_program             program           = nullptr;
    cl_kernel              kernel            = nullptr;
    BufferAllocator        *buffer_allocator = nullptr;
    std::vector<int>       fds               = {};
    char                   *buf_ptr          = nullptr;
    cl_mem_dmabuf_host_ptr src_buf_dmabuf    = {};
    cl_mem                 src_buffer        = {};
    cl_mem_dmabuf_host_ptr out_buf_dmabuf    = {};
    cl_mem                 out_buffer        = {};
    const size_t           global_work_size  = BUFFER_SIZE;
    int                    i                 = 0;
    int                    num_mismatches    = 0;
    struct dma_buf_sync    buf_sync          = {};
    cl_event               unmap_event       = nullptr;

    /*
     * Step 0: Create Buffer Allocator
     */
    buffer_allocator = CreateDmabufHeapBufferAllocator();
    if(buffer_allocator == nullptr)
    {
        std::cerr << "Error with CreateDmabufHeapBufferAllocator()\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    /*
     * Step 1: setup in order queue
     */
    errcode = clGetPlatformIDs(1, &platform, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    errcode = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errcode);
    CL_ASSERT_ERRCODE(errcode);
    queue = clCreateCommandQueueWithProperties(context, device, nullptr, &errcode);
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 2: Confirm required extensions
     */
    if (!confirm_extension_support(device, "cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for dmabuf-backed images is not supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    if (!confirm_extension_support(device, "cl_qcom_ext_host_ptr_iocoherent"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr_iocoherent needed for dmabuf-backed images is not supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    if (!confirm_extension_support(device, "cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for dmabuf-backed images is not supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    /*
     * Step 3: build the kernel
     */
    program = clCreateProgramWithSource(context, 1, &PROGRAM_SOURCE, nullptr, &errcode);
    CL_ASSERT_ERRCODE(errcode);
    errcode = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    kernel = clCreateKernel(program, "copybuf",  &errcode);
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 4: Create dmabuf-backed CL buffers
     */

    // Source buffer
    src_buf_dmabuf = make_dmabuf_buffer(buffer_allocator, device, BUFFER_SIZE, 0, CL_MEM_HOST_IOCOHERENT_QCOM);
    fds.push_back(src_buf_dmabuf.dmabuf_filedesc);

    src_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        BUFFER_SIZE,
        &src_buf_dmabuf,
        &errcode
    );
    CL_ASSERT_ERRCODE(errcode);

    // Output buffer
    out_buf_dmabuf = make_dmabuf_buffer(buffer_allocator, device, BUFFER_SIZE, 0, CL_MEM_HOST_IOCOHERENT_QCOM);
    fds.push_back(out_buf_dmabuf.dmabuf_filedesc);

    out_buffer = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        BUFFER_SIZE,
        &out_buf_dmabuf,
        &errcode
    );
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 5: Initialize the CL buffers
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(src_buf_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        CL_ASSERT_ERRCODE(errno);
    }
    // Initialize source buffer to 0xff
    buf_ptr = static_cast<char *>(clEnqueueMapBuffer(
        queue,
        src_buffer,
        CL_BLOCKING,
        CL_MAP_WRITE,
        0,
        BUFFER_SIZE,
        0,
        nullptr,
        nullptr,
        &errcode
    ));
    CL_ASSERT_ERRCODE(errcode);

    memset(buf_ptr, 0xff, BUFFER_SIZE);

    errcode = clEnqueueUnmapMemObject(queue, src_buffer, buf_ptr, 0, nullptr, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clWaitForEvents(1, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(src_buf_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
       std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
       CL_ASSERT_ERRCODE(errno);
    }

    // Ensure cache coherency.
    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(out_buf_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        CL_ASSERT_ERRCODE(errno);
    }
    // Initialize the output buffer to 0x00
    buf_ptr = static_cast<char *>(clEnqueueMapBuffer(
        queue,
        out_buffer,
        CL_BLOCKING,
        CL_MAP_WRITE,
        0,
        BUFFER_SIZE,
        0,
        nullptr,
        nullptr,
        &errcode
    ));
    CL_ASSERT_ERRCODE(errcode);

    memset(buf_ptr, 0x00, BUFFER_SIZE);

    errcode = clEnqueueUnmapMemObject(queue, out_buffer, buf_ptr, 0, nullptr, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clWaitForEvents(1, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(out_buf_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
       std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
       CL_ASSERT_ERRCODE(errno);
    }
    /*
     * Step 6: Set up the kernel arguments and run the kernel
     */
    errcode = clSetKernelArg(kernel, 0, sizeof(src_buffer), &src_buffer);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clSetKernelArg(kernel, 1, sizeof(out_buffer), &out_buffer);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        nullptr,
        &global_work_size,
        nullptr,
        0,
        nullptr,
        nullptr
    );
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 7: Verify the copy operation
     */

    // Ensure cache coherency.
    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if ( ioctl(out_buf_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        CL_ASSERT_ERRCODE(errno);
    }
    buf_ptr = static_cast<char *>(clEnqueueMapBuffer(
        queue,
        out_buffer,
        CL_BLOCKING,
        CL_MAP_READ,
        0,
        BUFFER_SIZE,
        0,
        nullptr,
        nullptr,
        &errcode
    ));
    CL_ASSERT_ERRCODE(errcode);

    for(i = 0; i < BUFFER_SIZE; i++)
    {
        if(buf_ptr[i] != (char)0xff)
        {
            num_mismatches++;
        }
    }

    errcode = clEnqueueUnmapMemObject(queue, out_buffer, buf_ptr, 0, nullptr, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clWaitForEvents(1, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    if ( ioctl(out_buf_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
       std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
       CL_ASSERT_ERRCODE(errno);
    }

    if(num_mismatches > 0)
    {
        errcode = EXIT_FAILURE;
        std::cout << "copybuf failed! Found: " << num_mismatches << " mismatches!\n";
    }
    else
    {
        errcode = EXIT_SUCCESS;
        std::cout << "copybuf passed!\n";
    }

    cleanup:
    if(src_buffer != nullptr)
    {
        clReleaseMemObject(src_buffer);
    }
    if(out_buffer != nullptr)
    {
        clReleaseMemObject(out_buffer);
    }
    if(kernel != nullptr)
    {
        clReleaseKernel(kernel);
    }
    if(program != nullptr)
    {
        clReleaseProgram(program);
    }
    if(queue != nullptr)
    {
        clReleaseCommandQueue(queue);
    }
    if(context != nullptr)
    {
        clReleaseContext(context);
    }
    for(auto fd: fds)
    {
        if (close(fd) < 0)
        {
            std::cerr << "Error " << errno << " closing dmabuf allocation fd: " << strerror(errno) << "\n";
            std::exit(errno);
        }
    }
    if (buffer_allocator != nullptr)
    {
        FreeDmabufHeapBufferAllocator(buffer_allocator);
    }

    return errcode;
}

int dmabuf_image()
{
    cl_int                 errcode                                   = 0;
    cl_device_id           device                                    = nullptr;
    cl_platform_id         platform                                  = nullptr;
    cl_context             context                                   = nullptr;
    cl_command_queue       queue                                     = nullptr;
    cl_program             program                                   = nullptr;
    cl_kernel              kernel                                    = nullptr;
    BufferAllocator        *buffer_allocator                         = nullptr;
    std::vector<int>       fds                                       = {};
    char                   *img_ptr                                  = nullptr;
    cl_image_format        src_img_format                            = {};
    cl_image_desc          src_img_desc                              = {};
    cl_mem_dmabuf_host_ptr src_img_dmabuf                            = {};
    cl_mem                 src_image                                 = {};
    cl_image_format        out_img_format                            = {};
    cl_image_desc          out_img_desc                              = {};
    cl_mem_dmabuf_host_ptr out_img_dmabuf                            = {};
    cl_mem                 out_image                                 = {};
    const size_t           global_work_size[]                        = {IMAGE_WIDTH, IMAGE_HEIGHT};
    int                    i                                         = 0;
    int                    j                                         = 0;
    int                    num_mismatches                            = 0;
    size_t                 origin[]                                  = {0, 0, 0};
    const size_t           src_img_region[]                          = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};
    const size_t           out_img_region[]                          = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};
    size_t                 row_pitch                                 = 0;
    size_t                 element_size                              = 0;
    size_t                 total_bytes                               = 0;
    struct dma_buf_sync    buf_sync                                  = {};
    cl_event               unmap_event                               = nullptr;

    /*
     * Step 0: Create Buffer Allocator
     * 创建DMSbuffer的内存分配器
     */
    buffer_allocator = CreateDmabufHeapBufferAllocator();
    if(buffer_allocator == nullptr)
    {
        std::cerr << "Error with CreateDmabufHeapBufferAllocator()\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    /*
     * Step 1: setup in order queue
     */
    errcode = clGetPlatformIDs(1, &platform, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    errcode = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errcode);
    CL_ASSERT_ERRCODE(errcode);
    queue = clCreateCommandQueueWithProperties(context, device, nullptr, &errcode);
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 2: Confirm required extensions
     */
    if (!confirm_extension_support(device, "cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for dmabuf-backed images is not supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    if (!confirm_extension_support(device, "cl_qcom_ext_host_ptr_iocoherent"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr_iocoherent needed for dmabuf-backed images is not supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    if (!confirm_extension_support(device, "cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for dmabuf-backed images is not supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    if (!confirm_extension_support(device, "cl_qcom_extended_query_image_info"))
    {
        std::cerr << "Extension cl_qcom_extended_query_image_info needed for dmabuf-backed images is not supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    /*
     * Step 3: build the kernel
     */
    program = clCreateProgramWithSource(context, 1, &PROGRAM_SOURCE, nullptr, &errcode);
    CL_ASSERT_ERRCODE(errcode);
    errcode = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    kernel = clCreateKernel(program, "copyimg",  &errcode);
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 4: Create dmabuf-backed CL images
     */

    // Source image 创建opencl的image对象非
    src_img_format.image_channel_order = CL_RGBA;
    src_img_format.image_channel_data_type = CL_UNORM_INT8;

    src_img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    src_img_desc.image_width = IMAGE_WIDTH;
    src_img_desc.image_height = IMAGE_HEIGHT;
    /// 获取设置返回：图像每行的实际字节数（即行间距）
    errcode = clGetDeviceImageInfoQCOM(device, src_img_desc.image_width, src_img_desc.image_height, &src_img_format,
                                       CL_IMAGE_ROW_PITCH, sizeof(src_img_desc.image_row_pitch), &src_img_desc.image_row_pitch, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    
    // 获取返回：这张图像的总的字节数大小
    errcode = clQueryImageInfoQCOM(device, CL_MEM_READ_ONLY, &src_img_format, &src_img_desc,
                                    CL_IMAGE_SIZE_QCOM, sizeof(total_bytes), &total_bytes, nullptr);
    CL_ASSERT_ERRCODE(errcode);

    /// 获取DMA buffer的句柄。并且放到句柄列表中
    src_img_dmabuf = make_dmabuf_buffer(buffer_allocator, device, total_bytes, 0, CL_MEM_HOST_IOCOHERENT_QCOM);
    fds.push_back(src_img_dmabuf.dmabuf_filedesc);

    /// 使用这个buffer句柄，来进行创建clCreateImage
    src_image = clCreateImage(
        context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        &src_img_format,
        &src_img_desc,
        &src_img_dmabuf,
        &errcode
    );
    CL_ASSERT_ERRCODE(errcode);

    // Output image  下面是同样的output Immage
    out_img_format.image_channel_order = CL_RGBA;
    out_img_format.image_channel_data_type = CL_UNORM_INT8;

    out_img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    out_img_desc.image_width = IMAGE_WIDTH;
    out_img_desc.image_height = IMAGE_HEIGHT;
    errcode = clGetDeviceImageInfoQCOM(device, out_img_desc.image_width, out_img_desc.image_height, &out_img_format,
                                       CL_IMAGE_ROW_PITCH, sizeof(out_img_desc.image_row_pitch), &out_img_desc.image_row_pitch, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    errcode = clQueryImageInfoQCOM(device, CL_MEM_READ_ONLY, &out_img_format, &out_img_desc,
                                    CL_IMAGE_SIZE_QCOM, sizeof(total_bytes), &total_bytes, nullptr);
    CL_ASSERT_ERRCODE(errcode);

    out_img_dmabuf = make_dmabuf_buffer(buffer_allocator, device, total_bytes, 0, CL_MEM_HOST_IOCOHERENT_QCOM);
    fds.push_back(out_img_dmabuf.dmabuf_filedesc);

    out_image = clCreateImage(
        context,
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        &out_img_format,
        &out_img_desc,
        &out_img_dmabuf,
        &errcode
    );
    CL_ASSERT_ERRCODE(errcode);



    // get element size
    errcode = clGetImageInfo(src_image, CL_IMAGE_ELEMENT_SIZE, sizeof(element_size), &element_size, nullptr);
    CL_ASSERT_ERRCODE(errcode);

    /*
     * 初始化OpenCL的images
     * Step 5: Initialize the CL images
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(src_img_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        CL_ASSERT_ERRCODE(errno);
    }
    // Initialize source buffer to 0xff
    row_pitch = 0;
    img_ptr = static_cast<char *>(clEnqueueMapImage(
        queue,
        src_image,
        CL_BLOCKING,
        CL_MAP_WRITE,
        origin,
        src_img_region,
        &row_pitch,
        nullptr,
        0,
        nullptr,
        nullptr,
        &errcode
    ));
    CL_ASSERT_ERRCODE(errcode);

    for (i = 0; i < src_img_desc.image_height; ++i)
    {
        memset(img_ptr + i * row_pitch, 0xff, src_img_desc.image_width * element_size);
    }



    errcode = clEnqueueUnmapMemObject(queue, src_image, img_ptr, 0, nullptr, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clWaitForEvents(1, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(src_img_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
       std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
       CL_ASSERT_ERRCODE(errno);
    }





    
    // Ensure cache coherency. 确保缓存一致性
    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(out_img_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        CL_ASSERT_ERRCODE(errno);
    }
    // Initialize the output buffer to 0x00
    row_pitch = 0;
    img_ptr = static_cast<char *>(clEnqueueMapImage(
        queue,
        out_image,
        CL_BLOCKING,
        CL_MAP_WRITE,
        origin,
        out_img_region,
        &row_pitch,
        nullptr,
        0,
        nullptr,
        nullptr,
        &errcode
    ));
    CL_ASSERT_ERRCODE(errcode);

    for (i = 0; i < out_img_desc.image_height; ++i)
    {
        memset(img_ptr + i * row_pitch, 0x00, out_img_desc.image_width * element_size);
    }

    errcode = clEnqueueUnmapMemObject(queue, out_image, img_ptr, 0, nullptr, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clWaitForEvents(1, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(out_img_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
       std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
       CL_ASSERT_ERRCODE(errno);
    }



    /*
     * 开始设置kernel的参数，然后进行运行kernel
     * Step 6: Set up the kernel arguments and run the kernel
     */
    errcode = clSetKernelArg(kernel, 0, sizeof(src_image), &src_image);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clSetKernelArg(kernel, 1, sizeof(out_image), &out_image);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clEnqueueNDRangeKernel(
        queue,
        kernel,
        2,
        nullptr,
        global_work_size,
        nullptr,
        0,
        nullptr,
        nullptr
    );
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 7: Verify the copy operation
     */

     /// 下面进行结果验证
    // Ensure cache coherency.
    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if ( ioctl(out_img_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        CL_ASSERT_ERRCODE(errno);
    }
    row_pitch = 0;
    img_ptr = static_cast<char *>(clEnqueueMapImage(
        queue,
        out_image,
        CL_BLOCKING,
        CL_MAP_READ,
        origin,
        out_img_region,
        &row_pitch,
        nullptr,
        0,
        nullptr,
        nullptr,
        &errcode
    ));
    CL_ASSERT_ERRCODE(errcode);

    for (i = 0; i < out_img_desc.image_height; ++i)
    {
        for(j = 0; j < out_img_desc.image_width * element_size; ++j)
        {
            if(img_ptr[i * row_pitch + j] != (char)0xff)
            {
                num_mismatches++;
            }
        }
    }

    errcode = clEnqueueUnmapMemObject(queue, out_image, img_ptr, 0, nullptr, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clWaitForEvents(1, &unmap_event);
    CL_ASSERT_ERRCODE(errcode);

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    if ( ioctl(out_img_dmabuf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
       std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
       CL_ASSERT_ERRCODE(errno);
    }

    if(num_mismatches > 0)
    {
        errcode = EXIT_FAILURE;
        std::cout << "copyimg failed! Found: " << num_mismatches << " mismatches!\n";
    }
    else
    {
        errcode = EXIT_SUCCESS;
        std::cout << "copyimg passed!\n";
    }

    cleanup:
    if(src_image != nullptr)
    {
        clReleaseMemObject(src_image);
    }
    if(out_image != nullptr)
    {
        clReleaseMemObject(out_image);
    }
    if(kernel != nullptr)
    {
        clReleaseKernel(kernel);
    }
    if(program != nullptr)
    {
        clReleaseProgram(program);
    }
    if(queue != nullptr)
    {
        clReleaseCommandQueue(queue);
    }
    if(context != nullptr)
    {
        clReleaseContext(context);
    }
    for(auto fd: fds)
    {
        if (close(fd) < 0)
        {
            std::cerr << "Error " << errno << " closing dmabuf allocation fd: " << strerror(errno) << "\n";
            std::exit(errno);
        }
    }
    if (buffer_allocator != nullptr)
    {
        FreeDmabufHeapBufferAllocator(buffer_allocator);
    }

    return errcode;
}


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <type>\n"
                  << "\t<type> = buffer | image\n"
                  << " This sample shows how to create and use dmabuf backed buffers and images";
        return EXIT_FAILURE;
    }
    const std::string arg(argv[1]);
    if(arg.compare("buffer") != 0 && arg.compare("image") != 0)
    {
        std::cerr << "Usage: " << argv[0] << " <type>\n"
                  << "\t<type> = buffer | image\n"
                  << " This sample shows how to create and use dmabuf backed buffers and images";
        return EXIT_FAILURE;
    }
    bool is_buffer = arg.compare("buffer") == 0;

    if(is_buffer)
    {
        return dmabuf_buffer();
    } else
    {
        return dmabuf_image();
    }
}
