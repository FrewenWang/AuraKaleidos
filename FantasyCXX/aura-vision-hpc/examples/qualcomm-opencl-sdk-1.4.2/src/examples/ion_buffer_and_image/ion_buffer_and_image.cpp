//--------------------------------------------------------------------------------------
// File: ion_buffer_and_image.cpp
// Desc: Standalone sample showcasing how to use ION backed buffers and images
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

// ION
#include <ion/ion.h>
#include <linux/msm_ion.h> // Vendor specific Ion heap IDs and flags.

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext.h>

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

// This define controls the allocation type for the buffer and image. If defined the allocation will be IOCOHERENT,
// otherwise it will be UNCACHED.
#define IOCOHERENT_ALLOCATION

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

cl_mem_ion_host_ptr make_ion_buffer(int ion_fd, cl_device_id device, size_t size, unsigned int ion_allocation_flags, cl_uint host_cache_policy)
{
    cl_int  errcode;
    cl_uint device_page_size;

    errcode = clGetDeviceInfo(device, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(device_page_size), &device_page_size, nullptr);
    if (errcode != CL_SUCCESS)
    {
        std::cerr << "Error " << errcode << " with clGetDeviceInfo for page size." << "\n";
        std::exit(errcode);
    }

    int fd = 0;
    errcode = ion_alloc_fd(ion_fd, size, device_page_size, ION_HEAP(ION_SYSTEM_HEAP_ID), ion_allocation_flags, &fd);
    if (errcode == -1)
    {
        std::cerr << "Error allocating ion memory\n";
        std::exit(EXIT_FAILURE);
    }

    void *host_addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (MAP_FAILED == host_addr)
    {
        close(fd);
        std::cerr << "Error " << errno << " mmapping fd to pointer: " << strerror(errno) << "\n";
        std::exit(errno);
    }

    cl_mem_ion_host_ptr ion_mem;
    ion_mem.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
    ion_mem.ext_host_ptr.host_cache_policy = host_cache_policy;
    ion_mem.ion_filedesc                   = fd;
    ion_mem.ion_hostptr                    = host_addr;

    return ion_mem;
}

int ion_buffer()
{
    cl_int              errcode                                   = 0;
    cl_device_id        device                                    = nullptr;
    cl_platform_id      platform                                  = nullptr;
    cl_context          context                                   = nullptr;
    cl_command_queue    queue                                     = nullptr;
    cl_program          program                                   = nullptr;
    cl_kernel           kernel                                    = nullptr;
    int                 ion_fd                                    = -1;
    std::vector<int>    fds                                       = {};
    char                *buf_ptr                                  = nullptr;
    cl_mem_ion_host_ptr src_buf_ion                               = {};
    cl_mem              src_buffer                                = {};
    cl_mem_ion_host_ptr out_buf_ion                               = {};
    cl_mem              out_buffer                                = {};
    const size_t        global_work_size                          = BUFFER_SIZE;
    int                 i                                         = 0;
    int                 num_mismatches                            = 0;

    /*
     * Step 0: Open ION
     */
    ion_fd = ion_open();
    if(ion_fd < 0)
    {
        std::cerr << "Error with ion_open()\n";
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
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for ION-backed images is not supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    if (!confirm_extension_support(device, "cl_qcom_ion_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ion_host_ptr needed for ION-backed images is not supported.\n";
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
     * Step 4: Create ION-backed CL buffers
     */

    // Source buffer
#ifdef IOCOHERENT_ALLOCATION
    src_buf_ion = make_ion_buffer(ion_fd, device, BUFFER_SIZE, ION_FLAG_CACHED, CL_MEM_HOST_IOCOHERENT_QCOM);
#else
    src_buf_ion = make_ion_buffer(ion_fd, device, BUFFER_SIZE, 0, CL_MEM_HOST_UNCACHED_QCOM);
#endif
    fds.push_back(src_buf_ion.ion_filedesc);

    src_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        BUFFER_SIZE,
        &src_buf_ion,
        &errcode
    );
    CL_ASSERT_ERRCODE(errcode);

    // Output buffer
#ifdef IOCOHERENT_ALLOCATION
    out_buf_ion = make_ion_buffer(ion_fd, device, BUFFER_SIZE, ION_FLAG_CACHED, CL_MEM_HOST_IOCOHERENT_QCOM);
#else
    out_buf_ion = make_ion_buffer(ion_fd, device, BUFFER_SIZE, 0, CL_MEM_HOST_UNCACHED_QCOM);
#endif
    fds.push_back(out_buf_ion.ion_filedesc);

    out_buffer = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        BUFFER_SIZE,
        &out_buf_ion,
        &errcode
    );
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 5: Initialize the CL buffers
     */

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

    errcode = clEnqueueUnmapMemObject(queue, src_buffer, buf_ptr, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(errcode);

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

    errcode = clEnqueueUnmapMemObject(queue, out_buffer, buf_ptr, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(errcode);

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

    errcode = clEnqueueUnmapMemObject(queue, out_buffer, buf_ptr, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(errcode);

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
            std::cerr << "Error " << errno << " closing ion allocation fd: " << strerror(errno) << "\n";
            std::exit(errno);
        }
    }
    if (ion_close(ion_fd) < 0)
    {
        std::cerr << "Error closing ion device fd.\n";
        std::exit(EXIT_FAILURE);
    }

    return errcode;
}

int ion_image()
{
    cl_int              errcode                                   = 0;
    cl_device_id        device                                    = nullptr;
    cl_platform_id      platform                                  = nullptr;
    cl_context          context                                   = nullptr;
    cl_command_queue    queue                                     = nullptr;
    cl_program          program                                   = nullptr;
    cl_kernel           kernel                                    = nullptr;
    int                 ion_fd                                    = -1;
    std::vector<int>    fds                                       = {};
    char                *img_ptr                                  = nullptr;
    cl_image_format     src_img_format                            = {};
    cl_image_desc       src_img_desc                              = {};
    cl_mem_ion_host_ptr src_img_ion                               = {};
    cl_mem              src_image                                 = {};
    cl_image_format     out_img_format                            = {};
    cl_image_desc       out_img_desc                              = {};
    cl_mem_ion_host_ptr out_img_ion                               = {};
    cl_mem              out_image                                 = {};
    const size_t        global_work_size[]                        = {IMAGE_WIDTH, IMAGE_HEIGHT};
    int                 i                                         = 0;
    int                 j                                         = 0;
    int                 num_mismatches                            = 0;
    size_t              origin[]                                  = {0, 0, 0};
    const size_t        src_img_region[]                          = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};
    const size_t        out_img_region[]                          = {IMAGE_WIDTH, IMAGE_HEIGHT, 1};
    size_t              row_pitch                                 = 0;
    size_t              element_size                              = 0;
    size_t              padding_in_bytes                          = 0;
    size_t              total_bytes                               = 0;

    /*
     * Step 0: Open ION
     */
    ion_fd = ion_open();
    if(ion_fd < 0)
    {
        std::cerr << "Error with ion_open()\n";
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
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for ION-backed images is not supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    if (!confirm_extension_support(device, "cl_qcom_ion_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ion_host_ptr needed for ION-backed images is not supported.\n";
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
     * Step 4: Create ION-backed CL images
     */

    // get device padding in bytes
    errcode = clGetDeviceInfo(device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(padding_in_bytes), &padding_in_bytes, nullptr);
    CL_ASSERT_ERRCODE(errcode);

    // Source image
    src_img_format.image_channel_order = CL_RGBA;
    src_img_format.image_channel_data_type = CL_UNORM_INT8;

    src_img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    src_img_desc.image_width = IMAGE_WIDTH;
    src_img_desc.image_height = IMAGE_HEIGHT;
    errcode = clGetDeviceImageInfoQCOM(device, src_img_desc.image_width, src_img_desc.image_height, &src_img_format,
                             CL_IMAGE_ROW_PITCH, sizeof(src_img_desc.image_row_pitch), &src_img_desc.image_row_pitch, nullptr);
    CL_ASSERT_ERRCODE(errcode);

    total_bytes = src_img_desc.image_row_pitch * src_img_desc.image_height + padding_in_bytes;

#ifdef IOCOHERENT_ALLOCATION
    src_img_ion = make_ion_buffer(ion_fd, device, total_bytes, ION_FLAG_CACHED, CL_MEM_HOST_IOCOHERENT_QCOM);
#else
    src_img_ion = make_ion_buffer(ion_fd, device, total_bytes, 0, CL_MEM_HOST_UNCACHED_QCOM);
#endif
    fds.push_back(src_img_ion.ion_filedesc);

    src_image = clCreateImage(
        context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        &src_img_format,
        &src_img_desc,
        &src_img_ion,
        &errcode
    );
    CL_ASSERT_ERRCODE(errcode);

    // Output image
    out_img_format.image_channel_order = CL_RGBA;
    out_img_format.image_channel_data_type = CL_UNORM_INT8;

    out_img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    out_img_desc.image_width = IMAGE_WIDTH;
    out_img_desc.image_height = IMAGE_HEIGHT;
    errcode = clGetDeviceImageInfoQCOM(device, out_img_desc.image_width, out_img_desc.image_height, &out_img_format,
                                       CL_IMAGE_ROW_PITCH, sizeof(out_img_desc.image_row_pitch), &out_img_desc.image_row_pitch, nullptr);
    CL_ASSERT_ERRCODE(errcode);

    total_bytes = out_img_desc.image_row_pitch * out_img_desc.image_height + padding_in_bytes;

#ifdef IOCOHERENT_ALLOCATION
    out_img_ion = make_ion_buffer(ion_fd, device, total_bytes, ION_FLAG_CACHED, CL_MEM_HOST_IOCOHERENT_QCOM);
#else
    out_img_ion = make_ion_buffer(ion_fd, device, total_bytes, 0, CL_MEM_HOST_UNCACHED_QCOM);
#endif
    fds.push_back(out_img_ion.ion_filedesc);

    out_image = clCreateImage(
        context,
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        &out_img_format,
        &out_img_desc,
        &out_img_ion,
        &errcode
    );
    CL_ASSERT_ERRCODE(errcode);

    // get element size
    errcode = clGetImageInfo(src_image, CL_IMAGE_ELEMENT_SIZE, sizeof(element_size), &element_size, nullptr);
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 5: Initialize the CL images
     */

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

    errcode = clEnqueueUnmapMemObject(queue, src_image, img_ptr, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(errcode);

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

    errcode = clEnqueueUnmapMemObject(queue, out_image, img_ptr, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(errcode);

    /*
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

    errcode = clEnqueueUnmapMemObject(queue, out_image, img_ptr, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(errcode);

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
            std::cerr << "Error " << errno << " closing ion allocation fd: " << strerror(errno) << "\n";
            std::exit(errno);
        }
    }
    if (ion_close(ion_fd) < 0)
    {
        std::cerr << "Error closing ion device fd.\n";
        std::exit(EXIT_FAILURE);
    }

    return errcode;
}


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <type>\n"
                  << "\t<type> = buffer | image\n"
                  << " This sample shows how to create and use ion backed buffers and images";
        return EXIT_FAILURE;
    }
    const std::string arg(argv[1]);
    if(arg.compare("buffer") != 0 && arg.compare("image") != 0)
    {
        std::cerr << "Usage: " << argv[0] << " <type>\n"
                  << "\t<type> = buffer | image\n"
                  << " This sample shows how to create and use ion backed buffers and images";
        return EXIT_FAILURE;
    }
    bool is_buffer = arg.compare("buffer") == 0;

    if(is_buffer)
    {
        return ion_buffer();
    } else
    {
        return ion_image();
    }
}