//--------------------------------------------------------------------------------------
// File: cl_wrapper.cpp
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2017, 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------
#include "util/cl_wrapper.h"

#include <iostream>

#include <sys/mman.h>

cl_wrapper::cl_wrapper(const cl_context_properties *context_properties, const cl_queue_properties *queue_properties)
{
    cl_platform_id platform;
    cl_int err;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetPlatformIDs." << "\n";
        std::exit(err);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_device, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetDeviceIDs." << "\n";
        std::exit(err);
    }

    m_context = clCreateContext(context_properties, 1, &m_device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateContext." << "\n";
        std::exit(err);
    }

    m_cmd_queue = clCreateCommandQueueWithProperties(m_context, m_device, queue_properties, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateCommandQueue." << "\n";
        std::exit(err);
    }

    m_buffer_allocator = CreateDmabufHeapBufferAllocator();
}

cl_wrapper::~cl_wrapper()
{
    for (auto kernel : m_kernels)
    {
        clReleaseKernel(kernel);
    }
    clReleaseCommandQueue(m_cmd_queue);
    for (auto program : m_programs)
    {
        clReleaseProgram(program);
    }
    clReleaseContext(m_context);

    for (auto &pair: m_host_ptrs)
    {
        if (munmap(pair.first, pair.second) < 0)
        {
            std::cerr << "Error " << errno << " munmap-ing dmabuf alloc: " << strerror(errno) << "\n";
            std::exit(errno);
        }
        pair.first = nullptr;
    }

    for (const auto fd : m_file_descs)
    {
        if (close(fd) < 0)
        {
            std::cerr << "Error " << errno << " closing dmabuf allocation fd: " << strerror(errno) << "\n";
            std::exit(errno);
        }
    }

    if(m_buffer_allocator)
    {
        FreeDmabufHeapBufferAllocator(m_buffer_allocator);
    }
}

cl_kernel cl_wrapper::make_kernel(const std::string &kernel_name, cl_program program)
{
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateKernel." << "\n";
        std::exit(err);
    }
    m_kernels.push_back(kernel);
    return kernel;
}

cl_context cl_wrapper::get_context() const
{
    return m_context;
}

cl_command_queue cl_wrapper::get_command_queue() const
{
    return m_cmd_queue;
}

cl_program cl_wrapper::make_program(const char **source, cl_uint source_len, const char *options)
{
    cl_int err = 0;
    cl_program program = clCreateProgramWithSource(m_context, source_len, source, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateProgramWithSource." << "\n";
        std::exit(err);
    }

    err = clBuildProgram(program, 0, nullptr, options, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clBuildProgram.\n";
        static const size_t LOG_SIZE = 2048;
        char log[LOG_SIZE];
        log[0] = 0;
        err = clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, LOG_SIZE, log, nullptr);
        if (err == CL_INVALID_VALUE)
        {
            std::cerr << "There was a build error, but there is insufficient space allocated to show the build logs.\n";
        }
        else
        {
            std::cerr << "Build error:\n" << log << "\n";
        }
        std::exit(EXIT_FAILURE);
    }

    m_programs.push_back(program);

    return program;
}

static std::string init_extension_string(cl_device_id device)
{
    static const size_t BUF_SIZE = 1024;
    char                extensions_buf[BUF_SIZE];
    std::memset(extensions_buf, 0, sizeof(extensions_buf));
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions_buf), extensions_buf, nullptr);
    return std::string(extensions_buf);
}

bool cl_wrapper::check_extension_support(const std::string &desired_extension) const
{
    static const std::string extensions = init_extension_string(m_device);
    if (extensions.size() == 0)
    {
        std::cerr << "Couldn't identify available OpenCL extensions\n";
        std::exit(EXIT_FAILURE);
    }

    return extensions.find(desired_extension) != std::string::npos;
}

size_t cl_wrapper::get_image_row_pitch(const cl_image_format& img_format, const cl_image_desc& img_desc) const
{
    size_t img_row_pitch = 0;
    cl_int err = clGetDeviceImageInfoQCOM(m_device, img_desc.image_width, img_desc.image_height, &img_format,
                                          CL_IMAGE_ROW_PITCH, sizeof(img_row_pitch), &img_row_pitch, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clGetDeviceImageInfoQCOM for CL_IMAGE_ROW_PITCH." << "\n";
        std::exit(err);
    }

    return img_row_pitch;
}

size_t cl_wrapper::get_max_workgroup_size(cl_kernel kernel) const
{
    size_t result = 0;

    cl_int err = clGetKernelWorkGroupInfo(kernel, m_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(result), &result, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetKernelWorkGroupInfo for CL_KERNEL_WORK_GROUP_SIZE." << "\n";
        std::exit(err);
    }

    return result;
}

cl_mem_dmabuf_host_ptr cl_wrapper::make_buffer_for_yuv_image(const cl_image_format& img_format, const cl_image_desc& img_desc)
{
    const size_t effective_img_height = ((img_desc.image_height + 511) / 512) * 512; // Round up to the nearest multiple of 512
    const size_t img_row_pitch = get_image_row_pitch(img_format, img_desc);

    cl_int err;
    size_t padding_in_bytes = 0;

    err = clGetDeviceInfo(m_device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(padding_in_bytes), &padding_in_bytes, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetDeviceInfo for padding." << "\n";
        std::exit(err);
    }

    const size_t y_plane_bytes  = img_row_pitch * effective_img_height;
    const size_t uv_plane_bytes = img_row_pitch * effective_img_height / 2;
    const size_t total_bytes    = y_plane_bytes + uv_plane_bytes + padding_in_bytes;

    return make_buffer(total_bytes);
}

cl_mem_dmabuf_host_ptr cl_wrapper::make_buffer_for_compressed_image(const cl_image_format& img_format, const cl_image_desc& img_desc)
{
    const bool valid_compressed_nv12    = img_format.image_channel_order        == CL_QCOM_COMPRESSED_NV12
                                          && img_format.image_channel_data_type == CL_UNORM_INT8;
    const bool valid_compressed_nv12_4r = img_format.image_channel_order        == CL_QCOM_COMPRESSED_NV12_4R
                                          && img_format.image_channel_data_type == CL_UNORM_INT8;
    const bool valid_compressed_p010    = img_format.image_channel_order        == CL_QCOM_COMPRESSED_P010
                                          && img_format.image_channel_data_type == CL_QCOM_UNORM_INT10;
    const bool valid_compressed_tp10    = img_format.image_channel_order        == CL_QCOM_COMPRESSED_TP10
                                          && img_format.image_channel_data_type == CL_QCOM_UNORM_INT10;
    const bool valid_compressed_rgba    = img_format.image_channel_order        == CL_QCOM_COMPRESSED_RGBA
                                          && img_format.image_channel_data_type == CL_UNORM_INT8;

    std::vector<bool> valid_compressed_images = {
        valid_compressed_nv12,
        valid_compressed_nv12_4r,
        valid_compressed_p010,
        valid_compressed_tp10,
        valid_compressed_rgba
    };

    if (std::none_of(valid_compressed_images.begin(), valid_compressed_images.end(), [](bool valid){ return valid; }))
    {
        std::cerr << "Unsupported image format for compressed image.\n";
        std::exit(EXIT_FAILURE);
    }

    static const size_t max_dims = 2048;
    if (img_desc.image_height > max_dims || img_desc.image_width > max_dims)
    {
        std::cerr << "For this example, the image dimensions must be less than or equal to " << max_dims << "\n";
        std::exit(EXIT_FAILURE);
    }

    // The size of this DMA-BUF buffer will be sufficient to hold an image where each dimension is <= 2048.
    // This is a loose upper bound only, however the general calculation is not within the scope of these examples. If
    // hardware supports the cl_qcom_extended_query_image_info extension we can query compressed image size directly.
    size_t total_bytes = 12681216;
    if(check_extension_support("cl_qcom_extended_query_image_info"))
    {
        cl_int err = clQueryImageInfoQCOM(m_device, CL_MEM_READ_ONLY, &img_format, &img_desc, CL_IMAGE_SIZE_QCOM, sizeof(total_bytes), &total_bytes, nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clQueryImageInfoQCOM for padding." << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    return make_buffer(total_bytes);
}

cl_mem_dmabuf_host_ptr cl_wrapper::make_buffer_for_nonplanar_image(const cl_image_format &img_format, const cl_image_desc &img_desc)
{
    cl_int err;
    size_t padding_in_bytes = 0;

    (void) img_format; // Unused, for now

    err = clGetDeviceInfo(m_device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(padding_in_bytes), &padding_in_bytes, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetDeviceInfo for padding." << "\n";
        std::exit(err);
    }

    const size_t total_bytes = img_desc.image_row_pitch * img_desc.image_height + padding_in_bytes;
    return make_buffer(total_bytes);
}

cl_mem_dmabuf_host_ptr cl_wrapper::make_buffer(size_t size)
{
    return make_buffer_internal(size, 0);
}

cl_mem_dmabuf_host_ptr cl_wrapper::make_buffer_internal(size_t size, unsigned int dmabuf_allocation_flags)
{
    int fd = DmabufHeapAlloc(m_buffer_allocator, "qcom,system", size, 0, 0);
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
    dmabuf_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
    dmabuf_mem.dmabuf_filedesc                = fd;
    dmabuf_mem.dmabuf_hostptr                 = host_addr;

    m_host_ptrs.push_back(std::make_pair(dmabuf_mem.dmabuf_hostptr, size));
    m_file_descs.push_back(fd);

    return dmabuf_mem;
}

cl_device_id cl_wrapper::get_device_id() const {
    return m_device;
}