//--------------------------------------------------------------------------------------
// File: cl_wrapper.h
// Desc:
//
// Author:      QUALCOMM
//
//               Copyright (c) 2017, 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

#include "util.h"

#include <linux/dma-buf.h>
#include <BufferAllocator/BufferAllocator.h>
#include <BufferAllocator/BufferAllocatorWrapper.h>

/**
 * \brief A wrapper around OpenCL setup/teardown code.
 *
 * All objects exposed are owned by the wrapper, and are cleaned up when it is destroyed. cl_wrapper uses DMA-BUF memory
 * allocations when creating buffers.
 */
class cl_wrapper {
public:
    /**
     * \brief Sets up OpenCL.
     */
    cl_wrapper(const cl_context_properties *context_properties = nullptr, const cl_queue_properties *queue_properties = nullptr);

    /**
     * \brief Frees associated OpenCL objects, including the results of make_kernel, make_program, and make_buffer.
     */
    ~cl_wrapper();

    /**
     * \brief Gets the cl_context associated with the wrapper for using in OpenCL functions.
     * @return
     */
    cl_context          get_context() const;

    /**
    * \brief Gets the cl_command_queue associated with the wrapper for using in OpenCL functions.
    * @return
    */
    cl_command_queue    get_command_queue() const;

    /**
     * \brief Makes a cl_kernel from the given program.
     *
     * @param kernel_name
     * @param program
     * @return
     */
    cl_kernel           make_kernel(const std::string &kernel_name, cl_program program);

    /**
     * Makes a cl_program (whose lifetime is managed by cl_wrapper) from the given source code strings.
     * @param source - The source code strings
     * @param source_len - The length of program_source
     * @param options - Options to pass to the compiler
     * @return
     */
    cl_program          make_program(const char **source, cl_uint source_len, const char *options = "");

    /**
     * \brief Checks if the wrapped device supports the desired extension via clGetDeviceInfo
     *
     * @param desired_extension
     * @return true if the desired_extension is supported, otherwise false
     */
    bool                check_extension_support(const std::string &desired_extension) const;

    /**
     * \brief Gets the required row pitch for the given image. Must be considered when accessing the underlying buffer.
     *
     * @param img_format [in] - The image format
     * @param img_desc [in] - The image description
     * @return the image row pitch
     */
    size_t              get_image_row_pitch(const cl_image_format &img_format, const cl_image_desc &img_desc) const;

    /**
     * \brief Gets the max workgroup size for the specified kernel.
     *
     * @param kernel
     * @return
     */
    size_t              get_max_workgroup_size(cl_kernel kernel) const;

    /**
     * \brief Gets the OpenCL device associated with the context
     * @return
     */
    cl_device_id        get_device_id() const;

    /**
     * \brief Makes an dmabuf buffer that can be used for a nonplanar image, e.g. CL_R or CL_RGB
     *
     * @param img_format [in]
     * @param img_desc [in]
     * @return
     */
    cl_mem_dmabuf_host_ptr make_buffer_for_nonplanar_image(const cl_image_format &img_format, const cl_image_desc &img_desc);

    /**
     * \brief Makes an dmabuf buffer that can be used for a YUV 4:2:0 image.
     *
     * @param img_format [in] - The image format
     * @param img_desc [in] - The image description
     * @param img_row_pitch [in] - The image pitch
     * @return
     */
    cl_mem_dmabuf_host_ptr make_buffer_for_yuv_image(const cl_image_format& img_format, const cl_image_desc& img_desc);

    /**
     * \brief Makes an dmabuf buffer that can be used for a compressed image.
     *
     * @param img_format [in] - The image format
     * @param img_desc [in] - The image description
     * @return
     */
    cl_mem_dmabuf_host_ptr make_buffer_for_compressed_image(const cl_image_format& img_format, const cl_image_desc& img_desc);

    /**
     * \brief Makes an dmabuf buffer of the specified size.
     *
     * @param size [in] - Desired buffer size
     * @return
     */
    cl_mem_dmabuf_host_ptr make_buffer(size_t size);

private:

    cl_mem_dmabuf_host_ptr
    make_buffer_internal(size_t size, unsigned int dmabuf_allocation_flags);

    // Data members
    cl_device_id m_device;
    cl_context m_context;
    cl_command_queue m_cmd_queue;
    std::vector<cl_program> m_programs;
    std::vector<cl_kernel> m_kernels;

    std::vector<int> m_file_descs;
    std::vector<std::pair<void*, size_t>> m_host_ptrs;

    BufferAllocator* m_buffer_allocator = nullptr;
};
