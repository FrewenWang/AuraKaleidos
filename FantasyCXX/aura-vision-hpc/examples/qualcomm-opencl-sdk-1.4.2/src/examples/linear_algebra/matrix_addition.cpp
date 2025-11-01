//--------------------------------------------------------------------------------------
// File: matrix_addition.cpp
// Desc: Computes the matrix sum C = A + B.
//
// Author:      QUALCOMM
//
//               Copyright (c) 2018, 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <cstdlib>
#include <cstring>
#include <iostream>

// Project includes
#include "util/cl_wrapper.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

static const char *PROGRAM_SOURCE = R"(
    __kernel void buffer_addition(__global const float *matrix_a,
                                  __global const float *matrix_b,
                                  __global       float *matrix_c)
    {
        const int wid_x = get_global_id(0);
        matrix_c[wid_x] = matrix_a[wid_x] + matrix_b[wid_x];
    }
)";

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Please specify input files.\n"
                     "\n"
                     "Usage: " << argv[0] << " <matrix A> <matrix B> [<output file>]\n"
                     "Computes the matrix sum C = A + B. See README.md for matrix input format.\n"
                     "If no file is specified for the output, then it is written to stdout.\n";
        return EXIT_FAILURE;
    }
    const std::string matrix_a_filename(argv[1]);
    const std::string matrix_b_filename(argv[2]);
    const bool        output_to_file = argc >= 4;
    const matrix_t    matrix_a       = load_matrix(matrix_a_filename);
    const matrix_t    matrix_b       = load_matrix(matrix_b_filename);
    const size_t      matrix_size    = matrix_a.width * matrix_a.height;
    const size_t      matrix_bytes   = matrix_size * sizeof(cl_float);
    const std::string output_filename(output_to_file ? argv[3] : "");

    if (matrix_a.width != matrix_b.width && matrix_a.height != matrix_b.height)
    {
        std::cerr << "Matrix A and B must have the same dimensions.\n";
        return EXIT_FAILURE;
    }

    cl_wrapper          wrapper;
    cl_program          program       = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel           kernel        = wrapper.make_kernel("buffer_addition", program);
    cl_context          context       = wrapper.get_context();
    cl_command_queue    command_queue = wrapper.get_command_queue();
    struct dma_buf_sync buf_sync      = {};
    cl_event            unmap_event   = nullptr;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for dmabuf-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for dmabuf-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create suitable dmabuf-backed buffers.
     */

    cl_int err =  CL_SUCCESS;

    cl_mem_dmabuf_host_ptr matrix_a_buf = wrapper.make_buffer(matrix_bytes);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(matrix_a_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    std::memcpy(matrix_a_buf.dmabuf_hostptr, matrix_a.elements.data(), matrix_bytes);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(matrix_a_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    cl_mem              matrix_a_mem     = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_bytes,
            &matrix_a_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix A." << "\n";
        return err;
    }

    cl_mem_dmabuf_host_ptr matrix_b_buf = wrapper.make_buffer(matrix_bytes);
    std::memcpy(matrix_b_buf.dmabuf_hostptr, matrix_b.elements.data(), matrix_bytes);
    cl_mem              matrix_b_mem     = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_bytes,
            &matrix_b_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix B." << "\n";
        return err;
    }

    cl_mem_dmabuf_host_ptr matrix_c_buf = wrapper.make_buffer(matrix_bytes);
    cl_mem              matrix_c_mem     = clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_bytes,
            &matrix_c_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix C." << "\n";
        return err;
    }

    /*
     * Step 2: Set up the kernel arguments
     */

    err = clSetKernelArg(kernel, 0, sizeof(matrix_a_mem), &matrix_a_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel, 1, sizeof(matrix_b_mem), &matrix_b_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel, 2, sizeof(matrix_c_mem), &matrix_c_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        return err;
    }

    /*
     * Step 3: Run the kernel.
     */

    const size_t global_work_size = matrix_size;
    err = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            1,
            nullptr,
            &global_work_size,
            nullptr,
            0,
            nullptr,
            nullptr
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel." << "\n";
        return err;
    }

    /*
     * Step 4: Copy the data out of the DMA buffer.
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if ( ioctl(matrix_c_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    cl_float *ptr = static_cast<cl_float *>(clEnqueueMapBuffer(
            command_queue,
            matrix_c_mem,
            CL_BLOCKING,
            CL_MAP_READ,
            0,
            matrix_bytes,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueMapBuffer." << "\n";
        return err;
    }

    matrix_t matrix_c;
    matrix_c.width  = matrix_a.width;
    matrix_c.height = matrix_a.height;
    matrix_c.elements.resize(matrix_size);
    std::memcpy(matrix_c.elements.data(), ptr, matrix_bytes);

    err = clEnqueueUnmapMemObject(command_queue, matrix_c_mem, ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueUnmapMemObject." << "\n";
        return err;
    }

    err = clWaitForEvents(1, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while waiting to unmap.\n";
        return err;
    }

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    if ( ioctl(matrix_c_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    clFinish(command_queue);

    if (output_to_file)
    {
        save_matrix(output_filename, matrix_c);
    }
    else
    {
        save_matrix(std::cout, matrix_c);
    }

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseMemObject(matrix_a_mem);
    clReleaseMemObject(matrix_b_mem);
    clReleaseMemObject(matrix_c_mem);

    return EXIT_SUCCESS;
}
