//--------------------------------------------------------------------------------------
// File: buffer_matrix_multiplication.cpp
// Desc: Demonstrates matrix multiplication using buffers
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
    // Each work item computes a 4-column by 8-row (8x4) section of the output matrix.
    // The inner loops read in a 1x4 section of matrix B, a 8x1 section of matrix A,
    // and accumulate the partial results for the corresponding 8x4 section of
    // matrix C.
    // The outer loop iterates over the width of matrix A and the height of matrix B
    // to get the complete result.
    __kernel void matmul_8x4_blocks(__global const float *matrix_a,
                                    __global const float *matrix_b,
                                    __global       float *matrix_c,
                                                   int    matrix_b_width,
                                                   int    matrix_a_width)
    {
        const int wid_x = get_global_id(0);
        const int wid_y = get_global_id(1);

        float  a[8];
        float4 b;
        float4 c[8];

        for (int i = 0; i < 8; ++i)
        {
            c[i] = (float4)(0.0f);
        }

        for (int j = 0; j < matrix_a_width; ++j)
        {
            b = vload4(0, matrix_b + j * matrix_b_width + (wid_x * 4));

    #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                a[i] = matrix_a[((wid_y * 8) + i) * matrix_a_width + j];
            }

    #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                c[i] += a[i] * b;
            }
        }

    #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            vstore4(c[i], 0, matrix_c + ((wid_y * 8) + i) * matrix_b_width + (wid_x * 4));
        }
    }

    // The "remainder" version calculates a single element of the output matrix per
    // work item.
    __kernel void matmul_remainder(__global const  float *matrix_a,
                                   __global const  float *matrix_b,
                                   __global        float *matrix_c,
                                                   int    x_rem_start,
                                                   int    y_rem_start,
                                                   int    matrix_b_width,
                                                   int    matrix_a_width)
    {
        const int wid_x = get_global_id(0) + x_rem_start;
        const int wid_y = get_global_id(1) + y_rem_start;

        float c     = 0.0f;
        int   a_idx = matrix_a_width * wid_y;
        int   b_idx = wid_x;

    #pragma unroll 8
        for (int i = 0; i < matrix_a_width; ++i)
        {
            c += matrix_a[a_idx] * matrix_b[b_idx];
            ++a_idx;
            b_idx += matrix_b_width;
        }

        const int c_idx = wid_x + matrix_b_width * wid_y;
        matrix_c[c_idx] = c;
    }
)";

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Please specify input files.\n"
                     "\n"
                     "Usage: " << argv[0] << " <matrix A> <matrix B> [<output file>]\n"
                     "Computes the matrix product C = A * B. See README.md for matrix input format.\n"
                     "There is no size restriction for the matrices. To the extent possible it\n"
                     "calculates the result using an efficient tiled algorithm. For the portion of\n"
                     "the result matrix not covered by tiles it uses a less efficient naive\n"
                     "implementation.\n"
                     "If no file is specified for the output, then it is written to stdout.\n";
        return EXIT_FAILURE;
    }

    const std::string matrix_a_filename(argv[1]);
    const std::string matrix_b_filename(argv[2]);
    const bool        output_to_file = argc >= 4;
    const matrix_t    matrix_a       = load_matrix(matrix_a_filename);
    const matrix_t    matrix_b       = load_matrix(matrix_b_filename);
    const size_t      matrix_a_size  = matrix_a.width * matrix_a.height;
    const size_t      matrix_a_bytes = matrix_a_size * sizeof(cl_float);
    const size_t      matrix_b_size  = matrix_b.width * matrix_b.height;
    const size_t      matrix_b_bytes = matrix_b_size * sizeof(cl_float);
    const std::string output_filename(output_to_file ? argv[3] : "");

    if (matrix_a.width != matrix_b.height)
    {
        std::cerr << "Can't multiply a matrix of dimensions "
                  << matrix_a.width << "x" << matrix_a.height << " "
                  << "by a matrix of dimensions "
                  << matrix_b.width << "x" << matrix_b.height << "\n";
        return EXIT_FAILURE;
    }

    matrix_t matrix_c;
    matrix_c.width  = matrix_b.width;
    matrix_c.height = matrix_a.height;
    const size_t matrix_c_size  = matrix_c.width * matrix_c.height;
    const size_t matrix_c_bytes = matrix_c_size * sizeof(cl_float);
    matrix_c.elements.resize(matrix_c_size);

    cl_wrapper          wrapper;
    cl_program          program       = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel           kernel_8x4    = wrapper.make_kernel("matmul_8x4_blocks", program);
    cl_kernel           kernel_rem    = wrapper.make_kernel("matmul_remainder", program);
    cl_context          context       = wrapper.get_context();
    cl_command_queue    command_queue = wrapper.get_command_queue();
    struct dma_buf_sync buf_sync      = {};
    cl_event            unmap_event   = nullptr;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for dmabuf-backed buffers is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for dmabuf-backed buffers is not supported.\n";
        return EXIT_FAILURE;
    }

    cl_int err = CL_SUCCESS;

    /*
     * Step 1: Create suitable dmabuf-backed buffers.
     */

    /*
     * Matrix A
     */

    cl_mem_dmabuf_host_ptr matrix_a_buf = wrapper.make_buffer(matrix_a_bytes);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(matrix_a_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    std::memcpy(matrix_a_buf.dmabuf_hostptr, matrix_a.elements.data(), matrix_a_bytes);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(matrix_a_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    cl_mem              matrix_a_mem     = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_a_bytes,
            &matrix_a_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix A." << "\n";
        return err;
    }

    /*
     * Matrix B
     */

    cl_mem_dmabuf_host_ptr matrix_b_buf = wrapper.make_buffer(matrix_b_bytes);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(matrix_b_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    std::memcpy(matrix_b_buf.dmabuf_hostptr, matrix_b.elements.data(), matrix_b_bytes);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(matrix_b_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    cl_mem              matrix_b_mem     = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_b_bytes,
            &matrix_b_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix B." << "\n";
        return err;
    }

    /*
     * Matrix C
     */

    cl_mem_dmabuf_host_ptr matrix_c_buf = wrapper.make_buffer(matrix_c_bytes);
    cl_mem              matrix_c_mem     = clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_c_bytes,
            &matrix_c_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix C." << "\n";
        return err;
    }

    /*
     * Step 2: Set up the kernel arguments for tiled kernel.
     */

    err = clSetKernelArg(kernel_8x4, 0, sizeof(matrix_a_mem), &matrix_a_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_8x4, 1, sizeof(matrix_b_mem), &matrix_b_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_8x4, 2, sizeof(matrix_c_mem), &matrix_c_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        return err;
    }

    const cl_int matrix_b_width  = matrix_b.width;
    const cl_int matrix_a_width  = matrix_a.width;

    err = clSetKernelArg(kernel_8x4, 3, sizeof(matrix_b_width), &matrix_b_width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_8x4, 4, sizeof(matrix_a_width), &matrix_a_width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4." << "\n";
        return err;
    }

    /*
     * Step 3: Run the 4x8 tiled kernel.
     */

    const size_t tiled_global_work_size[] = {static_cast<size_t>(matrix_b.width / 4), static_cast<size_t>(matrix_a.height / 8)};
    if (tiled_global_work_size[0] != 0 && tiled_global_work_size[1] != 0)
    {
        err = clEnqueueNDRangeKernel(
                command_queue,
                kernel_8x4,
                2,
                nullptr,
                tiled_global_work_size,
                nullptr,
                0,
                nullptr,
                nullptr
        );
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clEnqueueNDRangeKernel for tiled portion." << "\n";
            return err;
        }
    }

    /*
     * Step 4: Set up and run less efficient kernels for the edges of the result
     *         matrix that weren't covered by the tiled version.
     */

    const cl_int x_rem_start = (matrix_b.width / 4) * 4;
    const cl_int y_rem_start = (matrix_a.height / 8) * 8;

    err = clSetKernelArg(kernel_rem, 0, sizeof(matrix_a_mem), &matrix_a_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_rem, 1, sizeof(matrix_b_mem), &matrix_b_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_rem, 2, sizeof(matrix_c_mem), &matrix_c_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_rem, 3, sizeof(x_rem_start), &x_rem_start);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        return err;
    }

    const cl_int right_y_rem_start = 0;
    err = clSetKernelArg(kernel_rem, 4, sizeof(right_y_rem_start), &right_y_rem_start);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_rem, 5, sizeof(matrix_b_width), &matrix_b_width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 5." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_rem, 6, sizeof(matrix_a_width), &matrix_a_width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 6." << "\n";
        return err;
    }

    /*
     * Covers the remaining right side for the full height of the matrix.
     */

    const size_t right_rem_work_size[] = {static_cast<size_t>(matrix_b.width - x_rem_start), static_cast<size_t>(matrix_a.height)};
    if (right_rem_work_size[0] != 0 && right_rem_work_size[1] != 0)
    {
        err = clEnqueueNDRangeKernel(
                command_queue,
                kernel_rem,
                2,
                nullptr,
                right_rem_work_size,
                nullptr,
                0,
                nullptr,
                nullptr
        );
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clEnqueueNDRangeKernel for right remainder." << "\n";
            return err;
        }
    }

    const cl_int bottom_x_rem_start = 0;
    err = clSetKernelArg(kernel_rem, 3, sizeof(bottom_x_rem_start), &bottom_x_rem_start);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_rem, 4, sizeof(y_rem_start), &y_rem_start);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4." << "\n";
        return err;
    }

    /*
     * Covers the remaining bottom portion of the result matrix not covered above.
     */

    const size_t bottom_rem_work_size[] = {static_cast<size_t>(x_rem_start), static_cast<size_t>(matrix_a.height - y_rem_start)};
    if (bottom_rem_work_size[0] != 0 && bottom_rem_work_size[1] != 0)
    {
        err = clEnqueueNDRangeKernel(
                command_queue,
                kernel_rem,
                2,
                nullptr,
                bottom_rem_work_size,
                nullptr,
                0,
                nullptr,
                nullptr
        );
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clEnqueueNDRangeKernel for bottom remainder." << "\n";
            return err;
        }
    }

    /*
     * Step 5: Copy the data out of the ION buffer.
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
            matrix_c_bytes,
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

    std::memcpy(matrix_c.elements.data(), ptr, matrix_c_bytes);

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
