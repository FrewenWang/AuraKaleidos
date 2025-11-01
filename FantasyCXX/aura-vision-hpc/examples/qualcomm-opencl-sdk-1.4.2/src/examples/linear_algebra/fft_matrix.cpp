//--------------------------------------------------------------------------------------
// File: fft_matrix.cpp
// Desc: Runs a kernel that computes the 2D fast Fourier transform of the input matrix
//       and writes the real and imaginary parts of the output matrices using well-known
//       Cooley-Tukey algorithm.
//
// Author:      QUALCOMM
//
//               Copyright (c) 2018, 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include "util/cl_wrapper.h"
#include "util/util.h"
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

std::string PROGRAM_SOURCE = R"(
    int bitReverse(int ordered, int N)
    {
      int halfN = (N >> 1);

      for (size_t i = 0; i < halfN; i++)
      {
        int j = (N - i - 1);
        ordered = (ordered & (~(1 << i | 1 << j))) | (((ordered & (1 << i)) >> i) << j) | ((ordered & (1 << j)) >> j) << i;
      }
      return ordered;
    }

    void computeButterflyFirstIteration(int i, int local_size, int halfArraySize, int nHalfSize, int log2HalfSize, int log2ArraySize, __global float2* pInData, __local float4* pOutData)
    {

      while (i < halfArraySize)
      {
        int k = 0;
        int nOffset = i * 2;

        // Compute the butterflies
        int k00 = nOffset;
        int k01 = k00 + 1;

        float2 InData0 = pInData[bitReverse(k00, log2ArraySize)];
        float2 InData1 = pInData[bitReverse(k01, log2ArraySize)];
        float4 OutData;

        OutData.x = InData0.x + InData1.x;
        OutData.y = InData0.y + InData1.y;
        OutData.z = InData0.x - InData1.x;
        OutData.w = InData0.y - InData1.y;

        pOutData[k00 >> 1] = OutData;

        i += local_size;
      }
    }

    void computeButterflyIteration(int i, int local_size, int halfArraySize, int nHalfSize, int log2HalfSize, int log2ArraySize, __local float4* pInData, __local float4* pOutData)
    {
      while (i < (halfArraySize >> 1))
      {
        int nBufferlySize = 2 * nHalfSize;
        int k = (2 * i) & (nHalfSize - 1);
        int nOffset = ((2 * i) >> log2HalfSize)*nBufferlySize;

        // Compute the butterflies
        int k00 = nOffset + k;
        int k01 = k00 + nHalfSize;

        float recipBufferlySize = native_recip((float)nBufferlySize);
        float fCos = native_cos(2.0f * M_PI_F * k * recipBufferlySize);
        float fSin = native_sin(2.0f * M_PI_F * k * recipBufferlySize);

        float4 InData = pInData[k01 >> 1];

        float fTmp0 = fCos * InData.x + fSin * InData.y;
        float fTmp1 = fCos * InData.y - fSin * InData.x;

        fCos = native_cos(2.0f * M_PI_F * (k + 1) * recipBufferlySize);
        fSin = native_sin(2.0f * M_PI_F * (k + 1) * recipBufferlySize);

        float fTmp2 = fCos * InData.z + fSin * InData.w;
        float fTmp3 = fCos * InData.w - fSin * InData.z;

        InData = pInData[k00 >> 1];
        float4 OutData;

        OutData.x = InData.x - fTmp0;
        OutData.y = InData.y - fTmp1;
        OutData.z = InData.z - fTmp2;
        OutData.w = InData.w - fTmp3;

        pOutData[k01 >> 1] = OutData;

        OutData.x = InData.x + fTmp0;
        OutData.y = InData.y + fTmp1;
        OutData.z = InData.z + fTmp2;
        OutData.w = InData.w + fTmp3;

        pOutData[k00 >> 1] = OutData;

        i += local_size;
      }
    }

    void computeButterflyLastIteration(int i, int local_size, int halfArraySize, int nHalfSize, int log2HalfSize, int log2ArraySize, __local float4* pInData, __global float4* pOutData)
    {
      while (i < (halfArraySize >> 1))
      {
        int nBufferlySize = 2 * nHalfSize;
        int k = (2 * i) & (nHalfSize - 1);
        int nOffset = ((2 * i) >> log2HalfSize)*nBufferlySize;

        // Compute the butterflies
        int k00 = nOffset + k;
        int k01 = k00 + nHalfSize;

        float recipBufferlySize = native_recip((float)nBufferlySize);
        float fCos = native_cos(2.0f * M_PI_F * k * recipBufferlySize);
        float fSin = native_sin(2.0f * M_PI_F * k * recipBufferlySize);

        float4 InData = pInData[k01 >> 1];

        float fTmp0 = fCos * InData.x + fSin * InData.y;
        float fTmp1 = fCos * InData.y - fSin * InData.x;

        fCos = native_cos(2.0f * M_PI_F * (k + 1) * recipBufferlySize);
        fSin = native_sin(2.0f * M_PI_F * (k + 1) * recipBufferlySize);

        float fTmp2 = fCos * InData.z + fSin * InData.w;
        float fTmp3 = fCos * InData.w - fSin * InData.z;

        InData = pInData[k00 >> 1];
        float4 OutData;

        OutData.x = InData.x - fTmp0;
        OutData.y = InData.y - fTmp1;
        OutData.z = InData.z - fTmp2;
        OutData.w = InData.w - fTmp3;

        pOutData[k01 >> 1] = OutData;

        OutData.x = InData.x + fTmp0;
        OutData.y = InData.y + fTmp1;
        OutData.z = InData.z + fTmp2;
        OutData.w = InData.w + fTmp3;

        pOutData[k00 >> 1] = OutData;

        i += local_size;
      }
    }

    void __kernel FFT2D_LM(int nArraySize, int log2ArraySize, __global float2* pInData, __local float2* pScratch)
    {
      int local_id = get_local_id(0);
      int local_size = get_local_size(0);
      int offset = get_group_id(1) << log2ArraySize;
      int nHalfSize = 1;
      int log2HalfSize = 0;
      int halfArraySize = (nArraySize >> 1);

      computeButterflyFirstIteration(local_id, local_size, halfArraySize, nHalfSize, log2HalfSize, log2ArraySize, pInData + offset, pScratch);
      barrier(CLK_LOCAL_MEM_FENCE);

      nHalfSize = nHalfSize * 2;
      log2HalfSize++;
      for (int iteration = 1; iteration < log2ArraySize - 1; iteration++)
      {
        computeButterflyIteration(local_id, local_size, halfArraySize, nHalfSize, log2HalfSize, log2ArraySize, pScratch, pScratch);
        barrier(CLK_LOCAL_MEM_FENCE);

        nHalfSize = nHalfSize * 2;
        log2HalfSize++;
      }

      computeButterflyLastIteration(local_id, local_size, halfArraySize, nHalfSize, log2HalfSize, log2ArraySize, pScratch, pInData + offset);
    }

    __kernel void MatrixTranspose(const uint rows,
      const uint cols,
      __global float2* matrix,
      __global float2* matrixTranspose)
    {
      const uint i = get_global_id(0) << 1;
      const uint j = get_global_id(1) << 1;

      float4 temp = *(__global float4*)&matrix[mad24(j, cols, i)];
      float4 temp1 = *(__global float4*)&matrix[mad24(j, cols, i) + cols];

      *(__global float4*)&matrixTranspose[mad24(i, rows, j)] = (float4)(temp.s01, temp1.s01);
      *(__global float4*)&matrixTranspose[mad24(i, rows, j) + rows] = (float4)(temp.s23, temp1.s23);
    }
)";

/*
* Takes the difference between event profiling times (in nanoseconds) and prints the time delta
* in microseconds.
*/
double print_time_delta(const char *message, cl_ulong start, cl_ulong end)
{
    double delta_ns = (double)(end - start);
    double delta_us = delta_ns / 1000.0;
    std::cout << message << delta_us << "microseconds" << std::endl;
    return delta_us;
}

double print_execution_time(cl_event recording_dispatch_event)
{
    cl_ulong start, end;
    clGetEventProfilingInfo(recording_dispatch_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),&start, nullptr);
    clGetEventProfilingInfo(recording_dispatch_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    return print_time_delta("Start to end was ", start, end);
}

static bool is_power_of_2(size_t n);

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Please specify source and output matrices.\n"
                     "\n"
                     "Usage: " << argv[0] << " <source> <real output> <imaginary output>\n"
                     "Runs a kernel that computes the 2D fast Fourier transform of the matrix"
                     "<source>, and writes the real and imaginary parts of the output to the matrices\n"
                     "<real output> and <imaginary output>, respectively. We use the well-known\n"
                     "Cooley-Tukey algorithm.\n"
                     "The matrix must have width = height = a power of 2.\n";
        return EXIT_FAILURE;
    }
    const std::string src_matrix_filename(argv[1]);
    const std::string real_out_filename(argv[2]);
    const std::string imag_out_filename(argv[3]);

    cl_wrapper          wrapper;
    struct dma_buf_sync buf_sync    = {};
    cl_event            unmap_event = nullptr;

    // Optimized kernel from HW team
    const char *program_src = PROGRAM_SOURCE.c_str();
    cl_program program     = wrapper.make_program(&program_src, 1);
    cl_kernel fft2d = wrapper.make_kernel("FFT2D_LM", program);
    cl_kernel matrix_transpose = wrapper.make_kernel("MatrixTranspose", program);
    size_t fft2d_maxwg = wrapper.get_max_workgroup_size(fft2d);

    cl_context context         = wrapper.get_context();
    complex_matrix_t  src_matrix = load_real_matrix_as_complex_matrix(src_matrix_filename);

    if ((src_matrix.width != src_matrix.height)
        || !is_power_of_2(src_matrix.width))
    {
        std::cerr << "For this example, the width and height of the input matrix must be equal, and\n"
                  << "they must both be a power of 2. " << src_matrix_filename << " is " << src_matrix.width << "x"
                  << src_matrix.height << ".\n";
        return EXIT_FAILURE;
    }

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
     * Step 1: Create suitable DMA buffer-backed CL images.
     */

    const size_t        src_matrix_bytes = src_matrix.width * src_matrix.height * sizeof(cl_float2);
    cl_mem_dmabuf_host_ptr src_mem      = wrapper.make_buffer(src_matrix_bytes);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(src_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    std::memcpy(src_mem.dmabuf_hostptr, src_matrix.elements.data(), src_matrix_bytes);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(src_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    cl_int err;
    cl_mem src_matrix_mem = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            src_matrix_bytes,
            &src_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for source image." << "\n";
        return err;
    }

    cl_mem_dmabuf_host_ptr out_mem = wrapper.make_buffer(src_matrix_bytes);
    cl_mem out_matrix_mem = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            src_matrix_bytes,
            &out_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for real output matrix." << "\n";
        return err;
    }

    /*
     * Step 2: Set up and run the row- and column-pass kernels.
     */
    cl_int width = static_cast<cl_int>(src_matrix.width);
    cl_int height = static_cast<cl_int>(src_matrix.height);
    cl_int log_w = static_cast<cl_int>(std::log2(width));

    err = clSetKernelArg(fft2d, 0, sizeof(width), &width);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        return err;
    }

    err = clSetKernelArg(fft2d, 1, sizeof(log_w), &log_w);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        return err;
    }

    err = clSetKernelArg(fft2d, 2, sizeof(src_matrix_mem), &src_matrix_mem);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        return err;
    }

    err = clSetKernelArg(fft2d, 3, sizeof(cl_float2) * std::max(width, height), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4." << "\n";
        return err;
    }

    const cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue command_queue  = clCreateCommandQueueWithProperties(context, wrapper.get_device_id(),
                                                                         properties, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " Could not create command queue" << "\n";
        return err;
    }

    cl_event row_pass_time = nullptr;
    cl_event transpose_1_time = nullptr;
    cl_event transpose_2_time = nullptr;

    const size_t fft2d_global_work_size[]  = {std::min(static_cast<size_t>(fft2d_maxwg), static_cast<size_t>(width)), static_cast<size_t>(src_matrix.height)};
    const size_t fft2d_local_work_size[]  = {std::min(static_cast<size_t>(fft2d_maxwg), static_cast<size_t>(width)), 1};

    err = clEnqueueNDRangeKernel(
            command_queue,
            fft2d,
            2,
            nullptr,
            fft2d_global_work_size,
            fft2d_local_work_size,
            0,
            nullptr,
            &row_pass_time
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel." << "\n";
        return err;
    }

    // Now transpose the matrix
    err = clSetKernelArg(matrix_transpose, 0, sizeof(height), &height);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        return err;
    }

    err = clSetKernelArg(matrix_transpose, 1, sizeof(width), &width);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        return err;
    }

    err = clSetKernelArg(matrix_transpose, 2, sizeof(src_matrix_mem), &src_matrix_mem);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        return err;
    }

    err = clSetKernelArg(matrix_transpose, 3, sizeof(out_matrix_mem), &out_matrix_mem);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        return err;
    }

    const size_t transpose_global_size[] = {static_cast<size_t>(width >> 1), static_cast<size_t>(height >> 1)};
    const size_t transpose_local_size[]  = {4, 128};
    err = clEnqueueNDRangeKernel(
            command_queue,
            matrix_transpose,
            2,
            nullptr,
            transpose_global_size,
            transpose_local_size,
            0,
            nullptr,
            &transpose_1_time
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel." << "\n";
        return err;
    }

    // Exactly the same as before but on output
    err = clSetKernelArg(fft2d, 2, sizeof(out_matrix_mem), &out_matrix_mem);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        return err;
    }

    cl_event col_pass_time;

    err = clEnqueueNDRangeKernel(
            command_queue,
            fft2d,
            2,
            nullptr,
            fft2d_global_work_size,
            fft2d_local_work_size,
            0,
            nullptr,
            &col_pass_time
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel." << "\n";
        return err;
    }

    //Transpose
    err = clSetKernelArg(matrix_transpose, 2, sizeof(out_matrix_mem), &out_matrix_mem);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2." << "\n";
        return err;
    }

    err = clSetKernelArg(matrix_transpose, 3, sizeof(src_matrix_mem), &src_matrix_mem);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3." << "\n";
        return err;
    }

    err = clEnqueueNDRangeKernel(
            command_queue,
            matrix_transpose,
            2,
            nullptr,
            transpose_global_size,
            transpose_local_size,
            0,
            nullptr,
            &transpose_2_time
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel." << "\n";
        return err;
    }

    /*
     * Step 3: Copy the data out of the DMA buffer
     */
    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if ( ioctl(src_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }
    complex_matrix_t out_info;
    out_info.width  = src_matrix.width;
    out_info.height = src_matrix.height;
    out_info.elements.resize(out_info.width * out_info.height);
    cl_float *mat_ptr    = nullptr;
    mat_ptr = static_cast<cl_float *>(clEnqueueMapBuffer(
            command_queue,
            src_matrix_mem,
            CL_BLOCKING,
            CL_MAP_READ,
            0,
            src_matrix_bytes,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping real part output matrix for reading." << "\n";
        return err;
    }

    std::memcpy(out_info.elements.data(), mat_ptr, src_matrix_bytes);

    err = clEnqueueUnmapMemObject(command_queue, src_matrix_mem, mat_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping real part output matrix." << "\n";
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
    if ( ioctl(src_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    clFinish(command_queue);

    double total_gpu_time = 0.0;
    std::cout << "Row pass times" << std::endl;
    total_gpu_time += print_execution_time(row_pass_time);
    std::cout << "Col pass times" << std::endl;
    total_gpu_time += print_execution_time(col_pass_time);

    std::cout << "Transpose times are " << std::endl;
    total_gpu_time += print_execution_time(transpose_1_time);
    total_gpu_time += print_execution_time(transpose_2_time);

    std::cout << "Total GPU time = " << total_gpu_time << std::endl;

    save_matrix(real_out_filename, imag_out_filename, out_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseMemObject(src_matrix_mem);
    clReleaseMemObject(out_matrix_mem);
    clReleaseEvent(row_pass_time);
    clReleaseEvent(col_pass_time);
    clReleaseEvent(transpose_1_time);
    clReleaseEvent(transpose_2_time);

    return EXIT_SUCCESS;
}

bool is_power_of_2(size_t n)
{
    return n && !(n & (n - 1));
}
