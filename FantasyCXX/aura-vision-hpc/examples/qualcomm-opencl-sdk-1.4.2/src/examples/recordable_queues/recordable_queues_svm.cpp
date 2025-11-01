//--------------------------------------------------------------------------------------
// File: recordable_queues_svm.cpp
// Desc: Demonstrates how to use the recordable queues extension for CPU efficient
//       dispatch of multiple commands. This example multiplies two matrices then
//       repeatedly increment the result.
//
// Author:      QUALCOMM
//
//               Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>
#include "util/cl_wrapper.h"
#include "util/util.h"

#define CL_ASSERT_ERRCODE(err)                                                                  \
    if ((err) != CL_SUCCESS)                                                                    \
    {                                                                                           \
        std::cerr << "OpenCL API error code " << (err) << " on line " << __LINE__ << "\n";      \
        std::exit(err);                                                                         \
    }

static const char *PROGRAM_SOURCE = R"(
    // C := alpha*op(A)*op(B) + beta*C
    __kernel void sgemm_alpha_beta_B_trans_Ab_B(__global const int *A,
                                                __global const int *B,
                                                __global int *C,
                                                const int m,
                                                const int n,
                                                const int k,
                                                const int alpha,
                                                const int beta)
    {
        int pos;
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);
        if ((gx < n) && (gy < m))
        {
            int a;
            int b;
            int c = beta * C[gy * n + gx];
            for (pos = 0; pos < k; pos += 1 )
            {
                b = B[pos * n + gx];
                a = A[gy * k + pos];
                c += alpha * (a * b);
            }
            C[gy * n + gx] = c;
        }
    }

    __kernel void inc_square_matrix(__global int *A, const int m, const int i)
    {
        const int gx = get_global_id(0);
        const int gy = get_global_id(1);
        if ((gx < m) && (gy < m))
        {
            A[gy * m + gx] = A[gy * m + gx] + i;
        }
    }
)";

/*
 * Computes the local work size for an m by n matrix. The calculation ensures the local work size is
 * never too large.
 */
void compute_local_size(cl_kernel kernel, cl_device_id device, size_t m, size_t n, size_t *ret_local_size)
{
    size_t dim_0 = std::min(m, n);
    size_t dim_1 = 0;
    size_t max_work_group_size = 0;
    int err;

    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
    CL_ASSERT_ERRCODE(err);

    max_work_group_size = std::min(max_work_group_size, m*n);

    dim_1 = std::max((size_t)1, max_work_group_size/dim_0);

    ret_local_size[0] = dim_0;
    ret_local_size[1] = dim_1;
    ret_local_size[2] = 1;
}

/*
 * Takes the difference between event profiling times (in nanoseconds) and prints the time delta
 * in microseconds.
 */
void print_time_delta(const char *message, cl_ulong start, cl_ulong end)
{
    double delta_ns = (double)(end - start);
    double delta_us = delta_ns / 1000.0;
    std::cout << message << delta_us << "microseconds" << std::endl;
}

void print_execution_time(cl_event recording_dispatch_event)
{
    cl_ulong queued, submit, start, end;

    clGetEventProfilingInfo(recording_dispatch_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, nullptr);
    clGetEventProfilingInfo(recording_dispatch_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, nullptr);
    clGetEventProfilingInfo(recording_dispatch_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    clGetEventProfilingInfo(recording_dispatch_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);

    print_time_delta("Queued to submit was ", queued, submit);
    print_time_delta("Submit to start was ", submit, start);
    print_time_delta("Start to end was ", start, end);
}

/*
 * Takes a matrix, its size in bytes, and dimension (the matrix is assumed to be square),
 * maps the result, then prints the formatted matrix to stdout.
 */
void map_and_print_memobj_matrix(const char *message, cl_command_queue command_queue, cl_mem C,
                                 cl_uint matrix_bytes, cl_event *event_wait_list, int matrix_dim, int fd)
{
    // Ensure cache coherency.
    struct dma_buf_sync buf_sync    = {};
    int                 err         = CL_SUCCESS;
    cl_event            map_event   = nullptr;
    cl_event            unmap_event = nullptr;

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if ( ioctl(fd, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        CL_ASSERT_ERRCODE(errno);
    }

    cl_int *C_ptr = static_cast<cl_int *>(clEnqueueMapBuffer(
            command_queue,
            C,
            CL_NON_BLOCKING,
            CL_MAP_READ,
            0,
            matrix_bytes,
            event_wait_list ? 1 : 0,
            event_wait_list,
            &map_event,
            &err
    ));
    CL_ASSERT_ERRCODE(err);

    clWaitForEvents(1, &map_event);

    std::cout << "Result matrix " << message << ":" << std::endl;

    // Print result matrix
    for (int row = 0; row < matrix_dim; row++)
    {
        std::cout << "| ";
        for (int col = 0; col < matrix_dim; col++)
        {
            std::cout << C_ptr[row*matrix_dim + col] << " ";
        }
        std::cout << "|" << std::endl;
    }

    // Unmap buffer
    err = clEnqueueUnmapMemObject(command_queue, C, C_ptr, 0, nullptr, &unmap_event);
    CL_ASSERT_ERRCODE(err);

    err = clWaitForEvents(1, &unmap_event);
    CL_ASSERT_ERRCODE(err);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    if ( ioctl(fd, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        CL_ASSERT_ERRCODE(errno);
    }

    // Release event
    err = clReleaseEvent(map_event);
    CL_ASSERT_ERRCODE(err);
    err = clReleaseEvent(unmap_event);
    CL_ASSERT_ERRCODE(err);
}

void map_and_print_svm_matrix(const char *message, cl_command_queue command_queue, cl_int* C_data,
                                 cl_uint matrix_bytes, cl_event *event_wait_list, int matrix_dim)
{
    int err = CL_SUCCESS;
    cl_event map_event = nullptr;

    err = clEnqueueSVMMap(command_queue, CL_NON_BLOCKING, CL_MAP_READ, C_data, matrix_bytes, event_wait_list ? 1 : 0, event_wait_list, &map_event);
    CL_ASSERT_ERRCODE(err);

    clWaitForEvents(1, &map_event);

    std::cout << "Result matrix " << message << ":" << std::endl;

    // Print result matrix
    for (int row = 0; row < matrix_dim; row++)
    {
        std::cout << "| ";
        for (int col = 0; col < matrix_dim; col++)
        {
            std::cout << C_data[row*matrix_dim + col] << " ";
        }
        std::cout << "|" << std::endl;
    }

    // Unmap buffer
    err = clEnqueueSVMUnmap(command_queue, C_data, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(err);

    // Release event
    err = clReleaseEvent(map_event);
    CL_ASSERT_ERRCODE(err);
}

int main(int argc, char** argv)
{
    size_t global_size[3] = {0};
    size_t gemm_kernel_local_size[3] = {0};
    size_t inc_kernel_local_size[3] = {0};
    const int max_matrix_dim = 64;
    cl_mem C;
    cl_int k = 3;
    cl_int m = 2;
    cl_int n = m;
    cl_int alpha = 1;
    cl_int beta = 0;
    cl_int inc = 0;
    cl_int inc_by_amount = 1;
    int svm_fine_grain_flag = 0;
    cl_platform_id platform;
    cl_device_id device;
    size_t matrix_bytes;
    size_t matrix_c_bytes;
    static const char *HELP_MESSAGE = "\n"
                                      "Usage: recordable_queues <m> <k> <inc>\n"
                                      "Outputs a matrix of size <m> x <m>.  Computes the matrix product C=A*B\n"
                                      "then increments C <inc> number of times.\n"
                                      "As a formula the result is A*B + <inc>, where <inc> is an integer added to all elements.\n"
                                      "An initial sequence of kernels is recorded as: Matrix Multiply A*B, increment_1,...,increment_<inc>\n"
                                      "The initial sequence is dispatched.  A new B matrix is created which is twice as large as the original,\n"
                                      "then the recording is updated and dispatched with the new B. This sequence looks like:\n"
                                      "Matrix Multiply A*(2*B), increment_1,...,increment_<inc>\n";

    std::cout << HELP_MESSAGE;

    // Read <m>
    if(argc >= 2)
    {
        m = atoi(argv[1]);
        if(m < 1 || m > max_matrix_dim)
        {
            std::cerr << "<m> must be greater than or equal to 1 and no larger than " << max_matrix_dim << std::endl;
            std::exit(EXIT_FAILURE);
        }
        n = m;
    }

    // Read <k>
    if(argc >= 3)
    {
        k = atoi(argv[2]);
        if(k < 1 || m > max_matrix_dim)
        {
            std::cerr << "<k> must be greater than or equal to 1 and no larger than " << max_matrix_dim << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // Read <inc>
    if(argc >= 4)
    {
        inc = atoi(argv[3]);
        if(inc < 0 || m > max_matrix_dim)
        {
            std::cerr << "<inc> must be greater than or equal to 0 and no larger than " << max_matrix_dim << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    if(argc >= 5)
    {
        std::cerr << "Too many arguments." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << std::endl << "<m>=" << m << " <n>=" << n << " <k>=" << k << std::endl;

    // Init global work size
    global_size[0] = m;
    global_size[1] = m;

    cl_wrapper       wrapper;
    cl_program       program       = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel        gemm_kernel   = wrapper.make_kernel("sgemm_alpha_beta_B_trans_Ab_B", program);
    cl_kernel        inc_kernel    = wrapper.make_kernel("inc_square_matrix", program);
    cl_context       context       = wrapper.get_context();
    cl_command_queue command_queue = nullptr;
    size_t           dummy_global_size[3] = {1, 1, 1};
    size_t           dummy_local_size[3] = {1, 1, 1};
    cl_int           err;
    const cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for dmabuf-backed buffers is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    if (!wrapper.check_extension_support("cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for dmabuf-backed buffers is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    if (!wrapper.check_extension_support("cl_qcom_recordable_queues"))
    {
        std::cerr << "Extension cl_qcom_recordable_queues needed.\n";
        std::exit(EXIT_FAILURE);
    }

    cl_device_svm_capabilities svm_capabilities = 0;

    // Query device ID
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetPlatformIDs." << "\n";
        std::exit(err);
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetDeviceIDs." << "\n";
        std::exit(err);
    }
    command_queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    CL_ASSERT_ERRCODE(err);

    // Check if fine-grain SVM is supported
    err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &svm_capabilities, 0);
    CL_ASSERT_ERRCODE(err);
    if(CL_DEVICE_SVM_FINE_GRAIN_BUFFER & svm_capabilities)
    {
        svm_fine_grain_flag = CL_MEM_SVM_FINE_GRAIN_BUFFER;
    }

    // Calculate matrix sizes
    matrix_bytes = k*m*sizeof(cl_int);
    matrix_c_bytes = m*m*sizeof(cl_int);

    // Read max recordable queue size
    cl_uint max_recordable_queue_size = 0;

    err = clGetDeviceInfo(device, CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE, sizeof(cl_uint),
                          &max_recordable_queue_size, nullptr);
    CL_ASSERT_ERRCODE(err);

    // Check if the user tried to record too many kernels
    if((cl_uint)inc + 1u > max_recordable_queue_size)
    {
        std::cerr << "Cannot record more than " << max_recordable_queue_size << "kernels\n";
        std::exit(EXIT_FAILURE);
    }

    // Calculate local size
    compute_local_size(gemm_kernel, device, m, k, gemm_kernel_local_size);
    compute_local_size(inc_kernel, device, m, k, inc_kernel_local_size);

    /*
     * Step 1: Create suitable SVM and ION backed buffers.
     */

    /*
     * Matrix A
     */

    cl_int *A_data = static_cast<cl_int *>(clSVMAlloc(context, CL_MEM_READ_ONLY|svm_fine_grain_flag, matrix_bytes, 0));
    if(A_data == nullptr)
    {
        std::cerr << "Error " << err << " with clSVMAlloc for matrix A." << "\n";
        std::exit(err);
    }

    /*
     * Matrix B
     */

    cl_int *B_data = static_cast<cl_int *>(clSVMAlloc(context, CL_MEM_READ_ONLY|svm_fine_grain_flag, matrix_bytes, 0));
    if (B_data == nullptr)
    {
        std::cerr << "Error " << err << " with clSVMAlloc for matrix B." << "\n";
        std::exit(err);
    }

    /*
     * Matrix Bx2
     */

    cl_int *Bx2_data = static_cast<cl_int *>(clSVMAlloc(context, CL_MEM_READ_ONLY|svm_fine_grain_flag, matrix_bytes, 0));
    if (Bx2_data == nullptr)
    {
        std::cerr << "Error " << err << " with clSVMAlloc for matrix Bx2." << "\n";
        std::exit(err);
    }

    /*
     * Matrix C
     */

    cl_mem_dmabuf_host_ptr matrix_c_buf = wrapper.make_buffer(matrix_c_bytes);
    C     = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            matrix_c_bytes,
            &matrix_c_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix C." << "\n";
        std::exit(err);
    }

    err = clEnqueueSVMMap(command_queue, CL_BLOCKING, CL_MAP_WRITE, A_data, matrix_bytes, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(err);

    err = clEnqueueSVMMap(command_queue, CL_BLOCKING, CL_MAP_WRITE, B_data, matrix_bytes, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(err);

    err = clEnqueueSVMMap(command_queue, CL_BLOCKING, CL_MAP_WRITE, Bx2_data, matrix_bytes, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(err);

    // Initialize matrices
    for (int i = 0; i < k*m; i++)
    {
        A_data[i]= i;
        B_data[i]= 1;
        Bx2_data[i] = 2*B_data[i];
    }

    err = clEnqueueSVMUnmap(command_queue, A_data, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(err);

    err = clEnqueueSVMUnmap(command_queue, B_data, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(err);

    err = clEnqueueSVMUnmap(command_queue, Bx2_data, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(err);

    /*
     * Step 2: Set up the kernel arguments for the GEMM kernel. We are passing in nullptr for now for matrix A,
     * which is allowed by the OpenCL specification.  This is done with the knowledge that this argument
     * will be updated to a valid cl_mem object when the recording is enqueued.
     */

    err = clSetKernelArg(gemm_kernel, 0, sizeof(cl_mem),  nullptr);
    CL_ASSERT_ERRCODE(err);

    err = clSetKernelArgSVMPointer(gemm_kernel, 1, B_data);
    CL_ASSERT_ERRCODE(err);

    err = clSetKernelArg(gemm_kernel, 2, sizeof(cl_mem), &C);
    CL_ASSERT_ERRCODE(err);

    err = clSetKernelArg(gemm_kernel, 3, sizeof(cl_int), &m);
    CL_ASSERT_ERRCODE(err);

    err = clSetKernelArg(gemm_kernel, 4, sizeof(cl_int), &n);
    CL_ASSERT_ERRCODE(err);

    err = clSetKernelArg(gemm_kernel, 5, sizeof(cl_int), &k);
    CL_ASSERT_ERRCODE(err);

    err = clSetKernelArg(gemm_kernel, 6, sizeof(cl_int), &alpha);
    CL_ASSERT_ERRCODE(err);

    err = clSetKernelArg(gemm_kernel, 7, sizeof(cl_int), &beta);
    CL_ASSERT_ERRCODE(err);

    /*
     *  Step 3: Create recordable command queue and create new recording
     */
    cl_command_queue recordable_queue = clCreateCommandQueue(context, device, CL_QUEUE_RECORDABLE_QCOM, &err);
    CL_ASSERT_ERRCODE(err);

    cl_recording_qcom recording = clNewRecordingQCOM(recordable_queue, &err);
    CL_ASSERT_ERRCODE(err);


    /*
     * Step 4: Record GEMMM kernel, with dummy_global_work size.  Use clEnqueueRecordingQCOM to set the global work size
     */

    err = clEnqueueNDRangeKernel(recordable_queue, gemm_kernel, 2, nullptr, global_size, gemm_kernel_local_size, 0, nullptr, nullptr);
    CL_ASSERT_ERRCODE(err);

    /*
     * Step 5: Record the increment kernel N times.
     */

    // Record the first increment kernel to increment by 0 instead of 1, as well as have an incorrect global size,
    // and sub-optimal local size.  These will be set with the recordable commands queues API
    inc_by_amount = 0;

    if(inc > 0)
    {
        err = clSetKernelArg(inc_kernel, 0, sizeof(cl_mem), &C);
        CL_ASSERT_ERRCODE(err);

        err = clSetKernelArg(inc_kernel, 1, sizeof(cl_int), &m);
        CL_ASSERT_ERRCODE(err);

        err = clSetKernelArg(inc_kernel, 2, sizeof(cl_int), &inc_by_amount);
        CL_ASSERT_ERRCODE(err);

        err = clEnqueueNDRangeKernel(recordable_queue, inc_kernel, 2, nullptr, dummy_global_size, dummy_local_size, 0,
                                     nullptr, nullptr);
        CL_ASSERT_ERRCODE(err);
    }

    // Set N-1 kernels with the correct ndrange and kernel arguments and work size
    inc_by_amount = 1;
    for(int i = 1; i < inc; i++)
    {
        err = clSetKernelArg(inc_kernel, 0, sizeof(cl_mem), &C);
        CL_ASSERT_ERRCODE(err);

        err = clSetKernelArg(inc_kernel, 1, sizeof(cl_int), &m);
        CL_ASSERT_ERRCODE(err);

        err = clSetKernelArg(inc_kernel, 2, sizeof(cl_int), &inc_by_amount);
        CL_ASSERT_ERRCODE(err);

        err = clEnqueueNDRangeKernel(recordable_queue, inc_kernel, 2, nullptr, global_size, inc_kernel_local_size, 0, nullptr, nullptr);
        CL_ASSERT_ERRCODE(err);
    }

    err = clEndRecordingQCOM(recording);
    CL_ASSERT_ERRCODE(err);

    // Prepare array to tell clEnqueueRecordingSVMQCOM how to set svm arguments
    cl_array_arg_qcom update_kernel_svm_args[1] = {          {0 /* dispatch_index */,
                                                              0 /* arg_index */,
                                                              0 /* arg_size, always 0 for SVM */,
                                                              A_data /* arg_value */}};

    // Prepare array to tell clEnqueueRecordingQCOM how to update kernel arguments
    cl_array_arg_qcom update_kernel_args[1] = {       {1 /* dispatch_index */,
                                                       2 /* arg_index */,
                                                       sizeof(inc_by_amount) /* arg_size */,
                                                       &inc_by_amount /* arg_value */}};

    // Prepare array to tell clEnqueueRecordingQCOM how to update global work size
    cl_workgroup_qcom update_global_size[1] = {
            {
                    1 /* dispatch_index */,
                    global_size /* workgroup_size */
            }
    };

    // Prepare array to tell clEnqueueRecordingQCOM how to update inc_kernel's local work size
    cl_workgroup_qcom update_local_size[1] = {
            {
                    1 /* dispatch_index */,
                    inc_kernel_local_size /* work group_size */
            }
    };

    /*
     * Step 6:  Enqueue Recording
     */

    cl_event recording_dispatch_event = nullptr; // rename

    err = clEnqueueRecordingSVMQCOM(command_queue, recording,
                                 inc > 0 ? 1 : 0, inc > 0? update_kernel_args : nullptr,
                                 1, update_kernel_svm_args,
                                 0, nullptr,
                                 inc > 0 ? 1 : 0, inc > 0? update_global_size : nullptr,
                                 inc > 0 ? 1 : 0, inc > 0? update_local_size : nullptr,
                                 0, nullptr,
                                 0, nullptr, &recording_dispatch_event);
    CL_ASSERT_ERRCODE(err);

    /*
     * Step 7: Map and print matrix C to stdout
     */

    map_and_print_svm_matrix("A =", command_queue, A_data, matrix_bytes, nullptr, m);
    map_and_print_svm_matrix("B =", command_queue, B_data, matrix_bytes, nullptr, m);
    map_and_print_memobj_matrix("C = AB", command_queue, C, matrix_c_bytes, &recording_dispatch_event, m, matrix_c_buf.dmabuf_filedesc);
    print_execution_time(recording_dispatch_event);

    /*
     * Step 8: Print profiling times
     */

    cl_ulong queued, submit, start, end;

    clGetEventProfilingInfo(recording_dispatch_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, nullptr);
    clGetEventProfilingInfo(recording_dispatch_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, nullptr);
    clGetEventProfilingInfo(recording_dispatch_event, CL_PROFILING_COMMAND_START,  sizeof(cl_ulong), &start,  nullptr);
    clGetEventProfilingInfo(recording_dispatch_event, CL_PROFILING_COMMAND_END,    sizeof(cl_ulong), &end,    nullptr);

    print_time_delta("Queued to submit was ", queued, submit);
    print_time_delta("Submit to start was ", submit, start);
    print_time_delta("Start to end was ", start, end);

    /*
     * Step 9: Prepare array to set matrix Bx2 on argument 1
     */

    update_kernel_svm_args[0].dispatch_index    = 0;
    update_kernel_svm_args[0].arg_size          = 0;
    update_kernel_svm_args[0].arg_index         = 1;
    update_kernel_svm_args[0].arg_value         = Bx2_data;

    /*
     * Step 10: Enqueue Recording with the first dispatch index performing C = A*Bx2
     */

    err = clEnqueueRecordingSVMQCOM(command_queue, recording,
                                    0, nullptr,
                                    1, update_kernel_svm_args,
                                    0, nullptr,
                                    0, nullptr,
                                    0, nullptr,
                                    0, nullptr,
                                    0, nullptr, &recording_dispatch_event);
    CL_ASSERT_ERRCODE(err);

    // use clFinish and ignore the event this time
    err = clFinish(command_queue);
    CL_ASSERT_ERRCODE(err);

    /*
     * Step 11: Map and print matrix C again, expecting GEMM kernel to give C=A*Bx2
     */

    map_and_print_svm_matrix("A =", command_queue, A_data, matrix_bytes, nullptr, m);
    map_and_print_svm_matrix("B =", command_queue, Bx2_data, matrix_bytes, nullptr, m);
    map_and_print_memobj_matrix("C=A*B", command_queue, C, matrix_c_bytes, &recording_dispatch_event, m, matrix_c_buf.dmabuf_filedesc);
    print_execution_time(recording_dispatch_event);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseEvent(recording_dispatch_event);
    clSVMFree(context, A_data);
    clSVMFree(context, B_data);
    clReleaseMemObject(C);
    clReleaseRecordingQCOM(recording);
    clReleaseCommandQueue(recordable_queue);
    clReleaseCommandQueue(command_queue);

    return 0;
}
