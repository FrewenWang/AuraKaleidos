//--------------------------------------------------------------------------------------
// File: svm_matrix_multiplication.cpp
// Desc: Take input matrices A and B and calculate the product C several times on the
//       device. Between each multiplication, the host copies C to A. Synchronization is
//       achieved with shared virtual memory and locks instead of the more expensive
//       clFinish.
//
// Author:      QUALCOMM
//
//          Copyright (c) 2019, 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <atomic>
#include "util/cl_wrapper.h"
#include "util/util.h"
#include <CL/cl.h>

static const char *PROGRAM_SOURCE = R"(
    #define PROCEED_NEXT_MULTIPLICATION -1
    #define C_BLOCK_ROWS 8
    #define C_BLOCK_COLUMNS 4

    // Each work-item computes a 8 row by 4 column block of matrix_c = matrix_a * matrix_b.
    __kernel void matmul_8x4_blocks(__constant  float         *matrix_a,
                                    __constant  float         *matrix_b,
                                    __global    float         *matrix_c,
                                                int            matrix_a_width,
                                                int            matrix_b_width,
                                    __global    atomic_int    *done,
                                                int            iteration)
    {
        const int c_x_block       = get_global_id(0);
        const int c_y_block       = get_global_id(1);
        const int local_id        = get_local_linear_id();
        float     a[C_BLOCK_ROWS]; // 8 rows by 1 column of matrix A used in computing a partial element of matrix C.
        float4    b; // 1 row by 4 columns of matrix B used in computing a partial element of matrix C.
        float4    c[C_BLOCK_ROWS]; // 8 rows by 4 columns of matrix C computed by this work-item.

        // Wait for the host to copy the previously calculated matrix C to matrix A for this iteration.
        if(local_id == 0)
        {
            while(atomic_load_explicit(
                    &done[iteration - 1],
                    memory_order_acquire, // Also make non-atomic shared virtual memory updates visible here.
                    memory_scope_all_svm_devices) != PROCEED_NEXT_MULTIPLICATION)
            {
            }
        }
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);

        // Start with a zeroed block of scratch memory for matrix C.
        for (int c_block_y = 0; c_block_y < C_BLOCK_ROWS; ++c_block_y)
        {
            c[c_block_y] = (float4)(0.0f);
        }

        // The inner loops read in a 8 row by 1 column block of matrix A, a 1 row by 4 column block of matrix B,
        // and accumulate the partial results for the corresponding 8 row by 4 columns block of matrix C.
        // The outer loop iterates over the width of matrix A and the height of matrix B to get the complete result.
        for (int a_x = 0; a_x < matrix_a_width; ++a_x)
        {
            for (int c_block_y = 0; c_block_y < C_BLOCK_ROWS; ++c_block_y)
            {
                a[c_block_y] = matrix_a[(c_y_block * C_BLOCK_ROWS + c_block_y) * matrix_a_width + a_x];
            }
            b = vload4(
                    0, // offset
                    matrix_b + a_x * matrix_b_width + c_x_block * C_BLOCK_COLUMNS); // address
            for (int c_block_y = 0; c_block_y < C_BLOCK_ROWS; ++c_block_y)
            {
                c[c_block_y] += a[c_block_y] * b;
            }
        }

        // Store the completed block in matrix C.
        for (int c_block_y = 0; c_block_y < C_BLOCK_ROWS; ++c_block_y)
        {
            vstore4(
                    c[c_block_y], // source
                    0, // destination offset
                    matrix_c + (c_y_block * C_BLOCK_ROWS + c_block_y) * matrix_b_width + c_x_block * C_BLOCK_COLUMNS); //destination address
        }

        // Notify the host that this work-group is done. Once all are done, the host will proceed with copying
        // the freshly calculated matrix C to matrix A for the next iteration to use.
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        if(local_id == 0)
        {
            bool success = false;
            while (!success)
            {
                int expected = atomic_load(&done[iteration]);
                int desired = expected + 1;
                success = atomic_compare_exchange_weak_explicit(
                        &done[iteration],
                        &expected,
                        desired,
                        memory_order_release, // On success, also make non-atomic shared virtual memory updates visible elsewhere.
                        memory_order_relaxed, // On failure, don't worry about making non-atomic shared virtual memory updates visible elsewhere.
                        memory_scope_all_svm_devices);
            }
        }
    }
)";


int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6)
    {
        std::cerr <<
                "Please specify input files and local size.\n"
                "Usage:\n" <<
                argv[0] << " <matrix A> <matrix B> <local size x> <local size y> [<output file>]\n"
                "\n"
                "Calculates the product of <matrix A> and <matrix B> as matrix C several times on the device. Between\n"
                "each multiplication, the host copies C to A. Matrix A's number of rows must be divisible by 8 and\n"
                "matrix B's number of columns must be divisible by 4. See README.md for the matrix input format.\n"
                "<local size x> and <local size y> are the number of work-items per work-group.\n"
                "If no <output file> is specified, stdout is used.\n";
        return EXIT_FAILURE;
    }

    const bool        output_to_file = argc >= 6;
    matrix_t          matrix_a       = load_matrix(argv[1]);
    matrix_t          matrix_b       = load_matrix(argv[2]);
    const size_t      matrix_a_bytes = matrix_a.width * matrix_a.height * sizeof(cl_float);
    const size_t      matrix_b_bytes = matrix_b.width * matrix_b.height * sizeof(cl_float);
    const std::string output_filename(output_to_file ? argv[5] : "");

    if (matrix_a.width != matrix_b.height)
    {
        std::cerr
                << "Can't multiply matrix A of width " << matrix_a.width
                << " by matrix B of height " << matrix_b.height << ".\n"
                << "These dimensions must be the same.\n";
        return EXIT_FAILURE;
    }
    if (matrix_a.width != matrix_b.width)
    {
        std::cerr
                << "Can't copy matrix C of width " << matrix_b.width << " to matrix A of width " << matrix_a.width
                << ".\n"
                << "Matrix A and B's widths must be the same.\n";
        return EXIT_FAILURE;
    }

    matrix_t matrix_c;
    matrix_c.width  = matrix_b.width;
    matrix_c.height = matrix_a.height;
    const size_t matrix_c_size  = matrix_c.width * matrix_c.height;
    const size_t matrix_c_bytes = matrix_c_size * sizeof(cl_float);
    matrix_c.elements.resize(matrix_c_size);

    cl_int           err                    = CL_SUCCESS;
    cl_wrapper       wrapper;
    cl_context       context                = wrapper.get_context();
    cl_program       program                = wrapper.make_program(&PROGRAM_SOURCE, 1, "-cl-std=CL2.0");
    cl_kernel        kernel_8x4             = wrapper.make_kernel("matmul_8x4_blocks", program);
    cl_command_queue command_queue          = wrapper.get_command_queue();
    cl_float        *matrix_a_svm           = nullptr;
    cl_mem           matrix_a_mem           = nullptr;
    cl_float        *matrix_b_svm           = nullptr;
    cl_mem           matrix_b_mem           = nullptr;
    cl_float        *matrix_c_svm           = nullptr;
    cl_mem           matrix_c_mem           = nullptr;
    std::atomic_int *done                   = nullptr;
    static const int MATRIX_MULTIPLICATIONS = 8;
    const size_t     global_work_size[]     = { // Each work-item calculates a 8 row by 4 column block of matrix C.
            static_cast<size_t>(matrix_b.width / 4),
            static_cast<size_t>(matrix_a.height / 8)};
    const size_t     local_work_size[]      = {
            static_cast<size_t>(atoi(argv[3])),
            static_cast<size_t>(atoi(argv[4]))};

    /*
     * Step 1: Create shared virtual memory backed OpenCL buffers.
     */

    // Matrix A
    matrix_a_svm = static_cast<cl_float*>(clSVMAlloc(
            context,
            CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
            matrix_a_bytes,
            0));
    std::memcpy(matrix_a_svm, matrix_a.elements.data(), matrix_a_bytes);
    matrix_a_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, matrix_a_bytes, matrix_a_svm, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix A.\n";
        return err;
    }

    // Matrix B
    matrix_b_svm = static_cast<cl_float*>(clSVMAlloc(
            context,
            CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
            matrix_b_bytes,
            0));
    std::memcpy(matrix_b_svm, matrix_b.elements.data(), matrix_b_bytes);
    matrix_b_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, matrix_b_bytes, matrix_b_svm, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix B.\n";
        return err;
    }

    // Matrix C
    matrix_c_svm = static_cast<cl_float*>(clSVMAlloc(
            context,
            CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
            matrix_c_bytes,
            0));
    matrix_c_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, matrix_c_bytes, matrix_c_svm, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for matrix C.\n";
        return err;
    }

    /*
     * Step 2: Set up the kernel arguments.
     */

    const size_t full_work_groups[] = {
            global_work_size[0] / local_work_size[0],
            global_work_size[1] / local_work_size[1]};
    const size_t partial_work_groups_x = global_work_size[0] % local_work_size[0] == 0 ? 0 : 1;
    const size_t partial_work_groups_y = global_work_size[1] % local_work_size[1] == 0 ? 0 : 1;
    size_t work_groups = (full_work_groups[0] + partial_work_groups_x) * (full_work_groups[1] + partial_work_groups_y);

    err = clSetKernelArg(kernel_8x4, 0, sizeof(matrix_a_mem), &matrix_a_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0.\n";
        return err;
    }

    err = clSetKernelArg(kernel_8x4, 1, sizeof(matrix_b_mem), &matrix_b_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1.\n";
        return err;
    }

    err = clSetKernelArg(kernel_8x4, 2, sizeof(matrix_c_mem), &matrix_c_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2.\n";
        return err;
    }

    const cl_int matrix_a_width  = matrix_a.width;
    err = clSetKernelArg(kernel_8x4, 3, sizeof(matrix_a_width), &matrix_a_width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3.\n";
        return err;
    }

    const cl_int matrix_b_width  = matrix_b.width;
    err = clSetKernelArg(kernel_8x4, 4, sizeof(matrix_b_width), &matrix_b_width);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4.\n";
        return err;
    }

    const size_t done_size = (MATRIX_MULTIPLICATIONS + 1) * sizeof(std::atomic_int); // in bytes
    done = static_cast<std::atomic_int*>(clSVMAlloc(
            context,
            CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
            done_size,
            0));
    memset(done, 0, done_size);
    err = clSetKernelArgSVMPointer(kernel_8x4, 5, done);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 5.\n";
        return err;
    }

    /*
     * Step 3: Run the 8x4 tiled kernel.
     */

    static const int PROCEED_NEXT_MULTIPLICATION = -1;

    for(cl_int matmul_iteration = 1 ; matmul_iteration < MATRIX_MULTIPLICATIONS + 1; matmul_iteration++)
    {
        err = clSetKernelArg(kernel_8x4, 6, sizeof(matmul_iteration), &matmul_iteration);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clSetKernelArg for argument 6.\n";
            return err;
        }

        err = clEnqueueNDRangeKernel(
                command_queue,
                kernel_8x4,
                2,
                nullptr,
                global_work_size,
                local_work_size,
                0,
                nullptr,
                nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clEnqueueNDRangeKernel for tiled portion.\n";
            return err;
        }
    }

    err = clFlush(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " starting computation.\n";
        return err;
    }

    // Notify the device that it can start the first matrix multiplication.
    done[0].store(
            PROCEED_NEXT_MULTIPLICATION,
            std::memory_order_release); // Also make non-atomic shared virtual memory updates visible elsewhere.

    for (int matmul_iteration = 1 ; matmul_iteration < MATRIX_MULTIPLICATIONS + 1; matmul_iteration++)
    {
        // Wait for the device to finish the previous matrix A * matrix B calculation.
        while(done[matmul_iteration].load(
                std::memory_order_acquire // Also make non-atomic shared virtual memory updates visible here.
                ) != work_groups)
        {
        }

        // Copy matrix C, which was just calculated on the device, to matrix A for the device to use on it's next matrix
        // multiplication.
        std::memcpy(matrix_a_svm, matrix_c_svm, matrix_a_bytes);

        // Notify the device that it can start the next matrix A * matrix B calculation.
        done[matmul_iteration].store(
                PROCEED_NEXT_MULTIPLICATION,
                std::memory_order_release); // On success, also make non-atomic shared virtual memory updates visible elsewhere.
    }

    // Output the final matrix C that was calculated by the device.
    std::memcpy(matrix_c.elements.data(), matrix_c_svm, matrix_c_bytes);
    if (output_to_file)
    {
        save_matrix(output_filename, matrix_c);
        std::cout << "Output matrix written to \"" << output_filename << "\".\n";
    }
    else
    {
        save_matrix(std::cout, matrix_c);
    }

    /*
     * Step 4: Clean up OpenCL resources that aren't automatically handled by cl_wrapper.
     */

    clSVMFree(context, done);

    err = clReleaseMemObject(matrix_c_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " releasing matrix_c_mem.\n";
        return err;
    }
    clSVMFree(context, matrix_c_svm);

    err = clReleaseMemObject(matrix_b_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " releasing matrix_b_mem.\n";
        return err;
    }
    clSVMFree(context, matrix_b_svm);

    err = clReleaseMemObject(matrix_a_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " releasing matrix_a_mem.\n";
        return err;
    }
    clSVMFree(context, matrix_a_svm);

    return EXIT_SUCCESS;
}