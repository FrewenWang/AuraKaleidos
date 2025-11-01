//--------------------------------------------------------------------------------------
// File: qcom_perf_hint.cpp
// Desc: This example demonstrates the usage of cl_qcom_perf_hint.
//
// Author:      QUALCOMM
//
//          Copyright (c) 2014, 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>
#include "util/cl_wrapper.h"

static const char* PROGRAM_SOURCE = R"(
    __constant int NUM_INTERATIONS = 10;

    /*
     * Sample multiply-add 16-bit integer OpenCL kernel intended to provide heavy computation
     */
    __kernel void MAdd16(__global short *data) {
        int gid = get_global_id(0);
        short s = data[gid];
        short16 s0 = s + (short16)(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
        short16 s1 = s + (short16)(17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32);
        short16 s2 = s + (short16)(33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48);
        short16 s3 = s + (short16)(49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64);
        for (int index = 0; index < NUM_INTERATIONS; ++index) {
            s0=s2-s0*s1;    s1=s3-s1*s2;    s2=s0-s2*s3;    s3=s1-s3*s0;
            s0=s2-s0*s1;    s1=s3-s1*s2;    s2=s0-s2*s3;    s3=s1-s3*s0;
            s0=s2-s0*s1;    s1=s3-s1*s2;    s2=s0-s2*s3;    s3=s1-s3*s0;
            s0=s2-s0*s1;    s1=s3-s1*s2;    s2=s0-s2*s3;    s3=s1-s3*s0;
        }
        s0 = s0+s1+s2+s3;
        data[gid] = s0.s0+s0.s1+s0.s2+s0.s3+s0.s4+s0.s5+s0.s6+s0.s7+s0.s8+s0.s9+s0.sa+s0.sb+s0.sc+s0.sd+s0.se+s0.sf;
    }
)";

static const char* PROGRAM_COMPILER_OPTIONS = "-cl-mad-enable -cl-no-signed-zeros "
                                              "-cl-unsafe-math-optimizations -cl-finite-math-only";

// Which iteration in our regression we will change the performance hint
constexpr size_t KERNEL_TRANSITION_ITERATION = 5;
constexpr size_t NUMBER_OF_KERNEL_ITERATIONS = 10;

// Integer operations for default work group size
// Note: If the work group max size is less than this value,
//       the global work size will be tailored to smaller value
//       and remain an integer multiple.
constexpr size_t INTOPS_WORKGROUP_SIZE = 1024;

// Number of short integers we're operating on
constexpr size_t NUM_SHORTS_IN_BUFFER = INTOPS_WORKGROUP_SIZE * INTOPS_WORKGROUP_SIZE * 2;

int main(int argc, char** argv)
{
    /*
     * OpenCL setup via cl_wrapper, create one kernel, one context, and one command queue.
     *
     * Note: cl_wrapper's constructor takes properties for the cl_context that it creates, so lets pass in
     *       a performance hint to set the performance hint to low performance.
     */
    cl_context_properties context_properties[3] = {CL_CONTEXT_PERF_HINT_QCOM, CL_PERF_HINT_LOW_QCOM, 0};
    cl_command_queue_properties cmd_queue_properties[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

    cl_wrapper            wrapper(context_properties, cmd_queue_properties);
    cl_program            program       = wrapper.make_program(&PROGRAM_SOURCE, 1, PROGRAM_COMPILER_OPTIONS);
    cl_kernel             kernel        = wrapper.make_kernel("MAdd16", program);
    cl_context            context       = wrapper.get_context();
    cl_device_id          device_id     = wrapper.get_device_id();
    cl_command_queue      command_queue = wrapper.get_command_queue();
    cl_int                err           = CL_SUCCESS;

    /*
     * Confirm the required OpenCL extensions are supported.
     */
    if (!wrapper.check_extension_support("cl_qcom_perf_hint"))
    {
        std::cerr << "cl_qcom_perf_hint extension is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Setup example
     */
    std::random_device rd;                              // Random device to give our random number generator
    std::mt19937 gen(rd());                             // Random number generator
    std::uniform_int_distribution<cl_short> dist(0, 5); // Random number distribution for generator
    cl_short*    host_memory = nullptr;                 // Buffer to store random generated values to send to GPU
    size_t       global_work_size;                      // Maximum number of values in buffer to compute on GPU
    size_t       local_work_size;                       // KERNEL_WORK_GROUP_SIZE queried from device
    cl_ulong     low_perf_hint_run_time = 0;            // Accumulated kernel run-time for CL_PERF_HINT_LOW_QCOM hint in nanoseconds
    cl_ulong     high_perf_hint_run_time = 0;           // Accumulated kernel run-time for CL_PERF_HINT_HIGH_QCOM hint in nanoseconds
    cl_ulong     start_time      = 0;                   // Kernel event profiling start time
    cl_ulong     end_time        = 0;                   // Kernel event profiling end time
    cl_mem       read_write_buffer;                     // OpenCL buffer used by the kernel
    cl_event     event_kernel_execution;                // OpenCL event object used to track kernel run-time completion

    /*
     * Allocate our buffer of shorts
     */
    host_memory = reinterpret_cast<cl_short*>(calloc(NUM_SHORTS_IN_BUFFER, sizeof(cl_short)));

    /*
     * Allocate device memory
     */
    read_write_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(cl_short) * NUM_SHORTS_IN_BUFFER, nullptr, &err);
    if(err != CL_SUCCESS)
    {
        std::cerr << "clCreateBuffer failed with ErrorCode : " << err << "\n";
        return EXIT_FAILURE;
    }

    /*
     * Set kernel arguments
     */
    err = clSetKernelArg (kernel, 0, sizeof(cl_mem), &read_write_buffer);
    if(err != CL_SUCCESS)
    {
        std::cerr << "clSetKernelArg failed with ErrorCode : " << err << "\n";
        return EXIT_FAILURE;
    }

    /*
     * Query kernel work group size to determine if our global_work_size needs to be aligned to an integer
     * multiple.
     */
    err = clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size),
                                    &local_work_size, nullptr);
    if(err != CL_SUCCESS)
    {
        std::cerr << "clGetKernelWorkGroupInfo failed with ErrorCode : " << err << "\n";
        return EXIT_FAILURE;
    }

    /*
     * Set global work size to an integer multiple of the work group size in-case our global_work_size is not aligned
     */
    global_work_size = NUM_SHORTS_IN_BUFFER;
    global_work_size -= global_work_size % local_work_size;

    /*
     * Initialize host data, with the first half the same as the second
     *
     * For example with a buffer of 10 integers 0..9 the buffer would
     * look like:
     *
     * Index   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
     *         __________________________________________________________
     * Value:  0  1  2  3  4  5  6  7  8  9  9  8  7  6  5  4  3  2  1  0
     *
     * Only in this case our values are generated with a random number generator.
     */
    for (size_t index = 0U; index < global_work_size / 2; ++index)
    {
        host_memory[index] = host_memory[global_work_size - index - 1] = dist(gen);
    }

    /*
     * Send buffer to the GPU
     */
    err = clEnqueueWriteBuffer (command_queue, read_write_buffer, CL_BLOCKING, 0, NUM_SHORTS_IN_BUFFER * sizeof(cl_short),
                                host_memory, 0, nullptr, nullptr);
    if(err != CL_SUCCESS)
    {
        std::cerr << "clEnqueueWriteBuffer failed with ErrorCode : " << err << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Starting cl_qcom_perf_hint kernel regression, running 10 iterations: " << std::endl;

    /*
     * Run our kernel stress test which runs the kernel 10 times, 5 times at the performance hint created at
     * context creation (CL_PERF_HINT_LOW_QCOM), and 5 times at high performance (CL_PERF_HINT_HIGH_QCOM)
     * and stores the results of each pass.
     */
    for(size_t kernel_iteration = 0; kernel_iteration < NUMBER_OF_KERNEL_ITERATIONS; kernel_iteration++)
    {
        if(kernel_iteration == KERNEL_TRANSITION_ITERATION)
        {
            // Finish prior computation with low performance before switching modes.
            err = clFinish(command_queue);
            if(err != CL_SUCCESS)
            {
                std::cerr << "clFinish failed with ErrorCode : " << err << "\n";
                return EXIT_FAILURE;
            }

            err = clSetPerfHintQCOM(context, CL_PERF_HINT_HIGH_QCOM);
            if(err != CL_SUCCESS)
            {
                std::cerr << "clSetPerfHintQCOM failed with ErrorCode : " << err << "\n";
                return EXIT_FAILURE;
            }

            // Allow time for clock switching after new performance hint set
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(100ms);
        }

        /*
         * Launch kernel
         */
        err = clEnqueueNDRangeKernel(command_queue,kernel,1, nullptr, &global_work_size,
                                     nullptr,0,nullptr, &event_kernel_execution);
        if (err != CL_SUCCESS)
        {
            std::cerr << "clEnqueueWriteBuffer failed with ErrorCode : " << err << "\n";
            return err;
        }

        /*
         * Wait for kernel to be finished
         */
        err = clWaitForEvents (1, &event_kernel_execution);
        if (err != CL_SUCCESS)
        {
            std::cerr << "clWaitForEvents failed with ErrorCode : " << err << "\n";
            return err;
        }

        /*
         * Query profiling information from kernel execution event
         */
        err = clGetEventProfilingInfo(event_kernel_execution, CL_PROFILING_COMMAND_START,
                                      sizeof(start_time), &start_time, nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "clGetEventProfilingInfo failed with ErrorCode : " << err << "\n";
            return err;
        }

        err = clGetEventProfilingInfo(event_kernel_execution, CL_PROFILING_COMMAND_END,
                                      sizeof(end_time), &end_time, nullptr);
        if (err != CL_SUCCESS)
        {
            std::cerr << "clGetEventProfilingInfo failed with ErrorCode : " << err << "\n";
            return err;
        }

        /*
         * Profiling time stores execution time as nanoseconds, we can use std::chrono to easily manage this
         */
        std::chrono::nanoseconds ns(end_time - start_time);

        /*
         * Save the aggregate run-time dependent on what performance hint we're currently using, before the
         * kernel transition iteration we'll be in CL_PERF_HINT_LOW_QCOM performance, and afterwards we'll be
         * running in CL_PERF_HINT_HIGH_QCOM performance.
         */
        if(kernel_iteration < KERNEL_TRANSITION_ITERATION)
        {
            low_perf_hint_run_time += ns.count();
        }
        else
        {
            high_perf_hint_run_time += ns.count();
        }
    }

    /*
     * Calculate low perf hint test case average kernel time
     */
    {
        std::chrono::nanoseconds ns(low_perf_hint_run_time);
        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(ns);
        std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(ns) -
                                       std::chrono::duration_cast<std::chrono::microseconds>(ms);
        std::cout << "    Low performance hint runtime: " << ms.count() << "ms " << us.count() << "us" << std::endl;
    }

    /*
     * Calculate high perf hint change test case average kernel time
     */
    {
        std::chrono::nanoseconds ns(high_perf_hint_run_time);
        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(ns);
        std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(ns) -
                                       std::chrono::duration_cast<std::chrono::microseconds>(ms);
        std::cout << "    High performance hint runtime: " << ms.count() << "ms " << us.count() << "us" << std::endl;
    }

    /*
     * Clean up everything outside of the cl_wrapper's control
     */
    clReleaseMemObject(read_write_buffer);
    clReleaseEvent(event_kernel_execution);
    free(host_memory);

    return EXIT_SUCCESS;
}
