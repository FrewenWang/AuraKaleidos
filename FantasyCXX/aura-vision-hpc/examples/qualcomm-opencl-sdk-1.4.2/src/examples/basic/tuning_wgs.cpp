//--------------------------------------------------------------------------------------
// File: tuning_wgs.cpp
// Desc: This program shows how to tune the work group size of a kernel.
//
// Author:      QUALCOMM
//
//               Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <limits>
#include <cstdlib>
#include <fstream>
#include <iostream>

// Project includes
#include "util/cl_wrapper.h"

static const int PERCENT             = 100;
static const int PRECISION           = 2;
static const int DIMS                = 2;
static const int MAX_WORKGROUP_SIZE  = 1024;
static const int LOCAL_SIZE_MULTIPLE = 1;
static const int BUF_SIZE_X          = 710;
static const int BUF_SIZE_Y          = 1024;
static const double TIME_FACTOR      = 1000000.0;

static const char *PROGRAM_SOURCE = R"(
    __kernel void copybuf(__global uint* input, __global uint* output)
    {
        uint id = get_global_linear_id();
        output[id] = input[id];
    }
)";

int main(int argc, char** argv)
{
    static const      cl_queue_properties    QUEUE_PROPERTIES[]                = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_wrapper        wrapper(nullptr, QUEUE_PROPERTIES);
    cl_int            errcode                                                  = 0;
    int               source_buffer[BUF_SIZE_X * BUF_SIZE_Y]                   = {0};
    int               destination_buffer[BUF_SIZE_X * BUF_SIZE_Y]              = {0};
    cl_context        context                                                  = wrapper.get_context();
    cl_command_queue  queue                                                    = wrapper.get_command_queue();
    cl_program        program                                                  = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel         kernel                                                   = wrapper.make_kernel("copybuf", program);
    cl_event          evt                                                      = nullptr;
    cl_ulong          start_time                                               = 0;
    cl_ulong          end_time                                                 = 0;
    cl_mem            source_mem                                               = nullptr;
    cl_mem            destination_mem                                          = nullptr;
    double            default_exec_time                                        = 0;
    double            best_exec_time                                           = std::numeric_limits<double>::max();
    const size_t      global_work_size[]                                       = {BUF_SIZE_X, BUF_SIZE_Y};
    size_t            default_workgroup_size[]                                 = {0, 0};
    size_t            best_workgroup_size[]                                    = {0, 0};
    int               num_mismatches                                           = 0;

/*
 * Help Message
*/

    std::cout << "\nSample: " << argv[0] << "\n"
                 "\tThis sample implements a 2-Dimensional copy operation between source buffer and destination buffer.\n"
                 "\tUse it to find the best local work group size in terms of least execution time.\n";

/*
 * Step 1. Setup the source memory argument
*/

    for(int y = 0; y < BUF_SIZE_Y; y++)
    {
        for(int x = 0; x < BUF_SIZE_X; x++)
        {
            source_buffer[y * BUF_SIZE_X + x] = (y * BUF_SIZE_X + x);
        }
    }
    source_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                                sizeof(source_buffer), source_buffer, &errcode);
    if(errcode != CL_SUCCESS)
    {
        std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
    }
    errcode = clSetKernelArg(kernel, 0, sizeof(cl_mem), &source_mem);
    if(errcode != CL_SUCCESS)
    {
        std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
    }

/*
 * Step 2. Setup the destination memory argument
*/

    destination_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(source_buffer),
                                     nullptr, &errcode);
    if(errcode != CL_SUCCESS)
    {
        std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
    }
    errcode = clSetKernelArg(kernel, 1, sizeof(cl_mem), &destination_mem);
    if(errcode != CL_SUCCESS)
    {
        std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
    }

/*
 * Step 3. Finding the Best Work Group Size by
 *         Iteratively passing all workgroup size combinations to CLEnqueNDRangeKernel
*/

    for (size_t size_x = LOCAL_SIZE_MULTIPLE; size_x <= MAX_WORKGROUP_SIZE; size_x += LOCAL_SIZE_MULTIPLE)
    {
        for (size_t size_y = LOCAL_SIZE_MULTIPLE; size_y <= MAX_WORKGROUP_SIZE; size_y += LOCAL_SIZE_MULTIPLE)
        {
            if ((size_x * size_y) <= MAX_WORKGROUP_SIZE)
            {
                default_workgroup_size[0] = size_x;
                default_workgroup_size[1] = size_y;
                errcode = clEnqueueNDRangeKernel(
                        queue,
                        kernel,
                        DIMS,
                        nullptr,
                        global_work_size,
                        default_workgroup_size,
                        0,
                        nullptr,
                        &evt);
                errcode = clWaitForEvents(1, &evt);
                if(errcode != CL_SUCCESS)
                {
                    std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
                    return(0);
                }

/*
 * Verifying results for each workgroup size
*/

                errcode = clEnqueueReadBuffer(queue,
                                              destination_mem,
                                              CL_BLOCKING,
                                              0,
                                              BUF_SIZE_Y * BUF_SIZE_X * sizeof(int),
                                              destination_buffer, 0, nullptr, nullptr);
                if(errcode != CL_SUCCESS)
                {
                    std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
                    return(0);
                }
                for(int y = 0; y < BUF_SIZE_Y; y++ )
                {
                    for(int x = 0; x < BUF_SIZE_X; x++ )
                    {
                        if(destination_buffer[y * BUF_SIZE_X + x] != source_buffer[y * BUF_SIZE_X + x])
                        {
                            num_mismatches++;
                            std::cout<<destination_buffer[y * BUF_SIZE_X + x]<<"\t"<<source_buffer[y * BUF_SIZE_X + x]<<"\n";
                        }
                    }
                }
                if (num_mismatches != 0)
                {
                    std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
                    return (0);
                }

/*
 * Calculating execution time for each workgroup size
*/

                errcode = clGetEventProfilingInfo(
                        evt,
                        CL_PROFILING_COMMAND_START,
                        sizeof(start_time),
                        &start_time,
                        nullptr);
                if(errcode != CL_SUCCESS)
                {
                    std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
                    return(0);
                }
                errcode = clGetEventProfilingInfo(
                        evt,
                        CL_PROFILING_COMMAND_END,
                        sizeof(end_time),
                        &end_time,
                        nullptr);
                if(errcode != CL_SUCCESS)
                {
                    std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
                    return(0);
                }
                default_exec_time = (end_time - start_time)/TIME_FACTOR;

/*
 * Determining Best execution time
*/

                if (default_exec_time < best_exec_time)
                {
                    best_exec_time = default_exec_time;
                    best_workgroup_size[0] = default_workgroup_size[0];
                    best_workgroup_size[1] = default_workgroup_size[1];
                }
                clReleaseEvent(evt);
            }
        }
    }

/*
 * Determining Default execution time.
 * Default is Local Work Group Size = nullptr
*/

errcode = clEnqueueNDRangeKernel(
        queue,
        kernel,
        DIMS,
        nullptr,
        global_work_size,
        nullptr,
        0,
        nullptr,
        &evt);
errcode = clWaitForEvents(1, &evt);
if(errcode != CL_SUCCESS)
{
    std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
}

errcode = clGetEventProfilingInfo(
        evt,
        CL_PROFILING_COMMAND_START,
        sizeof(start_time),
        &start_time,
        nullptr);
if(errcode != CL_SUCCESS)
{
    std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
}
errcode = clGetEventProfilingInfo(
        evt,
        CL_PROFILING_COMMAND_END,
        sizeof(end_time),
        &end_time,
        nullptr);
if(errcode != CL_SUCCESS)
{
    std::cout<< "Nonzero error code "<<errcode<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
}
default_exec_time = (end_time - start_time)/TIME_FACTOR;

    std::cout<<"\nCL app successfully completed!\n\n"
        "The best work group size is:\nX\tY\n"
        <<best_workgroup_size[0]<<"\t"<<best_workgroup_size[1]<<"\n\n"
        "The Best Execution Time is:\t\t"<<best_exec_time<<" mili seconds\n"
        "The Default Execution Time is:\t\t"<<default_exec_time<<" mili seconds\n"
        "Improvement with Tuned workgroup is:\t"<<std::fixed<<std::setprecision(PRECISION)
        <<(default_exec_time-best_exec_time)/default_exec_time*PERCENT<<"%\n";

/*
 * Step 4. Clean up OpenCL resources that aren't automatically handled by cl_wrapper
*/

    clReleaseMemObject(destination_mem);
    clReleaseMemObject(source_mem);
}
