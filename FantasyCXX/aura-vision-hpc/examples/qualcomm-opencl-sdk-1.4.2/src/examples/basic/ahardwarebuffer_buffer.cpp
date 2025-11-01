//--------------------------------------------------------------------------------------
// File: ahardwarebuffer_buffer.cpp
// Desc: This program demonstrates the usage of cl_qcom_android_ahardwarebuffer_host_ptr
//       extension using OpenCL buffers.
//
// Author: QUALCOMM
//
//          Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <iostream>

// Project includes
#include "util/cl_wrapper.h"

// Library includes
#include <android/hardware_buffer.h>

static const int   WIDTH_IN_PIXELS  = 64;
static const int   HEIGHT_IN_PIXELS = 64;
static const int   BYTES_PER_PIXEL  = 4; // Using R8G8B8A8 format in this sample. Make sure to update if format is changed.
static const char *PROGRAM_SOURCE = R"(
    __kernel void square (
       __global int *in_out_buf)
    {
       int i = get_global_id(0);
       in_out_buf[i] *= in_out_buf[i];
    }
)";

int main(int argc, char** argv)
{
    cl_wrapper       wrapper;
    cl_program       program       = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel        kernel        = wrapper.make_kernel("square", program);
    cl_context       context       = wrapper.get_context();
    cl_command_queue command_queue = wrapper.get_command_queue();
    cl_int           err           = CL_SUCCESS;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */
    if (!wrapper.check_extension_support("cl_qcom_android_ahardwarebuffer_host_ptr"))
    {
        std::cerr << "cl_qcom_android_ahardwarebuffer_host_ptr extension is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create AHardwareBuffer_Desc and initialize structure.
     * Allocate a buffer using AHardwareBuffer_allocate that matches the passed AHardwareBuffer_Desc.
     */
    AHardwareBuffer_Desc ahb_desc = {0};
    ahb_desc.width  = WIDTH_IN_PIXELS;
    ahb_desc.height = HEIGHT_IN_PIXELS;
    ahb_desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM; // Make sure to update BYTES_PER_PIXEL if format is changed
    ahb_desc.layers = 1;
    ahb_desc.usage  = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;

    AHardwareBuffer *p_input_ahb;
    err = AHardwareBuffer_allocate(&ahb_desc, &p_input_ahb);
    if (err != CL_SUCCESS)
    {
        std::cerr << "AHardwareBuffer_allocate failed with error code " << err << "\n";
        return err;
    }

    /*
     * Step 2: Create an OpenCL buffer object that uses ahb_mem as its data store.
     */
    cl_mem_ahardwarebuffer_host_ptr ahb_mem;
    ahb_mem.ext_host_ptr.allocation_type   = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
    ahb_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_WRITEBACK_QCOM;
    ahb_mem.ahb_ptr                        = p_input_ahb;

    AHardwareBuffer_describe(p_input_ahb, &ahb_desc); // This is to get the stride from input buffer
    size_t buf_size = ahb_desc.height * ahb_desc.stride * BYTES_PER_PIXEL;

    cl_mem in_out_mem = clCreateBuffer(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            buf_size, // It is not necessary to pass buffer size. Driver can calcuclate buffer size, if '0' is passed here.
            &ahb_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for input android hardware buffer\n";
        return err;
    }

    /*
     * Step 3: Set up kernel argument
     */
    err = clSetKernelArg(kernel, 0, sizeof(in_out_mem), &in_out_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0.\n";
        return err;
    }

    /*
     * Step 4: Map buffer for writing and initialize
     */
    int *hostptr = static_cast<int *>(clEnqueueMapBuffer(
            command_queue,
            in_out_mem,
            CL_BLOCKING,
            CL_MAP_WRITE,
            0,
            buf_size,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while mapping in_out_mem.\n";
        return err;
    }

    int *locked_ptr = nullptr;
    // Fence value -1 is passed as clEnqueueMapBuffer in above call makes sure that writing is complete due to CL_BLOCKING flag.
    err = AHardwareBuffer_lock(p_input_ahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, (void **) &locked_ptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while locking p_input_ahb\n";
        return err;
    }

    size_t global_work_size =  buf_size/sizeof(int);
    for(size_t i = 0; i < global_work_size; i++)
    {
        hostptr[i] = i;
    }

    err = AHardwareBuffer_unlock(p_input_ahb, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while unlocking p_input_ahb\n";
        return err;
    }

    err = clEnqueueUnmapMemObject(command_queue, in_out_mem, hostptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "clEnqueueUnmapMemObject failed with error code " << err << "\n";
        return err;
    }

    /*
     * Step 5: Execute kernel
     */
    err = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            1, // work dimension of input buffer
            nullptr,
            &global_work_size,
            nullptr,
            0,
            nullptr,
            nullptr
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel.\n";
        return err;
    }

    clFinish(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while finishing the command queue.\n";
        return err;
    }

    /*
     * Step 6: Map buffer for reading and compare against reference values
     */
    int *reference = static_cast<int*>(malloc(buf_size));
    for(size_t i = 0; i < global_work_size; ++i)
    {
        reference[i] = i*i;
    }
    hostptr = static_cast<int*>(clEnqueueMapBuffer(
            command_queue,
            in_out_mem,
            CL_BLOCKING,
            CL_MAP_READ,
            0,
            buf_size,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while mapping output buffer.\n";
        return err;
    }

    err = AHardwareBuffer_lock(p_input_ahb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, nullptr, (void **) &locked_ptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while locking p_input_ahb\n";
        return err;
    }

    for(size_t i = 0; i < global_work_size; ++i)
    {
        if(hostptr[i] != reference[i])
        {
            std::cerr << "Mismatch at "<< i <<". Expected: " << reference[i] << "Actual: " << hostptr[i] << "\n";
            return EXIT_FAILURE;
        }
    }

    err = AHardwareBuffer_unlock(p_input_ahb, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while unlocking p_input_ahb\n";
        return err;
    }

    err = clEnqueueUnmapMemObject(command_queue, in_out_mem, hostptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while unmapping output buffer.\n";
        return err;
    }

    /*
     * Step 7: Clean up OpenCL resources that aren't automatically handled by cl_wrapper
     */
    clReleaseMemObject(in_out_mem);
    AHardwareBuffer_release(p_input_ahb);

    std::cout << "ahardwarebuffer_buffer sample executed successfully\n";

    return EXIT_SUCCESS;
}
