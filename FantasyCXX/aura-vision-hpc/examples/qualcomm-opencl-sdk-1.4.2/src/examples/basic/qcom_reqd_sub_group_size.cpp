//--------------------------------------------------------------------------------------
// File: qcom_reqd_sub_group_size.cpp
// Desc: Demonstrates minimal usage of the cl_qcom_reqd_sub_group_size extension. It can
//       be used to manually tune an NDRange's sub-group size for performance.
//
// Author: QUALCOMM
//
//             Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#include <iostream>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "util/cl_wrapper.h"

static const char *PROGRAM_SOURCE = R"(
    #pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

    // The following kernels perform meaningless work to use finite
    // resources and make it impossible to execute an unlimited amount.

    __attribute__((qcom_reqd_sub_group_size("half")))
    __kernel void half_sub_group(__global int *buffer)
    {
        size_t i = get_global_linear_id();
        buffer[i] = i;
    }

    __attribute__((qcom_reqd_sub_group_size("full")))
    __kernel void full_sub_group(__global int *buffer)
    {
        size_t i = get_global_linear_id();
        buffer[i] = i;
    }
)";

int main(int argc, char** argv)
{
    cl_int     err                   = CL_SUCCESS;
    cl_wrapper wrapper;
    cl_program program = wrapper.make_program(&PROGRAM_SOURCE, 1, "-cl-std=CL2.0");
    cl_kernel  half_sub_group_kernel = wrapper.make_kernel("half_sub_group", program);
    cl_kernel  full_sub_group_kernel = wrapper.make_kernel("full_sub_group", program);
    size_t     local_work_size[]     = {0};
    cl_mem     cl_buffer             = nullptr;

    if (argc != 1)
    {
        std::cerr <<
                  "The sample takes no arguments.\n"
                  "\n"
                  "Usage: " << argv[0] << "\n"
                  "\n"
                  "Demonstrates usage of cl_qcom_reqd_sub_group_size.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_reqd_sub_group_size"))
    {
        std::cerr <<
                  "Extension cl_qcom_reqd_sub_group_size needed for setting the sub-group size is\n"
                  "not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Get sub-group information.
     */

    cl_device_id device_id = wrapper.get_device_id();
    size_t sub_group_size = 0;
    size_t sub_group_count = 0;

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(local_work_size), &local_work_size[0], nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " getting the device's maximum work-group size.\n";
        return err;
    }

    err = clGetKernelSubGroupInfoKHR(
            half_sub_group_kernel,
            device_id,
            CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
            sizeof(local_work_size),
            local_work_size,
            sizeof(sub_group_size),
            &sub_group_size,
            nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " getting the half sub-group kernel's sub-group size.\n";
        return err;
    }
    err = clGetKernelSubGroupInfoKHR(
            half_sub_group_kernel,
            device_id,
            CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR,
            sizeof(local_work_size),
            local_work_size,
            sizeof(sub_group_count),
            &sub_group_count,
            nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " getting the half sub-group kernel's sub-group count.\n";
        return err;
    }
    std::cout <<
            "The half sub-group kernel's sub-group size is " << sub_group_size <<
            " work-items across " << sub_group_count << " sub-groups.\n";

    err = clGetKernelSubGroupInfoKHR(
            full_sub_group_kernel,
            device_id,
            CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
            sizeof(local_work_size),
            local_work_size,
            sizeof(sub_group_size),
            &sub_group_size,
            nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " getting the full sub-group kernel's sub-group size.\n";
        return err;
    }
    err = clGetKernelSubGroupInfoKHR(
            full_sub_group_kernel,
            device_id,
            CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR,
            sizeof(local_work_size),
            local_work_size,
            sizeof(sub_group_count),
            &sub_group_count,
            nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " getting the full sub-group kernel's sub-group count.\n";
        return err;
    }
    std::cout <<
            "The full sub-group kernel's sub-group size is " << sub_group_size <<
            " work-items across " << sub_group_count << " sub-groups.\n";

    /*
     * Step 2: Set up kernel arguments and run the kernels.
     */

    cl_command_queue command_queue = wrapper.get_command_queue();

    cl_buffer = clCreateBuffer(
            wrapper.get_context(),
            CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            local_work_size[0] * sizeof(cl_int),
            nullptr,
            &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " creating the OpenCL buffer.\n";
        return err;
    }
    err = clSetKernelArg(half_sub_group_kernel, 0, sizeof(cl_buffer), &cl_buffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " setting the half sub-group kernel's buffer.\n";
        return err;
    }
    err = clSetKernelArg(full_sub_group_kernel, 0, sizeof(cl_buffer), &cl_buffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " setting the full sub-group kernel's buffer.\n";
        return err;
    }

    err = clEnqueueNDRangeKernel(
            command_queue,
            half_sub_group_kernel,
            1,
            nullptr,
            local_work_size, // global_work_size
            local_work_size,
            0,
            nullptr,
            nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " enqueuing the half sub-group kernel.\n";
        return err;
    }

    err = clEnqueueNDRangeKernel(
            command_queue,
            full_sub_group_kernel,
            1,
            nullptr,
            local_work_size, // global_work_size
            local_work_size,
            0,
            nullptr,
            nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " enqueuing the full sub-group kernel.\n";
        return err;
    }

    err = clFinish(command_queue);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " finishing computation.\n";
        return err;
    }

    /*
     * Step 3: Clean up resources that aren't automatically handled by cl_wrapper.
     */

    err = clReleaseMemObject(cl_buffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " releasing the OpenCL buffer.\n";
        return err;
    }


    std::cout << "Success!\n";
    return EXIT_SUCCESS;
}