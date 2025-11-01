//--------------------------------------------------------------------------------------
// File: qcom_dot_product8.cpp
// Desc: This example demonstrates the usage of cl_qcom_dot_product8
//
// Author:      QUALCOMM
//
//          Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <cstdlib>
#include <iostream>
#include <vector>
#include <CL/cl.h>
#include "util/cl_wrapper.h"

static const char *PROGRAM_SOURCE = R"(
    #pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable
    __kernel void udot_product(
                  __global int *res,
                  __global uint *p0,
                  __global uint *p1,
                  __global int *acc)
    {
        int i = get_global_id(0);
        res[i] = qcom_udot8_acc(p0[i], p1[i], acc[i]);
    }
    __kernel void dot_product(
                  __global int *res,
                  __global uint *p0,
                  __global uint *p1,
                  __global int *acc)
    {
        int i = get_global_id(0);
        res[i] = qcom_dot8_acc(p0[i], p1[i], acc[i]);
    }
)";

int saturated_add(int a, int b)
{
    if (a > 0)
    {
        if (b > INT32_MAX - a)
        {
            return INT32_MAX;
        }
    } else if (b < INT32_MIN - a)
    {
        return INT32_MIN;
    }
    return a + b;
}

cl_int udot8_acc(cl_uint p0, cl_uint p1, cl_int acc)
{
    cl_uchar p0a = (p0 >> 24) & 0xFF;
    cl_uchar p0b = (p0 >> 16) & 0xFF;
    cl_uchar p0c = (p0 >> 8) & 0xFF;
    cl_uchar p0d = p0 & 0xFF;
    cl_uchar p1a = (p1 >> 24) & 0xFF;
    cl_uchar p1b = (p1 >> 16) & 0xFF;
    cl_uchar p1c = (p1 >> 8) & 0xFF;
    cl_uchar p1d = p1 & 0xFF;

    cl_int intermediate_res = p0a * p1a + p0b * p1b + p0c * p1c + p0d * p1d;
    return saturated_add(intermediate_res, acc);
}

cl_int dot8_acc(cl_uint p0, cl_uint p1, cl_int acc)
{
    cl_char p0a = (cl_char)((p0 >> 24) & 0xFF);
    cl_char p0b = (cl_char)((p0 >> 16) & 0xFF);
    cl_char p0c = (cl_char)((p0 >> 8) & 0xFF);
    cl_char p0d = (cl_char)(p0 & 0xFF);
    cl_uchar p1a = (p1 >> 24) & 0xFF;
    cl_uchar p1b = (p1 >> 16) & 0xFF;
    cl_uchar p1c = (p1 >> 8) & 0xFF;
    cl_uchar p1d = p1 & 0xFF;

    cl_int intermediate_res = p0a * p1a + p0b * p1b + p0c * p1c + p0d * p1d;
    return saturated_add(intermediate_res, acc);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <sign>\n" <<
                     "\t<sign> = signed | unsigned\n";
        return EXIT_FAILURE;
    }
    const std::string arg(argv[1]);
    if(arg.compare("signed") != 0 && arg.compare("unsigned") != 0)
    {
        std::cerr << "Usage: " << argv[0] << " <sign>\n" <<
                     "\t<sign> = signed | unsigned\n";
        return EXIT_FAILURE;
    }
    bool signed_dot = arg.compare("signed") == 0;

    cl_wrapper         wrapper;
    cl_program         program       = wrapper.make_program(&PROGRAM_SOURCE, 1, "-cl-std=CL2.0");
    cl_kernel          kernel        = wrapper.make_kernel((signed_dot) ? "dot_product" : "udot_product", program);
    cl_context         context       = wrapper.get_context();
    cl_command_queue   command_queue = wrapper.get_command_queue();
    cl_int             err           = CL_SUCCESS;

    // Setup example
    cl_uchar p0a = (signed_dot) ? -11 : 11;
    cl_uchar p0b = 22;
    cl_uchar p0c = (signed_dot) ? -33 : 33;
    cl_uchar p0d = 44;
    cl_uchar p1a = 55;
    cl_uchar p1b = 66;
    cl_uchar p1c = 77;
    cl_uchar p1d = 88;
    cl_uint  p0  = (p0a << 24) | (p0b << 16) | (p0c << 8) | p0d;
    cl_uint  p1  = (p1a << 24) | (p1b << 16) | (p1c << 8) | p1d;
    cl_int   acc = 9;
    std::cout << "p0: " << p0 << " p1: " << p1 << " acc: " << acc << std::endl;

    /*
     * Confirm the required OpenCL extensions are supported.
     */
    if (!wrapper.check_extension_support("cl_qcom_dot_product8"))
    {
        std::cerr << "cl_qcom_dot_product8 extension is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Allocate device memory
     */
    cl_mem res_mem = clCreateBuffer(context,CL_MEM_WRITE_ONLY, sizeof(cl_int),nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for res_mem\n";
        return err;
    }

    cl_mem p0_mem = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof (cl_uint), &p0, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for p0_mem\n";
        return err;
    }

    cl_mem p1_mem = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(cl_uint), &p1, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for p1_mem\n";
        return err;
    }

    cl_mem acc_mem = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof (cl_int), &acc, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer for acc_mem\n";
        return err;
    }

    /*
     * Set kernel arguments
     */
    err = clSetKernelArg(kernel, 0, sizeof(res_mem), &res_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0.\n";
        return err;
    }

    err = clSetKernelArg(kernel, 1, sizeof (p0_mem), &p0_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1.\n";
        return err;
    }

    err = clSetKernelArg(kernel, 2, sizeof(p1_mem), &p1_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2.\n";
        return err;
    }

    err = clSetKernelArg(kernel, 3, sizeof(acc_mem), &acc_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3.\n";
        return err;
    }

    /*
     * Launch kernel
     */
    size_t global_work_size = 1;
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
     * Map buffer for reading and compare against reference values
     */
    cl_int *hostptr = static_cast<cl_int*>(clEnqueueMapBuffer(
            command_queue,
            res_mem,
            CL_BLOCKING,
            CL_MAP_READ,
            0,
            sizeof(cl_int),
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

    cl_int ref = (signed_dot) ? dot8_acc(p0, p1, acc) : udot8_acc(p0, p1, acc);
    if(*hostptr != ref)
    {
        std::cerr << "Mismatch. Expected: " << ref << " Actual: " << *hostptr  << std::endl;
        return EXIT_FAILURE;
    } else
    {
        std::cout << "Reference: " << ref << " Actual: " << *hostptr << std::endl;
    }

    /*
     * Cleanup
     */
    err = clEnqueueUnmapMemObject(command_queue, res_mem, hostptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while unmapping output buffer.\n";
        return err;
    }

    clReleaseMemObject(res_mem);
    clReleaseMemObject(p0_mem);
    clReleaseMemObject(p1_mem);
    clReleaseMemObject(acc_mem);

    return EXIT_SUCCESS;
}
