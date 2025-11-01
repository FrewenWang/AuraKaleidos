//--------------------------------------------------------------------------------------
// File: qcom_bitreverse.cpp
// Desc: Demonstrates the usage of the cl_qcom_bitreverse extension that offers
//       accelerated reversal of bits in a unsigned integer by reversing the bits of
//       an array of prime numbers.
//
// Author: QUALCOMM
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
    #pragma OPENCL EXTENSION cl_qcom_bitreverse : enable
    __kernel void bitreverse (
       __global uint *io_array)
    {
       uint qcom_bitreverse(uint v);
       int i = get_global_id(0);
       io_array[i] = qcom_bitreverse(io_array[i]);
    }
)";

unsigned int bit_reverse(unsigned int num)
{
    unsigned int numOfBit = sizeof(num) * 8;
    unsigned int reverse_num = 0, i, temp;

    for (i = 0; i < numOfBit; i++)
    {
        temp = (num & (1 << i));
        if (temp)
            reverse_num |= (1 << ((numOfBit - 1) - i));
    }

    return reverse_num;
}

int main(int argc, char** argv)
{
    cl_wrapper              wrapper;
    cl_program              program       = wrapper.make_program(&PROGRAM_SOURCE, 1, "-cl-std=CL2.0");
    cl_kernel               kernel        = wrapper.make_kernel("bitreverse", program);
    cl_context              context       = wrapper.get_context();
    cl_command_queue        command_queue = wrapper.get_command_queue();
    cl_int                  err           = CL_SUCCESS;
    // A list of prime numbers
    std::vector<cl_uint>    numbers   = {947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013};

    /*
     * Confirm the required OpenCL extensions are supported.
     */
    if (!wrapper.check_extension_support("cl_qcom_bitreverse"))
    {
        std::cerr << "cl_qcom_bitreverse extension is not supported.\n";
        return EXIT_FAILURE;
    }

    cl_mem in_out_mem = clCreateBuffer(
            context,
            CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
            numbers.size()*sizeof(cl_uint),
            numbers.data(),
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer\n";
        return err;
    }

    err = clSetKernelArg(kernel, 0, sizeof(in_out_mem), &in_out_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0.\n";
        return err;
    }

    size_t global_work_size = numbers.size();
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
    cl_uint *hostptr = static_cast<cl_uint*>(clEnqueueMapBuffer(
            command_queue,
            in_out_mem,
            CL_BLOCKING,
            CL_MAP_READ,
            0,
            numbers.size()*sizeof(cl_uint),
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

    cl_uint *reference = static_cast<cl_uint*>(numbers.data());
    for(size_t i = 0; i < numbers.size(); ++i)
    {
        if(hostptr[i] != bit_reverse(reference[i]))
        {
            std::cerr << "Mismatch at "<< i <<". Expected: " << reference[i] << "Actual: " << hostptr[i] << "\n";
            return EXIT_FAILURE;
        }
        else
        {
            std::cout << "Reference " << "0x" << std::hex << bit_reverse(reference[i]) << " == " << "OpenCL 0x" << std::hex << hostptr[i] << std::endl;
        }
    }

    err = clEnqueueUnmapMemObject(command_queue, in_out_mem, hostptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while unmapping output buffer.\n";
        return err;
    }

    clReleaseMemObject(in_out_mem);

    return EXIT_SUCCESS;
}
