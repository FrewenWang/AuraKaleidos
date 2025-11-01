//--------------------------------------------------------------------------------------
// File: protected_android_hardware_buffer.cpp
// Desc: Demonstrates the cl_qcom_protected_context and
//       cl_qcom_android_ahardwarebuffer_host_ptr extensions.
//
// Author: QUALCOMM
//
//             Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#include <cstdlib>
#include <iostream>
#include <ui/GraphicBuffer.h>
#include <vndk/hardware_buffer.h>
#include <QtiGrallocDefs.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_ext_qcom.h>
#include "util/cl_wrapper.h"

static const char *PROGRAM_SOURCE = R"(
    __kernel void initialize_protected_memory(__global int *buffer, __write_only __global image2d_t image) {
        buffer[0] = 1;
        write_imageui(image, (int2)(0, 0), (uint4)(1, 0, 0, 0));
    }
)";

int main(int argc, char** argv)
{
    cl_int                             err                         = CL_SUCCESS;
    static const cl_context_properties CONTEXT_PROPERTIES[]        = {CL_CONTEXT_PROTECTED_QCOM, 1, 0};
    cl_wrapper                         wrapper(CONTEXT_PROPERTIES);
    cl_command_queue                   protected_command_queue     = wrapper.get_command_queue();
    cl_mem                             protected_cl_buffer         = nullptr;
    cl_mem                             protected_cl_image          = nullptr;

    if (argc != 1)
    {
        std::cerr <<
                "The sample takes no arguments.\n"
                "\n"
                "Usage: " << argv[0] << "\n"
                "\n"
                "0 is returned after initializing memory using the cl_qcom_protected_context\n"
                "and cl_qcom_android_ahardwarebuffer_host_ptr extensions. 1 is returned on\n"
                "failure.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_protected_context"))
    {
        std::cerr <<
                "Extension cl_qcom_protected_context needed for a protected context is not\n"
                "supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_android_ahardwarebuffer_host_ptr"))
    {
        std::cerr <<
                "Extension cl_qcom_android_ahardwarebuffer_host_ptr needed for Android Hardware\n"
                "Buffer (zero copy) is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create protected Android hardware buffer backed OpenCL memory objects.
     */

    int                  android_error           = android::NO_ERROR;
    cl_context           protected_context       = wrapper.get_context();
    static const int     IMAGE_WIDTH             = 1; // in pixels
    static const int     IMAGE_HEIGHT            = 1; // in pixels
    static const int     PIXEL_SIZE              = 4; // in bytes
    AHardwareBuffer_Desc buffer_description      = {0};
    AHardwareBuffer     *android_hardware_buffer = nullptr;
    AHardwareBuffer_Desc image_description       = {0};
    AHardwareBuffer     *android_hardware_image  = nullptr;

    // Allocate the backing protected Android hardware buffer.
    buffer_description.width = IMAGE_WIDTH;
    buffer_description.height = IMAGE_HEIGHT;
    buffer_description.layers = 1;
    buffer_description.format = HAL_PIXEL_FORMAT_RGBA_8888;
    buffer_description.usage =
            AHARDWAREBUFFER_USAGE_CPU_READ_NEVER |
            AHARDWAREBUFFER_USAGE_CPU_WRITE_NEVER |
            AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE |
            GRALLOC_USAGE_PRIVATE_UNCACHED |
            AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT;
    android_error = AHardwareBuffer_allocate(&buffer_description, &android_hardware_buffer);
    if(android_error != android::NO_ERROR)
    {
        std::cerr << "Error " << android_error << " allocating the protected Android hardware buffer.\n";
        return android_error;
    }

    // Create the protected OpenCL buffer.
    size_t buffer_size = buffer_description.height * buffer_description.stride * PIXEL_SIZE;
    cl_mem_ahardwarebuffer_host_ptr buffer_host_ptr = {{0}};
    buffer_host_ptr.ext_host_ptr.allocation_type = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
    buffer_host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    buffer_host_ptr.ahb_ptr = android_hardware_buffer;
    protected_cl_buffer = clCreateBuffer(
            protected_context,
            CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            buffer_size,
            &buffer_host_ptr,
            &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " creating the protected OpenCL buffer.\n";
        return err;
    }

    // Allocate the backing protected Android hardware image.
    image_description.width = IMAGE_WIDTH;
    image_description.height = IMAGE_HEIGHT;
    image_description.layers = 1;
    image_description.format = HAL_PIXEL_FORMAT_RGBA_8888;
    image_description.usage =
            AHARDWAREBUFFER_USAGE_CPU_READ_NEVER |
            AHARDWAREBUFFER_USAGE_CPU_WRITE_NEVER |
            AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE |
            GRALLOC_USAGE_PRIVATE_UNCACHED |
            AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT;
    android_error = AHardwareBuffer_allocate(&image_description, &android_hardware_image);
    if(android_error != android::NO_ERROR)
    {
        std::cerr << "Error " << android_error << " allocating the protected Android hardware image.\n";
        return android_error;
    }
    AHardwareBuffer_describe(android_hardware_image, &image_description); // get stride

    // Create the protected OpenCL image.
    static const cl_image_format IMAGE_FORMAT = {CL_RGBA, CL_UNSIGNED_INT8};
    cl_image_desc cl_image_description = {0};
    cl_image_description.image_type = CL_MEM_OBJECT_IMAGE2D;
    cl_image_description.image_width = IMAGE_WIDTH;
    cl_image_description.image_height = IMAGE_HEIGHT;
    cl_image_description.image_row_pitch = image_description.stride * PIXEL_SIZE;
    cl_mem_ahardwarebuffer_host_ptr image_host_ptr = {{0}};
    image_host_ptr.ext_host_ptr.allocation_type = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
    image_host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    image_host_ptr.ahb_ptr = android_hardware_image;
    protected_cl_image = clCreateImage(
            protected_context,
            CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &IMAGE_FORMAT,
            &cl_image_description,
            &image_host_ptr,
            &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " creating the protected OpenCL image.\n";
        return err;
    }

    /*
     * Step 2: Set up kernel arguments and run the kernel.
     */

    cl_program program = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel  kernel  = wrapper.make_kernel("initialize_protected_memory", program);

    err = clSetKernelArg(kernel, 0, sizeof(protected_cl_buffer), &protected_cl_buffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " setting kernel argument 0 to the protected buffer.\n";
        return err;
    }

    err = clSetKernelArg(kernel, 1, sizeof(protected_cl_image), &protected_cl_image);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " setting kernel argument 1 to the protected image.\n";
        return err;
    }

    static const size_t GLOBAL_WORK_SIZE[] = {1};
    static const size_t LOCAL_WORK_SIZE[] = {1};
    err = clEnqueueNDRangeKernel(
            protected_command_queue,
            kernel,
            1,
            nullptr,
            GLOBAL_WORK_SIZE,
            LOCAL_WORK_SIZE,
            0,
            nullptr,
            nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " enqueuing kernel to initialize protected memory.\n";
        return err;
    }

    /*
     * Step 3: Clean up resources that aren't automatically handled by cl_wrapper.
     */

    // Finish using the Android hardware buffer and image for safe release.
    err = clFinish(protected_command_queue);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " finishing computation.\n";
        return err;
    }

    err = clReleaseMemObject(protected_cl_image);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " releasing the protected OpenCL image.\n";
        return err;
    }

    AHardwareBuffer_release(android_hardware_image);

    err = clReleaseMemObject(protected_cl_buffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " releasing the protected OpenCL buffer.\n";
        return err;
    }

    AHardwareBuffer_release(android_hardware_buffer);


    std::cout <<
            "Allocated and initialized OpenCL memory objects with cl_qcom_protected_context\n"
            "and cl_qcom_android_ahardwarebuffer_host_ptr!\n";
    return EXIT_SUCCESS;
}