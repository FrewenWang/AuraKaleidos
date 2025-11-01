//--------------------------------------------------------------------------------------
// File: protected_android_native_buffer.cpp
// Desc: Demonstrates the cl_qcom_protected_context and
//       cl_qcom_android_native_buffer_host_ptr extensions.
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
                "and cl_qcom_android_native_buffer_host_ptr extensions. 1 is returned on failure.\n";
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

    if (!wrapper.check_extension_support("cl_qcom_android_native_buffer_host_ptr"))
    {
        std::cerr <<
                "Extension cl_qcom_android_native_buffer_host_ptr needed for Android Native\n"
                "Buffer (zero copy) is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create protected Android native buffer backed OpenCL memory objects.
     */

    android::status_t     android_err       = android::NO_ERROR;
    static const uint32_t WIDTH             = 1; // in pixels
    static const uint32_t HEIGHT            = 1; // in pixels
    cl_context            protected_context = wrapper.get_context();
    static const int      PIXEL_SIZE        = 4; // in bytes

    // Allocate the backing protected Android graphic buffer.
    static android::sp<android::GraphicBuffer> protected_graphic_buffer = new android::GraphicBuffer();
    android_err = protected_graphic_buffer->reallocate(
            WIDTH,
            HEIGHT,
            HAL_PIXEL_FORMAT_RGBA_8888,
            1,
            GRALLOC_USAGE_SW_READ_NEVER |
                    GRALLOC_USAGE_SW_WRITE_NEVER |
                    GRALLOC_USAGE_HW_TEXTURE |
                    GRALLOC_USAGE_PRIVATE_UNCACHED |
                    GRALLOC_USAGE_PROTECTED);
    if(android_err != android::NO_ERROR)
    {
        std::cerr << "Error " << android_err << " allocating the protected graphic buffer.\n";
        return android_err;
    }

    // Create the protected OpenCL buffer.
    size_t buffer_size = protected_graphic_buffer->getHeight() * protected_graphic_buffer->getStride() * PIXEL_SIZE;
    cl_mem_android_native_buffer_host_ptr buffer_host_ptr = {{0}};
    buffer_host_ptr.ext_host_ptr.allocation_type   = CL_MEM_ANDROID_NATIVE_BUFFER_HOST_PTR_QCOM;
    buffer_host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    buffer_host_ptr.anb_ptr = protected_graphic_buffer->getNativeBuffer();
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

    // Allocate the backing protected Android graphic image.
    static android::sp<android::GraphicBuffer> protected_graphic_image = new android::GraphicBuffer();
    android_err = protected_graphic_image->reallocate(
            WIDTH,
            HEIGHT,
            HAL_PIXEL_FORMAT_RGBA_8888,
            1,
            GRALLOC_USAGE_SW_READ_NEVER |
                    GRALLOC_USAGE_SW_WRITE_NEVER |
                    GRALLOC_USAGE_HW_TEXTURE |
                    GRALLOC_USAGE_PRIVATE_UNCACHED |
                    GRALLOC_USAGE_PROTECTED);
    if (android_err != android::NO_ERROR)
    {
        std::cerr << "Error " << android_err << " allocating the protected graphic image.\n";
        return android_err;
    }

    // Create the protected OpenCL image.
    static const cl_image_format IMAGE_FORMAT = {CL_RGBA, CL_UNSIGNED_INT8};
    cl_image_desc image_description  = {0};
    image_description.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_description.image_row_pitch = protected_graphic_image->getStride() * PIXEL_SIZE;
    image_description.image_width = protected_graphic_image->getWidth();
    image_description.image_height = protected_graphic_image->getHeight();
    cl_mem_android_native_buffer_host_ptr image_host_ptr = {{0}};
    image_host_ptr.ext_host_ptr.allocation_type   = CL_MEM_ANDROID_NATIVE_BUFFER_HOST_PTR_QCOM;
    image_host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    image_host_ptr.anb_ptr = protected_graphic_image->getNativeBuffer();
    protected_cl_image = clCreateImage(
            protected_context,
            CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &IMAGE_FORMAT,
            &image_description,
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
        std::cerr << "Error " << err << " setting kernel argument 0 to the protected OpenCL buffer.\n";
        return err;
    }

    err = clSetKernelArg(kernel, 1, sizeof(protected_cl_image), &protected_cl_image);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " setting kernel argument 1 to the protected OpenCL image.\n";
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

    // Finish using the Android graphic memory for safe release.
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

    err = clReleaseMemObject(protected_cl_buffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " releasing the protected OpenCL buffer.\n";
        return err;
    }


    std::cout <<
            "Allocated and initialized OpenCL memory objects with cl_qcom_protected_context\n"
            "and cl_qcom_android_native_buffer_host_ptr!\n";
    return EXIT_SUCCESS;
}