//--------------------------------------------------------------------------------------
// File: protected_ion.cpp
// Desc: Demonstrates the cl_qcom_protected_context and cl_qcom_ion_host_ptr extensions.
//
// Author: QUALCOMM
//
//             Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_ext_qcom.h>

// ION
#include <ion/ion.h>
#include <linux/msm_ion.h> // Vendor specific Ion heap IDs and flags.

#define CL_ASSERT_ERRCODE(errcode)                                                                  \
    if ((errcode) != CL_SUCCESS)                                                                    \
    {                                                                                               \
        std::cerr << "OpenCL API error code " << (errcode) << " on line " << __LINE__ << "\n";      \
    }

static const char *PROGRAM_SOURCE = R"(
__kernel void initialize_protected_memory(__global int *buffer, __write_only __global image2d_t image) {
    buffer[0] = 1;
    write_imageui(image, (int2)(0, 0), (uint4)(1, 0, 0, 0));
}
)";

bool confirm_extension_support(cl_device_id  device, const char *extnStr)
{
    cl_int      errcode = CL_SUCCESS;
    char        extensions[2048] = {0};

    errcode = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), &extensions, nullptr);

    if(errcode != CL_SUCCESS)
    {
        std::cerr << "Error " << errcode << " with clGetDeviceInfo for extensions." << "\n";
        std::exit(errcode);
    }
    return (strstr(extensions, extnStr) != nullptr);
}

int main(int argc, char** argv)
{
    cl_int                             errcode                      = 0;
    cl_device_id                       device                       = nullptr;
    cl_platform_id                     platform                     = nullptr;
    cl_context                         context                      = nullptr;
    cl_command_queue                   queue                        = nullptr;
    cl_program                         program                      = nullptr;
    cl_kernel                          kernel                       = nullptr;
    static const cl_context_properties CONTEXT_PROPERTIES[]         = {CL_CONTEXT_PROTECTED_QCOM, 1, 0};
    int                                ion_fd                       = -1;
    cl_uint                            device_page_size             = 0; // in bytes
    cl_uint                            ext_mem_padding              = 0; // in bytes
    static const size_t                GLOBAL_WORK_SIZE[]           = {1};
    static const size_t                LOCAL_WORK_SIZE[]            = {1};
    // Image
    int                                image_ion_fd                 = 0;
    cl_mem                             protected_cl_image           = nullptr;
    static const size_t                image_width                  = 1; // in pixels
    static const size_t                image_height                 = 1; // in pixels
    size_t                             image_row_pitch              = 0; // in bytes
    static const cl_image_format       image_format                 = {CL_RGBA, CL_UNSIGNED_INT8};
    cl_image_desc                      image_description            = {0};
    cl_mem_ion_host_ptr                image_host_ptr               = {{0}};
    size_t                             padded_image_size            = 0;
    // Buffer
    int                                buffer_ion_fd                = 0;
    cl_mem                             protected_cl_buffer          = nullptr;
    static const size_t                buffer_size                  = sizeof(cl_int); // in bytes
    cl_mem_ion_host_ptr                buffer_host_ptr              = {{0}};
    size_t                             padded_protected_buffer_size = 0;


    if (argc != 1)
    {
        std::cerr <<
                "The sample takes no arguments.\n"
                "\n"
                "Usage: " << argv[0] << "\n"
                "\n"
                "0 is returned after initializing memory using the cl_qcom_protected_context\n"
                "and cl_qcom_ion_host_ptr extensions. 1 is returned on failure.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    /*
     * Step 0: Open ION
     */
    ion_fd = ion_open();
    if(ion_fd < 0)
    {
        std::cerr << "Error with ion_open()\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    /*
     * Step 1: setup in order queue
     */
    errcode = clGetPlatformIDs(1, &platform, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    errcode = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    context = clCreateContext(CONTEXT_PROPERTIES, 1, &device, nullptr, nullptr, &errcode);
    CL_ASSERT_ERRCODE(errcode);
    queue = clCreateCommandQueueWithProperties(context, device, nullptr, &errcode);
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 2: Confirm the required OpenCL extensions are supported.
     */
    if (!confirm_extension_support(device, "cl_qcom_protected_context"))
    {
        std::cerr <<
                "Extension cl_qcom_protected_context needed for a protected context is not\n"
                "supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    if (!confirm_extension_support(device, "cl_qcom_ion_host_ptr"))
    {
        std::cerr <<
                "Extension cl_qcom_ion_host_ptr needed for ION (zero copy) memory is not\n"
                "supported.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    /*
     * Step 3: Create protected Ion backed OpenCL memory objects.
     */
    errcode = clGetDeviceInfo(device, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(device_page_size), &device_page_size, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    errcode = clGetDeviceInfo(device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(ext_mem_padding), &ext_mem_padding, nullptr);
    CL_ASSERT_ERRCODE(errcode);

    // Allocate the backing protected Ion buffer.
    padded_protected_buffer_size = buffer_size + ext_mem_padding; // in bytes
    errcode = ion_alloc_fd(
            ion_fd,
            padded_protected_buffer_size,
            device_page_size,
            ION_HEAP(ION_SECURE_HEAP_ID),
            ION_FLAG_SECURE | ION_FLAG_CP_PIXEL,
            &buffer_ion_fd);
    if (errcode == -1)
    {
        std::cerr << "Error " << errcode << " allocating the protected Ion buffer.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    // Create the protected OpenCL buffer.
    buffer_host_ptr.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_PROTECTED_QCOM;
    buffer_host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    buffer_host_ptr.ion_filedesc = buffer_ion_fd;
    buffer_host_ptr.ion_hostptr = nullptr;
    protected_cl_buffer = clCreateBuffer(
            context,
            CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            buffer_size,
            &buffer_host_ptr,
            &errcode);
    CL_ASSERT_ERRCODE(errcode);

    // Allocate the backing protected Ion image.
    errcode = clGetDeviceImageInfoQCOM(
        device,
        image_width,
        image_height,
        &image_format,
        CL_IMAGE_ROW_PITCH,
        sizeof(image_row_pitch),
        &image_row_pitch,
        nullptr);
    CL_ASSERT_ERRCODE(errcode);

    padded_image_size = image_height * image_row_pitch + ext_mem_padding;
    errcode = ion_alloc_fd(
            ion_fd,
            padded_image_size,
            device_page_size,
            ION_HEAP(ION_SECURE_HEAP_ID),
            ION_FLAG_SECURE | ION_FLAG_CP_PIXEL,
            &image_ion_fd);
    if (errcode == -1)
    {
        std::cerr << "Error " << errcode << " allocating the protected Ion image.\n";
        errcode = EXIT_FAILURE;
        goto cleanup;
    }

    // Create the protected OpenCL image.
    image_description.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_description.image_width = image_width;
    image_description.image_height = image_height;
    image_description.image_row_pitch = image_row_pitch;

    image_host_ptr.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_PROTECTED_QCOM;
    image_host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    image_host_ptr.ion_filedesc = image_ion_fd;
    image_host_ptr.ion_hostptr = nullptr;
    protected_cl_image = clCreateImage(
            context,
            CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &image_format,
            &image_description,
            &image_host_ptr,
            &errcode);
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 4: Set up kernel arguments and run the kernel.
     */
    program = clCreateProgramWithSource(context, 1, &PROGRAM_SOURCE, nullptr, &errcode);
    CL_ASSERT_ERRCODE(errcode);
    errcode = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    CL_ASSERT_ERRCODE(errcode);
    kernel = clCreateKernel(program, "initialize_protected_memory",  &errcode);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clSetKernelArg(kernel, 0, sizeof(protected_cl_buffer), &protected_cl_buffer);
    CL_ASSERT_ERRCODE(errcode);
    errcode = clSetKernelArg(kernel, 1, sizeof(protected_cl_image), &protected_cl_image);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clEnqueueNDRangeKernel(
            queue,
            kernel,
            1,
            nullptr,
            GLOBAL_WORK_SIZE,
            LOCAL_WORK_SIZE,
            0,
            nullptr,
            nullptr);
    CL_ASSERT_ERRCODE(errcode);

    errcode = clFinish(queue);
    CL_ASSERT_ERRCODE(errcode);

    /*
     * Step 5: Clean up resources
     */
cleanup:
    if(protected_cl_image != nullptr)
    {
        clReleaseMemObject(protected_cl_image);
    }
    if(protected_cl_buffer != nullptr)
    {
        clReleaseMemObject(protected_cl_buffer);
    }
    if(kernel != nullptr)
    {
        clReleaseKernel(kernel);
    }
    if(program != nullptr)
    {
        clReleaseProgram(program);
    }
    if(queue != nullptr)
    {
        clReleaseCommandQueue(queue);
    }
    if(context != nullptr)
    {
        clReleaseContext(context);
    }
    if (close(image_ion_fd) < 0)
    {
        std::cerr << "Error " << errcode << " closing the protected ION image.\n";
        errcode = EXIT_FAILURE;
    }
    if (close(buffer_ion_fd) < 0)
    {
        std::cerr << "Error " << errcode << " closing the protected ION buffer.\n";
        errcode = EXIT_FAILURE;
    }
    if (ion_close(ion_fd) < 0)
    {
        std::cerr << "Error closing ion device fd.\n";
        errcode = EXIT_FAILURE;
    }


    if(errcode == EXIT_SUCCESS)
    {
        std::cout <<
                  "Successfully allocated and initialized OpenCL memory objects with cl_qcom_protected_context\n"
                  "and cl_qcom_ion_host_ptr!\n";
    } else
    {
        std::cerr <<
                  "Failed to allocate and initialized OpenCL memory objects with cl_qcom_protected_context\n"
                  "and cl_qcom_ion_host_ptr!\n";
    }
    return errcode;
}