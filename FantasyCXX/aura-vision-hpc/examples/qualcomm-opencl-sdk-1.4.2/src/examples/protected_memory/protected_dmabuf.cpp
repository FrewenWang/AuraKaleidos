//--------------------------------------------------------------------------------------
// File: protected_dmabuf.cpp
// Desc: Demonstrates the cl_qcom_protected_context and cl_qcom_dmabuf_host_ptr extensions.
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
#include "util/cl_wrapper.h"

#include <vmmem_wrapper.h>

static const char *PROGRAM_SOURCE =
        "__kernel void initialize_protected_memory(__global int *buffer, __write_only __global image2d_t image) {\n"
        "    buffer[0] = 1;                                                                                      \n"
        "    write_imageui(image, (int2)(0, 0), (uint4)(1, 0, 0, 0));                                            \n"
        "}                                                                                                       \n";

int main(int argc, char** argv)
{
    cl_int                             err                         = CL_SUCCESS;
    static const cl_context_properties CONTEXT_PROPERTIES[]        = {CL_CONTEXT_PROTECTED_QCOM, 1, 0};
    cl_wrapper                         wrapper(CONTEXT_PROPERTIES);
    int                                buffer_fd               = 0;
    int                                image_fd                = 0;
    cl_mem                             protected_cl_buffer         = NULL;
    cl_mem                             protected_cl_image          = NULL;

    if (argc != 1)
    {
        std::cerr <<
                "The sample takes no arguments.\n"
                "\n"
                "Usage: " << argv[0] << "\n"
                "\n"
                "0 is returned after initializing memory using the cl_qcom_protected_context\n"
                "and cl_qcom_dmabuf_host_ptr extensions. 1 is returned on failure.\n";
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

    if (!wrapper.check_extension_support("cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr <<
                "Extension cl_qcom_dmabuf_host_ptr needed for dmabuf (zero copy) memory is not\n"
                "supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create protected dmabuf backed OpenCL memory objects.
     */

    cl_device_id                 device_id         = wrapper.get_device_id();
    cl_context                   protected_context = wrapper.get_context();
    cl_uint                      ext_mem_padding   = 0; // in bytes
    static const size_t          IMAGE_WIDTH       = 1; // in pixels
    static const size_t          IMAGE_HEIGHT      = 1; // in pixels
    BufferAllocator*             dmabuf_allocator  = CreateDmabufHeapBufferAllocator();
    size_t                       image_row_pitch   = 0; // in bytes
    static const size_t          BUFFER_SIZE       = sizeof(cl_int); // in bytes
    static const cl_image_format IMAGE_FORMAT      = {CL_RGBA, CL_UNSIGNED_INT8};

    err = clGetDeviceInfo(device_id, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(ext_mem_padding), &ext_mem_padding, NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " getting the extended memory padding.\n";
        return err;
    }

    // Allocate the backing protected dmabuf buffer.
    size_t padded_protected_buffer_size = BUFFER_SIZE + ext_mem_padding; // in bytes
    buffer_fd = DmabufHeapAlloc(dmabuf_allocator, "qcom,system", padded_protected_buffer_size, 0, 0);
    if(buffer_fd < 0)
    {
        std::cerr << "Error " << err << " allocating the protected dmabuf buffer.\n";
        return err;
    }

    // Create protected virtual memory for dmabuf buffer descriptor
    {
        VmMem *vmmem = CreateVmMem();
        if (vmmem == nullptr) {
            std::cerr << "CreateVmMem() failed to get an object\n";
            return -1;
        }

        VmHandle handle = FindVmByName(vmmem, (char*) "qcom,cp_pixel");
        if(handle < 0)
        {
            std::cerr << "Failed to find VM named qcom,cp_pixel";
            return handle;
        }

        VmHandle handle_arr[1] = {handle};
        uint32_t perm_arr[1] = {VMMEM_READ | VMMEM_WRITE};

        int ret = LendDmabuf(vmmem, buffer_fd, handle_arr, perm_arr, 1);
        if (ret < 0) {
            std::cerr << "LendDmabuf() failed, dma_buf_fd [" << buffer_fd << "] with errcode: << " << ret << "\n";
            return ret;
        }
        FreeVmMem(vmmem);
    }

    // Create the protected OpenCL buffer.
    cl_mem_dmabuf_host_ptr dmabuf_buf_mem = {};
    dmabuf_buf_mem.ext_host_ptr.allocation_type   = CL_MEM_DMABUF_HOST_PTR_PROTECTED_QCOM;
    dmabuf_buf_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    dmabuf_buf_mem.dmabuf_filedesc                = buffer_fd;
    dmabuf_buf_mem.dmabuf_hostptr                 = nullptr;
    protected_cl_buffer = clCreateBuffer(
            protected_context,
            CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            BUFFER_SIZE,
            &dmabuf_buf_mem,
            &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " creating the protected OpenCL buffer.\n";
        return err;
    }

    err = clGetDeviceImageInfoQCOM(
            device_id,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            &IMAGE_FORMAT,
            CL_IMAGE_ROW_PITCH,
            sizeof(image_row_pitch),
            &image_row_pitch,
            NULL);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " getting the image's row pitch.\n";
        return err;
    }

    // Allocate the backing protected dmabuf image.
    size_t padded_image_size = IMAGE_HEIGHT * image_row_pitch + ext_mem_padding;
    image_fd = DmabufHeapAlloc(dmabuf_allocator, "qcom,system", padded_image_size, 0, 0);
    if(image_fd < 0)
    {
        std::cerr << "Error " << err << " allocating the protected dmabuf image.\n";
        return err;
    }

    // Create protected virtual memory for dmabuf image descriptor
    {
        VmMem *vmmem = CreateVmMem();

        VmHandle handle = FindVmByName(vmmem, (char*) "qcom,cp_pixel");

        VmHandle handle_arr[1] = {handle};
        uint32_t perm_arr[1] = {VMMEM_READ | VMMEM_WRITE};

        int ret = LendDmabuf(vmmem, image_fd, handle_arr, perm_arr, 1);

        if (ret < 0) {
            std::cerr << "LendDmabuf() failed, dma_buf_fd [" << buffer_fd << "] with errcode: << " << ret << "\n";
            return ret;
        }
        FreeVmMem(vmmem);
    }

    // Create the protected OpenCL image.
    cl_image_desc image_description = {0};
    image_description.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_description.image_width = IMAGE_WIDTH;
    image_description.image_height = IMAGE_HEIGHT;
    image_description.image_row_pitch = image_row_pitch;

    cl_mem_dmabuf_host_ptr dmabuf_img_mem = {};
    dmabuf_img_mem.ext_host_ptr.allocation_type   = CL_MEM_DMABUF_HOST_PTR_PROTECTED_QCOM;
    dmabuf_img_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    dmabuf_img_mem.dmabuf_filedesc                = image_fd;
    dmabuf_img_mem.dmabuf_hostptr                 = nullptr;
    protected_cl_image = clCreateImage(
            protected_context,
            CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &IMAGE_FORMAT,
            &image_description,
            &dmabuf_img_mem,
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

    cl_command_queue protected_command_queue = wrapper.get_command_queue();
    static const size_t GLOBAL_WORK_SIZE[] = {1};
    static const size_t LOCAL_WORK_SIZE[] = {1};
    err = clEnqueueNDRangeKernel(
            protected_command_queue,
            kernel,
            1,
            NULL,
            GLOBAL_WORK_SIZE,
            LOCAL_WORK_SIZE,
            0,
            NULL,
            NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " enqueuing kernel to initialize protected memory.\n";
        return err;
    }

    err = clFinish(protected_command_queue);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " finishing computation.\n";
        return err;
    }

    /*
     * Step 3: Clean up resources that aren't automatically handled by cl_wrapper.
     */

    err = clReleaseMemObject(protected_cl_image);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " releasing the protected OpenCL image.\n";
        return err;
    }

    err = close(buffer_fd);
    if (err < 0)
    {
        std::cerr << "Error " << err << " closing the protected dmabuf buffer.\n";
        return err;
    }

    err = clReleaseMemObject(protected_cl_buffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " releasing the protected OpenCL buffer.\n";
        return err;
    }

    err = close(image_fd);
    if (err < 0)
    {
        std::cerr << "Error " << err << " closing the protected dmabuf image buffer.\n";
        return err;
    }

    FreeDmabufHeapBufferAllocator(dmabuf_allocator);

    std::cout <<
            "Allocated and initialized OpenCL memory objects with cl_qcom_protected_context\n"
            "and cl_qcom_dmabuf_host_ptr!\n";
    return EXIT_SUCCESS;
}