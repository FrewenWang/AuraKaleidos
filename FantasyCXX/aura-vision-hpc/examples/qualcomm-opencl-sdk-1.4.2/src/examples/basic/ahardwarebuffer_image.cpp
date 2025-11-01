//--------------------------------------------------------------------------------------
// File: ahardwarebuffer_image.cpp
// Desc: This program demonstrates the usage of cl_qcom_android_ahardwarebuffer_host_ptr
//       extension using OpenCL images.
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
    #pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
    __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    __kernel void copy2d(
       __read_only image2d_t input_mem,
       __write_only image2d_t output_mem,
       const unsigned int width,
       const unsigned int height,
       const unsigned int depth)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        if ((x >= width) || (y >= height))
            return;
        int2 pos = (int2) (x,y);
        float4 scaledPixel = read_imagef(input_mem,sampler,pos);
        write_imagef(output_mem, pos, scaledPixel);
    }
)";

int main(int argc, char** argv)
{
    cl_wrapper       wrapper;
    cl_program       program       = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel        kernel        = wrapper.make_kernel("copy2d", program);
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
    AHardwareBuffer_Desc input_ahb_desc;
    memset(&input_ahb_desc, 0, sizeof(AHardwareBuffer_Desc));
    input_ahb_desc.width  = WIDTH_IN_PIXELS;
    input_ahb_desc.height = HEIGHT_IN_PIXELS;
    input_ahb_desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    input_ahb_desc.layers = 1;
    input_ahb_desc.usage  = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;

    AHardwareBuffer *p_input_ahb;
    err = AHardwareBuffer_allocate(&input_ahb_desc, &p_input_ahb);
    if (err != CL_SUCCESS)
    {
        std::cerr << "AHardwareBuffer_allocate failed with error code " << err << "\n";
        return err;
    }

    /*
     * Step 2: Create an OpenCL input image object that uses input_ahb_mem as its data store.
     */
    cl_mem_ahardwarebuffer_host_ptr input_ahb_mem;
    input_ahb_mem.ext_host_ptr.allocation_type   = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
    input_ahb_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    input_ahb_mem.ahb_ptr                        = p_input_ahb;

    AHardwareBuffer_describe(p_input_ahb, &input_ahb_desc);

    cl_image_format image_format = {CL_RGBA, CL_UNORM_INT8};
    cl_image_desc image_desc = {};
    size_t image_width = 0;
    size_t image_height = 0;
    image_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width     = image_width = (size_t) input_ahb_desc.width;
    image_desc.image_height    = image_height = (size_t) input_ahb_desc.height;
    image_desc.image_row_pitch = (size_t) input_ahb_desc.stride * BYTES_PER_PIXEL;

    cl_mem in_mem = clCreateImage(context,
                            CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                            &image_format,
                            &image_desc,
                            &input_ahb_mem,
                            &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage\n";
        return err;
    }
    size_t element_size = 0;
    err = clGetImageInfo(in_mem, CL_IMAGE_ELEMENT_SIZE, sizeof(element_size), &element_size, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "clGetImageInfo failed with error code " << err << "\n";
        return err;
    }

    /*
     * Step 3: Create an OpenCL output image object that uses output_ahb_mem as its data store.
     */
    AHardwareBuffer_Desc output_ahb_desc;
    memcpy(&output_ahb_desc, &input_ahb_desc, sizeof(AHardwareBuffer_Desc));
    output_ahb_desc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER;

    AHardwareBuffer *p_output_ahb;
    err = AHardwareBuffer_allocate(&output_ahb_desc, &p_output_ahb);
    if (err != CL_SUCCESS)
    {
        std::cerr << "AHardwareBuffer_allocate failed with " << err << "\n";
        return err;
    }

    cl_mem_ahardwarebuffer_host_ptr output_ahb_mem;
    output_ahb_mem.ext_host_ptr.allocation_type   = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
    output_ahb_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    output_ahb_mem.ahb_ptr                        = p_output_ahb;

    AHardwareBuffer_describe(p_output_ahb, &output_ahb_desc);

    cl_mem out_mem = clCreateImage(context,
                            CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                            &image_format,
                            &image_desc,
                            &output_ahb_mem,
                            &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage\n";
        return err;
    }
    err = clGetImageInfo(out_mem, CL_IMAGE_ELEMENT_SIZE, sizeof(element_size), &element_size, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "clGetImageInfo failed with error code " << err << "\n";
        return err;
    }

    /*
     * Step 3: Set up kernel arguments
     */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0.\n";
        return err;
    }
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1.\n";
        return err;
    }
    size_t w = output_ahb_desc.width;
    err = clSetKernelArg(kernel, 2, sizeof(cl_int), &w);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2.\n";
        return err;
    }
    size_t h = output_ahb_desc.height;
    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &h);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3.\n";
        return err;
    }
    size_t d = output_ahb_desc.layers;
    err = clSetKernelArg(kernel, 4, sizeof(cl_int), &d);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4.\n";
        return err;
    }

    /*
     * Step 4: Map input image for writing and initialize
     */
    size_t origin[3] = {0,0,0}, region[3] = {image_width, image_height, 1}; // image depth is 1 for 2D images
    size_t image_row_pitch = 0;
    cl_uchar* input_data = static_cast<cl_uchar *>(clEnqueueMapImage(
            command_queue,
            in_mem,
            CL_BLOCKING,
            CL_MAP_WRITE,
            origin,
            region,
            &image_row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "clEnqueueMapImage failed with error code " << err << "\n";
        return err;
    }

    cl_uchar* locked_ptr = nullptr;
    // Fence value -1 is passed as CL_BLOCKING in above call makes sure that writing is complete due to CL_BLOCKING flag.
    err = AHardwareBuffer_lock(p_input_ahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, (void **) &locked_ptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while locking p_input_ahb\n";
        return err;
    }

    size_t image_buffer_size = input_ahb_desc.stride * input_ahb_desc.height * element_size * input_ahb_desc.layers;
    memset(input_data, 0x0, image_buffer_size);
    {
        cl_uchar non_trivial_byte = 0x01;
        for (size_t i = 0; i < image_row_pitch * image_height; i += image_row_pitch)
        {
            for(size_t z = 0; z < image_width * element_size; z++)
            {
                input_data[i + z] = non_trivial_byte;
                non_trivial_byte++;  // C standard specifies that unsigned overflows wrap to zero
            }
        }
    }

    err = AHardwareBuffer_unlock(p_input_ahb, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while unlocking p_input_ahb\n";
        return err;
    }

    err = clEnqueueUnmapMemObject(command_queue, in_mem, input_data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "clEnqueueUnmapMemObject failed with error code " << err << "\n";
        return err;
    }

    /*
     * Step 5: Copy input image to output image by executing copy2d kernel
     */
    size_t global_work_size[2] = {image_width, image_height};
    err = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            2, // work dimenion for 2D image
            nullptr,
            global_work_size,
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

    /*
     * Step 6: Map output image for reading and compare against reference values
     */
    cl_uchar* output_data = (cl_uchar*)clEnqueueMapImage(
            command_queue,
            out_mem,
            CL_BLOCKING,
            CL_MAP_WRITE,
            origin,
            region,
            &image_row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "clEnqueueMapImage failed with error code " << err << "\n";
        return err;
    }

    err = AHardwareBuffer_lock(p_output_ahb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, nullptr, (void **) &locked_ptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while locking p_output_ahb\n";
        return err;
    }

    for(size_t i = 0; i < image_height; i++)
    {
        for(size_t j = 0; j < image_width; j++)
        {
            for(size_t k = 0; k < element_size; k++)
            {
                size_t offset = i*image_row_pitch + j*element_size + k;
                if(input_data[offset] != output_data[offset])
                {
                    std::cerr << "Mismatch at row "<< i << " col "<< j << " channel "<< k << ". Expected : " << input_data[offset] << " Actual : "<< output_data[offset] <<".\n";
                    return EXIT_FAILURE;
                }
            }
        }
    }

    err = AHardwareBuffer_unlock(p_output_ahb, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while unlocking p_output_ahb\n";
        return err;
    }

    err = clEnqueueUnmapMemObject(command_queue, out_mem, output_data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "clEnqueueUnmapMemObject failed with error code " << err << "\n";
        return err;
    }

    /*
     * Step 7: Clean up cl resources that aren't automatically handled by cl_wrapper
     */
    clReleaseMemObject(in_mem);
    clReleaseMemObject(out_mem);
    AHardwareBuffer_release(p_input_ahb);
    AHardwareBuffer_release(p_output_ahb);

    std::cout << "ahardwarebuffer_image sample executed successfully\n";

    return EXIT_SUCCESS;
}
