//--------------------------------------------------------------------------------------
// File: qcom_filter_bicubic.cpp
// Desc: This program shows how to use the cl_qcom_filter_bicubic extension. It shows
//       how to use literal and kernel argument samplers.
//
// Author:      QUALCOMM
//
//          Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>

// Project includes
#include "util/cl_wrapper.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>

constexpr float      SCALE_FACTOR      = 2.0f;

static const char *PROGRAM_SOURCE = R"(
    #pragma OPENCL EXTENSION cl_qcom_filter_bicubic : enable
    __kernel void upscale_image_rgba(
        __read_only image2d_t input1Mem,
        __read_only image2d_t input2Mem,
        __write_only image2d_t outputMem,
        sampler_t argSampler
    )
    {
        const sampler_t litSampler = CLK_NORMALIZED_COORDS_TRUE |
                                     CLK_ADDRESS_CLAMP          |
                                     QCOM_CLK_FILTER_BICUBIC;

        int w = get_image_width(outputMem);
        int h = get_image_height(outputMem);

        int outX = get_global_id(0);
        int outY = get_global_id(1);
        int2 posOut = {outX, outY};

        float inX = outX / (float) w;
        float inY = outY / (float) h;
        float2 posIn = (float2) (inX, inY);

        float4 pixel1 = read_imagef(input1Mem, argSampler, posIn);
        float4 pixel2 = read_imagef(input2Mem, litSampler, posIn);
        write_imagef(outputMem, posOut, pixel1 + pixel2);
    }
)";

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Please specify input and output images.\n"
                     "\n"
                     "Usage: " << argv[0] << " <input1> <input2> <output>\n"
                                             "\n"
                                             "This sample performs hardware accelerated bicubic interpolation to upscale\n"
                                             "and combine the input images. Both input images must be RGBA format and\n"
                                             "have the same dimensions.\n";
        return EXIT_FAILURE;
    }
    const std::string in1_filename(argv[1]);
    const std::string in2_filename(argv[2]);
    const std::string out_filename(argv[3]);

    cl_wrapper           wrapper;
    cl_program           program           = wrapper.make_program(&PROGRAM_SOURCE, 1, "-cl-std=CL2.0");
    cl_kernel            kernel            = wrapper.make_kernel("upscale_image_rgba", program);
    cl_context           context           = wrapper.get_context();
    cl_command_queue     command_queue     = wrapper.get_command_queue();
    cl_int               err               = CL_SUCCESS;
    rgba_image_t         input1_image_info = load_rgba_image_data(in1_filename);
    rgba_image_t         input2_image_info = load_rgba_image_data(in2_filename);
    struct dma_buf_sync  buf_sync          = {};
    cl_event             unmap_event       = nullptr;


    /*
     * Step 0: Confirm the required OpenCL extensions are supported
     */

    if(!wrapper.check_extension_support("cl_qcom_filter_bicubic"))
    {
        std::cerr << "Extension cl_qcom_filter_bicubic is not supported on this device\n";
        return EXIT_FAILURE;
    }

    if(!wrapper.check_extension_support("cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for DMABUF-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    if(!wrapper.check_extension_support("cl_qcom_ext_host_ptr_iocoherent"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr_iocoherent needed for DMABUF-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    if(!wrapper.check_extension_support("cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for DMABUF-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create suitable dmabuf-backed CL images.
     */
    // Input image 1
    cl_image_format     input1_image_format        = {0};
    input1_image_format.image_channel_order = CL_RGBA;
    input1_image_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc       input1_image_desc          = {0};
    input1_image_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    input1_image_desc.image_width  = input1_image_info.width;
    input1_image_desc.image_height = input1_image_info.height;
    input1_image_desc.image_row_pitch = wrapper.get_image_row_pitch(input1_image_format, input1_image_desc);

    cl_mem_dmabuf_host_ptr input1_image_dmabuf_mem = wrapper.make_buffer_for_nonplanar_image(input1_image_format, input1_image_desc);
    cl_mem input1_image = clCreateImage(context,
                                       CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                       &input1_image_format,
                                       &input1_image_desc,
                                       &input1_image_dmabuf_mem,
                                       &err);

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for input 1 image." << "\n";
        return err;
    }

    // Input image 2
    cl_image_format     input2_image_format        = {0};
    input2_image_format.image_channel_order = CL_RGBA;
    input2_image_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc       input2_image_desc          = {0};
    input2_image_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    input2_image_desc.image_width  = input2_image_info.width;
    input2_image_desc.image_height = input2_image_info.height;
    input2_image_desc.image_row_pitch = wrapper.get_image_row_pitch(input2_image_format, input2_image_desc);

    cl_mem_dmabuf_host_ptr input2_image_dmabuf_mem = wrapper.make_buffer_for_nonplanar_image(input2_image_format, input2_image_desc);
    cl_mem input2_image = clCreateImage(context,
                                        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                        &input2_image_format,
                                        &input2_image_desc,
                                        &input2_image_dmabuf_mem,
                                        &err);

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for input 2 image." << "\n";
        return err;
    }

    // Output image
    cl_image_format     output_image_format        = {0};
    output_image_format.image_channel_order = CL_RGBA;
    output_image_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc       output_image_desc          = {0};
    output_image_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    output_image_desc.image_width = static_cast<size_t>(input1_image_desc.image_width * SCALE_FACTOR);
    output_image_desc.image_height = static_cast<size_t>(input1_image_desc.image_height * SCALE_FACTOR);
    output_image_desc.image_row_pitch = wrapper.get_image_row_pitch(output_image_format, output_image_desc);

    cl_mem_dmabuf_host_ptr output_image_dmabuf_mem = wrapper.make_buffer_for_nonplanar_image(output_image_format, output_image_desc);
    cl_mem output_image = clCreateImage(context,
                                        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                        &output_image_format,
                                        &output_image_desc,
                                        &output_image_dmabuf_mem,
                                        &err);

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image." << "\n";
        return err;
    }

    /*
     * Step 2: Copy data to input images.
     */

    size_t               origin[]            = {0,0,0};
    const size_t         input1_region[]      = {input1_image_desc.image_width, input1_image_desc.image_height,1};
    const size_t         input2_region[]      = {input2_image_desc.image_width, input2_image_desc.image_height,1};
    size_t               image_row_pitch     = 0;
    uint8_t*             image_host_ptr      = nullptr;
    size_t               element_size        = 0;

    err = clGetImageInfo (input1_image,
                          CL_IMAGE_ELEMENT_SIZE,
                          sizeof(size_t),
                          reinterpret_cast<void*>(&element_size),
                          nullptr);

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clGetImageInfo for element size." << "\n";
        return err;
    }

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(input1_image_dmabuf_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }
    // Image 1
    image_host_ptr = reinterpret_cast<uint8_t*>(clEnqueueMapImage(
        command_queue,
        input1_image,
        CL_TRUE,
        CL_MAP_WRITE,
        origin,
        input1_region,
        &image_row_pitch,
        nullptr,
        0,
        nullptr,
        nullptr,
        &err
    ));

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueMapImage when mapping the input 1 image for writing." << "\n";
        return err;
    }

    for(unsigned row = 0; row < input1_image_desc.image_height; ++row)
    {
        std::memcpy(&image_host_ptr[row * image_row_pitch], &input1_image_info.pixels[row * input1_image_desc.image_width * element_size], input1_image_desc.image_width * element_size);
    }

    err = clEnqueueUnmapMemObject(command_queue,
                                  input1_image,
                                  reinterpret_cast<void*>(image_host_ptr),
                                  0,
                                  nullptr,
                                  &unmap_event);

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueUnmapMemObject when unmapping the input 1 image." << "\n";
        return err;
    }

    err = clWaitForEvents(1, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while waiting to unmap.\n";
        return err;
    }

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(input1_image_dmabuf_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(input2_image_dmabuf_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }
    // Image 2
    image_host_ptr = reinterpret_cast<uint8_t*>(clEnqueueMapImage(
        command_queue,
        input2_image,
        CL_TRUE,
        CL_MAP_WRITE,
        origin,
        input2_region,
        &image_row_pitch,
        nullptr,
        0,
        nullptr,
        nullptr,
        &err
    ));

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueMapImage when mapping the input 2 image for writing." << "\n";
        return err;
    }

    for(unsigned row = 0; row < input2_image_desc.image_height; ++row)
    {
        std::memcpy(&image_host_ptr[row * image_row_pitch], &input2_image_info.pixels[row * input2_image_desc.image_width * element_size], input2_image_desc.image_width * element_size);
    }

    err = clEnqueueUnmapMemObject(command_queue,
                                  input2_image,
                                  reinterpret_cast<void*>(image_host_ptr),
                                  0,
                                  nullptr,
                                  &unmap_event);

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueUnmapMemObject when unmapping the input 2 image." << "\n";
        return err;
    }

    err = clWaitForEvents(1, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while waiting to unmap.\n";
        return err;
    }

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(input2_image_dmabuf_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }
    /*
     * Step 3: Setup other kernel arguments
     */

    cl_sampler sampler = clCreateSampler(
        context,
        CL_TRUE,
        CL_ADDRESS_CLAMP,
        CL_FILTER_BICUBIC_QCOM,
        &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateSampler." << "\n";
        return err;
    }

    /*
     * Step 4: Run the kernel
     */

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0 for input 1 image." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1 for input 2 image." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2 for output image." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel, 3, sizeof(cl_sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3 for sampler." << "\n";
        return err;
    }

    size_t work_size[2] = {output_image_desc.image_width, output_image_desc.image_height};
    err = clEnqueueNDRangeKernel(command_queue,
                                 kernel,
                                 2,
                                 nullptr,
                                 work_size,
                                 nullptr,
                                 0,
                                 nullptr,
                                 nullptr);

    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel for kernel." << "\n";
        return err;
    }

    err = clFinish(command_queue);

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clFinish." << "\n";
        return err;
    }

    /*
     * Step 5: Save the output image
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if ( ioctl(output_image_dmabuf_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    size_t               output_num_pixels    = output_image_desc.image_width * output_image_desc.image_height;
    rgba_image_t         output_image_info;
    output_image_info.width  = output_image_desc.image_width;
    output_image_info.height = output_image_desc.image_height;
    output_image_info.pixels.resize(output_num_pixels * element_size);

    const size_t output_region[]      = {output_image_desc.image_width, output_image_desc.image_height, 1};
    image_row_pitch                   = 0;
    image_host_ptr = reinterpret_cast<uint8_t*>(clEnqueueMapImage(
        command_queue,
        output_image,
        CL_TRUE,
        CL_MAP_READ,
        origin,
        output_region,
        &image_row_pitch,
        nullptr,
        0,
        nullptr,
        nullptr,
        &err
    ));

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueMapImage when mapping the output image for reading." << "\n";
        return err;
    }

    for(unsigned row = 0; row < output_image_desc.image_height; ++row)
    {
        std::memcpy(&output_image_info.pixels[row * output_image_desc.image_width * element_size], &image_host_ptr[row * image_row_pitch], output_image_desc.image_width * element_size);
    }

    err = clEnqueueUnmapMemObject(command_queue,
                                  output_image,
                                  reinterpret_cast<void*>(image_host_ptr),
                                  0,
                                  nullptr,
                                  &unmap_event);

    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueUnmapMemObject when unmapping the output image." << "\n";
        return err;
    }

    err = clWaitForEvents(1, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while waiting to unmap.\n";
        return err;
    }

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    if ( ioctl(output_image_dmabuf_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }
    save_rgba_image_data(out_filename, output_image_info);

    /*
     * Clean up resources
     */

    if(input1_image)
    {
        clReleaseMemObject(input1_image);
    }

    if(input2_image)
    {
        clReleaseMemObject(input2_image);
    }

    if(output_image)
    {
        clReleaseMemObject(output_image);
    }

    if(sampler)
    {
        clReleaseSampler(sampler);
    }

    return EXIT_SUCCESS;
}
