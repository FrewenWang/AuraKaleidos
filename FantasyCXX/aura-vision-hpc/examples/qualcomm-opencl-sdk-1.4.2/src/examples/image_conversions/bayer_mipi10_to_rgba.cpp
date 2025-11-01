//--------------------------------------------------------------------------------------
// File: bayer_mipi10_to_rgba.cpp
// Desc: This program converts BAYER MIPI10 to RGBA.
//
// Author:      QUALCOMM
//
//               Copyright (c) 2018, 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <cstdlib>
#include <cstring>
#include <iostream>

// Project includes
#include "util/cl_wrapper.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

static const char *PROGRAM_SOURCE = R"(
    // Illustrates a simple demosaicing scheme.
    // The idea of exploiting the GPU's bilinear interpolation capability can be
    // applied to other interpolation schemes.
    __kernel void bayer_to_rgba(__read_only  image2d_t bayer_image,
                                __write_only image2d_t rgba_image,
                                             sampler_t sampler)
    {
        const int    wid_x          = get_global_id(0);
        const int    wid_y          = get_global_id(1);
        const float2 coord          = (float2)(wid_x, wid_y) + 0.5f;
        const float4 bayer_pixels[] = {
            read_imagef(bayer_image, sampler, coord + (float2)(0.,  0.)),
            read_imagef(bayer_image, sampler, coord + (float2)(0.5, 0.)),
            read_imagef(bayer_image, sampler, coord + (float2)(0.,  0.5)),
            read_imagef(bayer_image, sampler, coord + (float2)(0.5, 0.5))
        };
        const float4 rgba_pixels[] = {
            (float4)(bayer_pixels[0].w, 0.5f * (bayer_pixels[0].y + bayer_pixels[0].z), bayer_pixels[0].x, 1.f),
            (float4)(bayer_pixels[1].w, 0.5f * (bayer_pixels[1].y + bayer_pixels[1].z), bayer_pixels[1].x, 1.f),
            (float4)(bayer_pixels[2].w, 0.5f * (bayer_pixels[2].y + bayer_pixels[2].z), bayer_pixels[2].x, 1.f),
            (float4)(bayer_pixels[3].w, 0.5f * (bayer_pixels[3].y + bayer_pixels[3].z), bayer_pixels[3].x, 1.f)
        };
        const int2 write_coord = (int2)(2 * wid_x, 2 * wid_y);
        write_imagef(rgba_image, write_coord + (int2)(0, 0), rgba_pixels[0]);
        write_imagef(rgba_image, write_coord + (int2)(1, 0), rgba_pixels[1]);
        write_imagef(rgba_image, write_coord + (int2)(0, 1), rgba_pixels[2]);
        write_imagef(rgba_image, write_coord + (int2)(1, 1), rgba_pixels[3]);
    }
)";

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Please specify source and output images.\n"
                     "\n"
                     "Usage: " << argv[0] << " <source image data file> <output image data file>\n"
                     "\n"
                     "Demonstrates conversion from Bayer image order to RGBA, a.k.a. \"demosaicing\".\n";
        return EXIT_FAILURE;
    }
    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper wrapper;
    cl_program           program              = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel            kernel               = wrapper.make_kernel("bayer_to_rgba", program);
    cl_context           context              = wrapper.get_context();
    cl_command_queue     command_queue        = wrapper.get_command_queue();
    bayer_mipi10_image_t src_bayer_image_info = load_bayer_mipi_10_image_data(src_image_filename);
    struct dma_buf_sync  buf_sync             = {};
    cl_event             unmap_event          = nullptr;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_other_image"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for Bayer image format and MIPI10 data type is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for DMA buffer-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for DMA buffer-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create suitable DMA buffer-backed CL images.
     */

    cl_image_format src_format;
    src_format.image_channel_order     = CL_QCOM_BAYER;
    src_format.image_channel_data_type = CL_QCOM_UNORM_MIPI10;

    cl_image_desc src_desc;
    std::memset(&src_desc, 0, sizeof(src_desc));
    src_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    src_desc.image_width     = src_bayer_image_info.width;
    src_desc.image_height    = src_bayer_image_info.height;
    src_desc.image_row_pitch = wrapper.get_image_row_pitch(src_format, src_desc);

    cl_mem_dmabuf_host_ptr src_mem = wrapper.make_buffer_for_nonplanar_image(src_format, src_desc);
    cl_int              err         = 0;
    cl_mem              src_image   = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &src_format,
            &src_desc,
            &src_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image." << "\n";
        return err;
    }

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if (ioctl(src_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    const size_t   origin[]     = {0, 0, 0};
    size_t         row_pitch    = 0 ;
    const size_t   src_region[] = {src_desc.image_width, src_desc.image_height, 1};
    unsigned char *image_ptr    = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            src_image,
            CL_BLOCKING,
            CL_MAP_WRITE,
            origin,
            src_region,
            &row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueMapImage for source image." << "\n";
        return err;
    }

    // Copies image data from the host to the DMA buffer.
    for (uint32_t i = 0; i < src_desc.image_height; ++i)
    {
        std::memcpy(
                image_ptr                          + i * src_desc.image_row_pitch,
                src_bayer_image_info.pixels.data() + i * src_desc.image_width / 4 * 5,
                src_desc.image_width / 4 * 5
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, src_image, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image." << "\n";
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
    if (ioctl(src_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    cl_image_format out_format;
    out_format.image_channel_order     = CL_RGBA;
    out_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_desc;
    std::memset(&out_desc, 0, sizeof(out_desc));
    out_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    out_desc.image_width     = src_bayer_image_info.width;
    out_desc.image_height    = src_bayer_image_info.height;
    out_desc.image_row_pitch = wrapper.get_image_row_pitch(out_format, out_desc);

    cl_mem_dmabuf_host_ptr out_mem = wrapper.make_buffer_for_nonplanar_image(out_format, out_desc);
    cl_mem out_image = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_format,
            &out_desc,
            &out_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image." << "\n";
        return err;
    }

    /*
     * Step 2: Set up kernel arguments and run the kernel.
     */

    cl_sampler sampler = clCreateSampler(
            context,
            CL_FALSE,
            CL_ADDRESS_CLAMP_TO_EDGE,
            CL_FILTER_LINEAR,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateSampler." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel, 0, sizeof(src_image), &src_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel, 1, sizeof(out_image), &out_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2." << "\n";
        return err;
    }

    const size_t global_work_size[] = {out_desc.image_width / 2, out_desc.image_height / 2};
    err = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            2,
            nullptr,
            global_work_size,
            nullptr,
            0,
            nullptr,
            nullptr
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel." << "\n";
        return err;
    }

    /*
     * Step 3: Copy the data out of the DMA buffer.
     */

    rgba_image_t out_image_info;
    out_image_info.width  = out_desc.image_width;
    out_image_info.height = out_desc.image_height;
    out_image_info.pixels.resize((out_image_info.width * 4) * out_image_info.height);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if (ioctl(out_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    const size_t out_region[] = {out_desc.image_width, out_desc.image_height, 1};
    row_pitch                 = 0;
    image_ptr                 = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            out_image,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            out_region,
            &row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping dest image buffer for reading." << "\n";
        return err;
    }

    // Copies image data from the DMA buffer to the host.
    for (uint32_t i = 0; i < out_desc.image_height; ++i)
    {
        std::memcpy(
                out_image_info.pixels.data() + i * out_desc.image_width * 4,
                image_ptr                    + i * row_pitch,
                out_desc.image_width * 4
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_image, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image." << "\n";
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
    if (ioctl(out_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    save_rgba_image_data(out_image_filename, out_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_image);
    clReleaseMemObject(out_image);

    return EXIT_SUCCESS;
}