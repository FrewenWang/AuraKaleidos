//--------------------------------------------------------------------------------------
// File: rgba_to_p010.cpp
// Desc: This program converts RGBA8888 to P010
//
// Author:      QUALCOMM
//
//               Copyright (c) 2018, 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <array>
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
    __kernel void
    rgba_to_p010(__read_only  image2d_t input_rgba,
                 __write_only image2d_t out_p010_y,
                 __write_only image2d_t out_p010_uv,
                              sampler_t sampler)
    {
        int2   coord;
        float4 ys[4];
        float4 uv;
        float4 rgba_avg;
        float4 rgbas[4];

        coord.x = 2 * get_global_id(0);
        coord.y = 2 * get_global_id(1);

        rgbas[0] = read_imagef(input_rgba, sampler, coord);
        rgbas[1] = read_imagef(input_rgba, sampler, coord + (int2)(0, 1));
        rgbas[2] = read_imagef(input_rgba, sampler, coord + (int2)(1, 0));
        rgbas[3] = read_imagef(input_rgba, sampler, coord + (int2)(1, 1));

        rgba_avg = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = 0; i < 4; ++i)
        {
            ys[i].x   = rgbas[i].x * 0.299f + rgbas[i].y * 0.587f + rgbas[i].z * 0.114f;
            rgba_avg += rgbas[i];
        }
        rgba_avg /= 4.0f;
        uv.x      = rgba_avg.x * -0.169f + rgba_avg.y * -0.331f + rgba_avg.z * 0.5f    + 0.5f;
        uv.y      = rgba_avg.x *  0.5f   + rgba_avg.y * -0.419f + rgba_avg.z * -0.081f + 0.5f;

        write_imagef(out_p010_y,  coord, ys[0]);
        write_imagef(out_p010_y,  coord + (int2)(0, 1), ys[1]);
        write_imagef(out_p010_y,  coord + (int2)(1, 0), ys[2]);
        write_imagef(out_p010_y,  coord + (int2)(1, 1), ys[3]);
        write_imagef(out_p010_uv, coord / 2, uv);
    }
)";

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <img data file> <out img data file> \n"
                  << "Input image file data should be in format CL_RGBA / CL_UNORM_INT8\n"
                  << "Demonstrates conversions from RGBA8888 to P010.\n"
                  << "Since P010 is a chroma subsampled format, the conversion is lossy.\n";
        return 0;
    }

    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper          wrapper;
    cl_program          program             = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel           rgba_to_p010_kernel = wrapper.make_kernel("rgba_to_p010", program);
    cl_context          context             = wrapper.get_context();
    rgba_image_t        src_rgba_image_info = load_rgba_image_data(src_image_filename);
    struct dma_buf_sync buf_sync            = {};
    cl_event            unmap_event         = nullptr;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_other_image"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for P010 image format is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    if (!wrapper.check_extension_support("cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for dmabuf-backed images is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    if (!wrapper.check_extension_support("cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for dmabuf-backed images is not supported.\n";
        std::exit(EXIT_FAILURE);
    }

    /*
     * Step 1: Create suitable DMA buffer-backed CL images. Note that planar formats (like P010) must be read only,
     * but you can write to child images derived from the planes. (See step 2 for deriving child images.)
     */

    cl_image_format out_p010_format;
    out_p010_format.image_channel_order     = CL_QCOM_P010;
    out_p010_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_p010_desc;
    std::memset(&out_p010_desc, 0, sizeof(out_p010_desc));
    out_p010_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_p010_desc.image_width  = src_rgba_image_info.width;
    out_p010_desc.image_height = src_rgba_image_info.height;

    cl_int err = 0;
    cl_mem_dmabuf_host_ptr out_p010_mem = wrapper.make_buffer_for_yuv_image(out_p010_format, out_p010_desc);
    cl_mem out_p010_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_p010_format,
            &out_p010_desc,
            &out_p010_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image." << "\n";
        std::exit(err);
    }

    cl_image_format src_rgba_format;
    src_rgba_format.image_channel_order     = CL_RGBA;
    src_rgba_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_rgba_desc;
    std::memset(&src_rgba_desc, 0, sizeof(src_rgba_desc));
    src_rgba_desc.image_type             = CL_MEM_OBJECT_IMAGE2D;
    src_rgba_desc.image_width            = src_rgba_image_info.width;
    src_rgba_desc.image_height           = src_rgba_image_info.height;
    const size_t img_row_pitch           = wrapper.get_image_row_pitch(src_rgba_format, src_rgba_desc);
    src_rgba_desc.image_row_pitch        = img_row_pitch;
    cl_mem_dmabuf_host_ptr src_rgba_mem = wrapper.make_buffer_for_nonplanar_image(src_rgba_format, src_rgba_desc);
    cl_mem src_rgba_image = clCreateImage(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &src_rgba_format,
            &src_rgba_desc,
            &src_rgba_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source RGBA image." << "\n";
        std::exit(err);
    }

    /*
     * Step 2: Separate planar P010 images into their component planes.
     */
    cl_image_format out_y_plane_format;
    out_y_plane_format.image_channel_order     = CL_QCOM_P010_Y;
    out_y_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_y_plane_desc;
    std::memset(&out_y_plane_desc, 0, sizeof(out_y_plane_desc));
    out_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_y_plane_desc.image_width  = out_p010_desc.image_width;
    out_y_plane_desc.image_height = out_p010_desc.image_height;
    out_y_plane_desc.mem_object   = out_p010_image;

    cl_mem out_y_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &out_y_plane_format,
            &out_y_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image y plane." << "\n";
        std::exit(err);
    }

    cl_image_format out_uv_plane_format;
    out_uv_plane_format.image_channel_order     = CL_QCOM_P010_UV;
    out_uv_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_uv_plane_desc;
    std::memset(&out_uv_plane_desc, 0, sizeof(out_uv_plane_desc));
    out_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    out_uv_plane_desc.image_width  = src_rgba_image_info.width;
    out_uv_plane_desc.image_height = src_rgba_image_info.height;
    out_uv_plane_desc.mem_object   = out_p010_image;

    cl_mem out_uv_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &out_uv_plane_format,
            &out_uv_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image uv plane." << "\n";
        std::exit(err);
    }

    /*
     * Step 3: Copy data to the input image.
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if (ioctl(src_rgba_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    cl_command_queue command_queue = wrapper.get_command_queue();
    const size_t     origin[]      = {0, 0, 0};
    const size_t     region[]      = {src_rgba_image_info.width, src_rgba_image_info.height, 1};
    size_t           row_pitch     = 0;
    unsigned char   *image_ptr     = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            src_rgba_image,
            CL_BLOCKING,
            CL_MAP_WRITE,
            origin,
            region,
            &row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping source image for writing." << "\n";
        std::exit(err);
    }

    // Copies image data to the DMA buffer from the host.
    for (uint32_t i = 0; i < out_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                image_ptr                          + i * row_pitch,
                src_rgba_image_info.pixels.data()  + i * src_rgba_image_info.width * 4,
                src_rgba_image_info.width * 4
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, src_rgba_image, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image." << "\n";
        std::exit(err);
    }

    err = clWaitForEvents(1, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while waiting to unmap.\n";
        return err;
    }

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if (ioctl(src_rgba_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    /*
     * Step 4: Set up other kernel arguments
     */

    cl_sampler sampler = clCreateSampler(
            context,
            CL_FALSE,
            CL_ADDRESS_CLAMP_TO_EDGE,
            CL_FILTER_NEAREST,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateSampler." << "\n";
        std::exit(err);
    }

    /*
     * Step 5: Run the kernel for both y- and uv-planes
     */

    err = clSetKernelArg(rgba_to_p010_kernel, 0, sizeof(src_rgba_image), &src_rgba_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for rgba_to_p010_kernel." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(rgba_to_p010_kernel, 1, sizeof(out_y_plane), &out_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for rgba_to_p010_kernel." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(rgba_to_p010_kernel, 2, sizeof(out_uv_plane), &out_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for rgba_to_p010_kernel." << "\n";
        std::exit(err);
    }

    err = clSetKernelArg(rgba_to_p010_kernel, 3, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 3 for rgba_to_p010_kernel." << "\n";
        std::exit(err);
    }

    const size_t work_size[] = {src_rgba_desc.image_width / 2, src_rgba_desc.image_height / 2};
    err = clEnqueueNDRangeKernel(
            command_queue,
            rgba_to_p010_kernel,
            2,
            nullptr,
            work_size,
            nullptr,
            0,
            nullptr,
            nullptr
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for rgba_to_p010_kernel." << "\n";
        std::exit(err);
    }

    err = clFinish(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while finishing the command queue.\n";
        return err;
    }

    /*
     * Step 6: Copy the data out of the DMA buffer for each plane.
     */

    p010_image_t out_p010_image_info;
    out_p010_image_info.y_width  = src_rgba_desc.image_width;
    out_p010_image_info.y_height = src_rgba_desc.image_height;
    out_p010_image_info.y_plane.resize(out_p010_image_info.y_width * out_p010_image_info.y_height * 2);
    out_p010_image_info.uv_plane.resize(out_p010_image_info.y_width * out_p010_image_info.y_height);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if (ioctl(out_p010_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    const size_t out_y_region[] = {out_p010_image_info.y_width, out_p010_image_info.y_height, 1};
    row_pitch                   = 0;
    image_ptr = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            out_y_plane,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            out_y_region,
            &row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping dest image y-plane for reading." << "\n";
        std::exit(err);
    }

    // Copies image data from the DMA buffer to the host.
    for (uint32_t i = 0; i < out_p010_image_info.y_height; ++i)
    {
        std::memcpy(
                out_p010_image_info.y_plane.data() + i * out_p010_image_info.y_width * 2,
                image_ptr                          + i * row_pitch,
                out_p010_image_info.y_width * 2
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_y_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image y-plane." << "\n";
        std::exit(err);
    }

    const size_t out_uv_region[] = {out_p010_image_info.y_width / 2, out_p010_image_info.y_height / 2, 1};
    row_pitch                    = 0;
    image_ptr = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            out_uv_plane,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            out_uv_region,
            &row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping dest image uv-plane for reading." << "\n";
        std::exit(err);
    }

    // Copies image data from the DMA buffer to the host.
    for (uint32_t i = 0; i < out_p010_image_info.y_height / 2; ++i)
    {
        std::memcpy(
                out_p010_image_info.uv_plane.data() + i * out_p010_image_info.y_width * 2,
                image_ptr                           + i * row_pitch,
                out_p010_image_info.y_width * 2
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_uv_plane, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image uv-plane." << "\n";
        std::exit(err);
    }

    err = clWaitForEvents(1, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while waiting to unmap.\n";
        return err;
    }

    clReleaseEvent(unmap_event);

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    if (ioctl(out_p010_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    save_p010_image_data(out_image_filename, out_p010_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(out_uv_plane);
    clReleaseMemObject(out_y_plane);
    clReleaseMemObject(out_p010_image);
    clReleaseMemObject(src_rgba_image);

    return EXIT_SUCCESS;
}