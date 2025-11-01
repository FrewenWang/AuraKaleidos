//--------------------------------------------------------------------------------------
// File: compressed_tp10_to_rgba.cpp
// Desc: This program converts a TP10 image to RGBA8888
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
    compressed_tp10_to_rgba(__read_only  image2d_t input_tp10,
                            __write_only image2d_t out_rgba,
                              sampler_t sampler)
    {
        int2   coord;
        float4 yuv;
        float4 rgba;

        coord.x = get_global_id(0);
        coord.y = get_global_id(1);

        yuv    = read_imagef(input_tp10,  sampler, coord);
        yuv.y  = (yuv.y - 0.5f) * 0.872f;
        yuv.z  = (yuv.z - 0.5f) * 1.23f;
        rgba.x = yuv.x + (1.140f * yuv.z);
        rgba.y = yuv.x - (0.395f * yuv.y) - (0.581f * yuv.z);
        rgba.z = yuv.x + (2.032f * yuv.y);
        rgba.w = 1.0f;
        write_imagef(out_rgba, coord, rgba);
    }


    __kernel void tp10_y_compress(__read_only  image2d_t src_image,
                                  __write_only image2d_t dest_image_y_plane,
                                           sampler_t sampler)
    {
        const int    wid_x       = get_global_id(0);
        const int    wid_y       = get_global_id(1);
        const int2   read_coord  = (int2)(3 * wid_x, wid_y);
        const int2   write_coord = (int2)(3 * wid_x, wid_y);
        const float4 pixels_in[] = {
            read_imagef(src_image, sampler, read_coord),
            read_imagef(src_image, sampler, read_coord + (int2)(1, 0)),
            read_imagef(src_image, sampler, read_coord + (int2)(2, 0)),
            };
        float        y_pixels_out[] = {pixels_in[0].s0, pixels_in[1].s0, pixels_in[2].s0};
        qcom_write_imagefv_3x1_n10t00(dest_image_y_plane, write_coord, y_pixels_out);
    }

    __kernel void tp10_uv_compress(__read_only  image2d_t src_image,
                                   __write_only image2d_t dest_image_uv_plane,
                                            sampler_t sampler)
    {
        const int    wid_x       = get_global_id(0);
        const int    wid_y       = get_global_id(1);
        const int2   read_coord  = (int2)(6 * wid_x, 2 * wid_y);
        const int2   write_coord = (int2)(3 * wid_x, wid_y);
        const float4 pixels_in[] = {
            read_imagef(src_image, sampler, read_coord),
            read_imagef(src_image, sampler, read_coord + (int2)(2, 0)),
            read_imagef(src_image, sampler, read_coord + (int2)(4, 0)),
            };
        float2       uv_pixels_out[] = {
            {pixels_in[0].s1, pixels_in[0].s2},
            {pixels_in[1].s1, pixels_in[1].s2},
            {pixels_in[2].s1, pixels_in[2].s2},
            };
        qcom_write_imagefv_3x1_n10t01(dest_image_uv_plane, write_coord, uv_pixels_out);
    }
)";

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input image> <output image> \n"
                     "Input image file data should be in format CL_QCOM_TP10 / CL_QCOM_UNORM_INT10.\n"
                     "Demonstrates conversions from compressed TP10 to RGBA8888.\n"
                     "Note that first it converts the input TP10 to compressed TP10, then to RGBA.\n";
        return EXIT_FAILURE;
    }

    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper          wrapper;
    cl_program          program                 = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel           compressed_tp10_to_rgba = wrapper.make_kernel("compressed_tp10_to_rgba", program);
    cl_kernel           tp10_y_compress         = wrapper.make_kernel("tp10_y_compress", program);
    cl_kernel           tp10_uv_compress        = wrapper.make_kernel("tp10_uv_compress", program);
    cl_context          context                 = wrapper.get_context();
    tp10_image_t        src_tp10_image_info     = load_tp10_image_data(src_image_filename);
    struct dma_buf_sync buf_sync                = {};
    cl_event            unmap_event             = nullptr;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_other_image"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for TP10 image format is not supported.\n";
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
     * Step 1: Create suitable DMA buffer-backed CL images. Note that planar formats (like TP10) must be read only,
     * but you can write to child images derived from the planes. (See step 2 for deriving child images.)
     */

    cl_image_format src_tp10_format;
    src_tp10_format.image_channel_order     = CL_QCOM_TP10;
    src_tp10_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc src_tp10_desc;
    std::memset(&src_tp10_desc, 0, sizeof(src_tp10_desc));
    src_tp10_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_tp10_desc.image_width  = src_tp10_image_info.y_width;
    src_tp10_desc.image_height = src_tp10_image_info.y_height;

    cl_int err = 0;
    cl_mem_dmabuf_host_ptr src_tp10_mem = wrapper.make_buffer_for_yuv_image(src_tp10_format, src_tp10_desc);
    cl_mem src_tp10_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &src_tp10_format,
            &src_tp10_desc,
            &src_tp10_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image." << "\n";
        return err;
    }

    cl_image_format ubwc_tp10_format;
    ubwc_tp10_format.image_channel_order     = CL_QCOM_COMPRESSED_TP10;
    ubwc_tp10_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc ubwc_tp10_desc;
    std::memset(&ubwc_tp10_desc, 0, sizeof(ubwc_tp10_desc));
    ubwc_tp10_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    ubwc_tp10_desc.image_width  = src_tp10_desc.image_width;
    ubwc_tp10_desc.image_height = src_tp10_desc.image_height;

    cl_mem_dmabuf_host_ptr ubwc_tp10_mem = wrapper.make_buffer_for_compressed_image(ubwc_tp10_format, ubwc_tp10_desc);
    cl_mem ubwc_tp10_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &ubwc_tp10_format,
            &ubwc_tp10_desc,
            &ubwc_tp10_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for UBWC image." << "\n";
        return err;
    }

    cl_image_format out_rgba_format;
    out_rgba_format.image_channel_order     = CL_RGBA;
    out_rgba_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_rgba_desc;
    std::memset(&out_rgba_desc, 0, sizeof(out_rgba_desc));
    out_rgba_desc.image_type             = CL_MEM_OBJECT_IMAGE2D;
    out_rgba_desc.image_width            = src_tp10_image_info.y_width;
    out_rgba_desc.image_height           = src_tp10_image_info.y_height;
    const size_t img_row_pitch           = wrapper.get_image_row_pitch(out_rgba_format, out_rgba_desc);
    out_rgba_desc.image_row_pitch        = img_row_pitch;
    cl_mem_dmabuf_host_ptr out_rgba_mem = wrapper.make_buffer_for_nonplanar_image(out_rgba_format, out_rgba_desc);
    cl_mem out_rgba_image = clCreateImage(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_rgba_format,
            &out_rgba_desc,
            &out_rgba_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output RGBA image." << "\n";
        return err;
    }

    /*
     * Step 2: Separate planar image into its component planes.
     */

    cl_image_format src_y_plane_format;
    src_y_plane_format.image_channel_order     = CL_QCOM_TP10_Y;
    src_y_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc src_y_plane_desc;
    std::memset(&src_y_plane_desc, 0, sizeof(src_y_plane_desc));
    src_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_y_plane_desc.image_width  = src_tp10_desc.image_width;
    src_y_plane_desc.image_height = src_tp10_desc.image_height;
    src_y_plane_desc.mem_object   = src_tp10_image;

    cl_mem src_y_plane = clCreateImage(
            context,
            CL_MEM_READ_ONLY,
            &src_y_plane_format,
            &src_y_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image y plane." << "\n";
        return err;
    }

    cl_image_format src_uv_plane_format;
    src_uv_plane_format.image_channel_order     = CL_QCOM_TP10_UV;
    src_uv_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc src_uv_plane_desc;
    std::memset(&src_uv_plane_desc, 0, sizeof(src_uv_plane_desc));
    src_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    src_uv_plane_desc.image_width  = src_tp10_image_info.y_width;
    src_uv_plane_desc.image_height = src_tp10_image_info.y_height;
    src_uv_plane_desc.mem_object   = src_tp10_image;

    cl_mem src_uv_plane = clCreateImage(
            context,
            CL_MEM_READ_ONLY,
            &src_uv_plane_format,
            &src_uv_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image uv plane." << "\n";
        return err;
    }

    cl_image_format ubwc_y_plane_format;
    ubwc_y_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_TP10_Y;
    ubwc_y_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc ubwc_y_plane_desc;
    std::memset(&ubwc_y_plane_desc, 0, sizeof(ubwc_y_plane_desc));
    ubwc_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    ubwc_y_plane_desc.image_width  = ubwc_tp10_desc.image_width;
    ubwc_y_plane_desc.image_height = ubwc_tp10_desc.image_height;
    ubwc_y_plane_desc.mem_object   = ubwc_tp10_image;

    cl_mem ubwc_y_plane = clCreateImage(
            context,
            CL_MEM_READ_WRITE,
            &ubwc_y_plane_format,
            &ubwc_y_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for UBWC image y plane." << "\n";
        return err;
    }

    cl_image_format ubwc_uv_plane_format;
    ubwc_uv_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_TP10_UV;
    ubwc_uv_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc ubwc_uv_plane_desc;
    std::memset(&ubwc_uv_plane_desc, 0, sizeof(ubwc_uv_plane_desc));
    ubwc_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    ubwc_uv_plane_desc.image_width  = ubwc_tp10_desc.image_width;
    ubwc_uv_plane_desc.image_height = ubwc_tp10_desc.image_height;
    ubwc_uv_plane_desc.mem_object   = ubwc_tp10_image;

    cl_mem ubwc_uv_plane = clCreateImage(
            context,
            CL_MEM_READ_WRITE,
            &ubwc_uv_plane_format,
            &ubwc_uv_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for UBWC image uv plane." << "\n";
        return err;
    }

    /*
     * Step 3: Copy data to input image planes. Note that for linear TP10 images you must observe row alignment
     * restrictions. (You may also write to the DMA buffer directly if you prefer, however using clEnqueueMapImage for
     * a child planar image will return the correct host pointer for the desired plane.)
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if (ioctl(src_tp10_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    cl_command_queue command_queue  = wrapper.get_command_queue();
    const size_t     origin[]       = {0, 0, 0};
    const size_t     src_y_region[] = {src_y_plane_desc.image_width, src_y_plane_desc.image_height, 1};
    size_t           row_pitch      = 0;
    unsigned char   *image_ptr      = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            src_y_plane,
            CL_BLOCKING,
            CL_MAP_WRITE,
            origin,
            src_y_region,
            &row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping source image y-plane buffer for writing." << "\n";
        return err;
    }
    // Copies image data to the DMA buffer from the host.
    for (uint32_t i = 0; i < src_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                image_ptr                          + i * row_pitch,
                src_tp10_image_info.y_plane.data() + i * src_y_plane_desc.image_width * 4 / 3,
                src_y_plane_desc.image_width * 4 / 3
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, src_y_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer." << "\n";
        return err;
    }

    // Note the discrepancy between the child plane image descriptor and the size required by clEnqueueMapImage.
    const size_t src_uv_region[] = {src_uv_plane_desc.image_width / 2, src_uv_plane_desc.image_height / 2, 1};
    row_pitch                    = 0;
    image_ptr = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            src_uv_plane,
            CL_BLOCKING,
            CL_MAP_WRITE,
            origin,
            src_uv_region,
            &row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping source image uv-plane buffer for writing." << "\n";
        return err;
    }

    // Copies image data to the DMA buffer from the host.
    for (uint32_t i = 0; i < src_uv_plane_desc.image_height / 2; ++i)
    {
        std::memcpy(
                image_ptr                           + i * row_pitch,
                src_tp10_image_info.uv_plane.data() + i * src_uv_plane_desc.image_width * 4 / 3,
                src_uv_plane_desc.image_width * 4 / 3
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, src_uv_plane, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image uv-plane data buffer." << "\n";
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
    if (ioctl(src_tp10_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    /*
     * Step 4: Set up all the kernels
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
        return err;
    }

    // Copy TP10 Y Plane
    err = clSetKernelArg(tp10_y_compress, 0, sizeof(src_tp10_image), &src_tp10_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_y_compress, 1, sizeof(ubwc_y_plane), &ubwc_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for tp10_y_compress kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_y_compress, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for tp10_y_compress kernel." << "\n";
        return err;
    }

    // Copy TP10 UV Plane
    err = clSetKernelArg(tp10_uv_compress, 0, sizeof(src_tp10_image), &src_tp10_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_uv_compress kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_uv_compress, 1, sizeof(ubwc_uv_plane), &ubwc_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for tp10_uv_compress kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_uv_compress, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for tp10_uv_compress kernel." << "\n";
        return err;
    }

    // Naive TP10 UBWC to RGBA
    err = clSetKernelArg(compressed_tp10_to_rgba, 0, sizeof(ubwc_tp10_image), &ubwc_tp10_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for compressed_tp10_to_rgba kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compressed_tp10_to_rgba, 1, sizeof(out_rgba_image), &out_rgba_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for compressed_tp10_to_rgba kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compressed_tp10_to_rgba, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for compressed_tp10_to_rgba kernel." << "\n";
        return err;
    }

    /*
     * Step 5: Run the kernels
     */

    size_t work_size[2];

    work_size[0] = src_tp10_desc.image_width / 3;
    work_size[1] = src_tp10_desc.image_height;
    err = clEnqueueNDRangeKernel(command_queue, tp10_y_compress, 2, nullptr, work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for tp10_y_compress kernel." << "\n";
        return err;
    }

    work_size[0] = src_tp10_desc.image_width / 6;
    work_size[1] = src_tp10_desc.image_height / 2;
    err = clEnqueueNDRangeKernel(command_queue, tp10_uv_compress, 2, nullptr, work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for tp10_uv_compress kernel." << "\n";
        return err;
    }

    err = clFlush(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while flushing the command queue.\n";
        return err;
    }

    work_size[0] = out_rgba_desc.image_width;
    work_size[1] = out_rgba_desc.image_height;
    err = clEnqueueNDRangeKernel(command_queue, compressed_tp10_to_rgba, 2, nullptr, work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for compressed_tp10_to_rgba kernel." << "\n";
        return err;
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

    rgba_image_t out_rgba_image_info;
    out_rgba_image_info.width  = out_rgba_desc.image_width;
    out_rgba_image_info.height = out_rgba_desc.image_height;
    out_rgba_image_info.pixels.resize(out_rgba_desc.image_width * out_rgba_image_info.height * 4);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if (ioctl(out_rgba_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    const size_t out_rgb_region[] = {out_rgba_desc.image_width, out_rgba_desc.image_height, 1};
    row_pitch                     = 0;
    image_ptr = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            out_rgba_image,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            out_rgb_region,
            &row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping dest image y-plane buffer for reading." << "\n";
        return err;
    }
    // Copies image data from the dmabuf buffer to the host
    for (uint32_t i = 0; i < out_rgba_desc.image_height; ++i)
    {
        std::memcpy(
                out_rgba_image_info.pixels.data() + i * out_rgba_desc.image_width * 4,
                image_ptr                         + i * row_pitch,
                out_rgba_desc.image_width * 4
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_rgba_image, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image out_rgba_image buffer." << "\n";
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
    if (ioctl(out_rgba_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    save_rgba_image_data(out_image_filename, out_rgba_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_tp10_image);
    clReleaseMemObject(ubwc_y_plane);
    clReleaseMemObject(ubwc_uv_plane);
    clReleaseMemObject(ubwc_tp10_image);
    clReleaseMemObject(out_rgba_image);

    return EXIT_SUCCESS;
}