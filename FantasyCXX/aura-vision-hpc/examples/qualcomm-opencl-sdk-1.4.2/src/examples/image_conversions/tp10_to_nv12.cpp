//--------------------------------------------------------------------------------------
// File: tp10_to_nv12.cpp
// Desc: This program converts a TP10 image to NV12 using vectorized read/writes.
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
    // The image format determines how the 4th argument to vector read operations is interpreted.
    // For YUV images:
    __constant int YUV_Y_PLANE = 0;
    __constant int YUV_U_PLANE = 1;
    __constant int YUV_V_PLANE = 2;
    
    // For Y-only images:
    static const int Y_Y_PLANE = 0;

    // For UV-only images:
    __constant int UV_U_PLANE = 0;
    __constant int UV_V_PLANE = 1;
    
    // Reads 2x2 from a UV-only image and writes 2x1 to a UV-only image
    __kernel void read_uv_2x2_write_uv_2x1(__read_only  image2d_t src_image,
                                           __write_only image2d_t dest_image_y_plane,
                                                        sampler_t sampler)
    {
        const int    wid_x              = get_global_id(0);
        const int    wid_y              = get_global_id(1);
        const float2 read_coord         = (float2)(2 * wid_x, 2 * wid_y) + 0.5;
        const int2   write_coord        = (int2)(2 * wid_x, 2 * wid_y);
        const float4 u_pixels_in        = qcom_read_imagef_2x2(src_image, sampler, read_coord, UV_U_PLANE);
        const float4 v_pixels_in        = qcom_read_imagef_2x2(src_image, sampler, read_coord, UV_V_PLANE);
        float2       uv_pixels_out[2][2] = {
           {{u_pixels_in.s3, v_pixels_in.s3}, {u_pixels_in.s2, v_pixels_in.s2}},
           {{u_pixels_in.s0, v_pixels_in.s0}, {u_pixels_in.s1, v_pixels_in.s1}},
           };
        qcom_write_imagefv_2x1_n8n01(dest_image_y_plane, write_coord,                uv_pixels_out[0]);
        qcom_write_imagefv_2x1_n8n01(dest_image_y_plane, write_coord + (int2)(0, 1), uv_pixels_out[1]);
    }

    __kernel void read_y_2x2x4_write_y_4x1(__read_only  image2d_t src_image,
                                           __write_only image2d_t dest_image_y_plane,
                                                        sampler_t sampler)
    {
        const int    wid_x              = get_global_id(0);
        const int    wid_y              = get_global_id(1);
        const float2 read_coord         = (float2)(4 * wid_x, 4 * wid_y) + 0.5;
        const int2   write_coord        = (int2)(4 * wid_x, 4 * wid_y);
        const float4 y_pixels_in[]      = {
            qcom_read_imagef_2x2(src_image, sampler, read_coord,                    YUV_Y_PLANE),
            qcom_read_imagef_2x2(src_image, sampler, read_coord + (float2)(2., 0.), YUV_Y_PLANE),
            qcom_read_imagef_2x2(src_image, sampler, read_coord + (float2)(0., 2.), YUV_Y_PLANE),
            qcom_read_imagef_2x2(src_image, sampler, read_coord + (float2)(2., 2.), YUV_Y_PLANE),
            };
        float        y_pixels_out[4][4] = {
           {y_pixels_in[0].s3, y_pixels_in[0].s2, y_pixels_in[1].s3, y_pixels_in[1].s2},
           {y_pixels_in[0].s0, y_pixels_in[0].s1, y_pixels_in[1].s0, y_pixels_in[1].s1},
           {y_pixels_in[2].s3, y_pixels_in[2].s2, y_pixels_in[3].s3, y_pixels_in[3].s2},
           {y_pixels_in[2].s0, y_pixels_in[2].s1, y_pixels_in[3].s0, y_pixels_in[3].s1},
           };
        qcom_write_imagefv_4x1_n8n00(dest_image_y_plane, write_coord,                y_pixels_out[0]);
        qcom_write_imagefv_4x1_n8n00(dest_image_y_plane, write_coord + (int2)(0, 1), y_pixels_out[1]);
        qcom_write_imagefv_4x1_n8n00(dest_image_y_plane, write_coord + (int2)(0, 2), y_pixels_out[2]);
        qcom_write_imagefv_4x1_n8n00(dest_image_y_plane, write_coord + (int2)(0, 3), y_pixels_out[3]);
    }
)";

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input image> <output image> \n"
                                             "Input image file data should be in format CL_QCOM_TP10 / CL_QCOM_UNORM_INT10.\n"
                                             "Demonstrates conversions from CL_QCOM_TP10 to NV12 using vectorized read/writes.\n";
        return EXIT_FAILURE;
    }

    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper          wrapper;
    cl_program          program               = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel           tp10_to_nv12_y        = wrapper.make_kernel("read_y_2x2x4_write_y_4x1", program);
    cl_kernel           tp10_to_nv12_uv       = wrapper.make_kernel("read_uv_2x2_write_uv_2x1", program);
    cl_context          context               = wrapper.get_context();
    cl_command_queue    command_queue         = wrapper.get_command_queue();
    tp10_image_t        input_tp10_image_file = load_tp10_image_data(src_image_filename);
    cl_sampler          sampler               = nullptr;
    cl_int              err                   = 0;
    struct dma_buf_sync buf_sync              = {};
    cl_event            unmap_event           = nullptr;

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
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for dmabuf-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for dmabuf-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create suitable DMA buffer-backed CL images. Note that planar formats (like TP10) must be read only,
     * but you can write to child images derived from the planes. (See step 2 for deriving child images.)
     */
    cl_image_format input_tp10_format;
    input_tp10_format.image_channel_order     = CL_QCOM_TP10;
    input_tp10_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc input_tp10_desc = {0};
    input_tp10_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    input_tp10_desc.image_width  = input_tp10_image_file.y_width;
    input_tp10_desc.image_height = input_tp10_image_file.y_height;

    cl_mem_dmabuf_host_ptr input_tp10_mem = wrapper.make_buffer_for_yuv_image(input_tp10_format, input_tp10_desc);
    cl_mem input_tp10_image = clCreateImage(
        context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        &input_tp10_format,
        &input_tp10_desc,
        &input_tp10_mem,
        &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image." << "\n";
        return err;
    }

    cl_image_format output_nv12_format;
    output_nv12_format.image_channel_order     = CL_QCOM_NV12;
    output_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc output_nv12_desc = {0};
    output_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    output_nv12_desc.image_width  = input_tp10_image_file.y_width;
    output_nv12_desc.image_height = input_tp10_image_file.y_height;

    cl_mem_dmabuf_host_ptr output_nv12_mem = wrapper.make_buffer_for_yuv_image(output_nv12_format, output_nv12_desc);
    cl_mem output_nv12_image;
    output_nv12_image = clCreateImage(
        context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
        &output_nv12_format,
        &output_nv12_desc,
        &output_nv12_mem,
        &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for NV12 image." << "\n";
        return err;
    }

    /*
     * Step 2: Separate planar TP10 images into their component planes.
     */
    cl_image_format input_tp10_y_format;
    input_tp10_y_format.image_channel_order     = CL_QCOM_TP10_Y;
    input_tp10_y_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc input_tp10_y_desc = {0};
    input_tp10_y_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    input_tp10_y_desc.image_width  = input_tp10_desc.image_width;
    input_tp10_y_desc.image_height = input_tp10_desc.image_height;
    input_tp10_y_desc.mem_object   = input_tp10_image;

    cl_mem input_tp10_y = clCreateImage(
        context,
        CL_MEM_READ_WRITE,
        &input_tp10_y_format,
        &input_tp10_y_desc,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image y plane." << "\n";
        return err;
    }

    cl_image_format input_tp10_uv_format;
    input_tp10_uv_format.image_channel_order     = CL_QCOM_TP10_UV;
    input_tp10_uv_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc input_tp10_uv_desc = {0};
    input_tp10_uv_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    input_tp10_uv_desc.image_width  = input_tp10_desc.image_width;
    input_tp10_uv_desc.image_height = input_tp10_desc.image_height;
    input_tp10_uv_desc.mem_object   = input_tp10_image;

    cl_mem input_tp10_uv = clCreateImage(
        context,
        CL_MEM_READ_WRITE,
        &input_tp10_uv_format,
        &input_tp10_uv_desc,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image uv plane." << "\n";
        return err;
    }

    /*
     * Step 3: Copy data to input image planes. Note that for linear TP10 images you must observe row alignment
     * restrictions. (You may also write to the DMA buffer directly if you prefer, however using clEnqueueMapImage for
     * a child planar image will return the correct host pointer for the desired plane.)
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if (ioctl(input_tp10_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    const size_t     origin[]       = {0, 0, 0};
    const size_t     src_y_region[] = {input_tp10_y_desc.image_width, input_tp10_y_desc.image_height, 1};
    size_t           row_pitch      = 0;
    unsigned char   *image_ptr      = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
        command_queue,
        input_tp10_y,
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
    for (uint32_t i = 0; i < input_tp10_y_desc.image_height; ++i)
    {
        std::memcpy(
            image_ptr                            + i * row_pitch,
            input_tp10_image_file.y_plane.data() + i * input_tp10_y_desc.image_width * 4 / 3,
            input_tp10_y_desc.image_width * 4 / 3
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, input_tp10_y, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer." << "\n";
        return err;
    }

    // Note the discrepancy between the child plane image descriptor and the size required by clEnqueueMapImage.
    const size_t src_uv_region[] = {input_tp10_uv_desc.image_width / 2, input_tp10_uv_desc.image_height / 2, 1};
    row_pitch                    = 0;
    image_ptr = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
        command_queue,
        input_tp10_uv,
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
    for (uint32_t i = 0; i < input_tp10_uv_desc.image_height / 2; ++i)
    {
        std::memcpy(
            image_ptr                             + i * row_pitch,
            input_tp10_image_file.uv_plane.data() + i * input_tp10_uv_desc.image_width * 4 / 3,
            input_tp10_uv_desc.image_width * 4 / 3
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, input_tp10_uv, image_ptr, 0, nullptr, &unmap_event);
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
    if (ioctl(input_tp10_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    /*
     * Step 4: Separate planar NV12 images into their component planes.
     */
    cl_image_format output_nv12_y_format;
    output_nv12_y_format.image_channel_order     = CL_QCOM_NV12_Y;
    output_nv12_y_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc output_nv12_y_desc = {0};
    output_nv12_y_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    output_nv12_y_desc.image_width  = input_tp10_desc.image_width;
    output_nv12_y_desc.image_height = input_tp10_desc.image_height;
    output_nv12_y_desc.mem_object   = output_nv12_image;

    cl_mem output_nv12_y = clCreateImage(
        context,
        CL_MEM_READ_WRITE,
        &output_nv12_y_format,
        &output_nv12_y_desc,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image y plane." << "\n";
        return err;
    }

    cl_image_format output_nv12_uv_format;
    output_nv12_uv_format.image_channel_order     = CL_QCOM_NV12_UV;
    output_nv12_uv_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc output_nv12_uv_desc = {0};
    output_nv12_uv_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    output_nv12_uv_desc.image_width  = input_tp10_desc.image_width;
    output_nv12_uv_desc.image_height = input_tp10_desc.image_height;
    output_nv12_uv_desc.mem_object   = output_nv12_image;

    cl_mem output_nv12_uv = clCreateImage(
        context,
        CL_MEM_READ_WRITE,
        &output_nv12_uv_format,
        &output_nv12_uv_desc,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image uv plane." << "\n";
        return err;
    }

    /*
     * Step 5: Set kernel arguments for both of our kernels.
     */
    sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateSampler." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_nv12_y, 0, sizeof(input_tp10_y), &input_tp10_y);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_nv12_y, 1, sizeof(output_nv12_y), &output_nv12_y);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_nv12_y, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_nv12_uv, 0, sizeof(input_tp10_uv), &input_tp10_uv);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_nv12_uv, 1, sizeof(output_nv12_uv), &output_nv12_uv);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_nv12_uv, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." << "\n";
        return err;
    }

    /*
     * Step 6: Run Kernels
     */
    size_t global_work_size[] = {0, 0};

    global_work_size[0] = (input_tp10_desc.image_width  + 3) / 4;
    global_work_size[1] = (input_tp10_desc.image_height + 3) / 4;
    err = clEnqueueNDRangeKernel(command_queue, tp10_to_nv12_y, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel." << "\n";
        return err;
    }

    global_work_size[0] = (input_tp10_desc.image_width + 3) / 4;
    global_work_size[1] = (input_tp10_desc.image_height + 3) / 4;
    err = clEnqueueNDRangeKernel(command_queue, tp10_to_nv12_uv, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel." << "\n";
        return err;
    }

    /*
     * Step 7: Allow kernels to finish
     */
    err = clFinish(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clFinish." << "\n";
        return err;
    }

    /*
     * Step 8: Copy the data out of the DMA buffer for each plane.
     */

    nv12_image_t out_nv12_image_info;
    out_nv12_image_info.y_width  = output_nv12_desc.image_width;
    out_nv12_image_info.y_height = output_nv12_desc.image_height;
    out_nv12_image_info.y_plane.resize(out_nv12_image_info.y_width * out_nv12_image_info.y_height);
    out_nv12_image_info.uv_plane.resize(out_nv12_image_info.y_width * out_nv12_image_info.y_height / 2);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if (ioctl(output_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    const size_t out_y_region[] = {output_nv12_y_desc.image_width, output_nv12_y_desc.image_height, 1};
    row_pitch      = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
        command_queue,
        output_nv12_y,
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
        std::cerr << "Error " << err << " mapping dest image y-plane buffer for reading." << "\n";
        return err;
    }

    // Copies image data from the DMA buffer to the host.
    for (uint32_t i = 0; i < output_nv12_y_desc.image_height; ++i)
    {
        std::memcpy(
            out_nv12_image_info.y_plane.data() + i * output_nv12_y_desc.image_width,
            image_ptr                          + i * row_pitch,
            output_nv12_y_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, output_nv12_y, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image y-plane buffer." << "\n";
        return err;
    }

    const size_t out_uv_region[] = {output_nv12_uv_desc.image_width / 2, output_nv12_uv_desc.image_height / 2, 1};
    row_pitch                    = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
        command_queue,
        output_nv12_uv,
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
        std::cerr << "Error " << err << " mapping dest image uv-plane buffer for reading." << "\n";
        return err;
    }

    // Copies image data from the DMA buffer to the host.
    for (uint32_t i = 0; i < output_nv12_uv_desc.image_height / 2; ++i)
    {
        std::memcpy(
            out_nv12_image_info.uv_plane.data() + i * output_nv12_uv_desc.image_width,
            image_ptr                           + i * row_pitch,
            output_nv12_uv_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, output_nv12_uv, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image uv-plane buffer." << "\n";
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
    if (ioctl(output_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    save_nv12_image_data(out_image_filename, out_nv12_image_info);

    /*
     * Step 9: Cleanup
     */
    clReleaseSampler(sampler);
    clReleaseMemObject(input_tp10_image);
    clReleaseMemObject(input_tp10_y);
    clReleaseMemObject(input_tp10_uv);
    clReleaseMemObject(output_nv12_image);
    clReleaseMemObject(output_nv12_y);
    clReleaseMemObject(output_nv12_uv);

    return 0;
}