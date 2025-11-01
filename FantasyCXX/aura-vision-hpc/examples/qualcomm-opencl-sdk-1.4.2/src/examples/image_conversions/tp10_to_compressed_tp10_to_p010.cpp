//--------------------------------------------------------------------------------------
// File: tp10_to_compressed_tp10_to_p010.cpp
// Desc: This program converts TP10 to compressed TP10 to P010
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
#include <vndk/hardware_buffer.h>
#include <ui/GraphicBuffer.h>

#define HAL_PIXEL_FORMAT_YCbCr_420_TP10_UBWC     0x7FA30C09
#define HAL_PIXEL_FORMAT_YCbCr_420_P010_VENUS    0x7FA30C0A

static const char *PROGRAM_SOURCE = R"(
    static const int Y_COMPONENT = 0;
    static const int U_COMPONENT = 1;
    static const int V_COMPONENT = 2;

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

        float2 uv_pixels_out[] = {
            {pixels_in[0].s1, pixels_in[0].s2},
            {pixels_in[1].s1, pixels_in[1].s2},
            {pixels_in[2].s1, pixels_in[2].s2},
        };

        qcom_write_imagefv_3x1_n10t01(dest_image_uv_plane, write_coord, uv_pixels_out);
    }

    // This kernel writes from a compressed TP10 image (both planes) to a P010 image.
    __kernel void compressed_tp10_to_p010(__read_only  image2d_t src_image,
                                          __write_only image2d_t dest_p010_y,
                                          __write_only image2d_t dest_p010_uv,
                                                       sampler_t sampler)
    {
        const int    wid_x              = get_global_id(0);
        const int    wid_y              = get_global_id(1);
        const float2 read_coord         = (float2)(4 * wid_x, 4 * wid_y) + 0.5;
        const int2   y_write_coord      = (int2)(4 * wid_x, 4 * wid_y);
        const int2   uv_write_coord     = (int2)(2 * wid_x, 2 * wid_y);

        const float4 y_pixels_in[]      = {
            qcom_read_imagef_2x2(src_image, sampler, read_coord,                    Y_COMPONENT),
            qcom_read_imagef_2x2(src_image, sampler, read_coord + (float2)(2., 0.), Y_COMPONENT),
            qcom_read_imagef_2x2(src_image, sampler, read_coord + (float2)(0., 2.), Y_COMPONENT),
            qcom_read_imagef_2x2(src_image, sampler, read_coord + (float2)(2., 2.), Y_COMPONENT),
        };

        float y_pixels_out[4][4] = {
            {y_pixels_in[0].s3, y_pixels_in[0].s2, y_pixels_in[1].s3, y_pixels_in[1].s2},
            {y_pixels_in[0].s0, y_pixels_in[0].s1, y_pixels_in[1].s0, y_pixels_in[1].s1},
            {y_pixels_in[2].s3, y_pixels_in[2].s2, y_pixels_in[3].s3, y_pixels_in[3].s2},
            {y_pixels_in[2].s0, y_pixels_in[2].s1, y_pixels_in[3].s0, y_pixels_in[3].s1},
        };

        qcom_write_imagefv_4x1_n10p00(dest_p010_y, y_write_coord,                y_pixels_out[0]);
        qcom_write_imagefv_4x1_n10p00(dest_p010_y, y_write_coord + (int2)(0, 1), y_pixels_out[1]);
        qcom_write_imagefv_4x1_n10p00(dest_p010_y, y_write_coord + (int2)(0, 2), y_pixels_out[2]);
        qcom_write_imagefv_4x1_n10p00(dest_p010_y, y_write_coord + (int2)(0, 3), y_pixels_out[3]);

        const float4 u_pixels_in       = qcom_read_imagef_2x2(src_image, sampler, read_coord, U_COMPONENT);
        const float4 v_pixels_in       = qcom_read_imagef_2x2(src_image, sampler, read_coord, V_COMPONENT);

        float2 uv_pixels_out[2][2] = {
            {{u_pixels_in.s3, v_pixels_in.s3}, {u_pixels_in.s2, v_pixels_in.s2}},
            {{u_pixels_in.s0, v_pixels_in.s0}, {u_pixels_in.s1, v_pixels_in.s1}},
        };

        qcom_write_imagefv_2x1_n10p01(dest_p010_uv, uv_write_coord,                uv_pixels_out[0]);
        qcom_write_imagefv_2x1_n10p01(dest_p010_uv, uv_write_coord + (int2)(0, 1), uv_pixels_out[1]);
    }
)";

cl_mem CreatePlaneFromImage(cl_context context, cl_channel_order channel_order, cl_channel_type channel_type,
    size_t width, size_t height, cl_mem image, cl_mem_flags flags, int *err)
{
    cl_image_format plane_format;
    plane_format.image_channel_order = channel_order;
    plane_format.image_channel_data_type = channel_type;

    cl_image_desc plane_desc;
    std::memset(&plane_desc, 0, sizeof(plane_desc));
    plane_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    plane_desc.image_width = width;
    plane_desc.image_height = height;
    plane_desc.mem_object = image;

    cl_mem plane = clCreateImage(
        context,
        flags,
        &plane_format,
        &plane_desc,
        nullptr,
        err
    );
    if (*err != CL_SUCCESS) {
        std::cerr << "Error " << *err << " with clCreateImage for plane." <<"\n";
        return nullptr;
    }
    *err = CL_SUCCESS;
    return plane;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input image> <output image> "
                                             "Input image file data should be in format CL_QCOM_TP10 / CL_QCOM_UNORM_INT10. "
                                             "Demonstrates conversions from CL_QCOM_TP10 to CL_QCOM_COMPRESSED_TP10 to P010 using vectorized read/writes. ";
        return EXIT_FAILURE;
    }

    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper       wrapper;
    cl_program       program                    = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel        tp10_to_tp10_compressed_y  = wrapper.make_kernel("tp10_y_compress", program);
    cl_kernel        tp10_to_tp10_compressed_uv = wrapper.make_kernel("tp10_uv_compress", program);
    cl_kernel        compressed_tp10_to_p010    = wrapper.make_kernel("compressed_tp10_to_p010", program);
    cl_context       context                    = wrapper.get_context();
    cl_command_queue command_queue              = wrapper.get_command_queue();
    tp10_image_t     input_tp10_image_file      = load_tp10_image_data(src_image_filename);
    cl_sampler       sampler                    = nullptr;
    cl_int           err                        = 0;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */
    if (!wrapper.check_extension_support("cl_qcom_other_image")) {
        std::cerr << "Extension cl_qcom_other_image needed for TP10 image format is not supported.";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_android_ahardwarebuffer_host_ptr")) {
        std::cerr << "Extension cl_qcom_android_ahardwarebuffer_host_ptr needed for graphic buffer support.";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_ext_host_ptr_iocoherent")) {
        std::cerr << "Extension cl_qcom_ext_host_ptr_iocoherent needed for specifying cache policy.\n";
        return EXIT_FAILURE;
    }

    std::vector<cl_image_format> formats;
    formats = get_image_formats(context, CL_MEM_READ_WRITE | CL_MEM_OTHER_IMAGE_QCOM);
    const bool rw_formats_supported =
        is_format_supported(formats, cl_image_format{CL_QCOM_P010_Y, CL_QCOM_UNORM_INT10}) &&
        is_format_supported(formats, cl_image_format{CL_QCOM_P010_UV, CL_QCOM_UNORM_INT10});
    if (!rw_formats_supported) {
        std::cerr << "For this example your device must support read-write CL_QCOM_P010_Y/UV "
                     "with CL_QCOM_UNORM_INT10 image format, but it does not.\n";
        std::cerr << "Supported read-write formats include:\n";
        print_formats(formats);
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create a GraphicBuffer then convert to AHardwareBuffer.
     */
    cl_image_format input_tp10_format;
    input_tp10_format.image_channel_order     = CL_QCOM_TP10;
    input_tp10_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc input_tp10_desc = {0};
    input_tp10_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    input_tp10_desc.image_width  = input_tp10_image_file.y_width;
    input_tp10_desc.image_height = input_tp10_image_file.y_height;

    android::GraphicBuffer *src_tp10_gb = new android::GraphicBuffer();
    src_tp10_gb->reallocate(input_tp10_image_file.y_width, input_tp10_image_file.y_height,
                            HAL_PIXEL_FORMAT_YCbCr_420_TP10_UBWC , 1,
                            android::GraphicBuffer::USAGE_SW_WRITE_OFTEN | android::GraphicBuffer::USAGE_HW_TEXTURE);
    AHardwareBuffer *p_input_ahb = src_tp10_gb->toAHardwareBuffer();
    if (p_input_ahb == nullptr) {
        std::cerr << "HAL_PIXEL_FORMAT_YCbCr_420_TP10_UBWC failed with error code " << err <<"\n";
        return err;
    }

    cl_image_format output_tp10_compressed_format;
    output_tp10_compressed_format.image_channel_order     = CL_QCOM_COMPRESSED_TP10;
    output_tp10_compressed_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc output_tp10_compressed_desc = {0};
    output_tp10_compressed_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    output_tp10_compressed_desc.image_width  = input_tp10_image_file.y_width;
    output_tp10_compressed_desc.image_height = input_tp10_image_file.y_height;

    android::GraphicBuffer *tp10_compressed_gb = new android::GraphicBuffer();
    tp10_compressed_gb->reallocate(input_tp10_image_file.y_width, input_tp10_image_file.y_height,
                                   HAL_PIXEL_FORMAT_YCbCr_420_TP10_UBWC, 1,
                                   android::GraphicBuffer::USAGE_HW_RENDER | android::GraphicBuffer::USAGE_HW_TEXTURE);
    AHardwareBuffer *p_tp10_compressed_ahb = tp10_compressed_gb->toAHardwareBuffer();
    if (p_tp10_compressed_ahb == nullptr) {
        std::cerr << "HAL_PIXEL_FORMAT_YCbCr_420_TP10_UBWC failed with error code " << err <<"\n";
        return err;
    }

    cl_image_format out_p010_format;
    out_p010_format.image_channel_order     = CL_QCOM_P010;
    out_p010_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_p010_desc;
    std::memset(&out_p010_desc, 0, sizeof(out_p010_desc));
    out_p010_desc.image_type             = CL_MEM_OBJECT_IMAGE2D;
    out_p010_desc.image_width            = input_tp10_image_file.y_width;
    out_p010_desc.image_height           = input_tp10_image_file.y_height;
    const size_t img_row_pitch           = wrapper.get_image_row_pitch(out_p010_format, out_p010_desc);
    out_p010_desc.image_row_pitch        = img_row_pitch;

    android::GraphicBuffer *out_p010_gb = new android::GraphicBuffer();
    out_p010_gb->reallocate(input_tp10_image_file.y_width, input_tp10_image_file.y_height,
                            HAL_PIXEL_FORMAT_YCbCr_420_P010_VENUS, 1,
                            android::GraphicBuffer::USAGE_HW_RENDER | android::GraphicBuffer::USAGE_HW_TEXTURE);
    AHardwareBuffer *p_output_ahb = out_p010_gb->toAHardwareBuffer();
    if (p_output_ahb == nullptr) {
        std::cerr << "HAL_PIXEL_FORMAT_YCbCr_420_P010_VENUS failed with error code " << err <<"\n";
        return err;
    }

    /*
     * Step 2: Create parent images for planar images.
     */
    cl_mem src_tp10_image;
    {
        cl_mem_ahardwarebuffer_host_ptr input_ahb_mem;
        input_ahb_mem.ext_host_ptr.allocation_type = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
        input_ahb_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
        input_ahb_mem.ahb_ptr = p_input_ahb;


        src_tp10_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &input_tp10_format,
            &input_tp10_desc,
            &input_ahb_mem,
            &err
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Error " << err << " with clCreateImage for source image." <<"\n";
            return err;
        }
    }

    cl_mem compressed_tp10_image;
    {
        cl_mem_ahardwarebuffer_host_ptr input_ahb_mem;
        input_ahb_mem.ext_host_ptr.allocation_type = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
        input_ahb_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
        input_ahb_mem.ahb_ptr = p_tp10_compressed_ahb;

        compressed_tp10_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &output_tp10_compressed_format,
            &output_tp10_compressed_desc,
            &input_ahb_mem,
            &err
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Error " << err << " with clCreateImage for compressed source image." <<"\n";
            return err;
        }
    }

    cl_mem out_p010_image;
    {
        cl_mem_ahardwarebuffer_host_ptr output_ahb_mem;
        output_ahb_mem.ext_host_ptr.allocation_type = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
        output_ahb_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
        output_ahb_mem.ahb_ptr = p_output_ahb;

        out_p010_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_p010_format,
            &out_p010_desc,
            &output_ahb_mem,
            &err
        );
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clCreateImage for output P010 image." <<"\n";
            return err;
        }
    }

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
        CL_MEM_READ_WRITE,
        &out_y_plane_format,
        &out_y_plane_desc,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for destination image y plane." <<"\n";
        return err;
    }

    cl_image_format out_uv_plane_format;
    out_uv_plane_format.image_channel_order     = CL_QCOM_P010_UV;
    out_uv_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_uv_plane_desc;
    std::memset(&out_uv_plane_desc, 0, sizeof(out_uv_plane_desc));
    out_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    out_uv_plane_desc.image_width  = out_p010_desc.image_width;
    out_uv_plane_desc.image_height = out_p010_desc.image_height;
    out_uv_plane_desc.mem_object   = out_p010_image;

    cl_mem out_uv_plane = clCreateImage(
        context,
        CL_MEM_READ_WRITE,
        &out_uv_plane_format,
        &out_uv_plane_desc,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for destination image uv plane." <<"\n";
        return err;
    }

    cl_mem src_y_plane = CreatePlaneFromImage(context, CL_QCOM_TP10_Y, CL_QCOM_UNORM_INT10, input_tp10_desc.image_width,
                                              input_tp10_desc.image_height, src_tp10_image,CL_MEM_READ_ONLY, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clCreateImage for source image y plane." <<"\n";
        return err;
    }

    cl_mem src_uv_plane = CreatePlaneFromImage(context, CL_QCOM_TP10_UV, CL_QCOM_UNORM_INT10, input_tp10_desc.image_width,
                                               input_tp10_desc.image_height, src_tp10_image, CL_MEM_READ_ONLY, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clCreateImage for source image uv plane." <<"\n";
        return err;
    }

    cl_mem compressed_y_plane = CreatePlaneFromImage(context, CL_QCOM_COMPRESSED_TP10_Y, CL_QCOM_UNORM_INT10, input_tp10_desc.image_width,
                                                     input_tp10_desc.image_height, compressed_tp10_image, CL_MEM_WRITE_ONLY, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clCreateImage for source compressed image y plane." <<"\n";
        return err;
    }

    cl_mem compressed_uv_plane = CreatePlaneFromImage(context, CL_QCOM_COMPRESSED_TP10_UV, CL_QCOM_UNORM_INT10, input_tp10_desc.image_width,
                                                      input_tp10_desc.image_height, compressed_tp10_image, CL_MEM_WRITE_ONLY, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clCreateImage for source image compressed uv plane." <<"\n";
        return err;
    }

    /*
     * Step 3: Copy data to input image planes. Note that for linear TP10 images you must observe row alignment
     * restrictions.
     */
    const size_t     origin[]       = {0, 0, 0};
    const size_t     src_y_region[] = {input_tp10_desc.image_width, input_tp10_desc.image_height, 1};
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
        std::cerr << "Error " << err << " mapping source image y-plane buffer for writing." <<"\n";
        return err;
    }

    {
        void* locked_ptr = nullptr;
        src_tp10_gb->lock(android::GraphicBuffer::USAGE_SW_READ_OFTEN, &locked_ptr);

        // Copies image data to the AHB buffer from the host
        for (uint32_t i = 0; i < input_tp10_desc.image_height; ++i) {
            std::memcpy(
                image_ptr                             + i * row_pitch,
                input_tp10_image_file.y_plane.data()  + i * input_tp10_desc.image_width * 4 / 3,
                input_tp10_desc.image_width * 4 / 3
            );
        }

        src_tp10_gb->unlock();
    }

    err = clEnqueueUnmapMemObject(command_queue, src_y_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer." << "\n";
        return err;
    }

    // For mapping the NV12_UV image for host access, the region dimensions should reflect the actual number of UV pixels
    // being read. This is different from the dimensions passed in to create the NV12_Y image. In that case, the width
    // and height of the parent YUV image are expected.
    const size_t src_uv_region[] = {input_tp10_image_file.y_width / 2, input_tp10_image_file.y_height / 2, 1};
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

    {
        void* locked_ptr = nullptr;
        src_tp10_gb->lock(android::GraphicBuffer::USAGE_SW_READ_OFTEN, &locked_ptr);

        // Copies image data to the AHB buffer from the host
        for (uint32_t i = 0; i < input_tp10_image_file.y_height / 2; ++i) {
            std::memcpy(
                image_ptr                             + i * row_pitch,
                input_tp10_image_file.uv_plane.data() + i * input_tp10_desc.image_width * 4 / 3,
                input_tp10_desc.image_width * 4 / 3
            );
        }

        src_tp10_gb->unlock();
    }

    err = clEnqueueUnmapMemObject(command_queue, src_uv_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image uv-plane data buffer." <<"\n";
        return err;
    }

    /*
     * Step 4: Set kernel arguments for our kernels.
     */
    sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateSampler." <<"\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_tp10_compressed_y, 0, sizeof(src_tp10_image), &src_tp10_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." <<"\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_tp10_compressed_y, 1, sizeof(compressed_y_plane), &compressed_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." <<"\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_tp10_compressed_y, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." <<"\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_tp10_compressed_uv, 0, sizeof(src_tp10_image), &src_tp10_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." <<"\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_tp10_compressed_uv, 1, sizeof(compressed_uv_plane), &compressed_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." <<"\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_tp10_compressed_uv, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." <<"\n";
        return err;
    }

    err = clSetKernelArg(compressed_tp10_to_p010, 0, sizeof(compressed_tp10_image), &compressed_tp10_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." <<"\n";
        return err;
    }

    err = clSetKernelArg(compressed_tp10_to_p010, 1, sizeof(out_y_plane), &out_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." <<"\n";
        return err;
    }

    err = clSetKernelArg(compressed_tp10_to_p010, 2, sizeof(out_uv_plane), &out_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." <<"\n";
        return err;
    }

    err = clSetKernelArg(compressed_tp10_to_p010, 3, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_y_compress kernel." <<"\n";
        return err;
    }

    /*
     * Step 6: Run Kernels
     */
    size_t global_work_size[] = {0, 0};

    global_work_size[0] = input_tp10_desc.image_width / 3;
    global_work_size[1] = input_tp10_desc.image_height;
    err = clEnqueueNDRangeKernel(command_queue, tp10_to_tp10_compressed_y, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel tp10_to_tp10_compressed_y." <<"\n";
        return err;
    }

    global_work_size[0] = input_tp10_desc.image_width / 6;
    global_work_size[1] = input_tp10_desc.image_height / 2;
    err = clEnqueueNDRangeKernel(command_queue, tp10_to_tp10_compressed_uv, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel tp10_to_tp10_compressed_uv." <<"\n";
        return err;
    }

    global_work_size[0] = work_units(input_tp10_desc.image_width, 4);
    global_work_size[1] = work_units(input_tp10_desc.image_height, 4);
    err = clEnqueueNDRangeKernel(command_queue, compressed_tp10_to_p010, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel compressed_tp10_to_p010." << "\n";
        return err;
    }

    err = clFinish(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clFinish." << "\n";
        return err;
    }

    /*
     * Step 7: Copy the data out of the ion buffer for each plane.
     */
    size_t image_size;
    out_p010_desc.image_array_size = 1;
    err = clQueryImageInfoQCOM(wrapper.get_device_id(), CL_MEM_READ_ONLY, &out_p010_format, &out_p010_desc, CL_IMAGE_SIZE_QCOM, sizeof(image_size), &image_size, nullptr);

    std::cout << "Output image size: " << image_size << std::endl;

    p010_image_t out_image_info;
    out_image_info.y_width  = out_p010_desc.image_width;
    out_image_info.y_height = out_p010_desc.image_height;
    out_image_info.y_plane.resize(out_image_info.y_width * out_image_info.y_height * 2);
    out_image_info.uv_plane.resize(out_image_info.y_width * out_image_info.y_height);

    const size_t out_y_region[] = {out_y_plane_desc.image_width, out_y_plane_desc.image_height, 1};
    row_pitch                   = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
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
        std::cerr << "Error " << err << " mapping dest image y-plane buffer for reading." << "\n";
        return err;
    }

    {
        void* locked_ptr;
        out_p010_gb->lock(android::GraphicBuffer::USAGE_SW_READ_OFTEN, &locked_ptr);

        // Copies image data from buffer to host
        for (uint32_t i = 0; i < out_y_plane_desc.image_height; ++i)
        {
            std::memcpy(
                out_image_info.y_plane.data() + i * out_y_plane_desc.image_width * 2,
                image_ptr                     + i * row_pitch,
                out_y_plane_desc.image_width * 2
            );
        }

        out_p010_gb->unlock();
    }

    err = clEnqueueUnmapMemObject(command_queue, out_y_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image y-plane buffer." <<"\n";
        return err;
    }

    const size_t out_uv_region[] = {out_uv_plane_desc.image_width / 2, out_uv_plane_desc.image_height / 2, 1};
    row_pitch                    = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
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
        std::cerr << "Error " << err << " mapping dest image uv-plane buffer for reading." <<"\n";
        return err;
    }

    {
        void* locked_ptr;
        out_p010_gb->lock(android::GraphicBuffer::USAGE_SW_READ_OFTEN, &locked_ptr);

        // Copies image data from the AHB buffer to the host
        for (uint32_t i = 0; i < out_uv_plane_desc.image_height / 2; ++i)
        {
            std::memcpy(
                out_image_info.uv_plane.data() + i * out_uv_plane_desc.image_width * 2,
                image_ptr                      + i * row_pitch,
                out_uv_plane_desc.image_width * 2
            );
        }

        out_p010_gb->unlock();
    }

    err = clEnqueueUnmapMemObject(command_queue, out_uv_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image uv-plane buffer." <<"\n";
        return err;
    }

    err = clFinish(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while finishing the command queue.\n";
        return err;
    }

    save_p010_image_data(out_image_filename, out_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_tp10_image);
    clReleaseMemObject(compressed_uv_plane);
    clReleaseMemObject(compressed_y_plane);
    clReleaseMemObject(compressed_tp10_image);
    clReleaseMemObject(out_uv_plane);
    clReleaseMemObject(out_y_plane);
    clReleaseMemObject(out_p010_image);

    return EXIT_SUCCESS;
}
