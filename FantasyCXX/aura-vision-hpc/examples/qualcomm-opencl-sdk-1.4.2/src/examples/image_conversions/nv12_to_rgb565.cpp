//--------------------------------------------------------------------------------------
// File: nv12_to_rgb565.cpp
// Desc: This program converts NV12 to RGB565.
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
#include <vector>

// Project includes
#include "util/cl_wrapper.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>
#include <vndk/hardware_buffer.h>
#include <ui/GraphicBuffer.h>

#define HAL_PIXEL_FORMAT_RGB_565            0x4
#define HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS 0x7FA30C04

static const char *PROGRAM_SOURCE = R"(
    __kernel void
     nv12_to_rgb565(
         __read_only  image2d_t input_nv12,
         __write_only image2d_t out_r,
                      sampler_t sampler)
    {
        int2    coord;
        float4  yuv;
        ushort3 rgb;
        ushort  output_packed_pixel;

        coord.x = get_global_id(0);
        coord.y = get_global_id(1);

        yuv = read_imagef(input_nv12, sampler, coord);
        yuv.y = (yuv.y - 0.5f) * 0.872f;
        yuv.z = (yuv.z - 0.5f) * 1.23f;
        rgb.x = convert_ushort(clamp((yuv.x + (1.140f * yuv.z)) * 31.0f, 0.0f, 31.0f));
        rgb.y = convert_ushort(clamp((yuv.x - (0.395f * yuv.y) - (0.581f * yuv.z)) * 63.0f, 0.0f, 63.0f));
        rgb.z = convert_ushort(clamp((yuv.x + (2.032f * yuv.y)) * 31.0f, 0.0f, 31.0f));

        output_packed_pixel = (rgb.x << 11) | (rgb.y << 5) | rgb.z;

        write_imageui(out_r, coord, output_packed_pixel);
    }
)";

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <img data file> <out img data file> \n"
                                             "Input image file data should be in format CL_QCOM_NV12 / CL_UNORM_INT8\n"
                                             "Demonstrates conversions from NV12 to RGB565\n";
        return EXIT_FAILURE;
    }

    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper   wrapper;
    cl_program   program                        = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel    nv12_to_rgb565_kernel          = wrapper.make_kernel("nv12_to_rgb565", program);
    cl_context   context                        = wrapper.get_context();
    nv12_image_t src_nv12_image_info            = load_nv12_image_data(src_image_filename);
    cl_int       err                            = CL_SUCCESS;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */
    if (!wrapper.check_extension_support("cl_qcom_other_image")) {
        std::cerr << "Extension cl_qcom_other_image needed for NV12 image format is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_android_ahardwarebuffer_host_ptr")) {
        std::cerr << "Extension cl_qcom_android_ahardwarebuffer_host_ptr needed for graphic buffer support.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_ext_host_ptr_iocoherent")) {
        std::cerr << "Extension cl_qcom_ext_host_ptr_iocoherent needed for specifying cache policy.\n";
        return EXIT_FAILURE;
    }

    std::vector<cl_image_format> formats;
    formats = get_image_formats(context, CL_MEM_READ_WRITE | CL_MEM_OTHER_IMAGE_QCOM);
    const bool rw_formats_supported =
        is_format_supported(formats, cl_image_format{CL_QCOM_NV12_Y,  CL_UNORM_INT8}) &&
        is_format_supported(formats, cl_image_format{CL_QCOM_NV12_UV, CL_UNORM_INT8});
    if (!rw_formats_supported) {
        std::cerr << "For this example your device must support read-write CL_QCOM_NV12_Y/UV"
                     "with CL_UNORM_INT8 image format, but it does not.\n";
        std::cerr << "Supported read-write formats include:\n";
        print_formats(formats);
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create suitable descriptors for CL images. Note that planar formats (like NV12) must be read only,
     * but you can write to child images derived from the planes. (See step 4 for deriving child images.)
     */
    cl_image_format src_nv12_format;
    src_nv12_format.image_channel_order = CL_QCOM_NV12;
    src_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_nv12_desc = {0};
    src_nv12_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    src_nv12_desc.image_width = src_nv12_image_info.y_width;
    src_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_image_format out_rgba_format;
    out_rgba_format.image_channel_order     = CL_R;
    out_rgba_format.image_channel_data_type = CL_UNSIGNED_INT16;

    cl_image_desc out_rgba_desc = {0};
    out_rgba_desc.image_type             = CL_MEM_OBJECT_IMAGE2D;
    out_rgba_desc.image_width            = src_nv12_image_info.y_width;
    out_rgba_desc.image_height           = src_nv12_image_info.y_height;
    const size_t img_row_pitch           = wrapper.get_image_row_pitch(out_rgba_format, out_rgba_desc);
    out_rgba_desc.image_row_pitch        = img_row_pitch;

    /*
     * Step 2: Create a GraphicBuffer then convert to AHardwareBuffer.
     */
    android::GraphicBuffer *src_nv12_gb = new android::GraphicBuffer();
    src_nv12_gb->reallocate(src_nv12_image_info.y_width, src_nv12_image_info.y_height,
                            HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS, 1,
                            android::GraphicBuffer::USAGE_SW_WRITE_OFTEN | android::GraphicBuffer::USAGE_HW_TEXTURE);
    AHardwareBuffer *p_input_ahb = src_nv12_gb->toAHardwareBuffer();
    if (p_input_ahb == nullptr) {
        std::cerr << "HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS failed with error code " << err << "\n";
        return err;
    }

    android::GraphicBuffer *rgb565_gb = new android::GraphicBuffer();
    rgb565_gb->reallocate(src_nv12_image_info.y_width, src_nv12_image_info.y_height,
                          HAL_PIXEL_FORMAT_RGB_565, 1,
                          android::GraphicBuffer::USAGE_HW_RENDER | android::GraphicBuffer::USAGE_HW_TEXTURE | android::GraphicBuffer::USAGE_SW_READ_OFTEN);
    AHardwareBuffer *p_output_ahb = rgb565_gb->toAHardwareBuffer();
    if (p_output_ahb == nullptr) {
        std::cerr << "HAL_PIXEL_FORMAT_RGB_565 failed with error code " << err << "\n";
        return err;
    }

    /*
     * Step 3: Create OpenCL image objects that uses AHardwareBuffer as its data store.
     */
    cl_mem src_nv12_image;
    {
        cl_mem_ahardwarebuffer_host_ptr input_ahb_mem;
        input_ahb_mem.ext_host_ptr.allocation_type = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
        input_ahb_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
        input_ahb_mem.ahb_ptr = p_input_ahb;

        src_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &src_nv12_format,
            &src_nv12_desc,
            &input_ahb_mem,
            &err
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Error " << err << " with clCreateImage for source image." << "\n";
            return err;
        }
    }

    cl_mem out_rgb565_image;
    {
        cl_mem_ahardwarebuffer_host_ptr output_ahb_mem;
        output_ahb_mem.ext_host_ptr.allocation_type = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
        output_ahb_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
        output_ahb_mem.ahb_ptr = p_output_ahb;

        out_rgb565_image = clCreateImage(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_rgba_format,
            &out_rgba_desc,
            &output_ahb_mem,
            &err
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Error " << err << " with clCreateImage for output image." << "\n";
            return err;
        }
    }

    /*
     * Step 4: Separate planar NV12 image into its component planes.
     */
    cl_image_format src_y_plane_format;
    src_y_plane_format.image_channel_order     = CL_QCOM_NV12_Y;
    src_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_y_plane_desc = {0};
    src_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_y_plane_desc.image_width  = src_nv12_image_info.y_width;
    src_y_plane_desc.image_height = src_nv12_image_info.y_height;
    src_y_plane_desc.mem_object   = src_nv12_image;

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
    src_uv_plane_format.image_channel_order     = CL_QCOM_NV12_UV;
    src_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_uv_plane_desc = {0};
    src_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    src_uv_plane_desc.image_width  = src_nv12_image_info.y_width;
    src_uv_plane_desc.image_height = src_nv12_image_info.y_height;
    src_uv_plane_desc.mem_object   = src_nv12_image;

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

    /*
     * Step 5: Copy data to input image planes. Note that for linear NV12 images you must observe row alignment
     * restrictions.
     */
    cl_command_queue command_queue  = wrapper.get_command_queue();
    const size_t     origin[]       = {0, 0, 0};
    const size_t     src_y_region[] = {src_nv12_desc.image_width, src_nv12_desc.image_height, 1};
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

    {
        void* locked_ptr = nullptr;
        src_nv12_gb->lock(android::GraphicBuffer::USAGE_SW_READ_OFTEN, &locked_ptr);

        // Copies image data to the AHB buffer from the host
        for (uint32_t i = 0; i < src_nv12_desc.image_height; ++i) {
            std::memcpy(
                image_ptr                          + i * row_pitch,
                src_nv12_image_info.y_plane.data() + i * src_nv12_desc.image_width,
                src_nv12_desc.image_width
            );
        }

        src_nv12_gb->unlock();
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
    const size_t src_uv_region[] = {src_nv12_image_info.y_width / 2, src_nv12_image_info.y_height / 2, 1};
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
        src_nv12_gb->lock(android::GraphicBuffer::USAGE_SW_READ_OFTEN, &locked_ptr);

        // Copies image data to the AHB buffer from the host
        for (uint32_t i = 0; i < src_nv12_image_info.y_height / 2; ++i) {
            std::memcpy(
                image_ptr + i * row_pitch,
                src_nv12_image_info.uv_plane.data() + i * src_nv12_image_info.y_width,
                src_nv12_image_info.y_width
            );
        }

        src_nv12_gb->unlock();
    }

    err = clEnqueueUnmapMemObject(command_queue, src_uv_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " unmapping source image uv-plane data buffer." << "\n";
        return err;
    }

    /*
     * Step 6: Set up kernel arguments
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

    err = clSetKernelArg(nv12_to_rgb565_kernel, 0, sizeof(src_nv12_image), &src_nv12_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for nv12_to_rgb565_kernel.\n";
        return err;
    }

    err = clSetKernelArg(nv12_to_rgb565_kernel, 1, sizeof(out_rgb565_image), &out_rgb565_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for nv12_to_rgb565_kernel.\n";
        return err;
    }

    err = clSetKernelArg(nv12_to_rgb565_kernel, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for nv12_to_rgb565_kernel.\n";
        return err;
    }

    // Run nv12 to rgb565 kernel
    {
        const size_t work_size[] = {src_nv12_desc.image_width, src_nv12_desc.image_height};
        err = clEnqueueNDRangeKernel(
            command_queue,
            nv12_to_rgb565_kernel,
            2,
            nullptr,
            work_size,
            nullptr,
            0,
            nullptr,
            nullptr
        );
        if (err != CL_SUCCESS) {
            std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for nv12_to_rgb565_kernel.\n";
            return err;
        }
    }

    err = clFinish(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while finishing the command queue.\n";
        return err;
    }

    /*
     * Step 7: Copy the data out of the image buffer taking alignment and padding into account.
     */
    size_t image_size = src_nv12_desc.image_width * src_nv12_desc.image_height * 2;

    std::vector<unsigned char> rgb565_image_raw;
    rgb565_image_raw.resize(image_size);

    const size_t out_region[] = {src_nv12_desc.image_width, src_nv12_desc.image_height, 1};
    row_pitch = 0;
    image_ptr = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
        command_queue,
        out_rgb565_image,
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
        std::cerr << "Error " << err << " mapping dest out_rgb565_image buffer for reading." << "\n";
        std::exit(err);
    }

    // Unlock graphics buffer and copy data to vector
    void* locked_ptr;
    rgb565_gb->lock(android::GraphicBuffer::USAGE_SW_READ_OFTEN, &locked_ptr);
    for (uint32_t i = 0; i < out_rgba_desc.image_height; ++i)
    {
        std::memcpy(
            rgb565_image_raw.data()   + i * out_rgba_desc.image_width * 2,
            image_ptr                 + i * row_pitch,
            out_rgba_desc.image_width     * 2
        );
    }
    rgb565_gb->unlock();

    err = clEnqueueUnmapMemObject(command_queue, out_rgb565_image, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest out_rgb565_image buffer." << "\n";
        std::exit(err);
    }

    err = clFinish(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while finishing the command queue.\n";
        return err;
    }

    save_raw_image_data(out_image_filename, rgb565_image_raw, image_size);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_nv12_image);
    clReleaseMemObject(out_rgb565_image);

    return EXIT_SUCCESS;
}
