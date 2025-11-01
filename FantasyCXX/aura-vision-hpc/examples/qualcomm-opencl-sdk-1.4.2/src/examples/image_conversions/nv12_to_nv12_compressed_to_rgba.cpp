//--------------------------------------------------------------------------------------
// File: nv12_to_nv12_compressed_to_rgba.cpp
// Desc: This program converts NV12 to compressed NV12 to RGBA8888
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

#define HAL_PIXEL_FORMAT_RGBA_8888               0x1
#define HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS      0x7FA30C04
#define HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS_UBWC 0x7FA30C06

static const char *PROGRAM_SOURCE = R"(
    __kernel void compress(__read_only  image2d_t src_plane,
                           __write_only image2d_t dst_plane,
                                        sampler_t sampler)
    {
        const int    wid_x = get_global_id(0);
        const int    wid_y = get_global_id(1);
        const int2   coord = (int2)(wid_x, wid_y);
        const float4 pixel = read_imagef(src_plane, sampler, coord);
        write_imagef(dst_plane, coord, pixel);
    }

    __kernel void nv12_to_rgba(
        __read_only image2d_t  input_nv12,
        __write_only image2d_t out_rgba,
        sampler_t sampler)
    {
        int2   coord;
        float4 yuv;
        float4 rgba;

        coord.x = get_global_id(0);
        coord.y = get_global_id(1);

        yuv = read_imagef(input_nv12, sampler, coord);
        yuv.y = (yuv.y - 0.5f) * 0.872f;
        yuv.z = (yuv.z - 0.5f) * 1.23f;
        rgba.x = yuv.x + (1.140f * yuv.z);
        rgba.y = yuv.x - (0.395f * yuv.y) - (0.581f * yuv.z);
        rgba.z = yuv.x + (2.032f * yuv.y);
        rgba.w = 1.0f;
        write_imagef(out_rgba, coord, rgba);
    }
)";

cl_mem CreatePlaneFromImage(cl_context context, cl_channel_order channel_order, cl_channel_type channel_type,
        size_t width, size_t height, cl_mem image, cl_mem_flags flags, int *err)
{
    cl_image_format plane_format;
    plane_format.image_channel_order = channel_order;
    plane_format.image_channel_data_type = channel_type;

    cl_image_desc plane_desc = {0};
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
        std::cerr << "Error " << *err << " with clCreateImage for plane." << "\n";
        return nullptr;
    }
    *err = CL_SUCCESS;
    return plane;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <img data file> <out img data file> \n"
                                             "Input image file data should be in format CL_QCOM_NV12 / CL_UNORM_INT8\n"
                                             "Demonstrates conversions from NV12 to compressed NV12 to RGBA8888\n";
        return EXIT_FAILURE;
    }

    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper wrapper;
    cl_program program = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel compressed_nv12_to_rgb_kernel = wrapper.make_kernel("nv12_to_rgba", program);
    cl_kernel nv12_to_compressed_nv12_kernel = wrapper.make_kernel("compress", program);
    cl_context context = wrapper.get_context();
    nv12_image_t src_nv12_image_info = load_nv12_image_data(src_image_filename);

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
    formats = get_image_formats(context, CL_MEM_READ_WRITE | CL_MEM_OTHER_IMAGE_QCOM | CL_MEM_COMPRESSED_IMAGE_QCOM);
    const bool rw_formats_supported =
        is_format_supported(formats, cl_image_format{CL_QCOM_COMPRESSED_NV12_Y,  CL_UNORM_INT8}) &&
        is_format_supported(formats, cl_image_format{CL_QCOM_COMPRESSED_NV12_UV, CL_UNORM_INT8});
    if (!rw_formats_supported) {
        std::cerr << "For this example your device must support CL_QCOM_COMPRESSED_NV12_Y/UV "
                     "with CL_UNORM_INT8 image format, but it does not.\n";
        std::cerr << "Supported read-write formats include:\n";
        print_formats(formats);
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create suitable descriptors for CL images. Note that planar formats (like NV12) must be read only,
     * but you can write to child images derived from the planes. (See step 2 for deriving child images.)
     */
    cl_image_format src_nv12_format;
    src_nv12_format.image_channel_order = CL_QCOM_NV12;
    src_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_nv12_desc = {0};
    src_nv12_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    src_nv12_desc.image_width = src_nv12_image_info.y_width;
    src_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_image_format src_compressed_nv12_format;
    src_compressed_nv12_format.image_channel_order = CL_QCOM_COMPRESSED_NV12;
    src_compressed_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_compressed_nv12_desc = {0};
    src_compressed_nv12_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    src_compressed_nv12_desc.image_width = src_nv12_image_info.y_width;
    src_compressed_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_int err = 0;

    /*
     * Step 1: Create a GraphicBuffer then convert to AHardwareBuffer.
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

    android::GraphicBuffer *nv12_compressed_gb = new android::GraphicBuffer();
    nv12_compressed_gb->reallocate(src_nv12_image_info.y_width, src_nv12_image_info.y_height,
                                   HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS_UBWC, 1,
                            android::GraphicBuffer::USAGE_HW_RENDER | android::GraphicBuffer::USAGE_HW_TEXTURE);
    AHardwareBuffer *p_nv12_compressed_ahb = nv12_compressed_gb->toAHardwareBuffer();
    if (p_nv12_compressed_ahb == nullptr) {
        std::cerr << "HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS_UBWC failed with error code " << err << "\n";
        return err;
    }

    android::GraphicBuffer *rgba_gb = new android::GraphicBuffer();
    rgba_gb->reallocate(src_nv12_image_info.y_width, src_nv12_image_info.y_height,
                        HAL_PIXEL_FORMAT_RGBA_8888, 1,
                        android::GraphicBuffer::USAGE_HW_RENDER | android::GraphicBuffer::USAGE_HW_TEXTURE);
    AHardwareBuffer *p_output_ahb = rgba_gb->toAHardwareBuffer();
    if (p_output_ahb == nullptr) {
        std::cerr << "HAL_PIXEL_FORMAT_RGBA_8888 failed with error code " << err << "\n";
        return err;
    }

    /*
     * Step 2: Create an OpenCL input image object that uses input_ahb_mem as its data store.
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

    cl_mem compressed_nv12_image;
    {
        cl_mem_ahardwarebuffer_host_ptr input_ahb_mem;
        input_ahb_mem.ext_host_ptr.allocation_type = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
        input_ahb_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
        input_ahb_mem.ahb_ptr = p_nv12_compressed_ahb;

        compressed_nv12_image = clCreateImage(
                context,
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                &src_compressed_nv12_format,
                &src_compressed_nv12_desc,
                &input_ahb_mem,
                &err
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Error " << err << " with clCreateImage for compressed image." << "\n";
            return err;
        }
    }

    cl_image_format out_rgba_format;
    out_rgba_format.image_channel_order     = CL_RGBA;
    out_rgba_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_rgba_desc = {0};
    out_rgba_desc.image_type             = CL_MEM_OBJECT_IMAGE2D;
    out_rgba_desc.image_width            = src_nv12_image_info.y_width;
    out_rgba_desc.image_height           = src_nv12_image_info.y_height;
    const size_t img_row_pitch           = wrapper.get_image_row_pitch(out_rgba_format, out_rgba_desc);
    out_rgba_desc.image_row_pitch        = img_row_pitch;

    cl_mem out_rgba_image;
    {
        cl_mem_ahardwarebuffer_host_ptr input_ahb_mem;
        input_ahb_mem.ext_host_ptr.allocation_type = CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM;
        input_ahb_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
        input_ahb_mem.ahb_ptr = p_output_ahb;

        out_rgba_image = clCreateImage(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_rgba_format,
            &out_rgba_desc,
            &input_ahb_mem,
            &err
        );
        if (err != CL_SUCCESS) {
            std::cerr << "Error " << err << " with clCreateImage for output RGBA image." << "\n";
            return err;
        }
    }

    /*
     * Step 2: Separate planar NV12 images into their component planes.
     */
    cl_mem src_y_plane = CreatePlaneFromImage(context, CL_QCOM_NV12_Y, CL_UNORM_INT8, src_nv12_desc.image_width,
            src_nv12_desc.image_height, src_nv12_image,CL_MEM_READ_ONLY, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clCreateImage for source image y plane." << "\n";
        return err;
    }

    cl_mem src_uv_plane = CreatePlaneFromImage(context, CL_QCOM_NV12_UV, CL_UNORM_INT8, src_nv12_desc.image_width,
                                               src_nv12_desc.image_height, src_nv12_image, CL_MEM_READ_ONLY, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clCreateImage for source image uv plane." << "\n";
        return err;
    }

    cl_mem compressed_y_plane = CreatePlaneFromImage(context, CL_QCOM_COMPRESSED_NV12_Y, CL_UNORM_INT8, src_nv12_desc.image_width,
                                                     src_nv12_desc.image_height, compressed_nv12_image, CL_MEM_WRITE_ONLY, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clCreateImage for source compressed image y plane." << "\n";
        return err;
    }

    cl_mem compressed_uv_plane = CreatePlaneFromImage(context, CL_QCOM_COMPRESSED_NV12_UV, CL_UNORM_INT8, src_nv12_desc.image_width,
                                                      src_nv12_desc.image_height, compressed_nv12_image, CL_MEM_WRITE_ONLY, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clCreateImage for source image compressed uv plane." << "\n";
        return err;
    }

    /*
     * Step 3: Copy data to input image planes. Note that for linear NV12 images you must observe row alignment
     * restrictions. (You may also write to the ion buffer directly if you prefer, however using clEnqueueMapImage for
     * a child planar image will return the correct host pointer for the desired plane.)
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
    if (err != CL_SUCCESS)
    {
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

    // Set kernel arguments for NV12_Y to NV12_UBWC_Y
    err = clSetKernelArg(nv12_to_compressed_nv12_kernel, 0, sizeof(src_y_plane), &src_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for nv12_to_compressed_nv12_kernel.\n";
        return err;
    }

    err = clSetKernelArg(nv12_to_compressed_nv12_kernel, 1, sizeof(compressed_y_plane), &compressed_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for nv12_to_compressed_nv12_kernel.\n";
        return err;
    }

    err = clSetKernelArg(nv12_to_compressed_nv12_kernel, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for nv12_to_compressed_nv12_kernel.\n";
        return err;
    }

    // Compress the NV12 Y-Plane
    {
        const size_t work_size[] = {src_nv12_desc.image_width, src_nv12_desc.image_height};
        err = clEnqueueNDRangeKernel(
            command_queue,
            nv12_to_compressed_nv12_kernel,
            2,
            nullptr,
            work_size,
            nullptr,
            0,
            nullptr,
            nullptr
        );
        if (err != CL_SUCCESS) {
            std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for nv12_to_compressed_nv12_kernel.\n";
            return err;
        }
    }

    // Set kernel arguments for NV12_UV to NV12_UBWC_UV
    err = clSetKernelArg(nv12_to_compressed_nv12_kernel, 0, sizeof(src_uv_plane), &src_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for nv12_to_compressed_nv12_kernel.\n";
        return err;
    }

    err = clSetKernelArg(nv12_to_compressed_nv12_kernel, 1, sizeof(compressed_uv_plane), &compressed_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for nv12_to_compressed_nv12_kernel.\n";
        return err;
    }

    err = clSetKernelArg(nv12_to_compressed_nv12_kernel, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for nv12_to_compressed_nv12_kernel.\n";
        return err;
    }

    // Compress the NV12 UV-Plane
    {
        const size_t work_size[] = {src_nv12_desc.image_width / 2, src_nv12_desc.image_height / 2};
        err = clEnqueueNDRangeKernel(
            command_queue,
            nv12_to_compressed_nv12_kernel,
            2,
            nullptr,
            work_size,
            nullptr,
            0,
            nullptr,
            nullptr
        );
        if (err != CL_SUCCESS) {
            std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for nv12_to_compressed_nv12_kernel.\n";
            return err;
        }
    }

    // Set kernel arguments for NV12_UBWC to RGBA8888
    err = clSetKernelArg(compressed_nv12_to_rgb_kernel, 0, sizeof(src_nv12_image), &compressed_nv12_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for compressed_nv12_to_rgb_kernel kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compressed_nv12_to_rgb_kernel, 1, sizeof(out_rgba_image), &out_rgba_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for compressed_nv12_to_rgb_kernel kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compressed_nv12_to_rgb_kernel, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for compressed_nv12_to_rgb_kernel kernel." << "\n";
        return err;
    }

    // Run compressed nv12 to rgba
    {
        const size_t work_size[] = {out_rgba_desc.image_width, out_rgba_desc.image_height};
        err = clEnqueueNDRangeKernel(
                command_queue,
                compressed_nv12_to_rgb_kernel,
                2,
                nullptr,
                work_size,
                nullptr,
                0,
                nullptr,
                nullptr
        );
    }
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for compressed_nv12_to_rgb_kernel kernel." << "\n";
        return err;
    }

    err = clFinish(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while finishing the command queue.\n";
        return err;
    }

    /*
     * Step 6: Copy the data out of the buffer for each plane.
     */
    rgba_image_t out_rgba_image_info;
    out_rgba_image_info.width  = out_rgba_desc.image_width;
    out_rgba_image_info.height = out_rgba_desc.image_height;
    out_rgba_image_info.pixels.resize(out_rgba_desc.image_width * out_rgba_image_info.height * 4);

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
        std::cerr << "Error " << err << " mapping dest out_rgba_image buffer." << "\n";
        return err;
    }

    void* locked_ptr;
    rgba_gb->lock(android::GraphicBuffer::USAGE_SW_READ_OFTEN, &locked_ptr);

    // Copies image data from the buffer to the host
    for (uint32_t i = 0; i < out_rgba_desc.image_height; ++i)
    {
        std::memcpy(
                out_rgba_image_info.pixels.data() + i * out_rgba_desc.image_width * 4,
                image_ptr                         + i * row_pitch,
                out_rgba_desc.image_width * 4
        );
    }
    rgba_gb->unlock();

    err = clEnqueueUnmapMemObject(command_queue, out_rgba_image, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest out_rgba_image buffer." << "\n";
        return err;
    }

    err = clFinish(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while finishing the command queue.\n";
        return err;
    }

    save_rgba_image_data(out_image_filename, out_rgba_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_nv12_image);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(compressed_nv12_image);
    clReleaseMemObject(compressed_y_plane);
    clReleaseMemObject(compressed_uv_plane);
    clReleaseMemObject(out_rgba_image);

    return EXIT_SUCCESS;
}
