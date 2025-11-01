//--------------------------------------------------------------------------------------
// File: nv21_to_compressed_nv12.cpp
// Desc: This program converts linear NV21 to compressed NV12
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

// Android Native Buffer header
#include <ui/GraphicBuffer.h>

/**
 * \brief Makes an Android Native Buffer of the specified size.
 *
 * @param width [in] - width of buffer in Pixels
 * @param height [in] - height of buffer in Pixels
 * @param pGraphicBuffer [out] - GraphicBuffer structure created for Android Native Buffer
 * @return anbMem - Android Native Buffer
 */
cl_mem_android_native_buffer_host_ptr
make_anb_buffer(size_t width, size_t height, android::sp<android::GraphicBuffer>& pGraphicBuffer)
{
    pGraphicBuffer = new android::GraphicBuffer();
    static const unsigned int usage = GRALLOC_USAGE_SW_WRITE_OFTEN |
                                      GRALLOC_USAGE_HW_TEXTURE     |
                                      GRALLOC_USAGE_SW_READ_OFTEN  |
                                      GRALLOC_USAGE_HW_RENDER;
    pGraphicBuffer->reallocate(width, height, HAL_PIXEL_FORMAT_RGBA_8888, 1, usage);
    android::status_t error = pGraphicBuffer->initCheck();
    if (error != android::NO_ERROR)
    {
        std::cerr << "Error allocating graphics buffer.\n";
        std::exit(EXIT_FAILURE);
    }

    cl_mem_android_native_buffer_host_ptr anbMem = {{0}};
    anbMem.ext_host_ptr.allocation_type = CL_MEM_ANDROID_NATIVE_BUFFER_HOST_PTR_QCOM;
    anbMem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_WRITEBACK_QCOM;
    anbMem.anb_ptr = pGraphicBuffer->getNativeBuffer();
    return anbMem;
}

static const char *PROGRAM_SOURCE = R"(
    static const int GROUP_SIZE_DIMS = 2;       // based off work dims
    __kernel void
    nv21_to_compressed_nv12(__read_only  image2d_t input_nv21,
                            __write_only image2d_t out_compressed_nv12_y,
                            __write_only image2d_t out_compressed_nv12_uv,
                                         sampler_t sampler)
    {
        int2   coord;
        float4 uv_component;
        float4 y_component;

        coord.x = get_global_id(0) * GROUP_SIZE_DIMS;
        coord.y = get_global_id(1) * GROUP_SIZE_DIMS;

        // Swizzle the UV values compared to the NV21 kernel.
        uv_component.xy = read_imagef(input_nv21, sampler, coord).zy;
        write_imagef(out_compressed_nv12_uv, coord / GROUP_SIZE_DIMS, uv_component);

        // Get Y components
        y_component = read_imagef(input_nv21, sampler, coord);
        write_imagef(out_compressed_nv12_y, coord, y_component);

        y_component = read_imagef(input_nv21, sampler, coord + (int2)(1, 0));
        write_imagef(out_compressed_nv12_y, coord + (int2)(1, 0), y_component);

        y_component = read_imagef(input_nv21, sampler, coord + (int2)(1, 1));
        write_imagef(out_compressed_nv12_y, coord + (int2)(1, 1), y_component);

        y_component = read_imagef(input_nv21, sampler, coord + (int2)(0, 1));
        write_imagef(out_compressed_nv12_y, coord + (int2)(0, 1), y_component);
    }
    __kernel void decompress(__read_only  image2d_t src_plane,
                             __write_only image2d_t dst_plane,
                                          sampler_t sampler)
    {
        const int2   coord = (int2)(get_global_id(0), get_global_id(1));
        const float4 pixel = read_imagef(src_plane, sampler, coord);
        write_imagef(dst_plane, coord, pixel);
    }
)";

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Please specify source and output images.\n"
                     "\n"
                     "Usage: " << argv[0] << " <src img data file> <out img data file> \n"
                     "\n"
                     "Demonstrates conversions from linear NV21 to compressed NV12 formats.\n"
                     "The input NV21 image is converted to compressed NV12, then the\n"
                     "compressed NV12 image is decompressed and written to the output file\n"
                     "for comparison.\n"
                     "\n"
                     "Compressed image formats may be saved to disk, however be advised that the format\n"
                     "is specific to each GPU.\n";
        return EXIT_FAILURE;
    }

    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper            wrapper;
    cl_int                err                            = 0;
    cl_program            program                        = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel             nv21_to_compressed_nv12_kernel = wrapper.make_kernel("nv21_to_compressed_nv12", program);
    cl_kernel             decompress_kernel              = wrapper.make_kernel("decompress", program);
    cl_context            context                        = wrapper.get_context();
    nv12_image_t          src_nv21_image_info            = load_nv12_image_data(src_image_filename);
    static const uint32_t MAX_DIMENSION                  = 2048;

    // The size of the Android Native Buffer will be sufficient to hold an image where each dimension is <= 2048.
    // This is a loose upper bound only, however the general calculation is not within the scope of these examples.
    if(src_nv21_image_info.y_height > MAX_DIMENSION || src_nv21_image_info.y_width > MAX_DIMENSION)
    {
        std::cerr << "Dimensions of the image are too large, consider increasing the MAX_DIMENSIONS for the "
                     "Android Native Buffer allocation.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_other_image"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for NV12 image format is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_ext_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for Android Native Buffer-backed"
                     " images is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_ion_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_ion_host_ptr needed for Android Native Buffer-backed"
                     " images is not supported.\n";
        return EXIT_FAILURE;
    }

    if(!wrapper.check_extension_support("cl_qcom_android_native_buffer_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_android_native_buffer_host_ptr for Android Native Buffer-backed"
                     " images is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create suitable Android Native Buffer backed OpenCL images
     */

    cl_image_format src_nv21_format;
    src_nv21_format.image_channel_order     = CL_QCOM_NV12; // use NV12 until native NV21 support
    src_nv21_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_nv21_desc = {0};
    src_nv21_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_nv21_desc.image_width  = src_nv21_image_info.y_width;
    src_nv21_desc.image_height = src_nv21_image_info.y_height;

    android::sp<android::GraphicBuffer> pGraphicBufferInput = nullptr;
    // The size of this buffer will be sufficient to hold an image where each dimension is <= MAX_DIMENSION.
    cl_mem_android_native_buffer_host_ptr src_nv21_anb_mem = make_anb_buffer(MAX_DIMENSION, MAX_DIMENSION, pGraphicBufferInput);
    cl_mem src_nv21_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &src_nv21_format,
            &src_nv21_desc,
            &src_nv21_anb_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for input image.\n";
        return err;
    }

    cl_image_format compressed_nv12_format;
    compressed_nv12_format.image_channel_order     = CL_QCOM_COMPRESSED_NV12;
    compressed_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc compressed_nv12_desc = {0};
    compressed_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_nv12_desc.image_width  = src_nv21_image_info.y_width;
    compressed_nv12_desc.image_height = src_nv21_image_info.y_height;

    android::sp<android::GraphicBuffer> pGraphicBufferCompressed = nullptr;
    // The size of this buffer will be sufficient to hold an image where each dimension is <= MAX_DIMENSION.
    cl_mem_android_native_buffer_host_ptr compressed_nv12_anb_mem = make_anb_buffer(MAX_DIMENSION, MAX_DIMENSION, pGraphicBufferCompressed);
    cl_mem compressed_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &compressed_nv12_format,
            &compressed_nv12_desc,
            &compressed_nv12_anb_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image.\n";
        return err;
    }

    cl_image_format out_nv12_format;
    out_nv12_format.image_channel_order     = CL_QCOM_NV12;
    out_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_nv12_desc = {0};
    out_nv12_desc.image_type             = CL_MEM_OBJECT_IMAGE2D;
    out_nv12_desc.image_width            = src_nv21_image_info.y_width;
    out_nv12_desc.image_height           = src_nv21_image_info.y_height;
    android::sp<android::GraphicBuffer> pGraphicBufferOutput = nullptr;
    // The size of this buffer will be sufficient to hold an image where each dimension is <= MAX_DIMENSION.
    cl_mem_android_native_buffer_host_ptr out_nv12_anb_mem = make_anb_buffer(MAX_DIMENSION, MAX_DIMENSION, pGraphicBufferOutput);
    cl_mem out_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_nv12_format,
            &out_nv12_desc,
            &out_nv12_anb_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image.\n";
        return err;
    }

    /*
     * Step 2: Separate planar images into their component planes.
     */

    cl_image_format src_y_plane_format;
    src_y_plane_format.image_channel_order     = CL_QCOM_NV12_Y;
    src_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_y_plane_desc = {0};
    src_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_y_plane_desc.image_width  = src_nv21_desc.image_width;
    src_y_plane_desc.image_height = src_nv21_desc.image_height;
    src_y_plane_desc.mem_object   = src_nv21_image;

    cl_mem src_y_plane = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            &src_y_plane_format,
            &src_y_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for input image y plane.\n";
        return err;
    }

    cl_image_format src_uv_plane_format;
    src_uv_plane_format.image_channel_order     = CL_QCOM_NV12_UV;
    src_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_uv_plane_desc = {0};
    src_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane are halved in each dimension.
    src_uv_plane_desc.image_width  = src_nv21_desc.image_width;
    src_uv_plane_desc.image_height = src_nv21_desc.image_height;
    src_uv_plane_desc.mem_object   = src_nv21_image;

    cl_mem src_uv_plane = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            &src_uv_plane_format,
            &src_uv_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for input image uv plane.\n";
        return err;
    }

    cl_image_format compressed_y_plane_format;
    compressed_y_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_NV12_Y;
    compressed_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc compressed_y_plane_desc = {0};
    compressed_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_y_plane_desc.image_width  = compressed_nv12_desc.image_width;
    compressed_y_plane_desc.image_height = compressed_nv12_desc.image_height;
    compressed_y_plane_desc.mem_object   = compressed_nv12_image;

    cl_mem compressed_y_plane = clCreateImage(
            context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
            &compressed_y_plane_format,
            &compressed_y_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image y plane.\n";
        return err;
    }

    cl_image_format compressed_uv_plane_format;
    compressed_uv_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_NV12_UV;
    compressed_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc compressed_uv_plane_desc = {0};
    compressed_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane are halved in each dimension.
    compressed_uv_plane_desc.image_width  = compressed_nv12_desc.image_width;
    compressed_uv_plane_desc.image_height = compressed_nv12_desc.image_height;
    compressed_uv_plane_desc.mem_object   = compressed_nv12_image;

    cl_mem compressed_uv_plane = clCreateImage(
            context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
            &compressed_uv_plane_format,
            &compressed_uv_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image uv plane.\n";
        return err;
    }

    cl_image_format out_y_plane_format;
    out_y_plane_format.image_channel_order     = CL_QCOM_NV12_Y;
    out_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_y_plane_desc = {0};
    out_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_y_plane_desc.image_width  = out_nv12_desc.image_width;
    out_y_plane_desc.image_height = out_nv12_desc.image_height;
    out_y_plane_desc.mem_object   = out_nv12_image;

    cl_mem out_y_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
            &out_y_plane_format,
            &out_y_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image y plane.\n";
        return err;
    }

    cl_image_format out_uv_plane_format;
    out_uv_plane_format.image_channel_order     = CL_QCOM_NV12_UV;
    out_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_uv_plane_desc = {0};
    out_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_uv_plane_desc.image_width  = out_nv12_desc.image_width;
    out_uv_plane_desc.image_height = out_nv12_desc.image_height;
    out_uv_plane_desc.mem_object   = out_nv12_image;

    cl_mem out_uv_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
            &out_uv_plane_format,
            &out_uv_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image uv plane.\n";
        return err;
    }

    /*
     * Step 3: Copy data to input image planes. Note that for linear NV21 images you must observe row alignment
     * restrictions. (You may also write to the Ion buffer directly if you prefer, however using clEnqueueMapImage for
     * a child planar image will return the correct host pointer for the desired plane.)
     */

    void *lock = nullptr;
    cl_command_queue command_queue  = wrapper.get_command_queue();
    const size_t     origin[]       = {0, 0, 0};

    const size_t     src_y_region[] = {src_y_plane_desc.image_width, src_y_plane_desc.image_height, 1};
    size_t           row_pitch      = 0;  // in bytes
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
    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping source image y-plane buffer for writing.\n";
        return err;
    }
    // Copies image data to the Android Native Buffer from the host
    android::status_t graphic_buffer_error = {0};
    graphic_buffer_error = pGraphicBufferInput->lock(android::GraphicBuffer::USAGE_SW_WRITE_OFTEN, &lock);
    if (graphic_buffer_error != android::NO_ERROR)
    {
        std::cerr << "Error locking pGraphicBufferInput for Y plane copy.\n";
        std::exit(EXIT_FAILURE);
    }
    for (uint32_t i = 0; i < src_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                image_ptr                          + i * row_pitch,
                src_nv21_image_info.y_plane.data() + i * src_y_plane_desc.image_width,
                src_y_plane_desc.image_width
        );
    }
    pGraphicBufferInput->unlock();

    err = clEnqueueUnmapMemObject(command_queue, src_y_plane, image_ptr, 0, nullptr, nullptr);
    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer.\n";
        return err;
    }

    // there are 2 components per pixel (U and V)
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
        std::cerr << "Error " << err << " mapping source image uv-plane buffer for writing.\n";
        return err;
    }
    // Copies image data to the Android Native Buffer from the host
    graphic_buffer_error = pGraphicBufferInput->lock(android::GraphicBuffer::USAGE_SW_WRITE_OFTEN, &lock);
    if (graphic_buffer_error != android::NO_ERROR)
    {
        std::cerr << "Error locking pGraphicBufferInput for UV plane copy.\n";
        std::exit(EXIT_FAILURE);
    }
    for (uint32_t i = 0; i < src_uv_plane_desc.image_height / 2; ++i)
    {
        std::memcpy(
                image_ptr                           + i * row_pitch,
                src_nv21_image_info.uv_plane.data() + i * src_uv_plane_desc.image_width,
                src_uv_plane_desc.image_width
        );
    }
    pGraphicBufferInput->unlock();

    err = clEnqueueUnmapMemObject(command_queue, src_uv_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image uv-plane data buffer.\n";
        return err;
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
        std::cerr << "Error " << err << " with clCreateSampler.\n";
        return err;
    }

    /*
     * Step 5: Run the kernels.
     */

    size_t work_size[] = {0, 0};

    // Run the NV21   ---- >   Compressed NV12    kernel
    err = clSetKernelArg(nv21_to_compressed_nv12_kernel, 0, sizeof(src_nv21_image), &src_nv21_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for nv21_to_compressed_nv12_kernel.\n";
        return err;
    }

    err = clSetKernelArg(nv21_to_compressed_nv12_kernel, 1, sizeof(compressed_y_plane), &compressed_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for nv21_to_compressed_nv12_kernel.\n";
        return err;
    }

    err = clSetKernelArg(nv21_to_compressed_nv12_kernel, 2, sizeof(compressed_uv_plane), &compressed_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for nv21_to_compressed_nv12_kernel.\n";
        return err;
    }

    err = clSetKernelArg(nv21_to_compressed_nv12_kernel, 3, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 3 for nv21_to_compressed_nv12_kernel.\n";
        return err;
    }

    // run the kernel in blocks of 2x2 YUV values
    work_size[0] = compressed_nv12_desc.image_width / 2;
    work_size[1] = compressed_nv12_desc.image_height / 2;
    err = clEnqueueNDRangeKernel(
            command_queue,
            nv21_to_compressed_nv12_kernel,
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
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for nv21_to_compressed_nv12_kernel.\n";
        return err;
    }

    err = clFlush(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while flushing the command queue.\n";
        return err;
    }

    // Run the decompress kernel on Y plane
    err = clSetKernelArg(decompress_kernel, 0, sizeof(compressed_y_plane), &compressed_y_plane);
    if(err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for decompress_kernel.\n";
        return err;
    }
    err = clSetKernelArg(decompress_kernel, 1, sizeof(out_y_plane), &out_y_plane);
    if(err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for decompress_kernel.\n";
        return err;
    }
    err = clSetKernelArg(decompress_kernel, 2, sizeof(sampler), &sampler);
    if(err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for decompress_kernel.\n";
        return err;
    }

    work_size[0] = compressed_nv12_desc.image_width;
    work_size[1] = compressed_nv12_desc.image_height;
    err = clEnqueueNDRangeKernel(
            command_queue,
            decompress_kernel,
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
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for decompress_kernel.\n";
        return err;
    }
    // Run the decompress kernel on UV plane
    err = clSetKernelArg(decompress_kernel, 0, sizeof(compressed_uv_plane), &compressed_uv_plane);
    if(err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for decompress_kernel.\n";
        return err;
    }
    err = clSetKernelArg(decompress_kernel, 1, sizeof(out_uv_plane), &out_uv_plane);
    if(err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for decompress_kernel.\n";
        return err;
    }

    // there are 2 components per pixel (U and V)
    work_size[0] = compressed_nv12_desc.image_width / 2;
    work_size[1] = compressed_nv12_desc.image_height / 2;
    err = clEnqueueNDRangeKernel(
            command_queue,
            decompress_kernel,
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
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for decompress_kernel.\n";
        return err;
    }

    /*
     * Step 6: Copy the data out of the Android Native Buffer for each plane.
     */

    nv12_image_t out_nv12_image_info;
    out_nv12_image_info.y_width  = out_nv12_desc.image_width;
    out_nv12_image_info.y_height = out_nv12_desc.image_height;
    out_nv12_image_info.y_plane.resize(out_nv12_image_info.y_width * out_nv12_image_info.y_height);
    // Height of UV planes is half the total image height
    out_nv12_image_info.uv_plane.resize(out_nv12_image_info.y_width * out_nv12_image_info.y_height / 2);

    const size_t out_y_region[] = {out_nv12_desc.image_width, out_nv12_desc.image_height, 1};
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
        std::cerr << "Error " << err << " mapping dest image y-plane buffer for reading.\n";
        return err;
    }

    // Copies Y plane image data from the Android Native Buffer to the host
    graphic_buffer_error = pGraphicBufferOutput->lock(android::GraphicBuffer::USAGE_SW_READ_OFTEN, &lock);
    if (graphic_buffer_error != android::NO_ERROR)
    {
        std::cerr << "Error locking pGraphicBufferOutput for Y plane copy.\n";
        std::exit(EXIT_FAILURE);
    }
    for (uint32_t i = 0; i < out_nv12_desc.image_height; ++i)
    {
        std::memcpy(
                out_nv12_image_info.y_plane.data() + i * out_nv12_desc.image_width,
                image_ptr                          + i * row_pitch,
                out_nv12_desc.image_width
        );
    }
    pGraphicBufferOutput->unlock();

    err = clEnqueueUnmapMemObject(command_queue, out_y_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image y plane.\n";
        return err;
    }

    // there are 2 components per pixel (U and V)
    const size_t out_uv_region[] = {out_nv12_desc.image_width / 2, out_nv12_desc.image_height / 2, 1};
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
        std::cerr << "Error " << err << " mapping dest image uv-plane buffer for reading.\n";
        return err;
    }

    // Copies UV plane image data from the Android Native Buffer to the host
    graphic_buffer_error = pGraphicBufferOutput->lock(android::GraphicBuffer::USAGE_SW_READ_OFTEN, &lock);
    if (graphic_buffer_error != android::NO_ERROR)
    {
        std::cerr << "Error locking pGraphicBufferOutput for UV plane copy.\n";
        std::exit(EXIT_FAILURE);
    }
    // Height of UV plane is half the total image height
    for (uint32_t i = 0; i < out_nv12_desc.image_height / 2; ++i)
    {
        std::memcpy(
                out_nv12_image_info.uv_plane.data() + i * out_nv12_desc.image_width,
                image_ptr                           + i * row_pitch,
                out_nv12_desc.image_width
        );
    }
    pGraphicBufferOutput->unlock();

    err = clEnqueueUnmapMemObject(command_queue, out_uv_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image uv plane.\n";
        return err;
    }

    err = clFinish(command_queue);
    if(err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " calling clFinish on command_queue\n";
        return err;
    }
    save_nv12_image_data(out_image_filename, out_nv12_image_info);

    // Clean up OpenCL resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_nv21_image);
    clReleaseMemObject(compressed_uv_plane);
    clReleaseMemObject(compressed_y_plane);
    clReleaseMemObject(compressed_nv12_image);
    clReleaseMemObject(out_uv_plane);
    clReleaseMemObject(out_y_plane);
    clReleaseMemObject(out_nv12_image);

    return EXIT_SUCCESS;
}