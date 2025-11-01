//--------------------------------------------------------------------------------------
// File: compressed_nv12_to_p010.cpp
// Desc: This program converts compressed NV12 to P010
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
    compressed_nv12_to_p010(__read_only  image2d_t input_compressed_nv12,
                            __write_only image2d_t out_p010_y,
                            __write_only image2d_t out_p010_uv,
                                         sampler_t sampler)
    {
        int2   coord;
        float4 in;
        float4 out;
        float4 uv;
        float4 ys;

        coord.x = get_global_id(0) * 2;
        coord.y = get_global_id(1) * 2;

        ys    = qcom_read_imagef_2x2(input_compressed_nv12, sampler, convert_float2(coord) + 0.5f, 0);
        uv.xy = read_imagef(input_compressed_nv12, sampler, coord).yz;
        write_imagef(out_p010_y, coord, (float4)(ys.s3));
        coord += (int2)(1, 0);
        write_imagef(out_p010_y, coord, (float4)(ys.s2));
        coord += (int2)(0, 1);
        write_imagef(out_p010_y, coord, (float4)(ys.s1));
        coord -= (int2)(1, 0);
        write_imagef(out_p010_y, coord, (float4)(ys.s0));
        write_imagef(out_p010_uv, coord / 2, uv);
    }

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
)";

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <img data file> <out img data file> \n"
                     "Input image file data should be in format CL_QCOM_NV12 / CL_UNORM_INT8\n"
                     "Demonstrates conversions from compressed NV12 to P010\n"
                     "First the input uncompressed NV12 image is compressed, and then the conversion is performed.\n";
        return EXIT_FAILURE;
    }

    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper          wrapper;
    cl_program          program                        = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel           compressed_nv12_to_p010_kernel = wrapper.make_kernel("compressed_nv12_to_p010", program);
    cl_kernel           compress_kernel                = wrapper.make_kernel("compress", program);
    cl_context          context                        = wrapper.get_context();
    nv12_image_t        src_nv12_image_info            = load_nv12_image_data(src_image_filename);
    struct dma_buf_sync buf_sync                       = {};
    cl_event            unmap_event                    = nullptr;

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
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for DMA buffer-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for DMA buffer-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create suitable dmabuf buffer-backed CL images. Note that planar formats (like NV12) must be read only,
     * but you can write to child images derived from the planes. (See step 2 for deriving child images.)
     */
    cl_image_format src_nv12_format;
    src_nv12_format.image_channel_order     = CL_QCOM_NV12;
    src_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_nv12_desc;
    std::memset(&src_nv12_desc, 0, sizeof(src_nv12_desc));
    src_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_nv12_desc.image_width  = src_nv12_image_info.y_width;
    src_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_int err = 0;
    cl_mem_dmabuf_host_ptr src_nv12_mem = wrapper.make_buffer_for_yuv_image(src_nv12_format, src_nv12_desc);
    cl_mem src_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &src_nv12_format,
            &src_nv12_desc,
            &src_nv12_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image." << "\n";
        return err;
    }

    cl_image_format compressed_nv12_format;
    compressed_nv12_format.image_channel_order     = CL_QCOM_COMPRESSED_NV12;
    compressed_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc compressed_nv12_desc;
    std::memset(&compressed_nv12_desc, 0, sizeof(compressed_nv12_desc));
    compressed_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_nv12_desc.image_width  = src_nv12_image_info.y_width;
    compressed_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_mem_dmabuf_host_ptr compressed_nv12_mem = wrapper.make_buffer_for_compressed_image(compressed_nv12_format, compressed_nv12_desc);
    cl_mem compressed_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &compressed_nv12_format,
            &compressed_nv12_desc,
            &compressed_nv12_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source image." << "\n";
        return err;
    }

    cl_image_format out_p010_format;
    out_p010_format.image_channel_order     = CL_QCOM_P010;
    out_p010_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_p010_desc;
    std::memset(&out_p010_desc, 0, sizeof(out_p010_desc));
    out_p010_desc.image_type             = CL_MEM_OBJECT_IMAGE2D;
    out_p010_desc.image_width            = src_nv12_image_info.y_width;
    out_p010_desc.image_height           = src_nv12_image_info.y_height;
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
        return err;
    }

    /*
     * Step 2: Separate planar images into their component planes.
     */
    cl_image_format src_y_plane_format;
    src_y_plane_format.image_channel_order     = CL_QCOM_NV12_Y;
    src_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_y_plane_desc;
    std::memset(&src_y_plane_desc, 0, sizeof(src_y_plane_desc));
    src_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_y_plane_desc.image_width  = src_nv12_desc.image_width;
    src_y_plane_desc.image_height = src_nv12_desc.image_height;
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

    cl_image_desc src_uv_plane_desc;
    std::memset(&src_uv_plane_desc, 0, sizeof(src_uv_plane_desc));
    src_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane are halved in each dimension.
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

    cl_image_format compressed_y_plane_format;
    compressed_y_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_NV12_Y;
    compressed_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc compressed_y_plane_desc;
    std::memset(&compressed_y_plane_desc, 0, sizeof(compressed_y_plane_desc));
    compressed_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_y_plane_desc.image_width  = compressed_nv12_desc.image_width;
    compressed_y_plane_desc.image_height = compressed_nv12_desc.image_height;
    compressed_y_plane_desc.mem_object   = compressed_nv12_image;

    cl_mem compressed_y_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &compressed_y_plane_format,
            &compressed_y_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image y plane." << "\n";
        return err;
    }

    cl_image_format compressed_uv_plane_format;
    compressed_uv_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_NV12_UV;
    compressed_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc compressed_uv_plane_desc;
    std::memset(&compressed_uv_plane_desc, 0, sizeof(compressed_uv_plane_desc));
    compressed_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane are halved in each dimension.
    compressed_uv_plane_desc.image_width  = src_nv12_image_info.y_width;
    compressed_uv_plane_desc.image_height = src_nv12_image_info.y_height;
    compressed_uv_plane_desc.mem_object   = compressed_nv12_image;

    cl_mem compressed_uv_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &compressed_uv_plane_format,
            &compressed_uv_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image uv plane." << "\n";
        return err;
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
            CL_MEM_WRITE_ONLY,
            &out_y_plane_format,
            &out_y_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image y plane." << "\n";
        return err;
    }

    cl_image_format out_uv_plane_format;
    out_uv_plane_format.image_channel_order     = CL_QCOM_P010_UV;
    out_uv_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_uv_plane_desc;
    std::memset(&out_uv_plane_desc, 0, sizeof(out_uv_plane_desc));
    out_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane are halved in each dimension.
    out_uv_plane_desc.image_width  = out_p010_desc.image_width;
    out_uv_plane_desc.image_height = out_p010_desc.image_height;
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
        return err;
    }

    /*
     * Step 3: Copy data to input image planes. Note that for linear NV12 images you must observe row alignment
     * restrictions. (You may also write to the dmabuf buffer directly if you prefer, however using clEnqueueMapImage for
     * a child planar image will return the correct host pointer for the desired plane.)
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if (ioctl(src_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
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
    // Copies image data to the dmabuf buffer from the host
    for (uint32_t i = 0; i < src_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                image_ptr                          + i * row_pitch,
                src_nv12_image_info.y_plane.data() + i * src_y_plane_desc.image_width,
                src_y_plane_desc.image_width
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
    // Copies image data to the dmabuf buffer from the host
    for (uint32_t i = 0; i < src_uv_plane_desc.image_height / 2; ++i)
    {
        std::memcpy(
                image_ptr                           + i * row_pitch,
                src_nv12_image_info.uv_plane.data() + i * src_uv_plane_desc.image_width,
                src_uv_plane_desc.image_width
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
    if (ioctl(src_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
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
        return err;
    }

    /*
     * Step 5: Run the kernels.
     */
    size_t work_size[] = {0, 0};

    err = clSetKernelArg(compress_kernel, 0, sizeof(src_y_plane), &src_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for compress_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compress_kernel, 1, sizeof(compressed_y_plane), &compressed_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for compress_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compress_kernel, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for compress_kernel." << "\n";
        return err;
    }

    work_size[0] = src_nv12_desc.image_width;
    work_size[1] = src_nv12_desc.image_height;
    err = clEnqueueNDRangeKernel(
            command_queue,
            compress_kernel,
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
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for compress_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compress_kernel, 0, sizeof(src_uv_plane), &src_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for compress_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compress_kernel, 1, sizeof(compressed_uv_plane), &compressed_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for compress_kernel." << "\n";
        return err;
    }

    work_size[0] = src_nv12_desc.image_width / 2;
    work_size[1] = src_nv12_desc.image_height / 2;
    err = clEnqueueNDRangeKernel(
            command_queue,
            compress_kernel,
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
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for compress_kernel." << "\n";
        return err;
    }

    err = clFlush(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while flushing the command queue.\n";
        return err;
    }

    err = clSetKernelArg(compressed_nv12_to_p010_kernel, 0, sizeof(compressed_nv12_image), &compressed_nv12_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for compressed_nv12_to_p010_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compressed_nv12_to_p010_kernel, 1, sizeof(out_y_plane), &out_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for compressed_nv12_to_p010_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compressed_nv12_to_p010_kernel, 2, sizeof(out_uv_plane), &out_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for compressed_nv12_to_p010_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(compressed_nv12_to_p010_kernel, 3, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 3 for compressed_nv12_to_p010_kernel." << "\n";
        return err;
    }

    work_size[0] = out_p010_desc.image_width / 2;
    work_size[1] = out_p010_desc.image_height / 2;
    err = clEnqueueNDRangeKernel(
            command_queue,
            compressed_nv12_to_p010_kernel,
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
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for compressed_nv12_to_p010_kernel." << "\n";
        return err;
    }

    err = clFinish(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while finishing the command queue.\n";
        return err;
    }

    /*
     * Step 6: Copy the data out of the dmabuf buffer for each plane.
     */

    p010_image_t out_p010_image_info;
    out_p010_image_info.y_width  = out_p010_desc.image_width;
    out_p010_image_info.y_height = out_p010_desc.image_height;
    out_p010_image_info.y_plane.resize(out_p010_image_info.y_width * out_p010_image_info.y_height * 2);
    out_p010_image_info.uv_plane.resize(out_p010_image_info.y_width * out_p010_image_info.y_height);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if (ioctl(out_p010_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    const size_t out_y_region[] = {out_p010_desc.image_width, out_p010_desc.image_height, 1};
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
        std::cerr << "Error " << err << " mapping dest image y-plane buffer for reading." << "\n";
        return err;
    }
    // Copies image data from the dmabuf buffer to the host
    for (uint32_t i = 0; i < out_p010_desc.image_height; ++i)
    {
        std::memcpy(
                out_p010_image_info.y_plane.data() + i * out_p010_desc.image_width * 2,
                image_ptr                          + i * row_pitch,
                out_p010_desc.image_width * 2
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_y_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image y plane." << "\n";
        return err;
    }

    const size_t out_uv_region[] = {out_p010_desc.image_width / 2, out_p010_desc.image_height / 2, 1};
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
        std::cerr << "Error " << err << " mapping dest image uv-plane buffer for reading." << "\n";
        return err;
    }
    // Copies image data from the dmabuf buffer to the host
    for (uint32_t i = 0; i < out_p010_desc.image_height / 2; ++i)
    {
        std::memcpy(
                out_p010_image_info.uv_plane.data() + i * out_p010_desc.image_width * 2,
                image_ptr                           + i * row_pitch,
                out_p010_desc.image_width * 2
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_uv_plane, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image uv plane." << "\n";
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
    if (ioctl(out_p010_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    save_p010_image_data(out_image_filename, out_p010_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_nv12_image);
    clReleaseMemObject(compressed_uv_plane);
    clReleaseMemObject(compressed_y_plane);
    clReleaseMemObject(compressed_nv12_image);
    clReleaseMemObject(out_uv_plane);
    clReleaseMemObject(out_y_plane);
    clReleaseMemObject(out_p010_image);

    return EXIT_SUCCESS;
}