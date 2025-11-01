//--------------------------------------------------------------------------------------
// File: p010_to_compressed_tp10.cpp
// Desc: This program converts a P010 to compressed TP10
//
// Author:      QUALCOMM
//
//               Copyright (c) 2017, 2021 QUALCOMM Technologies, Inc.
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
    // The image format determines how the 4th argument to vector read operations is interpreted.
    // For YUV images:
    static const int Y_COMPONENT = 0;
    static const int U_COMPONENT = 1;
    static const int V_COMPONENT = 2;
    // This kernel writes from a P010 image (both planes) to a compressed TP10 image.
    __kernel void p010_to_tp10(__read_only  image2d_t src_p010,
                               __write_only image2d_t dest_tp10_y,
                               __write_only image2d_t dest_tp10_uv,
                                            sampler_t sampler)
    {
        const int    wid_x          = get_global_id(0);
        const int    wid_y          = get_global_id(1);
        const int2   read_coord     = (int2)(6 * wid_x, wid_y);
        const int2   y_write_coord  = (int2)(6 * wid_x, wid_y);
        const int2   uv_write_coord = (int2)(3 * wid_x, wid_y / 2);
        const float4 pixels_in[]    = {
            read_imagef(src_p010, sampler, read_coord               ),
            read_imagef(src_p010, sampler, read_coord + (int2)(1, 0)),
            read_imagef(src_p010, sampler, read_coord + (int2)(2, 0)),
            read_imagef(src_p010, sampler, read_coord + (int2)(3, 0)),
            read_imagef(src_p010, sampler, read_coord + (int2)(4, 0)),
            read_imagef(src_p010, sampler, read_coord + (int2)(5, 0)),
            };
        float        y_pixels_out[2][3] = {
            {pixels_in[0].s0, pixels_in[1].s0, pixels_in[2].s0},
            {pixels_in[3].s0, pixels_in[4].s0, pixels_in[5].s0},
            };
        float2       uv_pixels_out[3]   = {
            {pixels_in[0].s1, pixels_in[0].s2},
            {pixels_in[2].s1, pixels_in[2].s2},
            {pixels_in[4].s1, pixels_in[4].s2},
            };
        qcom_write_imagefv_3x1_n10t00(dest_tp10_y, y_write_coord,                y_pixels_out[0]);
        qcom_write_imagefv_3x1_n10t00(dest_tp10_y, y_write_coord + (int2)(3, 0), y_pixels_out[1]);
        if (wid_y % 2 == 0) qcom_write_imagefv_3x1_n10t01(dest_tp10_uv, uv_write_coord, uv_pixels_out);
    }

    // This kernel writes from a compressed TP10 image (both planes) to a P010 image.
    __kernel void tp10_to_p010(__read_only  image2d_t src_image,
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
        float        y_pixels_out[4][4] = {
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
        float2       uv_pixels_out[2][2] = {
           {{u_pixels_in.s3, v_pixels_in.s3}, {u_pixels_in.s2, v_pixels_in.s2}},
           {{u_pixels_in.s0, v_pixels_in.s0}, {u_pixels_in.s1, v_pixels_in.s1}},
           };
        qcom_write_imagefv_2x1_n10p01(dest_p010_uv, uv_write_coord,                uv_pixels_out[0]);
        qcom_write_imagefv_2x1_n10p01(dest_p010_uv, uv_write_coord + (int2)(0, 1), uv_pixels_out[1]);
    }
)";

// Some macros for use with tuples below
#define SRC_IMG(x) std::get<0>((x))
#define DST_IMG(x) std::get<1>((x))
#define WIDTH(x)   std::get<2>((x))
#define HEIGHT(x)  std::get<3>((x))

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Please specify source and output images.\n"
                     "\n"
                     "Usage: " << argv[0] << " <source image data file> <output image data file>\n"
                     "\n"
                     "Demonstrates conversions to and from p010 and compressed tp10 formats.\n"
                     "The input image is compressed and then decompressed, with the result written\n"
                     "to the specified output file for comparison. (The compression is not lossy so\n"
                     "they are identical.)\n"
                     "\n"
                     "Compressed image formats may be saved to disk, however be advised that the format\n"
                     "is specific to each GPU.\n";
        return EXIT_FAILURE;
    }
    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper          wrapper;
    cl_program          program             = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel           p010_to_tp10_kernel = wrapper.make_kernel("p010_to_tp10", program);
    cl_kernel           tp10_to_p010_kernel = wrapper.make_kernel("tp10_to_p010", program);
    cl_context          context             = wrapper.get_context();
    p010_image_t        src_p010_image_info = load_p010_image_data(src_image_filename);
    struct dma_buf_sync buf_sync            = {};
    cl_event            unmap_event         = nullptr;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_other_image"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for P010 image format is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_compressed_image"))
    {
        std::cerr << "Extension cl_qcom_compressed_image needed for reading/writing compressed images is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_vector_image_ops"))
    {
        std::cerr << "Extension cl_qcom_vector_image_ops needed for vector image reads/writes is not supported.\n";
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
     * Step 1: Create suitable DMA buffer-backed CL images. Note that planar formats (like P010) must be read only,
     * but you can write to child images derived from the planes. (See step 2 for deriving child images.)
     */

    cl_image_format src_format;
    src_format.image_channel_order     = CL_QCOM_P010;
    src_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc src_desc;
    std::memset(&src_desc, 0, sizeof(src_desc));
    src_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_desc.image_width  = src_p010_image_info.y_width;
    src_desc.image_height = src_p010_image_info.y_height;

    cl_int err = 0;
    cl_mem_dmabuf_host_ptr src_mem = wrapper.make_buffer_for_yuv_image(src_format, src_desc);
    cl_mem src_image = clCreateImage(
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

    cl_image_format compressed_format;
    compressed_format.image_channel_order     = CL_QCOM_COMPRESSED_TP10;
    compressed_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc compressed_desc;
    std::memset(&compressed_desc, 0, sizeof(compressed_desc));
    compressed_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_desc.image_width  = src_p010_image_info.y_width;
    compressed_desc.image_height = src_p010_image_info.y_height;

    cl_mem_dmabuf_host_ptr compressed_mem = wrapper.make_buffer_for_compressed_image(compressed_format,
                                                                                          compressed_desc);
    cl_mem compressed_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &compressed_format,
            &compressed_desc,
            &compressed_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for compressed image." << "\n";
        return err;
    }

    cl_image_format out_format;
    out_format.image_channel_order     = CL_QCOM_P010;
    out_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc out_desc;
    std::memset(&out_desc, 0, sizeof(out_desc));
    out_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_desc.image_width  = src_p010_image_info.y_width;
    out_desc.image_height = src_p010_image_info.y_height;

    cl_mem_dmabuf_host_ptr out_mem = wrapper.make_buffer_for_yuv_image(out_format, out_desc);
    cl_mem out_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
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
     * Step 2: Separate planar P010 images into their component planes.
     */

    cl_image_format src_y_plane_format;
    src_y_plane_format.image_channel_order     = CL_QCOM_P010_Y;
    src_y_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc src_y_plane_desc;
    std::memset(&src_y_plane_desc, 0, sizeof(src_y_plane_desc));
    src_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_y_plane_desc.image_width  = src_p010_image_info.y_width;
    src_y_plane_desc.image_height = src_p010_image_info.y_height;
    src_y_plane_desc.mem_object   = src_image;

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
    src_uv_plane_format.image_channel_order     = CL_QCOM_P010_UV;
    src_uv_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc src_uv_plane_desc;
    std::memset(&src_uv_plane_desc, 0, sizeof(src_uv_plane_desc));
    src_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    src_uv_plane_desc.image_width  = src_p010_image_info.y_width;
    src_uv_plane_desc.image_height = src_p010_image_info.y_height;
    src_uv_plane_desc.mem_object   = src_image;

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
    compressed_y_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_TP10_Y;
    compressed_y_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc compressed_y_plane_desc;
    std::memset(&compressed_y_plane_desc, 0, sizeof(compressed_y_plane_desc));
    compressed_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    compressed_y_plane_desc.image_width  = compressed_desc.image_width;
    compressed_y_plane_desc.image_height = compressed_desc.image_height;
    compressed_y_plane_desc.mem_object   = compressed_image;

    cl_mem compressed_y_plane = clCreateImage(
            context,
            CL_MEM_READ_WRITE,
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
    compressed_uv_plane_format.image_channel_order     = CL_QCOM_COMPRESSED_TP10_UV;
    compressed_uv_plane_format.image_channel_data_type = CL_QCOM_UNORM_INT10;

    cl_image_desc compressed_uv_plane_desc;
    std::memset(&compressed_uv_plane_desc, 0, sizeof(compressed_uv_plane_desc));
    compressed_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    compressed_uv_plane_desc.image_width  = compressed_desc.image_width;
    compressed_uv_plane_desc.image_height = compressed_desc.image_height;
    compressed_uv_plane_desc.mem_object   = compressed_image;

    cl_mem compressed_uv_plane = clCreateImage(
            context,
            CL_MEM_READ_WRITE,
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
    out_y_plane_desc.image_width  = out_desc.image_width;
    out_y_plane_desc.image_height = out_desc.image_height;
    out_y_plane_desc.mem_object   = out_image;

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
        std::cerr << "Error " << err << " with clCreateImage for destination image y plane." << "\n";
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
    out_uv_plane_desc.image_width  = out_desc.image_width;
    out_uv_plane_desc.image_height = out_desc.image_height;
    out_uv_plane_desc.mem_object   = out_image;

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
        std::cerr << "Error " << err << " with clCreateImage for destination image uv plane." << "\n";
        return err;
    }

    /*
     * Step 3: Copy data to input image planes. Note that for linear P010 images you must observe row alignment
     * restrictions. (You may also write to the DMA buffer directly if you prefer, however using clEnqueueMapImage for
     * a child planar image will return the correct host pointer for the desired plane.)
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if (ioctl(src_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    cl_command_queue command_queue  = wrapper.get_command_queue();
    const size_t     origin[]       = {0, 0, 0};
    const size_t     src_y_region[] = {src_y_plane_desc.image_width, src_y_plane_desc.image_height, 1};
    size_t           row_pitch      = 0;
    unsigned char   *image_ptr      = static_cast<unsigned char *>(clEnqueueMapImage(
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
                src_p010_image_info.y_plane.data() + i * src_y_plane_desc.image_width * 2,
                src_y_plane_desc.image_width * 2
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
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
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
                src_p010_image_info.uv_plane.data() + i * src_uv_plane_desc.image_width * 2,
                src_uv_plane_desc.image_width * 2
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
    if (ioctl(src_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    /*
     * Step 4: Run the kernels
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
     * Step 5: Run the kernels
     */

    err = clSetKernelArg(p010_to_tp10_kernel, 0, sizeof(src_image), &src_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for p010_to_tp10_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(p010_to_tp10_kernel, 1, sizeof(compressed_y_plane), &compressed_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for p010_to_tp10_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(p010_to_tp10_kernel, 2, sizeof(compressed_uv_plane), &compressed_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for p010_to_tp10_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(p010_to_tp10_kernel, 3, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 3 for p010_to_tp10_kernel." << "\n";
        return err;
    }

    const size_t p010_to_tp10_work_size[] = {work_units(src_desc.image_width, 6), src_desc.image_height};
    err = clEnqueueNDRangeKernel(
            command_queue,
            p010_to_tp10_kernel,
            2,
            nullptr,
            p010_to_tp10_work_size,
            nullptr,
            0,
            nullptr,
            nullptr
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for p010_to_tp10_kernel." << "\n";
        return err;
    }

    err = clFlush(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while flushing the command queue.\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_p010_kernel, 0, sizeof(compressed_image), &compressed_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for tp10_to_p010_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_p010_kernel, 1, sizeof(out_y_plane), &out_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for tp10_to_p010_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_p010_kernel, 2, sizeof(out_uv_plane), &out_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for tp10_to_p010_kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(tp10_to_p010_kernel, 3, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 3 for tp10_to_p010_kernel." << "\n";
        return err;
    }

    const size_t tp10_to_p010_work_size[] = {work_units(src_desc.image_width, 4), work_units(src_desc.image_height, 4)};
    err = clEnqueueNDRangeKernel(
            command_queue,
            tp10_to_p010_kernel,
            2,
            nullptr,
            tp10_to_p010_work_size,
            nullptr,
            0,
            nullptr,
            nullptr
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for tp10_to_p010_kernel." << "\n";
        return err;
    }

    err = clFlush(command_queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while flushing the command queue.\n";
        return err;
    }

    /*
     * Step 6: Copy the data out of the DMA buffer for each plane.
     */

    p010_image_t out_image_info;
    out_image_info.y_width  = out_desc.image_width;
    out_image_info.y_height = out_desc.image_height;
    out_image_info.y_plane.resize(out_image_info.y_width * out_image_info.y_height * 2);
    out_image_info.uv_plane.resize(out_image_info.y_width * out_image_info.y_height);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if (ioctl(out_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

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

    // Copies image data from the DMA buffer to the host.
    for (uint32_t i = 0; i < out_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                out_image_info.y_plane.data() + i * out_y_plane_desc.image_width * 2,
                image_ptr                     + i * row_pitch,
                out_y_plane_desc.image_width * 2
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_y_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image y-plane buffer." << "\n";
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
        std::cerr << "Error " << err << " mapping dest image uv-plane buffer for reading." << "\n";
        return err;
    }

    // Copies image data from the DMA buffer to the host.
    for (uint32_t i = 0; i < out_uv_plane_desc.image_height / 2; ++i)
    {
        std::memcpy(
                out_image_info.uv_plane.data() + i * out_uv_plane_desc.image_width * 2,
                image_ptr                      + i * row_pitch,
                out_uv_plane_desc.image_width * 2
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_uv_plane, image_ptr, 0, nullptr, &unmap_event);
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
    if (ioctl(out_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    save_p010_image_data(out_image_filename, out_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_image);
    clReleaseMemObject(compressed_uv_plane);
    clReleaseMemObject(compressed_y_plane);
    clReleaseMemObject(compressed_image);
    clReleaseMemObject(out_uv_plane);
    clReleaseMemObject(out_y_plane);
    clReleaseMemObject(out_image);

    return EXIT_SUCCESS;
}