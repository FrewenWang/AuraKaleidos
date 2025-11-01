//--------------------------------------------------------------------------------------
// File: nv12_readwrite_image.cpp
// Desc: Demonstrates use of optimized YUV ReadWrite using Qualcomm extensions to OpenCL.
//       The input NV12 image is read in 4X1 fashion in Y-Plane and 2X1 fashion in UV-Plane.
//       The files are written to output files for comparison.
//
// Author:      QUALCOMM
//
//               Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <cstdlib>
#include <iostream>
#include <iomanip>

// Project includes
#include "util/cl_wrapper.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

static const int CL_IMAGE_WIDTH_ALIGNMENT_IN_ELEMENTS_PLANAR = 512;

// Kernel for copying from planar_Y to planar_Y using VectorWrite4x1.
static const char* programCopyVectorWrite4x1_UNORM8_Y = R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void write_4x1_unorm8_y2y (  __read_only image2d_t   srcImage,
                                          __write_only image2d_t  dstImage,
                                          sampler_t               sampler)
    {
        int     wid_x               = get_global_id(0);
        int     wid_y               = get_global_id(1);
        half    packed_pixel_y[4]   = {0};
        int2    coord_packed        = (int2)(4 * wid_x, wid_y);
        int2    coord_y0            = (int2)(4 * wid_x, wid_y);
        int2    coord_y1            = (int2)(4 * wid_x + 1, wid_y);
        int2    coord_y2            = (int2)(4 * wid_x + 2, wid_y);
        int2    coord_y3            = (int2)(4 * wid_x + 3, wid_y);
        half4   read_pixel_y0       = read_imageh(srcImage, sampler, coord_y0);
        half4   read_pixel_y1       = read_imageh(srcImage, sampler, coord_y1);
        half4   read_pixel_y2       = read_imageh(srcImage, sampler, coord_y2);
        half4   read_pixel_y3       = read_imageh(srcImage, sampler, coord_y3);
        packed_pixel_y[0]           = read_pixel_y0.x;
        packed_pixel_y[1]           = read_pixel_y1.x;
        packed_pixel_y[2]           = read_pixel_y2.x;
        packed_pixel_y[3]           = read_pixel_y3.x;
        qcom_write_imagehv_4x1_n8n00(dstImage, coord_packed, packed_pixel_y);
    }
)";

// Kernel for copying from planar_UV to planar_UV using VectorWrite2x1.
static const char* programCopyVectorWrite2x1_UNORM8_UV = R"(
    __kernel void write_2x1_unorm8_uv2uv (__read_only image2d_t   srcImage,
                                          __write_only image2d_t  dstImage,
                                          sampler_t               sampler)
    {
        int     wid_x               = get_global_id(0);
        int     wid_y               = get_global_id(1);
        float2  packed_pixel_uv[2]  = {(float2)(0,0)};
        int2    coord_packed        = (int2)(2 * wid_x, wid_y);
        int2    coord_uv0           = (int2)(2 * wid_x, wid_y);
        int2    coord_uv1           = (int2)(2 * wid_x + 1, wid_y);
        float4  read_pixel_uv0      = read_imagef(srcImage, sampler, coord_uv0);
        float4  read_pixel_uv1      = read_imagef(srcImage, sampler, coord_uv1);
        packed_pixel_uv[0].x        = read_pixel_uv0.x;
        packed_pixel_uv[0].y        = read_pixel_uv0.y;
        packed_pixel_uv[1].x        = read_pixel_uv1.x;
        packed_pixel_uv[1].y        = read_pixel_uv1.y;
        qcom_write_imagefv_2x1_n8n01(dstImage, coord_packed, packed_pixel_uv);
    }
)";

// Kernel for copying from planar_YUV to planar_YUV.
static const char* programCopyYUVImage2YImageAndUVImage = R"(
    __kernel void copy_yuv_image_to_y_image_and_uv_image (__read_only image2d_t   srcYUVImage,
                                                          __write_only image2d_t  dstYImage,
                                                          __write_only image2d_t  dstUVImage,
                                                          sampler_t               sampler)
    {
        int     wid_x       = get_global_id(0);
        int     wid_y       = get_global_id(1);
        int2    coord_yuv   = (int2)(wid_x, wid_y);
        int2    coord_y     = (int2)(wid_x, wid_y);
        int2    coord_uv    = (int2)(wid_x >> 1, wid_y >> 1);
        float4  yuv_pixel   = read_imagef(srcYUVImage, sampler, coord_yuv);
        float4  y_pixel     = (float4)(yuv_pixel.x, 0, 0, 0);
        float4  uv_pixel    = (float4)(yuv_pixel.y, yuv_pixel.z, 0, 0);
        write_imagef(dstYImage, coord_y, y_pixel);
        if(!(wid_x & 1) && !(wid_y & 1))
        {
            write_imagef(dstUVImage, coord_uv, uv_pixel);
        }
    }
)";

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Please specify source image.\n"
                     "\n"
                     "Usage: " << argv[0] << " <source image data file>\n"
                     "       <regular output image data file> <optimized output image data file>\n"
                     "\n"
                     "Demonstrates use of optimized YUV ReadWrite using Qualcomm extensions to OpenCL.\n"
                     "The input NV12 image is read in 4X1 fashion in Y-Plane and 2X1 fashion in UV-Plane.\n"
                     "The files are written to output files for comparison.\n"
                     "\n";
        return EXIT_FAILURE;
    }
    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename_regular(argv[2]);
    const std::string out_image_filename_optimized(argv[3]);

    static const cl_queue_properties QUEUE_PROPERTIES[]  = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_wrapper                       wrapper(nullptr, QUEUE_PROPERTIES);
    cl_event                         map_event[3];
    cl_int                           err                 = CL_SUCCESS;
    cl_context                       context             = wrapper.get_context();
    cl_command_queue                 queue               = wrapper.get_command_queue();
    nv12_image_t                     src_nv12_image_info = load_nv12_image_data(src_image_filename);
    struct dma_buf_sync              buf_sync            = {};
    cl_event                         unmap_event         = nullptr;

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

    std::vector<cl_image_format> formats;
    formats = get_image_formats(context, CL_MEM_READ_WRITE);

    if (src_nv12_image_info.y_width % CL_IMAGE_WIDTH_ALIGNMENT_IN_ELEMENTS_PLANAR)
    {
        std::cerr << "src_image_params.image_width does not meet alignment requirement.\n";
    return EXIT_FAILURE;
    }

    /*
     * Step 1: Setting up resources
     */

    cl_image_format src_nv12_format;
    src_nv12_format.image_channel_order     = CL_QCOM_NV12;
    src_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc       src_nv12_desc;
    std::memset(&src_nv12_desc, 0, sizeof(src_nv12_desc));
    src_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_nv12_desc.image_width  = src_nv12_image_info.y_width;
    src_nv12_desc.image_height = src_nv12_image_info.y_height;

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

    /*
     * Step 3: Copy data to input image planes. Note that for linear NV12 images you must observe row alignment
     * restrictions. (You may also write to the DMA buffer directly if you prefer, however using clEnqueueMapImage for
     * a child planar image will return the correct host pointer for the desired plane.)
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if (ioctl(src_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for writes. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    const size_t     origin[]       = {0, 0, 0};
    const size_t     src_y_region[] = {src_y_plane_desc.image_width, src_y_plane_desc.image_height, 1};
    size_t           row_pitch      = 0;
    unsigned char   *image_ptr      = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            queue,
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
                src_nv12_image_info.y_plane.data() + i * src_y_plane_desc.image_width,
                src_y_plane_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(queue, src_y_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer." << "\n";
        return err;
    }

    // Note the discrepancy between the child plane image descriptor and the size required by clEnqueueMapImage.
    const size_t src_uv_region[] = {src_uv_plane_desc.image_width/2, src_uv_plane_desc.image_height/2 , 1};
    row_pitch                    = 0;
    image_ptr = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            queue,
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
    for (uint32_t i = 0; i < src_uv_plane_desc.image_height/ 2; ++i)
    {
        std::memcpy(
                image_ptr                           + i * row_pitch,
                src_nv12_image_info.uv_plane.data() + i * src_uv_plane_desc.image_width,
                src_uv_plane_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(queue, src_uv_plane, image_ptr, 0, nullptr, &unmap_event);
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

    // creating regular output image
    cl_image_format output_regular_nv12_format;
    output_regular_nv12_format.image_channel_order     = CL_QCOM_NV12;
    output_regular_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc output_regular_nv12_desc;
    std::memset(&output_regular_nv12_desc, 0, sizeof(output_regular_nv12_desc));
    output_regular_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    output_regular_nv12_desc.image_width  = src_nv12_image_info.y_width;
    output_regular_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_mem_dmabuf_host_ptr output_regular_nv12_mem = wrapper.make_buffer_for_yuv_image(output_regular_nv12_format, output_regular_nv12_desc);

    cl_mem output_regular_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &output_regular_nv12_format,
            &output_regular_nv12_desc,
            &output_regular_nv12_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output regular image." << "\n";
        return err;
    }

    // creating image for output optimized image
    cl_image_format output_optimized_nv12_format;
    output_optimized_nv12_format.image_channel_order     = CL_QCOM_NV12;
    output_optimized_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc output_optimized_nv12_desc;
    std::memset(&output_optimized_nv12_desc, 0, sizeof(output_optimized_nv12_desc));
    output_optimized_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    output_optimized_nv12_desc.image_width  = src_nv12_image_info.y_width;
    output_optimized_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_mem_dmabuf_host_ptr output_optimized_nv12_mem = wrapper.make_buffer_for_yuv_image(output_optimized_nv12_format, output_optimized_nv12_desc);

    cl_mem output_optimized_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &output_optimized_nv12_format,
            &output_optimized_nv12_desc,
            &output_optimized_nv12_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output optimized image." << "\n";
        return err;
    }

    // creating y plane for regular image read-write
    cl_image_format output_regular_y_plane_format;
    output_regular_y_plane_format.image_channel_order     = CL_QCOM_NV12_Y;
    output_regular_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc output_regular_y_plane_desc;
    std::memset(&output_regular_y_plane_desc, 0, sizeof(output_regular_y_plane_desc));
    output_regular_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    output_regular_y_plane_desc.image_width  = src_nv12_image_info.y_width;
    output_regular_y_plane_desc.image_height = src_nv12_image_info.y_height;
    output_regular_y_plane_desc.mem_object   = output_regular_nv12_image;

    cl_mem output_regular_y_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &output_regular_y_plane_format,
            &output_regular_y_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for destination REGULAR image y plane." << "\n";
        return err;
    }

    // creating uv plane for regular image read-write
    cl_image_format output_regular_uv_plane_format;
    output_regular_uv_plane_format.image_channel_order     = CL_QCOM_NV12_UV;
    output_regular_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc output_regular_uv_plane_desc;
    std::memset(&output_regular_uv_plane_desc, 0, sizeof(output_regular_uv_plane_desc));
    output_regular_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    output_regular_uv_plane_desc.image_width  = src_nv12_image_info.y_width;
    output_regular_uv_plane_desc.image_height = src_nv12_image_info.y_height;
    output_regular_uv_plane_desc.mem_object   = output_regular_nv12_image;

    cl_mem output_regular_uv_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &output_regular_uv_plane_format,
            &output_regular_uv_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source regular image uv plane." << "\n";
        return err;
    }

    // creating y plane for optimized image read-write
    cl_image_format output_optimized_y_plane_format;
    output_optimized_y_plane_format.image_channel_order     = CL_QCOM_NV12_Y;
    output_optimized_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc output_optimized_y_plane_desc;
    std::memset(&output_optimized_y_plane_desc, 0, sizeof(output_optimized_y_plane_desc));
    output_optimized_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    output_optimized_y_plane_desc.image_width  = src_nv12_image_info.y_width;
    output_optimized_y_plane_desc.image_height = src_nv12_image_info.y_height;
    output_optimized_y_plane_desc.mem_object   = output_optimized_nv12_image;

    cl_mem output_optimized_y_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &output_optimized_y_plane_format,
            &output_optimized_y_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source optimized image y plane." << "\n";
        return err;
    }

    // creating uv plane for optimized image
    cl_image_format output_optimized_uv_plane_format;
    output_optimized_uv_plane_format.image_channel_order     = CL_QCOM_NV12_UV;
    output_optimized_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc output_optimized_uv_plane_desc;
    std::memset(&output_optimized_uv_plane_desc, 0, sizeof(output_optimized_uv_plane_desc));
    output_optimized_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    output_optimized_uv_plane_desc.image_width  = src_nv12_image_info.y_width;
    output_optimized_uv_plane_desc.image_height = src_nv12_image_info.y_height;
    output_optimized_uv_plane_desc.mem_object   = output_optimized_nv12_image;

    cl_mem output_optimized_uv_plane = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &output_optimized_uv_plane_format,
            &output_optimized_uv_plane_desc,
            nullptr,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for source optimized image uv plane." << "\n";
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
        std::cerr << "Error " << err << " with clCreateSampler." << "\n";
        return err;
    }

    /*
     * Step 5: Calling Kernels
     */

    cl_program program_y = wrapper.make_program(&programCopyVectorWrite4x1_UNORM8_Y, 1);
    cl_kernel kernel_y = wrapper.make_kernel("write_4x1_unorm8_y2y", program_y);
    err = clSetKernelArg(kernel_y, 0, sizeof(cl_mem), &src_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for optimized y read-write kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_y, 1, sizeof(cl_mem), &output_optimized_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for optimized y read-write kernel." << "\n";
        return err;
    }
    err = clSetKernelArg(kernel_y, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {

        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for optimized y read-write kernel." << "\n";
        return err;
    }

    const size_t work_size[] = {src_nv12_image_info.y_width/4, src_nv12_image_info.y_height};
    err = clEnqueueNDRangeKernel(
            queue,
            kernel_y,
            2,
            nullptr,
            work_size,
            nullptr,
            0,
            nullptr,
            &map_event[0]
    );

    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for optimized y read-write kernel." << "\n";
        return err;
    }

    // for optimized uv copy
    cl_program program_uv = wrapper.make_program(&programCopyVectorWrite2x1_UNORM8_UV, 1);
    cl_kernel kernel_uv = wrapper.make_kernel("write_2x1_unorm8_uv2uv", program_uv);

    err = clSetKernelArg(kernel_uv, 0, sizeof(cl_mem), &src_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for optimized uv read-write kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_uv, 1, sizeof(cl_mem), &output_optimized_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for optimized uv read-write kernel." << "\n";
        return err;
    }
    err = clSetKernelArg(kernel_uv, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {

        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for optimized uv read-write kernel." << "\n";
        return err;
    }

    const size_t work_size_optimized_uv[] = {src_nv12_image_info.y_width/4, src_nv12_image_info.y_height/2};
    err = clEnqueueNDRangeKernel(
            queue,
            kernel_uv,
            2,
            nullptr,
            work_size_optimized_uv,
            nullptr,
            0,
            nullptr,
            &map_event[1]
    );

    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for optimized uv read-write kernel." << "\n";
        return err;
    }

    // for regular yuv copy
    cl_program program_yuv = wrapper.make_program(&programCopyYUVImage2YImageAndUVImage, 1);
    cl_kernel kernel_yuv = wrapper.make_kernel("copy_yuv_image_to_y_image_and_uv_image", program_yuv);
    err = clSetKernelArg(kernel_yuv, 0, sizeof(cl_mem), &src_nv12_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for regular yuv read-write." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_yuv, 1, sizeof(cl_mem), &output_regular_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for regular yuv read-write." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel_yuv, 2, sizeof(cl_mem), &output_regular_uv_plane);
    if (err != CL_SUCCESS)
    {

        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for regular yuv read-write." << "\n";
        return err;
    }
    err = clSetKernelArg(kernel_yuv, 3, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {

        std::cerr << "\tError " << err << " with clSetKernelArg for argument 3 for regular yuv read-write." << "\n";
        return err;
    }

    const size_t work_size_regular[] = {src_nv12_image_info.y_width, src_nv12_image_info.y_height};
    err = clEnqueueNDRangeKernel(
            queue,
            kernel_yuv,
            2,
            nullptr,
            work_size_regular,
            nullptr,
            0,
            nullptr,
            &map_event[2]
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for regular yuv read-write kernel." << "\n";
        return err;
    }

    /*
     * Step 6: Copy the data out of the DMA buffer to output files.
     */

    nv12_image_t output_optimized_nv12_image_info;
    output_optimized_nv12_image_info.y_width  = output_optimized_nv12_desc.image_width;
    output_optimized_nv12_image_info.y_height = output_optimized_nv12_desc.image_height;
    output_optimized_nv12_image_info.y_plane.resize(output_optimized_nv12_image_info.y_width * output_optimized_nv12_image_info.y_height);
    output_optimized_nv12_image_info.uv_plane.resize(output_optimized_nv12_image_info.y_width * output_optimized_nv12_image_info.y_height/2);

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if (ioctl(output_optimized_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }
    if (ioctl(src_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }
    if (ioctl(output_regular_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error preparing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    const size_t src_optimized_y_region[] = {output_optimized_y_plane_desc.image_width, output_optimized_y_plane_desc.image_height, 1};
    row_pitch                   = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            queue,
            output_optimized_y_plane,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            src_optimized_y_region,
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
    for (uint32_t i = 0; i < output_optimized_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                output_optimized_nv12_image_info.y_plane.data() + i * output_optimized_y_plane_desc.image_width,
                image_ptr                          + i * row_pitch,
                output_optimized_y_plane_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(queue, output_optimized_y_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image y-plane buffer." << "\n";
        return err;
    }

    const size_t src_optimized_uv_region[] = {output_optimized_uv_plane_desc.image_width/2, output_optimized_uv_plane_desc.image_height/2, 1};
    row_pitch                    = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            queue,
            output_optimized_uv_plane,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            src_optimized_uv_region,
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
    for (uint32_t i = 0; i < output_optimized_uv_plane_desc.image_height/2; ++i)
    {
        std::memcpy(
                output_optimized_nv12_image_info.uv_plane.data() + i * output_optimized_uv_plane_desc.image_width,
                image_ptr                           + i * row_pitch,
                output_optimized_uv_plane_desc.image_width
        );
    }
    err = clEnqueueUnmapMemObject(queue, output_optimized_uv_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image uv-plane buffer." << "\n";
        return err;
    }

    // regular output file creation and data copy
    nv12_image_t output_regular_nv12_image_info;
    output_regular_nv12_image_info.y_width  = output_regular_nv12_desc.image_width;
    output_regular_nv12_image_info.y_height = output_regular_nv12_desc.image_height;
    output_regular_nv12_image_info.y_plane.resize(output_regular_nv12_image_info.y_width * output_regular_nv12_image_info.y_height);
    output_regular_nv12_image_info.uv_plane.resize(output_regular_nv12_image_info.y_width * output_regular_nv12_image_info.y_height/2);

    const size_t src_regular_y_region[] = {output_regular_y_plane_desc.image_width, output_regular_y_plane_desc.image_height, 1};
    row_pitch = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            queue,
            output_regular_y_plane,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            src_regular_y_region,
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
    for (uint32_t i = 0; i < output_regular_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                output_regular_nv12_image_info.y_plane.data() + i * output_regular_y_plane_desc.image_width,
                image_ptr                          + i * row_pitch,
                output_regular_y_plane_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(queue, output_regular_y_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image y-plane buffer." << "\n";
        return err;
    }

    const size_t src_regular_uv_region[] = {output_regular_uv_plane_desc.image_width/2, output_regular_uv_plane_desc.image_height/2, 1};
    row_pitch                    = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            queue,
            output_regular_uv_plane,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            src_regular_uv_region,
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
    for (uint32_t i = 0; i < output_regular_uv_plane_desc.image_height/2; ++i)
    {
        std::memcpy(
                output_regular_nv12_image_info.uv_plane.data() + i * output_regular_uv_plane_desc.image_width,
                image_ptr                           + i * row_pitch,
                output_regular_uv_plane_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(queue, output_regular_uv_plane, image_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image uv-plane buffer." << "\n";
        return 1;
    }

    save_nv12_image_data(out_image_filename_regular, output_regular_nv12_image_info);
    save_nv12_image_data(out_image_filename_optimized, output_optimized_nv12_image_info);

    /*
     * Step 7: Verifying Results
     */

    // Comparing Source and Regular read-write image
    const size_t     ver_region[] = {src_nv12_image_info.y_width, src_nv12_image_info.y_height, 1};
    unsigned char   *image_ptr_s      = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            queue,
            src_nv12_image,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            ver_region,
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

    unsigned char   *image_ptr_r      = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            queue,
            output_regular_nv12_image,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            ver_region,
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

    // Comparing image data from the host
    for (uint32_t i = 0; i < src_nv12_image_info.y_height; ++i)
    {
        err = std::memcmp(
                image_ptr_s + i * src_nv12_image_info.y_width,
                image_ptr_r + i * src_nv12_image_info.y_width,
                src_y_plane_desc.image_width
        );
        if (err != 0)
        {
            std::cerr << "Error " << err << " source and regular read-write image failed." << "\n";
            return err;
        }
    }

    err = clEnqueueUnmapMemObject(queue, src_nv12_image, image_ptr_s, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer." << "\n";
        return err;
    }

    err = clEnqueueUnmapMemObject(queue, output_regular_nv12_image, image_ptr_r, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer." << "\n";
        return err;
    }

    // Comparing Source and Optimized read-write image
    image_ptr_s = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            queue,
            src_nv12_image,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            ver_region,
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

    unsigned char   *image_ptr_o      = reinterpret_cast<unsigned char *>(clEnqueueMapImage(
            queue,
            output_optimized_nv12_image,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            ver_region,
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

    // Comparing image data from the host
    for (uint32_t i = 0; i < src_nv12_image_info.y_height; ++i)
    {
        err = std::memcmp(
                image_ptr_s + i * src_nv12_image_info.y_width,
                image_ptr_o + i * src_nv12_image_info.y_width,
                src_y_plane_desc.image_width
        );
        if (err != 0)
        {
            std::cerr << "Error " << err << " source and optimized read-write image failed." << "\n";
            return err;
        }
    }

    err = clEnqueueUnmapMemObject(queue, src_nv12_image, image_ptr_s, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer." << "\n";
        return err;
    }

    err = clEnqueueUnmapMemObject(queue, output_optimized_nv12_image, image_ptr_o, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping source image y-plane data buffer." << "\n";
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
    if (ioctl(output_regular_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }
    if (ioctl(src_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }
    if (ioctl(output_optimized_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync))
    {
        std::cerr << "Error finalizing the DMA buffer for reads. The DMA_BUF_IOCTL_SYNC operation returned "
                  << strerror(errno) << "!\n";
        return errno;
    }

    /*
     * Step 8: Calculating execution time for Regular and Optimized read-write
     */

    cl_ulong start_time = 0;
    cl_ulong end_time = 0;
    err = clGetEventProfilingInfo(
            map_event[0],
            CL_PROFILING_COMMAND_START,
            sizeof(start_time),
            &start_time,
            nullptr);
    if(err != CL_SUCCESS)
    {
        std::cout<< "Nonzero error code "<<err<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
    }
    err = clGetEventProfilingInfo(
            map_event[0],
            CL_PROFILING_COMMAND_END,
            sizeof(end_time),
            &end_time,
            nullptr);
    if(err != CL_SUCCESS)
    {
        std::cout<< "Nonzero error code "<<err<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
    }

    cl_ulong optim_yuv_exec_time = (end_time - start_time);
    start_time = 0;
    end_time = 0;
    err = clGetEventProfilingInfo(
            map_event[1],
            CL_PROFILING_COMMAND_START,
            sizeof(start_time),
            &start_time,
            nullptr);
    if(err != CL_SUCCESS)
    {
        std::cout<< "Nonzero error code "<<err<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
    }
    err = clGetEventProfilingInfo(
            map_event[1],
            CL_PROFILING_COMMAND_END,
            sizeof(end_time),
            &end_time,
            nullptr);
    if(err != CL_SUCCESS)
    {
        std::cout<< "Nonzero error code "<<err<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
    }
    optim_yuv_exec_time += (end_time - start_time);

    start_time = 0;
    end_time = 0;
    err = clGetEventProfilingInfo(
            map_event[2],
            CL_PROFILING_COMMAND_START,
            sizeof(start_time),
            &start_time,
            nullptr);
    if(err != CL_SUCCESS)
    {
        std::cout<< "Nonzero error code "<<err<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
    }
    err = clGetEventProfilingInfo(
            map_event[2],
            CL_PROFILING_COMMAND_END,
            sizeof(end_time),
            &end_time,
            nullptr);
    if(err != CL_SUCCESS)
    {
        std::cout<< "Nonzero error code "<<err<<" at"<< __FILE__<<":"<<__LINE__<<"\n";
        return(0);
    }

    cl_ulong reg_yuv_exec_time = (end_time - start_time);
    std::cout<<"\nTest Executed Successfully\n"<<"optimization time: "
    <<optim_yuv_exec_time<<" ms\nregular time is: "<<reg_yuv_exec_time<<" ms\n"
    <<"%age benifit: "<<std::fixed<<std::setprecision(2)
    <<((((reg_yuv_exec_time-optim_yuv_exec_time)*100)/(double)reg_yuv_exec_time))<<"%\n";

    err = clFinish(queue);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " while finishing the command queue.\n";
        return err;
    }

    /*
     * Step 9: Release memory
     */

    clReleaseSampler(sampler);
    clReleaseMemObject(src_nv12_image);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(output_regular_nv12_image);
    clReleaseMemObject(output_optimized_nv12_image);
    clReleaseMemObject(output_regular_y_plane);
    clReleaseMemObject(output_regular_uv_plane);
    clReleaseMemObject(output_optimized_y_plane);
    clReleaseMemObject(output_optimized_uv_plane);

    return EXIT_SUCCESS;
}