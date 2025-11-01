//--------------------------------------------------------------------------------------
// File: convolution.cpp
// Desc: Runs a kernel demonstrating Gaussian blur with runtime constant promotion.
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
#include "util/half_float.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>
#include <CL/cl_ext_qcom.h>

static const char *PROGRAM_SOURCE = R"(
    __kernel void convolution(__read_only image2d_t      src_image,
                              __global    unsigned char *dest_image,
                                          int            dest_row_pitch,
                                          sampler_t      sampler,
                              __constant  float         *conv_filter __attribute__((max_constant_size(9 * 4))))
    {
        const int  wid_x            = get_global_id(0);
        const int  wid_y            = get_global_id(1);
        const int2 src_coord        = (int2)(2 * wid_x, 2 * wid_y);
        float4     convolved_pixels = (float4)(0, 0, 0, 0);
        float      source_pixels[16];
        for (int i = 0; i < 4; ++i)
        {
           for (int j = 0; j < 4; ++j)
           {
               const int  idx     = i * 4 + j;
               const int2 offset  = (int2)(j - 1, i - 1);
               source_pixels[idx] = read_imagef(src_image, sampler, src_coord + offset).x;
           }
        }
        for (int i = 0; i < 3; ++i)
        {
           for (int j = 0; j < 3; ++j)
           {
               const int  filter_idx = i * 3 + j;
               const int  pixel_idx1 = i * 4 + j;
               const int  pixel_idx2 = (i + 1) * 4 + j;
               convolved_pixels.s0 += conv_filter[filter_idx] * source_pixels[pixel_idx1];
               convolved_pixels.s1 += conv_filter[filter_idx] * source_pixels[pixel_idx1 + 1];
               convolved_pixels.s2 += conv_filter[filter_idx] * source_pixels[pixel_idx2];
               convolved_pixels.s3 += conv_filter[filter_idx] * source_pixels[pixel_idx2 + 1];
           }
        }
        convolved_pixels *= 255.0;
        const uchar2 res1  = convert_uchar2(convolved_pixels.s01);
        const uchar2 res2  = convert_uchar2(convolved_pixels.s23);
        vstore2(res1, wid_x, dest_image + (src_coord.y       * dest_row_pitch));
        vstore2(res2, wid_x, dest_image + ((src_coord.y + 1) * dest_row_pitch));
    }
)";

static const cl_uint  CONV_FILTER_WIDTH  = 3;
static const cl_uint  CONV_FILTER_HEIGHT = 3;

// This filter corresponds to a 3x3 Gaussian blur.
static cl_float CONVOLUTION_FILTER[CONV_FILTER_HEIGHT * CONV_FILTER_WIDTH] = {
    (1.f / 16.f), (1.f / 8.f), (1.f / 16.f),
    (1.f / 8.f),  (1.f / 4.f), (1.f / 8.f),
    (1.f / 16.f), (1.f / 8.f), (1.f / 16.f)
};

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Please specify source and output images.\n"
                     "\n"
                     "Usage: " << argv[0] << " <source image data file> <output image data file>\n"
                     "Runs a kernel demonstrating Gaussian blur with runtime constant promotion.\n"
                     "Additionally demonstrates that images and buffers can use the same underlying\n"
                     "dmabuf memory, by writing to an OpenCL buffer and reading results from an image\n"
                     "that share the same DMA buffer.\n";
        return EXIT_FAILURE;
    }
    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper          wrapper;
    cl_program          program             = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel           y_plane_kernel      = wrapper.make_kernel("convolution", program);
    cl_context          context             = wrapper.get_context();
    nv12_image_t        src_nv12_image_info = load_nv12_image_data(src_image_filename);
    struct dma_buf_sync buf_sync            = {};
    cl_event            unmap_event         = nullptr;

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
        std::cerr << "Extension cl_qcom_ext_host_ptr needed for dmabuf-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_dmabuf_host_ptr"))
    {
        std::cerr << "Extension cl_qcom_dmabuf_host_ptr needed for dmabuf-backed images is not supported.\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create suitable DMA buffer-backed CL images. Note that planar formats (like NV12) must be read only,
     * but you can write to child images derived from the planes. (See step 1 for deriving child images.)
     */

    cl_image_format src_nv12_format;
    src_nv12_format.image_channel_order     = CL_QCOM_NV12;
    src_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_nv12_desc;
    std::memset(&src_nv12_desc, 0, sizeof(src_nv12_desc));
    src_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    src_nv12_desc.image_width  = src_nv12_image_info.y_width;
    src_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_mem_dmabuf_host_ptr src_nv12_mem = wrapper.make_buffer_for_yuv_image(src_nv12_format, src_nv12_desc);
    cl_int err;
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

    cl_image_format out_nv12_format;
    out_nv12_format.image_channel_order     = CL_QCOM_NV12;
    out_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_nv12_desc;
    std::memset(&out_nv12_desc, 0, sizeof(out_nv12_desc));
    out_nv12_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_nv12_desc.image_width  = src_nv12_image_info.y_width;
    out_nv12_desc.image_height = src_nv12_image_info.y_height;

    cl_mem_dmabuf_host_ptr out_nv12_mem = wrapper.make_buffer_for_yuv_image(out_nv12_format, out_nv12_desc);
    cl_mem out_nv12_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &out_nv12_format,
            &out_nv12_desc,
            &out_nv12_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for output image." << "\n";
        return err;
    }

    // Note: images and buffers can be backed by the same underlying dmabuf memory.
    const int out_img_row_pitch = static_cast<int>(wrapper.get_image_row_pitch(out_nv12_format, out_nv12_desc));
    cl_mem out_nv12_buffer = clCreateBuffer(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            out_img_row_pitch * out_nv12_desc.image_height,
            &out_nv12_mem,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer." << "\n";
        return err;
    }

    /*
     * Step 2: Separate planar NV12 images into their component planes.
     */

    cl_image_format src_y_plane_format;
    src_y_plane_format.image_channel_order     = CL_QCOM_NV12_Y;
    src_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc src_y_plane_desc;
    std::memset(&src_y_plane_desc, 0, sizeof(src_y_plane_desc));
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

    cl_image_format out_y_plane_format;
    out_y_plane_format.image_channel_order     = CL_QCOM_NV12_Y;
    out_y_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_y_plane_desc;
    std::memset(&out_y_plane_desc, 0, sizeof(out_y_plane_desc));
    out_y_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    out_y_plane_desc.image_width  = out_nv12_desc.image_width;
    out_y_plane_desc.image_height = out_nv12_desc.image_height;
    out_y_plane_desc.mem_object   = out_nv12_image;

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

    /*
     * Step 3: Copy data to input image planes. Note that for linear NV12 images you must observe row alignment
     * restrictions. (You may also write to the DMA buffer directly if you prefer, however using clEnqueueMapImage for
     * a child planar image will return the correct host pointer for the desired plane.)
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(src_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    cl_command_queue command_queue  = wrapper.get_command_queue();
    const size_t     origin[]       = {0, 0, 0};
    const size_t     src_y_region[] = {src_y_plane_desc.image_width, src_y_plane_desc.image_height, 1};
    size_t           row_pitch      = 0;
    unsigned char   *image_ptr      = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            src_y_plane,
            CL_TRUE,
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

    // Copies image data to the DMA buffer from the host
    for (uint32_t i = 0; i < src_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                image_ptr                          + i * row_pitch,
                src_nv12_image_info.y_plane.data() + i * src_y_plane_desc.image_width,
                src_y_plane_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, src_y_plane, image_ptr, 0, nullptr, &unmap_event);
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

    buf_sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    if ( ioctl(src_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
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

    cl_mem convolution_kernel = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(CONVOLUTION_FILTER),
            CONVOLUTION_FILTER,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateBuffer." << "\n";
        return err;
    }

    /*
     * Step 5: Run the kernel separately for y-plane only
     */

    err = clSetKernelArg(y_plane_kernel, 0, sizeof(src_y_plane), &src_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0 for y-plane kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(y_plane_kernel, 1, sizeof(out_nv12_buffer), &out_nv12_buffer);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1 for y-plane kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(y_plane_kernel, 2, sizeof(out_img_row_pitch), &out_img_row_pitch);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2 for y-plane kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(y_plane_kernel, 3, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3 for y-plane kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(y_plane_kernel, 4, sizeof(convolution_kernel), &convolution_kernel);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 4 for y-plane kernel." << "\n";
        return err;
    }

    const size_t y_plane_work_size[] = {out_y_plane_desc.image_width / 2, out_y_plane_desc.image_height / 2};
    err = clEnqueueNDRangeKernel(
            command_queue,
            y_plane_kernel,
            2,
            nullptr,
            y_plane_work_size,
            nullptr,
            0,
            nullptr,
            nullptr
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel for y-plane kernel." << "\n";
        return err;
    }

    clFinish(command_queue);

    /*
     * Step 6: Copy the data out of the DMA buffer for each plane.
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if ( ioctl(out_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    nv12_image_t out_nv12_image_info;
    out_nv12_image_info.y_width  = out_nv12_desc.image_width;
    out_nv12_image_info.y_height = out_nv12_desc.image_height;
    out_nv12_image_info.y_plane.resize(out_nv12_image_info.y_width * out_nv12_image_info.y_height);
    out_nv12_image_info.uv_plane = src_nv12_image_info.uv_plane;

    const size_t out_y_region[] = {out_y_plane_desc.image_width, out_y_plane_desc.image_height, 1};
    row_pitch                   = 0;
    image_ptr = static_cast<unsigned char *>(clEnqueueMapImage(
            command_queue,
            out_y_plane,
            CL_TRUE,
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

    // Copies image data from the DMA buffer to the host
    for (uint32_t i = 0; i < out_y_plane_desc.image_height; ++i)
    {
        std::memcpy(
                out_nv12_image_info.y_plane.data() + i * out_y_plane_desc.image_width,
                image_ptr                          + i * row_pitch,
                out_y_plane_desc.image_width
        );
    }

    err = clEnqueueUnmapMemObject(command_queue, out_y_plane, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping dest image y-plane buffer." << "\n";
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
    if ( ioctl(out_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    clFinish(command_queue);

    save_nv12_image_data(out_image_filename, out_nv12_image_info);

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseSampler(sampler);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_nv12_image);
    clReleaseMemObject(out_nv12_buffer);
    clReleaseMemObject(out_y_plane);
    clReleaseMemObject(out_nv12_image);

    return EXIT_SUCCESS;
}
