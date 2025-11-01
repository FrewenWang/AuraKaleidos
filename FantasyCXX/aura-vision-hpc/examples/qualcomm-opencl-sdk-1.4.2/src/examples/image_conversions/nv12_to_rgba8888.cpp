//--------------------------------------------------------------------------------------
// File: nv12_to_rgba8888.cpp
// Desc: This program converts NV12 to RGBA8888
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
     nv12_to_rgba(__read_only image2d_t input_nv12,
                 __write_only image2d_t out_rgba, sampler_t sampler)
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

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <img data file> <out img data file> \n"
                     "Input image file data should be in format CL_QCOM_NV12 / CL_UNORM_INT8\n"
                     "Demonstrates conversions from NV12 to RGBA8888\n";
        return EXIT_FAILURE;
    }

    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);

    cl_wrapper          wrapper;
    cl_program          program             = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel           nv12_to_rgb_kernel  = wrapper.make_kernel("nv12_to_rgba", program);
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
    const bool rw_formats_supported =
            is_format_supported(formats, cl_image_format{CL_RGBA,  CL_UNORM_INT8});
    if (!rw_formats_supported)
    {
        std::cerr << "For this example your device must support read-write CL_RGBA"
                     "with CL_UNORM_INT8 image format, but it does not.\n";
        std::cerr << "Supported read-write formats include:\n";
        print_formats(formats);
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create suitable DMA buffer-backed CL images. Note that planar formats (like NV12) must be read only,
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

    cl_image_format out_rgba_format;
    out_rgba_format.image_channel_order     = CL_RGBA;
    out_rgba_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_rgba_desc;
    std::memset(&out_rgba_desc, 0, sizeof(out_rgba_desc));
    out_rgba_desc.image_type             = CL_MEM_OBJECT_IMAGE2D;
    out_rgba_desc.image_width            = src_nv12_image_info.y_width;
    out_rgba_desc.image_height           = src_nv12_image_info.y_height;
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
        std::cerr << "Error " << err << " with clCreateImage for output RGB image." << "\n";
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

    // For mapping the NV12_UV image for host access, the region dimensions should reflect the actual number of UV pixels
    // being read. This is different from the dimensions passed in to create the NV12_Y image. In that case, the width
    // and height of the parent YUV image are expected.
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
     * Step 5: Run the kernel for both y- and uv-planes
     */

    err = clSetKernelArg(nv12_to_rgb_kernel, 0, sizeof(src_nv12_image), &src_nv12_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 0 for nv12_to_rgb_kernel kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(nv12_to_rgb_kernel, 1, sizeof(out_rgba_image), &out_rgba_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 1 for nv12_to_rgb_kernel kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(nv12_to_rgb_kernel, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "\tError " << err << " with clSetKernelArg for argument 2 for nv12_to_rgb_kernel kernel." << "\n";
        return err;
    }

    const size_t work_size[] = {out_rgba_desc.image_width,out_rgba_desc.image_height};
    err = clEnqueueNDRangeKernel(
            command_queue,
            nv12_to_rgb_kernel,
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
        std::cerr << "\tError " << err << " with clEnqueueNDRangeKernel for nv12_to_rgb_kernel kernel." << "\n";
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
    // Copies image data from the DMA buffer to the host.
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
    clReleaseMemObject(src_nv12_image);
    clReleaseMemObject(out_rgba_image);

    return EXIT_SUCCESS;
}