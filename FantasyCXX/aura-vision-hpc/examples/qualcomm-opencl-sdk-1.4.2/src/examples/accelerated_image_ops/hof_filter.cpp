//--------------------------------------------------------------------------------------
// File: hof_filter.cpp
// Desc: Demonstrates the use of high order filters with the
//       cl_qcom_accelerated_image_ops extensions
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
    __kernel void hof_copy(__read_only   image2d_t               src_image,
                           __write_only  image2d_t               dest_image,
                           sampler_t                             sampler,
                           __read_only   qcom_weight_image_t     weight_image,
                           __global      float2                  *coord_offset)
    {
        int     wid_x            = get_global_id(0);
        int     wid_y            = get_global_id(1);
        int2    coord            = (int2)(wid_x, wid_y);
        float2  src_coord        = (float2)((float)wid_x + coord_offset[0].x, (float)wid_y + coord_offset[0].y);
        float4  convolved_pixel  = qcom_convolve_imagef(src_image, sampler, src_coord, weight_image);
        write_imagef(dest_image, coord, convolved_pixel);
    }
)";

static const cl_uint    NUM_PHASES = 4;
static const cl_uint    NUM_SEPARABLE_PHASES = 2;
static const cl_uint    FILTER_SIZE_X = 3;
static const cl_uint    FILTER_SIZE_Y = 3;

static const cl_half zero = to_half(0.f);
static const cl_half one = to_half(1.f);
static const cl_half threefourth = to_half(.75f);
static const cl_half half = to_half(.5f);
static const cl_half threeeigth = to_half(.375f);

static cl_half NON_SEPARABLE_WEIGHTS[NUM_PHASES * FILTER_SIZE_X * FILTER_SIZE_Y] = {
    // phase 0
    zero,   zero,       zero,
    zero,   one,        zero,
    zero,   zero,       zero,
    // phase 1
    zero,   zero,       zero,
    zero,   threefourth,      zero,
    zero,   zero,       zero,
    // phase 2
    zero,   zero,       zero,
    zero,   half, zero,
    zero,   zero,       zero,
    // phase 3
    zero,   zero,       zero,
    zero,   threeeigth,       zero,
    zero,   zero,       zero,
};

static cl_half SEPARABLE_WEIGHTS[NUM_SEPARABLE_PHASES * (FILTER_SIZE_X + FILTER_SIZE_Y)] = {
    // phase 0
    zero,   one,        zero,
    zero,   one,        zero,
    // phase 1
    zero,   threefourth,zero,
    zero,   half,       zero
};

std::string printFilter(cl_bool is_separable, size_t num_phases,
                        size_t filter_size_x, size_t filter_size_y, void *filter_weights)
{
    int i;
    std::string description;
    if (is_separable)
    {
        i = 0;
        for(size_t j = 0; j < num_phases;j++)
        {
            for(size_t x = 0; x <  filter_size_x;x++)
                description += std::to_string(to_float (((cl_half *)filter_weights)[i++])) + " ";
            for(size_t y = 0; y <  filter_size_y;y++)
                description += std::to_string(to_float (((cl_half *)filter_weights)[i++])) + " ";
            description += '\n';
        }
    }
    else
    {
        i= 0;
        for(size_t j = 0; j < num_phases;j++)
        {
            for (size_t row = 0; row < filter_size_y;row++)
            {
                for (size_t column = 0; column < filter_size_x; column++)
                    description += std::to_string(to_float (((cl_half *)filter_weights)[i++])) + " ";
                description += '\n';
            }
        }
    }
    return description;
}


int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Please specify source and output images.\n"
                     "\n"
                     "Usage: " << argv[0] << " <source image data file> <output image data file> <filter_type>\n"
                     "\t<filter_type> = separable | non-separable\n"
                     "Demonstrates the use of high order filters with the cl_qcom_accelerated_image_ops extensions. Runs a kernel that uses\n"
                     "qcom_convolve_imagef with a 4 phase 3x3 filter. Depending on the input this will use separable / non-separable filters\n"
                     "For each case, we run two versions of the kernel, one that adds a 0.5f offset to each coordinate and another that adds 1.25f to each coordinate\n"
                     "Since phase calculation is relative to the pixel center, this will result in the selection of phases: 0 and 3 respectively\n";
        return EXIT_FAILURE;
    }

    const std::string src_image_filename(argv[1]);
    const std::string out_image_filename(argv[2]);
    const std::string filter_type(argv[3]);

    bool separable = filter_type.compare("separable") == 0;

    cl_wrapper           wrapper;
    cl_program           program             = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel            y_plane_kernel      = wrapper.make_kernel("hof_copy", program);
    cl_kernel            uv_plane_kernel     = wrapper.make_kernel("hof_copy", program);
    cl_context           context             = wrapper.get_context();
    nv12_image_t         src_nv12_image_info = load_nv12_image_data(src_image_filename);
    struct dma_buf_sync  buf_sync            = {};
    cl_event             unmap_event         = nullptr;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

    if (!wrapper.check_extension_support("cl_qcom_other_image"))
    {
        std::cerr << "Extension cl_qcom_other_image needed for NV12 image format is not supported.\n";
        return EXIT_FAILURE;
    }

    if (!wrapper.check_extension_support("cl_qcom_accelerated_image_ops"))
    {
        std::cerr << "Extension cl_qcom_accelerated_image_ops needed for qcom_convolve_imagef is not supported.\n";
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
        std::cerr << "Error " << err << " with clCreateImage for downscaled image." << "\n";
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

    cl_image_format out_uv_plane_format;
    out_uv_plane_format.image_channel_order     = CL_QCOM_NV12_UV;
    out_uv_plane_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc out_uv_plane_desc;
    std::memset(&out_uv_plane_desc, 0, sizeof(out_uv_plane_desc));
    out_uv_plane_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    // The image dimensions for the uv-plane derived image must be the same as the parent image, even though the
    // actual dimensions of the uv-plane differ by a factor of 2 in each dimension.
    out_uv_plane_desc.image_width  = out_nv12_desc.image_width;
    out_uv_plane_desc.image_height = out_nv12_desc.image_height;
    out_uv_plane_desc.mem_object   = out_nv12_image;

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

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(src_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    /*
     * Step 3: Copy data to input image planes. Note that for linear NV12 images you must observe row alignment
     * restrictions. (You may also write to the DMA buffer directly if you prefer, however using clEnqueueMapImage for
     * a child planar image will return the correct host pointer for the desired plane.)
     */

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
            CL_TRUE,
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

    // Copies image data to the DMA buffer from the host
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

    cl_image_format weight_image_format;
    weight_image_format.image_channel_order     = CL_R;
    weight_image_format.image_channel_data_type = CL_HALF_FLOAT;

    cl_weight_image_desc_qcom weight_image_desc;
    std::memset(&weight_image_desc, 0, sizeof(weight_image_desc));
    weight_image_desc.image_desc.image_type        = CL_MEM_OBJECT_WEIGHT_IMAGE_QCOM;
    weight_image_desc.image_desc.image_width       = FILTER_SIZE_X;
    weight_image_desc.image_desc.image_height      = FILTER_SIZE_Y;
    weight_image_desc.image_desc.image_array_size  = separable ? NUM_SEPARABLE_PHASES : NUM_PHASES;
    weight_image_desc.image_desc.image_row_pitch   = 0;
    weight_image_desc.image_desc.image_slice_pitch = 0;

    weight_image_desc.weight_desc.center_coord_x = 1;
    weight_image_desc.weight_desc.center_coord_y = 1;
    weight_image_desc.weight_desc.flags          =  separable ? CL_WEIGHT_IMAGE_SEPARABLE_QCOM : 0;

    cl_mem weight_image = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &weight_image_format,
            reinterpret_cast<cl_image_desc *>(&weight_image_desc),
            separable ? static_cast<void *>(SEPARABLE_WEIGHTS) : static_cast<void *>(NON_SEPARABLE_WEIGHTS),
            &err
    );
    if (err != CL_SUCCESS) {
        std::cerr << "Error " << err << " with clCreateImage for weight image." << "\n";
        return err;
    }

    std::cout << "Running a 4-phase convolution with " << (separable ? "separable " : "non-separable ") << "weights" << "\n";
    std::cout << printFilter(separable, weight_image_desc.image_desc.image_array_size,
                             weight_image_desc.image_desc.image_width, weight_image_desc.image_desc.image_height,
                             separable ? static_cast<void *>(SEPARABLE_WEIGHTS) : static_cast<void *>(NON_SEPARABLE_WEIGHTS));


    /*
     * Step 5: Run the kernel separately for y- and uv-planes
     */

    err = clSetKernelArg(y_plane_kernel, 0, sizeof(src_y_plane), &src_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0 for y-plane kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(y_plane_kernel, 1, sizeof(out_y_plane), &out_y_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1 for y-plane kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(y_plane_kernel, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2 for y-plane kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(y_plane_kernel, 3, sizeof(weight_image), &weight_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3 for y-plane kernel." << "\n";
        return err;
    }


    err = clSetKernelArg(uv_plane_kernel, 0, sizeof(src_uv_plane), &src_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0 for uv-plane kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(uv_plane_kernel, 1, sizeof(out_uv_plane), &out_uv_plane);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1 for uv-plane kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(uv_plane_kernel, 2, sizeof(sampler), &sampler);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 2 for uv-plane kernel." << "\n";
        return err;
    }

    err = clSetKernelArg(uv_plane_kernel, 3, sizeof(weight_image), &weight_image);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 3 for uv-plane kernel." << "\n";
        return err;
    }

    cl_float2 sub_pixel_offsets[2];
    sub_pixel_offsets[0].x = sub_pixel_offsets[0].y = 0.5f;
    sub_pixel_offsets[1].x = sub_pixel_offsets[1].y = 1.25f;

    for (int k = 0; k < sizeof(sub_pixel_offsets)/sizeof(sub_pixel_offsets[0]); k++)
    {
        cl_mem offset_buffer = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof (cl_float2), &sub_pixel_offsets[k], &err);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clCreateBuffer for offset_buffer\n";
            return err;
        }

        err = clSetKernelArg(y_plane_kernel, 4, sizeof(offset_buffer), &offset_buffer);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clSetKernelArg for argument 4 for y-plane kernel." << "\n";
            return err;
        }

        err = clSetKernelArg(uv_plane_kernel, 4, sizeof(offset_buffer), &offset_buffer);
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clSetKernelArg for argument 4 for uv-plane kernel." << "\n";
            return err;
        }


        const size_t y_plane_work_size[] = {out_y_plane_desc.image_width, out_y_plane_desc.image_height};
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

        const size_t uv_plane_work_size[] = {out_uv_plane_desc.image_width / 2, out_uv_plane_desc.image_height / 2};
        err = clEnqueueNDRangeKernel(
                command_queue,
                uv_plane_kernel,
                2,
                nullptr,
                uv_plane_work_size,
                nullptr,
                0,
                nullptr,
                nullptr
        );
        if (err != CL_SUCCESS)
        {
            std::cerr << "Error " << err << " with clEnqueueNDRangeKernel for uv-plane kernel." << "\n";
            return err;
        }

        err = clFinish(command_queue);
        if (err != CL_SUCCESS)
        {
            std::cerr << "\tError " << err << " with clFinish." << "\n";
            return err;
        }

        buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
        if ( ioctl(out_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
        {
            std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
            return errno;
        }

        /*
         * Step 6: Copy the data out of the DMA buffer for each plane.
         */

        nv12_image_t out_nv12_image_info;
        out_nv12_image_info.y_width  = out_nv12_desc.image_width;
        out_nv12_image_info.y_height = out_nv12_desc.image_height;
        out_nv12_image_info.y_plane.resize(out_nv12_image_info.y_width * out_nv12_image_info.y_height);
        out_nv12_image_info.uv_plane.resize(out_nv12_image_info.y_width * out_nv12_image_info.y_height / 2);

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
                CL_TRUE,
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

        // Copies image data from the DMA buffer to the host
        for (uint32_t i = 0; i < out_uv_plane_desc.image_height / 2; ++i)
        {
            std::memcpy(
                    out_nv12_image_info.uv_plane.data() + i * out_uv_plane_desc.image_width,
                    image_ptr                           + i * row_pitch,
                    out_uv_plane_desc.image_width
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
        if ( ioctl(out_nv12_mem.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
        {
            std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
            return errno;
        }

        err = clFinish(command_queue);
        if (err != CL_SUCCESS)
        {
            std::cerr << "\tError " << err << " with clFinish." << "\n";
            return err;
        }

        const std::string out_file_name = out_image_filename + "_" + (separable ? "separable" : "nonseparable") + "_" + std::to_string(sub_pixel_offsets[k].x);
        std::cout << "Output for coord offsets: " << sub_pixel_offsets[k].x << "," << sub_pixel_offsets[k].y << " saved to " <<  out_file_name << " \n";
        save_nv12_image_data(out_file_name, out_nv12_image_info);
        clReleaseMemObject(offset_buffer);
    }

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseMemObject(weight_image);
    clReleaseSampler(sampler);
    clReleaseMemObject(src_uv_plane);
    clReleaseMemObject(src_y_plane);
    clReleaseMemObject(src_nv12_image);
    clReleaseMemObject(out_uv_plane);
    clReleaseMemObject(out_y_plane);
    clReleaseMemObject(out_nv12_image);

    return EXIT_SUCCESS;
}
