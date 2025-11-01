//--------------------------------------------------------------------------------------
// File: image_matrix_transpose.cpp
// Desc: Demonstrates transposing matrices with images
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

static const char *PROGRAM_SOURCE = R"(
    __kernel void transpose(__read_only  image2d_t matrix,
                            __write_only image2d_t matrix_t)
    {
        const int wid_x     = get_global_id(0);
        const int wid_y     = get_global_id(1);
        const float4 rows[] = {
            read_imagef(matrix, (int2)(wid_x, 4 * wid_y + 0)),
            read_imagef(matrix, (int2)(wid_x, 4 * wid_y + 1)),
            read_imagef(matrix, (int2)(wid_x, 4 * wid_y + 2)),
            read_imagef(matrix, (int2)(wid_x, 4 * wid_y + 3)),
            };
        write_imagef(matrix_t, (int2)(wid_y, 4 * wid_x + 0), (float4)(rows[0].x, rows[1].x, rows[2].x, rows[3].x));
        write_imagef(matrix_t, (int2)(wid_y, 4 * wid_x + 1), (float4)(rows[0].y, rows[1].y, rows[2].y, rows[3].y));
        write_imagef(matrix_t, (int2)(wid_y, 4 * wid_x + 2), (float4)(rows[0].z, rows[1].z, rows[2].z, rows[3].z));
        write_imagef(matrix_t, (int2)(wid_y, 4 * wid_x + 3), (float4)(rows[0].w, rows[1].w, rows[2].w, rows[3].w));
    }
)";

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Please specify input file.\n"
                     "\n"
                     "Usage: " << argv[0] << " <input matrix> [<output file>]\n"
                     "Given a matrix, computes its transpose.\n"
                     "There is no size restriction for the matrix, but it may be padded with extra elements.\n"
                     "If no file is specified for the output, then it is written to stdout.\n";
        return EXIT_FAILURE;
    }

    const std::string matrix_a_filename(argv[1]);
    const bool        output_to_file = argc >= 3;
    const matrix_t    matrix_a       = load_matrix(matrix_a_filename);
    const std::string output_filename(output_to_file ? argv[2] : "");

    matrix_t matrix_b;
    matrix_b.width              = matrix_a.height;
    matrix_b.height             = matrix_a.width;
    const size_t matrix_b_size  = matrix_b.width * matrix_b.height;
    matrix_b.elements.resize(matrix_b_size);

    cl_wrapper          wrapper;
    cl_program          program       = wrapper.make_program(&PROGRAM_SOURCE, 1);
    cl_kernel           kernel        = wrapper.make_kernel("transpose", program);
    cl_context          context       = wrapper.get_context();
    cl_command_queue    command_queue = wrapper.get_command_queue();
    struct dma_buf_sync buf_sync      = {};
    cl_event            unmap_event   = nullptr;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported.
     */

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

    cl_int err = CL_SUCCESS;

    /*
     * Step 1: Create suitable dmabuf-backed images.
     */

    /*
     * Matrix A
     */

    cl_image_format matrix_a_format;
    matrix_a_format.image_channel_order     = CL_RGBA;
    matrix_a_format.image_channel_data_type = CL_FLOAT;

    cl_image_desc matrix_a_desc;
    std::memset(&matrix_a_desc, 0, sizeof(matrix_a_desc));
    matrix_a_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    matrix_a_desc.image_width     = ((matrix_a.width + 3) / 4);
    matrix_a_desc.image_height    = ((matrix_a.height + 3) / 4) * 4;
    matrix_a_desc.image_row_pitch = wrapper.get_image_row_pitch(matrix_a_format, matrix_a_desc);

    cl_mem_dmabuf_host_ptr matrix_a_buf = wrapper.make_buffer_for_nonplanar_image(matrix_a_format,
                                                                                       matrix_a_desc);
    cl_mem matrix_a_mem = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &matrix_a_format,
            &matrix_a_desc,
            &matrix_a_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for matrix A." << "\n";
        return err;
    }

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    if ( ioctl(matrix_a_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    char         *image_ptr;
    const size_t  origin[]          = {0, 0, 0};
    const size_t  matrix_a_region[] = {matrix_a_desc.image_width, matrix_a_desc.image_height, 1};
    size_t        row_pitch         = 0;
    image_ptr = static_cast<char *>(clEnqueueMapImage(
            command_queue,
            matrix_a_mem,
            CL_BLOCKING,
            CL_MAP_WRITE,
            origin,
            matrix_a_region,
            &row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " mapping matrix A image." << "\n";
        return err;
    }

    for (size_t i = 0; i < matrix_a_desc.image_height; ++i)
    {
        if (i < static_cast<size_t>(matrix_a.height))
        {
            const size_t unpadded_row_size = sizeof(cl_float) * matrix_a.width;
            std::memcpy(
                    image_ptr                + i * row_pitch,
                    matrix_a.elements.data() + i * matrix_a.width,
                    unpadded_row_size
            );
            const size_t remaining_bytes = row_pitch - unpadded_row_size;
            std::memset(image_ptr + (i * row_pitch) + unpadded_row_size, 0, remaining_bytes);
        }
        else
        {
            std::memset(image_ptr + i * row_pitch, 0, row_pitch);
        }
    }

    err = clEnqueueUnmapMemObject(command_queue, matrix_a_mem, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " unmapping matrix A image." << "\n";
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
    if ( ioctl(matrix_a_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer writes. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    /*
     * Matrix B
     */

    cl_image_format matrix_b_format;
    matrix_b_format.image_channel_order     = CL_RGBA;
    matrix_b_format.image_channel_data_type = CL_FLOAT;

    cl_image_desc matrix_b_desc;
    std::memset(&matrix_b_desc, 0, sizeof(matrix_b_desc));
    matrix_b_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    matrix_b_desc.image_width     = ((matrix_b.width + 3) / 4);
    matrix_b_desc.image_height    = ((matrix_b.height + 3) / 4) * 4;
    matrix_b_desc.image_row_pitch = wrapper.get_image_row_pitch(matrix_b_format, matrix_b_desc);

    cl_mem_dmabuf_host_ptr matrix_b_buf = wrapper.make_buffer_for_nonplanar_image(matrix_b_format,
                                                                                       matrix_b_desc);
    cl_mem matrix_b_mem = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
            &matrix_b_format,
            &matrix_b_desc,
            &matrix_b_buf,
            &err
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clCreateImage for matrix B." << "\n";
        return err;
    }

    /*
     * Step 2: Set up the kernel arguments
     */

    err = clSetKernelArg(kernel, 0, sizeof(matrix_a_mem), &matrix_a_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 0." << "\n";
        return err;
    }

    err = clSetKernelArg(kernel, 1, sizeof(matrix_b_mem), &matrix_b_mem);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clSetKernelArg for argument 1." << "\n";
        return err;
    }

    /*
     * Step 3: Run the kernel.
     */

    const size_t global_work_size[] = {matrix_a_desc.image_width, matrix_a_desc.image_height / 4};
    err = clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            2,
            nullptr,
            global_work_size,
            nullptr,
            0,
            nullptr,
            nullptr
    );
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueNDRangeKernel." << "\n";
        return err;
    }

    /*
     * Step 4: Copy the data out of the DMA buffer.
     */

    buf_sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if ( ioctl(matrix_b_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    const size_t  matrix_b_region[] = {matrix_b_desc.image_width, matrix_b_desc.image_height, 1};
    image_ptr = static_cast<char *>(clEnqueueMapImage(
            command_queue,
            matrix_b_mem,
            CL_BLOCKING,
            CL_MAP_READ,
            origin,
            matrix_b_region,
            &row_pitch,
            nullptr,
            0,
            nullptr,
            nullptr,
            &err
    ));
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueMapImage for matrix B." << "\n";
        return err;
    }

    for (size_t i = 0; i < static_cast<size_t>(matrix_b.height); ++i)
    {
        const size_t unpadded_row_size = sizeof(cl_float) * matrix_b.width;
        std::memcpy(
                matrix_b.elements.data() + i * matrix_b.width,
                image_ptr                + i * row_pitch,
                unpadded_row_size
        );
    }
    err = clEnqueueUnmapMemObject(command_queue, matrix_b_mem, image_ptr, 0, nullptr, &unmap_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error " << err << " with clEnqueueUnmapMemObject." << "\n";
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
    if ( ioctl(matrix_b_buf.dmabuf_filedesc, DMA_BUF_IOCTL_SYNC, &buf_sync) )
    {
        std::cerr << "Error preparing the cache for buffer reads. The DMA_BUF_IOCTL_SYNC operation returned " << strerror(errno) << "!\n";
        return errno;
    }

    clFinish(command_queue);

    if (output_to_file)
    {
        save_matrix(output_filename, matrix_b);
    }
    else
    {
        save_matrix(std::cout, matrix_b);
    }

    // Clean up cl resources that aren't automatically handled by cl_wrapper
    clReleaseMemObject(matrix_a_mem);
    clReleaseMemObject(matrix_b_mem);

    return EXIT_SUCCESS;
}
