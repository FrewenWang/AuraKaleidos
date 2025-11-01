//--------------------------------------------------------------------------------------
// File: qcom_extended_query_image_info.cpp
// Desc: This program shows how to the cl_qcom_extended_query_image_info extensions by
//       querying different image parameters for various image formats.
//
// Author:      QUALCOMM
//
//          Copyright (c) 2021 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

// Std includes
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <array>
#include <tuple>

// Project includes
#include "util/cl_wrapper.h"
#include "util/util.h"

// Library includes
#include <CL/cl.h>

#define CL_ASSERT_ERRCODE(err)                                                                  \
    if ((err) != CL_SUCCESS)                                                                    \
    {                                                                                           \
        std::cerr << "OpenCL API error code " << (err) << " on line " << __LINE__ << "\n";      \
        return EXIT_FAILURE;                                                                    \
    }
int main(int argc, char** argv)
{
    if (argc != 1)
    {
        std::cerr << "Usage: " << argv[0] << "\n"
                                             "\n"
                                             "This example shows how to use the cl_qcom_extended_query_image_info extension.\n";
        return EXIT_FAILURE;
    }

    cl_wrapper       wrapper;
    cl_device_id     device          = wrapper.get_device_id();
    cl_int           err             = CL_SUCCESS;

    /*
     * Step 0: Confirm the required OpenCL extensions are supported
     */
    if(!wrapper.check_extension_support("cl_qcom_extended_query_image_info")) {
        std::cerr << "Extension cl_qcom_extended_query_image_info not supported on this device\n";
        return EXIT_FAILURE;
    }

    /*
     * Step 1: Create the image descriptors. You do NOT need to create the image to use the
     * cl_qcom_extended_query_image_info extension
     */

    // RGBA
    cl_image_format rgba_format = {};
    rgba_format.image_channel_order = CL_RGBA;
    rgba_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc rgba_desc = {};
    rgba_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    rgba_desc.image_width = 4000;
    rgba_desc.image_height = 3000;

    // BayerMipi
    cl_image_format bayer_mipi_format;
    bayer_mipi_format.image_channel_order     = CL_QCOM_BAYER;
    bayer_mipi_format.image_channel_data_type = CL_QCOM_UNORM_MIPI10;

    cl_image_desc bayer_mipi_desc;
    bayer_mipi_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    bayer_mipi_desc.image_width     = 2000;
    bayer_mipi_desc.image_height    = 1500;

    // NV12
    cl_image_format nv12_format = {};
    nv12_format.image_channel_order = CL_QCOM_NV12;
    nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc nv12_desc = {};
    nv12_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    nv12_desc.image_width = 1000;
    nv12_desc.image_height = 500;

    // NV12_Y
    cl_image_format nv12_y_format = {};
    nv12_y_format.image_channel_order = CL_QCOM_NV12_Y;
    nv12_y_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc nv12_y_desc = {};
    nv12_y_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    nv12_y_desc.image_width = 1000;
    nv12_y_desc.image_height = 500;

    // NV12_UV
    cl_image_format nv12_uv_format = {};
    nv12_uv_format.image_channel_order = CL_QCOM_NV12_UV;
    nv12_uv_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc nv12_uv_desc = {};
    nv12_uv_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    nv12_uv_desc.image_width = 1000;
    nv12_uv_desc.image_height = 500;

    // compressed NV12
    cl_image_format ubwc_nv12_format = {};
    ubwc_nv12_format.image_channel_order = CL_QCOM_COMPRESSED_NV12;
    ubwc_nv12_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc ubwc_nv12_desc = {};
    ubwc_nv12_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    ubwc_nv12_desc.image_width = 1920;
    ubwc_nv12_desc.image_height = 1080;

    // compressed NV12_Y
    cl_image_format ubwc_nv12_y_format = {};
    ubwc_nv12_y_format.image_channel_order = CL_QCOM_COMPRESSED_NV12_Y;
    ubwc_nv12_y_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc ubwc_nv12_y_desc = {};
    ubwc_nv12_y_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    ubwc_nv12_y_desc.image_width = 1920;
    ubwc_nv12_y_desc.image_height = 1080;

    // compressed NV12_UV
    cl_image_format ubwc_nv12_uv_format = {};
    ubwc_nv12_uv_format.image_channel_order = CL_QCOM_COMPRESSED_NV12_UV;
    ubwc_nv12_uv_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc ubwc_nv12_uv_desc = {};
    ubwc_nv12_uv_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    ubwc_nv12_uv_desc.image_width = 1920;
    ubwc_nv12_uv_desc.image_height = 1080;

    // RGBA Image Array
    cl_image_format rgba_array_format = {};
    rgba_array_format.image_channel_order = CL_RGBA;
    rgba_array_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc  rgba_array_desc = {};
    rgba_array_desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    rgba_array_desc.image_width = 4000;
    rgba_array_desc.image_height = 3000;
    rgba_array_desc.image_depth = 1;
    rgba_array_desc.image_array_size = 2;

    std::array<std::tuple<cl_image_format, cl_image_desc>, 9> image_descriptions = {
        std::make_tuple(rgba_format, rgba_desc),
        std::make_tuple(bayer_mipi_format, bayer_mipi_desc),
        std::make_tuple(nv12_format, nv12_desc),
        std::make_tuple(nv12_y_format, nv12_y_desc),
        std::make_tuple(nv12_uv_format, nv12_uv_desc),
        std::make_tuple(ubwc_nv12_format, ubwc_nv12_desc),
        std::make_tuple(ubwc_nv12_y_format, ubwc_nv12_y_desc),
        std::make_tuple(ubwc_nv12_uv_format, ubwc_nv12_uv_desc),
        std::make_tuple(rgba_array_format, rgba_array_desc),
    };

    /*
     * Step 2: Query image attributes
     */
    size_t row_pitch, slice_pitch, row_alignment, height_alignment, slice_alignment, image_size, element_size, base_address_alignment;
    for(size_t i = 0; i < image_descriptions.size(); i++)
    {
        cl_image_format format = std::get<0>(image_descriptions[i]);
        cl_image_desc desc     = std::get<1>(image_descriptions[i]);
        bool is_2d_array       = desc.image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY;

        err = clQueryImageInfoQCOM(device, CL_MEM_READ_ONLY, &format, &desc, CL_IMAGE_ROW_PITCH, sizeof(row_pitch), &row_pitch, nullptr);
        CL_ASSERT_ERRCODE(err);

        err = clQueryImageInfoQCOM(device, CL_MEM_READ_ONLY, &format, &desc, CL_IMAGE_ROW_ALIGNMENT_QCOM, sizeof(row_alignment), &row_alignment, nullptr);
        CL_ASSERT_ERRCODE(err);

        err = clQueryImageInfoQCOM(device, CL_MEM_READ_ONLY, &format, &desc, CL_IMAGE_HEIGHT_ALIGNMENT_QCOM, sizeof(height_alignment), &height_alignment, nullptr);
        CL_ASSERT_ERRCODE(err);

        err = clQueryImageInfoQCOM(device, CL_MEM_READ_ONLY, &format, &desc, CL_IMAGE_SIZE_QCOM, sizeof(image_size), &image_size, nullptr);
        CL_ASSERT_ERRCODE(err);

        err = clQueryImageInfoQCOM(device, CL_MEM_READ_ONLY, &format, &desc, CL_IMAGE_ELEMENT_SIZE, sizeof(element_size), &element_size, nullptr);
        CL_ASSERT_ERRCODE(err);

        err = clQueryImageInfoQCOM(device, CL_MEM_READ_ONLY, &format, &desc, CL_IMAGE_BASE_ADDRESS_ALIGNMENT_QCOM, sizeof(base_address_alignment), &base_address_alignment, nullptr);
        CL_ASSERT_ERRCODE(err);

        if(is_2d_array)
        {
            err = clQueryImageInfoQCOM(device, CL_MEM_READ_ONLY, &format, &desc, CL_IMAGE_SLICE_PITCH, sizeof(slice_pitch), &slice_pitch, nullptr);
            CL_ASSERT_ERRCODE(err);

            err = clQueryImageInfoQCOM(device, CL_MEM_READ_ONLY, &format, &desc, CL_IMAGE_SLICE_ALIGNMENT_QCOM, sizeof(slice_alignment), &slice_alignment, nullptr);
            CL_ASSERT_ERRCODE(err);
        }

        std::cout << "\nimage channel order = " << get_image_order(format) << (is_2d_array ? "_2D_ARRAY" : "")
                  << ", image_channel_data_type = " << get_image_datatype(format)
                  << ", image_width = " << desc.image_width
                  << ", image height = " << desc.image_height << std::endl;

        std::cout << "\tRow Pitch: "              << row_pitch << std::endl;
        std::cout << "\tRow Alignment: "          << row_alignment << std::endl;
        std::cout << "\tHeight Alignment: "       << height_alignment << std::endl;
        std::cout << "\tImage Size: "             << image_size << std::endl;
        std::cout << "\tElement Size: "           << element_size << std::endl;
        std::cout << "\tBase Address Alignment: " << base_address_alignment << std::endl;
        if(is_2d_array)
        {
            std::cout << "\tSlice Pitch: "            << slice_pitch << std::endl;
            std::cout << "\tSlice Alignment: "        << slice_alignment << std::endl;
        }
    }

    return EXIT_SUCCESS;
}