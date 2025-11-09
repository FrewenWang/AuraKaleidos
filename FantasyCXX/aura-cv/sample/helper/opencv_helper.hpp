#ifndef AURA_OPENCV_HELPER_HPP__
#define AURA_OPENCV_HELPER_HPP__

// aura2.0 headers
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/context.h"

// opencv headers
#include "opencv2/core/mat.hpp"

#include <map>

/**
 * @brief Converts a aura::ElemType to an OpenCV element type.
 *
 * @param elem_type The input aura::ElemType.
 * @param channel The number of channels.
 *
 * @return The converted OpenCV element type, or -1 if conversion fails.
 */
AURA_INLINE DT_S32 AuraElemType2OpenCV(aura::ElemType elem_type, DT_S32 channel)
{
    static const std::map<aura::ElemType, DT_S32> type_map =
    {
        {aura::ElemType::U8,  CV_8U},
        {aura::ElemType::S8,  CV_8S},
        {aura::ElemType::U16, CV_16U},
        {aura::ElemType::S16, CV_16S},
        {aura::ElemType::U32, CV_32S},
        {aura::ElemType::S32, CV_32S},
        {aura::ElemType::F16, CV_16F},
        {aura::ElemType::F32, CV_32F},
        {aura::ElemType::F64, CV_64F},
    };

    return (type_map.count(elem_type) > 0) ? CV_MAKETYPE(type_map.at(elem_type), channel) : -1;
}

/**
 * @brief Converts an OpenCV element type to a aura::ElemType.
 *
 * @param elem_type The input OpenCV element type.
 *
 * @return The converted aura::ElemType, or aura::ElemType::INVALID if conversion fails.
 */
AURA_INLINE aura::ElemType OpenCVElemType2Aura(DT_S32 elem_type)
{
    static const std::map<int, aura::ElemType> type_map =
    {
        {CV_8U,  aura::ElemType::U8},
        {CV_8S,  aura::ElemType::S8},
        {CV_16U, aura::ElemType::U16},
        {CV_16S, aura::ElemType::S16},
        {CV_32S, aura::ElemType::S32},
        {CV_16F, aura::ElemType::F16},
        {CV_32F, aura::ElemType::F32},
        {CV_64F, aura::ElemType::F64},
    };

    auto it = type_map.find(elem_type);
    if (it != type_map.end())
    {
        return it->second;
    }

    return aura::ElemType::INVALID;
}

/**
 * @brief Converts aura::Mat to cv::Mat.
 *
 * @param src The input aura::Mat.
 * @param deep_clone Whether to perform a deep clone (default is shallow copy).
 *
 * @return The converted cv::Mat, or an empty cv::Mat if conversion fails.
 */
AURA_INLINE cv::Mat AuraMat2OpenCVMat(aura::Context *ctx, aura::Mat &src, DT_BOOL deep_clone = DT_FALSE)
{
    if (!src.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "src mat is invalid");
        return cv::Mat();
    }

    DT_S32 width  = src.GetSizes().m_width;
    DT_S32 height = src.GetSizes().m_height;
    DT_S32 stride = src.GetStrides().m_width;

    DT_S32 cv_type = AuraElemType2OpenCV(src.GetElemType(), src.GetSizes().m_channel);
    if (-1 == cv_type)
    {
        AURA_ADD_ERROR_STRING(ctx, "unsupported elem type");
        return cv::Mat();
    }

    cv::Mat dst_shallow(height, width, cv_type, src.GetData(), stride);
    if (dst_shallow.empty())
    {
        AURA_ADD_ERROR_STRING(ctx, "empty cv mat for shallow copy");
        return cv::Mat();
    }

    if (deep_clone)
    {
        cv::Mat cv_mat_deep = dst_shallow.clone();
        if (cv_mat_deep.empty())
        {
            AURA_ADD_ERROR_STRING(ctx, "empty cv mat for deep clone");
            return cv::Mat();
        }

        return cv_mat_deep;
    }

    return dst_shallow;
}

/**
 * @brief Converts cv::Mat to aura::Mat by performing a deep clone.
 *
 * @param ctx The aura::Context pointer.
 * @param src The input OpenCV cv::Mat.
 * @param deep_clone Whether to perform a deep clone (default is shallow copy).
 *
 * @return The converted aura::Mat, or an empty aura::Mat if conversion fails.
 */
AURA_INLINE aura::Mat OpenCVMat2AuraMat(aura::Context *ctx, cv::Mat &src, DT_BOOL deep_clone = DT_FALSE)
{
    if (src.empty())
    {
        AURA_ADD_ERROR_STRING(ctx, "OpencvMatToAuraMat failed\n");
        return aura::Mat();
    }

    DT_S64 total_bytes = src.step[0] * src.rows;
    DT_S32 cv_data_type = src.type() & CV_MAT_DEPTH_MASK;

    aura::Buffer buffer = aura::Buffer(AURA_MEM_HEAP, total_bytes, total_bytes, src.data, src.data, 0);
    if (!buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "buffer is invalid\n");
        return aura::Mat();
    }

    aura::ElemType aura_elem_type = OpenCVElemType2Aura(cv_data_type);
    if (aura::ElemType::INVALID == aura_elem_type)
    {
        AURA_ADD_ERROR_STRING(ctx, "unsupported mat elem type \n");
        return aura::Mat();
    }

    aura::Mat dst_shallow(ctx, aura_elem_type, aura::Sizes3(src.rows, src.cols,
                          src.channels()), buffer, aura::Sizes(0, static_cast<DT_S32>(src.step[0])));
    if (!dst_shallow.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "dst_shallow init failed");
        return aura::Mat();
    }

    if (deep_clone)
    {
        aura::Mat dst_deep(ctx, aura_elem_type, dst_shallow.GetSizes(), AURA_MEM_DEFAULT);
        if (!dst_deep.IsValid())
        {
            AURA_ADD_ERROR_STRING(ctx, "dst_deep init failed");
            return aura::Mat();
        }

        if (dst_shallow.CopyTo(dst_deep) != aura::Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "CopyTo failed");
            return aura::Mat();
        }

        return dst_deep;
    }

    return dst_shallow;
}

#endif // AURA_OPENCV_HELPER_HPP__