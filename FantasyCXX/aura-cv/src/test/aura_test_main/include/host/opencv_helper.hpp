

#ifndef AURA_TEST_TEST_MAIN_HOST_OPENCV_HELPER_HPP__
#define AURA_TEST_TEST_MAIN_HOST_OPENCV_HELPER_HPP__

#include "aura/runtime/mat.h"
#include "aura/ops/core.h"

#include <opencv2/opencv.hpp>

#include <map>

namespace aura
{

AURA_INLINE DT_S32 ElemTypeToOpencv(ElemType elem_type, DT_S32 channel)
{
    std::map<ElemType, DT_S32> type_map
    {
        {ElemType::U8, CV_8U},   {ElemType::S8, CV_8S},
        {ElemType::U16, CV_16U}, {ElemType::S16, CV_16S},
        {ElemType::U32, CV_32S}, {ElemType::S32, CV_32S},
        {ElemType::F16, CV_16F}, {ElemType::F32, CV_32F},
        {ElemType::F64, CV_64F},
    };

    DT_S32 cv_elem_type = (type_map.count(elem_type) > 0) ? type_map[elem_type] : -1;

    DT_S32 type = -1;

    if(cv_elem_type != -1)
    {
        type = CV_MAKETYPE(cv_elem_type, channel);
    }

    return type;
}

AURA_INLINE cv::Mat MatToOpencv(Mat &mat)
{
    DT_S32 width  = mat.GetSizes().m_width;
    DT_S32 height = mat.GetSizes().m_height;
    DT_S32 stride = mat.GetStrides().m_width;

    DT_S32 cv_type = ElemTypeToOpencv(mat.GetElemType(), mat.GetSizes().m_channel);

    return cv::Mat(height, width, cv_type, mat.GetData(), stride);
}

AURA_INLINE DT_S32 BorderTypeToOpencv(BorderType border_type)
{
    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            return cv::BorderTypes::BORDER_CONSTANT;
        }
        case BorderType::REPLICATE:
        {
            return cv::BorderTypes::BORDER_REPLICATE;
        }
        case BorderType::REFLECT_101:
        {
            return cv::BorderTypes::BORDER_REFLECT101;
        }
        default:
        {
            return -1;
        }
    }
}

AURA_INLINE DT_S32 GetCVDepth(ElemType type)
{
    std::map<ElemType, DT_S32> type_map
    {
        {ElemType::U8, CV_8U},   {ElemType::S8, CV_8S},
        {ElemType::U16, CV_16U}, {ElemType::S16, CV_16S},
        {ElemType::U32, CV_32S}, {ElemType::S32, CV_32S},
        {ElemType::F16, CV_16F}, {ElemType::F32, CV_32F},
        {ElemType::F64, CV_64F},
    };

    DT_S32 cv_depth = (type_map.count(type) > 0) ? type_map[type] : -1;
    return cv_depth;
}

} // namespace aura

#endif // AURA_TEST_TEST_MAIN_HOST_OPENCV_HELPER_HPP__
