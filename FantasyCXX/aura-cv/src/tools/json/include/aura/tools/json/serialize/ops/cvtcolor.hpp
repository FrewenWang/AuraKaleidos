#ifndef AURA_TOOLS_JSON_SERIALIZE_OPS_CVTCOLOR_HPP__
#define AURA_TOOLS_JSON_SERIALIZE_OPS_CVTCOLOR_HPP__

#include "aura/tools/json/json.hpp"
#include "aura/ops/cvtcolor.h"

namespace aura
{

/**
 * @brief Define the json serialize method for CvtColorType enumeration.
 *
 */
AURA_JSON_SERIALIZE_ENUM(CvtColorType, {
    {CvtColorType::INVALID,          "Invalid"},
    {CvtColorType::BGR2BGRA,         "BGR2BGRA"},
    {CvtColorType::RGB2RGBA,         "BGR2BGRA"},
    {CvtColorType::BGRA2BGR,         "BGRA2BGR"},
    {CvtColorType::RGBA2RGB,         "BGRA2BGR"},
    {CvtColorType::BGR2RGB,          "BGR2RGB"},
    {CvtColorType::RGB2BGR,          "BGR2RGB"},
    {CvtColorType::BGR2GRAY,         "BGR2GRAY"},
    {CvtColorType::RGB2GRAY,         "RGB2GRAY"},
    {CvtColorType::GRAY2BGR,         "GRAY2BGR"},
    {CvtColorType::GRAY2RGB,         "GRAY2BGR"},
    {CvtColorType::GRAY2BGRA,        "GRAY2BGRA"},
    {CvtColorType::GRAY2RGBA,        "GRAY2BGRA"},
    {CvtColorType::BGRA2GRAY,        "BGRA2GRAY"},
    {CvtColorType::RGBA2GRAY,        "RGBA2GRAY"},
    {CvtColorType::RGB2YUV_NV12,     "RGB2YUV_NV12"},
    {CvtColorType::RGB2YUV_NV21,     "RGB2YUV_NV21"},
    {CvtColorType::RGB2YUV_YU12,     "RGB2YUV_YU12"},
    {CvtColorType::RGB2YUV_YV12,     "RGB2YUV_YV12"},
    {CvtColorType::RGB2YUV_Y444,     "RGB2YUV_Y444"},
    {CvtColorType::RGB2YUV_NV12_601, "RGB2YUV_NV12_601"},
    {CvtColorType::RGB2YUV_NV21_601, "RGB2YUV_NV21_601"},
    {CvtColorType::RGB2YUV_YU12_601, "RGB2YUV_YU12_601"},
    {CvtColorType::RGB2YUV_YV12_601, "RGB2YUV_YV12_601"},
    {CvtColorType::RGB2YUV_Y444_601, "RGB2YUV_Y444_601"},
    {CvtColorType::YUV2RGB_NV12,     "YUV2RGB_NV12"},
    {CvtColorType::YUV2RGB_NV21,     "YUV2RGB_NV21"},
    {CvtColorType::YUV2RGB_YU12,     "YUV2RGB_YU12"},
    {CvtColorType::YUV2RGB_YV12,     "YUV2RGB_YV12"},
    {CvtColorType::YUV2RGB_Y422,     "YUV2RGB_Y422"},
    {CvtColorType::YUV2RGB_UYVY,     "YUV2RGB_Y422"},
    {CvtColorType::YUV2RGB_YUYV,     "YUV2RGB_YUYV"},
    {CvtColorType::YUV2RGB_YUY2,     "YUV2RGB_YUYV"},
    {CvtColorType::YUV2RGB_YVYU,     "YUV2RGB_YVYU"},
    {CvtColorType::YUV2RGB_Y444,     "YUV2RGB_Y444"},
    {CvtColorType::YUV2RGB_NV12_601, "YUV2RGB_NV12_601"},
    {CvtColorType::YUV2RGB_NV21_601, "YUV2RGB_NV21_601"},
    {CvtColorType::YUV2RGB_YU12_601, "YUV2RGB_YU12_601"},
    {CvtColorType::YUV2RGB_YV12_601, "YUV2RGB_YV12_601"},
    {CvtColorType::YUV2RGB_Y422_601, "YUV2RGB_Y422_601"},
    {CvtColorType::YUV2RGB_UYVY_601, "YUV2RGB_Y422_601"},
    {CvtColorType::YUV2RGB_YUYV_601, "YUV2RGB_YUYV_601"},
    {CvtColorType::YUV2RGB_YUY2_601, "YUV2RGB_YUYV_601"},
    {CvtColorType::YUV2RGB_YVYU_601, "YUV2RGB_YVYU_601"},
    {CvtColorType::YUV2RGB_Y444_601, "YUV2RGB_Y444_601"},
    {CvtColorType::BAYERBG2BGR,      "BAYERBG2BGR"},
    {CvtColorType::BAYERGB2BGR,      "BAYERGB2BGR"},
    {CvtColorType::BAYERRG2BGR,      "BAYERRG2BGR"},
    {CvtColorType::BAYERGR2BGR,      "BAYERGR2BGR"},
})

} // namespace aura

#endif // AURA_TOOLS_JSON_SERIALIZE_OPS_CVTCOLOR_HPP__