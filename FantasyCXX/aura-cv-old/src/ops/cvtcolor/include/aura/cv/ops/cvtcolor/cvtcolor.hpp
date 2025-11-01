//
// Created by Frewen.Wang on 25-9-14.
//
#pragma once

#include "aura/utils/core/types/defs.hpp"
#include "aura/cv/ops/core/base_op.hpp"

namespace aura::cv
{

class AURA_EXPORTS CvtColor : public BaseOp
{
public:
    CvtColor(Context *ctx, const OpTarget &target = OpTarget::Default());

    Status SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type);
};

AURA_EXPORTS Status ICvtColor(Context *ctx, const std::vector<Mat> &src, std::vector<Mat> &dst, CvtColorType type, const OpTarget &target = OpTarget::Default());



/**
 * 定义图像格式转换的类型。注意需要和OpenCV保持同步
 */
enum class CvtColorType {
    INVALID = 0,
    BGR2BGRA = 100,
    RGB2RGBA = BGR2BGRA,
    BGRA2BGR,
    RGBA2RGB = BGRA2BGR,
    BGR2RGB,
    RGB2BGR = BGR2RGB,
    BGR2GRAY,
    RGB2GRAY,
    GRAY2BGR,
    GRAY2RGB = GRAY2BGR,
    GRAY2BGRA,
    GRAY2RGBA = GRAY2BGRA,
    BGRA2GRAY,
    RGBA2GRAY,

    RGB2YUV_NV12 = 200,
    RGB2YUV_NV21,
    RGB2YUV_YU12,
    RGB2YUV_YV12,
    RGB2YUV_Y444,
    RGB2YUV_NV12_601,
    RGB2YUV_NV21_601,
    RGB2YUV_YU12_601,
    RGB2YUV_YV12_601,
    RGB2YUV_Y444_601,
    RGB2YUV_NV12_P010,
    RGB2YUV_NV21_P010,

    YUV2RGB_NV12 = 300,
    YUV2RGB_NV21,
    YUV2RGB_YU12,
    YUV2RGB_YV12,
    YUV2RGB_Y422,
    YUV2RGB_UYVY = YUV2RGB_Y422,
    YUV2RGB_YUYV,
    YUV2RGB_YUY2 = YUV2RGB_YUYV,
    YUV2RGB_YVYU,
    YUV2RGB_Y444,
    YUV2RGB_NV12_601,
    YUV2RGB_NV21_601,
    YUV2RGB_YU12_601,
    YUV2RGB_YV12_601,
    YUV2RGB_Y422_601,
    YUV2RGB_UYVY_601 = YUV2RGB_Y422_601,
    YUV2RGB_YUYV_601,
    YUV2RGB_YUY2_601 = YUV2RGB_YUYV_601,
    YUV2RGB_YVYU_601,
    YUV2RGB_Y444_601,
    YUV2RGB_NV12_601_P010,
    YUV2RGB_NV21_601_P010,

    BAYERBG2BGR = 400,
    BAYERGB2BGR,
    BAYERRG2BGR,
    BAYERGR2BGR,
};
}
