//
// Created by LiWendong on 22-9-13.
//

#if defined (BUILD_FASTCV)

#include "FastcvUtil.h"

using namespace aura::va_cv;

namespace aura::vision {

fcvBorderType vision::FastcvUtil::cvtBorderType(VBorderMode opencvType) {
    fcvBorderType fcvType;
    switch (opencvType) {
    case BORDER_REPLICATE:
        fcvType = FASTCV_BORDER_REPLICATE;
        break;
    case BORDER_CONSTANT:
        fcvType = FASTCV_BORDER_CONSTANT;
        break;
    default:
        fcvType = FASTCV_BORDER_UNDEFINED;
    }
    return fcvType;
}

fcvInterpolationType vision::FastcvUtil::cvtInterpolationType(VInterMode opencvType) {
    fcvInterpolationType fcvType;
    switch (opencvType) {
        case INTER_NEAREST:
            fcvType = FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
            break;
        case INTER_LINEAR:
            fcvType = FASTCV_INTERPOLATION_TYPE_BILINEAR;
            break;
        case INTER_AREA:
            fcvType = FASTCV_INTERPOLATION_TYPE_AREA;
            break;
        default:
            fcvType = FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
    }
    return fcvType;
}

}

#endif