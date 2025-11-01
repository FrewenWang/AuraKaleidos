#if defined (BUILD_FASTCV)
#pragma once

#include "fastcv.h"
#include "cv.h"

namespace aura::vision {

class FastcvUtil {
public:
    static fcvBorderType cvtBorderType(va_cv::VBorderMode opencvType);

    static fcvInterpolationType cvtInterpolationType(va_cv::VInterMode opencvType);

};

}
#endif