#ifndef VISION_CVT_COLOR_H
#define VISION_CVT_COLOR_H

#include "vision/core/common/VTensor.h"
#include "vision/core/common/VConstants.h"

namespace aura::va_cv {

class CvtColor {
public:
    static void cvt_color(const vision::VTensor& src, vision::VTensor& dst, int code);

private:
    static void cvt_color_naive(const vision::VTensor& src, vision::VTensor& dst, int code);
#ifdef BUILD_OPENCV
    static void cvt_color_opencv(const vision::VTensor& src, vision::VTensor& dst, int code);
#endif
    static void cvt_color_sse(const vision::VTensor& src, vision::VTensor& dst, int code);
#ifdef BUILD_FASTCV
    static void cvt_color_fastcv(const vision::VTensor& src, vision::VTensor& dst, int code);
#endif

#if defined (USE_NEON) and __ARM_NEON
    static void cvt_color_neon(const vision::VTensor& src, vision::VTensor& dst, int code);
    static void nv_to_bgr_neon(const uint8_t* src, uint8_t* dst, int srcw, int srch, int x_num, int y_num);
#endif
    static void nv_to_bgr_naive(const unsigned char* src, unsigned char* dst, int srcw, int srch, int x_num, int y_num);
};

} // namespace aura::va_cv

#endif //VISION_CVT_COLOR_H
