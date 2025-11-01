#ifndef VISION_CROP_H
#define VISION_CROP_H

#include "vision/core/common/VTensor.h"
#include "vision/core/common/VStructs.h"

namespace aura::va_cv {

class Crop {
public:
    static void crop(const vision::VTensor& src, vision::VTensor& dst, const vision::VRect& rect);

    static void crop(const vision::VTensor &src, int &srcFormat, vision::VTensor &dst, const vision::VRect &rect);

private:
    static void cropOpencv(const vision::VTensor &src, int &format, vision::VTensor &dst, const vision::VRect &rect);

    static void crop_opencv(const vision::VTensor& src, vision::VTensor& dst, const vision::VRect& rect);

    static void crop_naive(const vision::VTensor& src, vision::VTensor& dst, const vision::VRect& rect);

    static void crop_sse(const vision::VTensor& src, vision::VTensor& dst, const vision::VRect& rect);

    static void crop_neon(const vision::VTensor& src, vision::VTensor& dst, const vision::VRect& rect);

    static void crop_neon_hwc_rgb_ir(const vision::VTensor& src, vision::VTensor& dst,
									 int crop_left, int crop_top, int crop_width, int crop_height);

    static void crop_neon_chw_rgb(const vision::VTensor& src, vision::VTensor& dst,
								  int crop_left, int crop_top, int crop_width, int crop_height);

    static void crop_naive_hwc_rgb(const vision::VTensor& src, vision::VTensor& dst,
								   int crop_left, int crop_top, int crop_width, int crop_height);

    static void crop_naive_chw(const vision::VTensor& src, vision::VTensor& dst,
							   int crop_left, int crop_top, int crop_width, int crop_height);
};

} // namespace aura::va_cv

#endif //VISION_CROP_H
