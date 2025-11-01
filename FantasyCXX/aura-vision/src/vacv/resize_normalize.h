#ifndef VISION_RESIZE_NORMALIZE_H
#define VISION_RESIZE_NORMALIZE_H

#include "vision/core/common/VTensor.h"
#include "cv.h"

namespace aura::va_cv {

class ResizeNormalize {
public:
    static void resize_normalize(const vision::VTensor& src, vision::VTensor& dst,
								 VSize dsize, double fx = 0, double fy = 0,
								 int interpolation = INTER_LINEAR,
								 const vision::VTensor& mean = vision::VTensor(),
								 const vision::VTensor& stddev = vision::VTensor());

    static void resize_normalize(const cv::Mat& src, cv::Mat& dst,
                                 VSize dsize, double fx = 0, double fy = 0,
                                 int interpolation = INTER_LINEAR,
                                 const vision::VTensor& mean = vision::VTensor(),
                                 const vision::VTensor& stddev = vision::VTensor());

private:
    static void resize_normalize_opencv(const vision::VTensor& src, vision::VTensor& dst,
										VSize dsize, double fx = 0, double fy = 0,
										int interpolation = INTER_LINEAR,
										const vision::VTensor& mean = vision::VTensor(),
										const vision::VTensor& stddev = vision::VTensor());

    static void resize_normalize_naive(const vision::VTensor& src, vision::VTensor& dst,
									   VSize dsize, double fx = 0, double fy = 0,
									   int interpolation = INTER_LINEAR,
									   const vision::VTensor& mean = vision::VTensor(),
									   const vision::VTensor& stddev = vision::VTensor());

    static void resize_normalize_sse(const vision::VTensor& src, vision::VTensor& dst,
									 VSize dsize, double fx = 0, double fy = 0,
									 int interpolation = INTER_LINEAR,
									 const vision::VTensor& mean = vision::VTensor(),
									 const vision::VTensor& stddev = vision::VTensor());

    static void resize_normalize_neon(const vision::VTensor& src, vision::VTensor& dst,
									  VSize dsize, double fx = 0, double fy = 0,
									  int interpolation = INTER_LINEAR,
									  const vision::VTensor& mean = vision::VTensor(),
									  const vision::VTensor& stddev = vision::VTensor());

#if defined (BUILD_FASTCV)
    static void resize_normalize_fastcv(const vision::VTensor& src, vision::VTensor& dst,
                                        VSize dsize, double fx = 0, double fy = 0,
                                        int interpolation = INTER_LINEAR,
                                        const vision::VTensor& mean = vision::VTensor(),
                                        const vision::VTensor& stddev = vision::VTensor());
#endif
};

} // namespace aura::va_cv

#endif //VISION_RESIZE_NORMALIZE_H
