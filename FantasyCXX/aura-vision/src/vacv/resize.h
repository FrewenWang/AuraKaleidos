#ifndef VISION_RESIZE_H
#define VISION_RESIZE_H

#include "cv.h"
#include "vision/core/common/VTensor.h"

namespace aura::va_cv {

class Resize {
public:
    static void resize(const vision::VTensor &src, vision::VTensor &dst,
                       VSize dsize, double fx = 0, double fy = 0,
                       int interpolation = INTER_LINEAR);
    
    static void resizeNoNormalize(const vision::VTensor &src, vision::VTensor &dst,
                                  VSize dsize, double fx = 0, double fy = 0,
                                  int interpolation = INTER_LINEAR);

private:
    static void resize_opencv(const vision::VTensor &src, vision::VTensor &dst,
                              VSize dsize, double fx = 0, double fy = 0,
                              int interpolation = INTER_LINEAR);

    static void resize_naive(const vision::VTensor &src, vision::VTensor &dst,
                             VSize dsize, double fx = 0, double fy = 0,
                             int interpolation = INTER_LINEAR);

    static void resize_sse(const vision::VTensor &src, vision::VTensor &dst,
                           VSize dsize, double fx = 0, double fy = 0,
                           int interpolation = INTER_LINEAR);

    static void resize_neon(const vision::VTensor &src, vision::VTensor &dst,
                            VSize dsize, double fx = 0, double fy = 0,
                            int interpolation = INTER_LINEAR);
    
    static void resizeWithFloatOpencv(const vision::VTensor &src, vision::VTensor &dst,
                                      VSize dsize, double fx = 0, double fy = 0,
                                      int interpolation = INTER_LINEAR);
    
    static void resizeWithFloatFastcv(const vision::VTensor &src, vision::VTensor &dst,
                                      VSize dsize, double fx = 0, double fy = 0,
                                      int interpolation = INTER_LINEAR);

#if defined (BUILD_FASTCV)
    static bool resize_fastcv(const vision::VTensor &src, vision::VTensor& dst, VSize dsize, double fx, double fy, int interpolation = INTER_LINEAR);
#endif
};

} // namespace aura::vision

#endif //VISION_RESIZE_H
