#ifndef VISION_WARP_AFFINE_NORMALIZE_H
#define VISION_WARP_AFFINE_NORMALIZE_H

#include "vision/core/common/VTensor.h"
#include "cv.h"

namespace aura::va_cv {

class WarpAffineNormalize {
public:
    static void warp_affine_normalize(const vision::VTensor &src, vision::VTensor &dst,
                                      const vision::VTensor &M, VSize dsize,
                                      int flags = INTER_LINEAR,
                                      int borderMode = BORDER_CONSTANT,
                                      const VScalar &borderValue = VScalar(),
                                      const vision::VTensor &mean = vision::VTensor(),
                                      const vision::VTensor &stddev = vision::VTensor());

    static void warp_affine_normalize(const vision::VTensor &src, vision::VTensor &dst,
                                      float scale, float rot, VSize dsize,
                                      const VScalar &aux_param = VScalar(),
                                      int flags = INTER_LINEAR,
                                      int borderMode = BORDER_CONSTANT,
                                      const VScalar &borderValue = VScalar(),
                                      const vision::VTensor &mean = vision::VTensor(),
                                      const vision::VTensor &stddev = vision::VTensor());

private:
    static void warp_affine_normalize_opencv(const vision::VTensor &src, vision::VTensor &dst,
                                             const vision::VTensor &M, VSize dsize,
                                             int flags = INTER_LINEAR,
                                             int borderMode = BORDER_CONSTANT,
                                             const VScalar &borderValue = VScalar(),
                                             const vision::VTensor &mean = vision::VTensor(),
                                             const vision::VTensor &stddev = vision::VTensor());

    static void warp_affine_normalize_naive(const vision::VTensor &src, vision::VTensor &dst,
                                            const vision::VTensor &M, VSize dsize,
                                            int flags = INTER_LINEAR,
                                            int borderMode = BORDER_CONSTANT,
                                            const VScalar &borderValue = VScalar(),
                                            const vision::VTensor &mean = vision::VTensor(),
                                            const vision::VTensor &stddev = vision::VTensor());

    static void warp_affine_normalize_sse(const vision::VTensor &src, vision::VTensor &dst,
                                          const vision::VTensor &M, VSize dsize,
                                          int flags = INTER_LINEAR,
                                          int borderMode = BORDER_CONSTANT,
                                          const VScalar &borderValue = VScalar(),
                                          const vision::VTensor &mean = vision::VTensor(),
                                          const vision::VTensor &stddev = vision::VTensor());

    static void warp_affine_normalize_neon(const vision::VTensor &src, vision::VTensor &dst,
                                           const vision::VTensor &M, VSize dsize,
                                           int flags = INTER_LINEAR,
                                           int borderMode = BORDER_CONSTANT,
                                           const VScalar &borderValue = VScalar(),
                                           const vision::VTensor &mean = vision::VTensor(),
                                           const vision::VTensor &stddev = vision::VTensor());

    static void warp_affine_normalize_opencv(const vision::VTensor &src, vision::VTensor &dst,
                                             float scale, float rot, VSize dsize,
                                             const VScalar &aux_param = VScalar(),
                                             int flags = INTER_LINEAR,
                                             int borderMode = BORDER_CONSTANT,
                                             const VScalar &borderValue = VScalar(),
                                             const vision::VTensor &mean = vision::VTensor(),
                                             const vision::VTensor &stddev = vision::VTensor());
};

} // namespace va_cv

#endif //VISION_WARP_AFFINE_NORMALIZE_H
