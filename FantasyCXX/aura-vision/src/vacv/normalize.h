#ifndef VISION_NORMALIZE_H
#define VISION_NORMALIZE_H

#include "vision/core/common/VTensor.h"

namespace aura::va_cv {

class Normalize {
public:
    static void normalize(const vision::VTensor &src, vision::VTensor &dst,
                          const vision::VTensor &mean, const vision::VTensor &stddev);

private:
    static void normalize_opencv(const vision::VTensor &src, vision::VTensor &dst,
                                 const vision::VTensor &mean, const vision::VTensor &stddev);

    static void normalize_naive(const vision::VTensor &src, vision::VTensor &dst,
                                const vision::VTensor &mean, const vision::VTensor &stddev);

    static void normalize_sse(const vision::VTensor &src, vision::VTensor &dst,
                              const vision::VTensor &mean, const vision::VTensor &stddev);

    static void normalize_neon(const vision::VTensor &src, vision::VTensor &dst,
                               const vision::VTensor &mean, const vision::VTensor &stddev);
};

} // namespace aura::va_cv

#endif //VISION_NORMALIZE_H
