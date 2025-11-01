#ifndef VISION_MATCH_TEMPLATE_H
#define VISION_MATCH_TEMPLATE_H

#include "vision/core/common/VTensor.h"

namespace aura::va_cv {

class MatchTemplate {
public:
    static void match_template(const vision::VTensor &src, const vision::VTensor &target,
                               vision::VTensor &result, int method);

    static void minMaxIdx(const vision::VTensor &src,
                          double *minVal,
                          double *maxVal,
                          int *minIdx = nullptr,
                          int *maxIdx = nullptr,
                          const vision::VTensor &mask = vision::VTensor());

private:
    static void match_template_opencv(const vision::VTensor &src, const vision::VTensor &target,
                                      vision::VTensor &result, int method);

    static void match_template_naive(const vision::VTensor &src, const vision::VTensor &target,
                                     vision::VTensor &result, int method);

    static void match_template_sse(const vision::VTensor &src, const vision::VTensor &target,
                                   vision::VTensor &result, int method);

    static void match_template_neon(const vision::VTensor &src, const vision::VTensor &target,
                                    vision::VTensor &result, int method);
};

} // namespace aura::va_cv

#endif //VISION_MATCH_TEMPLATE_H
