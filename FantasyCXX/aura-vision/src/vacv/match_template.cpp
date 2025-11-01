#include "match_template.h"

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#include "util/TensorConverter.h"

namespace aura::va_cv {

using namespace aura::vision;

void MatchTemplate::match_template(const vision::VTensor& src, const vision::VTensor& target,
								   vision::VTensor& result, int method) {
#ifdef USE_OPENCV
    match_template_opencv(src, target, result, method);
#else
#if defined (USE_NEON) and __ARM_NEON
    match_template_neon(src, target, result, method);
#elif defined (USE_SSE)
    match_template_sse(src, target, result, method);
#else
    match_template_naive(src, target, result, method);
#endif
#endif // USE_OPENCV
}

void MatchTemplate::match_template_opencv(const vision::VTensor& src, const vision::VTensor& target,
										  vision::VTensor& result, int method) {
#ifdef USE_OPENCV
    const auto& mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    const auto& mat_target = vision::TensorConverter::convert_to<cv::Mat>(target);
    cv::Mat mat_dst;
    cv::matchTemplate(mat_src, mat_target, mat_dst, method);
    result = vision::TensorConverter::convert_from<cv::Mat>(mat_dst, true);
#endif // USE_OPENCV
}

void MatchTemplate::minMaxIdx(const vision::VTensor& src, double* minVal, double* maxVal,
							  int* minIdx, int* maxIdx, const vision::VTensor& mask) {
#ifdef USE_OPENCV
    const auto& mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    const auto& mat_mask = vision::TensorConverter::convert_to<cv::Mat>(mask);
    cv::minMaxIdx(mat_src, minVal, maxVal, minIdx, maxIdx, mat_mask);
#endif // USE_OPENCV
}

void MatchTemplate::match_template_naive(const vision::VTensor& src, const vision::VTensor& target,
										 vision::VTensor& result, int method) {
    // todo:
}

void MatchTemplate::match_template_sse(const vision::VTensor& src, const vision::VTensor& target,
									   vision::VTensor& result, int method) {
    // todo:
}

void MatchTemplate::match_template_neon(const vision::VTensor& src, const vision::VTensor& target,
										vision::VTensor& result, int method) {
    // todo:
}

} // namespace aura::va_cv