#include "imencode.h"

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#include "util/TensorConverter.h"

namespace aura::va_cv {

void ImEncode::imencode(const vision::VTensor& src, std::vector<unsigned char>& buf, const char* format) {
#ifdef USE_OPENCV
    const auto& mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
#ifdef WITH_OCV_HIGHGUI
    cv::imencode(format, mat_src, buf);
#endif
#endif // USE_OPENCV
}

} // namespace aura::va_cv