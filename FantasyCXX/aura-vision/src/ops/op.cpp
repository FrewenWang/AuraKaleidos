#include "op.h"

#include "Memory.h"

#include "WarpAffine.h"

namespace aura::vision {
namespace op {

using namespace aura::vision;

// ---------------------------------------------------------------------------------------------------------------------
// 常规操作 Operator
// ---------------------------------------------------------------------------------------------------------------------

void memcpy(void *dst, const void *src, size_t size) {
    Memory::memoryCopy(dst, src, size);
}


// ---------------------------------------------------------------------------------------------------------------------
// 图像处理 Operator
// ---------------------------------------------------------------------------------------------------------------------

void warpAffine(cv::Mat &src, cv::Mat &dst, cv::Mat &M, cv::Size &dsize,
                const int &flags, const int &borderMode, const cv::Scalar &borderValue) {
    WarpAffine::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue);
}

} // namespace op
} // namespace aura::vision