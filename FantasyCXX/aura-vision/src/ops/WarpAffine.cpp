//
// Created by LiWendong on 23-3-15.
//

#include "WarpAffine.h"

namespace aura::vision {
namespace op {

void WarpAffine::warpAffine(cv::Mat &src, cv::Mat &dst, cv::Mat &M, cv::Size &dsize,
                            const int &flags, const int &borderMode, const cv::Scalar &borderValue) {
    cv::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue);
}

} // namespace op
} // namespace aura::vision