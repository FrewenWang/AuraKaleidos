#include "cv.h"
#include "crop.h"
#include "cvt_color.h"
#include "imencode.h"
#include "match_template.h"
#include "normalize.h"
#include "resize.h"
#include "resize_normalize.h"
#include "warp_affine.h"
#include "warp_affine_normalize.h"

#include "vision/util/PerfUtil.h"
#include "opencv2/core/core.hpp"
#include "util/TensorConverter.h"

namespace aura::va_cv {

using namespace aura::vision;

void resize(const VTensor &src, VTensor &dst,
            VSize dsize, double fx, double fy,
            int interpolation) {
    PERF_AUTO(PerfUtil::global(), "resize")
    Resize::resize(src, dst, dsize, fx, fy, interpolation);
}

void resizeNoNormalize(const VTensor &src, VTensor &dst,
                       VSize dsize, double fx, double fy,
                       int interpolation) {
    PERF_AUTO(PerfUtil::global(), "resizeNoNormalize")
    Resize::resizeNoNormalize(src, dst, dsize, fx, fy, interpolation);
}

void cvt_color(const VTensor& src, VTensor& dst, int code) {
    PERF_AUTO(PerfUtil::global(), "cvt_color")
    CvtColor::cvt_color(src, dst, code);
}

void normalize(const VTensor& src, VTensor& dst,
			   const VTensor& mean, const VTensor& stddev) {
    PERF_AUTO(PerfUtil::global(), "normalize")
    Normalize::normalize(src, dst, mean, stddev);
}

void warp_affine(const VTensor& src, VTensor& dst,
				 const VTensor& M, VSize dsize, int flags,
				 int borderMode, const VScalar& borderValue) {
    PERF_AUTO(PerfUtil::global(), "warp_affine")
    WarpAffine::warp_affine(src, dst, M, dsize, flags, borderMode, borderValue);
}

void warp_affine(const VTensor& src, VTensor& dst,
				 float scale, float rot, VSize dsize,
				 const VScalar& aux_param, int flags,
				 int borderMode, const VScalar& borderValue) {
    PERF_AUTO(PerfUtil::global(), "warp_affine")
    WarpAffine::warp_affine(src, dst, scale, rot, dsize, aux_param, flags, borderMode, borderValue);
}

void resize_normalize(const vision::VTensor& src, vision::VTensor& dst,
					  VSize dsize, double fx, double fy,
					  int interpolation,
					  const vision::VTensor& mean,
					  const vision::VTensor& stddev) {
    PERF_AUTO(PerfUtil::global(), "resize_normalize")
    ResizeNormalize::resize_normalize(src, dst, dsize, fx, fy, interpolation, mean, stddev);
}

void resize_normalize(const cv::Mat& src, cv::Mat& dst,
                      VSize dsize, double fx, double fy,
                      int interpolation,
                      const vision::VTensor& mean,
                      const vision::VTensor& stddev) {
    PERF_AUTO(PerfUtil::global(), "resize_normalize")
    ResizeNormalize::resize_normalize(src, dst, dsize, fx, fy, interpolation, mean, stddev);
}

void warp_affine_normalize(const vision::VTensor& src, vision::VTensor& dst,
						   const vision::VTensor& M, VSize dsize,
						   int flags,
						   int borderMode,
						   const VScalar& borderValue,
						   const vision::VTensor& mean,
						   const vision::VTensor& stddev) {
    PERF_AUTO(PerfUtil::global(), "warp_affine_normalize")
    WarpAffineNormalize::warp_affine_normalize(src, dst, M, dsize, flags, borderMode, borderValue, mean, stddev);
}

void warp_affine_normalize(const vision::VTensor& src, vision::VTensor& dst,
						   float scale, float rot, VSize dsize,
						   const VScalar& aux_param,
						   int flags, int borderMode,
						   const VScalar& borderValue,
						   const vision::VTensor& mean,
						   const vision::VTensor& stddev) {
    PERF_AUTO(PerfUtil::global(), "warp_affine_normalize")
    WarpAffineNormalize::warp_affine_normalize(src, dst, scale, rot, dsize, aux_param,
            flags, borderMode, borderValue, mean, stddev);
}

void crop(const vision::VTensor& src, vision::VTensor& dst, const vision::VRect& rect) {
    PERF_AUTO(PerfUtil::global(), "crop")
    Crop::crop(src, dst, rect);
}

void crop(const vision::VTensor& src, int srcFormat, vision::VTensor &dst, const vision::VRect &rect) {
    PERF_AUTO(PerfUtil::global(), "crop2")
    Crop::crop(src, dst, rect);
}

void match_template(const vision::VTensor& src, const vision::VTensor& target,
					vision::VTensor& result, int method) {
    PERF_AUTO(PerfUtil::global(), "match_template")
    MatchTemplate::match_template(src, target, result, method);
}

void minMaxIdx(const vision::VTensor& src, double* minVal, double* maxVal,
			   int* minIdx, int* maxIdx, const vision::VTensor& mask) {
    PERF_AUTO(PerfUtil::global(), "minMaxIdx")
    MatchTemplate::minMaxIdx(src, minVal, maxVal, minIdx, maxIdx, mask);
}

void imencode(const vision::VTensor& src, std::vector<unsigned char>& buf, const char* format) {
    PERF_AUTO(PerfUtil::global(), "imencode")
    ImEncode::imencode(src, buf, format);
}

bool variance(const vision::VTensor &src, float &variance) {
    PERF_AUTO(PerfUtil::global(), "variance")
#ifdef USE_OPENCV
    if (src.c != 3 || src.dType != FP32) {
        return false;
    }

    cv::Mat rbgData;
    rbgData = TensorConverter::convert_to<cv::Mat>(src, true);

    std::vector<cv::Mat_<float>> channels;
    cv::split(rbgData, channels);
    cv::Mat_<float> &r = channels[0];
    cv::Mat_<float> &g = channels[1];
    cv::Mat_<float> &b = channels[2];

    cv::Scalar r_stdDev;        // 均值
    cv::Scalar r_meanStdDev;    // 标准差
    cv::meanStdDev(r - g,  r_stdDev, r_meanStdDev);

    cv::Scalar g_stdDev;
    cv::Scalar g_meanStdDev;
    cv::meanStdDev(g - b,  g_stdDev, g_meanStdDev);

    cv::Scalar b_stdDev;
    cv::Scalar b_meanStdDev;
    cv::meanStdDev(b - r,  b_stdDev, b_meanStdDev);

    variance = (r_meanStdDev[0] * r_meanStdDev[0] +
                      g_meanStdDev[0] * g_meanStdDev[0] +
                      b_meanStdDev[0] * b_meanStdDev[0]) / 3.0;
    return true;
#else
    // 此函数的作用是计算NFT4浮点的范围的近似均值和方差，沿维度描述元素的描述符被视为随机vars。
//    fcvDescriptorSampledMeanAndVar36f32();

    //此函数的作用在灰度图像中计算矩形的强度均值和方差。
//    fcvImageIntensityStats();
    return false;
#endif
}

} // namespace aura::va_cv
