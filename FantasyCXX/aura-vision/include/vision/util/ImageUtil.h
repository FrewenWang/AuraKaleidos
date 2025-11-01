//
// Created by Li,Wendong on 2019-05-14.
//

#ifndef VISION_NATIVE_IMAGE_UTIL_H
#define VISION_NATIVE_IMAGE_UTIL_H
#include "vision/core/common/VTensor.h"
#include "vacv/cv.h"
#ifdef BUILD_NCNN
#include "mat.h"
#endif

namespace cv {
    class Mat;
}

namespace aura::vision {
    class VPoint;

    const unsigned int R2YI = 4899;
    const unsigned int G2YI = 9617;
    const unsigned int B2YI = 1868;
    const unsigned int B2UI = 9241;
    const unsigned int R2VI = 11682;

    const unsigned char ITUR_BT_602_CY = 37;
    const unsigned char ITUR_BT_602_CUB = 65;
    const unsigned char ITUR_BT_602_CUG = 13;
    const unsigned char ITUR_BT_602_CVG = 26;
    const unsigned char ITUR_BT_602_CVR = 51;
    const unsigned char ITUR_BT_602_SHIFT = 5;

    class ImageUtil {
    public:

        static int get_landmarks_x_min(VPoint* landmarks, std::vector<int> list);

        static int get_landmarks_x_max(VPoint* landmarks, std::vector<int> list);

        static int get_landmarks_y_min(VPoint* landmarks, std::vector<int> list);

        static int get_landmarks_y_max(VPoint* landmarks, std::vector<int> list);

        static void save_img(cv::Mat &src, char* feature_name);

        static float calc_image_brightness(short width, short height, unsigned char *frame);

#ifdef BUILD_NCNN
        static void resize_and_normalize(int src_w, int src_h, unsigned char *src_data, int pixel_type,
                                         int dst_w, int dst_h, ncnn::Mat &dst_data, float *means = NULL,
                                         float *stds = NULL);
#endif

        static cv::Mat normalize(cv::Mat& in);

        static cv::Mat normalize_with_mean_stddev(cv::Mat& in, const std::vector<double>& mean,
                                                  const std::vector<double>& stddev);

        static void mean_stddev(cv::Mat& in, std::vector<double>& mean, std::vector<double>& stddev);

        static cv::Mat fix_image_size(cv::Mat& in, int w, int h);

        template<class T>
        static void image_hwc_to_chw(T* in, T* out, int w, int h, int c) {
            int count = 0;
            int step = h * w;
            for (int i = 0; i < c; ++i) {
                for (int j = 0; j < step; ++j) {
                    out[count] = in[j * c + i];
                    count += 1;
                }
            }
        }

        template<class T>
        static void image_chw_to_hwc(T* in, T* out, int w, int h, int c) {
            int count = 0;
            int step = h * w;
            for (int i = 0; i < step; ++i) {
                for (int j = 0; j < c; ++j) {
                    out[count] = in[j * step + i];
                    count += 1;
                }
            }
        }

        /**
         * BGR转换为NV21格式
         * @param src 输入BGR图像数据
         * @param dst 输出NV21图像数据
         * @param width 图像宽度
         * @param height 图像高度
         */
        static void bgr2nv21(unsigned char *src, unsigned char *dst, int width, int height);

        /**
         * @brief BGR转换为NV21格式
         * @param frame 输入BGR图像数据
         * @return 转化而成的nv21数据
         */
        static cv::Mat bgr2nv21(const cv::Mat &frame);

        /**
         * @brief BGR转换为UYVY格式
         * @param frame 输入BGR图像数据
         * @return 转化而成的UYVY格式
         */
        static void bgr2uyvy(const cv::Mat &rgb, cv::Mat &uyvy);

        /**
         * 颜色转换
         * @param src
         * @param dst
         * @param width
         * @param height
         */
        static void cvt_color(cv::Mat& src, cv::Mat& dst, int code);

        /**
         * YUV 转 rgb
         * @param src
         * @param dst
         * @param width
         * @param height
         */
        static void cvt_color_yuv2rgb(unsigned char *src, int *dst, int width, int height);

#if __ARM_NEON
        /**
         * YV12转换为BGR
         * @param src
         * @param dst
         * @param w
         * @param h
         * @param bIdx
         */
        static void cvt_color_yuv2bgr_yv12(unsigned char *src, unsigned char *dst, int w, int h, int bIdx = 0);

        /**
         * NV21转换为BGR
         * @param src
         * @param dst
         * @param w
         * @param h
         * @param bIdx
         */
        static void cvt_color_yuv2bgr_nv21(unsigned char *src, unsigned char *dst, int w, int h, int bIdx = 0);
#endif // __ARM_NEON

        static void by_landmark_get_warp_affine(const VPoint* src_landmarks,
                const std::vector<VPoint>& base_landmarks, cv::Mat& rotation_matrix);

        static void get_warp_params(const VPoint* landmark, const VPoint* ref_landmark,
                float& scale, float& rot, va_cv::VScalar& aux_coeff);

        static void get_warp_params(const VPoint* landmark, const VPoint* ref_landmark, VTensor& rot_matrix);
    };

} // namespace vision
#endif //VISION_NATIVE_IMAGE_UTIL_H
