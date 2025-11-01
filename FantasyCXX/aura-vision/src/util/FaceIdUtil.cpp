#include "vision/util/FaceIdUtil.h"

#include <cmath>
#include <vector>

#include "config/static_config/ref_landmark/ref_face_crop.h.in"
#include "vacv/cv.h"
#include "vision/core/common/VStructs.h"
#include "vision/core/common/VTensor.h"
#include "vision/util/ImageUtil.h"
#ifdef BUILD_3D_LANDMARK
#include "face_3d_points.h"
#endif

namespace aura::vision {

float FaceIdUtil::compare_face_features(const float* first, const float* second) {
    float result = 0.0f;
    if (first == nullptr || second == nullptr) {
        return result;
    }

    float norm1 = 0.000001f;
    float norm2 = 0.000001f;
    float dot = 0.0f;
    for (int i = 0; i < FEATURE_COUNT; i++) {
        dot += first[i] * second[i];
        norm1 += first[i] * first[i];
        norm2 += second[i] * second[i];
    }
    result = dot / sqrt(norm1 * norm2);
    return result;
}

static constexpr float _k_face_feature_beta = 0.067;

bool FaceIdUtil::update_face_feature(const float *storage_data, const float *cur_data, float *output_data) {
    if (storage_data == nullptr || cur_data == nullptr || output_data == nullptr) {
        return false;
    }

    float contrary_beta = 1.0f - _k_face_feature_beta;
    for (int i = 0; i < FEATURE_COUNT; ++i) {
        output_data[i] = _k_face_feature_beta * cur_data[i] + contrary_beta * storage_data[i];
    }
    return true;
}

int FaceIdUtil::crop_face(short cameraLightType,int frameConvertBgrFormat,const float* landmarks,
                          unsigned char *frame, int w, int h, int resized_w, int resized_h,
                          unsigned char *resized_frame) {
    VTensor image;
    // 图像格式转换RGB和IR的转化方法不同
    if (cameraLightType == CAMERA_LIGHT_TYPE_RGB) {
		VTensor input(w, h * 3 / 2, frame, INT8);
        va_cv::cvt_color(input, image, frameConvertBgrFormat);
    } else {
		VTensor input(w, h, frame, INT8);
        va_cv::cvt_color(input, image,COLOR_GRAY2BGR);
    }

    auto* input_lmk = new VPoint[LM_2D_106_COUNT];
    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        input_lmk[i].x = landmarks[2 * i];
        input_lmk[i].y = landmarks[2 * i + 1];
    }

    float scale = 1.f;
    float rot = 0.f;
    va_cv::VScalar aux_params;
	VTensor cropped;
    ImageUtil::get_warp_params(input_lmk, get_face_crop_ref_lmk(), scale, rot, aux_params);
    va_cv::warp_affine(image, cropped, scale, rot, va_cv::VSize(resized_w, resized_h), aux_params);

    std::vector<unsigned char> encoded;
    va_cv::imencode(cropped, encoded, ".jpg");
    auto frame_len = encoded.size();
    memcpy(resized_frame, encoded.data(), frame_len);

    delete[] input_lmk;
    return frame_len;

//    VPoint *pts_landmarks = new VPoint[LM_2D_106_COUNT];
//    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
//        pts_landmarks[i].x = vision::aig_landmark[2 * i];
//        pts_landmarks[i].y = vision::aig_landmark[2 * i + 1];
//    }
//    cv::Mat image_ir;
//    cv::Mat image_light;
//    // 可见光和ir需要分别转换
//    if (config::sCameraLightType == CAMERA_LIGHT_TYPE_RGB) {
//        cv::Mat image_rgb_temp = cv::Mat(h * 1.5, w, CV_8UC1, _frame);
//        cv::cvtColor(image_rgb_temp, image_light, config::_s_frame_convert_format);
//    } else {
//        cv::Mat image_ir_temp = cv::Mat(h, w, CV_8UC1, _frame);
//        cv::cvtColor(image_ir_temp, image_ir, CV_GRAY2BGRA);
//    }

//    float lefteyex = 0;
//    float lefteyey = 0;
//    float righteyex = 0;
//    float righteyey = 0;
//    for (int i = FLM_61_R_EYE_LEFT_CORNER * 2; i < FLM_71_NOSE_BRIDGE1 * 2; i++) {
//        if (i % 2 == 0) {
//            lefteyex += landmarks[i];
//        } else {
//            lefteyey += landmarks[i];
//        }
//    }
//    for (int i = FLM_51_L_EYE_LEFT_CORNER * 2; i < FLM_61_R_EYE_LEFT_CORNER * 2; i++) {
//        if (i % 2 == 0) {
//            righteyex += landmarks[i];
//        } else {
//            righteyey += landmarks[i];
//        }
//    }
//    lefteyex /= 10;
//    lefteyey /= 10;
//    righteyex /= 10;
//    righteyey /= 10;
//    float dx = righteyex - lefteyex;
//    float dy = righteyey - lefteyey;
//    // 得到一个目标弧度值
//    float srcDegrees = static_cast<float>((std::atan2(dy, dx) * ANGEL_180 / M_PI) - ANGEL_180);
//    /**
//    *  计算 Base 数据的眼部中心点
//    */
//    float base_left_eye_x = 0;
//    float base_left_eye_y = 0;
//    float base_right_eye_x = 0;
//    float base_right_eye_y = 0;
//    for (int i = FLM_61_R_EYE_LEFT_CORNER; i < FLM_71_NOSE_BRIDGE1; i++) {
//
//        base_left_eye_x += pts_landmarks[i].x;
//        base_left_eye_y += pts_landmarks[i].y;
//
//    }
//    for (int i = FLM_51_L_EYE_LEFT_CORNER; i < FLM_61_R_EYE_LEFT_CORNER; i++) {
//        base_right_eye_x += pts_landmarks[i].x;
//        base_right_eye_y += pts_landmarks[i].y;
//    }
//    base_left_eye_x /= 10;
//    base_left_eye_y /= 10;
//    base_right_eye_x /= 10;
//    base_right_eye_y /= 10;
//    float base_dx = base_right_eye_x - base_left_eye_x;
//    float base_dy = base_right_eye_y - base_left_eye_y;
//    // 根据弧度值计算角度
//    int baseDegrees = int(std::atan2(base_dy, base_dx) * ANGEL_180 / M_PI) - ANGEL_180;
//    /**
//     *   计算 Base 的 x,y 均值
//     */
//    float base_mean_x = 0.0f;
//    float base_mean_y = 0.0f;
//    for (int j = 0; j < LM_2D_106_COUNT; ++j) {
//        base_mean_x += pts_landmarks[j].x;
//        base_mean_y += pts_landmarks[j].y;
//    }
//    base_mean_x = base_mean_x / LM_2D_106_COUNT;
//    base_mean_y = base_mean_y / LM_2D_106_COUNT;
//
//    /**
//     *  计算输入 landmark 的 x 均值
//     */
//    float input_mean_x = 0.0f;
//    float input_mean_y = 0.0f;
//    for (int k = 0; k < 212; ++k) {
//        if (k % 2 == 0) {
//            input_mean_x += landmarks[k];
//        } else {
//            input_mean_y += landmarks[k];
//        }
//    }
//    input_mean_x = input_mean_x / LM_2D_106_COUNT;
//    input_mean_y = input_mean_y / LM_2D_106_COUNT;
//    /**
//     * 计算 base 根号平方和
//     */
//    float sum_base_x = 0.0f;
//    float sum_base_y = 0.0f;
//    for (int l = 0; l < LM_2D_106_COUNT; ++l) {
//        sum_base_x += std::pow(pts_landmarks[l].x - base_mean_x, 2);
//        sum_base_y += std::pow(pts_landmarks[l].y - base_mean_y, 2);
//    }
//    float base_sign = std::sqrt(sum_base_x + sum_base_y);
//    /**
//     * 计算输入的根号平方和
//     */
//    float sum_input_x = 0.0f;
//    float sum_input_y = 0.0f;
//    for (int l = 0; l < 212; ++l) {
//        if (l % 2 == 0) {
//            sum_input_x += std::pow(landmarks[l] - input_mean_x, 2);
//        } else {
//            sum_input_y += std::pow(landmarks[l] - input_mean_y, 2);
//        }
//    }
//    float input_sign = std::sqrt(sum_input_x + sum_input_y);
//    float scale =
//            (base_sign / LM_2D_106_COUNT) / (input_sign / LM_2D_106_COUNT);
//    cv::Mat mat = cv::getRotationMatrix2D(cv::Point(0, 0), srcDegrees, scale);
//    mat.at<double>(0, 2) = base_mean_x - mat.at<double>(0, 0) * input_mean_x - mat.at<double>(0, 1) * input_mean_y;
//    mat.at<double>(1, 2) = base_mean_y - mat.at<double>(1, 0) * input_mean_x - mat.at<double>(1, 1) * input_mean_y;
//    cv::Mat output;
//    if (config::sCameraLightType == CAMERA_LIGHT_TYPE_RGB) {
//        cv::warpAffine(image_light, output, mat, cv::Size(desirew, desireh));
//    } else {
//        cv::warpAffine(image_ir, output, mat, cv::Size(desirew, desireh));
//    }
////        memcpy(frameResize, output.data, desireh * desirew * 4);
//    std::vector<unsigned char> out_image;
//    cv::imencode(".jpg", output, out_image);
//    int datalen = out_image.size();
//    for (int i = 0; i < datalen; i++) {
//        resized_frame[i] = out_image[i];
//    }
//
//    delete[] pts_landmarks;
//    return datalen;
}
bool FaceIdUtil::convertPoints3Dto2D(std::vector<float> &vec3dPoints, std::vector<float> &vec2dPoints, float &focalX,
                                     float &focalY, float &opticalCenterX, float &opticalCenterY) {
#ifdef BUILD_3D_LANDMARK
    return cockpitcv::convert3dpointTo2dpoint(vec3dPoints, vec2dPoints, focalX, focalY, opticalCenterX, opticalCenterY);
#endif
    VLOGW("TAG", "Unsupported Operation: convert points from 3D to 2D");
    return false;
}

} // namespace aura::vision