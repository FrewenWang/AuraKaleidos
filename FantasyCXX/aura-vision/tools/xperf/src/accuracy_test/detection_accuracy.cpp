#include "detection_accuracy.h"

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unordered_map>

#include "json.h"
#include "opencv2/opencv.hpp"

#include "vision/VisionAbility.h"
#include "util/image_process_util.h"


namespace xperf{

using namespace vision;

static VisionInitializer *initializer = nullptr;
static VisionService* g_service = nullptr;
static VisionRequest* g_request = nullptr;
static VisionResult* g_result = nullptr;
static cv::Mat g_yuv_image;
static int input_width;
static int input_height;


void DetectionAccuracy::test_setup() {
    input_width = 1280;
    input_height = 720;

    initializer = new VisionInitializer;
    initializer->init();

    g_service = new VisionService;
    g_service->init();
    g_request = g_service->make_request();
    g_result = g_service->make_result();

    g_service->set_config(ParamKey::USE_INTERNAL_MEM, true);
    g_service->set_config(ParamKey::FRAME_CONVERT_FORMAT, COLOR_YUV2BGR_NV21);
    g_service->set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_IR);
    // 单帧检测设置RELEASE_MODE为BENCHMARK_TEST
    g_service->set_config(ParamKey::RELEASE_MODE, BENCHMARK_TEST);
    // 测试集单帧图片只接受一张人脸输出
    g_service->set_config(ParamKey::FACE_MAX_COUNT, 1);
    g_service->set_config(ParamKey::FACE_NEED_CHECK_COUNT, 1);
    g_service->set_switches(
            std::unordered_map<short, bool> {
                    {ABILITY_FACE_RECT,                     true},
                    {ABILITY_FACE_LANDMARK,                 true},
            });
}

void format_json_result(VisionResult* result, Json::Value &detect_result_json, std::string eval_feature) {
    FaceInfo *face = result->get_face_result()->_face_infos[0];
    bool has_face = face->hasFace();
    detect_result_json["has_face"] = Json::Value(has_face);
    int face_id = face->_id;
    detect_result_json["face_id"] = Json::Value(face_id);
    // 人脸框坐标
    Json::Value face_rect_json;
    face_rect_json[0] = face->_rect_lt.x;
    face_rect_json[1] = face->_rect_lt.y;
    face_rect_json[2] = face->_rect_rb.x;
    face_rect_json[3] = face->_rect_rb.y;
    detect_result_json["face_rect"] = face_rect_json;

    GestureInfo *gesture = result->get_gesture_result()->_gesture_infos[0];

    if (eval_feature == "dms") {
        // dms五分类
        float state_dangerous = face->_danger_drive_state;
        detect_result_json["dangerous_state"] = Json::Value(state_dangerous);

    } else if (eval_feature == "face_recognize") {
        // 人脸特征提取
        Json::Value face_features_json;
        float* face_features = face->_feature;
        for (int i = 0; i < FEATURE_COUNT; i++) {
            face_features_json[i] = face_features[i];
        };
        detect_result_json["face_features"] = face_features_json;
    } else if (eval_feature == "gesture_type") {
        // 手势框类型
        float gesture_type = gesture->_type;
        detect_result_json["gesture_type"] = Json::Value(gesture_type);

        // 手势框坐标
        Json::Value gesture_rect_json;
        gesture_rect_json[0] = gesture->_rect_lt.x;
        gesture_rect_json[1] = gesture->_rect_lt.y;
        gesture_rect_json[2] = gesture->_rect_rb.x;
        gesture_rect_json[3] = gesture->_rect_rb.y;
        detect_result_json["gesture_rect"] = gesture_rect_json;
    } else if (eval_feature == "face_liveness_ir" || eval_feature == "face_liveness_rgb") {
        // 无感活检
        float state_no_interact_living = face->stateNoInteractLivingSingle;
        detect_result_json["no_interact_living_state"] = Json::Value(state_no_interact_living);
    } else if (eval_feature == "face_call") {
        // 打电话
        int left_call_state = face->_left_ear_call;
        int right_call_state = face->_right_ear_call;
        bool call_state = (left_call_state == 1) || (right_call_state == 1);
        detect_result_json["call_state"] = call_state;
    } else if (eval_feature == "face_attribute_rgb" || eval_feature == "face_attribute_ir") {
        // 人脸属性
        detect_result_json["glass_state"] = face->_state_glass_single;
        detect_result_json["age_state"] = face->_state_age_single;
        detect_result_json["gender_state"] = face->_state_gender_single;
        detect_result_json["race_state"] = face->_state_race_single;
    }  else if (eval_feature == "face_eye_center") {
        // 眼球模型，指标测试输入是眼睛图片，只取left即可
        detect_result_json["eye_centroid_x"] = face->_eye_centroid_left.x;
        detect_result_json["eye_centroid_y"] = face->_eye_centroid_left.y;

        Json::Value eyelid_landmark_json;
        VPoint* eye_lmks_left = face->_eye_lmk_8_left;
        for (int i = 0; i < 8; i++) {
            eyelid_landmark_json[i * 2] = eye_lmks_left[i].x;
            eyelid_landmark_json[i * 2 + 1] = eye_lmks_left[i].y;
        };
        detect_result_json["eye_lmks"] = eyelid_landmark_json;
    } else if (eval_feature == "face_cover") {
        // 遮挡模型
        float state_cover = face->_state_cover;
        detect_result_json["face_cover_state"] = Json::Value(state_cover);
    } else if (eval_feature == "face_emotion") {
        // 遮挡模型
        float state_emotion = face->_state_emotion_single;
        detect_result_json["face_emotion_state"] = Json::Value(state_emotion);
    }
}

void test_face_landmark_eye_state(std::string eval_img_path, std::string eval_result_path) {
    std::cout << "DetectionAccuracy" << " detect folder:" << eval_img_path.c_str() << std::endl;
    struct dirent *dirp;
    DIR* dir = opendir(eval_img_path.c_str());
    FaceInfo *face;
    while ((dirp = readdir(dir)) != nullptr) {
        if (dirp->d_type != DT_REG) {
            continue;
        }
        std::string file_name(dirp->d_name);
        std::string img_path(eval_img_path);
        img_path.append("/").append(file_name);
        cv::Mat image = cv::imread(img_path);
        cv::Mat fixed_image;
        fixed_image = xperf::ImageProcessUtil::fix_image_size(image, input_width, input_height);
        g_yuv_image = xperf::ImageProcessUtil::bgr_to_yuv(fixed_image);
        g_request->_width = input_width;
        g_request->_height = input_height;
        g_request->_frame = g_yuv_image.data;

        g_service->detect(g_request, g_result);
        face = g_result->get_face_result()->_face_infos[0];
        float eye_close_confidence = face->_eye_close_confidence;
        std::string write_file_name = std::to_string(eye_close_confidence) + "_" + file_name;
#ifdef WITH_OCV_HIGHGUI
        cv::imwrite(eval_result_path + "/" + write_file_name, image);
#endif
    }

    closedir(dir);
}

void DetectionAccuracy::test_execute(std::string eval_feature, std::string eval_img_path, std::string eval_result_path) {
    if (eval_feature == "face_landmark_eye_close") {
        // face landmark的睁闭眼置信度检测，输出阈值即可，不用保存为json结果文件
        test_face_landmark_eye_state(eval_img_path, eval_result_path);
        return;
    }

    if (eval_feature == "dms") {
        // dms五分类
        g_service->set_switch(ABILITY_FACE_DANGEROUS_DRIVING, true);
    } else if (eval_feature == "face_recognize") {
        // 人脸特征提取
        g_service->set_switch(ABILITY_FACE_FEATURE, true);
    } else if (eval_feature == "gesture_type") {
        g_service->set_switch(ABILITY_GESTURE, true);
        g_service->set_switch(ABILITY_GESTURE_RECT, true);
        g_service->set_switch(ABILITY_GESTURE_LANDMARK, true);
        g_service->set_switch(ABILITY_GESTURE_TYPE, true);
    } else if (eval_feature == "face_liveness_ir" || eval_feature == "face_liveness_rgb") {
        // 无感活检
        g_service->set_switch(ABILITY_FACE_NO_INTERACTIVE_LIVING, true);
        if (eval_feature == "face_liveness_rgb") {
            g_service->set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_RGB);
        } else {
            g_service->set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_IR);
        }
    } else if (eval_feature == "face_call") {
        // 打电话
        g_service->set_switch(ABILITY_FACE_CALL, true);
    } else if (eval_feature == "face_attribute_rgb" || eval_feature == "face_attribute_ir") {
        // 人脸属性
        g_service->set_switch(ABILITY_FACE_ATTRIBUTE, true);
        if (eval_feature == "face_attribute_rgb") {
            g_service->set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_RGB);
        } else {
            g_service->set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_IR);
        }
    }  else if (eval_feature == "face_eye_center") {
        g_service->set_switch(ABILITY_FACE_RECT, false);
        g_service->set_switch(ABILITY_FACE_LANDMARK, false);

        // 眼球模型
        g_service->set_switch(ABILITY_FACE_EYE_CENTER, true);
        // 眼球模型测试集输入为60x60单个人眼图片
        input_width = 60;
        input_height = 60;
    } else if (eval_feature == "face_cover") {
        // 遮挡模型
        g_service->set_switch(ABILITY_FACE_QUALITY, true);
    } else if (eval_feature == "face_emotion") {
        // 表情检测模型
        g_service->set_switch(ABILITY_FACE_EMOTION, true);
    }

    Json::Reader reader;
    // Json::StyledWriter sw;
    Json::FastWriter fw;
    Json::Value batch_result_json;

    // make sure the json file(eval_result_path) exists
    std::ifstream in(eval_result_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "DetectionAccuracy:eval_result_path doesn't exists!" << std::endl;
        return;
    }
    if (!reader.parse(in, batch_result_json)) {
        std::cout << "DetectionAccuracy:eval_result_path parse json failed!" << std::endl;
        return;
    }
    in.close();

    std::cout << "DetectionAccuracy" << " detect folder:" << eval_img_path.c_str() << std::endl;
    struct dirent *dirp;
    DIR* dir = opendir(eval_img_path.c_str());
    while ((dirp = readdir(dir)) != nullptr) {
        if (dirp->d_type != DT_REG) {
            continue;
        }
        std::string file_name(dirp->d_name);
        std::string img_path(eval_img_path);
        img_path.append("/").append(file_name);
        cv::Mat image = cv::imread(img_path);
        cv::Mat fixed_image;

        fixed_image = xperf::ImageProcessUtil::fix_image_size(image, input_width, input_height);
        g_yuv_image = xperf::ImageProcessUtil::bgr_to_yuv(fixed_image);

        g_request->clear_all();
        g_result->clear_all();
        g_request->_width = input_width;
        g_request->_height = input_height;
        g_request->_frame = g_yuv_image.data;

        g_service->detect(g_request, g_result);
        Json::Value detect_result_json;
        format_json_result(g_result, detect_result_json, eval_feature);
        batch_result_json[file_name] = Json::Value(detect_result_json);
    }

    //缩进输出
    std::ofstream os;
    os.open(eval_result_path, std::ios::out);
    os << fw.write(batch_result_json);
    os.close();

    closedir(dir);
}

void DetectionAccuracy::test_cleanup() {
    g_service->recycle_request(g_request);
    g_service->recycle_result(g_result);

    delete initializer;
    initializer = nullptr;

    delete g_service;
    g_service = nullptr;
}

}


