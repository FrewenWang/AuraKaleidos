#include <chrono>
#include <iostream>
#include <thread>

#include "opencv2/opencv.hpp"
#include "vision/VisionAbility.h"

#include "util/image_util.h"
#include "util/draw_util.h"

using namespace vision;

void init_service1(VisionService* service) {
    service->init();
    service->set_config(ParamKey::USE_INTERNAL_MEM, true);
    service->set_config(ParamKey::FRAME_CONVERT_FORMAT, COLOR_YUV2BGR_NV21);
    service->set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_IR);
    service->set_switches(
            std::unordered_map<short, bool> {
                    {ABILITY_ALL,                           true},
                    {ABILITY_FACE,                          true},
                    {ABILITY_FACE_RECT,                     true},
                    {ABILITY_FACE_LANDMARK,                 true},
                    {ABILITY_FACE_2DTO3D,                   true},
                    {ABILITY_FACE_ATTRIBUTE,                true},
                    {ABILITY_FACE_EMOTION,                  true},
                    {ABILITY_FACE_FEATURE,                  true},
                    {ABILITY_FACE_QUALITY,                  true},
                    {ABILITY_FACE_DANGEROUS_DRIVING,        true},
                    {ABILITY_FACE_FATIGUE,                  true},
                    {ABILITY_FACE_ATTENTION,                true},
                    {ABILITY_FACE_HEAD_BEHAVIOR,            true},
                    {ABILITY_FACE_INTERACTIVE_LIVING,       true},
                    {ABILITY_FACE_CALL,                     true}
    });
}

void init_service2(VisionService* service) {
    service->init();
    service->set_config(ParamKey::USE_INTERNAL_MEM, true);
    service->set_config(ParamKey::FRAME_CONVERT_FORMAT, COLOR_YUV2BGR_NV21);
    service->set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_IR);
    service->set_switches(
            std::unordered_map<short, bool> {
                    {ABILITY_GESTURE,                       true},
                    {ABILITY_GESTURE_RECT,                  true},
                    {ABILITY_GESTURE_LANDMARK,              true},
                    {ABILITY_GESTURE_TYPE,                  true},
                    {ABILITY_GESTURE_DYNAMIC,               true}
    });
}

void detect(VisionService* service, cv::Mat frame, int id) {
    auto yuv_image = ImageUtil::bgr_to_yuv(frame);
    auto* req = service->make_request();
    auto* res = service->make_result();

    req->clear_all();
    res->clear_all();

    req->_width = 1280;
    req->_height = 720;
    req->_frame = yuv_image.data;

    service->detect(req, res);

    DrawUtil::draw(frame, res, id);

    service->recycle_request(req);
    service->recycle_result(res);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << std::endl << "dual_cam [cam_id] [cam_id]" << std::endl;
        return -1;
    }

    int cam1_id = atoi(argv[1]);
    int cam2_id = atoi(argv[2]);
    std::cout << "open camera id: " << cam1_id << " " << cam2_id << std::endl;

    cv::VideoCapture cap1(cam1_id);
    cap1.set(CV_CAP_PROP_MODE, 0);
    cap1.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

    cv::VideoCapture cap2(cam2_id);
    cap2.set(CV_CAP_PROP_MODE, 0);
    cap2.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap2.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!cap1.isOpened()) {
        std::cerr << "Fail to open camera " << cam1_id << std::endl;
        return -1;
    }
    if (!cap2.isOpened()) {
        std::cerr << "Fail to open camera " << cam2_id << std::endl;
        return -1;
    }

    // 多实例仅需要一次初始化，完成模型加载
    VisionInitializer initializer;
    initializer.init();

    // 定义VisionService多实例
    VisionService service1;
    VisionService service2;
    // 每个实例单独初始化，包括参数配置和开关设置
    init_service1(&service1);
    init_service2(&service2);

    cv::Mat frame1;
    cv::Mat frame2;    
    while(true) {
        cap1 >> frame1;
        if (frame1.empty()) {
            std::cerr << "frame1 is empty" << std::endl;
            break;
        }

        cap2 >> frame2;
        if (frame2.empty()) {
            std::cerr << "frame2 is empty" << std::endl;
            break;
        }

        // 分别调用detect接口进行检测
        detect(&service1, frame1, 1);
        detect(&service2, frame2, 2);

        cv::waitKey(1);
    }
}