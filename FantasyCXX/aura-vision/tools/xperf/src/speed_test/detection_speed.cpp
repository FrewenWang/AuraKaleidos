#include "detection_speed.h"

#include <unordered_map>

#include "opencv2/opencv.hpp"
#include "vision/VisionAbility.h"

#include "profiler/speed_profiler.h"
#include "util/ability_id_to_str.h"
#include "util/image_process_util.h"

namespace xperf {

using namespace vision;

static VisionInitializer *initializer = nullptr;
static VisionService* g_service = nullptr;
static VisionRequest* g_request = nullptr;
static VisionResult* g_result = nullptr;
static std::string g_img_path = "../res/test_face.jpg";
static cv::Mat g_yuv_image;

void profile_setup() {
    cv::Mat image = cv::imread(g_img_path.c_str());
    g_yuv_image = ImageProcessUtil::bgr_to_yuv(image);
    g_request->_width = 1280;
    g_request->_height = 720;
    g_request->_frame = g_yuv_image.data;
}

void profile_cleanup() {
    g_request->clear_all();
    g_result->clear_all();
}

void DetectionSpeed::test_setup(std::string img_path) {
    g_img_path = std::move(img_path);
    initializer = new VisionInitializer;
    initializer->init();

    g_service = new VisionService;
    g_service->init();
    g_request = g_service->make_request();
    g_result = g_service->make_result();

    g_service->set_config(ParamKey::USE_INTERNAL_MEM, true);
    g_service->set_config(ParamKey::FRAME_CONVERT_FORMAT, COLOR_YUV2BGR_YV12);
    g_service->set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_IR);
    // 单帧检测设置RELEASE_MODE为BENCHMARK_TEST
    g_service->set_config(ParamKey::RELEASE_MODE, BENCHMARK_TEST);
    g_service->set_switches(
            std::unordered_map<short, bool> {
                    {ABILITY_ALL,                           true},
                    {ABILITY_FACE,                          true},
                    {ABILITY_FACE_DETECTION,                true},
                    {ABILITY_FACE_ATTRIBUTE,                true},
                    {ABILITY_FACE_EMOTION,                  true},
                    {ABILITY_FACE_FEATURE,                  true},
                    {ABILITY_FACE_QUALITY,                  true},
                    {ABILITY_FACE_DANGEROUS_DRIVING,        true},
                    {ABILITY_FACE_FATIGUE,                  true},
                    {ABILITY_FACE_ATTENTION,                true},
                    {ABILITY_FACE_HEAD_BEHAVIOR,            true},
                    {ABILITY_FACE_INTERACTIVE_LIVING,       true},
                    {ABILITY_FACE_CALL,                     true},
                    {ABILITY_FACE_NO_INTERACTIVE_LIVING,    true},
                    {ABILITY_FACE_EYE_GAZE,                 true},
                    {ABILITY_FACE_EYE_TRACKING,             true},
                    {ABILITY_GESTURE,                       true},
                    {ABILITY_GESTURE_RECT,                  true},
                    {ABILITY_GESTURE_LANDMARK,              true},
                    {ABILITY_GESTURE_TYPE,                  true},
                    {ABILITY_FACE_EYE_CENTER,               true}
            });

    profile_setup();
}

void DetectionSpeed::test_cleanup() {
    g_service->recycle_request(g_request);
    g_service->recycle_result(g_result);

    delete initializer;
    initializer = nullptr;

    delete g_service;
    g_service = nullptr;
}

template <AbilityId id>
struct DetectFunctor {
    void operator() () const {
        g_request->set_specific_ability(id);
        g_service->detect(g_request, g_result);
    }
};

void DetectionSpeed::test_execute() {
    std::vector<SpeedProfiler::SpeedResult> test_result;
    SpeedProfiler::TestFuncList func_list {
            DETECT_FUNC(ABILITY_FACE_RECT),
            DETECT_FUNC(ABILITY_FACE_LANDMARK),
            DETECT_FUNC(ABILITY_FACE_NO_INTERACTIVE_LIVING),
            DETECT_FUNC(ABILITY_FACE_DANGEROUS_DRIVING),
            DETECT_FUNC(ABILITY_FACE_CALL),
            DETECT_FUNC(ABILITY_FACE_QUALITY),
            DETECT_FUNC(ABILITY_FACE_ATTRIBUTE),
            DETECT_FUNC(ABILITY_FACE_EMOTION),
            DETECT_FUNC(ABILITY_FACE_EYE_GAZE),
            DETECT_FUNC(ABILITY_FACE_EYE_CENTER),
            DETECT_FUNC(ABILITY_FACE_EYE_TRACKING),
            DETECT_FUNC(ABILITY_FACE_ATTENTION),
            DETECT_FUNC(ABILITY_FACE_HEAD_BEHAVIOR),
            DETECT_FUNC(ABILITY_FACE_FATIGUE),
            DETECT_FUNC(ABILITY_FACE_FEATURE),
            DETECT_FUNC(ABILITY_GESTURE_RECT),
            DETECT_FUNC(ABILITY_GESTURE_LANDMARK)
    };
    SpeedProfiler::profile(func_list, &profile_setup, &profile_cleanup, test_result);
}

} // namespace xperf