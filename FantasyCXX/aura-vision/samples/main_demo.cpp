#include "../include/vision/VisionAbility.h"

#include "util/DebugUtil.h"
#include "vision/util/ImageUtil.h"
#include "vision/util/log.h"
#include "vgui.h"
#include "json/json.h"
#include "opencv2/opencv.hpp"

#include "InfoPrinter.h"
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <time.h>
#include <unordered_map>

#include <sys/stat.h>

#include "fileHelper.h"

using namespace std;
using namespace aura::vision;
using namespace vision_demo;
using namespace cv;

static const char *TAG = "VisionDemo";

#ifdef SAVE_IMG
// set mp4 file format and size
static cv::VideoWriter writer;
#endif

enum DemoType {
    IMAGE = 0,
    CAMERA = 1,
    VIDEO = 2,
    IMAGE_FOLDER = 3,
    SHOW_RESOURCES = 4,
    VIDEO_FOLDER = 5,
};

enum ResType {
    UNKNOWN = 0,
    IMG_UYVY = 1,
    IMG_GRAY = 2,
    IMG_RGB = 3,
    IMG_BGR = 4,
    IMG_JPG = 5,
    IMG_PNG = 6,
    IMGS_UYVY = 7,
    VIDEO_UYVY = 8,
    VIDEO_3GP = 9,
    JPG2UYVY = 10
};

DemoType sDemoType = IMAGE;
const char *kDemoType = "-dt";

string sTestImg = "";
const char *kTestImg = "-im";

string sModelPath = "";
const char *kModelPath = "-mp";

int sShowImg = 1;
const char *kShowImg = "-sh";

int sLoopCount = 1;
const char *kLoopCount = "-lc";

int sFrameDelay = 0;
const char *kFrameDelay = "-fd";

int sScheduleMethod = SchedulerMethod::NAIVE;
const char *kScheduleMethod = "-sm";

int sScheduleDagThreads = 2;
const char *kScheduleDagThreads = "-st";

int sQnnLoopModelId = 0;
const char *kQnnLoopModelId = "-ql";

int sDelayToStart = 0;
const char *kDelayToStart = "-ds";

int sSourceIndex = 1;
const char *kSourceIndex = "-source";

int sResType = ResType::UNKNOWN;
const char *kResType = "-rt";

int sImgWidth = 1600;
const char *kImgWidth = "-iw";

int sImgHeight = 1300;
const char *kImgHeight = "-ih";

// 接收外部设置读取图像的格式
FrameFormat gFrameFormat = FrameFormat::BGR;
const char *kFrameFormat = "-if";

string sSaveImgDir = "";
const char *kSaveImgDir = "-sid";

static int sSaveImageCheckResultStatus = 0;
static const char *sSaveImageCheckResult = "-sir";

std::mutex detectingMutex;
bool isDetecting = false;

#if defined(BUILD_QNX)
std::string sGenerateAbilityTableDir = "/data/ability_demo.json";
bool sGenerateAbilityTableFlag = false;
const char *pGenerateAbilityTable = "-gat";
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------------------------------------------------

//字符串分割函数
std::vector<std::string> split(std::string str, std::string pattern) {
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern;//扩展字符串以方便操作
    int size = str.size();
    for (int i = 0; i < size; i++) {
        pos = str.find(pattern, i);
        if (pos < size) {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

// from larger to smaller
bool compare_timestamp(const std::string file1, const std::string file2) {
    std::vector<std::string> f1_split = split(file1, "_");
    std::vector<std::string> f2_split = split(file2, "_");

    std::string f1_time = split(f1_split[f1_split.size() - 1], ".")[0];
    std::string f2_time = split(f2_split[f2_split.size() - 1], ".")[0];
    return f1_time < f2_time;
}

void time_to_str(std::string &timestamp_str, std::string &date_str) {
    //此处转化为东八区北京时间，如果是其它时区需要按需求修改
    int64 milli = atol(timestamp_str.c_str()) + (int64)8 * 60 * 60 * 1000;
    auto mTime = std::chrono::milliseconds(milli);
    auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(mTime);
    auto tt = std::chrono::system_clock::to_time_t(tp);
    std::tm *now = std::gmtime(&tt);

    char str[64];
    sprintf(str, "%4d-%02d-%02d %02d:%02d:%02d",
            now->tm_year + 1900, now->tm_mon + 1, now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
    date_str = str;
}

void print_result(VisionService *service, VisionResult *result) {
    // 打印耗时
#ifdef ENABLE_PERF
//    std::cout << "\nPerf information: " << std::endl;
//    std::cout << std::setw(45) << std::setfill('=') << " " << std::endl;
//    auto perf_records = result->getPerfUtil()->get_records();
//    for (auto& rec : perf_records) {
//        std::cout << std::left << std::setw(30) << std::setfill(' ') << rec.first << " : "
//                  << std::setw(4) << std::setfill(' ')<< rec.second << " ms" << std::endl;
//    }
//    std::cout << std::right << std::setw(45) << std::setfill('=') << " " << std::endl;
//    std::cout << std::endl;
#endif
    InfoPrinter::print(service, result);
}

static bool isFileExistsStat(string& name) {
    struct stat buffer = {0};
    return (stat(name.c_str(), &buffer) == 0);
}

static void drawAllResultShowAndSave(vision::VisionService *service,
                                     cv::Mat &image,
                                     vision::VisionResult *result,
                                     std::string saveImgPath) {

    if(sSaveImageCheckResultStatus == 1 || sShowImg == 1) {
        // cv::Mat im = cv::Mat(req->height, req->width, CV_8UC1, req->gray.data);
        // 需要画哪些图像就调用那个画图函数
        // 升级opencv4.6后imshow()方法存在问题，暂时关闭显示
        vision_demo::GUI::drawAllResult(service, image, result);
    }

    if (sSaveImageCheckResultStatus == 1) {
        if(!saveImgPath.empty()) {
            vision_demo::GUI::saveDrawImg(image, saveImgPath);
        }
    }

    if (sShowImg == 1) {
        vision_demo::GUI::showImg(image);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// showResources
// ---------------------------------------------------------------------------------------------------------------------

void readFile(string &filepath, unsigned char *buffer) {
    int size = 0;
    std::ifstream file(filepath, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        file.read((char *)buffer, size);
    }
    file.close();
}

void showImg(string &imgfile, int w, int h, int c, int type) {
    auto *buffer = (unsigned char *)malloc(w * h * c);
    readFile(imgfile, buffer);
    auto img = cv::Mat(h, w, type, buffer);
    imshow("image", img);
    cv::waitKey(0);
    free(buffer);
}

// use to convert
//	auto img = cv::Mat(1300, 1600, CV_8UC2, yuv);
//	cv::cvtColor(img, mat_dst, FrameConvertFormat::COLOR_YUV2RGB_UYVY);

void showResources() {
    switch (sResType) {
    case ResType::IMG_UYVY: {
        showImg(sTestImg, sImgWidth, sImgHeight, 2, CV_16UC1);
        break;
    }
    case ResType::IMG_GRAY: {
        showImg(sTestImg, sImgWidth, sImgHeight, 1, CV_8UC1);
        break;
    }
    case ResType::IMG_RGB:
    case ResType::IMG_BGR: {
        showImg(sTestImg, sImgWidth, sImgHeight, 3, CV_8UC3);
        break;
    }
    case ResType::IMG_JPG: {
        cv::Mat img = cv::imread(sTestImg);
        cv::imshow("img", img);
        cv::waitKey(0);
        break;
    }
    case ResType::VIDEO_UYVY: {
        break;
    }
    case ResType::VIDEO_3GP: {
        break;
    }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// detect_with_camera
// ---------------------------------------------------------------------------------------------------------------------

void subThread(cv::Mat frame, VisionRequest *req, VisionService *service, VisionResult *res) {
    if (frame.empty()) {
        return;
    }
    cv::Mat fixed_image = ImageUtil::fix_image_size(frame, sImgWidth, sImgHeight);
    //        auto yuv_image = ImageUtil::bgr2nv21(fixed_image);
    req->setFrame(fixed_image.data);
    service->detect(req, res);
    std::string path = "";
    if(!sSaveImgDir.empty()) {
        path = sSaveImgDir + "/";
        path.append(std::to_string(req->timestamp)).append(".png");
    }
    drawAllResultShowAndSave(service, fixed_image, res, path);
    req->clear();
    res->clear();
    std::lock_guard<std::mutex> lk(detectingMutex);
    isDetecting = false;
}

void detect_with_camera(VisionService *service) {
// OSX系统前置摄像头的device_id为0.支持的分辨率为720P
#ifdef BUILD_OSX
    cv::VideoCapture capture(0);
    capture.set(3, sImgWidth);  // 1280 CAP_PROP_FRAME_WIDTH
    capture.set(4, sImgHeight); // 720  CAP_PROP_FRAME_HEIGHT
#else
    cv::VideoCapture capture(0);
    // 默认OpenCV的UI的框就是为我们图片尺寸大小，如果当前摄像头不支持此分辨率输出可以进行调整
    capture.set(3, sImgWidth);  // 1280 CAP_PROP_FRAME_WIDTH
    capture.set(4, sImgHeight); // 720  CAP_PROP_FRAME_HEIGHT
#endif
    if (!capture.isOpened()) {
        VLOGE(TAG, "Fail to open video!!!!");
        return;
    }
    auto *req = service->make_request();
    req->vehicleInfo->speed = 70;
    req->vehicleInfo->steeringWheelAngle = -5;
    auto *res = service->make_result();
    req->width = sImgWidth;
    req->height = sImgHeight;
    req->format = FrameFormat::RGB;
    while (true) {
        cv::Mat frame;
        capture >> frame;
        std::lock_guard<std::mutex> lk(detectingMutex);
        if (!isDetecting) {
            isDetecting = true;
            std::thread t(subThread, frame, req, service, res);
            t.detach();
        }
    }
    service->recycle_request(req);
    service->recycle_result(res);
}

// ---------------------------------------------------------------------------------------------------------------------
// detect_with_folder
// ---------------------------------------------------------------------------------------------------------------------
void detect_with_folder(VisionService *service, std::string src_path, std::string saveImgDir) {
    auto *req = service->make_request();
    req->vehicleInfo->speed = 70;
    req->vehicleInfo->steeringWheelAngle = -5;
    auto *res = service->make_result();
    auto startTime = std::chrono::system_clock::now();
    req->width = sImgWidth;
    req->height = sImgHeight;

    cv::Mat fixed_image;
    cv::Mat yuv_image;
    cv::Mat image;

    unsigned char * buffer = nullptr;

    std::string folder_path;
    if (!src_path.empty()) {
        folder_path = src_path;
    } else {
        folder_path = "../../../../vision/examples/res";
    }
    struct dirent *dirp;
    DIR *dir = opendir((char *)folder_path.data());
    std::vector<std::string> file_vector;

    while ((dirp = readdir(dir)) != nullptr) {
        std::string file_name(dirp->d_name);
        if (file_name == "." || file_name == ".." || file_name == ".DS_Store") {
            continue;
        }
        file_vector.push_back(file_name);
    }
    std::sort(file_vector.begin(), file_vector.end(), compare_timestamp);
    std::string ds_file(".DS_Store");
    for (int i = 0; i < file_vector.size(); ++i) {
        std::string filename = file_vector[i];
        std::vector<std::string> filename_split = split(filename, "_");
        std::string timestap = split(filename_split[filename_split.size() - 1], ".")[0];
        std::string date_str;
        time_to_str(timestap, date_str);
        std::cout << date_str << std::endl;
        std::string img_path(folder_path);
        std::string out_path(saveImgDir);
        img_path.append("/").append(filename);
        out_path.append("/").append(filename);
        req->clear();
        res->clear();
        std::cout << "[ImgPath] " + img_path << std::endl;
        image = cv::imread(img_path);
        if (req->format == FrameFormat::YUV_420_NV21) {
            fixed_image = ImageUtil::fix_image_size(image, sImgWidth, sImgHeight);
            yuv_image = ImageUtil::bgr2nv21(fixed_image);
            buffer = yuv_image.data;
        } else if (req->format == FrameFormat::YUV_422_UYVY) {
            fixed_image = ImageUtil::fix_image_size(image, sImgWidth, sImgHeight);
            // 改造自NV21的算法逻辑
            yuv_image.create(sImgHeight, sImgWidth, CV_16UC1);
            ImageUtil::bgr2uyvy(fixed_image, yuv_image);
            buffer = yuv_image.data;
        } else if (req->format == FrameFormat::BGR) {
            fixed_image = ImageUtil::fix_image_size(image, sImgWidth, sImgHeight);
            buffer = fixed_image.data;
        }
        req->setFrame(buffer);
        std::time_t nowTime = std::time(nullptr);
        req->timestamp = ((int64_t)nowTime) * 1000;
        service->detect(req, res);
        print_result(service, res);
        drawAllResultShowAndSave(service, fixed_image, res, out_path);
#ifdef SAVE_IMG
        writer.write(fixed_image);
#endif
    }
    service->recycle_request(req);
    service->recycle_result(res);

    closedir(dir);
}
// ---------------------------------------------------------------------------------------------------------------------
// detect_with_video
// ---------------------------------------------------------------------------------------------------------------------
void detect_with_video(VisionService *service, std::string &src) {
    cv::VideoCapture capture(0);
    capture.open(src);
    if (!capture.isOpened()) {
        VLOGE(TAG, "Fail to open video");
        return;
    }
    double totalFrameCount = capture.get(7); // CAP_PROP_FRAME_COUNT
    VLOGD(TAG, "total frame count:  %f", totalFrameCount);
    double frameToStart = 1;
    capture.set(1, frameToStart); // CAP_PROP_POS_FRAMES
    VLOGD(TAG, "detect frame from:  %f", frameToStart);
    //获取本地视频的帧率
    double rate = capture.get(5); // CAP_PROP_FPS
    VLOGD(TAG, "this video frame rate: %f", rate);

    double currentFrame = frameToStart;
    cv::Mat frame;
    while (currentFrame < totalFrameCount) {
        if (!capture.read(frame)) {
            VLOGE(TAG, "read video capture failed");
            return;
        }
        frame = ImageUtil::fix_image_size(frame, sImgWidth, sImgHeight);
        auto image = ImageUtil::bgr2nv21(frame);
        auto *req = service->make_request();
        auto *res = service->make_result();
        req->vehicleInfo->speed = 70;
        req->vehicleInfo->steeringWheelAngle = -5;
        req->width = sImgWidth;
        req->height = sImgHeight;
        req->setFrame(image.data);
        std::time_t nowTime = std::time(nullptr);
        req->timestamp = ((int64_t)nowTime) * 1000;
        // 执行模型的检测
        service->detect(req, res);
        print_result(service, res);
        std::string path = "";
        if(!sSaveImgDir.empty()) {
            path = sSaveImgDir + "/";
            path.append(std::to_string(req->timestamp)).append(".png");
        }
        drawAllResultShowAndSave(service, image, res, path);
#ifdef SAVE_IMG
        writer.write(image);
#endif
        // 清楚数据
        req->clear();
        res->clear();
        service->recycle_request(req);
        service->recycle_result(res);
        currentFrame++;
    }
    VLOGD(TAG, "detect video frame end: %f", currentFrame);
}

void detect_with_video_folder(VisionService *service, std::string &path) {
    std::vector<std::string> file_vector;
    fileHelper::getDirFile(path, file_vector);

    do {
        for (auto it : file_vector) {
            detect_with_video(service, it);
        }
    } while (sLoopCount == -1);
}

// ---------------------------------------------------------------------------------------------------------------------
// detect_with_image
// ---------------------------------------------------------------------------------------------------------------------

void detect_with_image(VisionService *service, std::string &image_path) {
    std::string img_path;
    if (!image_path.empty()) {
        img_path = image_path;
    } else {
        img_path = "../../../../vision/examples/res/test_face.jpg";
    }

    auto *req = service->make_request();
    req->vehicleInfo->speed = 70;
    req->vehicleInfo->steeringWheelAngle = -5;
    auto *res = service->make_result();
    auto startTime = std::chrono::system_clock::now();
    req->width = sImgWidth;
    req->height = sImgHeight;

    cv::Mat fixed_image;
    cv::Mat yuv_image;

    unsigned char * buffer = nullptr;

    if (req->format == FrameFormat::YUV_420_NV21) {
        cv::Mat image = cv::imread(img_path);
        fixed_image = ImageUtil::fix_image_size(image, sImgWidth, sImgHeight);
        yuv_image = ImageUtil::bgr2nv21(fixed_image);
        buffer = yuv_image.data;
        // DBG_PRINT_ARRAY((char *)yuv_image.data, 100, "image_raw_data_nv21");
    } else if (req->format == FrameFormat::YUV_422_UYVY) {
        cv::Mat image = cv::imread(img_path);
        fixed_image = ImageUtil::fix_image_size(image, sImgWidth, sImgHeight);
        // 改造自NV21的算法逻辑
        yuv_image.create(sImgHeight, sImgWidth, CV_16UC1);
        ImageUtil::bgr2uyvy(fixed_image, yuv_image);
        buffer = yuv_image.data;
        // cv::imshow("nv21_convert", yuv_image);
        // cv::waitKey(0);
    } else if (req->format == FrameFormat::BGR) {
        cv::Mat image = cv::imread(img_path);
        fixed_image = ImageUtil::fix_image_size(image, sImgWidth, sImgHeight);
        buffer = fixed_image.data;
        // DBG_PRINT_ARRAY((char *)fixed_image.data, 100, "image_raw_data_bgr");
        // cv::imshow("image", fixed_image);
        // cv::waitKey(0);
    } else if (req->format == FrameFormat::UYVY_BUFFER) {
        // 待检测图像为UYVY原图时，走此处直接读取UYVY的原始数据文件，后续转灰度时与 FrameFormat::YUV_422_UYVY 操作一致
        buffer = (unsigned char *) malloc(sImgWidth * sImgHeight * 3);
        readFile(img_path, buffer);
        // showImg(img_path, sImgWidth, sImgHeight, 2, CV_16UC1);
        FILE *fp = fopen(img_path.c_str(), "rb");
        unsigned long size = fread(buffer, 1, 1600 * 2 * 1300, fp);
        fclose(fp);
    }

    for (int i = 1; sLoopCount == -1 || i <= sLoopCount; i++) {
        auto loopStart = std::chrono::system_clock::now();
        req->clear();
        res->clear();
        req->vehicleInfo->speed = 70.f;
        req->vehicleInfo->steeringWheelAngle = -5.f;
        req->setFrame(buffer);
        std::time_t nowTime = std::time(nullptr);
        req->timestamp = ((int64_t)nowTime) * 1000;
        service->detect(req, res);
        print_result(service, res);
        std::string path = "";
        if(!sSaveImgDir.empty()) {
            path = sSaveImgDir + "/";
            path.append(std::to_string(req->timestamp)).append(".png");
        }
        drawAllResultShowAndSave(service, fixed_image, res, sSaveImgDir);
#ifdef SAVE_IMG
        writer.write(fixed_image);
#endif

        PerfUtil *perf = res->getPerfUtil();
        if (perf) {
            perf->printDetectRecords();
        }

        //        std::int64_t execTime = PerfUtil::global()->getTotalTime();
        //        auto records = PerfUtil::global()->get_records();

        auto now = std::chrono::system_clock::now();
        auto loopTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - loopStart).count();
        std::cout << "[Perf]: ===> [CurLoopTime]: " << loopTime << " (millis)" << std::endl;
        //        auto totalTime = (now - startTime).count() / 1000;
        //        std::cout << "======> [TotalTime]: " << totalTime << " [TotalCount]: "  << i << " [AvgLoopTime]: " << (totalTime / i) << " [CurLoopTime]: " << loopTime << " [CurQnnTime]: " << execTime << " (millis)" << std::endl; std::cout << "===---> "; for (auto it : records) {
        //            std::cout << "[" << it.first << " : " << it.second << "] ";
        //        }
        //        std::cout << std::endl;

        int dur = sFrameDelay - loopTime;
        if (dur > 0) {
            std::cout << "[Perf]: ===> Sleep for " << dur << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(dur));
        }
    }
    service->recycle_request(req);
    service->recycle_result(res);
}

// ---------------------------------------------------------------------------------------------------------------------
// Main Methods
// ---------------------------------------------------------------------------------------------------------------------

void detect(VisionService *service, DemoType type, std::string src_path = "", std::string dst_path = "") {
    switch (type) {
    case DemoType::CAMERA:
        detect_with_camera(service);
        break;
    case DemoType::IMAGE:
        detect_with_image(service, src_path);
        break;
    case DemoType::VIDEO:
        detect_with_video(service, src_path);
        break;
    case DemoType::IMAGE_FOLDER:
        detect_with_folder(service, src_path, dst_path);
        break;
    case DemoType::SHOW_RESOURCES:
        showResources();
        break;
    case DemoType::VIDEO_FOLDER:
        detect_with_video_folder(service, src_path);
        break ;
    default:
        break;
    }
}

void init(VisionService *service) {
    PerfUtil::qnnLoopModel = sQnnLoopModelId;

    service->init();
    service->set_config(ParamKey::USE_INTERNAL_MEM, true);
    service->set_config(ParamKey::FRAME_WIDTH, sImgWidth);
    service->set_config(ParamKey::FRAME_HEIGHT, sImgHeight);
    // 注意： 如果是要和算法原始模型对齐。需要使用FrameFormat::BGR
    // 注意： 如果是要和服务层读图模式对齐的话。需要使用FrameFormat::YUV_422_UYVY
    // YUV_420_NV21、YUV_422_UYVY、BGR
    service->set_config(ParamKey::FRAME_FORMAT, (short)gFrameFormat);  // FrameFormat::BGR
    service->set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_IR);
    // 注意：如果仅是单独测试图片检测结果。需要开启BenchmarkTest
    // service->set_config(ParamKey::RELEASE_MODE, BENCHMARK_TEST);
    service->set_config(ParamKey::SCHEDULE_METHOD, sScheduleMethod);
    service->set_config(ParamKey::SCHEDULE_DAG_THREAD_COUNT, sScheduleDagThreads);
    // 开启人脸3D视线的一点标定的逻辑
    service->set_config(ParamKey::EYE_GAZE_CALIB_SWITCHER, 1.0);

    auto abilityMap = std::unordered_map<short, bool>{
        {ABILITY_FACE,                       true},
        {ABILITY_FACE_RECT,                  true},
        {ABILITY_FACE_LANDMARK,              true},
        {ABILITY_FACE_3D_LANDMARK,           true},
        {ABILITY_FACE_ATTRIBUTE,             true},
        {ABILITY_FACE_EMOTION,               true},
        {ABILITY_FACE_FEATURE,               true},
        {ABILITY_FACE_QUALITY,               true},
        {ABILITY_FACE_DANGEROUS_DRIVING,     true},
        {ABILITY_FACE_FATIGUE,               true},
        {ABILITY_FACE_ATTENTION,             true},
        {ABILITY_FACE_HEAD_BEHAVIOR,         false},
        {ABILITY_FACE_INTERACTIVE_LIVING,    true},
        {ABILITY_FACE_CALL,                  true},
        {ABILITY_FACE_NO_INTERACTIVE_LIVING, true},
        {ABILITY_FACE_EYE_GAZE,              true},                 // 关闭原来视线检测逻辑
        {ABILITY_FACE_3D_EYE_GAZE,           true},
        {ABILITY_FACE_EYE_TRACKING,          false},
        {ABILITY_GESTURE,                    true},
        {ABILITY_GESTURE_RECT,               true},
        {ABILITY_GESTURE_LANDMARK,           true},
        {ABILITY_GESTURE_TYPE,               false},
        {ABILITY_GESTURE_DYNAMIC,            false},
        {ABILITY_PLAY_PHONE_DETECT,          true},
        {ABILITY_FACE_EYE_CENTER,            true},
        {ABILITY_FACE_MOUTH_LANDMARK,        true},
        {ABILITY_LIVING_DETECTION, true},
        {ABILITY_BODY_HEAD_SHOULDER,         true},
        {ABILITY_BODY_LANDMARK,              true},
        {ABILITY_CAMERA_COVER,               true},
        {ABILITY_FACE_LIP_MOVEMENT,          true}
    };

    service->set_switches(abilityMap);
#if 0
    if(sGenerateAbilityTableFlag == true || isFileExistsStat(sGenerateAbilityTableDir) == false) {
        Json::Value root;
        for (auto it : abilityMap) {
            root[std::to_string(it.first)] = it.second;
        }

        Json::Value describe;

        root.toStyledString();

        Json::StyledWriter sw;
        std::ofstream os;
        os.open(sGenerateAbilityTableDir);
        os << sw.write(root);
        os.close();
    }

    Json::Reader reader;
    Json::Value value;
    std::ifstream infile(sGenerateAbilityTableDir);

    if(!infile.is_open() || infile.bad()) {
        return;
    }

    if (!reader.parse(infile, value)) {
        infile.close();
        return;
    }

    Json::Value::Members members = value.getMemberNames();
    for (Json::Value::Members::iterator it = members.begin(); it != members.end(); it++) {
        short key = atoi((*it).c_str());
        GUI::sAbilityMap[key] = value[*it].asBool();
    }
#endif
}

void printUsage();
void parseArgs(int argc, char *argv[]);

int main(int argc, char **argv) {
    if (argc == 1) {
        printUsage();
        return 0;
    }
    parseArgs(argc, argv);

    if (sDelayToStart > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(sDelayToStart));
    }

    // 设置日志开关结果
    Logger::init();
    Logger::setLogLevel(LogLevel::VERBOSE);
    Logger::setLogDest(vision::LogDest::ALL);
    VisionInitializer initializer;
    initializer.init(sModelPath);
//    initializer.init("./vision_model.bin");
    VisionService service(sSourceIndex);
    init(&service);

#ifdef SAVE_IMG
    std::string fileName("./calibration.mp4");
    if (argc > 2) {
        fileName.assign(argv[2]);
    }
    writer.open(fileName, CV_FOURCC('M', 'J', 'P', 'G'),
                10, cv::Size(sImgWidth, sImgHeight));
#endif
    detect(&service, sDemoType, sTestImg, sSaveImgDir);
#ifdef SAVE_IMG
    writer.release();
#endif

    // 测试设置设置所有的QNNPredictor进行休眠
    VisionInitializer::setInferenceCmd(SOURCE_1, InferenceCmd::CMD_INFERENCE_POWER_POWER_SAVE);
    VisionInitializer::setInferenceCmd(SOURCE_2, InferenceCmd::CMD_INFERENCE_POWER_POWER_SAVE);
    VisionInitializer::setInferenceCmd(SOURCE_3, InferenceCmd::CMD_INFERENCE_POWER_POWER_SAVE);

    initializer.deinit();

    return 0;
}

void printUsage() {
    fprintf(stdout, "USAGE:\n"
                    "       -dt         integer | Demo Type  Valid values are:\n"
                    "                   IMAGE            = 0 (default)\n"
                    "                   CAMERA           = 1\n"
                    "                   VIDEO            = 2\n"
                    "                   IMAGE_FOLDER     = 3\n"
                    "                   SHOW_RESOURCES   = 4\n"
                    "                   VIDEO_FOLDER     = 5\n"
                    "       -lc         integer | Loop Count [Default 1], -1 for while \n"
                    "       -im         string  | Image path or VIDEO path\n"
                    "       -sh         integer | Show image; [default 0: 不显示],[1: 显示]\n"
                    "       -fd         integer | Frame delay millis [default 0]\n"
                    "       -sm         integer | Schedule method [Naive - 0, DAG - 1(default)]\n"
                    "       -st         integer | DAG schedule threads num [default 2]\n"
                    "       -ql         integer | Qnn loop model id [default 0]\n"
                    "       -ds         integer | Delay seconds to start for debugging\n"
                    "       -rt         integer | Resource type for -dt=SHOW_RESOURCES\n"
                    "       -iw         integer | Image width  [default 1600]\n"
                    "       -ih         integer | Image height [default 1300]\n"
                    "       -mp         string  | Model file path\n"
                    "       -source     string  | Source Type [Source1 - 1, Source2 - 2]\n"
                    "       -if         integer | Image Format Valid values are:\n"
                    "                   UNKNOWN         = 0 (default)\n"
                    "                   YUV_422_UYVY    = 1\n"
                    "                   YUV_420_NV21    = 2\n"
                    "                   YUV_420_NV12    = 3\n"
                    "                   BGR             = 4\n"
                    "                   RGB             = 5\n"
                    "                   UYVY_BUFFER     = 6\n"
                    "       -sid        string  | Save Image Dir\n"
                    "       -gat        generate ability table\n"
                    "       -sir        save check result to image [default 0:不保存],[1:保存]\n"
    );
}

void parseArgs(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], kDemoType, strlen(kDemoType)) == 0) {
            ++i;
            if (i < argc) {
                sDemoType = (DemoType)atoi(argv[i]);
                printf("DetectType : %d\n", sDemoType);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kDemoType);
            }
        } else if (strncmp(argv[i], kTestImg, strlen(kTestImg)) == 0) {
            ++i;
            if (i < argc) {
                sTestImg = argv[i];
                printf("TestImage : %s\n", sTestImg.c_str());
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kTestImg);
            }
        } else if (strncmp(argv[i], kScheduleMethod, strlen(kScheduleMethod)) == 0) {
            ++i;
            if (i < argc) {
                sScheduleMethod = atoi(argv[i]);
                printf("TestImage : %d\n", sScheduleMethod);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kScheduleMethod);
            }
        } else if (strncmp(argv[i], kLoopCount, strlen(kLoopCount)) == 0) {
            ++i;
            if (i < argc) {
                sLoopCount = atoi(argv[i]);
                printf("LoopCount : %d\n", sLoopCount);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kLoopCount);
            }
        } else if (strncmp(argv[i], kFrameDelay, strlen(kFrameDelay)) == 0) {
            ++i;
            if (i < argc) {
                sFrameDelay = atoi(argv[i]);
                printf("FrameDelay : %d\n", sFrameDelay);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kFrameDelay);
            }
        } else if (strncmp(argv[i], kScheduleDagThreads, strlen(kScheduleDagThreads)) == 0) {
            ++i;
            if (i < argc) {
                sScheduleDagThreads = atoi(argv[i]);
                printf("ScheduleDagThreads : %d\n", sScheduleDagThreads);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kScheduleDagThreads);
            }
        } else if (strncmp(argv[i], kQnnLoopModelId, strlen(kQnnLoopModelId)) == 0) {
            ++i;
            if (i < argc) {
                sQnnLoopModelId = atoi(argv[i]);
                printf("QnnLoopModelId : %d\n", sQnnLoopModelId);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kQnnLoopModelId);
            }
        } else if (strncmp(argv[i], kDelayToStart, strlen(kDelayToStart)) == 0) {
            ++i;
            if (i < argc) {
                sDelayToStart = atoi(argv[i]);
                printf("DelayToStart : %d\n", sDelayToStart);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kDelayToStart);
            }
        } else if (strncmp(argv[i], kResType, strlen(kResType)) == 0) {
            ++i;
            if (i < argc) {
                sResType = atoi(argv[i]);
                printf("ShowImg : %d\n", sResType);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kResType);
            }
        } else if (strncmp(argv[i], kImgWidth, strlen(kImgWidth)) == 0) {
            ++i;
            if (i < argc) {
                sImgWidth = atoi(argv[i]);
                printf("ImgWidth : %d\n", sImgWidth);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kImgWidth);
            }
        } else if (strncmp(argv[i], kImgHeight, strlen(kImgHeight)) == 0) {
            ++i;
            if (i < argc) {
                sImgHeight = atoi(argv[i]);
                printf("ImgHeight : %d\n", sImgHeight);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kImgHeight);
            }
        } else if (strncmp(argv[i], kSourceIndex, strlen(kSourceIndex)) == 0) {
            ++i;
            if (i < argc) {
                sSourceIndex = atoi(argv[i]);
                printf("SourceIndex : %d\n", sSourceIndex);
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kSourceIndex);
            }
        } else if (strncmp(argv[i], "--help", 6) == 0) {
            ++i;
            if (i < argc) {
                printUsage();
            }
        } else if (strncmp(argv[i], kModelPath, strlen(kModelPath)) == 0) {
            ++i;
            if (i < argc) {
                sModelPath = argv[i];
                printf("ModelPath : %s\n", sModelPath.c_str());
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kModelPath);
            }
        } else if (strncmp(argv[i], kFrameFormat, strlen(kModelPath)) == 0) {
            ++i;
            if (i < argc) {
                gFrameFormat = static_cast<FrameFormat>(atoi(argv[i]));
                printf("kFrameFormat : %d\n", static_cast<int>(gFrameFormat));
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kFrameFormat);
            }
        } else if (strncmp(argv[i], kSaveImgDir, strlen(kSaveImgDir)) == 0) {
            ++i;
            if (i < argc) {
                sSaveImgDir = argv[i];
                printf("SaveImgDir : %s\n", sSaveImgDir.c_str());
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", kSaveImgDir);
            }
        } else if (strncmp(argv[i], sSaveImageCheckResult, strlen(sSaveImageCheckResult)) == 0) {
            ++i;
            if (i < argc) {
                sSaveImageCheckResultStatus = atoi(argv[i]);
                printf("SaveImageCheckResultStatus : %d\n", sSaveImageCheckResultStatus);
            } else {
                fprintf(stderr, "ERROR: missing param sSaveImageCheckResultStatus\n");
            }
        } else if (strncmp(argv[i], kShowImg, strlen(kShowImg)) == 0) {
            ++i;
            if (i < argc) {
                sShowImg = atoi(argv[i]);
                printf("sShowImg : %d\n", sShowImg);
            } else {
                fprintf(stderr, "ERROR: missing param sShowImg\n");
            }
#if defined(BUILD_QNX)
        } else if (strncmp(argv[i], pGenerateAbilityTable, strlen(pGenerateAbilityTable)) == 0) {
            ++i;
            if (i < argc) {
                sGenerateAbilityTableDir = argv[i];
                sGenerateAbilityTableFlag = true;
                printf("GenerateAbilityTableDir : %s\n", sGenerateAbilityTableDir.c_str());
            } else {
                fprintf(stderr, "ERROR: missing param with option %s\n", pGenerateAbilityTable);
            }
#endif
        }
    }
}