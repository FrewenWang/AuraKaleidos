#include <fstream>
#include <functional>
#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"
#include "vision/VisionAbility.h"

using namespace aura::vision;

using DetectCallback = std::function<void(FaceInfo*)>;

static const std::string face_db_file = "face_feature.bin";
static const std::string face_img_1 = "../res/face1.jpg";
static const std::string face_img_2 = "../res/face2.jpg";
static const std::string face_img_3 = "../res/face3.jpg";
static const int FEAT_LEN = 1024; // 人脸特征数据长度
static const float FEAT_COMP_THRESHOLD = 0.45F; // 人脸比对阈值

/**
 * BGR转换为NV21格式
 */ 
void bgr2nv21(unsigned char *src, unsigned char *dst, int width, int height) {
    if (src == nullptr || dst == nullptr || width % 2 != 0 || height % 2 != 0) {
        return;
    }

    static const unsigned int R2YI = 4899;
    static const unsigned int G2YI = 9617;
    static const unsigned int B2YI = 1868;
    static const unsigned int B2UI = 9241;
    static const unsigned int R2VI = 11682;
    static unsigned short shift = 14;
    static unsigned int coeffs[5] = {B2YI, G2YI, R2YI, B2UI, R2VI};
    static unsigned int offset = 128 << shift;

    unsigned char *y_plane = dst;
    unsigned char *vu_plane = dst + width * height;

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; ++c) {
            int Y = (unsigned int) (src[0] * coeffs[0] + src[1] * coeffs[1] + src[2] * coeffs[2]) >> shift;
            *y_plane++ = (unsigned char) Y;

            if (r % 2 == 0 && c % 2 == 0) {
                int U = (unsigned int) ((src[0] - Y) * coeffs[3] + offset) >> shift;
                int V = (unsigned int) ((src[2] - Y) * coeffs[4] + offset) >> shift;

                vu_plane[0] = (unsigned char) V;
                vu_plane[1] = (unsigned char) U;
                vu_plane += 2;
            }
            src += 3;
        }
    }
}

/**
 * 读取人脸特征数据文件
 */ 
int read_file(std::string file_name, void* buf) {
    if (buf == nullptr) {
        return -1;
    }
    std::ifstream ifs(file_name, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        std::cerr << "read file failed! (" << file_name << ")" << std::endl;
        return -1;
    }

    auto len = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (len != FEAT_LEN * sizeof(float)) {
        std::cerr << "feature file length error!" << std::endl;
        return -1;
    }

    ifs.read((char*)buf, len);
    ifs.close();

    return 0;
}

/**
 * 写入人脸特征数据到文件
 */ 
int write_file(std::string file_name, const void* buf, int len) {
    std::ofstream ofs(file_name, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "write file failed! (" << file_name << ")" << std::endl;
        return -1;
    }

    ofs.write((const char*)buf, len);
    ofs.close();

    return 0;
}

/**
 * 初始化SDK及参数配置
 */ 
void init(VisionService* service) {
    service->init(); // SDK初始化
    service->set_config(ParamKey::USE_INTERNAL_MEM, true); // 内部创建人脸数据所需的内存，若设为false，则需要外部传入内存
    service->set_config(ParamKey::FRAME_CONVERT_FORMAT, COLOR_YUV2BGR_NV21); // 颜色转换格式，对应输入为NV21
    service->set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_RGB); // 设置图片颜色模式为RGB，会根据颜色选择对应的模型
    service->set_config(ParamKey::RELEASE_MODE, BENCHMARK_TEST); // 调试模式（产品发布是不需要设置）
    service->set_config(ParamKey::NATIVE_LOG_LEVEL, static_cast<float>(LogLevel::ERROR)); // 日志级别
    // 设置检测开关
    service->set_switches(
            std::unordered_map<short, bool> {
                    {ABILITY_ALL,                           true}, // 总的开关，总是打开
                    {ABILITY_FACE,                          true}, // 人脸能力的总开关，检测人脸时总是打开
                    {ABILITY_FACE_RECT,                     true}, // 人脸框检测开关
                    {ABILITY_FACE_LANDMARK,                 true}, // 人脸关键点检测开关
                    {ABILITY_FACE_QUALITY,                  true}, // 人脸质量（遮挡）检测开关
                    {ABILITY_FACE_FEATURE,                  true}  // 人脸特征检测开关
    });
}

/**
 * 读取图片数据
 */ 
cv::Mat read_image(std::string img_path, int& w, int& h) {
    auto image = cv::imread(img_path);
    std::cout << "image w=" << image.cols << ", h=" << image.rows << std::endl;
    cv::Mat yuv_image(image.rows * 3 / 2, image.cols, CV_8UC1);
    bgr2nv21(image.data, yuv_image.data, image.cols, image.rows); // 将读取到RGB转换为NV2格式
    return yuv_image;
}

/**
 * 人脸检测
 */ 
int detect(VisionService* service, std::string img_path, const DetectCallback& callback) {
    if (!service) {
        std::cerr << "vision service is nullptr" << std::endl;
        return -1;
    }

    int width = 0;
    int height = 0;
    auto yuv = read_image(img_path, width, height); // 读取图片，并转换nv21格式，SDK目前接受YUV输入格式
    if (yuv.empty()) {
        std::cerr << "read image failed!" << std::endl;
        return -1;
    }

    auto* req = service->make_request(); // 创建请求
    auto* res = service->make_result(); // 创建结果
    req->clear_all(); // 清空请求数据
    res->clear_all(); // 清空结果数据

    req->_width = width; // 请求数据的宽
    req->_height = height; // 请求数据的高
    req->_frame = yuv.data; // 请求的图片帧数据

    service->detect(req, res); // 执行检测

    // 若检测成功，则执行回调
    if (callback != nullptr) {
        callback(res->get_face_result()->_face_infos[0]);
    }

    service->recycle_request(req); // 回收请求数据到对象池
    service->recycle_result(res); // 回收结果数据到对象池

    return 0;
}

void print_face_info(FaceInfo* fi) {
    std::cout << "detect face: " << std::boolalpha << fi->hasFace() << std::endl // 是否检测到人脸
              << "face pose: yaw=" << fi->_head_deflection.yaw  // 人脸姿态角yaw（左右偏转角）
              << ", pitch=" << fi->_head_deflection.pitch << std::endl // 人脸姿态角pitch（上下偏转角）
              << "face position: (" << fi->_rect_lt.x << ", " << fi->_rect_lt.y << ")" // 人脸检测框位置（像素）
              << ", (" << fi->_rect_rb.x << ", " << fi->_rect_rb.y << ")" << std::endl 
              << "face mask: " << ((int)fi->_state_window_cover == F_QUALITY_COVER_GOOD ? "No" : "Yes") // 人脸是否遮挡（如戴口罩）
              << std::endl;
}

/**
 * 注册人脸，提取人脸特征，并存储到本地文件
 */
void register_face(VisionService* service, const std::string& img_path) {
    detect(service, img_path, [](FaceInfo* fi) {
        print_face_info(fi); // 输出检测到的人脸基本信息

        if (!fi->hasFace()) { // 未检测到人脸，则返回
            return;
        }

        auto ret = write_file(face_db_file, fi->_feature, FEAT_LEN * sizeof(float)); // 人脸特征写入文件
        if (ret == 0) {
            std::cout << "register SUCCESS!" << std::endl;
        } else {
            std::cout << "register FAILED!" << std::endl;
        }
    });
}

/**
 * 识别人脸，当前人脸与已注册的人脸特征进行比对
 */
void recognize(VisionService* service, const std::string& img_path) {
    detect(service, img_path, [](FaceInfo* fi) {
        print_face_info(fi); // 输出检测到的人脸基本信息

        if (!fi->hasFace()) { // 未检测到人脸，则返回
            return;
        }

        float* feat_saved = new float[FEAT_LEN];
        read_file(face_db_file, feat_saved); // 从本地文件读取已注册的人脸

        auto score = FaceIdUtil::compare_face_features(feat_saved, fi->_feature); // 人脸特征比对

        std::cout << "compare score=" << score << std::endl;
        if (score > FEAT_COMP_THRESHOLD) {
            std::cout << "recognize SUCCESS!" << std::endl; // 比对分值大于阈值，则识别成功
        } else {
            std::cout << "recognize FAILED!" << std::endl;
        }

        delete[] feat_saved;
    });
}

int main(int argc, char** argv) {
    std::string detect_mode = "register";
    /* 运行方法：
     * ./face_detect [mode]
     * mode有三种选项：
     *  register -- 注册人脸
     *  recognize -- 识别人脸
     *  mask -- 戴口罩检测
    */
    if (argc > 1) {
        detect_mode = argv[1];
    }

    VisionInitializer initializer; // 初始化器，用于模型加载
    initializer.init(); // 加载模型

    VisionService service; // 视觉能力API接口类
    init(&service); // 初始化service，设置相关参数

    if (detect_mode == "register") {
        register_face(&service, face_img_1); // 注册人脸
    } else if (detect_mode == "recognize") {
        recognize(&service, face_img_2); // 识别人脸
    } else if (detect_mode == "mask") {
        recognize(&service, face_img_3); // 戴口罩检测
    } else {
        std::cerr << "usage: face_detect [mode]" << std::endl
                  << "mode:\tregister: register a face;"
                  << "\n\trecognize: compare a face to the registerd face template"
                  << "\n\tmask: detect a face with mask" << std::endl; 
    }
    return 0;
}