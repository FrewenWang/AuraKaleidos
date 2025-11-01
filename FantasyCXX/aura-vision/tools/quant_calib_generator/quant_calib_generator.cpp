#include <array>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#include "opencv2/opencv.hpp"
#include "vision/VisionAbility.h"

#include "detector/FaceNoInteractLivingDetector.h"
#include "detector/FaceRectDetector.h"
#include "detector/FaceLandmarkDetector.h"
#include "detector/FaceQualityDetector.h"
#include "detector/FaceAttributeDetector.h"
#include "detector/FaceEmotionDetector.h"
#include "detector/FaceFeatureDetector.h"
#include "detector/FaceEyeCenterDetector.h"
#include "detector/FaceDangerousDriveDetector.h"
#include "detector/FaceCallDetector.h"
#include "detector/GestureRectDetector.h"
#include "detector/GestureLandmarkDetector.h"
#include "util/quant_calib_data_util.hpp"

#include "vision/config/runtime_config/RtConfig.h"

using namespace vision;

static std::string remote_model_quant_data_path("baiduiov@172.20.72.11:/home/baiduiov/vision-space/datasets/QuantData/");

std::string exec_cmd(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe)
    {
        return "";
    }

    while (fgets(buffer.data(), 128, pipe) != NULL) {
        result += buffer.data();
    }
    pclose(pipe);

    return result;
}

bool exec_scp(const std::string& src_path, const std::string& dst_path) {
    if (src_path.empty() or src_path.empty()) {
        std::cout << "[Error]: params invalid" << std::endl;
        return false;
    }
    auto cmd = std::string("scp -pr ") + src_path + " " + dst_path;
    exec_cmd(cmd);
    return true;
}

cv::Mat bgr_to_yuv(const cv::Mat& frame) {
    cv::Mat yuv_image;
    yuv_image.create(frame.rows * 3 / 2, frame.cols, CV_8UC1);
    cv::cvtColor(frame, yuv_image, cv::COLOR_BGR2YUV_YV12);
    return yuv_image;
}

void save_raw(const std::string& file_name, const void* data, int len) {
    std::ofstream ofs(file_name, std::ios::binary);
    if (!data || len == 0) {
        std::cerr << "data is null, ignore writing to file" << std::endl;
    }
    ofs.write((const char*)data, len);
    ofs.close();
}

void update_index(const std::string& file_name, const std::string& line) {
    if (line.empty()) {
        return;
    }
    std::ofstream ofs(file_name, std::ios::app);
    ofs << line << std::endl;
    ofs.close();
}

void write_shape(const std::string& file_name, int c, int h, int w) {
    std::ofstream ofs(file_name);
    ofs << c << " " << h << " " << w << " " << std::endl;
    ofs.close();
}

bool scan_for_images(const std::string& dir, std::vector<std::string>& images) {
    if (dir.empty()) {
        return false;
    }

    DIR* dp = nullptr;
    dp = opendir(dir.c_str());
    if (dp == nullptr) {
        std::cerr << "open directory failed! (" << dir << ")" << std::endl;
        return false;
    }

    struct dirent* entry = nullptr;
    while ((entry = readdir(dp))) {
        if (entry->d_type == DT_DIR) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            std::string path = dir + "/" + entry->d_name;
            std::cout << "Search subdir: " << entry->d_name << std::endl;
            scan_for_images(path, images);
        } else if(entry->d_type == DT_REG) {
            const char* ext = strrchr(entry->d_name,'.');
            if((!ext) || (ext == entry->d_name)) {
                continue;
            } else {
                if(strcmp(ext, ".jpg") == 0 ||
                   strcmp(ext, ".jpeg") == 0 ||
                   strcmp(ext, ".bmp") == 0 ||
                   strcmp(ext, ".png") == 0) {
                   std::string path = dir + "/" + entry->d_name;
                   images.emplace_back(path);
                }
            }
        }
    }

    closedir(dp);
    return true;
}

void print_help(char* exe_name) {
    std::cout << exe_name << " [src_img_dir] [-f] [-g]" << std::endl
              << "Usage:" << std::endl
              << "  src_img_dir:    image set used to generate the calibration set, REQUIRED" << std::endl
              << "  -f --face:      prepare calibration set for face related abilities, OPTIONAL" << std::endl
              << "  -g --gesture:   prepare calibration set for gesture related abilities, OPTIONAL" << std::endl;
}

void mkdir_if_absent(std::string dir) {
    mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir((dir + "/val").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

cv::Mat fix_image_size(cv::Mat &in, int w, int h) {
    int cols = in.cols;
    int rows = in.rows;

    if (cols == w && rows == h) {
        return in;
    }

    int new_w = 0;
    int new_h = 0;

    cv::Mat resized;

    float scale_width = cols * 1.f / w;
    float scale_height = rows * 1.f / h;

    if (scale_width > scale_height) {
        new_w = w;
        new_h = static_cast<int>(rows / scale_width);
    } else {
        new_h = h;
        new_w = static_cast<int>(cols / scale_height);
    }
    cv::resize(in, resized, cv::Size(new_w, new_h));

    int delta_rows = h - new_h;
    int delta_cols = w - new_w;
    int top = delta_rows / 2;
    int bottom = delta_rows - top;
    int left = delta_cols / 2;
    int right = delta_cols - left;
    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar{174, 174, 174});
    return out;
}

#define MAKE_FACE_CALIB true
#define MAKE_GESTURE_CALIB false

int main(int argc, char** argv) {
    	
    VisionInitializer initializer;
    initializer.init();
    // load model and set config
    VisionService service(1);
    service.init();

    RtConfig * rtConfig = service.getRtConfig().get();
	service.set_config(ParamKey::FRAME_CONVERT_GRAY_FORMAT, COLOR_YUV2GRAY_NV21);
	service.set_config(ParamKey::FRAME_CONVERT_RGB_FORMAT, COLOR_YUV2RGB_NV21);
    service.set_config(ParamKey::FRAME_CONVERT_BGR_FORMAT, COLOR_YUV2BGR_NV21);
    service.set_config(ParamKey::CAMERA_LIGHT_TYPE, CAMERA_LIGHT_TYPE_IR);
    service.set_config(ParamKey::NATIVE_LOG_LEVEL, 0);

    // parse arguments
    std::string img_dir;
    bool make_face_calib = MAKE_FACE_CALIB;
    bool make_gesture_calib = MAKE_GESTURE_CALIB;
    if (argc <= 1) {
        print_help(argv[0]);
        return -1;
    } else {
        img_dir = argv[1];
        if (argc > 2) {
            if (strcmp(argv[2], "-f") == 0 || strcmp(argv[2], "--face") == 0) {
                make_face_calib = true;
                make_gesture_calib = false;
            } else if (strcmp(argv[2], "-g") == 0 || strcmp(argv[2], "--gesture") == 0) {
                make_gesture_calib = true;
                make_face_calib = false;
            }
            if (argc > 3) {
                if (strcmp(argv[3], "-f") == 0 || strcmp(argv[3], "--face") == 0) {
                    make_face_calib = true;
                } else if (strcmp(argv[3], "-g") == 0 || strcmp(argv[3], "--gesture") == 0) {
                    make_gesture_calib = true;
                }
            }
        }
    }
    std::cout << "make_face_calib=" << make_face_calib << ", make_gesture_calib=" << make_gesture_calib << std::endl;

    // iterate over the dir for images
    std::vector<std::string> image_list;
    std::cout << "Search image dir: " << img_dir << std::endl;
    if (!scan_for_images(img_dir, image_list)) {
        return -1;
    }
    std::cout << "Found images totally: " << image_list.size() << std::endl;

    auto* res = VisionResult::obtain(rtConfig);
    
    // create detectors
#define CREATE_F_DETECTOR(det) std::dynamic_pointer_cast<AbsDetector<FaceInfo>>(std::make_shared<det>())
    std::vector <std::pair<std::string, std::shared_ptr<AbsDetector<FaceInfo>>>> face_detectors {
            {"faceRect",       CREATE_F_DETECTOR(FaceRectDetector)},
            {"faceLandmark",   CREATE_F_DETECTOR(FaceLandmarkDetector)},
            {"faceQuality",    CREATE_F_DETECTOR(FaceQualityDetector)},
            {"faceAttribute",  CREATE_F_DETECTOR(FaceAttributeDetector)},
            {"faceEmotion",    CREATE_F_DETECTOR(FaceEmotionDetector)},
            {"faceEyeCenter",  CREATE_F_DETECTOR(FaceEyeCenterDetector)},
            {"faceLiveness",   CREATE_F_DETECTOR(FaceLivenessDetector)},
            {"faceFeature",    CREATE_F_DETECTOR(FaceFeatureDetector)},
            {"DangerousDrive", CREATE_F_DETECTOR(FaceDangerousDriveDetector)},
            {"PhoneCall",      CREATE_F_DETECTOR(FaceCallDetector)}
    };
#undef CREATE_F_DETECTOR

    // 人脸相关模型量化集保存路径创建
    if (make_face_calib) {
        mkdir_if_absent("faceRect");
        mkdir_if_absent("faceLandmark");
        mkdir_if_absent("faceQuality");
        mkdir_if_absent("faceAttribute");
        mkdir_if_absent("faceEmotion");
        mkdir_if_absent("faceEyeCenter");
        mkdir_if_absent("faceLiveness");
        mkdir_if_absent("faceFeature");
        mkdir_if_absent("dangerousDrive");
        mkdir_if_absent("phoneCall");

        for (auto& det : face_detectors) {
            auto shape = QuantCalibDataUtil::get_input_shape<FaceInfo>(det.second);
            if (shape.size() < 3) {
                continue;
            }
            write_shape(det.first + "/input_shape.txt", shape[0], shape[1], shape[2]);
        }
    }

#define CREATE_G_DETECTOR(det) std::dynamic_pointer_cast<AbsDetector<GestureInfo>>(std::make_shared<det>())
    std::vector <std::pair <std::string, std::shared_ptr<AbsDetector<GestureInfo>>>> gest_detectors {
            {"GestureRect",     CREATE_G_DETECTOR(GestureRectDetector)},
            {"GestureLandmark", CREATE_G_DETECTOR(GestureLandmarkDetector)}
    };
#undef CREATE_G_DETECTOR

    // 手势相关模型量化集保存路径创建
    if (make_gesture_calib) {
        mkdir_if_absent("GestureRect");
        mkdir_if_absent("GestureLandmark");

        for (auto& det : gest_detectors) {
            auto shape = QuantCalibDataUtil::get_input_shape<GestureInfo>(det.second);
            if (shape.size() < 3) {
                continue;
            }
            write_shape(det.first + "/input_shape.txt", shape[0], shape[1], shape[2]);
        }
    }

    int f_index = 0;
    int g_index = 0;
    std::cout << "Begin to prepare the calibration set..." << std::endl;
	VisionRequest *request = service.make_request();
    for (const auto& img_file : image_list) {
        // read images
        cv::Mat image = cv::imread(img_file);
        if (image.empty()) {
            std::cerr << "read image failed! (" << img_file << ")" << std::endl;
            continue;
        }
        image = fix_image_size(image, 1280, 720);

        auto yuv_image = bgr_to_yuv(image);
//        VFrameInfo frame;
//        frame.width = 1280;
//        frame.height = 720;
//        frame.data = yuv_image.data;
		request->width = 1280;
		request->height = 720;
		request->format = FrameFormat::YUV_420_NV21;
		request->frame = yuv_image.data;
        res->clearAll();

        if (make_face_calib) {
            // face related models
            auto* face_result = res->getFaceResult();
            // detect face rect and landmark
            auto face_rect_det = face_detectors[0].second;
            face_rect_det->init(rtConfig);
//            face_rect_det->detect(frame, face_result->_face_infos, res->get_perf_util());
			face_rect_det->detect(request, res);
            auto face_lmk_det = face_detectors[1].second;
            face_lmk_det->init(rtConfig);
//            face_lmk_det->detect(frame, face_result->_face_infos, res->get_perf_util());
			face_rect_det->detect(request, res);
            FaceInfo* fi = face_result->faceInfos[0];
            if (!fi || !fi->hasFace()) {
                continue;
            }
            f_index++;
            if (f_index % 10 == 0) {
                std::cout << "processed (face): " << f_index << std::endl;
            }
            // prepare small images
            for (auto& det : face_detectors) {
                auto prepared = QuantCalibDataUtil::prepare_calib_data<FaceInfo>(det.second, rtConfig, request, face_result->faceInfos);
                if (prepared.empty()) {
                    std::cout << det.first << " prepared is empty, ignore ..." << std::endl;
                    continue;
                }
                // write raw image
                std::string file_name = "val/" + std::to_string(f_index) + ".raw";
                save_raw(det.first + "/" + file_name, prepared.data, prepared.len());
                // update index file
                update_index(det.first + "/val_list.txt", file_name);
            }
        }

        if (make_gesture_calib) {
            // gesture related models
            auto* gest_result = res->getGestureResult();
            // prepare detect for gesture rect
            auto gest_rect_det = gest_detectors[0].second;
            gest_rect_det->init(rtConfig);
//            gest_rect_det->detect(frame, gest_result->_gesture_infos, res->get_perf_util());
			gest_rect_det->detect(request, res);
            GestureInfo* gi = gest_result->gestureInfos[0];
            if (!gi || !gi->hasGesture()) {
                continue;
            }
            g_index++;
            if (g_index % 10 == 0) {
                std::cout << "processed (gesture): " << g_index << std::endl;
            }
            // prepare small images
            for (auto& det : gest_detectors) {
                auto prepared = QuantCalibDataUtil::prepare_calib_data<GestureInfo>(det.second, rtConfig, request,
                                                                                    gest_result->gestureInfos);
                if (prepared.empty()) {
                    continue;
                }
                // write raw image
                std::string file_name = "val/" + std::to_string(g_index) + ".raw";
                save_raw(det.first + "/" + file_name, prepared.data, prepared.len());
                // update index file
                update_index(det.first + "/val_list.txt", file_name);
            }
        }
    }

    // push quantize image data to remote(for snpe)
    if (make_face_calib) {
        for (auto& det : face_detectors) {
            std::cout << "push " << det.first << " quantize image to remote " << remote_model_quant_data_path << std::endl;
            exec_scp(det.first, remote_model_quant_data_path);
        }
    }

    if (make_gesture_calib) {
        for (auto &det : gest_detectors) {
            std::cout << "push " << det.first << " quantize image to remote " << remote_model_quant_data_path
                      << std::endl;
            exec_scp(det.first, remote_model_quant_data_path);
        }
    }

    std::cout << "=== DONE! ===" << std::endl;
}
