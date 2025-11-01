#include "DebugUtil.h"
#include "vision/util/ImageUtil.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

#ifdef OPENCV4
#include <opencv2/imgcodecs/legacy/constants_c.h>
#endif

using namespace aura::vision;

void DebugUtil::print_info(const char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("[debug info] ");
    vprintf(format, args);
    va_end(args);
}

void DebugUtil::print_array(const int *data, int len, const std::string &tag) {
    if (!data || len <= 0) {
        return;
    }

    print_info("%s: ", tag.c_str());
    std::stringstream ss;
    for (int i = 0; i < len; ++i) {
        ss << data[i] << " ";
    }
    ss << "\n";
    print_info(ss.str().c_str());
}

void DebugUtil::print_array(const char *data, int len, const std::string &tag) {
    if (!data) {
        return;
    }
    print_info("%s: ", tag.c_str());
    std::stringstream ss;
    for (int i = 0; i < len; ++i) {
        ss << (int)(data[i]) << " ";
    }
    ss << "\n";
    print_info(ss.str().c_str());
}

void DebugUtil::print_array(const unsigned char *data, int len, const std::string &tag) {
    if (!data) {
        return;
    }

    print_info("%s: ", tag.c_str());
    std::stringstream ss;
    for (int i = 0; i < len; ++i) {
        ss << (int)(data[i]) << " ";
    }
    ss << "\n";
    print_info(ss.str().c_str());
}

void DebugUtil::print_array(const float *data, int len, const std::string &tag) {
    if (!data) {
        return;
    }

    print_info("%s: ", tag.c_str());
    std::stringstream ss;
    for (int i = 0; i < len; ++i) {
        ss << data[i] << " ";
    }
    ss << "\n";
    print_info(ss.str().c_str());
}

void DebugUtil::print_array(const double *data, int len, const std::string &tag) {
    if (!data) {
        return;
    }

    print_info("%s: ", tag.c_str());
    std::stringstream ss;
    for (int i = 0; i < len; ++i) {
        ss << data[i] << " ";
    }
    ss << "\n";
    print_info(ss.str().c_str());
}

void DebugUtil::print_points(const std::vector<VPoint> &points, const std::string &tag) {
    if (points.empty()) {
        return;
    }

    std::stringstream ss;
    ss << tag << ": ";
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        ss << "[" << points[i].x << " " << points[i].y << "] ";
    }
    ss << "\n";
    print_info(ss.str().c_str());
}

void DebugUtil::print_points(const VPoint *points, int len, const std::string &tag) {
    if (!points) {
        return;
    }

    std::stringstream ss;
    // ss << tag << ": ";
    // for (int i = 0; i < len; ++i) {
    //     // 竖行进行输出所有特征值. 部分情况下测试需要
    //     ss << points[i].x ;
    //     ss << "\n";
    //     ss << points[i].y << "";
    //     ss << "\n";
    // }
    ss << tag << ":[ ";
    for (int i = 0; i < len; ++i) {
        ss << points[i].x << " " << points[i].y << " ";
    }
    ss << "]\n";
    print_info(ss.str().c_str());
}

void DebugUtil::printPoints(const VPoint3 *points, int len, const std::string &tag) {
    if (!points) {
        return;
    }

    std::stringstream ss;
    ss << tag << ": ";
    for (int i = 0; i < len; ++i) {
        ss << "[" << points[i].x << " " << points[i].y << " " << points[i].z << "] ";
    }
    ss << "\n";
    print_info(ss.str().c_str());
}

void DebugUtil::printPoints(const std::vector<float> &points, const std::string &tag) {
    std::stringstream ss;
    ss << tag << ":[ ";
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        ss << points[i] << ",";
    }
    ss << "\n";
    print_info(ss.str().c_str());
}

void DebugUtil::printRect(VRect &rect, const std::string &tag) {
    std::stringstream ss;
    ss << tag << ":[";
    ss << rect.left << ",";
    ss << rect.top << ",";
    ss << rect.right << ",";
    ss << rect.bottom;
    ss << "]";
    ss << "\n";
    print_info(ss.str().c_str());
}

void DebugUtil::print_face_info(FaceInfo *info, DebugUtil::DebugType type) {
    if (!info) {
        print_info("face_info is nullptr\n");
        return;
    }
    switch (type) {
    case FACE_RECT:
        print_info("face rect confidence: %f \n", info->rectConfidence);
        print_points({info->rectLT, info->rectRB}, "face_rect");
        break;
    case FACE_LMK:
        print_info("face landmark confidence: %f\n", info->landmarkConfidence);
        print_points(info->landmark2D106, 30, "face_landmark");
        break;
    case FACE_LMK_3D:
        printPoints(info->landmark3D68, 20, "face_3d_landmark");
        break;
    case FACE_FEATURE:
        // 打印前20的face feature
        print_array(info->feature, 20, "face_feature");
        break;
    case FACE_LIVE:
        print_info("live_status: %d\n", int(info->stateNoInteractLivingSingle));
        break;
    case FACE_ATTRIBUTE:
        print_info("face_attributes:\nage=%d, gender=%d, race=%d, glasses=%d\n", int(info->stateAge),
                   int(info->stateGender), int(info->stateRace), int(info->stateGlass));
        break;
    case FACE_CALL:
        print_info("call_status: %d\n", int(info->stateCall));
        break;
    case FACE_DRIVE:
        print_info("drive status: %d\n", int(info->stateDangerDrive));
        break;
    case FACE_EMOTION:
        print_info("face emotion: %d\n", int(info->stateEmotion));
        break;
    default:
        break;
    }
}

void DebugUtil::print_gest_info(GestureInfo *info, DebugUtil::DebugType type) {
    if (!info) {
        print_info("face_info is nullptr\n");
        return;
    }
    switch (type) {
    case GEST_RECT:
        print_info("gesture rect confidence: %f\n", info->rectConfidence);
        print_points({info->rectLT, info->rectRB}, "gest rect");
        break;
    case GEST_LMK:
        print_info("gest landmark confidence: %f\n", info->landmarkConfidence);
        print_points(info->landmark21, 21, "gest_landmark");
        break;
    default:
        break;
    }
}

void DebugUtil::save_image_all(const std::string &path, const cv::Mat &mat) {
    save_raw_image(path, mat);
    save_bmp_image(path, mat);
}

void DebugUtil::save_bmp_image(const std::string &path, const cv::Mat &mat) {
    if (create_dir_if_none("debug_save") != 0) {
        return;
    }
    std::string n_path = "debug_save/" + path;
    if (n_path.substr(n_path.size() - 4) != ".bmp") {
        n_path += ".bmp";
    }
    int c = mat.channels();
    cv::Mat uint8_mat;
    if (mat.depth() != CV_8U) {
        mat.convertTo(uint8_mat, CV_8UC(c));
    } else {
        uint8_mat = mat;
    }
#ifdef WITH_OCV_HIGHGUI
    cv::imwrite(n_path, uint8_mat);
#endif
    print_info("write bmp image to %s\n", n_path.c_str());
}

void DebugUtil::save_jpg_image(const std::string& path, const cv::Mat &mat) {
    if (create_dir_if_none("debug_save") != 0) {
        return;
    }
    std::string n_path = "debug_save/" + path;
    if (n_path.substr(n_path.size() - 4) != ".jpg") {
        n_path += ".jpg";
    }
#ifdef WITH_OCV_HIGHGUI
    cv::imwrite(n_path, mat);
#endif
    print_info("write jpg image to %s\n", n_path.c_str());
}

void DebugUtil::save_raw_image(const std::string& path, const cv::Mat &mat, RawImgFmt type) {
    if (create_dir_if_none("debug_save") != 0) {
        return;
    }
    std::string n_path = "debug_save/" + path;
    int elemsize = mat.elemSize1();
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    int total = w * h * c;
    if (n_path.substr(n_path.size() - 4) != ".bin") {
        n_path += ".bin";
    }

    std::ofstream ofs(n_path, std::ios::binary);
    unsigned char* data = nullptr;
    // support float only!
    if (type == CHW && c > 1) {
        data = (unsigned char*)malloc(total * elemsize);
        //        if (elemsize > 1) {
        //            ImageUtil::image_hwc_to_chw<float>((float*)mat.data, (float*)data, w, h, c);
        //        } else {
        //            ImageUtil::image_hwc_to_chw<unsigned char>(mat.data, data, w, h, c);
        //        }
        ofs.write((char*)data, total * elemsize);
        free(data);
    } else {
        data = mat.data;
        ofs.write((char*)data, total * elemsize);
    }
    ofs.close();
    print_info("write binary image to %s\n", n_path.c_str());
}

cv::Mat DebugUtil::read_image(const std::string& path, ReadImgType type) {
    cv::Mat mat;
#ifdef WITH_OCV_HIGHGUI
    if (type == GRAY) {
        // mat = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    } else {
        // mat = cv::imread(path);
    }
#endif
    return mat;
}

cv::Mat DebugUtil::read_raw_image(const std::string &path, int w, int h, int c, RawImgType type,
                                  RawImgFmt src_fmt, RawImgFmt dst_fmt) {
    std::ifstream ifs(path, std::ios::binary);
    int elemSize = 1;
    static std::vector<int> mat_d_type {
        CV_8UC1,
        CV_8UC3,
        CV_32FC1,
        CV_32FC3
    };
    int dtype = 0;
    switch (type) {
        case U8:
            elemSize = 1;
            dtype = mat_d_type[c / 3];
            break;
        case F32:
            elemSize = 4;
            dtype = mat_d_type[2 + c / 3];
            break;
        case F64:
            elemSize = 8;
            dtype = mat_d_type[2 + c / 3];
            break;
        default:
            break;
    }
    int total_size = w * h * c * elemSize;

    unsigned char* data = nullptr;
    // double to float
    if (elemSize == 8) {
        unsigned char* d_data = (unsigned char*)malloc(total_size);
        ifs.read((char*)d_data, total_size);
        total_size = w * h * c * 4;
        float* f_data = (float*)malloc(total_size);
        for (int i = 0; i < w * h * c; ++i) {
            f_data[i] = static_cast<float>(((double*)d_data)[i]);
        }
        free(d_data);
        data = (unsigned char*)f_data;
    } else {
        data = (unsigned char*)malloc(total_size);
        ifs.read((char*)data, total_size);
    }

    // transpose if necessary
    unsigned char* mat_data = nullptr;
    cv::Mat mat(h, w, dtype);

    if (src_fmt == CHW && dst_fmt == HWC && c > 1) {
        if (elemSize == 1) {
            unsigned char *t_data = (unsigned char *) malloc(total_size);
            ImageUtil::image_chw_to_hwc<unsigned char>(data, t_data, w, h, c);
            mat_data = t_data;
        } else {
            float *t_data = (float *) malloc(total_size);
            ImageUtil::image_chw_to_hwc<float>((float*)data, t_data, w, h, c);
            mat_data = (unsigned char*)t_data;
        }
        free(data);
    } else if (src_fmt == HWC && dst_fmt == CHW && c > 1) {
        if (elemSize == 1) {
            unsigned char *t_data = (unsigned char *) malloc(total_size);
            ImageUtil::image_hwc_to_chw<unsigned char>(data, t_data, w, h, c);
            mat_data = t_data;
        } else {
            float *t_data = (float *) malloc(total_size);
            ImageUtil::image_hwc_to_chw<float>((float*)data, t_data, w, h, c);
            mat_data = (unsigned char*)t_data;
        }
        free(data);
    }
    else {
        mat_data = data;
    }

    memcpy(mat.data, mat_data, total_size);
    free(mat_data);
    return mat;
}

void DebugUtil::readRaw(const std::string &path, void *data,int len) {
    int size = 0;
    std::ifstream file(path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        if(size != len){
            print_info("originRaw size: %d, dstRaw size: %d",size,len);
        }
        file.seekg(0, file.beg);
        file.read((char*)data, size);
        file.close();
        print_info("read binary raw image: %s \n",path.c_str());
    } else {
        print_info("fail to read binary raw image: %s \n",path.c_str());
    }
}

int DebugUtil::create_dir_if_none(const std::string& dir_path) {
    if (0 != access(dir_path.c_str(), 0))
    {
        return mkdir(dir_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    return 0;
}

cv::Mat DebugUtil::compare_image(const cv::Mat& img1, const cv::Mat& img2) {
    print_info("img1 w=%d, h=%d, c=%d", img1.cols, img1.rows, img1.channels());
    print_info("img2 w=%d, h=%d, c=%d", img2.cols, img2.rows, img2.channels());
    cv::Mat diff_mat;
    if (img1.rows != img2.rows || img1.cols != img2.cols || img1.channels() != img2.channels()) {
        print_info("img size is different, cannot be compared!");
        return diff_mat;
    }

    diff_mat = img1 - img2;
#ifdef WITH_OCV_HIGHGUI
    cv::imshow("diff", diff_mat);
#endif
    return diff_mat;
}

#ifdef BUILD_NCNN

void DebugUtil::print_mat(ncnn::Mat& mat, const std::string& tag) {
    int w = mat.w;
    int h = mat.h;
    int c = mat.c;
    int cstep = mat.cstep;
    if (w * h * c == 0) {
        print_info("mat is empty!");
        return;
    }

    std::stringstream ss;
    ss << tag << " w=" << w << ", h=" << h << ", c=" << c << "\n";
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < w * h; ++j) {
            ss << mat[cstep * i + j] << " ";
        }
        ss << "\n";
    }
    print_info(ss.str().c_str());
}

void DebugUtil::save_raw_image(const std::string& path, const ncnn::Mat &mat, RawImgFmt type) {
    if (create_dir_if_none("debug_save") != 0) {
        return;
    }
    std::string n_path = "debug_save/" + path;
    int elemsize = mat.elemsize;
    int total = mat.total();
    int w = mat.w;
    int h = mat.h;
    int c = mat.c;
    if (n_path.substr(n_path.size() - 4) != ".bin") {
        n_path += ".bin";
    }

    std::ofstream ofs(n_path, std::ios::binary);
    unsigned char* data = nullptr;
    // support float only!
    if (type == HWC && c > 1) {
        data = (unsigned char*)malloc(total * elemsize);
        ImageUtil::image_chw_to_hwc<float>((float*)mat.data, (float*)data, w, h, c);
        ofs.write((char*)data, total * elemsize);
        free(data);
    } else {
        data = (unsigned char*)mat.data;
        ofs.write((char*)data, total * elemsize);
    }
    ofs.close();
    print_info("write binary image to %s\n", n_path.c_str());
}

#endif