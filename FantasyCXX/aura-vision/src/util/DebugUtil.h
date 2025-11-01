#pragma once

#include <cstdio>
#include <cstring>
#include <stdarg.h>
#include "opencv2/opencv.hpp"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/bean/GestureInfo.h"
#include "vision/core/common/VStructs.h"
#ifdef BUILD_NCNN
#include "mat.h"
#endif

// #define DEBUG_SUPPORT
#ifdef DEBUG_SUPPORT
#define DBG_PRINT(format, ...) vision::DebugUtil::print_info(format, ##__VA_ARGS__)
#define DBG_PRINT_MAT(mat, ...) vision::DebugUtil::print_mat(mat, ##__VA_ARGS__)
#define DBG_PRINT_ARRAY(data, len, ...) vision::DebugUtil::print_array(data, len, ##__VA_ARGS__)
#define DBG_PRINT_POINTS(data, tag) vision::DebugUtil::printPoints(data, tag)
#define DBG_PRINT_RECT(rect, tag) vision::DebugUtil::printRect(rect, tag)
#define DBG_PRINT_FACE_RECT(faceInfo) vision::DebugUtil::print_face_info(faceInfo, vision::DebugUtil::FACE_RECT)
#define DBG_PRINT_FACE_LMK(faceInfo) vision::DebugUtil::print_face_info(faceInfo, vision::DebugUtil::FACE_LMK)
#define DBG_PRINT_FACE_LMK_3D(faceInfo) vision::DebugUtil::print_face_info(faceInfo, vision::DebugUtil::FACE_LMK_3D)
#define DBG_PRINT_FACE_FEATURE(faceInfo) vision::DebugUtil::print_face_info(faceInfo, vision::DebugUtil::FACE_FEATURE)
#define DBG_PRINT_FACE_ATTRIBUTE(faceInfo) vision::DebugUtil::print_face_info(faceInfo, vision::DebugUtil::FACE_ATTRIBUTE)
#define DBG_PRINT_FACE_LIVE(faceInfo) vision::DebugUtil::print_face_info(faceInfo, vision::DebugUtil::FACE_LIVE)
#define DBG_PRINT_FACE_CALL(faceInfo) vision::DebugUtil::print_face_info(faceInfo, vision::DebugUtil::FACE_CALL)
#define DBG_PRINT_FACE_DRIVE(faceInfo) vision::DebugUtil::print_face_info(faceInfo, vision::DebugUtil::FACE_DRIVE)
#define DBG_PRINT_FACE_EMOTION(faceInfo) vision::DebugUtil::print_face_info(faceInfo, vision::DebugUtil::FACE_EMOTION)
#define DBG_PRINT_GEST_RECT(gestInfo) vision::DebugUtil::print_gest_info(gestInfo, vision::DebugUtil::GEST_RECT)
#define DBG_PRINT_GEST_LMK(gestInfo) vision::DebugUtil::print_gest_info(gestInfo, vision::DebugUtil::GEST_LMK)
#define DBG_IMG(name, mat) vision::DebugUtil::save_image_all(name, mat)
#define DBG_RAW(name, data) vision::DebugUtil::save_raw_image(name, data)
//#define DBG_READ_RAW(path, w, h, c, type) vision::DebugUtil::read_raw_image(path, w, h, c, type)
#define DBG_READ_RAW(path, data, size) vision::DebugUtil::readRaw(path, data, size)
#else
#define DBG_PRINT(format, ...)
#define DBG_PRINT_MAT(mat, ...)
#define DBG_PRINT_ARRAY(data, len, ...)
#define DBG_PRINT_POINTS(data, tag)
#define DBG_PRINT_RECT(rect, tag)
#define DBG_PRINT_FACE_RECT(faceInfo)
#define DBG_PRINT_FACE_LMK(faceInfo)
#define DBG_PRINT_FACE_LMK_3D(faceInfo)
#define DBG_PRINT_FACE_FEATURE(faceInfo)
#define DBG_PRINT_FACE_ATTRIBUTE(faceInfo)
#define DBG_PRINT_FACE_LIVE(faceInfo)
#define DBG_PRINT_FACE_CALL(faceInfo)
#define DBG_PRINT_FACE_DRIVE(faceInfo)
#define DBG_PRINT_FACE_EMOTION(faceInfo)
#define DBG_PRINT_GEST_RECT(gestInfo)
#define DBG_PRINT_GEST_LMK(gestInfo)
#define DBG_IMG(name, mat)
#define DBG_RAW(name, data)
// #define DBG_READ_RAW(path,w,h,c,type)
#define DBG_READ_RAW(path,data,size)
#endif



namespace aura::vision {
class DebugUtil {
public:
    enum DebugType {
        FACE_RECT,
        FACE_LMK,
        FACE_LMK_3D,
        FACE_FEATURE,
        FACE_ATTRIBUTE,
        FACE_LIVE,
        FACE_CALL,
        FACE_DRIVE,
        FACE_EMOTION,
        GEST_RECT,
        GEST_LMK
    };

    enum RawImgType {
        U8,
        F32,
        F64
    };

    enum RawImgFmt {
        CHW,
        HWC
    };

    enum ReadImgType {
        COLOR,
        GRAY
    };

    static void print_info(const char* format, ...);

    static void print_array(const float* data, int len, const std::string& tag="");
    static void print_array(const char* data, int len, const std::string& tag="");
    static void print_array(const unsigned char* data, int len, const std::string& tag="");
    static void print_array(const int* data, int len, const std::string& tag="");
    static void print_array(const double* data, int len, const std::string& tag="");

    static void print_points(const std::vector<VPoint>& points, const std::string& tag="");
    static void print_points(const VPoint* points, int len, const std::string& tag="");
    /**
     * 打印 float 关键点
     * @param points of vector
     * @param tag name tag
     */
    static void printPoints(const std::vector<float> &points, const std::string &tag = "");
    /**
     * 打印 3D 关键点
     * @param points  3D关键点
     * @param len     关键点长度
     * @param tag     信息标签
     */
    static void printPoints(const VPoint3 *points, int len, const std::string &tag = "");

    /**
     * 打印Rect框坐标的信息
     * @param rect
     * @param type
     */
    static void printRect(VRect &rect, const std::string &tag = "");

    static void save_image_all(const std::string& path, const cv::Mat& mat);
    static void save_raw_image(const std::string& path, const cv::Mat& mat, RawImgFmt dst_type = HWC);
    static void save_bmp_image(const std::string& path, const cv::Mat& mat);
    static void save_jpg_image(const std::string& path, const cv::Mat& mat);

    static cv::Mat read_raw_image(const std::string &path, int w, int h, int c, RawImgType type,
                                  RawImgFmt src_fmt = HWC, RawImgFmt dst_fmt = HWC);
    /**
     * 读取RAW数据。 需要注意原始
     * @param path raw数据路径
     * @return
     */
    static void readRaw(const std::string &path,void* data,int size);
    static cv::Mat read_image(const std::string& path, ReadImgType type = COLOR);

    static void print_face_info(FaceInfo* info, DebugType type);
    static void print_gest_info(GestureInfo* info, DebugType type);

    static cv::Mat compare_image(const cv::Mat& img1, const cv::Mat& img2);

#ifdef BUILD_NCNN
    static void print_mat(ncnn::Mat& mat, const std::string& tag = "");
    static void save_raw_image(const std::string& path, const ncnn::Mat& mat, RawImgFmt dst_type = CHW);
#endif

private:
    static int create_dir_if_none(const std::string& dir_path);
};
} // namespace ncnn
