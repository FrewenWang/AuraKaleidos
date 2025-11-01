
#ifdef BUILD_EXPERIMENTAL

#ifndef VISION_PDM_UTIL_H
#define VISION_PDM_UTIL_H

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include "pdm.h"
#include <dirent.h>

namespace aura::vision {

    class PdmUtil{
    public:

        /**
         * @brief extract 68 landmarks from 106 landmarks
         * @param dst: results of 68 points
         * @param src: input 106 points
         */
        static void point_temp_106_to_68(float *dst, float *src);

        /**
         * @brief extract 51 landmarks from 106 landmarks
         * @param dst: results of 51 points
         * @param src: input 106 points
         */
        static void point_temp_106_to_51(float *dst, float *src);

        /**
         * @brief extract 68 landmarks from 72 landmarks
         * @param dst: results of 68 points
         * @param src: input 72 points
         */
        static void point_temp_72_to_68(float *dst, float *src);

        /**
         * @brief extract 51 landmarks from 72 landmarks
         * @param dst: results of 51 points
         * @param src: input 72 points
         */
        static void point_temp_72_to_51(float *dst, float *src);

        /**
         * @brief extract 106 landmarks from 68 and 72 landmarks
         * @param dst: results of 106 points
         * @param src68: input 68 points
         * @param src72: input 72 points
         */
        static void point_temp_68and72_to_106(float *dst, float *src68, float *src72);

        /**
         * @brief draw coordinate according to face rotation
         * @param img: input image
         * @param rotation: rotation matrix of face
         * @param camera_matrix: camera intrincs
         */
        static void draw_face_coor(cv::Mat &img, cv::Matx33f &rotation);

        /**
         * @brief Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
         * */
        static cv::Matx33f euler2rotation_matrix(const cv::Vec3f &eulerAngles);

        /**
         * @brief Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
         * */
        static cv::Vec3f rotation_matrix2euler(const cv::Matx33f &rotation_matrix);

        static cv::Vec3f euler2axis_angle(const cv::Vec3f &euler);

        static cv::Vec3f axis_angle2euler(const cv::Vec3f &axis_angle);

        static cv::Matx33f axis_angle2rotation_matrix(const cv::Vec3f &axis_angle);

        static cv::Vec3f rotation_matrix2axis_angle(const cv::Matx33f &rotation_matrix);

        /**
         * @brief Generally useful 3D functions
         * */
        static void project(cv::Mat_<float> &dest, const cv::Mat_<float> &mesh, float fx, float fy, float cx, float cy);

        /**
         * @brief Point set and landmark manipulation functions
         * Using Kabsch's algorithm for aligning shapes
         * This assumes that align_from and align_to are already mean normalised
         * */
        static cv::Matx22f align_shapes_kabsch2d(const cv::Mat_<float> &align_from, const cv::Mat_<float> &align_to);

        /**
         * @brief Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
         * */
        static cv::Matx22f align_shapes_with_scale(cv::Mat_<float> &src, cv::Mat_<float> dst);
    };
}

#endif /* utils_h */

#endif