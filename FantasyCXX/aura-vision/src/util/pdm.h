#ifdef BUILD_EXPERIMENTAL

#ifndef VISION_PDM_H
#define VISION_PDM_H

#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <string>

namespace aura::vision {

    /**
     * @brief A linear 3D Point Distribution Model (constructed using Non-Rigid structure from motion or PCA)
     * @brief Only describes the model but does not contain an instance of it (no local or global parameters are stored here)
     * @brief Contains the utility functions to help manipulate the model
     * */
    class Pdm {
    public:

        Pdm() { ; }

        /**
         * @brief A copy constructor
         * */
        Pdm(const Pdm &other);


        /**
         * @brief 加载配置文件
         * @param location 配置文件路径
         * */
        void read(std::string location);

        /**
         * @brief 加载内存数据
         * @param modelmem 配置文件的内存数据
         * */
        bool read(const char * modelmem, int mem_size);

        /**
         * @brief Number of vertices
         * */
        inline int number_of_points() const { return _mean_shape.rows / 3; }

        /**
         * @brief Listing the number of modes of variation
         * */
        inline int number_of_modes() const { return _princ_comp.cols; }

        /**
         * @brief Compute shape in object space (3D)
         * Compute the 3D representation of shape (in object space) using the local parameters
         * */
        void calc_shape_3d(cv::Mat_<float> &out_shape, const cv::Mat_<float> &params_local) const;

        /**
         * @brief Compute shape in image space (2D)
         * Get the 2D shape (in image space) from global and local parameters
         * */
        void calc_shape_2d(cv::Mat_<float> &out_shape, const cv::Mat_<float> &params_local,
                           const cv::Vec6f &params_global) const;

        /**
         * @brief provided the bounding box of a face and the local parameters (with optional rotation), generates the global parameters that can generate the face with the provided bounding box
         * This all assumes that the bounding box describes face from left outline to right outline of the face and chin to eyebrows
         * */
        void calc_params(cv::Vec6f &out_params_global, const cv::Rect_<float> &bounding_box,
                         const cv::Mat_<float> &params_local, const cv::Vec3f rotation = cv::Vec3f(0.0f));

        /**
         * @brief Provided the landmark location compute global and local parameters best fitting it (can provide optional rotation for potentially better results)
         * */
        void calc_params(cv::Vec6f &out_params_global, cv::Mat_<float> &out_params_local,
                         const cv::Mat_<float> &landmark_locations, const cv::Vec3f rotation = cv::Vec3f(0.0f));

        /**
         * @brief provided the model parameters, compute the bounding box of a face
         * The bounding box describes face from left outline to right outline of the face and chin to eyebrows
         * */
        void calc_bounding_box(cv::Rect_<float> &out_bounding_box, const cv::Vec6f &params_global,
                               const cv::Mat_<float> &params_local);

        /**
         * @brief Helpers for computing Jacobians, and Jacobians with the weight matrix
         * Calculate the PdmUtil's Jacobian over rigid parameters (rotation, translation and scaling), the additional input W represents trust for each of the landmarks and is part of Non-Uniform RLMS
         * */
        void compute_rigid_jacobian(const cv::Mat_<float> &params_local, const cv::Vec6f &params_global,
                                    cv::Mat_<float> &Jacob, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w);

        /**
         * @brief Calculate the PdmUtil's Jacobian over all parameters (rigid and non-rigid), the additional input W represents trust for each of the landmarks and is part of Non-Uniform RLMS
         * */
        void
        compute_jacobian(const cv::Mat_<float> &params_local, const cv::Vec6f &params_global, cv::Mat_<float> &Jacobian,
                         const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w);

        /**
         * @brief Given the current parameters, and the computed delta_p compute the updated parameters
         * */
        void
        update_model_parameters(const cv::Mat_<float> &delta_p, cv::Mat_<float> &params_local, cv::Vec6f &params_global);

        /**
         * @brief Skipping lines that start with # (together with empty lines)
         * */
        void skip_comments(std::ifstream &stream);

        /**
         * @brief 提取 landmark 围成的人脸框
         * */
        void
        extract_bounding_box(const cv::Mat_<float> &landmarks, float &min_x, float &max_x, float &min_y, float &max_y);

        void read_mat(std::ifstream &stream, cv::Mat &output_mat);

        void read_mat(const char * modelmem, cv::Mat &output_mat, int& offset);

    private:

        void string_skip(const char * data, char c, int& offset);

        /**
         * The 3D mean shape vector of the PDM [x1,..,xn,y1,...yn,z1,...,zn]
         * */
        cv::Mat_<float> _mean_shape;

        /**
         * Principal components or variation bases of the model,
         * */
        cv::Mat_<float> _princ_comp;

        /**
         * Eigenvalues (variances) corresponding to the bases
         * */
        cv::Mat_<float> _eigen_values;

        /**
         * 模型文件大小
         * */
         int _model_mem_size;
    };

}

#endif
#endif