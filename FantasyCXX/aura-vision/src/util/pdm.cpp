#ifdef BUILD_EXPERIMENTAL
#include "pdm.h"
#include <fstream>
#include "pdm_util.h"
#include "opencv2/opencv.hpp"
#include <cmath>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aura::vision {

    /**
     * @brief Orthonormalising the 3x3 rotation matrix
     * */
    void Orthonormalise(cv::Matx33f &R) {

        cv::SVD svd(R, cv::SVD::MODIFY_A);

        // get the orthogonal matrix from the initial rotation matrix
        cv::Mat_<float> X = svd.u * svd.vt;

        // This makes sure that the handedness is preserved and no reflection happened
        // by making sure the determinant is 1 and not -1
        cv::Mat_<float> W = cv::Mat_<float>::eye(3, 3);
        W(2, 2) = determinant(X);
        cv::Mat Rt = svd.u * W * svd.vt;

        Rt.copyTo(R);
    }

    Pdm::Pdm(const Pdm &other) {
        // Make sure the matrices are allocated properly
        this->_mean_shape = other._mean_shape.clone();
        this->_princ_comp = other._princ_comp.clone();
        this->_eigen_values = other._eigen_values.clone();
    }

    void Pdm::calc_shape_3d(cv::Mat_<float> &out_shape, const cv::Mat_<float> &p_local) const {
        out_shape.create(_mean_shape.rows, _mean_shape.cols);
        out_shape = _mean_shape + _princ_comp * p_local;
    }

    void Pdm::calc_shape_2d(cv::Mat_<float> &out_shape, const cv::Mat_<float> &params_local,
                            const cv::Vec6f &params_global) const {
        int n = this->number_of_points();

        float s = params_global[0]; // scaling factor
        float tx = params_global[4]; // x offset
        float ty = params_global[5]; // y offset
        // get the rotation matrix from the euler angles
        cv::Vec3f euler(params_global[1], params_global[2], params_global[3]);
        cv::Matx33f currRot = PdmUtil::euler2rotation_matrix(euler);

        // get the 3D shape of the object
        cv::Mat_<float> Shape_3D = _mean_shape + _princ_comp * params_local;
        // create the 2D shape matrix (if it has not been defined yet)
        if ((out_shape.rows != _mean_shape.rows) || (out_shape.cols != 1)) {
            out_shape.create(2 * n, 1);
        }

        // for every vertex
        for (int i = 0; i < n; i++) {
            // Transform this using the weak-perspective mapping to 2D from 3D
            out_shape.at<float>(i, 0) = s * (currRot(0, 0) * Shape_3D.at<float>(i, 0) +
                                             currRot(0, 1) * Shape_3D.at<float>(i + n, 0) +
                                             currRot(0, 2) * Shape_3D.at<float>(i + n * 2, 0)) + tx;
            out_shape.at<float>(i + n, 0) = s * (currRot(1, 0) * Shape_3D.at<float>(i, 0) +
                                                 currRot(1, 1) * Shape_3D.at<float>(i + n, 0) +
                                                 currRot(1, 2) * Shape_3D.at<float>(i + n * 2, 0)) + ty;
        }
    }

    void Pdm::calc_params(cv::Vec6f &out_params_global, const cv::Rect_<float> &bounding_box,
                          const cv::Mat_<float> &params_local, const cv::Vec3f rotation) {

        // get the shape instance based on local params
        cv::Mat_<float> current_shape(_mean_shape.size());

        calc_shape_3d(current_shape, params_local);

        // rotate the shape
        cv::Matx33f rotation_matrix = PdmUtil::euler2rotation_matrix(rotation);

        cv::Mat_<float> reshaped = current_shape.reshape(1, 3);

        cv::Mat rotated_shape = (cv::Mat(rotation_matrix) * reshaped);

        // Get the width of expected shape
        double min_x;
        double max_x;
        cv::minMaxLoc(rotated_shape.row(0), &min_x, &max_x);

        double min_y;
        double max_y;
        cv::minMaxLoc(rotated_shape.row(1), &min_y, &max_y);

        float width = (float) abs(min_x - max_x);
        float height = (float) abs(min_y - max_y);

        float scaling = ((bounding_box.width / width) + (bounding_box.height / height)) / 2.0f;

        // The estimate of face center also needs some correction
        float tx = bounding_box.x + bounding_box.width / 2;
        float ty = bounding_box.y + bounding_box.height / 2;

        // Correct it so that the bounding box is just around the minimum and maximum point in the initialised face
        tx = tx - scaling * (min_x + max_x) / 2.0f;
        ty = ty - scaling * (min_y + max_y) / 2.0f;

        out_params_global = cv::Vec6f(scaling, rotation[0], rotation[1], rotation[2], tx, ty);
    }

    void Pdm::skip_comments(std::ifstream &stream) {
        while (stream.peek() == '#' || stream.peek() == '\n' || stream.peek() == ' ' || stream.peek() == '\r') {
            std::string skipped;
            std::getline(stream, skipped);
        }
    }

    void
    Pdm::extract_bounding_box(const cv::Mat_<float> &landmarks, float &min_x, float &max_x, float &min_y,
                              float &max_y) {
        if (landmarks.cols == 1) {
            int n = landmarks.rows / 2;
            cv::MatConstIterator_<float> landmarks_it = landmarks.begin();

            for (int i = 0; i < n; ++i) {
                float val = *landmarks_it++;

                if (i == 0 || val < min_x)
                    min_x = val;

                if (i == 0 || val > max_x)
                    max_x = val;
            }

            for (int i = 0; i < n; ++i) {
                float val = *landmarks_it++;

                if (i == 0 || val < min_y)
                    min_y = val;

                if (i == 0 || val > max_y)
                    max_y = val;
            }
        } else {
            int n = landmarks.rows;
            for (int i = 0; i < n; ++i) {
                float val_x = landmarks.at<float>(i, 0);
                float val_y = landmarks.at<float>(i, 1);

                if (i == 0 || val_x < min_x)
                    min_x = val_x;

                if (i == 0 || val_x > max_x)
                    max_x = val_x;

                if (i == 0 || val_y < min_y)
                    min_y = val_y;

                if (i == 0 || val_y > max_y)
                    max_y = val_y;
            }
        }
    }

    void Pdm::calc_bounding_box(cv::Rect_<float> &out_bounding_box, const cv::Vec6f &params_global,
                                const cv::Mat_<float> &params_local) {
        // get the shape instance based on local params
        cv::Mat_<float> current_shape;
        // 第一次调用传入参数都为0,current_shape在函数内被赋值，与3D模型一致
        calc_shape_2d(current_shape, params_local, params_global);
        // Get the width of expected shape
        float min_x, max_x, min_y, max_y;
        extract_bounding_box(current_shape, min_x, max_x, min_y, max_y);
        float width = abs(min_x - max_x);
        float height = abs(min_y - max_y);
        //返回矩形框
        out_bounding_box = cv::Rect_<float>(min_x, min_y, width, height);
    }

    void
    Pdm::compute_rigid_jacobian(const cv::Mat_<float> &p_local, const cv::Vec6f &params_global,
                                cv::Mat_<float> &Jacob,
                                const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w) {

        // number of verts
        int n = this->number_of_points();

        Jacob.create(n * 2, 6);

        float X, Y, Z;

        float s = params_global[0];

        cv::Mat_<float> shape_3D;
        this->calc_shape_3d(shape_3D, p_local);

        // Get the rotation matrix
        cv::Vec3f euler(params_global[1], params_global[2], params_global[3]);
        cv::Matx33f currRot = PdmUtil::euler2rotation_matrix(euler);

        float r11 = currRot(0, 0);
        float r12 = currRot(0, 1);
        float r13 = currRot(0, 2);
        float r21 = currRot(1, 0);
        float r22 = currRot(1, 1);
        float r23 = currRot(1, 2);
        float r31 = currRot(2, 0);
        float r32 = currRot(2, 1);
        float r33 = currRot(2, 2);

        cv::MatIterator_<float> Jx = Jacob.begin();
        cv::MatIterator_<float> Jy = Jx + n * 6;

        for (int i = 0; i < n; i++) {

            X = shape_3D.at<float>(i, 0);
            Y = shape_3D.at<float>(i + n, 0);
            Z = shape_3D.at<float>(i + n * 2, 0);

            // The rigid jacobian from the axis angle rotation matrix approximation using small angle assumption (R * R')
            // where R' = [1, -wz, wy
            //             wz, 1, -wx
            //             -wy, wx, 1]
            // And this is derived using the small angle assumption on the axis angle rotation matrix parametrisation

            // scaling term
            *Jx++ = (X * r11 + Y * r12 + Z * r13);
            *Jy++ = (X * r21 + Y * r22 + Z * r23);

            // rotation terms
            *Jx++ = (s * (Y * r13 - Z * r12));
            *Jy++ = (s * (Y * r23 - Z * r22));
            *Jx++ = (-s * (X * r13 - Z * r11));
            *Jy++ = (-s * (X * r23 - Z * r21));
            *Jx++ = (s * (X * r12 - Y * r11));
            *Jy++ = (s * (X * r22 - Y * r21));

            // translation terms
            *Jx++ = 1.0f;
            *Jy++ = 0.0f;
            *Jx++ = 0.0f;
            *Jy++ = 1.0f;

        }

        cv::Mat Jacob_w = cv::Mat::zeros(Jacob.rows, Jacob.cols, Jacob.type());

        Jx = Jacob.begin();
        Jy = Jx + n * 6;

        cv::MatIterator_<float> Jx_w = Jacob_w.begin<float>();
        cv::MatIterator_<float> Jy_w = Jx_w + n * 6;

        // Iterate over all Jacobian values and multiply them by the weight in diagonal of W
        for (int i = 0; i < n; i++) {
            float w_x = W.at<float>(i, i);
            float w_y = W.at<float>(i + n, i + n);

            for (int j = 0; j < Jacob.cols; ++j) {
                *Jx_w++ = *Jx++ * w_x;
                *Jy_w++ = *Jy++ * w_y;
            }
        }

        Jacob_t_w = Jacob_w.t();
    }

    void
    Pdm::compute_jacobian(const cv::Mat_<float> &params_local, const cv::Vec6f &params_global,
                          cv::Mat_<float> &Jacobian,
                          const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w) {

        // number of vertices
        int n = this->number_of_points();

        // number of non-rigid parameters
        int m = this->number_of_modes();

        Jacobian.create(n * 2, 6 + m);

        float X, Y, Z;

        float s = params_global[0];

        cv::Mat_<float> shape_3D;
        this->calc_shape_3d(shape_3D, params_local);

        cv::Vec3f euler(params_global[1], params_global[2], params_global[3]);
        cv::Matx33f currRot = PdmUtil::euler2rotation_matrix(euler);

        float r11 = currRot(0, 0);
        float r12 = currRot(0, 1);
        float r13 = currRot(0, 2);
        float r21 = currRot(1, 0);
        float r22 = currRot(1, 1);
        float r23 = currRot(1, 2);
        float r31 = currRot(2, 0);
        float r32 = currRot(2, 1);
        float r33 = currRot(2, 2);

        cv::MatIterator_<float> Jx = Jacobian.begin();
        cv::MatIterator_<float> Jy = Jx + n * (6 + m);
        cv::MatConstIterator_<float> Vx = this->_princ_comp.begin();
        cv::MatConstIterator_<float> Vy = Vx + n * m;
        cv::MatConstIterator_<float> Vz = Vy + n * m;

        for (int i = 0; i < n; i++) {

            X = shape_3D.at<float>(i, 0);
            Y = shape_3D.at<float>(i + n, 0);
            Z = shape_3D.at<float>(i + n * 2, 0);

            // The rigid jacobian from the axis angle rotation matrix approximation using small angle assumption (R * R')
            // where R' = [1, -wz, wy
            //             wz, 1, -wx
            //             -wy, wx, 1]
            // And this is derived using the small angle assumption on the axis angle rotation matrix parametrisation

            // scaling term
            *Jx++ = (X * r11 + Y * r12 + Z * r13);
            *Jy++ = (X * r21 + Y * r22 + Z * r23);

            // rotation terms
            *Jx++ = (s * (Y * r13 - Z * r12));
            *Jy++ = (s * (Y * r23 - Z * r22));
            *Jx++ = (-s * (X * r13 - Z * r11));
            *Jy++ = (-s * (X * r23 - Z * r21));
            *Jx++ = (s * (X * r12 - Y * r11));
            *Jy++ = (s * (X * r22 - Y * r21));

            // translation terms
            *Jx++ = 1.0f;
            *Jy++ = 0.0f;
            *Jx++ = 0.0f;
            *Jy++ = 1.0f;

            for (int j = 0; j < m; j++, ++Vx, ++Vy, ++Vz) {
                // How much the change of the non-rigid parameters (when object is rotated) affect 2D motion
                *Jx++ = (s * (r11 * (*Vx) + r12 * (*Vy) + r13 * (*Vz)));
                *Jy++ = (s * (r21 * (*Vx) + r22 * (*Vy) + r23 * (*Vz)));
            }
        }

        // Adding the weights here
        cv::Mat Jacob_w = Jacobian.clone();

        if (cv::trace(W)[0] != W.rows) {
            Jx = Jacobian.begin();
            Jy = Jx + n * (6 + m);

            cv::MatIterator_<float> Jx_w = Jacob_w.begin<float>();
            cv::MatIterator_<float> Jy_w = Jx_w + n * (6 + m);

            // Iterate over all Jacobian values and multiply them by the weight in diagonal of W
            for (int i = 0; i < n; i++) {
                float w_x = W.at<float>(i, i);
                float w_y = W.at<float>(i + n, i + n);

                for (int j = 0; j < Jacobian.cols; ++j) {
                    *Jx_w++ = *Jx++ * w_x;
                    *Jy_w++ = *Jy++ * w_y;
                }
            }
        }
        Jacob_t_w = Jacob_w.t();

    }

    void Pdm::update_model_parameters(const cv::Mat_<float> &delta_p, cv::Mat_<float> &params_local,
                                      cv::Vec6f &params_global) {

        // The scaling and translation parameters can be just added
        params_global[0] += delta_p.at<float>(0, 0);
        params_global[4] += delta_p.at<float>(4, 0);
        params_global[5] += delta_p.at<float>(5, 0);

        // get the original rotation matrix
        cv::Vec3f eulerGlobal(params_global[1], params_global[2], params_global[3]);
        cv::Matx33f R1 = PdmUtil::euler2rotation_matrix(eulerGlobal);

        // construct R' = [1, -wz, wy
        //               wz, 1, -wx
        //               -wy, wx, 1]
        cv::Matx33f R2 = cv::Matx33f::eye();

        R2(1, 2) = -1.0 * (R2(2, 1) = delta_p.at<float>(1, 0));
        R2(2, 0) = -1.0 * (R2(0, 2) = delta_p.at<float>(2, 0));
        R2(0, 1) = -1.0 * (R2(1, 0) = delta_p.at<float>(3, 0));

        // Make sure it's orthonormal
        Orthonormalise(R2);

        // Combine rotations
        cv::Matx33f R3 = R1 * R2;

        // Extract euler angle (through axis angle first to make sure it's legal)
        cv::Vec3f axis_angle = PdmUtil::rotation_matrix2axis_angle(R3);
        cv::Vec3f euler = PdmUtil::axis_angle2euler(axis_angle);

        params_global[1] = euler[0];
        params_global[2] = euler[1];
        params_global[3] = euler[2];

        // Local parameter update, just simple addition
        if (delta_p.rows > 6) {
            params_local = params_local + delta_p(cv::Rect(0, 6, 1, this->number_of_modes()));
        }

    }

    void Pdm::calc_params(cv::Vec6f &out_params_global, cv::Mat_<float> &out_params_local,
                          const cv::Mat_<float> &landmark_locations, const cv::Vec3f rotation) {
        int m = this->number_of_modes();
        int n = this->number_of_points();
        //初始化为1
        cv::Mat_<int> visi_ind_2D(n * 2, 1, 1);
        cv::Mat_<int> visi_ind_3D(3 * n, 1, 1);

        int visi_count = n;

        for (int i = 0; i < n; ++i) {
            // If the landmark is invisible indicate this
            if (landmark_locations.at<float>(i) == 0) {
                visi_ind_2D.at<int>(i) = 0;
                visi_ind_2D.at<int>(i + n) = 0;
                visi_ind_3D.at<int>(i) = 0;
                visi_ind_3D.at<int>(i + n) = 0;
                visi_ind_3D.at<int>(i + 2 * n) = 0;

                visi_count--;
            }
        }

        // As this might be subsampled have special versions
        cv::Mat_<float> M(visi_count * 3, _mean_shape.cols, 0.0); //_mean_shape.cols:1
        cv::Mat_<float> V(visi_count * 3, _princ_comp.cols, 0.0); //_princ_comp.cols:30
        visi_count = 0;
        for (int i = 0; i < n * 3; ++i) {
            if (visi_ind_3D.at<int>(i) == 1) {
//            M_t.push_back(this->_mean_shape.at<double>(i));
//            V_t.push_back(this->_princ_comp.at<double>(i));
                this->_mean_shape.row(i).copyTo(M.row(visi_count));
                this->_princ_comp.row(i).copyTo(V.row(visi_count));
                visi_count++;
//			cv::vconcat(M, this->_mean_shape.row(i), M);//leisheng526
//			cv::vconcat(V, this->_princ_comp.row(i), V);//leisheng526
            }
        }

        cv::Mat_<float> m_old = this->_mean_shape.clone();
        cv::Mat_<float> v_old = this->_princ_comp.clone();

//    cv::Mat_<double> M(M_t.size(), _mean_shape.cols, (double*)M_t.data());//leisheng526
//    cv::Mat_<double> V(V_t.size(), _princ_comp.cols, (double*)V_t.data());//leisheng526

        this->_mean_shape = M;
        this->_princ_comp = V;

        // The new number of points
        n = M.rows / 3;

        // Extract the relevant landmark locations, visible landmark locations
        cv::Mat_<float> landmark_locs_vis(n * 2, 1, 0.0f);
        int k = 0;
        for (int i = 0; i < visi_ind_2D.rows; ++i) {
            if (visi_ind_2D.at<int>(i) == 1) {
                landmark_locs_vis.at<float>(k) = landmark_locations.at<float>(i);
                k++;
            }
        }

        // Compute the initial global parameters
        float min_x, max_x, min_y, max_y;
        extract_bounding_box(landmark_locs_vis, min_x, max_x, min_y, max_y);

        float width = abs(min_x - max_x);
        float height = abs(min_y - max_y);

        cv::Rect_<float> model_bbox;
        //这次运行params_local, params_global都为0,model_box应该是3D模型的人脸框，里面的旋转矩阵也是单位对角阵
        calc_bounding_box(model_bbox, cv::Vec6f(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                          cv::Mat_<double>(this->number_of_modes(), 1, 0.0));
        cv::Rect_<float> bbox(min_x, min_y, width, height);

        float scaling = ((width / model_bbox.width) + (height / model_bbox.height)) / 2.0f;

        cv::Vec3f rotation_init = rotation;
        //初始化对角单位矩阵
        cv::Matx33f R = PdmUtil::euler2rotation_matrix(rotation_init);
        //translation为什么是这样？
        cv::Vec2f translation((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f);

        cv::Mat_<float> loc_params(this->number_of_modes(), 1, 0.0);
        cv::Vec6f glob_params(scaling, rotation_init[0], rotation_init[1], rotation_init[2], translation[0],
                              translation[1]);

        // get the 3D shape of the object,M:_mean_shape visble,V:vector,what's the multi result?
        cv::Mat_<float> shape_3D = M + V * loc_params;
        cv::Mat_<float> curr_shape(2 * n, 1);

        // for every vertex
        for (int i = 0; i < n; i++) {
            // Transform this using the weak-perspective mapping to 2D from 3D
            curr_shape.at<float>(i, 0) = scaling *
                                         (R(0, 0) * shape_3D.at<float>(i, 0) + R(0, 1) * shape_3D.at<float>(i + n, 0) +
                                          R(0, 2) * shape_3D.at<float>(i + n * 2, 0)) + translation[0];
            curr_shape.at<float>(i + n, 0) = scaling * (R(1, 0) * shape_3D.at<float>(i, 0) +
                                                        R(1, 1) * shape_3D.at<float>(i + n, 0) +
                                                        R(1, 2) * shape_3D.at<float>(i + n * 2, 0)) + translation[1];
        }
        // perspective error
        float currError = cv::norm(curr_shape - landmark_locs_vis);

        cv::Mat_<float> regularisations = cv::Mat_<float>::zeros(1, 6 + m);

        float reg_factor = 1;

        // Setting the regularisation to the inverse of eigenvalues
        cv::Mat(reg_factor / this->_eigen_values).copyTo(regularisations(cv::Rect(6, 0, m, 1)));
        regularisations = cv::Mat::diag(regularisations.t());

        cv::Mat_<float> WeightMatrix = cv::Mat_<float>::eye(n * 2, n * 2);

        int not_improved_in = 0;

        for (size_t i = 0; i < 1000; ++i) {
            // get the 3D shape of the object
            shape_3D = M + V * loc_params;
            shape_3D = shape_3D.reshape(1, 3);

            cv::Matx23f R_2D(R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2));

            cv::Mat_<float> curr_shape_2D = scaling * shape_3D.t() * cv::Mat(R_2D).t();
            curr_shape_2D.col(0) = curr_shape_2D.col(0) + translation(0);
            curr_shape_2D.col(1) = curr_shape_2D.col(1) + translation(1);
            curr_shape_2D = cv::Mat(curr_shape_2D.t()).reshape(1, n * 2);

            cv::Mat_<float> error_resid;
            cv::Mat(landmark_locs_vis - curr_shape_2D).convertTo(error_resid, CV_32F);
            cv::Mat_<float> J, J_w_t;
            this->compute_jacobian(loc_params, glob_params, J, WeightMatrix, J_w_t);
            // projection of the meanshifts onto the jacobians (using the weighted Jacobian, see Baltrusaitis 2013)
            cv::Mat_<float> J_w_t_m = J_w_t * error_resid;

            // Add the regularisation term
            J_w_t_m(cv::Rect(0, 6, 1, m)) =
                    J_w_t_m(cv::Rect(0, 6, 1, m)) - regularisations(cv::Rect(6, 6, m, m)) * loc_params;
            cv::Mat_<float> Hessian = regularisations.clone(); //106 35*35
            // Perform matrix multiplication in OpenBLAS (fortran call)
            //float alpha1 = 1.0;
            //float beta1 = 1.0;
            //sgemm_("N", "N", &J.cols, &J_w_t.rows, &J_w_t.cols, &alpha1, (float*)J.data, &J.cols, (float*)J_w_t.data, &J_w_t.cols, &beta1, (float*)Hessian.data, &J.cols);
//openblas leisheng526
            // Above is a fast (but ugly) version of
            Hessian = J_w_t * J + regularisations;

            // Solve for the parameter update (from Baltrusaitis 2013 based on eq (36) Saragih 2011)
            cv::Mat_<float> param_update;
#ifdef OPENCV2
            cv::solve(Hessian, J_w_t_m, param_update, CV_CHOLESKY);
#endif
            // To not overshoot, have the gradient decent rate a bit smaller
            //param_update = 0.5 * param_update;
            param_update = 0.75 * param_update;//new value leisheng526
            update_model_parameters(param_update, loc_params, glob_params);

            scaling = glob_params[0];
            rotation_init[0] = glob_params[1];
            rotation_init[1] = glob_params[2];
            rotation_init[2] = glob_params[3];

            translation[0] = glob_params[4];
            translation[1] = glob_params[5];

            R = PdmUtil::euler2rotation_matrix(rotation_init);

            R_2D(0, 0) = R(0, 0);
            R_2D(0, 1) = R(0, 1);
            R_2D(0, 2) = R(0, 2);
            R_2D(1, 0) = R(1, 0);
            R_2D(1, 1) = R(1, 1);
            R_2D(1, 2) = R(1, 2);

            curr_shape_2D = scaling * shape_3D.t() * cv::Mat(R_2D).t();
            curr_shape_2D.col(0) = curr_shape_2D.col(0) + translation(0);
            curr_shape_2D.col(1) = curr_shape_2D.col(1) + translation(1);

            curr_shape_2D = cv::Mat(curr_shape_2D.t()).reshape(1, n * 2);

            float error = cv::norm(curr_shape_2D - landmark_locs_vis);

            if (0.999 * currError < error) {
                not_improved_in++;
                //if (not_improved_in == 5)
                if (not_improved_in == 3)//new value leisheng526
                {
                    break;
                }
            }

            currError = error;

        }

        out_params_global = glob_params;
        out_params_local = loc_params;

        this->_mean_shape = m_old;
        this->_princ_comp = v_old;


    }

    void Pdm::read_mat(std::ifstream &stream, cv::Mat &output_mat) {
        // Read in the number of rows, columns and the data type
        int row, col, type;

        stream >> row >> col >> type;

        output_mat = cv::Mat(row, col, type);

        switch (output_mat.type()) {
            case CV_64FC1: {
                cv::MatIterator_<double> begin_it = output_mat.begin<double>();
                cv::MatIterator_<double> end_it = output_mat.end<double>();

                while (begin_it != end_it) {
                    stream >> *begin_it++;
                }
            }
                break;
            case CV_32FC1: {
                cv::MatIterator_<float> begin_it = output_mat.begin<float>();
                cv::MatIterator_<float> end_it = output_mat.end<float>();

                while (begin_it != end_it) {
                    stream >> *begin_it++;
                }
            }
                break;
            case CV_32SC1: {
                cv::MatIterator_<int> begin_it = output_mat.begin<int>();
                cv::MatIterator_<int> end_it = output_mat.end<int>();
                while (begin_it != end_it) {
                    stream >> *begin_it++;
                }
            }
                break;
            case CV_8UC1: {
                cv::MatIterator_<uchar> begin_it = output_mat.begin<uchar>();
                cv::MatIterator_<uchar> end_it = output_mat.end<uchar>();
                while (begin_it != end_it) {
                    stream >> *begin_it++;
                }
            }
                break;
            default:
                printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", __FILE__, __LINE__, output_mat.type());
                abort();


        }
    }

    void Pdm::read(std::string location) {

//    ifstream patchesFile(location.c_str(), ios_base::in);
        std::ifstream pdmLoc(location, std::ios_base::in);
        if (!pdmLoc) {
            std::cout << "Wrong model path.\n";
            exit(1);
        }
        skip_comments(pdmLoc);

        // Reading mean values
        cv::Mat_<double> mean_shape_d;
        read_mat(pdmLoc, mean_shape_d);
        mean_shape_d.convertTo(_mean_shape, CV_32F); // Moving things to floats for speed

        skip_comments(pdmLoc);

        // Reading principal components
        cv::Mat_<double> princ_comp_d;
        read_mat(pdmLoc, princ_comp_d);
        princ_comp_d.convertTo(_princ_comp, CV_32F);

        skip_comments(pdmLoc);

        // Reading eigenvalues
        cv::Mat_<double> eigen_values_d;
        read_mat(pdmLoc, eigen_values_d);
        eigen_values_d.convertTo(_eigen_values, CV_32F);
    }


    bool Pdm::read(const char *modelmem, int mem_size) {
        _model_mem_size = mem_size;

        int offset = 0;

        try {
            cv::Mat mean_shape_d;
            read_mat(modelmem, mean_shape_d, offset);
            mean_shape_d.convertTo(_mean_shape, CV_32F);

            cv::Mat princ_comp_d;
            read_mat(modelmem, princ_comp_d, offset);
            princ_comp_d.convertTo(_princ_comp, CV_32F);

            cv::Mat eigen_values_d;
            read_mat(modelmem, eigen_values_d, offset);
            eigen_values_d.convertTo(_eigen_values, CV_32F);
        }
        catch (...) {
            return false;
        }

        return true;
    }

    void Pdm::string_skip(const char *data, char c, int &offset) {
        for (int i = 0; i < (_model_mem_size - offset); ++i) {
            if (*(data + i) == c) {
                offset += (i + 1);
                return;
            }
        }
        throw "error";
    }

    void Pdm::read_mat(const char *modelmem, cv::Mat &output_mat, int &offset) {
        int rows = 0;
        int cols = 0;
        int type = 0;

        string_skip(modelmem + offset, '#', offset);
        string_skip(modelmem + offset, '\n', offset);

        sscanf((modelmem + offset), "%d", &rows);
        string_skip(modelmem + offset, '\n', offset);

        sscanf((modelmem + offset), "%d", &cols);
        string_skip(modelmem + offset, '\n', offset);

        sscanf((modelmem + offset), "%d", &type);
        string_skip(modelmem + offset, '\n', offset);

        output_mat.create(cv::Size(cols, rows), type);
        switch (output_mat.type()) {
            case CV_64FC1: {
                double *data = (double *) output_mat.data;
                double value = 0.0;
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        sscanf((modelmem + offset), "%lf", &value);
                        *data++ = value;
                        string_skip(modelmem + offset, ' ', offset);
                    }
                    string_skip(modelmem + offset, '\n', offset);
                }
            }
                break;
            case CV_32FC1: {
                float *data = (float *) output_mat.data;
                float value = 0.0;
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        sscanf((modelmem + offset), "%f", &value);
                        *data++ = value;
                        string_skip(modelmem + offset, ' ', offset);
                    }
                    string_skip(modelmem + offset, '\n', offset);
                }
            }
                break;
            case CV_32SC1: {
                int *data = (int *) output_mat.data;
                int value = 0;
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        sscanf((modelmem + offset), "%d", &value);
                        *data++ = value;
                        string_skip(modelmem + offset, ' ', offset);
                    }
                    string_skip(modelmem + offset, '\n', offset);
                }
            }
                break;
            case CV_8UC1: {
                uchar *data = (uchar *) output_mat.data;
                uchar value = '\0';
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        sscanf((modelmem + offset), "%c", &value);
                        *data++ = value;
                        string_skip(modelmem + offset, ' ', offset);
                    }
                    string_skip(modelmem + offset, '\n', offset);
                }
            }
                break;
            default:
                throw "error";
        }
    }
}
#endif