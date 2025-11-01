
#pragma once

#include "opencv2/opencv.hpp"

#include "AbsFaceDetector.h"
#include "inference/CustomizePredictor.h"
#include "util/mat_math.h"
#include "util/pdm.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

    /**
     * @brief 人脸2D关键点转换3D点功能
     * 算法原理：
     * */
    class Face2Dto3DDetector : public AbsFaceDetector {
    public:
        Face2Dto3DDetector();
        ~Face2Dto3DDetector() override;

        int init(RtConfig* cfg) override;
        int init_params();
        bool _pdm_model_init_state;
//        int doDetect(VFrameInfo& frame, FaceInfo** infos, PerfUtil* perf) override;
        int doDetect(VisionRequest *request, VisionResult *result) override;
    protected:
        bool init_pdm(const char *param_mem, int mem_size);
//        int prepare(VFrameInfo& frame, FaceInfo** infos, TensorArray& prepared) override;
        int prepare(VisionRequest *request, FaceInfo** infos, TensorArray& prepared) override;
        int process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) override;
        int post(VisionRequest *request, TensorArray& infer_results, FaceInfo** infos) override;
        /** 模型推算参数的集数 */
        void calcparams_calculate(FaceInfo *faceinfo);

        std::shared_ptr<CustomizePredictor> _customize_predictor;

    private:
        Pdm _pdm106;

        cv::Vec6f _out_params_global;
        cv::Mat_<float> _out_params_local;
        cv::Vec3f _vec_trans;
        cv::Vec3f _vec_rot;
        float _x;
        float _y;
        float _z;

        cv::Mat _landmarks_2d_106;
        cv::Mat _landmarks_3d_106;
        cv::Mat_<float> _points106;

        cv::Mat_<float> _shape3d106;
        cv::Mat_<float> _shape2d106;
    };

} //namespace vision
