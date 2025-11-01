
#ifdef BUILD_EXPERIMENTAL

#include "inference/CustomizePredictor.h"
#include "inference/InferenceRegistry.h"
#include "Face2dTo3dDetector.h"
#include "util/DebugUtil.h"
#include "vacv/cv.h"
#include "vision/util/log.h"

using namespace aura::vision;

Face2Dto3DDetector::Face2Dto3DDetector() :
        _pdm_model_init_state(false) {
    _x = 0.0f;
    _y = 0.0f;
    _z = 0.0f;
    _predictor = std::dynamic_pointer_cast<AbsPredictor>(InferRegistry::get(ModelId::VISION_TYPE_FACE_2DTO3D));
    TAG = "Face2Dto3DDetector";
    mPerfTag += TAG;
}

Face2Dto3DDetector::~Face2Dto3DDetector() = default;

bool Face2Dto3DDetector::init_pdm(const char *param_mem, int mem_size) {
    if (!_pdm106.read(param_mem, mem_size)) {
        return false;
    }

    return true;
}

int Face2Dto3DDetector::init(RtConfig* cfg) {
	mRtConfig = cfg;
    V_RET(Error::OK);
}

int Face2Dto3DDetector::init_params() {
    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_2DTO3D);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "face 2dto3d predictor not registered!");

    _customize_predictor = std::dynamic_pointer_cast<CustomizePredictor>(_predictor);
    int mem_size = _customize_predictor->get_customize_param_size();
    auto* param_mem = _customize_predictor->get_customize_param_data();

    VLOGD(TAG, "init 2Dto3D pdm...");
    _pdm_model_init_state = init_pdm(param_mem, mem_size);
    if (_pdm_model_init_state) {
        VLOGD(TAG, "load 2Dto3D pdm model success...");
    } else {
        VLOGD(TAG, "load 2Dto3D pdm model failed...");
        V_RET(Error::MODEL_INIT_ERR);
    }

    _landmarks_2d_106 = cv::Mat(LM_2D_106_COUNT, 2, CV_64FC1);
    _landmarks_3d_106 = cv::Mat(LM_2D_106_COUNT, 3, CV_64FC1);
    _points106 = cv::Mat_<float>(LM_2D_106_COUNT * 2, 1, 0.0f);

    _shape3d106 = cv::Mat_<float>(LM_2D_106_COUNT * 3, 1);
    _shape2d106 = cv::Mat_<float>(LM_2D_106_COUNT * 3, 1);

    V_RET(Error::OK);
}

int Face2Dto3DDetector::prepare(VisionRequest *request, FaceInfo** infos, TensorArray& prepared) {
    V_RET(Error::OK);
}

int Face2Dto3DDetector::process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) {
    V_RET(Error::OK);
}

int Face2Dto3DDetector::post(VisionRequest *request, TensorArray& infer_results, FaceInfo** infos) {
    V_RET(Error::OK);
}

int Face2Dto3DDetector::doDetect(VisionRequest *request, VisionResult *result) {
    if (!_pdm_model_init_state) {
        return static_cast<int>(Error::MODEL_INIT_ERR);
    }
	FaceInfo** infos = result->getFaceResult()->faceInfos;
    for (int i = 0; i < mRtConfig->faceNeedDetectCount; ++i) {
        FaceInfo *face = infos[i];
        V_CHECK_CONT(face->isNotDetectType()); // 判断当前人脸是否是模型检出
        calcparams_calculate(face);
    }
    return static_cast<int>(Error::OK);
}

void Face2Dto3DDetector::calcparams_calculate(FaceInfo *faceinfo) {
    for (int i = 0; i < LM_2D_106_COUNT; i++) {
        _points106(i) = faceinfo->landmark2D106[i].x;
        _points106(i + LM_2D_106_COUNT) = faceinfo->landmark2D106[i].y;
    }

    _pdm106.calc_params(_out_params_global, _out_params_local, _points106);
    _pdm106.calc_shape_3d(_shape3d106, _out_params_local);
    _pdm106.calc_shape_2d(_shape2d106, _out_params_local, _out_params_global);
    for (int i = 0; i < _pdm106.number_of_points(); ++i) {
        faceinfo->landmark2D106[i].x = _shape2d106(i);
        faceinfo->landmark2D106[i].y = _shape2d106(i + _pdm106.number_of_points());
    }

    // TODO 需要根据不同的摄像头参数进行计算。而不是直接按照默认参数(待完成)
    // 雷宇提供的的相机默认参数。
    float fx = 500.0f * (mRtConfig->frameWidth / 640.0f);
    float fy = 500.0f * (mRtConfig->frameHeight / 480.0f);
    fx = (fx + fy) / 2.0f;
    fy = fx;
    float cx = mRtConfig->frameWidth / 2.0f;
    float cy = mRtConfig->frameHeight / 2.0f;

    // 设置实际的摄像机参数
    // fx fy x轴 y轴的焦距    cx cy
    double camd[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_64FC1, camd);

    _z = fx / _out_params_global[0];
    _x = ((_out_params_global[4] - mRtConfig->frameWidth / 2.0) * (1.0 / fx)) * _z;
    _y = ((_out_params_global[5] - mRtConfig->frameHeight / 2.0) * (1.0 / fy)) * _z;

    _vec_trans[0] = _x;
    _vec_trans[1] = _y;
    _vec_trans[2] = _z;

    _vec_rot[0] = _out_params_global[1];
    _vec_rot[1] = _out_params_global[2];
    _vec_rot[2] = _out_params_global[3];

    cv::solvePnP(_landmarks_3d_106, _landmarks_2d_106, camera_matrix,
                 cv::Mat_<double>(), _vec_rot, _vec_trans, true);

    //* 180 / M_PI;
    faceinfo->headDeflection.pitch = _vec_rot(0) * ANGEL_180 / M_PI;
    faceinfo->headDeflection.yaw = _vec_rot(1) * ANGEL_180 / M_PI;
    faceinfo->headDeflection.roll = _vec_rot(2) * ANGEL_180 / M_PI;
    VLOGD(TAG, "Face euler angle, pitch:%f \tyaw:%f \troll:%f",
                faceinfo->headDeflection.pitch, faceinfo->headDeflection.yaw, faceinfo->headDeflection.roll);
    faceinfo->headTranslation.x = _vec_trans(0);
    faceinfo->headTranslation.y = _vec_trans(1);
    faceinfo->headTranslation.z = _vec_trans(2);
    VLOGD(TAG, "Face trans vec, x:%f \ty:%f \tz:%f",
          faceinfo->headTranslation.x, faceinfo->headTranslation.y, faceinfo->headTranslation.z);
}

#endif