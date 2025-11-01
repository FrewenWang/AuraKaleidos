#include "InferenceConverter.hpp"

namespace aura::vision{
// model id
const std::unordered_map<std::string, ModelId> InferenceConverter::modelIdMap{
        {"source1_camera_cover",        VISION_TYPE_SOURCE1_CAMERA_COVER},
        {"face_rect",                   VISION_TYPE_FACE_RECT},
        {"face_landmark",               VISION_TYPE_FACE_LANDMARK},
        {"face_feature",                VISION_TYPE_FACE_FEATURE},
        {"face_no_interact_living_rgb", VISION_TYPE_FACE_NO_INTERACT_LIVING_RGB},
        {"face_no_interact_living_ir",  VISION_TYPE_FACE_NO_INTERACT_LIVING_IR},
        {"face_2dto3d",                 VISION_TYPE_FACE_2DTO3D},
        {"face_attribute_rgb",          VISION_TYPE_FACE_ATTRIBUTE_RGB},
        {"face_attribute_ir",           VISION_TYPE_FACE_ATTRIBUTE_IR},
        {"face_call",                   VISION_TYPE_FACE_CALL},
        {"face_dangerous_driving",      VISION_TYPE_FACE_DANGEROUS_DRIVING},
        {"face_drink",                  VISION_TYPE_FACE_DRINK},
        {"face_emotion",                VISION_TYPE_FACE_EMOTION},
        {"face_eye_center",             VISION_TYPE_FACE_EYE_CENTER},
        {"face_mouth_landmark",         VISION_TYPE_FACE_MOUTH_LANDMARK},
        {"face_eye_gaze",               VISION_TYPE_FACE_EYE_GAZE},
        {"face_quality",                VISION_TYPE_FACE_QUALITY},
        {"gesture_rect",                VISION_TYPE_GESTURE_RECT},
        {"gesture_landmark",            VISION_TYPE_GESTURE_LANDMARK},
        {"person_body",                 VISION_TYPE_PERSON_BODY},
        {"person_landmark",             VISION_TYPE_PERSON_LANDMARK},
        {"biology_category",            VISION_TYPE_BIOLOGY_CATEGORY},
        {"face_reconstruct",            VISION_TYPE_FACE_RECONSTRUCT},
        {"head_shoulder",               VISION_TYPE_HEAD_SHOULDER},
        {"body_landmark",               VISION_TYPE_BODY_LANDMARK}
};

// dtype
const std::unordered_map <std::string, DType> InferenceConverter::dTypeMap{
        {"fp32", FP32},
        {"fp16", FP16},
        {"int8", INT8}
};

// infer type
const std::unordered_map<std::string, InferType> InferenceConverter::inferTypeMap{
        {"ncnn",        NCNN},
        {"tnn",         TNN},
        {"paddle_lite", PADDLE_LITE},
        {"TF_lite",     TF_LITE},
        {"snpe",        SNPE},
        {"qnn",         QNN},
        {"onnx",        ONNX}
};

} // namespace aura::vision