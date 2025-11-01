#ifndef VISION_FACE_ID_UTIL_H
#define VISION_FACE_ID_UTIL_H

#include <math.h>

#include "vision/core/common/VMacro.h"

namespace aura::vision {

class VA_PUBLIC FaceIdUtil {
public:
    /**
     * @brief Get the compare score of the two face feature vectors,
     * normally this is used for identify the face from registered database
     * @param first face feature vector
     * @param second face feature vector
     * @return the compare score of the two face features
     */
    static float compare_face_features(const float *first, const float *second);

    /**
     * 计算两个图片数据的相似度
     * @param first
     * @param second
     * @return
     */
    template <typename T> static float compare_image_data(const T *first, const T *second, int len) {
        float result = 0.0f;
        if (first == nullptr || second == nullptr) {
            return result;
        }

        float norm1 = 0.000001f;
        float norm2 = 0.000001f;
        float dot = 0.0f;
        for (int i = 0; i < len; i++) {
            dot += static_cast<float>(first[i]) * static_cast<float>(second[i]);
            norm1 += static_cast<float>(first[i]) * static_cast<float>(first[i]);
            norm2 += static_cast<float>(second[i]) * static_cast<float>(second[i]);
        }
        result = dot / sqrt(norm1 * norm2);
        return result;
    }

    /**
     * @brief Update an old face feature by the current face,
     * this is used for consistently updating the database to
     * capture the recent differences of the face
     * @param old_face the face feature stored in the database
     * @param current_face the face feature extracted through the current face
     * @param output the output new face feature
     * @return update result
     */
    static bool update_face_feature(const float *storage_data, const float *cur_data, float *output_data);

    /**
     * @brief Crop and align the face
     * @param cameraLightType  RGB:0,IR:1
     * @param frameConvertBgrFormat
     * @param landmaks face landmarks
     * @param _frame image frame
     * @param w image width
     * @param h image height
     * @param resized_w cropped width
     * @param resized_h cropped height
     * @param resized_frame the aligned face image data
     */
    static int crop_face(short cameraLightType, int frameConvertBgrFormat, const float *landmarks, unsigned char *frame,
                         int w, int h, int resized_w, int resized_h, unsigned char *resized_frame);

    /**
     * @brief convert landmarks 3D to 2D
     * @param vec3dPoints  3D landmark points
     * @param vec2dPoints  2D landmark points
     * @param focalX  camera focal length pixelX
     * @param focalY camera focal length pixelY
     * @param opticalCenterX  camera optical center x
     * @param opticalCenterY camera optical center y
     */
    static bool convertPoints3Dto2D(std::vector<float> &vec3dPoints, std::vector<float> &vec2dPoints, float &focalX,
                             float &focalY, float &opticalCenterX, float &opticalCenterY);
};

} // namespace vision

#endif // VISION_FACE_ID_UTIL_H
