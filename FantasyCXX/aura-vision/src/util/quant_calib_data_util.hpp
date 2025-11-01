#ifndef VISION_QUANT_CALIB_DATA_UTIL_HPP
#define VISION_QUANT_CALIB_DATA_UTIL_HPP

#include <string>

#include "detector/AbsDetector.h"
#include "vision/config/runtime_config/RtConfig.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/bean/GestureInfo.h"

namespace aura::vision {

class QuantCalibDataUtil {
public:
    template <typename T>
    static inline VTensor prepare_calib_data(std::shared_ptr<AbsDetector<T>>& detector, RtConfig *rtConfig, VisionRequest *request, T** infos) {
        TensorArray prepared;
        detector->init(rtConfig);
        detector->prepare(request, infos, prepared);
        if (!prepared.empty()) {
            return prepared[0];
        } else {
            return VTensor{};
        }
    }

    template <typename T>
    static inline std::vector<int> get_input_shape(std::shared_ptr<AbsDetector<T>>& detector) {
        std::vector<int> shape_data;
        if (detector->_predictor) {
            auto inputs = detector->_predictor->get_input_desc();
            auto shape = inputs[0].shape();
            shape_data.resize(3);
            shape_data[0] = shape.c();
            shape_data[1] = shape.h();
            shape_data[2] = shape.w();
        }
        return shape_data;
    }
};

} // namespace aura::vision

#endif //VISION_QUANT_CALIB_DATA_UTIL_HPP
