#pragma once

#include "vision/core/common/VTensor.h"

namespace aura::vision {

class TensorConverter {
public:
    template<typename T>
    static T convert_to(const VTensor &tensor, bool copy = false);

    template<typename T>
    static VTensor convert_from(const T &mat, bool copy = false);

};

} // namespace aura::vision
