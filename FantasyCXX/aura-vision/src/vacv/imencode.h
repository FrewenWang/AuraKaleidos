#ifndef VISION_IMENCODE_H
#define VISION_IMENCODE_H

#include <vector>

#include "vision/core/common/VTensor.h"

namespace aura::va_cv {

class ImEncode {
public:
    static void imencode(const vision::VTensor& src, std::vector<unsigned char>& buf, const char* format);
};

} // namespace aura::va_cv

#endif //VISION_IMENCODE_H
