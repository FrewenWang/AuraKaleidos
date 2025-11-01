#ifndef VISION_VFRAME_H
#define VISION_VFRAME_H

#include "vision/core/common/VTensor.h"

namespace aura::vision{

struct VFrameInfo {
    int width;
    int height;
    FrameFormat format;
    void* data;

    VTensor gray; // grey image
    VTensor rgb;  // rgb image,
	VTensor bgr;  // format is B-G-R

    VFrameInfo() : width(0), height(0), format(FrameFormat::UNKNOWN), data(nullptr), gray{}, rgb{} {};
    VFrameInfo(int w, int h, FrameFormat f, void* raw) : width(w), height(h), format(f), data(raw), gray{}, rgb{} {}

    VFrameInfo& operator= (const VFrameInfo& f) {
        if (this == &f) {
            return *this;
        }
        width = f.width;
        height = f.height;
        format = f.format;
        data = f.data;
		gray = f.gray;
        rgb = f.rgb;
        return *this;
    }

    bool has_grey() const { return !gray.empty(); }
    bool has_rgb() const { return !rgb.empty(); }
    bool has_data() const { return data != nullptr; }
};

} // namespace vision

#endif //VISION_VFRAME_H
