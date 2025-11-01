#include "lmk_converter.h"
#include "vision/core/common/VConstants.h"

namespace aura::vision {
void LmkConverter::get_68_point(FaceInfo* face) {
    int count = LM_2D_106_COUNT;
    int index = 0;
    for (int i = 0; i < count; i++) {
        if (i <= 32 && i % 2 == 0) { // 0-16
            index = i / 2;
        } else if (i >= 33 && i <= 37) { // 17-21
            index = i - 16;
        } else if (i >= 42 && i <= 46) { // 22-26
            index = i - 20;
        } else if (i >= 71 && i <= 74) { // 27-30
            index = i - 44;
        } else if (i >= 81 && i <= 85) { // 31-35
            index = i - 50;
        } else if (i >= 51 && i <= 56) { // 36-41
            index = i - 15;
        } else if (i >= 61 && i <= 66) { // 42-47
            index = i - 19;
        } else if (i >= 86) { // 48-67
            index = i - 38;
        }
        face->landmark2D68[index] = face->landmark2D106[i];
    }
}
} // namespace