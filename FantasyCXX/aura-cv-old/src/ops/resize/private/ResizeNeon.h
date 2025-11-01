#ifndef VISION_RESIZE_NEON_H
#define VISION_RESIZE_NEON_H

#include <cstdint>

namespace aura::aura_cv {

class ResizeNeon {

public:
    /**
     * 使用Neon进行加速的单通道双线性差值的resize功能
     * @param src
     * @param w_in
     * @param h_in
     * @param dst
     * @param w_out
     * @param h_out
     */
    static void resizeNeonInterLinearOneChannel(const uint8_t *src, int w_in, int h_in,
                                                uint8_t *dst, int w_out, int h_out);
    
    /**
     * 使用Neon进行加速的三通道双线性差值的resize功能
     * @param src
     * @param w_in
     * @param h_in
     * @param dst
     * @param w_out
     * @param h_out
     */
    static void resizeNeonInterLinearThreeChannel(const uint8_t *src, int w_in, int h_in,
                                                  uint8_t *dst, int w_out, int h_out);
    
};

}
#endif //VISION_RESIZE_NEON_H
