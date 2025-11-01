//
// Created by Li,Wendong on 2019-06-01.
//

#ifndef VISION_FRAMEINFO_H
#define VISION_FRAMEINFO_H

#include <sstream>

namespace aura::vision {

/**
* @brief frame lightness status
*/
enum FrameLightnessStatus : short {
    F_LIGHTNESS_UNKNOW = 0,
    F_LIGHTNESS_BRIGHT = 1,
    F_LIGHTNESS_DARK = 2,
    F_LIGHTNESS_NORMAL = 3
};

/**
* @brief frame occlusion status
*/
enum FaceOcclusionStatus : short {
    F_NONE_OCCLUSION = 0,       // none occlusion
    F_CAMERA_OCCLUSION = 1,     // camera side occlusion
    F_FACE_OCCLUSION = 2        // face side occlusion
};

/**
* @brief frame occlusion status
*/
enum FrameSpoofStatus : short {
    F_SPOOF_UNKNOWN = 0,            // normal
    F_SPOOF_PHOTO_ATTACK = 1,       // spoofed by device side
    F_SPOOF_SCREEN_ATTACK = 2       // spoofed by camera side
};

/**
 * @brief Image frame info
 */
class FrameInfo {
public:
    std::string tag{};
    short width = 0;
    short height = 0;
    unsigned char *frame = nullptr;
    uint64_t timestamp = 0;                /// frame timestamp,nanoseconds
    short brightness = 0;
    short state_frame_lightness = 0;   /// frame lightness status 0-unknow, 1-bright, 2-dark, 3-normal
    short state_spoof = 0;             /// frame spoof state, 0-UNKNOW, 1-PHOTO_ATTACK, 2-SCREEN_ATTACK
    short state_frame_occlusion = 0;   /// frame occlusion status 0-none occlusion, 1-camera side occlusion, 2-face side occlusion

    void copy(const FrameInfo &info);
    void clearAll();
    void clear();

    FrameInfo() noexcept;
    FrameInfo(const FrameInfo &) noexcept;
    FrameInfo(FrameInfo &&) noexcept;
    FrameInfo &operator=(const FrameInfo &) noexcept;
    FrameInfo &operator=(FrameInfo &&) noexcept;
    ~FrameInfo() = default;
    void toString(std::stringstream &ss) const;
};

} // namespace vision

#endif //VISION_FRAMEINFO_H
