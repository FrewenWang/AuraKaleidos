#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace alice::inference {

std::vector<float> load_image_nchw(
    const std::string& filename,
    std::int64_t channels,
    std::int64_t height,
    std::int64_t width);

}  // namespace alice::inference
