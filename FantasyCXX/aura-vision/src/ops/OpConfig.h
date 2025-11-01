//
// Created by LiWendong on 23-3-15.
//

#pragma once

namespace aura::vision {
namespace op {

#if __ARM_NEON
#define SUPPORT_NEON 1
#if __aarch64__
#define SUPPORT_AARCH64 1
#endif
#endif

} // namespace op
} // namespace aura::vision