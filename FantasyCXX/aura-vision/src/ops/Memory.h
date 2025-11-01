//
// Created by LiWendong on 23-3-15.
//

#pragma once

#include <stddef.h>

namespace aura::vision {
namespace op {

class Memory {
public:
    static void memoryCopy(void *a, const void *b, size_t &c);
};

} // namespace op
} // namespace aura::vision