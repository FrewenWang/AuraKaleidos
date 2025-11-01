#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <sstream>

#define safe_unlikely(x) (__builtin_expect((x), 0))
namespace safe {

/**
 * @brief 自定义的使用安全的Vector
 */
#if defined(USE_SAFE_VECTOR) && defined(__aarch64__) || defined(_M_ARM64)
template<typename T, typename Allocator = std::allocator<T>>
class vector : public std::vector<T, Allocator> {
public:
    using std::vector<T, Allocator>::vector;
    typedef typename std::vector<T, Allocator>::size_type size_type;
    using std::vector<T, Allocator>::_M_impl;

    __attribute__((always_inline)) T& operator[](size_type index) {
        if (safe_unlikely(index >= static_cast<size_type>(_M_impl._M_finish - _M_impl._M_start))) {
            std::ostringstream oss;
            oss << "Index out of range: " << index << " size: " << (_M_impl._M_finish - _M_impl._M_start);
            throw std::out_of_range(oss.str());
        }
        return *(_M_impl._M_start + index);
    }

    __attribute__((always_inline)) const T& operator[](size_type index) const {
        if (safe_unlikely(index >= static_cast<size_type>(_M_impl._M_finish - _M_impl._M_start))) {
            std::ostringstream oss;
            oss << "Index out of range: " << index << " size: " << (_M_impl._M_finish - _M_impl._M_start);
            throw std::out_of_range(oss.str());
        }
        return *(_M_impl._M_start + index);
    }
};
#else
template<typename T, typename Allocator = std::allocator<T>>
using vector = std::vector<T, Allocator>;
#endif

}
#