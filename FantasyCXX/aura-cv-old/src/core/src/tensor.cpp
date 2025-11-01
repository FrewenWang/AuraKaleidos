//
// Created by Frewen.Wang on 2022/9/29.
//

#include "aura/cv/tensor.h"
#include "aura/cv/Allocator.h"
#include "opencv2/core/core.hpp"

namespace aura::cv {
// ======================== 工具函数宏定义 =====================================
// 如果是intel的编译器，并且不是win32设备
#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#  define FETCH_ADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define FETCH_ADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define FETCH_ADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define FETCH_ADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define FETCH_ADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define FETCH_ADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
// thread-unsafe
static inline int FETCH_ADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif

ATensor::ATensor() : width(0), height(0), channel(0), stride(0), dims(0), data(nullptr), dType(FP32), dLayout(NCHW),
                     refCount(nullptr) {
}

ATensor::ATensor(int w, DLayout layout, DType type) : stride(w), dims(1), data(nullptr),
                                                      refCount(nullptr) {
    create(w, height, channel, layout, type);
}

ATensor::ATensor(int w, int h, DLayout layout, DType type)
        : stride(w * h), dims(2), data(nullptr),
          refCount(nullptr) {
    create(w, h, layout, type);
}

ATensor::ATensor(int w, int h, int c, DLayout layout, DType type)
        : stride(w * h), dims(3), data(nullptr),
          refCount(nullptr) {
    create(w, h, c, layout, type);
}


ATensor::ATensor(int w, void *data, DLayout layout, DType type)
        : width(w), height(1), channel(1), stride(w), dims(1), data(data), dType(type), dLayout(layout),
          refCount(nullptr) {
}

ATensor::ATensor(int w, int h, void *data, DLayout layout, DType type)
        : width(w), height(1), channel(1), stride(w), dims(1), data(data), dType(type), dLayout(layout),
          refCount(nullptr) {
}

ATensor::ATensor(int w, int h, int c, void *data, DLayout layout, DType type)
        : width(w), height(1), channel(1), stride(w), dims(1), data(data), dType(type), dLayout(layout),
          refCount(nullptr) {
}

ATensor::ATensor(const ATensor &t) {
    // increase refcount
    t.addRef();
    width = t.width;
    height = t.height;
    channel = t.channel;
    stride = t.stride;
    dims = t.dims;
    data = t.data;
    dType = t.dType;
    dLayout = t.dLayout;
    
    tName = t.tName;
    refCount = t.refCount;
}

ATensor::~ATensor() {
    release();
}

void ATensor::create(int w, int h, int c, DLayout layout, DType type, DAllocType allocType) {
    if (w == width && h == height && c == channel && dType == type && dLayout == layout && allocType == allocateType
        && data != nullptr) {
        return;
    }
    release();
    dType = type;
    dLayout = layout;
    allocType = allocateType;
    
    width = w;
    height = h;
    channel = c;
    // 步长
    stride = w * h;
    mPixelSize = width * height * channel;
    // 计算Tensor的大小，如果type为FP64。则每个像素占8个字节。
    // FP32则每个像素占4个字节，FP16则每个像素占2个字节，INT8则每个像素占1个字节
    if (dType == FP64) {
        mByteSize = stride * c * 8;
    } else if (dType == FP32) {
        mByteSize = stride * c * 4;
    } else if (dType == FP16) {
        mByteSize = stride * c * 2;
    } else if (dType == INT8) {
        mByteSize = stride * c;
    } else {
        mByteSize = stride * c;
    }
    
    if (h == 1 && c == 1) {
        dims = 1;
    } else if (c == 1) {
        dims = 2;
    } else {
        dims = 3;
    }
    
    if (mByteSize > 0) {
        if (allocType == MALLOC) {
            data = Allocator::allocate(Allocator::alignSize(mByteSize, 4));
        } else if (allocType == FAST_CV_ALLOC) {
            data = Allocator::allocateInFastCV(mByteSize, 128);
        }
    }
    
}


void ATensor::release() {
    if (refCount && FETCH_ADD(refCount, -1) == 1) {
        if (allocateType == MALLOC) {
            Allocator::deallocate(data);
        } else if (allocateType == FAST_CV_ALLOC) {
            Allocator::deallocateInFastCV(data);
        }
        Allocator::deallocate(refCount);
    }
}

bool ATensor::empty() const {
    return data == nullptr;
}

size_t ATensor::size() const {
    return width * height * channel;
}

ATensor ATensor::clone() const {
    // 如果当前ATensor是空的。则重新创建ATensor对象
    if (empty()) {
        return ATensor();
    }
    ATensor t;
    t.create(width, height, channel, dLayout, dType);
    
    if (size() > 0) {
        memcpy(t.data, data, len());
    }
    return t;
}

void ATensor::addRef() const {

}

void ATensor::create(int w, DLayout layout, DType type) {

}

void ATensor::create(int w, int h, DLayout layout, DType type) {

}

void ATensor::setName(const std::string &name) {

}

std::string ATensor::getName() const {
    return std::string();
}

int ATensor::getRefCount() const {
    return 0;
}

unsigned char *ATensor::asUChar() const {
    return nullptr;
}

int ATensor::getStride() const {
    return 0;
}

size_t ATensor::len() const {
    return 0;
}

/**
 * 将图像数据由hwc(后C)转换成chw(前C)
 * @tparam T
 * @param in
 * @param out
 * @param w
 * @param h
 * @param c
 */
template<typename T>
void hwc2chw(T *in, T *out, int w, int h, int c) {
    int count = 0;
    // 数据计算法的步长，
    int step = h * w;
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < step; ++j) {
            out[count] = in[j * c + i];
            count += 1;
        }
    }
}

/**
 * 将图像数据由chw(前C)转换成hwc(后C)
 * @tparam T
 * @param in
 * @param out
 * @param w
 * @param h
 * @param c
 */
template<typename T>
void chw2hwc(T *in, T *out, int w, int h, int c) {
    int count = 0;
    int step = h * w;
    for (int i = 0; i < step; ++i) {
        for (int j = 0; j < c; ++j) {
            out[count] = in[j * step + i];
            count += 1;
        }
    }
}


template<>
::cv::Mat ATensor::convertTo<::cv::Mat>(const ATensor &tensor, bool copy) {
    if (tensor.empty()) {
        return {};
    }
    int w = tensor.width;
    int h = tensor.height;
    int c = tensor.channel;
    auto dtype = tensor.dType;
    
    int mat_type;
    if (dtype == FP32) {
        mat_type = CV_32FC(c);
    } else if (dtype == FP16) {
        mat_type = CV_16UC(c);
    } else if (dtype == INT8) {
        mat_type = CV_8UC(c);
    } else if (dtype == FP64) {
        mat_type = CV_64FC(c);
    } else {
        // 代码规范：通用异常的规范不应给抛出，应该抛出具体的异常信息
        // throw std::runtime_error("TensorConverter exception when converted to cv::Mat, dtype not supported!");
        throw std::invalid_argument("TensorConverter exception when converted to cv::Mat, dType not supported!");
    }
    // 如果做数据拷贝
    if (copy) {
        // 实例化一个mat对象
        ::cv::Mat mat(h, w, mat_type);
        // 进行数据拷贝
        memcpy(mat.data, tensor.data, tensor.len());
        return mat;
    }
    return {h, w, mat_type, tensor.data};
}


} // namespace aura::aura_cv