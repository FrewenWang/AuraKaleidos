#include "vision/core/common/VTensor.h"

#include "vision/util/log.h"
#include "vision/util/VaAllocator.h"

#if defined (USE_NEON) and __ARM_NEON
#include "arm_neon.h"
#endif

namespace aura::vision {

#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#  define IOV_FETCH_ADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define IOV_FETCH_ADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define IOV_FETCH_ADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define IOV_FETCH_ADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define IOV_FETCH_ADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define IOV_FETCH_ADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
// thread-unsafe
static inline int IOV_FETCH_ADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif

static const char *TAG = "Tensor";

VTensor::VTensor()
    : w(0), h(0), c(0), stride(0), mPixelSize(0), dims(0), data(nullptr), dType(FP32), dLayout(NCHW),
      _name(""), _ref_count(nullptr) {
}

VTensor::VTensor(int w, DType type, DLayout layout)
    : stride(w), mPixelSize(stride * c), dims(1), data(nullptr),
      _name(""), _ref_count(nullptr) {
    create(w, layout, type);
}

VTensor::VTensor(int w, int h, DType type, DLayout layout)
    : stride(w * h), mPixelSize(stride * c), dims(2), data(nullptr),
      _name(""), _ref_count(nullptr) {
    create(w, h, layout, type);
}

VTensor::VTensor(int w, int h, int c, DType type, DLayout layout)
    : stride(w * h), mPixelSize(stride * c), dims(3), data(nullptr),
      _name(""), _ref_count(nullptr) {
    create(w, h, c, layout, type);
}

VTensor::VTensor(int w, DLayout layout, DType type)
    : VTensor(w, type, layout) {
}

VTensor::VTensor(int w, int h, DLayout layout, DType type)
    : VTensor(w, h, type, layout) {
}

VTensor::VTensor(int w, int h, int c, DLayout layout, DType type)
    : VTensor(w, h, c, type, layout) {
}

VTensor::VTensor(int w, void *data, DType type, DLayout layout)
    : w(w), h(1), c(1), stride(w), mPixelSize(stride * c), dims(1), data(data), dType(type), dLayout(layout),
      _name(""), _ref_count(nullptr) {
}

VTensor::VTensor(int w, int h, void *data, DType type, DLayout layout)
    : w(w), h(h), c(1), stride(w * h), mPixelSize(stride * c), dims(2), data(data), dType(type), dLayout(layout),
      _name(""), _ref_count(nullptr) {
}

VTensor::VTensor(int w, int h, int c, void *data, DType type, DLayout layout)
    : w(w), h(h), c(c), stride(w * h), mPixelSize(stride * c), dims(3), data(data), dType(type), dLayout(layout),
      _name(""), _ref_count(nullptr) {
}

VTensor::VTensor(int w, void *data, DLayout layout, DType type)
    : VTensor(w, data, type, layout) {
}

VTensor::VTensor(int w, int h, void *data, DLayout layout, DType type)
    : VTensor(w, h, data, type, layout) {
}

VTensor::VTensor(int w, int h, int c, void *data, DLayout layout, DType type)
    : VTensor(w, h, c, data, type, layout) {
}

VTensor::~VTensor() {
    release();
}

VTensor &VTensor::operator=(const VTensor &t) {
    if (this == &t) {
        return *this;
    }
    // increase refcount
    t.addRef();
    // release current data
    release();
    w = t.w;
    h = t.h;
    c = t.c;
    stride = t.stride;
    mPixelSize = t.mPixelSize;
    mByteSize = t.mByteSize;
    dims = t.dims;
    data = t.data;
    dType = t.dType;
    dLayout = t.dLayout;
    _name = t._name;
    _ref_count = t._ref_count;
    allocType = t.allocType;
    return *this;
}

VTensor::VTensor(const VTensor &t) {
    // increase refcount
    t.addRef();
    w = t.w;
    h = t.h;
    c = t.c;
    stride = t.stride;
    mPixelSize = t.mPixelSize;
    mByteSize = t.mByteSize;
    dims = t.dims;
    data = t.data;
    dType = t.dType;
    dLayout = t.dLayout;
    _name = t._name;
    _ref_count = t._ref_count;
}

void VTensor::create(int _w, DLayout layout, DType type) {
    create(_w, 1, 1, type, layout, MALLOC);
}

void VTensor::create(int _w, int _h, DLayout layout, DType type) {
    create(_w, _h, 1, type, layout, MALLOC);
}

void VTensor::create(int _w, int _h, int _c, DLayout layout, DType type) {
    create(_w, _h, _c, type, layout, MALLOC);
}

void VTensor::create(int _w, DType type, DLayout layout) {
    create(_w, 1, 1, type, layout, MALLOC);
}

void VTensor::create(int _w, int _h, DType type, DLayout layout) {
    create(_w, _h, 1, type, layout, MALLOC);
}

void VTensor::create(int _w, int _h, int _c, DType type, DLayout layout, DAllocType _allocType) {
    if (w == _w && h == _h && c == _c && dType == type && dLayout == layout && allocType == _allocType &&
        data != nullptr)
        return;

    release();

    dType = type;
    dLayout = layout;
    allocType = _allocType;

    w = _w;
    h = _h;
    c = _c;
    stride = w * h;
    mPixelSize = stride * c;
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
            data = VaAllocator::allocate(VaAllocator::align_size(mByteSize, 4));
        } else if (allocType == FCV_ALLOC) {
            data = VaAllocator::allocateInFcv(mByteSize, 128);
        }
        _ref_count = (int *)VaAllocator::allocate((int)sizeof(*_ref_count));
        *_ref_count = 1;
    }
}

void VTensor::release() {
    if (_ref_count && IOV_FETCH_ADD(_ref_count, -1) == 1) {
        if (allocType == MALLOC) {
            VaAllocator::deallocate(data);
        } else if (allocType == FCV_ALLOC) {
            VaAllocator::deallocateInFcv(data);
        }
        VaAllocator::deallocate(_ref_count);
    }

    data = nullptr;
    dType = FP32;
    dLayout = NCHW;
    mPixelSize = 0;
    mByteSize = 0;
    stride = 0;
    w = 0;
    h = 0;
    c = 0;
    y = nullptr;
    u = nullptr;
    v = nullptr;
    uv = nullptr;
    _ref_count = nullptr;
    _name = std::string();
}

VTensor VTensor::clone() const {
    if (empty()) {
        return VTensor(0, dLayout);
    }
    VTensor t;
    t.create(w, h, c, dType, dLayout);
    if (size() > 0) {
        memcpy(t.data, data, len());
    }
    return t;
}

template<typename T>
void hwc_to_chw(T *in, T *out, int w, int h, int c) {
    int count = 0;
    int step = h * w;
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < step; ++j) {
            out[count] = in[j * c + i];
            count += 1;
        }
    }
}

template<typename T>
void chw_to_hwc(T *in, T *out, int w, int h, int c) {
    int count = 0;
    int step = h * w;
    for (int i = 0; i < step; ++i) {
        for (int j = 0; j < c; ++j) {
            out[count] = in[j * step + i];
            count += 1;
        }
    }
}

#if defined (USE_NEON) and __ARM_NEON
void hwc_to_chw_neon_u8_rgb(uint8_t *src_data, uint8_t *dst_data, int w, int h, int c) {
    int stride = w * h;

    int num8x16 = int(stride / 16);
    int remain = stride % 16;
    uint8x16x3_t intlv_rgb;
    int i = 0;
    for (i = 0; i < num8x16; i++) {
        intlv_rgb = vld3q_u8(src_data + 3 * 16 * i);
        vst1q_u8(dst_data + 16 * i, intlv_rgb.val[0]);
        vst1q_u8(dst_data + stride + 16 * i, intlv_rgb.val[1]);
        vst1q_u8(dst_data + stride * 2 + 16 * i, intlv_rgb.val[2]);
    }
    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            *(dst_data + 16 * i + j) = *(src_data + 3 * 16 * i + j * 3);
            *(dst_data + stride + 16 * i + j) = *(src_data+ 3 * 16 * i + j * 3 + 1);
            *(dst_data + stride * 2 + 16 * i + j) = *(src_data + 3 * 16 * i + j * 3 + 2);
        }
    }
}

void chw_to_hwc_neon_u8_rgb(uint8_t *src_data, uint8_t *dst_data, int w, int h, int c) {
    int stride = w * h;

    int num8x16 = stride / 16;
    int remain = stride % 16;
    uint8x16x3_t intlv_rgb;
    uint8x16_t intlv_b;
    uint8x16_t intlv_g;
    uint8x16_t intlv_r;
    int i = 0;
    for (i = 0; i < num8x16; i++) {
        intlv_b = vld1q_u8(src_data + 16 * i);
        intlv_g = vld1q_u8(src_data + stride + 16 * i);
        intlv_r = vld1q_u8(src_data + stride * 2 + 16 * i);
        intlv_rgb.val[0] = intlv_b;
        intlv_rgb.val[1] = intlv_g;
        intlv_rgb.val[2] = intlv_r;
        vst3q_u8(dst_data +  3 * 16 * i, intlv_rgb);
    }
    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            *(dst_data + 3 * 16 * i + j * 3) = *(src_data + 16 * i + j);
            *(dst_data + 3 * 16 * i + j * 3 + 1) = *(src_data + stride + 16 * i + j);
            *(dst_data + 3 * 16 * i + j * 3 + 2) = *(src_data + stride * 2 + 16 * i + j);
        }
    }
}

void hwc_to_chw_neon_fp32_rgb(float *src_data, float *dst_data, int w, int h, int c) {
    int stride = w * h;

    int num32x4 = stride / 4;
    int remain = stride % 4;
    float32x4x3_t fp32lv_rgb;
    int i = 0;
    for (i = 0; i < num32x4; i++) {
        fp32lv_rgb = vld3q_f32(src_data + 3 * 4 * i);
        vst1q_f32(dst_data + 4 * i, fp32lv_rgb.val[0]);
        vst1q_f32(dst_data + stride + 4 * i, fp32lv_rgb.val[1]);
        vst1q_f32(dst_data + stride * 2 + 4 * i, fp32lv_rgb.val[2]);
    }
    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            *(dst_data + 4 * i + j) = *(src_data + 3 * 4 * i + j * 3);
            *(dst_data + stride + 4 * i + j) = *(src_data+ 3 * 4 * i + j * 3 + 1);
            *(dst_data + stride * 2 + 4 * i + j) = *(src_data + 3 * 4 * i + j * 3 + 2);
        }
    }
}

void chw_to_hwc_neon_fp32_rgb(float *src_data, float *dst_data, int w, int h, int c) {
    int stride = w * h;

    int num32x4 = stride / 4;
    int remain = stride % 4;
    float32x4x3_t intlv_rgb;
    float32x4_t intlv_b;
    float32x4_t intlv_g;
    float32x4_t intlv_r;
    int i = 0;
    for (i = 0; i < num32x4; i++) {
        intlv_b = vld1q_f32(src_data + 4 * i);
        intlv_g = vld1q_f32(src_data + stride + 4 * i);
        intlv_r = vld1q_f32(src_data + stride * 2 + 4 * i);
        intlv_rgb.val[0] = intlv_b;
        intlv_rgb.val[1] = intlv_g;
        intlv_rgb.val[2] = intlv_r;
        vst3q_f32(dst_data + 3 * 4 * i, intlv_rgb);
    }
    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            *(dst_data + 3 * 4 * i + j * 3) = *(src_data + 4 * i + j);
            *(dst_data + 3 * 4 * i + j * 3 + 1) = *(src_data + stride + 4 * i + j);
            *(dst_data + 3 * 4 * i + j * 3 + 2) = *(src_data + stride * 2 + 4 * i + j);
        }
    }
}
void u8_to_f32_neon(uint8_t* src, float* dst, int len) {
    int num8x16 = len / 16;
    int remain = len % 16;

    uint8x16_t uint8X16;
    uint8x8_t uint8X8_low;
    uint8x8_t uint8X8_high;
    uint16x8_t uint16X8_low;
    uint16x4_t uint16X4_low;
    uint32x4_t uint32X4_low;
    float32x4_t float32X4_low;
    uint16x4_t uint16X4_high;
    uint32x4_t uint32X4_high;
    float32x4_t float32X4_high;

    int i = 0;
    for (i = 0; i < num8x16; i++) {
        // 取16个像素
        uint8x16_t uint8X16 = vld1q_u8(src + i * 16);
        uint8x8_t uint8X8_low = vget_low_u8(uint8X16);

        uint16X8_low = vmovl_u8(uint8X8_low);
        uint16X4_low = vget_low_u16(uint16X8_low);
        uint32X4_low = vmovl_u16(uint16X4_low);
        float32X4_low = vcvtq_f32_u32(uint32X4_low);
        uint16X4_high = vget_high_u16(uint16X8_low);
        uint32X4_high = vmovl_u16(uint16X4_high);
        float32X4_high = vcvtq_f32_u32(uint32X4_high);

        vst1q_f32(dst + i * 16, float32X4_low);
        vst1q_f32(dst + i * 16 + 4, float32X4_high);

        uint8X8_high = vget_high_u8(uint8X16);
        uint16X8_low = vmovl_u8(uint8X8_high);
        uint16X4_low = vget_low_u16(uint16X8_low);
        uint32X4_low = vmovl_u16(uint16X4_low);
        float32X4_low = vcvtq_f32_u32(uint32X4_low);
        uint16X4_high = vget_high_u16(uint16X8_low);
        uint32X4_high = vmovl_u16(uint16X4_high);
        float32X4_high = vcvtq_f32_u32(uint32X4_high);

        vst1q_f32(dst + i * 16 + 8, float32X4_low);
        vst1q_f32(dst + i * 16 + 12, float32X4_high);
    }

    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            *(dst + i * 16 + j) = static_cast<float>(*(src + i * 16 + j));
        }
    }
}

void f32_to_u8_neon(float* src, uint8_t* dst, int len) {
    int num8x16 = len / 16;
    int remain = len % 16;

    int i = 0;
    for (i = 0; i < num8x16; i++) {
        // 取16个像素
        float32x4_t fp32x4_0 = vld1q_f32(src + i*16);
        float32x4_t fp32x4_1 = vld1q_f32(src + i*16 + 4);
        float32x4_t fp32x4_2 = vld1q_f32(src + i*16 + 8);
        float32x4_t fp32x4_3 = vld1q_f32(src + i*16 + 12);

        // float32->uint32
        uint32x4_t u32x4_0 = vcvtq_u32_f32(fp32x4_0);
        uint32x4_t u32x4_1 = vcvtq_u32_f32(fp32x4_1);
        uint32x4_t u32x4_2 = vcvtq_u32_f32(fp32x4_2);
        uint32x4_t u32x4_3 = vcvtq_u32_f32(fp32x4_3);

        // u32->u16
        uint16x4_t u16x4_0 = vmovn_u32(u32x4_0);
        uint16x4_t u16x4_1 = vmovn_u32(u32x4_1);
        uint16x4_t u16x4_2 = vmovn_u32(u32x4_2);
        uint16x4_t u16x4_3 = vmovn_u32(u32x4_3);

        // u16 combine
        uint16x8_t u16x8_0 = vcombine_u16(u16x4_0, u16x4_1);
        uint16x8_t u16x8_1 = vcombine_u16(u16x4_2, u16x4_3);

        // u16->u8
        uint8x8_t u8x8_0 = vmovn_u16(u16x8_0);
        uint8x8_t u8x8_1 = vmovn_u16(u16x8_1);

        uint8x16_t u8x16 = vcombine_u8(u8x8_0, u8x8_1);
        vst1q_u8(dst + i * 16, u8x16);
    }

    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            *(dst + i * 16 + j) = static_cast<uint8_t>(*(src + i * 16 + j));
        }
    }
}
#endif

VTensor VTensor::changeLayout(DLayout layout) {
    if (empty()) {
        dLayout = layout;
        return VTensor(0, layout);
    }

    if (c == 1 || dLayout == layout) {
        dLayout = layout;
        return clone();
    }

    VTensor t;
    t.create(w, h, c, dType, layout);

    if (dLayout == NHWC && layout == NCHW) {
        if (dType == FP32) {
#if defined (USE_NEON) and __ARM_NEON
            if (c == 3) {
                hwc_to_chw_neon_fp32_rgb((float*)data, (float*)t.data, w, h, c);
            } else {
                hwc_to_chw<float>((float*)data, (float*)t.data, w, h, c);
            }
#else
            hwc_to_chw<float>((float *)data, (float *)t.data, w, h, c);
#endif
        } else if (dType == FP16) {
            hwc_to_chw<short>((short *)data, (short *)t.data, w, h, c);
        } else if (dType == INT8) {
#if defined (USE_NEON) and __ARM_NEON
            if (c == 3) {
                hwc_to_chw_neon_u8_rgb((uint8_t*)data, (uint8_t*)t.data, w, h, c);
            } else {
                hwc_to_chw<char>((char*)data, (char*)t.data, w, h, c);
            }
#else
            hwc_to_chw<char>((char *)data, (char *)t.data, w, h, c);
#endif
        }
    } else if (dLayout == NCHW && layout == NHWC) {
        if (dType == FP32) {
#if defined (USE_NEON) and __ARM_NEON
            if (c == 3) {
                chw_to_hwc_neon_fp32_rgb((float*)data, (float*)t.data, w, h, c);
            } else {
                chw_to_hwc<float>((float*)data, (float*)t.data, w, h, c);
            }
#else
            chw_to_hwc<float>((float *)data, (float *)t.data, w, h, c);
#endif
        } else if (dType == FP16) {
            chw_to_hwc<short>((short *)data, (short *)t.data, w, h, c);
        } else if (dType == INT8) {
#if defined (USE_NEON) and __ARM_NEON
            if (c == 3) {
                chw_to_hwc_neon_u8_rgb((uint8_t*)data, (uint8_t*)t.data, w, h, c);
            } else {
                chw_to_hwc<char>((char*)data, (char*)t.data, w, h, c);
            }
#else
            chw_to_hwc<char>((char *)data, (char *)t.data, w, h, c);
#endif
        }
    }

    return t;
}

VTensor VTensor::changeDType(DType type) {
    if (empty()) {
        return VTensor();
    }

    if (dType == type) {
        return clone();
    }

    VTensor t;
    t.create(w, h, c, type, dLayout);

    // copy date
    // todo: accelerate
    if (type == FP32 && dType == INT8) {
#if defined (USE_NEON) and __ARM_NEON
        u8_to_f32_neon((uint8_t*)data, (float*)t.data, t.size());
#else
        auto *dst_ptr = (float *)t.data;
        auto *src_ptr = (unsigned char *)data;
        for (int i = 0; i < (int)size(); ++i) {
            dst_ptr[i] = static_cast<float>(src_ptr[i]);
        }
#endif
    } else if (type == INT8 && dType == FP32) {
#if defined (USE_NEON) and __ARM_NEON
        f32_to_u8_neon((float*)data, (uint8_t*)t.data, t.size());
#else
        auto *dst_ptr = (unsigned char *)t.data;
        auto *src_ptr = (float *)data;
        for (int i = 0; i < (int)size(); ++i) {
            dst_ptr[i] = static_cast<char>(src_ptr[i]);
        }
#endif
    } else {
        // todo: support more conversions
        VLOGE(TAG, "Not supported yet to convert from dType: %d to dType: %d", dType, type);
    }
    return t;
}

bool VTensor::empty() const {
    return data == nullptr || mPixelSize == 0;
}

size_t VTensor::size() const {
    return mPixelSize;
}

size_t VTensor::len() const {
    return mByteSize;
}

void VTensor::setName(const std::string &name) {
    _name = name;
}

std::string VTensor::getName() const {
    return _name;
}

int VTensor::getRefCount() const {
    if (_ref_count) {
        return *_ref_count;
    } else {
        return 0;
    }
}

void VTensor::addRef() const {
    if (_ref_count) {
        IOV_FETCH_ADD(_ref_count, 1);
    }
}

unsigned char *VTensor::asUChar() const {
    return (unsigned char *)data;
}

int VTensor::getStride() const {
    return w * c;
}

} // namespace aura::vision
