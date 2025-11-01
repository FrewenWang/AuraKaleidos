//
// Created by Frewen.Wong on 2022/4/23.
//
#pragma once

#include <cstddef>
#include <iostream>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace aura::cv
{
/// Data type of inference
enum DType {
    FP32 = 0, FP16 = 1, INT8 = 2, FP64 = 3, D_TYPE_UNKNOWN
};

/// Data Layout
enum DLayout {
    NCHW = 0, NHWC = 1
};

/// Data Allocate Type
enum DAllocType {
    MALLOC = 0, FAST_CV_ALLOC = 1
};


/// Basic data structure
class ATensor {
public:
    /**
     * 默认构造函数
     */
    ATensor();

    /**
     *
     * @param w
     * @param layout
     * @param dType
     */
    explicit ATensor(int w, DLayout layout = NCHW, DType type = FP32);

    explicit ATensor(int w, int h, DLayout layout = NCHW, DType type = FP32);

    explicit ATensor(int w, int h, int c, DLayout layout = NCHW, DType type = FP32);

    /**
     *
     * @param w
     * @param h
     * @param c
     * @param data
     * @param layout
     * @param type
     */
    explicit ATensor(int w, void *data, DLayout layout = NCHW, DType type = FP32);

    explicit ATensor(int w, int h, void *data, DLayout layout = NCHW, DType type = FP32);

    explicit ATensor(int w, int h, int c, void *data, DLayout layout = NCHW, DType type = FP32);

    ATensor(const ATensor &t);

    /**
     * 析构函数
     */
    ~ATensor();

    /**
     *
     * @param w
     * @param h
     * @param c
     * @param layout
     * @param type
     * @param allocType
     */
    void create(int w, DLayout layout = NCHW, DType type = FP32);

    void create(int w, int h, DLayout layout = NCHW, DType type = FP32);

    void create(int w, int h, int c, DLayout layout = NCHW, DType type = FP32, DAllocType allocType = MALLOC);

    void release();

    bool empty() const;

    size_t size() const;

    size_t len() const;

    ATensor clone() const;

    void setName(const std::string &name);

    std::string getName() const;

    int getRefCount() const;

    unsigned char *asUChar() const;

    int getStride() const;

    int width = 0;
    int height = 0;
    int channel = 0;
    int stride = 0;
    /** ATensor的像素大小 */
    int mPixelSize = 0;
    /** ATensor的字节大小 */
    int mByteSize = 0;
    int dims = 0;
    void *data = nullptr;
    DType dType = DType::INT8;
    DLayout dLayout = DLayout::NCHW;
    DAllocType allocateType = DAllocType::MALLOC;

public:
    /**
     * 定义模板方法，用于将AATensor转换成对应类型的数据
     * @tparam T
     * @param ATensor
     * @param copy
     * @return
     */
    template<typename T>
    static T convertTo(const ATensor &ATensor, bool copy = false);

    template<typename T>
    static ATensor convertFrom(const T &mat, bool copy = false);

private:
    void addRef() const;

    std::string tName = "";
    int *refCount = nullptr;
};

using ATensorArray = std::vector<ATensor>;
using ATensorPtr = std::shared_ptr<ATensor>;

}
