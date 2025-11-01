#ifndef VISION_TENSOR_H
#define VISION_TENSOR_H

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace aura::vision {

/// Data type of inference
enum DType {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    FP64 = 3,
    DTYPE_UNKNOWN
};

/// Data layout
enum DLayout {
    NCHW = 0,
    NHWC = 1
};

enum DAllocType {
	MALLOC = 0,
	FCV_ALLOC = 1
};

/**
 * Basic data structure
 * VisionAbility能力层的使用的数据结构VTensor
 * VTensor的DLayout和OpenCV以及QNN的处理格式保持一致，默认使用NHWC(后C格式)
 * 如果使用QNN推理，则不需要进行changeLayout的转换
 * 如果使用ONNX推理，需要在OnnxPredictor进行changeLayout的转换
 */
class VTensor {
public:
    VTensor();

    explicit VTensor(int w, DLayout layout = NHWC, DType dtype = FP32);

    VTensor(int w, int h, DLayout layout = NHWC, DType dtype = FP32);

    VTensor(int w, int h, int c, DLayout layout = NHWC, DType type = FP32);

    explicit VTensor(int w, DType type = FP32, DLayout layout = NHWC);

    VTensor(int w, int h, DType type = FP32, DLayout layout = NHWC);

    VTensor(int w, int h, int c, DType type = FP32, DLayout layout = NHWC);

    VTensor(int w, void *data, DType type = FP32, DLayout layout = NHWC);

    VTensor(int w, int h, void *data, DType type = FP32, DLayout layout = NHWC);

    VTensor(int w, int h, int c, void *data, DType type = FP32, DLayout layout = NHWC);

    VTensor(int w, void *data, DLayout layout = NHWC, DType type = FP32);

    VTensor(int w, int h, void *data, DLayout layout = NHWC, DType type = FP32);

    VTensor(int w, int h, int c, void *data, DLayout layout = NHWC, DType type = FP32);

    VTensor(const VTensor& t);

    ~VTensor();

    VTensor& operator=(const VTensor& t);
    VTensor clone() const;

    /**
     * 进行Tensor的数据格式的转换。
     * 注意：如果channel(c) 是1的话。则不会进行转换，返回其自身
     * @param layout
     * @return
     */
    VTensor changeLayout(DLayout layout);
    VTensor changeDType(DType dtype);

    void create(int w, DType dtype = FP32, DLayout layout = NCHW);
    void create(int w, int h, DType dtype = FP32, DLayout layout = NCHW);
	void create(int w, int h, int c, DType dtype = FP32, DLayout layout = NCHW, DAllocType allocType = MALLOC);

    void create(int w, DLayout layout = NCHW, DType dtype = FP32);
    void create(int w, int h, DLayout layout = NCHW, DType dtype = FP32);
    void create(int w, int h, int c, DLayout layout = NCHW, DType dtype = FP32);

    void release();
    bool empty() const;
    /**
     * @return 返回所占用的像素数
     */
    size_t size() const;
    /**
     * @return 返回所占用的字节数
     */
    size_t len() const;
    void setName(const std::string& name);
    std::string getName() const;
    int getRefCount() const;
	unsigned char * asUChar() const;
	int getStride() const;

    int w = 0;
    int h = 0;
    int c = 0;
    int stride = 0;
    int mPixelSize = 0;
    int mByteSize = 0;
    int dims = 0;
    void* data = nullptr;
    DType dType = DType::INT8;
    DLayout dLayout = DLayout::NHWC;
    DAllocType allocType = DAllocType::MALLOC;

	// 用于转换图像帧时临时存储 YUV 数据，由外部传入，本类内不做创建和销毁
	unsigned char *y = nullptr;
	unsigned char *u = nullptr;
	unsigned char *v = nullptr;
	unsigned char *uv = nullptr;

private:
    void addRef() const;
    std::string _name;
    int* _ref_count = nullptr;
};

using TensorArray = std::vector<VTensor>;
using TensorPtr = std::shared_ptr<VTensor>;

} // namespace vision

#endif //VISION_TENSOR_H
