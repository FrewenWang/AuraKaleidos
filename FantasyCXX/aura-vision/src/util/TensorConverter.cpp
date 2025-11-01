#include <stdexcept>
#include "util/DebugUtil.h"
#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif
#ifdef USE_SNPE
#include "DlSystem/ITensorFactory.hpp"
#include "SNPE/SNPEFactory.hpp"
#endif

#ifdef USE_QNN
//#include "QnnTypes.h"
//#include "IOTensor.hpp"
//#include "DataUtil.hpp"
//#include "QnnWrapperUtils.hpp"
//using namespace qnn;
//using namespace qnn::tools;
//using namespace qnn::tools::datautil;
//using namespace qnn::tools::iotensor;
#endif
#ifdef USE_PADDLE_LITE
#include "paddle_api.h"
#endif
#ifdef BUILD_NCNN
#include "net.h"
#endif
#ifdef USE_TF_LITE
#include "tensorflow/lite/c/c_api_internal.h"
#endif

#include "TensorConverter.h"
#include "vision/util/log.h"

namespace aura::vision {

static const char* TAG = "TensorConverter";

template <typename T> VTensor TensorConverter::convert_from(const T& mat, bool copy) {
    VLOGE("TensorConverter", "Unsupported typename for TensorConverter");
    return VTensor();
}

#ifdef BUILD_NCNN
template <>
ncnn::Mat TensorConverter::convert_to<ncnn::Mat>(const VTensor & tensor, bool copy) {
    ncnn::Mat mat;
    int w = tensor.w;
    int h = tensor.h;
    int c = tensor.c;

    if (tensor.dtype == FP32) {
        mat.create(w, h, c, (size_t)4);
        if (!copy && tensor.stride == static_cast<int>(mat.cstep)) {
            mat = ncnn::Mat(w, h, c, tensor.data, (size_t)4);
        } else {
            auto* src = (char*) tensor.data;
            auto* dst = (char*) mat.data;
            for (int i = 0; i < c; ++i) {
                memcpy(dst, src, w * h * 4);
                dst += mat.cstep * 4;
                src += w * h * 4;
            }
        }
    } else if (tensor.dtype == FP16) {
        mat.create(w, h, c, (size_t)2);
        if (!copy && tensor.stride == static_cast<int>(mat.cstep)) {
            mat = ncnn::Mat(w, h, c, tensor.data, (size_t)2);
        } else {
            auto* src = (char*) tensor.data;
            auto* dst = (char*) mat.data;
            for (int i = 0; i < c; ++i) {
                memcpy(dst, src, w * h * 2);
                dst += mat.cstep * 2;
                src += w * h * 2;
            }
        }
    } else if (tensor.dtype == INT8) {
        mat.create(w, h, c, (size_t)1);
        if (!copy && tensor.stride == static_cast<int>(mat.cstep)) {
            mat = ncnn::Mat(w, h, c, tensor.data, (size_t)1);
        } else {
            auto* src = (char*) tensor.data;
            auto* dst = (char*) mat.data;
            for (int i = 0; i < c; ++i) {
                memcpy(dst, src, w * h);
                dst += mat.cstep;
                src += w * h;
            }
        }
    }

    return mat;
}

template <> VTensor TensorConverter::convert_from<ncnn::Mat>(const ncnn::Mat& mat, bool copy) {
    VTensor t;
    int w = mat.w;
    int h = mat.h;
    int c = mat.c;

    if (mat.elemsize == 4) {
        t.create(w, h, c, FP32, NCHW);
        if (!copy && t.stride == static_cast<int>(mat.cstep)) {
            t = VTensor(w, h, c, mat.data, FP32, NCHW);
        } else {
            auto* src = (char*) mat.data;
            auto* dst = (char*) t.data;
            for (int i = 0; i < c; ++i) {
                memcpy(dst, src, w * h * 4);
                src += mat.cstep * 4;
                dst += w * h * 4;
            }
        }
    } else if (mat.elemsize == 2) {
        t.create(w, h, c, FP16, NCHW);
        if (!copy && t.stride == static_cast<int>(mat.cstep)) {
            t = VTensor(w, h, c, mat.data, FP16, NCHW);
        } else {
            auto* src = (char*) mat.data;
            auto* dst = (char*) t.data;
            for (int i = 0; i < c; ++i) {
                memcpy(dst, src, w * h * 2);
                src += mat.cstep * 2;
                dst += w * h * 2;
            }
        }
    } else if (mat.elemsize == 1) {
        t.create(w, h, c, INT8, NCHW);
        if (!copy && t.stride == static_cast<int>(mat.cstep)) {
            t = VTensor(w, h, c, mat.data, INT8, NCHW);
        } else {
            auto* src = (char*) mat.data;
            auto* dst = (char*) t.data;
            for (int i = 0; i < c; ++i) {
                memcpy(dst, src, w * h);
                src += mat.cstep;
                dst += w * h;
            }
        }
    }

    return t;
}
#endif // NCNN

#ifdef USE_SNPE
    template <>
    std::unique_ptr<zdl::DlSystem::ITensor> TensorConverter::convert_to<std::unique_ptr<zdl::DlSystem::ITensor>>(const Tensor& tensor, bool copy) {
        zdl::DlSystem::Dimension dims[4];
        dims[0] = 1;
        dims[1] = tensor.h;
        dims[2] = tensor.w;
        dims[3] = tensor.c;
        size_t size = 4;
        zdl::DlSystem::TensorShape tensorShape(dims, size);
        std::unique_ptr<zdl::DlSystem::ITensor> itenso_ptr;

        // TODO only consider the situation which snpe input dtype is float32 in this version
        if (!copy) {
            throw std::runtime_error("use copy when create a snpe ITensor");
        } else {
            auto* src = (float*) tensor.data;
            itenso_ptr = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(tensorShape);
            std::copy(src, src + tensor.size(), itenso_ptr->begin());
        }

        return itenso_ptr;
    }


    template <>
    Tensor TensorConverter::convert_from<zdl::DlSystem::ITensor>(const zdl::DlSystem::ITensor& itensor, bool copy) {

        Tensor t;

        auto itensor_shape = itensor.getShape();
        auto* dims = itensor_shape.getDimensions();
        size_t dim_count = itensor_shape.rank();

        int w, h, c = 1;
        if (dim_count == 4) {
            h = dims[1];
            w = dims[2];
            c = dims[3];
        } else if (dim_count == 2) {
            h = dims[0];
            w = dims[1];
        } else {
            // TODO handle other dim_count for snpe output ITensor
            VLOGE("TensorConverter", "unhandled snpe output tensor dims : %d", dim_count);
            throw std::runtime_error("unhandled snpe output tensor dims");
        }

        // TODO only consider the situation which snpe output dtype is float32 in this version
        t.create(w, h, c, FP32, NHWC);

        if (!copy) {
            throw std::runtime_error("only collect Snpe ITensor data by memory copy");
        } else {
            std::copy(itensor.cbegin(), itensor.cend(), (float*)t.data);
        }
        if (c > 1) {
            // snpe ITensor data layout:NHWC, change to NCHW
            t = t.change_layout(NCHW);
        }
        return t;
    }
#endif

#ifdef  USE_QNN
    /*
    template <>
    std::unique_ptr<Qnn_Tensor_t> TensorConverter::convert_to<std::unique_ptr<Qnn_Tensor_t>>(const Tensor& tensor, bool copy) {

    }

    template <>
    Tensor TensorConverter::convert_from<qnn_wrapper_api::QnnOutputInfo>(const qnn_wrapper_api::QnnOutputInfo& qnn_tensor, bool copy) {

        Tensor t;

        std::vector<size_t> dims;
        for (int i = 0; i < qnn_tensor.dims.size(); i ++) {
            dims.push_back(qnn_tensor.dims[i]);
        }

        uint8_t* bufferToWrite = reinterpret_cast<uint8_t*>(qnn_tensor.data);
        int length = qnn_tensor.length;

        int dim_count = qnn_tensor.dims.size();
        int w, h, c = 1;
        if (dim_count == 4) {
            h = dims[1];
            w = dims[2];
            c = dims[3];
        } else if (dim_count == 2) {
            h = dims[0];
            w = dims[1];
        } else {
            // TODO handle other dim_count for qnn output ITensor
            VLOGE("TensorConverter", "unhandled qnn output tensor dims : %d", dim_count);
            throw std::runtime_error("unhandled qnn output tensor dims");
        }

        // TODO only consider the situation which qnn output dtype is float32 in this version
        t.create(w, h, c, FP32, NHWC);

        if (!copy) {
            throw std::runtime_error("only collect qnn ITensor data by memory copy");
        } else {
            memcpy(t.data, bufferToWrite, length);
        }
        if (c > 1) {
            // qnn Tensor data layout:NHWC, change to NCHW
            t = t.change_layout(NCHW);
        }
        return t;
    }
     */
#endif

#ifdef USE_OPENCV
template <>
cv::Mat TensorConverter::convert_to<cv::Mat>(const VTensor & tensor, bool copy) {
    if (tensor.empty()) {
        return {};
    }

    int w = tensor.w;
    int h = tensor.h;
    int c = tensor.c;
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
        throw std::runtime_error("TensorConverter exception when converted to cv::Mat, dtype not supported!");
    }

    if (copy) {
        cv::Mat mat(h, w, mat_type);
        memcpy(mat.data, tensor.data, tensor.len());
        return mat;
    }
    return {h, w, mat_type, tensor.data};
}

template <> VTensor TensorConverter::convert_from<cv::Mat>(const cv::Mat& mat, bool copy) {
    if (mat.empty()) {
        return {};
    }

    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    auto depth = mat.depth();
    DType dtype = INT8;
    switch (depth) {
        case CV_8U:
        case CV_8S:
            dtype = INT8;
            break;
        case CV_16U:
        case CV_16S:
            dtype = FP16;
            break;
        case CV_32S:
        case CV_32F:
            dtype = FP32;
            break;
        case CV_64F:
            dtype = FP64;
            break;
        default:
            throw std::runtime_error("TensorConverter exception when converted from cv::Mat, depth not supported!");
    }

    if (copy) {
        VTensor t(w, h, c, dtype, NHWC);
        memcpy(t.data, mat.data, t.len());
        return t;
    }
    return {w, h, c, mat.data, dtype, NHWC};
}
#endif

#ifdef USE_PADDLE_LITE
template <>
VTensor TensorConverter::convert_from<paddle::lite_api::Tensor>(const paddle::lite_api::Tensor& lite_tensor, bool copy) {
    auto shape = lite_tensor.shape();
    auto dims = shape.size();
    if (dims == 0) {
        VLOGW(TAG, "lite_tensor dim is zero!");
        return VTensor{};
    }

    // NOTE: 目前只支持 NCHW 格式，以及 float32数据类型！
    int c = 1;
    int h = 1;
    int w = 1;
    switch (static_cast<int>(dims)) {
        case 1:
            w = shape[0];
            break;
        case 2:
            h = shape[0];
            w = shape[1];
            break;
        case 3:
            c = shape[0];
            h = shape[1];
            w = shape[2];
            break;
        case 4:
            c = shape[1];
            h = shape[2];
            w = shape[3];
            break;
        default: break;
    }

    if (copy) {
        VTensor t(w, h, c, FP32, NCHW);
        // DBG_PRINT_ARRAY((float *)(lite_tensor.data<float>()),100,"paddle-lite_post_data");
        memcpy(t.data, lite_tensor.data<float>(), t.len());
        return t;
    }
    return VTensor(w, h, c, (void*)(lite_tensor.data<float>()), FP32, NCHW);
}
#endif

#ifdef USE_TF_LITE
template <>
Tensor TensorConverter::convert_from<TfLiteTensor>(const TfLiteTensor& tf_tensor, bool copy) {
    auto dim_size = tf_tensor.dims->size;
    auto shape = tf_tensor.dims->data;
    if (dim_size == 0) {
        VLOGW(TAG, "tf_tensor dim is zero!");
        return Tensor{};
    }

    // NOTE: 目前只支持 NCHW 格式，以及 float32数据类型！
    int c = 1;
    int h = 1;
    int w = 1;
    switch (static_cast<int>(dim_size)) {
        case 1:
            w = shape[0];
            break;
        case 2:
            h = shape[0];
            w = shape[1];
            break;
        case 3:
            h = shape[0];
            w = shape[1];
            c = shape[2];
            break;
        case 4:
            h = shape[1];
            w = shape[2];
            c = shape[3];
            break;
        default: break;
    }

    float* buf = tf_tensor.data.f;
    if (copy) {
        Tensor t(w, h, c, FP32, NCHW);
        memcpy(t.data, buf, tf_tensor.bytes);
        return t;
    }
    return Tensor(w, h, c, (void*)buf, FP32, NCHW);
}
#endif

} // namespace aura::vision