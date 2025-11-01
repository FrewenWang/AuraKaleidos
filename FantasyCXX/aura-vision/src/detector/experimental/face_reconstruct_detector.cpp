#include "face_reconstruct_detector.h"

#include "opencv2/opencv.hpp"

#include "inference/InferenceRegistry.h"
#include "util/math_utils.h"
#include "util/TensorConverter.h"
#include "vision/util/VaAllocator.h"
#include "vacv/cv.h"
#include "vision/util/ImageUtil.h"

namespace vision {

static const char* TAG = "FaceReconstructDetector";

FaceReconstructDetector::FaceReconstructDetector() {
    init_params();
    _predictor = InferRegistry::get(ModelId::VISION_TYPE_FACE_RECONSTRUCT);
}

FaceReconstructDetector::~FaceReconstructDetector() = default;

int FaceReconstructDetector::detect(VFrameInfo& frame, FaceInfo** infos, PerfUtil* perf) {
#ifdef ENABLE_PERF
    _perf = perf;
#endif
    for (int i = 0; i < static_cast<int>(config::_s_face_max_count); ++i) {
        auto *face = infos[i];
        V_CHECK_CONT(static_cast<int>(face->_id) == 0); // 是否有人脸

        TensorArray prepared;
        TensorArray predicted;
        VA_CHECK_CONT(prepare(frame, &face, prepared) != 0);
        VA_CHECK_CONT(process(prepared, predicted) != 0);
        VA_CHECK_CONT(post(predicted, &face) != 0);
    }
    V_RET(Error::OK);
}

int FaceReconstructDetector::prepare(VFrameInfo& frame, FaceInfo** infos, TensorArray& prepared) {
    AUTO_PERF(_perf, std::string(TAG) + "-prepare");
    auto *face = *infos;
    CHECK_OR_MAKE_RGB(frame);

    // image rescale
    cv::Mat src_image = TensorConverter::convert_to<cv::Mat>(frame.rgb); // BGR
    auto max_size = MAX(frame.width, frame.height);
    float rescale = 1000.f / max_size;
    int new_size_w = static_cast<int>(frame.width * rescale);
    int new_size_h = static_cast<int>(frame.height * rescale);
    cv::Mat src_image_resized;
    if (max_size > 1000) {
        cv::resize(src_image, src_image_resized, {new_size_w, new_size_h});
    }
    _in_resized_h = src_image_resized.rows;
    _in_resized_w = src_image_resized.cols;
    VLOGD(TAG, "resized_h=%d, resized_w=%d", _in_resized_h, _in_resized_w);
    // convert to rgb
    cv::Mat image; // RGB 
    cv::cvtColor(src_image_resized, image, CV_BGR2RGB);

    _image = TensorConverter::convert_from<cv::Mat>(image, true);

    // int left = 351;
    // int top = 173;
    // int right = 644;
    // int bottom = 467;

    CLAMP(face->_rect_lt.x, 0, frame.width);
    CLAMP(face->_rect_lt.y, 0, frame.height);
    CLAMP(face->_rect_rb.x, 0, frame.width);
    CLAMP(face->_rect_rb.y, 0, frame.height);

    float coeff = 1.1f;
    int left = static_cast<int>(face->_rect_lt.x * rescale);
    int top = static_cast<int>(face->_rect_lt.y / coeff * rescale);
    int right = static_cast<int>(face->_rect_rb.x * rescale);
    int bottom = static_cast<int>(face->_rect_rb.y * rescale);
    V_CHECK_COND(!(right - left > 0 && bottom - top > 0),
                  Error::PREPARE_ERR, "crop face size invalid!");
    VLOGD(TAG, "left=%d, top=%d, right=%d, bottom=%d", left, top, right, bottom);              

    // get warp affine matrix              
    float old_size = (right - left + bottom - top) / 2.f;
    float center_x = right - (right - left) / 2.f;
    float center_y = bottom - (bottom - top) / 2.f + old_size * 0.14f;
    int size = static_cast<int>(old_size * 1.58f);
    VLOGD(TAG, "center=[%f, %f], size=%d", center_x, center_y, size);
    cv::Point2f srcPoint[3];
	cv::Point2f dstPoint[3];
	srcPoint[0] = cv::Point2f(center_x - size / 2.f, center_y - size / 2.f);
	srcPoint[1] = cv::Point2f(center_x - size / 2.f, center_y + size / 2.f);
	srcPoint[2] = cv::Point2f(center_x + size / 2.f, center_y - size / 2.f);
	dstPoint[0] = cv::Point2f(0, 0);
	dstPoint[1] = cv::Point2f(0, _k_input_height - 1);
	dstPoint[2] = cv::Point2f(_k_input_width - 1, 0);
	// cv::Mat warpMat(cv::Size(2, 3), CV_32F);
	auto warpMat = cv::getAffineTransform(srcPoint, dstPoint);
    auto* warp_data = (float*)_warp_matrix.data;
    warp_data[0] = warpMat.at<double>(0, 0);
    warp_data[1] = warpMat.at<double>(0, 1);
    warp_data[2] = warpMat.at<double>(0, 2);
    warp_data[3] = warpMat.at<double>(1, 0);
    warp_data[4] = warpMat.at<double>(1, 1);
    warp_data[5] = warpMat.at<double>(1, 2);
    warp_data[6] = 0.f;
    warp_data[7] = 0.f;
    warp_data[8] = 1.f;

    // crop face
    cv::Mat image_f;
    image.convertTo(image_f, CV_32FC3);
    image_f = image_f / 255.f;
    printf("image_f=\n");
    float* data = image_f.ptr<float>(0);
    for (int i = 0; i < 30; ++i) {
        printf("%f ", data[i]);
    }
    printf("\n");
    cv::Mat dst_image;
    cv::warpAffine(image_f, dst_image, warpMat, cv::Size(_k_input_width, _k_input_height));

    auto* dptr = warpMat.ptr<double>(0);
    for (int i = 0; i < 6; ++i) {
        printf("%f ", dptr[i]);
    }
    printf("\n");

    printf("dst_image=\n");
    data = dst_image.ptr<float>(0);
    for (int i = 0; i < 30; ++i) {
        printf("%f ", data[i]);
    }
    printf("\n");

    prepared.clear();
    prepared.emplace_back(TensorConverter::convert_from<cv::Mat>(dst_image, true));    
    
    VA_RET(Error::OK);
}

int FaceReconstructDetector::process(TensorArray& inputs, TensorArray& outputs) {
    AUTO_PERF(_perf, std::string(TAG) + "-process");
    MAKE_PREDICTOR_IF_INVALID(ModelId::VISION_TYPE_FACE_RECONSTRUCT);
    VA_CHECK_NULL_RET_MSG(_predictor, Error::PREDICTOR_NULL_ERR, "Prn predictor not registered!");
    VA_CHECK(_predictor->predict(inputs, outputs));
    VA_RET(Error::OK);
}

int FaceReconstructDetector::post(TensorArray& infer_results, FaceInfo** faces) {
    AUTO_PERF(_perf, std::string(TAG) + "-post");
    VLOGD(TAG, "prn output size: %d", static_cast<int>(infer_results.size()));
    V_CHECK_COND(infer_results.size() < 1, Error::INFER_ERR, "Prn infer results size error");
 
    auto& output = infer_results[0];
    _out_w = output.w;
    _out_h = output.h;
    _out_c = output.c;
    auto* out_data = (float *) output.data;
    for (size_t i = 0; i < output.size(); ++i) {
        out_data[i] *= (256.f * 1.1f);
    }

    // printf("\nprn infer results:\n");
    // for (auto i = 0; i < 30; ++i) {
    //     printf("%f ", out_data[i]);
    // } 
    // printf("\n");

    // change layout
    output.layout = NHWC;
    _cropped_vertices = output.change_layout(NCHW);
    auto* vert_z_data = (float*)_cropped_vertices.data + output.stride * 2;
    Tensor z(output.w, output.h, FP32);
    auto* z_data = (float*)z.data;
    float coeff = *(float*)_warp_matrix.data;
    for (auto i = 0; i < output.stride; ++i) {
        z_data[i] = vert_z_data[i] / coeff;
        vert_z_data[i] = 1.f;
    }

    // printf("\ncropped vertices:\n");
    // for (auto i = 0; i < 30; ++i) {
    //     printf("%f ", ((float*)_cropped_vertices.data)[i]);
    // } 
    // printf("\n");

    // printf("warp:\n");
    // for (auto i = 0; i < 9; ++i) {
    //     printf("%f ", ((float*)_warp_matrix.data)[i]);
    // } 
    // printf("\n");

    float inv_warp[9];
    if (!MathUtils::matrix_3x3_invert((float*)_warp_matrix.data, inv_warp)) {
        VLOGE(TAG, "warp matrix does not have inverse matrix!");
        VA_RET(Error::POST_ERR);
    }
    // printf("inv_warp:\n");
    // for (auto i = 0; i < 9; ++i) {
    //     printf("%f ", inv_warp[i]);
    // } 
    // printf("\n");

    // final pos map
    VLOGD(TAG, "output w=%d, h=%d, c=%d, stride=%d", output.w, output.h, output.c, output.stride);
    auto* vert_data = (float*)_vertices.data;
    auto* crop_vert_x = (float*)_cropped_vertices.data;
    auto* crop_vert_y = (float*)_cropped_vertices.data + output.stride;
    auto* crop_vert_z = (float*)_cropped_vertices.data + output.stride * 2;
    for (auto i = 0; i < output.stride; ++i) {
        vert_data[3 * i] = crop_vert_x[i] * inv_warp[0] + crop_vert_y[i] * inv_warp[1] + crop_vert_z[i] * inv_warp[2];
        vert_data[3 * i + 1] = crop_vert_x[i] * inv_warp[3] + crop_vert_y[i] * inv_warp[4] + crop_vert_z[i] * inv_warp[5];
        vert_data[3 * i + 2] = z_data[i];
    }

    // printf("\nvertices:\n");
    // for (auto i = 0; i < 30; ++i) {
    //     printf("%f ", vert_data[i]);
    // } 
    // printf("\n");

    VA_RET(Error::OK);
}

Tensor FaceReconstructDetector::get_vertices() const {
    if (_face_ind.empty()) {
        VLOGE(TAG, "face indices is empty, which should be set firstly!");
        return _vertices;
    }
    VLOGD(TAG, "face_ind_stride=%d", _face_ind.stride);
    Tensor face_vert(_face_ind.stride, 1, 3, FP32, NHWC);
    auto* vert_data = static_cast<float*>(_vertices.data);
    auto* face_vert_data = static_cast<float*>(face_vert.data);
    auto* ind_data = static_cast<float*>(_face_ind.data);
    for (int i = 0; i < _face_ind.stride; ++i) {
        auto idx = static_cast<int>(ind_data[i]);
        face_vert_data[3 * i] = vert_data[3 * idx];
        face_vert_data[3 * i + 1] = vert_data[3 * idx + 1];
        face_vert_data[3 * i + 2] = vert_data[3 * idx + 2];
    }

    return face_vert;
}

Tensor FaceReconstructDetector::get_colors(const Tensor& vertices) {
    VLOGD(TAG, "get color");
    Tensor colors{vertices.w, vertices.h, vertices.c, FP32, NHWC};
    auto* data = static_cast<float*>(vertices.data);
    auto* color_data = static_cast<float*>(colors.data);
    auto* img_data = static_cast<char*>(_image.data);

    VLOGD(TAG, "vertices stride=%d", vertices.stride);
    for (int i = 0; i < vertices.stride; ++i) {
        int x = static_cast<int>(MIN(MAX(data[3 * i], 0), _in_resized_w - 1));
        int y = static_cast<int>(MIN(MAX(data[3 * i + 1], 0), _in_resized_h - 1));
        int idx = y * _in_resized_w + x;
        color_data[3 * i] = img_data[3 * idx] / 255.f;
        color_data[3 * i + 1] = img_data[3 * idx + 1] / 255.f;
        color_data[3 * i + 2] = img_data[3 * idx + 2] / 255.f;
    }
    printf("\n");

    printf("get colors:\n");
    for (auto i = 0; i < 30; ++i) {
        printf("%f ", color_data[i]);
    } 
    printf("\n");

    return colors;
}

void FaceReconstructDetector::generate_texture() {
    VLOGD(TAG, "get texture");
    cv::Mat img_mat = TensorConverter::convert_to<cv::Mat>(_image);
    cv::Mat map_x(_vertices.w, _vertices.h, CV_32FC1);
    cv::Mat map_y(_vertices.w, _vertices.h, CV_32FC1);
    float* mx_data = (float*)map_x.data;
    float* my_data = (float*)map_y.data;
    float* vert_data = (float*)_vertices.data;
    VLOGD(TAG, "get_texture, w=%d, h=%d", _vertices.w, _vertices.h);
    VLOGD(TAG, "get_texture, src_mat w=%d, h=%d", img_mat.cols, img_mat.rows);
    for (auto i = 0; i < _vertices.stride; ++i) {
        mx_data[i] = vert_data[3 * i];
        my_data[i] = vert_data[3 * i + 1];
    }
    cv::Mat tex_mat;
    cv::cvtColor(img_mat, img_mat, CV_RGB2BGR);
    cv::remap(img_mat, tex_mat, map_x, map_y, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar{0, 0, 0});
    VLOGD(TAG, "get_texture, dst_mat w=%d, h=%d", tex_mat.cols, tex_mat.rows);
#ifdef WITH_OCV_HIGHGUI
    cv::imwrite(_texture_path + "/3dface_texture.png", tex_mat);
#endif
}

Tensor FaceReconstructDetector::generate_uv_coords() {
    if (_face_ind.empty()) {
        VLOGE(TAG, "face indices is empty, which should be set firstly!");
        return Tensor{};
    }

    Tensor face_uv(_face_ind.stride, 1, 3, FP32, NHWC);
    float* uv_data = static_cast<float*>(_uv_coords.data);
    auto* face_uv_data = static_cast<float*>(face_uv.data);
    auto* ind_data = static_cast<float*>(_face_ind.data);
    VLOGD(TAG, "face_ind stride=%d", _face_ind.stride);
    for (int i = 0; i < _face_ind.stride; ++i) {
        auto idx = static_cast<int>(ind_data[i]);
        face_uv_data[3 * i] = uv_data[2 * idx] / 256.f;
        face_uv_data[3 * i + 1] = uv_data[2 * idx + 1] / 256.f;
        face_uv_data[3 * i + 2] = 0;
    }

    printf("uv_coords:\n");
    for (auto i = 0; i < 30; ++i) {
        printf("%f ", uv_data[i]);
    } 
    printf("\n");

    printf("face_uv_data:\n");
    for (auto i = 0; i < 30; ++i) {
        printf("%f ", face_uv_data[i]);
    } 
    printf("\n");

    return face_uv;
}

void FaceReconstructDetector::set_face_ind(const Tensor& face_ind) {
    _face_ind = face_ind;
}

void FaceReconstructDetector::set_texture_path(const std::string& path) {
    _texture_path = path;
}

void FaceReconstructDetector::init_params() {
    _warp_matrix.create(3, 3, FP32, NCHW);
    _vertices.create(_k_input_width, _k_input_height, 3, FP32, NHWC);
    _uv_coords.create(_k_input_width, _k_input_height, 2, FP32, NHWC);

    float* uv_data = static_cast<float*>(_uv_coords.data);
    for (int i = 0; i < _k_input_height; ++i) {
        for (int j = 0; j < _k_input_width; ++j) {
            int idx = i * _k_input_width + j;
            uv_data[2 * idx] = j;
            uv_data[2 * idx + 1] = i;
        }
    }
}

} // namespace vision