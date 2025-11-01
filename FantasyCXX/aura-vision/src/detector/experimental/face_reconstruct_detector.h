#ifndef VISION_PRN_DETECTOR_H
#define VISION_PRN_DETECTOR_H

#include "detector/AbsDetector.h"
#include "vision/core/bean/FaceInfo.h"

namespace vision {

class FaceReconstructDetector : public AbsDetector<FaceInfo> {
public:
    FaceReconstructDetector();
    ~FaceReconstructDetector() override;

    int detect(VFrameInfo& frame, FaceInfo** infos, PerfUtil* perf) override;
    void set_face_ind(const Tensor& face_ind);
    Tensor get_vertices() const;
    Tensor get_colors(const Tensor& vertices);
    void generate_texture();
    Tensor generate_uv_coords();
    void set_texture_path(const std::string& path);

protected:
    int prepare(VFrameInfo& frame, FaceInfo** infos, TensorArray& prepared) override;
    int process(TensorArray& inputs, TensorArray& outputs) override;
    int post(TensorArray& infer_results, FaceInfo** infos) override;

private:
    void init_params();

    Tensor _cropped_vertices;
    Tensor _vertices;
    Tensor _pos_map;
    Tensor _warp_matrix;
    Tensor _image;
    Tensor _face_ind;
    Tensor _texture;
    Tensor _uv_coords;

    int _out_w;
    int _out_h;
    int _out_c;

    int _in_resized_h;
    int _in_resized_w;

    std::string _texture_path;

    const int _k_input_width = 256;
    const int _k_input_height = 256;
};

} // namespace vision

#endif //VISION_PRN_DETECTOR_H
