#include "model_info.h"

#include <sstream>
#include <string>
#include "vision/core/common/VMacro.h"
#include "util/InferenceConverter.hpp"
#include "vision/util/VaAllocator.h"

using namespace std;

namespace aura::vision{

const std::string ModelMagic::_str_magic = "bdiovvision"; // NOLINT

std::string ModelMagic::magic() {
    static std::string hex_magic = str_to_hex(_str_magic);
    return hex_magic;
}

std::string ModelMagic::str_to_hex(const std::string& str) {
    const std::string hex = "0123456789ABCDEF";

    string ss("");
    for (auto& s : str) {
        ss += (hex[(unsigned char)s >> (unsigned char)4]);
        ss += (hex[(unsigned char)s & (unsigned char)0xf]);
    }
    return ss;

//    std::stringstream ss;

//    for (auto& s : str) {
//        ss << hex[(unsigned char)s >> (unsigned char)4]
//           << hex[(unsigned char)s & (unsigned char)0xf];
//    }
//    return ss.str();
}

std::string ModelInfo::to_string(bool verbose) const {
    string ss("");
    ss += "ModelName:";
    ss += InferenceConverter::get_tag<ModelId>(id);
    ss += ",Version:";
    ss += version;
    if (id == VISION_TYPE_FACE_FEATURE) {
        ss += ",versionCode:";
        ss += std::to_string(versionCode);
        ss += ",extends:";
        ss += extends;
    }

    if (verbose) {
        ss += ",Dtype:";
        ss += InferenceConverter::get_tag<DType>(dtype);
        ss += ",InferType:";
        ss += std::to_string(V_TO_INT(infer_type));
        ss += ",BlobsSize:";
        ss += std::to_string(blobs.size());
    }

    ss += "\n";
    return ss;

//    std::stringstream ss;
//    ss << "ModelName:" << TagIdConverter::get_tag<ModelId>(id)
//       << ", Version:" << version;
//
//    if (verbose) {
//        ss << ", Dtype:" << TagIdConverter::get_tag<DType>(dtype)
//           << ", InferType:" << V_TO_INT(infer_type)
//           << ", BlobsSize:" << blobs.size();
//    }
//
//    ss << std::endl;
//    return ss.str();
}

void ModelInfo::release() {
    for (auto &blob : blobs) {
        if (blob.data != nullptr) {
            VaAllocator::deallocate(blob.data);
        }
        blob.len = 0;
        blob.data = nullptr;
    }
}

} // namespace