#include "vision/initializer/VisionInitializer.h"
#include "vision/core/common/VConstants.h"
#include "vision/core/common/VMacro.h"
#include "vision/core/version.h"
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include "sys/mman.h"

/// 要生成build_config.h需要使用cmake的configure_file的函数进行
#include "build_config.h"
#include "config/static_config/model_ports/model_port_config.h"
#include "inference/InferenceRegistry.h"
#include "model_ir/internal_blob_config.h"
#include "model_ir/model_info.h"
#include "model_ir/model_parser.h"
#include "proto/model_port.plain.h"
#include "util/InferenceConverter.hpp"

#ifdef BUILD_FASTCV
#include "fastcv.h"
#endif

#ifdef BUILD_IOS
#include "vision/manager/VisionManagerRegistry.h"
#include "vision/manager/AbsVisionManager.h"
#include "manager/ImageBrightnessManager.h"
#include "manager/FaceRectManager.h"
#include "manager/FaceLandmarkManager.h"
#include "manager/FaceQualityManager.h"
#include "manager/FaceInteractLivingManager.h"
#include "manager/FaceFeatureManager.h"
#endif

using namespace std;

namespace aura::vision {

static const char *TAG = "VisionInitializer";
static bool gFastCVInitedFlag = false;

class VisionInitializer::Impl {
public:
    Impl();

    /**
     * 初始化
     * @param modelPath  模型路径
     * @return
     */
    int init(const string &modelPath);

    /**
     * 使用Model的Buffer对象初始化
     * @param mem
     * @param len
     * @return
     */
    int initFromMemory(const void *mem, int len);

    /**
     * 使用Model的文件路径进行初始化
     * @param file
     * @return
     */
    int initFromFile(const char *file);

    /**
     * 反初始化
     * @return
     */
    int deinit();

    /**
     * 获取加载的模型信息
     * 用于业务层判断使用的模型的信息
     * @return
     */
    std::string getModelInfo();

private:
    int register_predictor(ModelInfo &model);

    std::unique_ptr<ModelParser> modelParser;
    std::shared_ptr<ModelPort> modelPortCfg;
    std::unordered_map<short, PortDesc> modelPortDescMap;
    std::vector<ModelInfo> models;
};

VisionInitializer::Impl::Impl() {
    modelParser = std::unique_ptr<ModelParser>(new ModelParser());
    modelPortCfg = ModelPortConfig::get_config();

    // construct model port_desc map from runtime config
    for (const auto &desc: modelPortCfg->model_port()) {
        const auto &tag = desc.ability();
        modelPortDescMap[InferenceConverter::getId<ModelId>(tag)] = desc;
    }
}

int VisionInitializer::Impl::init(const string &modelPath) {
    int ret = V_TO_INT(Error::INIT_FAILURE);
    VLOGI(TAG, "%s modelPath[%s]", VisionInitializer::getVersion().c_str(), modelPath.c_str());
    void *model_blob = nullptr;

#if !USE_EXTERNAL_MODEL
    // Read model from internal
    model_blob = static_cast<void *>(model_resource::_g_internal_model_blob + model_resource::INTERNAL_MODEL_OFFSET);
    V_CHECK_NULL_RET_INFO(model_blob, Error::MODEL_INIT_ERR, "No model data found!");
#if MODEL_LEN
    ret = initFromMemory(model_blob, MODEL_LEN);
    if (ret != 0) {
        return ret;
    }
#else
    V_RET(Error::MODEL_INIT_ERR);
#endif
#else
    return initFromFile(modelPath.c_str()); // Read model from external
#endif
    return 0;
}

int VisionInitializer::Impl::initFromMemory(const void *mem, int len) {
    int ret = V_TO_INT(Error::INIT_FAILURE);
    // parse from memory
    V_CHECK_MSG(modelParser->parse(mem, len, models), "Parse model from memory FAILED!");

    int valid_cnt = 0;
    for (auto &model: models) {
        // search model port config
        if (modelPortDescMap.find(model.id) != modelPortDescMap.end()) {
            model.port_desc = modelPortDescMap[model.id];
        }
        VLOGD(TAG, "model version: %s", model.version.c_str());

        // register predictor
        ret = register_predictor(model);
        if (ret != 0) {
            VLOGE(TAG, "Init predictor with model_id=%s, infer_type=%d FAILED! ret=%d",
                  InferenceConverter::get_tag(model.id).c_str(), model.infer_type, ret);
            continue;
        }
        valid_cnt++;
    }

    // release model memory : QNN 的模型会加载到后端，此处的 model 可以 release 掉，其他推理库需要根据情况判断
    for (auto &model: models) {
        if (model.infer_type == InferType::QNN) {
            model.release();
        }
    }

    VLOGW(TAG, "Register %d models successfully!", valid_cnt);
    if (valid_cnt < static_cast<int>(models.size())) {
        VLOGE(TAG, "Register %d models FAILED! init failed!!!", V_TO_INT(models.size()) - valid_cnt);
        V_RET(Error::MODEL_INIT_ERR);
    }

#ifdef BUILD_IOS
    // iOS 平台下主动调用 mananger register
    VisionManagerRegistry::registerVisionManagerCreator("ImageBrightnessManager", static_cast<int>(ABILITY_IMAGE_BRIGHTNESS), []() {
        return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<ImageBrightnessManager>());
    });
    VisionManagerRegistry::registerVisionManagerCreator("FaceRectManager", static_cast<int>(ABILITY_FACE_RECT), []() {
        return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceRectManager>());
    });

    VisionManagerRegistry::registerVisionManagerCreator("FaceLandmarkManager", static_cast<int>(ABILITY_FACE_LANDMARK), []() {
        return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceLandmarkManager>());
    });

    VisionManagerRegistry::registerVisionManagerCreator("FaceQualityManager", static_cast<int>(ABILITY_FACE_QUALITY), []() {
        return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceQualityManager>());
    });

    VisionManagerRegistry::registerVisionManagerCreator("FaceInteractLivingManager", static_cast<int>(ABILITY_FACE_INTERACTIVE_LIVING), []() {
        return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceInteractLivingManager>());
    });

    VisionManagerRegistry::registerVisionManagerCreator("FaceFeatureManager", static_cast<int>(ABILITY_FACE_FEATURE), []() {
        return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceFeatureManager>());
    });
#endif

// 启用 FastCV 时需要初始化模式和内存
#ifdef BUILD_FASTCV
        if (!gFastCVInitedFlag) {
            // @see fastcv.h -> fcvOperationMode
            ret = fcvSetOperationMode(FASTCV_OP_CPU_OFFLOAD);
            if (ret != 0) {
                VLOGE(TAG, "Fail to fcvSetOperationMode, return code is %d", ret);
                V_RET(Error::INIT_FAILURE_FASTCV);
            }
            fcvMemInit();
            // get fastcv version
            char version[32];
            fcvGetVersion(version, 32);
            VLOGW(TAG, "fcv version: %s", version);
            gFastCVInitedFlag = true;
        } else {
            VLOGI(TAG, "fcv has inited !!!!");
        }
#endif
    V_RET(Error::OK);
}

int VisionInitializer::Impl::initFromFile(const char *file) {
    VLOGD(TAG, "init model from file");
#ifdef BUILD_LINUX
    // 以内存映射方式读取文件，linux上耗时接近于0
    int fd = open(file, O_RDONLY);
    V_CHECK_COND(fd == -1, Error::MODEL_INIT_ERR, "open model file failed!");
    auto len = (int) lseek(fd, 0, SEEK_END);
    auto addr = (char *) mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    auto ret = initFromMemory(addr, len);
    munmap(addr, len);
#else
    std::FILE *modelFile = std::fopen(file, "rb");
    if (modelFile == nullptr)
    V_CHECK_COND(modelFile == nullptr, Error::MODEL_INIT_ERR, "open model file failed!");

    std::fseek(modelFile, 0, SEEK_END);
    auto size = (int) std::ftell(modelFile);
    std::rewind(modelFile);

    auto buffer = new char[size];
    std::fread(buffer, 1, size, modelFile);
    std::fclose(modelFile);
    auto ret = initFromMemory(buffer, size);
    delete []buffer;
#endif

    V_RET(ret);
}

int VisionInitializer::Impl::deinit() {
    // clear predictors
    InferRegistry::clear();

    // release model memory
    for (auto &model: models) {
        model.release();
    }
    models.clear();

#ifdef BUILD_FASTCV
    // FastCV 一次进程只初始化一次。不进行反初始化
    // fcvMemDeInit();
    // fcvCleanUp();
#endif

    V_RET(Error::OK);
}

int VisionInitializer::Impl::register_predictor(vision::ModelInfo &model) {
    // 初始化Source1的Predictor
    auto predictor1 = InferenceFactory::create_predictor(model);
    V_CHECK_NULL_RET_ERR(predictor1, Error::PREDICTOR_NULL_ERR, "create_predictor1 NULL!!!");
    InferRegistry::insert(SOURCE_1, model.id, std::move(predictor1));
#if !defined(BUILD_ANDROID) and !defined(BUILD_IOS)
    // 初始化Source2的Predictor
    auto predictor2 = InferenceFactory::create_predictor(model);
    V_CHECK_NULL_RET_ERR(predictor2, Error::PREDICTOR_NULL_ERR, "create_predictor2 NULL!!!");
    InferRegistry::insert(SOURCE_2, model.id, std::move(predictor2));
#endif
    V_RET(Error::OK);
}

std::string VisionInitializer::Impl::getModelInfo() {
    if (models.empty()) {
        return "model is empty";
    }

    std::string model_info;
    for (auto &model: models) {
        model_info += model.to_string();
    }
    return model_info;
}

VisionInitializer::VisionInitializer() {
//    _impl = std::make_unique<VisionInitializer::Impl>();
    impl = std::unique_ptr<VisionInitializer::Impl>(new VisionInitializer::Impl());
}

VisionInitializer::~VisionInitializer() = default;

int VisionInitializer::init(string modelPath) {
    return impl->init(modelPath);
}

int VisionInitializer::initFromMemory(const void *mem, int len) {
    return impl->initFromMemory(mem, len);
}

int VisionInitializer::initFromFile(const char *file) {
    return impl->initFromFile(file);
}

int VisionInitializer::deinit() {
    auto res = impl->deinit();
    impl.reset();
    return res;
}

std::string VisionInitializer::getModelInfo() {
    return impl->getModelInfo();
}

void VisionInitializer::setDebugConfig(const string &tag, const LogLevel &level,
                                       const LogDest &dest, const string &fileDir) {
    // 设置AppName会同步修改
    Logger::setAppName(tag);
    Logger::setLogLevel(level);
    Logger::setLogDest(dest);
    Logger::setLogFileDir(fileDir);
}

std::string VisionInitializer::getVersion() {
    return VisionVersion::getVersion();
}

bool VisionInitializer::setInferenceCmd(int source, int cmd) {
    return InferRegistry::setInferenceCmd(source, cmd);
}

} // namespace aura::vision

