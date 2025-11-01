#ifndef VISION_VISION_INITIALIZER_H
#define VISION_VISION_INITIALIZER_H

#include <memory>
#include <string>
#include "vision/core/common/VMacro.h"

namespace aura::vision {

/**
 * @brief Model loader & initializer
 * VisionInitializer will load all the computation resources,
 * this should be called before invocation of the VisionService APIs
 */
class VA_PUBLIC VisionInitializer {
public:
    /**
     * @brief Init from internal memory
     * This is easy and direct, no extra input is needed,
     * model resources have been wrapped into the shared library
     * @return init error (0 for success)
     */
    int init(std::string modelPath = "./libvision_model.bin");

    /**
     * @brief Init from other memory address
     * The start address and memory length are needed,
     * this is typically for the case that the models are loaded by client app
     * @param mem start address of the memory
     * @param len total length of the memory
     * @return init error (0 for success)
     */
    int initFromMemory(const void* mem, int len);

    /**
     * @brief Init from extra file
     * @param file file name of the model
     * @return init error (0 for success) 
     */
    int initFromFile(const char* file);

    /**
     * @brief release the model resouces
     * @return init error (0 for success) 
     */
    int deinit();

    /**
     * @brief get model versions
     * @return model versions string
     */
    std::string getModelInfo();

    /**
     * @brief 设置进行调试日志打印的相关配置
     *
     * @param tag   日志打印的 Tag
     * @param level 日志打印的等级
     * @param dest  日志输出目的
     * @param fileDir  日志输出目的
     */
    static void setDebugConfig(const std::string &tag, const LogLevel &level,
                               const LogDest &dest, const std::string &fileDir = "");
    /**
     * 获取SDK版本号
     * @return
     */
    static std::string getVersion();
    /**
     * 设置给推理器的指令，具体指令参见：
     * @see VConstant::InferenceCmd
     * @return
     */
    static bool setInferenceCmd(int source, int cmd);

public:
    VisionInitializer();
    ~VisionInitializer();

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace vision

#endif //VISION_VISION_INITIALIZER_H
