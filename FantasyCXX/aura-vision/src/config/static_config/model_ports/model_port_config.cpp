#include "model_port_config.h"
#include "plainbuffer/text_format.h"
#include "vision/util/log.h"

namespace aura::vision{

static const char* TAG = "ModelPortConfig";

// NOLINTNEXTLINE
std::string ModelPortConfig::_model_port_cfg_str = {
#include MODEL_PORT_PROTOTXT_IN_PATH
};

std::shared_ptr<ModelPort> ModelPortConfig::get_config() {
    auto port = std::make_shared<ModelPort>();
    auto ret = plainbuffer::TextFormat::ParseFromString(_model_port_cfg_str, port.get());
    if (!ret) {
        VLOGE(TAG, "Parse runtime config FAILED!");
        return nullptr;
    }
    return port;
}

} // namespace aura::vision