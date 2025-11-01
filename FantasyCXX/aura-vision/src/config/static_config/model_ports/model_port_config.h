#ifndef VISION_MODEL_PORT_CONFIG_H
#define VISION_MODEL_PORT_CONFIG_H

#include <memory>
#include <string>

#include "proto/model_port.plain.h"

namespace aura::vision{

class ModelPortConfig {
public:
    static std::shared_ptr<ModelPort> get_config();

private:
    static std::string _model_port_cfg_str;
};

} // namespace aura::vision

#endif //VISION_MODEL_PORT_CONFIG_H
