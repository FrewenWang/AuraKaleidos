

#ifndef AURA_RUNTIME_NN_SNPE_MODEL_HPP__
#define AURA_RUNTIME_NN_SNPE_MODEL_HPP__

#include "nn_model.hpp"

namespace aura
{

class SnpeModel : public NNModel
{
public:
    SnpeModel(Context *ctx, const ModelInfo &model_info);

    const std::vector<std::string>& GetOutputLayerNames() const;

private:
    std::vector<std::string> m_output_layer_names;
};

}// namespace aura

#endif // AURA_RUNTIME_NN_SNPE_MODEL_HPP__