#ifndef AURA_RUNTIME_NN_NP_MODEL_HPP__
#define AURA_RUNTIME_NN_NP_MODEL_HPP__

#include "nn_model.hpp"

namespace aura
{

class NpModel : public NNModel
{
public:
    struct TensorAttr
    {
        MI_S32 zero_point;
        MI_F32 scale;
        ElemType elem_type;
        std::vector<MI_S32> sizes;
    };

    NpModel(Context *ctx, const ModelInfo &model_info);

    const std::vector<std::pair<std::string, TensorAttr>>& GetInputAttrs() const;
    const std::vector<std::pair<std::string, TensorAttr>>& GetOutputAttrs() const;

private:
    std::vector<std::pair<std::string, TensorAttr>> m_input_attrs;
    std::vector<std::pair<std::string, TensorAttr>> m_output_attrs;
};

}// namespace aura

#endif // AURA_RUNTIME_NN_NP_MODEL_HPP__