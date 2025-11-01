#ifndef AURA_RUNTIME_NN_XNN_MODEL_HPP__
#define AURA_RUNTIME_NN_XNN_MODEL_HPP__

#include "nn_model.hpp"

namespace aura
{

class XnnModel : public NNModel
{
public:
    XnnModel(Context *ctx, const ModelInfo &model_info);
};

}// namespace aura

#endif // AURA_RUNTIME_NN_XNN_MODEL_HPP__