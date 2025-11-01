
#ifndef AURA_RUNTIME_NN_QNN_MODEL_HPP__
#define AURA_RUNTIME_NN_QNN_MODEL_HPP__

#include "nn_model.hpp"

namespace aura
{

class QnnModel : public NNModel
{
public:
    QnnModel(Context *ctx, const ModelInfo &model_info);
};

}// namespace aura

#endif // AURA_RUNTIME_NN_QNN_MODEL_HPP__