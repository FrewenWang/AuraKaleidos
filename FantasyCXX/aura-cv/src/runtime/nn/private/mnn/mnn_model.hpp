#ifndef AURA_RUNTIME_NN_MNN_MODEL_HPP__
#define AURA_RUNTIME_NN_MNN_MODEL_HPP__

#include "nn_model.hpp"

namespace aura
{

enum class MnnPrecision
{
    PRECISION_NORMAL = 0,       /*!< Normal precision level */
    PRECISION_HIGH,             /*!< High precision level */
    PRECISION_LOW               /*!< Low precision level */
};

enum class MnnMemory
{
    MEMORY_NORMAL = 0,          /*!< Normal memory usage level */
    MEMORY_HIGH,                /*!< High memory usage level */
    MEMORY_LOW                  /*!< Low memory usage level */
};

enum class MnnTuning
{
    GPU_TUNING_NONE   = 1 << 0, /*!< No GPU tuning */
    GPU_TUNING_HEAVY  = 1 << 1, /*!< Heavy GPU tuning */
    GPU_TUNING_WIDE   = 1 << 2, /*!< Wide GPU tuning */
    GPU_TUNING_NORMAL = 1 << 3, /*!< Normal GPU tuning */
    GPU_TUNING_FAST   = 1 << 4  /*!< Fast GPU tuning */
};

enum class MnnCLMem
{
    GPU_MEMORY_NONE   = 0,      /*!< No GPU memory type */
    GPU_MEMORY_BUFFER = 1 << 6, /*!< GPU memory type: buffer */
    GPU_MEMORY_IAURA  = 1 << 7  /*!< GPU memory type: iaura */
};

class MnnModel : public NNModel
{
public:
    MnnModel(Context *ctx, const ModelInfo &model_info);

    MnnPrecision GetPrecision() const;
    MnnMemory GetMemory() const;
    MnnTuning GetTuning() const;
    MnnCLMem GetCLMem() const;

private:
    enum MnnPrecision m_precision;
    enum MnnMemory    m_memory;
    enum MnnTuning    m_tuning;
    enum MnnCLMem     m_cl_mem;
};

}// namespace aura

#endif // AURA_RUNTIME_NN_MNN_MODEL_HPP__