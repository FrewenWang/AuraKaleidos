

#ifndef AURA_RUNTIME_NN_NN_MODEL_HPP__
#define AURA_RUNTIME_NN_NN_MODEL_HPP__

#include "aura/runtime/nn/nn_engine.hpp"
#include "aura/runtime/core.h"
#include "aura/runtime/memory.h"
#include "aura/runtime/array.h"

#define AURA_NN_MODEL_MAGIC             0x4D694E4E      // MiNN
#define AURA_NN_MODEL_ENCRYPT_LENGTH    (1024 * 1024)   // 1MB

namespace aura
{

#pragma pack(push, 1)
struct MinnHeader
{
    MI_S32 magic_num = 0;
    struct
    {
        MI_S16 major = 0;
        MI_S16 minor = 0;
    } version;
};

struct MinnDataV1
{
    MI_S16 framework    = 0;
    struct
    {
        MI_S16 major    = 0;
        MI_S16 minor    = 0;
        MI_S16 patch    = 0;
    } framework_version;
    struct
    {
        MI_S16 major    = 0;
        MI_S16 minor    = 0;
    } model_version;
    MI_S16 backend_type = 0;
};
#pragma pack(pop)

struct ModelInfo
{
    ModelInfo()
    {}

    ModelInfo(const Buffer &minn_buffer, const std::string &decrypt_key, const std::string &minn_file_name = std::string())
              : minn_buffer(minn_buffer), decrypt_key(decrypt_key), minn_file_name(minn_file_name)
    {}

    Buffer minn_buffer;
    std::string decrypt_key;
    std::string minn_file_name;
};

class NNModel
{
public:
    NNModel(Context *ctx, const ModelInfo &model_info);
    virtual ~NNModel();

    AURA_DISABLE_COPY_AND_ASSIGN(NNModel);

    MI_U32 GetMinnVersion() const;
    std::string GetFrameWorkVersion() const;
    std::string GetModelVersion() const;
    std::string GetBackendType() const;
    std::string GetVersion() const;
    std::string GetModelName() const;
    Buffer GetModelBuffer();
    AURA_VOID ReleaseModelBuffer();
    MI_BOOL IsValid() const;
    MatMap MapMatNames(const MatMap &mat_map, MI_BOOL is_input) const;
    TensorDescMap MapTensorDescNames(const TensorDescMap &tensor_desc_map, MI_BOOL is_input) const;

    static Buffer CreateModelBufferFromFile(Context *ctx, const std::string &minn_file);
#if defined(AURA_BUILD_HOST)
    static Buffer MapModelBufferFromFile(Context *ctx, const std::string &minn_file);
#endif // AURA_BUILD_HOST

protected:
    Context     *m_ctx;
    MI_U32      m_minn_version;
    std::string m_framework_version;
    std::string m_model_version;
    std::string m_backend_type;
    MI_S64      m_data_offset;
    Buffer      m_model_buffer;
    ModelInfo   m_model_info;
    MI_BOOL     m_is_valid;
    std::unordered_map<std::string, std::string> m_input_names_map;
    std::unordered_map<std::string, std::string> m_output_names_map;
};

}// namespace aura

#endif // AURA_RUNTIME_NN_NN_MODEL_HPP__