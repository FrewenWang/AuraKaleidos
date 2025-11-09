

#ifndef AURA_RUNTIME_NN_NB_MODEL_HPP__
#define AURA_RUNTIME_NN_NB_MODEL_HPP__

#include "aura/runtime/nn/nn_engine.hpp"
#include "aura/runtime/core.h"
#include "aura/runtime/memory.h"
#include "aura/runtime/array.h"

#define AURA_NB_MODEL_MAGIC     0x4D694E42      // MiNB

namespace aura
{

#pragma pack(push, 1)
struct MinbHeader
{
    DT_S32 magic_num    = 0;
    struct
    {
        DT_S16 major    = 0;
        DT_S16 minor    = 0;
    } version;
    DT_U32 packinfo_len = 0;
};

struct MinbDate
{
    DT_U16 year   = 0;
    DT_U16 month  = 0;
    DT_U16 day    = 0;
    DT_U16 hour   = 0;
    DT_U16 minute = 0;
    DT_U16 second = 0;
};

struct ContainerVersion
{
    DT_S16 major = 0;
    DT_S16 minor = 0;
};
#pragma pack(pop)

class AURA_EXPORTS NBModel
{
public:
    NBModel(Context *ctx, const std::string &minb_file);
    ~NBModel();

    AURA_DISABLE_COPY_AND_ASSIGN(NBModel);

    DT_U32 GetMinbVersion() const;
    std::string GetDate() const;
    std::string GetAuthor() const;
    std::string GetDescription() const;
    std::string GetVersion() const;
    std::vector<std::string> GetMinnModelNames() const;
    Buffer GetModelBuffer(const std::string minn_model_name);
    std::string GetMinnModelInfo(const std::string minn_model_name, const std::string attr = "all");
    DT_BOOL IsValid() const;
    DT_VOID Release();

private:
    Context         *m_ctx;
    std::string      m_minb_file;
    DT_S64           m_minb_file_offset;
    DT_U32           m_minb_version;
    DT_BOOL          m_is_valid;
    Buffer           m_minb_buffer;
    MinbDate         m_minb_date;
    std::string      m_author;
    std::string      m_description;
    ContainerVersion m_container_version;
    DT_U16           m_minn_num;

    struct MinnPackInfo
    {
        std::string minn_model_name = "";
        DT_U64      minn_model_size = 0;
    };
    std::vector<MinnPackInfo> m_minn_infos;
};

}// namespace aura

#endif // AURA_RUNTIME_NN_NB_MODEL_HPP__