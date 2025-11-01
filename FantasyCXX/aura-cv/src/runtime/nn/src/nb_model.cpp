#include "nb_model.hpp"
#include "nn_executor_impl.hpp"

namespace aura
{

NBModel::NBModel(Context *ctx, const std::string &minb_file) : m_ctx(ctx), m_minb_file(minb_file), m_minb_file_offset(0), m_minb_version(0), m_is_valid(MI_FALSE), m_minn_num(0)
{
    FILE *fp = MI_NULL;

    do
    {
        if (MI_NULL == m_ctx)
        {
            break;
        }

        MinbHeader minb_header;
        size_t header_bytes = 0;

        fp = fopen(m_minb_file.c_str(), "rb");
        if (MI_NULL == fp)
        {
            std::string info = "open model container: " + m_minb_file + " failed";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            break;
        }

        header_bytes = fread(&minb_header, 1, sizeof(minb_header), fp);
        if (header_bytes != sizeof(minb_header))
        {
            std::string info = "minb_file " +  m_minb_file + " need fread " + std::to_string(sizeof(minb_header)) + ", but actual only fread " + std::to_string(header_bytes);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            break;
        }

        m_minb_file_offset += sizeof(minb_header);

        if (minb_header.magic_num != AURA_NB_MODEL_MAGIC)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "minb model header check failed");
            break;
        }

        m_minb_version = (minb_header.version.major << 16) | minb_header.version.minor;
        if (0x010000 == m_minb_version)
        {
            size_t packinfo_bytes                  = 0;
            MI_BOOL minn_infos_valid               = MI_TRUE;
            MI_S64 minn_packinfo_offset            = 0;
            std::string minb_packinfo_deccrypt_key = "MiNB";

            m_minb_buffer = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(m_ctx, AURA_MEM_HEAP, minb_header.packinfo_len, 0));
            if (!m_minb_buffer.IsValid())
            {
                std::string info = "m_minb_buffer alloc " + std::to_string(minb_header.packinfo_len) + " failed";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                break;
            }

            packinfo_bytes = fread(static_cast<MI_U8*>(m_minb_buffer.m_data), 1, minb_header.packinfo_len, fp);
            if (static_cast<MI_U32>(packinfo_bytes) != minb_header.packinfo_len)
            {
                std::string info = "file fread size(" + std::to_string(packinfo_bytes) + "," + std::to_string(minb_header.packinfo_len) + ") not match";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                break;
            }

            if (NNDecrypt(m_ctx, m_minb_buffer, m_minb_buffer, minb_packinfo_deccrypt_key, m_minb_buffer.m_size) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NNDecrypt m_minb_buffer failed");
                break;
            }

            if (NNDeserialize(m_ctx, m_minb_buffer, minn_packinfo_offset, m_minb_date) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NNDeserialize m_minb_date failed");
                break;
            }

            if (NNDeserialize(m_ctx, m_minb_buffer, minn_packinfo_offset, m_author) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NNDeserialize m_author failed");
                break;
            }

            if (NNDeserialize(m_ctx, m_minb_buffer, minn_packinfo_offset, m_description) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NNDeserialize m_description failed");
                break;
            }

            if (NNDeserialize(m_ctx, m_minb_buffer, minn_packinfo_offset, m_container_version) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NNDeserialize m_container_version failed");
                break;
            }

            if (NNDeserialize(m_ctx, m_minb_buffer, minn_packinfo_offset, m_minn_num) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NNDeserialize m_minn_num failed");
                break;
            }

            m_minn_infos.resize(m_minn_num);
            for (MI_S32 idx = 0; idx < m_minn_num; idx++)
            {
                if (NNDeserialize(m_ctx, m_minb_buffer, minn_packinfo_offset, m_minn_infos[idx].minn_model_name) != Status::OK)
                {
                    std::string info = "NNDeserialize get minn_model_name(" + std::to_string(idx) + " / " + std::to_string(m_minn_num) + ") failed";
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    minn_infos_valid = MI_FALSE;
                    break;
                }

                if (NNDeserialize(m_ctx, m_minb_buffer, minn_packinfo_offset, m_minn_infos[idx].minn_model_size) != Status::OK)
                {
                    std::string info = "NNDeserialize get minn_model_size(" + std::to_string(idx) + " / " + std::to_string(m_minn_num) + ") failed";
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    minn_infos_valid = MI_FALSE;
                    break;
                }
            }

            if (!minn_infos_valid)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "NNDeserialize m_minn_infos failed");
                break;
            }

            m_minb_file_offset += minn_packinfo_offset;
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported minb version");
            break;
        }

        m_is_valid = MI_TRUE;
    } while (0);

    if (fp != MI_NULL)
    {
        fclose(fp);
    }
}

NBModel::~NBModel()
{
    Release();
    m_is_valid = MI_FALSE;
}

AURA_VOID NBModel::Release()
{
    if (m_minb_buffer.IsValid())
    {
        AURA_FREE(m_ctx, m_minb_buffer.m_origin);
        m_minb_buffer.Clear();
    }
}

MI_U32 NBModel::GetMinbVersion() const
{
    return m_minb_version;
}

std::string NBModel::GetDate() const
{
    // e.g. 2024.12.13 10:16:33
    std::string minb_pack_date_str = std::to_string(m_minb_date.year) + "." + std::to_string(m_minb_date.month) + "." + std::to_string(m_minb_date.day) + " " +
                                     std::to_string(m_minb_date.hour) + ":" + std::to_string(m_minb_date.minute) + ":" + std::to_string(m_minb_date.second);
    return minb_pack_date_str;
}

std::string NBModel::GetAuthor() const
{
    return m_author;
}

std::string NBModel::GetDescription() const
{
    // e.g. it's a test model to show minb info
    return m_description;
}

std::string NBModel::GetVersion() const
{
    // e.g. 28.16
    std::string minb_pack_ver_str = std::to_string(m_container_version.major) + "." + std::to_string(m_container_version.minor);
    return minb_pack_ver_str;
}

std::vector<std::string> NBModel::GetMinnModelNames() const
{
    std::vector<std::string> minn_model_names;
    for (MI_S32 idx = 0; idx < m_minn_num; idx++)
    {
        minn_model_names.push_back(m_minn_infos[idx].minn_model_name);
    }
    return minn_model_names;
}

Buffer NBModel::GetModelBuffer(const std::string minn_model_name)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid model");
        return Buffer();
    }

    MI_S64 minb_model_offset   = m_minb_file_offset;
    MI_U64 dst_minn_model_size = 0;

    for (MI_S32 idx = 0; idx < m_minn_num; idx++)
    {
        if (m_minn_infos[idx].minn_model_name == minn_model_name)
        {
            dst_minn_model_size = m_minn_infos[idx].minn_model_size;
            break;
        }
        else
        {
            minb_model_offset += m_minn_infos[idx].minn_model_size;
        }
    }

    if (0 == dst_minn_model_size)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid minn model name");
        return Buffer();
    }

#if defined(AURA_BUILD_HOST)
    Buffer minn_buffer = NNModel::MapModelBufferFromFile(m_ctx, m_minb_file);
    if (!minn_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MapModelBufferFromFile failed");
        return Buffer();
    }
#else
    Buffer minn_buffer = NNModel::CreateModelBufferFromFile(m_ctx, m_minb_file);
    if (!minn_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CreateModelBufferFromFile failed");
        return Buffer();
    }
#endif // AURA_BUILD_HOST

    minn_buffer.Resize(dst_minn_model_size, minb_model_offset);
    return minn_buffer;
}

std::string NBModel::GetMinnModelInfo(const std::string minn_model_name, const std::string attr)
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid model");
        return std::string();
    }

    if ((attr != "all") && (attr != "model_version") && (attr != "framework") && (attr != "framework_version"))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid attr");
        return std::string();
    }

    Buffer minn_buffer;
    MinnHeader minn_header;
    MinnDataV1 minn_data;
    std::string backend_str;
    std::string framework_str;
    std::string minn_model_info;
    MI_S64 data_offset  = 0;
    MI_U32 minn_version = 0;

    minn_buffer = this->GetModelBuffer(minn_model_name);
    if (!minn_buffer.IsValid())
    {
        std::string info = "create " + minn_model_name + " minn_buffer fail!\n";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return minn_model_info;
    }

    if (NNDeserialize(m_ctx, minn_buffer, data_offset, minn_header) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "NNDeserialize minn_header failed");
        goto EXIT;
    }

    if (minn_header.magic_num != AURA_NN_MODEL_MAGIC)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "model minn_header check failed");
        goto EXIT;
    }

    minn_version = (minn_header.version.major << 16) | minn_header.version.minor;
    if ((0x010000 == minn_version) || (0x010001 == minn_version))
    {
        if (NNDeserialize(m_ctx, minn_buffer, data_offset, minn_data) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "NNDeserialize minn_data failed");
            goto EXIT;
        }

        switch (minn_data.framework)
        {
            case 1:
            {
                framework_str = "qnn";
                break;
            }
            case 2:
            {
                framework_str = "snpe";
                break;
            }
            case 10:
            {
                framework_str = "np";
                break;
            }
            case 20:
            {
                framework_str = "xnn";
                break;
            }
            case 30:
            {
                framework_str = "mnn";
                break;
            }
            default:
            {
                AURA_ADD_ERROR_STRING(m_ctx, "unsupport framework");
                goto EXIT;
            }
        }

        switch (minn_data.backend_type)
        {
            case 1:
            {
                backend_str = "cpu";
                break;
            }
            case 2:
            {
                backend_str = "gpu";
                break;
            }
            case 3:
            {
                backend_str = "npu";
                break;
            }
            default:
            {
                AURA_ADD_ERROR_STRING(m_ctx, "unsupport backend");
                goto EXIT;
            }
        }

        if ("all" == attr)
        {
            minn_model_info  = "-------- [" + minn_model_name + "] --------\n";
            minn_model_info += "size byte          : " + std::to_string(minn_buffer.m_size) + "\n";
            minn_model_info += "backend            : " + backend_str + "\n";
            minn_model_info += "framework          : " + framework_str + "\n";
            minn_model_info += "framework version  : " + std::to_string(minn_data.framework_version.major) + "." + std::to_string(minn_data.framework_version.minor) + "." + std::to_string(minn_data.framework_version.patch) + "\n";
            minn_model_info += "user version       : " + std::to_string(minn_data.model_version.major) + "." + std::to_string(minn_data.model_version.minor) + "\n";
        }
        else if ("model_version" == attr)
        {
            minn_model_info = std::to_string(minn_data.model_version.major) + "." + std::to_string(minn_data.model_version.minor);
        }
        else if ("framework" == attr)
        {
            minn_model_info = framework_str;
        }
        else if ("framework_version" == attr)
        {
            minn_model_info = std::to_string(minn_data.framework_version.major) + "." + std::to_string(minn_data.framework_version.minor) + "." + std::to_string(minn_data.framework_version.patch);
        }
    }

EXIT:
    AURA_FREE(m_ctx, minn_buffer.m_origin);
    minn_buffer.Clear();
    return minn_model_info;
}

MI_BOOL NBModel::IsValid() const
{
    return m_is_valid;
}

} // namespace aura