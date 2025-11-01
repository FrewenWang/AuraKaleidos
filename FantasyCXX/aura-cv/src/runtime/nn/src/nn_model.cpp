#include "nn_model.hpp"
#include "nn_executor_impl.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace aura
{

static Status NNDeserialize(Context *ctx, const Buffer &minn_buffer, MI_S64 &offset,
                            std::unordered_map<std::string, std::string> &names_map)
{
    MI_U32 map_len = 0;
    if (NNDeserialize(ctx, minn_buffer, offset, map_len) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
        return Status::ERROR;
    }

    for (MI_U32 i = 0; i < map_len; i++)
    {
        std::string ori_name, mapped_name;

        if (NNDeserialize(ctx, minn_buffer, offset, ori_name) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
            return Status::ERROR;
        }
        if (NNDeserialize(ctx, minn_buffer, offset, mapped_name) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
            return Status::ERROR;
        }

        names_map[ori_name] = mapped_name;
    }

    return Status::OK;
}

static Status NNDeserialize(Context *ctx, const Buffer &minn_buffer, MI_S64 &offset, Buffer &buffer)
{
    MI_S64 size = 0;

    if (NNDeserialize(ctx, minn_buffer, offset, size) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
        return Status::ERROR;
    }

    if (0 == size)
    {
        AURA_ADD_ERROR_STRING(ctx, "empty buffer");
        return Status::ERROR;
    }

    if (offset + size > minn_buffer.m_size)
    {
        std::string info = "minn_buffer overflow, curr minn buffer data pos is " + std::to_string(offset) + ", again read " + \
                           std::to_string(size) + "bytes, will excess minn_buffer size " + std::to_string(minn_buffer.m_size);
        AURA_ADD_ERROR_STRING(ctx, info.c_str() );
        return Status::ERROR;
    }

    buffer = minn_buffer;
    buffer.Resize(size, offset);
    offset += size;

    return Status::OK;
}

NNModel::NNModel(Context *ctx, const ModelInfo &model_info) : m_ctx(ctx), m_data_offset(0), m_model_info(model_info), m_is_valid(MI_FALSE)
{
    do
    {
        if (MI_NULL == ctx)
        {
            break;
        }

        if (!m_model_info.minn_buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "minn_buffer is invalid");
            break;
        }

        if (model_info.decrypt_key.empty())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "decrypt_key is empty");
            break;
        }

        MinnHeader header;
        if (NNDeserialize(m_ctx, m_model_info.minn_buffer, m_data_offset, header) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
            break;
        }

        if (header.magic_num != AURA_NN_MODEL_MAGIC)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "model header check failed");
            break;
        }

        m_minn_version = (header.version.major << 16) | header.version.minor;
        if ((1 == header.version.major) && (header.version.minor <= 2))
        {
            MinnDataV1 data;

            if (NNDeserialize(m_ctx, m_model_info.minn_buffer, m_data_offset, data) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
                break;
            }

            if (1 == data.framework)
            {
                m_framework_version = "qnn.v" + std::to_string(data.framework_version.major) + "." +
                                      std::to_string(data.framework_version.minor) + "." + std::to_string(data.framework_version.patch);
            }
            else if (2 == data.framework)
            {
                m_framework_version = "snpe.v" + std::to_string(data.framework_version.major) + "." +
                                      std::to_string(data.framework_version.minor) + "." + std::to_string(data.framework_version.patch);
            }
            else if (10 == data.framework)
            {
                m_framework_version = "np.v" + std::to_string(data.framework_version.major) + "." +
                                      std::to_string(data.framework_version.minor) + "." + std::to_string(data.framework_version.patch);
            }
            else if (20 == data.framework)
            {
                m_framework_version = "xnn.v" + std::to_string(data.framework_version.major) + "." +
                                      std::to_string(data.framework_version.minor) + "." + std::to_string(data.framework_version.patch);
            }
            else if (30 == data.framework)
            {
                m_framework_version = "mnn.v" + std::to_string(data.framework_version.major) + "." +
                                      std::to_string(data.framework_version.minor) + "." + std::to_string(data.framework_version.patch);
            }
            else
            {
                AURA_ADD_ERROR_STRING(m_ctx, "unsupported frame type");
                break;
            }

            m_model_version = "v" + std::to_string(data.model_version.major) + "." + std::to_string(data.model_version.minor);

            if (1 == data.backend_type)
            {
                m_backend_type = NNBackendToString(NNBackend::CPU);
            }
            else if (2 == data.backend_type)
            {
                m_backend_type = NNBackendToString(NNBackend::GPU);
            }
            else if (3 == data.backend_type)
            {
                m_backend_type = NNBackendToString(NNBackend::NPU);
            }
            else
            {
                AURA_ADD_ERROR_STRING(m_ctx, "unsupported backend type");
                break;
            }
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported minn version");
            break;
        }

        if ((1 == header.version.major) && (header.version.minor > 0))
        {
            Status ret = NNDeserialize(m_ctx, m_model_info.minn_buffer, m_data_offset, m_input_names_map);
            ret |= NNDeserialize(m_ctx, m_model_info.minn_buffer, m_data_offset, m_output_names_map);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
                break;
            }
        }

        m_is_valid = MI_TRUE;
    } while (0);
}

NNModel::~NNModel()
{
    ReleaseModelBuffer();
    m_is_valid = MI_FALSE;
    m_model_info = ModelInfo();
}

MI_U32 NNModel::GetMinnVersion() const
{
    return m_minn_version;
}

std::string NNModel::GetFrameWorkVersion() const
{
    return m_framework_version;
}

std::string NNModel::GetModelVersion() const
{
    return m_model_version;
}

std::string NNModel::GetBackendType() const
{
    return m_backend_type;
}

std::string NNModel::GetVersion() const
{
    std::string nn_model_version = "minn(v" + std::to_string(m_minn_version >> 16) + "." + std::to_string(m_minn_version & 0xffff) + ")"
                                   " model(" + GetModelVersion() + ":" + GetFrameWorkVersion() + ")";
    return nn_model_version;
}

std::string NNModel::GetModelName() const
{
    return m_model_info.minn_file_name;
}

Buffer NNModel::GetModelBuffer()
{
    if (!m_is_valid)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid model");
        return Buffer();
    }

    if (m_model_buffer.IsValid())
    {
        return m_model_buffer;
    }

    Buffer minn_buffer;

    if (NNDeserialize(m_ctx, m_model_info.minn_buffer, m_data_offset, minn_buffer) != Status::OK)
    {
        ReleaseModelBuffer();
        AURA_ADD_ERROR_STRING(m_ctx, "NNDeserialize failed");
        return Buffer();
    }

#if defined(AURA_BUILD_HEXAGON)
    MI_U32 addr_offset = AURA_ALIGN(reinterpret_cast<MI_U64>(minn_buffer.m_origin), 32) - reinterpret_cast<MI_U64>(minn_buffer.m_origin);
    m_model_buffer = m_model_info.minn_buffer;
    m_model_buffer.Resize(minn_buffer.m_size, addr_offset);
#else
    m_model_buffer = minn_buffer;
#endif // AURA_BUILD_HEXAGON

    MI_U64 decrypt_size = ((m_minn_version < 0x010002) || (minn_buffer.m_size < AURA_NN_MODEL_ENCRYPT_LENGTH)) ? minn_buffer.m_size : AURA_NN_MODEL_ENCRYPT_LENGTH;
    if (NNDecrypt(m_ctx, minn_buffer, m_model_buffer, m_model_info.decrypt_key, decrypt_size) != Status::OK)
    {
        ReleaseModelBuffer();
        AURA_ADD_ERROR_STRING(m_ctx, "NNDecrypt failed");
        return Buffer();
    }

    return m_model_buffer;
}

AURA_VOID NNModel::ReleaseModelBuffer()
{
    if (!m_model_info.minn_file_name.empty())
    {
#if defined(AURA_BUILD_HOST)
        if ((AURA_MEM_HEAP == m_model_info.minn_buffer.m_type) && (m_model_info.minn_buffer.m_property != 0))
        {
            munmap(m_model_info.minn_buffer.m_origin, m_model_info.minn_buffer.m_capacity);
        }
        else
#endif // AURA_BUILD_HOST
        {
            AURA_FREE(m_ctx, m_model_info.minn_buffer.m_origin);
        }
    }
    m_model_info.minn_buffer.Clear();
    m_model_info = ModelInfo();
}

MI_BOOL NNModel::IsValid() const
{
    return m_is_valid;
}

static MatMap MapMatNamesImpl(Context *ctx, const MatMap &mat_map, const std::unordered_map<std::string, std::string> &names_map)
{
    MatMap mapped;

    if (names_map.size() == 0)
    {
        return mat_map;
    }

    if (names_map.size() != mat_map.size())
    {
        AURA_ADD_ERROR_STRING(ctx, "mat map size not match the names map size");
        return MatMap();
    }

    for (const auto &pair : mat_map)
    {
        std::string ori_name = pair.first;
        if (names_map.find(ori_name) == names_map.end())
        {
            AURA_ADD_ERROR_STRING(ctx, ("mat name: " + ori_name + " not exist in names map").c_str());
            return MatMap();
        }

        std::string mapped_name = names_map.at(ori_name);
        mapped[mapped_name] = pair.second;
    }

    return mapped;
}

MatMap NNModel::MapMatNames(const MatMap &mat_map, MI_BOOL is_input) const
{
    if (is_input)
    {
        return MapMatNamesImpl(m_ctx, mat_map, m_input_names_map);
    }
    else
    {
        return MapMatNamesImpl(m_ctx, mat_map, m_output_names_map);
    }
}

static TensorDescMap MapTensorDescNamesImpl(Context *ctx, const TensorDescMap &tensor_desc_map,
                                            const std::unordered_map<std::string, std::string> &names_map)
{
    TensorDescMap mapped;

    if (names_map.size() == 0)
    {
        return tensor_desc_map;
    }

    if (names_map.size() != tensor_desc_map.size())
    {
        AURA_ADD_ERROR_STRING(ctx, "tensor desc map size not match the names map size");
        return TensorDescMap();
    }

    for (const auto &pair : tensor_desc_map)
    {
        std::string mapped_name = pair.first;

        MI_BOOL find_flag = MI_FALSE;
        for (const auto &name_pair : names_map)
        {
            if (name_pair.second == mapped_name)
            {
                mapped[name_pair.first] = pair.second;
                find_flag = MI_TRUE;
                break;
            }
        }

        if (MI_FALSE == find_flag)
        {
            AURA_ADD_ERROR_STRING(ctx, ("mapped name: " + mapped_name + " not exist in names map").c_str());
            return TensorDescMap();
        }
    }

    return mapped;
}

TensorDescMap NNModel::MapTensorDescNames(const TensorDescMap &tensor_desc_map, MI_BOOL is_input) const
{
    if (is_input)
    {
        return MapTensorDescNamesImpl(m_ctx, tensor_desc_map, m_input_names_map);
    }
    else
    {
        return MapTensorDescNamesImpl(m_ctx, tensor_desc_map, m_output_names_map);
    }
}

Buffer NNModel::CreateModelBufferFromFile(Context *ctx, const std::string &minn_file)
{
    if (MI_NULL == ctx)
    {
        return Buffer();
    }

    if (minn_file.empty())
    {
        AURA_ADD_ERROR_STRING(ctx, "minn_file is empty");
        return Buffer();
    }

    struct stat st;
    if (stat(minn_file.c_str(), &st) != 0)
    {
        AURA_ADD_ERROR_STRING(ctx, ("stat model: " + minn_file + " failed").c_str());
        return Buffer();
    }

    FILE *fp = fopen(minn_file.c_str(), "rb");
    if (MI_NULL == fp)
    {
        AURA_ADD_ERROR_STRING(ctx, ("open model: " + minn_file + " failed").c_str());
        return Buffer();
    }

    Buffer buffer = ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, st.st_size, 0));
    if (!buffer.IsValid())
    {
        fclose(fp);
        AURA_ADD_ERROR_STRING(ctx, ("alloc " + std::to_string(st.st_size) + " failed").c_str());
        return Buffer();
    }

    MI_S64 bytes = fread(buffer.m_data, 1, st.st_size, fp);
    if (bytes != st.st_size)
    {
        fclose(fp);
        AURA_FREE(ctx, buffer.m_data);
        AURA_ADD_ERROR_STRING(ctx, ("expect read " + std::to_string(st.st_size) + "bytes, but got " + std::to_string(bytes) + " bytes").c_str());
        return Buffer();
    }

    fclose(fp);
    return buffer;
}

#if defined(AURA_BUILD_HOST)
Buffer NNModel::MapModelBufferFromFile(Context *ctx, const std::string &minn_file)
{
    if (MI_NULL == ctx)
    {
        return Buffer();
    }

    if (minn_file.empty())
    {
        AURA_ADD_ERROR_STRING(ctx, "minn_file is empty");
        return Buffer();
    }

    struct stat st;
    if (stat(minn_file.c_str(), &st) != 0)
    {
        AURA_ADD_ERROR_STRING(ctx, ("stat model: " + minn_file + " failed").c_str());
        return Buffer();
    }

    MI_S32 fd = open(minn_file.c_str(), O_RDONLY);
    if (fd < 0)
    {
        AURA_ADD_ERROR_STRING(ctx, ("open model: " + minn_file + " failed").c_str());
        return Buffer();
    }

    AURA_VOID *map_addr = mmap(MI_NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (MAP_FAILED == map_addr)
    {
        close(fd);
        AURA_ADD_ERROR_STRING(ctx, ("mmap model: " + minn_file + " failed").c_str());
        return Buffer();
    }

    close(fd);

#  if defined(__USE_MISC)
    if (madvise(map_addr, st.st_size, MADV_NOHUGEPAGE))
    {
        munmap(map_addr, st.st_size);
        AURA_ADD_ERROR_STRING(ctx, ("madvise model: " + minn_file + " failed").c_str());
        return Buffer();
    }
#  endif // __USE_MISC

    return Buffer(AURA_MEM_HEAP, st.st_size, st.st_size, map_addr, map_addr, fd);
}
#endif // AURA_BUILD_HOST

} // namespace aura