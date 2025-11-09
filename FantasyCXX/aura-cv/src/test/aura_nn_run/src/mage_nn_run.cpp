#include "aura_nn_run.hpp"
#include "nb_model.hpp"

#include "aura/runtime/core.h"

#include <dlfcn.h>
#include <sys/stat.h>

#include <queue>
#include <regex>
#include <chrono>
#include <string>
#include <thread>
#include <fstream>
#include <ostream>
#include <iomanip>
#include <numeric>
#include <unordered_set>

using namespace aura;

#define ASSGIN_VALUE(config, key, value)  \
if (!value.empty())                       \
{                                         \
    config[key] = value;                  \
}                                         \

#pragma pack(push, 1)
struct MinnDataV1
{
    DT_S16 framework;
    struct FrameVersion
    {
        DT_S16 major;
        DT_S16 minor;
        DT_S16 patch;
    } framework_version;
    struct ModelVersion
    {
        DT_S16 major;
        DT_S16 minor;
    } model_version;
    DT_S16 backend_type;
};

struct MinnHeader
{
    DT_S32 magic_num;
    struct Version
    {
        DT_S16 major;
        DT_S16 minor;
    } version;
};
#pragma pack(pop)

template <typename Tp>
AURA_INLINE Status NNDeserialize(FILE *handle, Tp &val)
{
    if (fread(&val, sizeof(Tp), 1, handle) != 1)
    {
        std::cout << "fread failed" << std::endl;
        return Status::ERROR;
    }

    return Status::OK;
}

AURA_INLINE std::vector<std::string> NNSplit(const std::string &src, DT_CHAR separator)
{
    std::vector<std::string> result;
    std::string value;
    std::istringstream tokenized_string_stream(src);

    while (getline(tokenized_string_stream, value, separator))
    {
        result.push_back(value);
    }

    return result;
}

Status MakeDirectories(const std::string &path)
{
    std::vector<std::string> multy_dirs = NNSplit(path, '/');

    std::string cur_dir;
    for (const auto &dir : multy_dirs)
    {
        cur_dir += dir + "/";

        struct stat stat_buf;
        if (stat(cur_dir.c_str(), &stat_buf) != 0)
        {
            if (mkdir(cur_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
            {
                std::cout << "fail to create dir:" << path << std::endl;
                return Status::ERROR;
            }
        }
    }

    return Status::OK;
}

AURA_INLINE ElemType GetType(const std::string &type)
{
    if (type.empty())
    {
        return ElemType::INVALID;
    }

    ElemType elem_type;

    if ("s8" == type)
    {
        elem_type = ElemType::S8;
    }
    else if ("u8" == type)
    {
        elem_type = ElemType::U8;
    }
    else if ("u16" == type)
    {
        elem_type = ElemType::U16;
    }
    else if ("s16" == type)
    {
        elem_type = ElemType::S16;
    }
    else if ("u32" == type)
    {
        elem_type = ElemType::U32;
    }
    else if ("s32" == type)
    {
        elem_type = ElemType::S32;
    }
    else if ("f16" == type)
    {
        elem_type = ElemType::F16;
    }
    else if ("f32" == type)
    {
        elem_type = ElemType::F32;
    }
    else
    {
        return ElemType::INVALID;
    }

    return elem_type;
}

AURA_INLINE std::vector<std::string> Split(const std::string &tokenized_string, DT_CHAR separator)
{
    std::vector<std::string> split_string;
    std::istringstream tokenized_string_stream(tokenized_string);
    while (!tokenized_string_stream.eof())
    {
        std::string value;
        getline(tokenized_string_stream, value, separator);
        if (!value.empty())
        {
            split_string.emplace_back(value);
        }
    }
    return split_string;
}

AURA_INLINE std::string SanitizeName(const std::string &name)
{
    std::string sanitized_name = std::regex_replace(name, std::regex("\\W+"), "_");
    if (!std::isalpha(sanitized_name[0]) && sanitized_name[0] != '_')
    {
        sanitized_name = "_" + sanitized_name;
    }
    return sanitized_name;
}

AURA_INLINE DT_VOID DumpOut(const CommandParam &param, const std::string &input_file_name, const MatMap &mat_map)
{
    size_t last_slash_idx = input_file_name.rfind('/');
    std::string file_with_extension = (last_slash_idx != std::string::npos) ?
                                       input_file_name.substr(last_slash_idx + 1) : input_file_name;

    size_t last_dot_idx = file_with_extension.rfind('.');
    std::string out_file_prefix = (last_dot_idx != std::string::npos) ?
                                   file_with_extension.substr(0, last_dot_idx) : file_with_extension;

    std::string outdir_name;
    if ((("mnn" == param.platform) || ("qnn" == param.platform)) && !param.dump_path.empty())
    {
        outdir_name = param.dump_path + "/"
                                      + out_file_prefix
                                      + std::string("_graph")
                                      + std::to_string(std::atoi(param.qnn_graph_ids.c_str()))
                                      + std::string("_");
    }
    else
    {
        outdir_name = param.output_path + "/"
                                        + out_file_prefix
                                        + std::string("_graph")
                                        + std::to_string(std::atoi(param.qnn_graph_ids.c_str()))
                                        + std::string("_");
    }

    for (auto &mat : mat_map)
    {
        std::string replace_name = SanitizeName(mat.first);
        mat.second->Dump(outdir_name + "_" + replace_name + ".bin");
    }
}

AURA_INLINE DT_VOID InferenceDelay(DT_S32 ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

AURA_INLINE DT_BOOL CheckParamValid(const std::string &options, const std::vector<std::vector<std::string>> &command_set)
{
    for (DT_S32 idx = 0; idx < (DT_S32)command_set.size(); idx++)
    {
        DT_BOOL is_valid_set = DT_TRUE;
        for (DT_S32 idy = 0; idy < (DT_S32)command_set[idx].size(); idy++)
        {
            if (options.find(command_set[idx][idy]) == std::string::npos)
            {
                is_valid_set = DT_FALSE;
                break;
            }
        }

        if (is_valid_set)
        {
            return DT_TRUE;
        }
    }

    return DT_FALSE;
}

AURA_INLINE DT_BOOL StringContains(const std::string &str, const std::string &sub_str)
{
    return str.compare(0, str.size(), sub_str) == 0;
}

AURA_INLINE DT_BOOL IsModelContainer(const std::string &model_name)
{
    return (model_name.substr(model_name.rfind('.') + 1) == "minb");
}

AURA_INLINE std::string ParseStringOptions(std::vector<std::string> &options, const std::string &key_short, const std::string &key_long,
                                    size_t offset, const std::string &default_value)
{
    std::string result;

    if (options.empty() || offset > 1)
    {
        return result;
    }

    for (size_t idx = 0; idx < options.size(); idx++)
    {
        if (StringContains(options[idx], key_short) || StringContains(options[idx], key_long))
        {
            if (0 == offset)
            {
                result = options[idx + 1];
            }
            else
            {
                if (idx + 1 + offset < options.size())
                {
                    result = options[idx + 1 + offset];
                }
                else
                {
                    result = default_value;
                }
            }

            break;
        }
    }
    return result;
}

static Status ReadInputList(Context *ctx, const std::string &input_list, const TensorDescMap &input_tensor_desc, std::vector<std::vector<InputDataInfo>> &input_list_data_info)
{
    std::ifstream file_list_stream(input_list);
    if (!file_list_stream.is_open())
    {
        AURA_LOGE(ctx, AURA_TAG, "fail to open input list %s\n", input_list.c_str());
        return Status::ERROR;
    }

    std::vector<std::string> lines;
    std::string file_line;
    while (std::getline(file_list_stream, file_line))
    {
        // remove blank line
        if (file_line.empty())
        {
            continue;
        }

        // remove lines with # or %s
        if ((file_line.find('#') != std::string::npos) || (file_line.find('%') != std::string::npos))
        {
            AURA_LOGD(ctx, AURA_TAG, "file line is ignore, which contains invalid symbol # or %,\n");
            continue;
        }

        //remove CR
        file_line.erase(std::remove(file_line.begin(), file_line.end(), '\r'), file_line.end());

        lines.push_back(file_line);
    }

    std::string separator = ":=";
    for (DT_S32 i = 0; i < (DT_S32)lines.size(); i++)
    {
        std::vector<InputDataInfo> tenor_file_names_vec;
        std::vector<std::string> tenor_file_names = Split(lines[i], ' ');
        for (auto input_info : tenor_file_names)
        {
            InputDataInfo info;
            auto position = input_info.find(separator);
            if (position != std::string::npos)
            {
                info.tensor_name = input_info.substr(0, position);
                info.file_name   = input_info.substr(position + separator.size());
            }

            if (!input_tensor_desc.count(info.tensor_name))
            {
                AURA_LOGE(ctx, AURA_TAG, "tensor name %s in input list is not in model\n", info.tensor_name.c_str());
                file_list_stream.close();
                return Status::ERROR;
            }

            tenor_file_names_vec.push_back(info);
        }

        if (tenor_file_names_vec.size() != input_tensor_desc.size())
        {
            AURA_LOGE(ctx, AURA_TAG, "tensor num in input list is different from tensor num in model, model input tensor is %d, but input list get %d\n",
                      (DT_S32)input_tensor_desc.size(), (DT_S32)tenor_file_names_vec.size());
            file_list_stream.close();
            return Status::ERROR;
        }

        input_list_data_info.push_back(tenor_file_names_vec);
    }

    if (input_list_data_info.size() <= 0)
    {
        AURA_LOGE(ctx, AURA_TAG, "input list file %s is empty or context is invalid\n", input_list.c_str());
        file_list_stream.close();
        return Status::ERROR;
    }

    file_list_stream.close();

    return Status::OK;
}

static ElemType GetElemTypeFromFile(Context *ctx, const std::string &elemtype_file, const std::string &tensor_name)
{
    ElemType elem_type = ElemType::INVALID;

    std::ifstream file_elemtype_stream(elemtype_file);
    if (!file_elemtype_stream.is_open())
    {
        AURA_LOGE(ctx, AURA_TAG, "fail to open elemtype file %s\n", elemtype_file.c_str());
        return elem_type;
    }

    std::string file_line;
    std::getline(file_elemtype_stream, file_line);
    if (file_line.empty())
    {
        AURA_LOGE(ctx, AURA_TAG, "cant find %s elemtype in file %s\n", tensor_name.c_str(), elemtype_file.c_str());
        file_elemtype_stream.close();
        return elem_type;
    }

    std::string separator = ":=";
    std::vector<std::string> tenor_name_elemtypes = Split(file_line, ' ');
    for (auto name_elemtype : tenor_name_elemtypes)
    {
        auto position = name_elemtype.find(separator);
        if (position != std::string::npos)
        {
            std::string cur_tensor_name  = name_elemtype.substr(0, position);
            std::string cur_elemtype_str = name_elemtype.substr(position + separator.size());
            if (cur_tensor_name == tensor_name)
            {
                elem_type = GetType(cur_elemtype_str);
                break;
            }
        }
    }

    file_elemtype_stream.close();
    return elem_type;
}

static ElemType GetElemTypeFromParam(Context *ctx, const std::string &elemtype_str, const std::string &tensor_name)
{
    ElemType elem_type = ElemType::INVALID;

    elem_type = GetType(elemtype_str);
    if (elem_type != ElemType::INVALID)
    {
        return elem_type;
    }

    elem_type = GetElemTypeFromFile(ctx, elemtype_str, tensor_name);

    return elem_type;
}

static Status ParsePlatform(Context *ctx, const std::string &model_path, std::string &platform)
{
    Status ret = Status::ERROR;

    do
    {
        if (model_path.empty())
        {
            AURA_LOGE(ctx, AURA_TAG, "model_file is empty\n");
            break;
        }

        FILE *handle = fopen(model_path.c_str(), "rb");
        if (DT_NULL == handle)
        {
            AURA_LOGE(ctx, AURA_TAG, "fail to open model %s\n", model_path.c_str());
            break;
        }

        constexpr DT_S32 AURA_NN_MODEL_MAGIC = 0x4D694E4E; // MiNN

        MinnHeader header;
        memset(&header, 0, sizeof(MinnHeader));

        if (NNDeserialize(handle, header) != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "NNDeserialize failed\n");
            fclose(handle);
            break;
        }

        if (header.magic_num != AURA_NN_MODEL_MAGIC)
        {
            AURA_LOGE(ctx, AURA_TAG, "model header check failed\n");
            fclose(handle);
            break;
        }

        if ((1 == header.version.major) && (header.version.minor <= 2))
        {
            MinnDataV1 data;
            memset(&data, 0, sizeof(MinnDataV1));

            if (NNDeserialize(handle, data) != Status::OK)
            {
                AURA_LOGE(ctx, AURA_TAG, "NNDeserialize failed\n");
                fclose(handle);
                break;
            }

            std::string framework_version;

            if (1 == data.framework)
            {
                platform = "qnn";
                framework_version = "qnn.v" + std::to_string(data.framework_version.major) + "." +
                                      std::to_string(data.framework_version.minor) + "." + std::to_string(data.framework_version.patch);
            }
            else if (2 == data.framework)
            {
                platform = "snpe";
                framework_version = "snpe.v" + std::to_string(data.framework_version.major) + "." +
                        std::to_string(data.framework_version.minor) + "." + std::to_string(data.framework_version.patch);
            }
            else if (10 == data.framework)
            {
                framework_version = "np.v" + std::to_string(data.framework_version.major) + "." +
                                      std::to_string(data.framework_version.minor) + "." + std::to_string(data.framework_version.patch);
                platform = "np";
            }
            else if (20 == data.framework)
            {
                framework_version = "xnn.v" + std::to_string(data.framework_version.major) + "." +
                                      std::to_string(data.framework_version.minor) + "." + std::to_string(data.framework_version.patch);
                platform = "xnn";
            }
            else if (30 == data.framework)
            {
                framework_version = "mnn.v" + std::to_string(data.framework_version.major) + "." +
                                      std::to_string(data.framework_version.minor) + "." + std::to_string(data.framework_version.patch);
                platform = "mnn";
            }
            else
            {
                AURA_LOGE(ctx, AURA_TAG, "unsupported framework type\n");
                fclose(handle);
                break;
            }

            std::string model_version = "v" + std::to_string(data.model_version.major) + "." + std::to_string(data.model_version.minor);

            AURA_LOGI(ctx, AURA_TAG, "=========[model info]=========\n");
            AURA_LOGI(ctx, AURA_TAG, "model version    : %s\n", model_version.c_str());
            AURA_LOGI(ctx, AURA_TAG, "framework version: %s\n", framework_version.c_str());
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "unsupported minn version\n");
            fclose(handle);
            break;
        }

        fclose(handle);
        ret = Status::OK;
    } while (0);

    return ret;
}

static Status ParsePlatform(Context *ctx, const std::string &model_path, const std::string &minn_name, std::string &platform)
{
    Status ret = Status::ERROR;
    std::string model_version;
    std::string framework_version;

    std::shared_ptr<NBModel> nb_model = std::make_shared<NBModel>(ctx, model_path);
    if (!nb_model->IsValid())
    {
        AURA_LOGE(ctx, AURA_TAG, "create nb model failed\n");
        return ret;
    }

    platform = nb_model->GetMinnModelInfo(minn_name, "framework");
    if (platform.empty())
    {
        AURA_LOGE(ctx, AURA_TAG, "get framework failed\n");
        return ret;
    }

    model_version = nb_model->GetMinnModelInfo(minn_name, "model_version");
    if (model_version.empty())
    {
        AURA_LOGE(ctx, AURA_TAG, "get model_version failed\n");
        return ret;
    }
    model_version = "v" + model_version;

    framework_version = nb_model->GetMinnModelInfo(minn_name, "framework_version");
    if (framework_version.empty())
    {
        AURA_LOGE(ctx, AURA_TAG, "get framework_version failed\n");
        return ret;
    }
    framework_version = platform + ".v" + framework_version;

    AURA_LOGI(ctx, AURA_TAG, "=========[model info]=========\n");
    AURA_LOGI(ctx, AURA_TAG, "model version    : %s\n", model_version.c_str());
    AURA_LOGI(ctx, AURA_TAG, "framework version: %s\n", framework_version.c_str());

    ret = Status::OK;
    return ret;
}

static Status GetTensorInfo(Context *ctx, const CommandParam &param)
{
    if (DT_NULL == ctx)
    {
        AURA_LOGE(ctx, AURA_TAG, "bad args, ctx is nullptr...\n");
        return Status::ERROR;
    }

    NNEngine *nn_engine = ctx->GetNNEngine();
    if (DT_NULL == nn_engine)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_engine is null\n");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    // init model
    NNConfig config;
    if (param.backend != "")
    {
        config["backend"] = param.backend;
    }

    std::shared_ptr<NNExecutor> nn_executor = DT_NULL;
    if (!param.model_path.empty())
    {
        nn_executor = nn_engine->CreateNNExecutor(param.model_path, param.password, config);
    }
    else if (!param.model_container_path.empty())
    {
        nn_executor = nn_engine->CreateNNExecutor(param.model_container_path, param.minn_name, param.password, config);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "'--model' or '--nb_model' is required\n");
        return ret;
    }

    if (DT_NULL == nn_executor)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor is null\n");
        return ret;
    }

    if (nn_executor->Initialize() != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor Initialize fail\n");
        return ret;
    }

    // get tensor info
    // get input tensor info
    std::vector<TensorDescMap> input_tensor_info = nn_executor->GetInputs();
    if (input_tensor_info.size() <= 0)
    {
        AURA_LOGE(ctx, AURA_TAG, "GetInputs fail, input tensor num <= 0\n");
        return ret;
    }

    AURA_LOGD(ctx, AURA_TAG, "input tensor info: %s\n", TensorDescMapToString(input_tensor_info).c_str());

    // get output tensor info
    std::vector<TensorDescMap> output_tensor_info = nn_executor->GetOutputs();
    if (output_tensor_info.size() <= 0)
    {
        AURA_LOGE(ctx, AURA_TAG, "GetOutputs fail, output tensor num <= 0\n");
        return ret;
    }

    AURA_LOGD(ctx, AURA_TAG, "ouput tensor info: %s\n", TensorDescMapToString(output_tensor_info).c_str());

    ret = Status::OK;

    return ret;
}

static Status GetMinbInfo(Context *ctx, const CommandParam &param)
{
    if (DT_NULL == ctx)
    {
        AURA_LOGE(ctx, AURA_TAG, "bad args, ctx is nullptr...\n");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    size_t minn_num = 0;
    std::vector<std::string> minn_names;
    std::shared_ptr<NBModel> nb_model = DT_NULL;
    NNEngine *nn_engine = ctx->GetNNEngine();
    if (DT_NULL == nn_engine)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_engine is null\n");
        return Status::ERROR;
    }

    if (!IsModelContainer(param.model_container_path))
    {
        AURA_LOGE(ctx, AURA_TAG, "%s is not .minb file, not support '--get_minb_info' option\n", param.model_container_path.c_str());
        return ret;
    }

    nb_model = std::make_shared<NBModel>(ctx, param.model_container_path);
    if (!nb_model->IsValid())
    {
        AURA_LOGE(ctx, AURA_TAG, "create nb model failed\n");
        return ret;
    }

    AURA_LOGI(ctx, AURA_TAG, "=================[MINB INFO]=================\n");
    AURA_LOGI(ctx, AURA_TAG, " - date          : %s\n", nb_model->GetDate().c_str());
    AURA_LOGI(ctx, AURA_TAG, " - author        : %s\n", nb_model->GetAuthor().c_str());
    AURA_LOGI(ctx, AURA_TAG, " - description   : %s\n", nb_model->GetDescription().c_str());
    AURA_LOGI(ctx, AURA_TAG, " - version       : %s\n", nb_model->GetVersion().c_str());

    minn_names = nb_model->GetMinnModelNames();
    minn_num   = minn_names.size();
    for (size_t idx = 0; idx < minn_num; idx++)
    {
        std::string minn_model_info = nb_model->GetMinnModelInfo(minn_names[idx]);
        if (minn_model_info.empty())
        {
            AURA_LOGE(ctx, AURA_TAG, "minb GetMinnModelInfo for %s fail\n", minn_names[idx].c_str());
            return ret;
        }

        AURA_LOGD(ctx, AURA_TAG, " - #%zu minn info  :\n%s", idx + 1, minn_model_info.c_str());
    }

    ret = Status::OK;
    return ret;
}

Status AuraNnRun::NetRun(Context *ctx, const CommandParam &param)
{
    if (DT_NULL == ctx)
    {
        AURA_LOGE(ctx, AURA_TAG, "bad args, ctx is nullptr...\n");
        return Status::ERROR;
    }

    NNEngine *nn_engine = ctx->GetNNEngine();
    if (DT_NULL == nn_engine)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_engine is null\n");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    static std::vector<DT_F32> init_time;
    static std::vector<DT_F32> forward_time;

    static DT_F32 consume_time = 0.f;
    static DT_S32 count = 0;

    // 1. model init
    aura::Time start = aura::Time::Now();

    if (param.memory_tag_info)
    {
        param.func_memProfiler_trace_start(("AURA_NN_init_" + std::to_string(count)).c_str());
    }

    NNConfig config;
    ASSGIN_VALUE(config, "perf_level",        param.perf_level);
    ASSGIN_VALUE(config, "profiling_path",    param.profiling_path);
    ASSGIN_VALUE(config, "log_level",         param.log_level);
    ASSGIN_VALUE(config, "backend",           param.backend);
    ASSGIN_VALUE(config, "profiling_level",   param.profiling_level);
    ASSGIN_VALUE(config, "mnn_precision",     param.mnn_precision);
    ASSGIN_VALUE(config, "mnn_memory",        param.mnn_memory);
    ASSGIN_VALUE(config, "htp_mem_step_size", param.htp_mem_step_size);
    ASSGIN_VALUE(config, "snpe_unsigned_pd",  param.snpe_unsigned_pd);
    ASSGIN_VALUE(config, "qnn_graph_ids",     param.qnn_graph_ids);
    ASSGIN_VALUE(config, "qnn_udo_path",      param.qnn_udo_path);
    ASSGIN_VALUE(config, "mnn_dump_layers",   param.mnn_dump_layers);
    ASSGIN_VALUE(config, "mnn_tuning",        param.mnn_tuning);
    ASSGIN_VALUE(config, "mnn_clmem",         param.mnn_clmem);

    std::shared_ptr<NNExecutor> nn_executor = DT_NULL;

    if (!param.model_path.empty())
    {
        nn_executor = nn_engine->CreateNNExecutor(param.model_path, param.password, config);
    }
    else if (!param.model_container_path.empty())
    {
        nn_executor = nn_engine->CreateNNExecutor(param.model_container_path, param.minn_name, param.password, config);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "'--model' or '--nb_model' is required\n");
        return ret;
    }

    if (DT_NULL == nn_executor)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor is null\n");
        return ret;
    }

    if (nn_executor->Initialize() != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor Initialize fail\n");
        return ret;
    }

    if (param.memory_tag_info)
    {
        param.func_memProfiler_trace_end(("AURA_NN_init_" + std::to_string(count)).c_str());
    }

    aura::Time end = aura::Time::Now();
    consume_time = end.AsMilliSec() - start.AsMilliSec();
    init_time.push_back(consume_time);

    if (param.show_time)
    {
        AURA_LOGD(ctx, AURA_TAG, "nn_init_time = %fms\n", consume_time);
    }

    // 2. model forward
    std::vector<TensorDescMap> input_tensor_info  = nn_executor->GetInputs();
    std::vector<TensorDescMap> output_tensro_info = nn_executor->GetOutputs();

    DT_S8 qnn_graph_ids = std::atoi(param.qnn_graph_ids.c_str());
    TensorDescMap &input_tensor_desc  = input_tensor_info[qnn_graph_ids];
    TensorDescMap &output_tensor_desc = output_tensro_info[qnn_graph_ids];

    MatMap input_mat_map;
    MatMap output_mat_map;

    std::vector<Mat> input_mat_vec;
    std::vector<Mat> output_mat_vec;

    MatFactory factory(ctx);

    // create input and output mat map
    auto prepare_io_mat = [&ctx, &param, &factory](const TensorDescMap &tensor_desc_map, MatMap &matmap, std::vector<Mat> &mat_vec, DT_BOOL is_input) -> Status
    {
        for (const auto &tensor_desc : tensor_desc_map)
        {
            const std::string tensor_name = tensor_desc.first;

            ElemType elem_type = GetElemTypeFromParam(ctx, param.src_type, tensor_name);
            if (ElemType::INVALID == elem_type)
            {
                AURA_LOGE(ctx, AURA_TAG, "element type of %s is invalid\n", tensor_name.c_str());
                return Status::ERROR;
            }

            const std::vector<DT_S32> &sizes = tensor_desc.second.sizes;
            DT_S32 elem_counts = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<DT_S32>());

            if ((param.use_random_inputs) && (is_input))
            {
                DT_F32 min = 0.f;
                DT_F32 max = 255.f;
                mat_vec.push_back(factory.GetRandomMat(min, max, elem_type, {1, 1, elem_counts}, AURA_MEM_DEFAULT));
            }
            else
            {
                mat_vec.push_back(factory.GetEmptyMat(elem_type, {1, 1, elem_counts}, AURA_MEM_DEFAULT));
            }
            matmap[tensor_name] = &(mat_vec.back());

            if (!matmap[tensor_name]->IsValid())
            {
                AURA_LOGE(ctx, AURA_TAG, "GetEmptyMat failed\n");
                return Status::ERROR;
            }
        }

        return Status::OK;
    };

    if (prepare_io_mat(input_tensor_desc, input_mat_map, input_mat_vec, DT_TRUE) != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "prepare input mat failed\n");
        return Status::ERROR;
    }

    if (prepare_io_mat(output_tensor_desc, output_mat_map, output_mat_vec, DT_FALSE) != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "prepare output mat failed\n");
        return Status::ERROR;
    }

    // read input list and check to get <input_name, file_path> pair
    std::vector<std::vector<InputDataInfo>> input_list_data_info;
    if (!param.use_random_inputs)
    {
        if (ReadInputList(ctx, param.input_list, input_tensor_desc, input_list_data_info) != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "ReadInputList failed\n");
            return Status::ERROR;
        }
    }

    DT_S32 batch_num = param.use_random_inputs ? 1 : input_list_data_info.size();

    for (DT_S32 batch_cnt = 0; batch_cnt < batch_num; batch_cnt++)
    {
        // read mat data from file
        if (!param.use_random_inputs)
        {
            for (const auto &input_data_info : input_list_data_info[batch_cnt])
            {
                if (input_mat_map.find(input_data_info.tensor_name) == input_mat_map.end())
                {
                    AURA_LOGE(ctx, AURA_TAG, "input name %s not found in model\n", input_data_info.tensor_name.c_str());
                    return Status::ERROR;
                }

                if (input_mat_map[input_data_info.tensor_name]->Load(input_data_info.file_name) != Status::OK)
                {
                    AURA_LOGE(ctx, AURA_TAG, "input %s load %s fail\n", input_data_info.tensor_name.c_str(), input_data_info.file_name.c_str());
                    return Status::ERROR;
                }
            }
        }

        if (param.register_mem)
        {
            // memory register
            AnyParams any_params;
            any_params["input_matmap"]  = input_mat_map;
            any_params["output_matmap"] = output_mat_map;
            ret = nn_executor->Update("register_mem", any_params);
            if (ret != Status::OK)
            {
                AURA_LOGE(ctx, AURA_TAG, "register failed\n");
                return Status::ERROR;
            }
        }

        for (DT_S32 loop_inner_cnt = 0; loop_inner_cnt < param.loop_num_inner; loop_inner_cnt++)
        {
            // forward
            if (param.memory_tag_info)
            {
                param.func_memProfiler_trace_start(("AURA_NN_Forward_" + std::string("loop_")  + std::to_string(count)
                                                                       + std::string("file_")  + std::to_string(batch_cnt)
                                                                       + std::string("graph_") + std::to_string(qnn_graph_ids)).c_str());
            }

            start = aura::Time::Now();

            ret = nn_executor->Forward(input_mat_map, output_mat_map, qnn_graph_ids);
            if (ret != Status::OK)
            {
                AURA_LOGE(ctx, AURA_TAG, "nn_executor Forward fail\n");
                return ret;
            }

            end = aura::Time::Now();
            consume_time = end.AsMilliSec() - start.AsMilliSec();
            forward_time.push_back(consume_time);

            if (param.show_time)
            {
                AURA_LOGD(ctx, AURA_TAG, "nn_forward_time = %fms\n", consume_time);
            }

            if (param.memory_tag_info)
            {
                param.func_memProfiler_trace_end(("AURA_NN_Forward_" + std::string("loop_")  + std::to_string(count)
                                                                     + std::string("file_")  + std::to_string(batch_cnt)
                                                                     + std::string("graph_") + std::to_string(qnn_graph_ids)).c_str());
            }

            if (param.inference_interval > 0)
            {
                InferenceDelay(param.inference_interval);
            }
            count++;
        }

        if (param.register_mem)
        {
            AnyParams any_params;
            any_params["input_matmap"]  = input_mat_map;
            any_params["output_matmap"] = output_mat_map;
            ret = nn_executor->Update("deregister_mem", any_params);
            if (ret != Status::OK)
            {
                AURA_LOGE(ctx, AURA_TAG, "deregister failed\n");
                return Status::ERROR;
            }
        }

        // dump
        if (!param.use_random_inputs)
        {
            DumpOut(param, input_list_data_info[batch_cnt][0].file_name, output_mat_map);
        }
    }

    if ((DT_TRUE == param.show_time) && (count == (param.loop_num * batch_num)))
    {
        std::sort(init_time.begin(), init_time.end());
        std::sort(forward_time.begin(), forward_time.end());

        DT_F32 init_min_time = init_time.front();
        DT_F32 init_max_time = init_time.back();
        DT_F32 init_avr_time = 0.f;
        DT_F32 forward_min_time = forward_time.front();
        DT_F32 forward_max_time = forward_time.back();
        DT_F32 forward_avr_time = 0.f;

        DT_F64 sum = 0.0;
        for (DT_S32 i = 0; i < (DT_S32)init_time.size(); i++)
        {
            sum += init_time[i];
        }
        init_avr_time = sum / init_time.size();

        sum = 0.0;
        for (DT_S32 i = 0; i < (DT_S32)forward_time.size(); i++)
        {
            sum += forward_time[i];
        }
        forward_avr_time = sum / forward_time.size();

        AURA_LOGD(ctx, AURA_TAG, "=========[time info]=========\n");
        AURA_LOGD(ctx, AURA_TAG, "nn init:    min = %fms max = %fms avg = %fms\n", init_min_time, init_max_time, init_avr_time);
        AURA_LOGD(ctx, AURA_TAG, "nn forward: min = %fms max = %fms avg = %fms\n", forward_min_time, forward_max_time, forward_avr_time);
    }

    return ret;
}

Status AuraNnRun::ParseCommandLine(DT_S32 argc, DT_CHAR *argv[])
{
    // check required options
    std::string check_opt;
    std::vector<std::string> options;
    for (DT_S32 i = 1; i < argc; ++i)
    {
        options.emplace_back(argv[i]);
        check_opt += std::string(argv[i]) + " ";
    }

    std::vector<std::vector<std::string>> command_set
    {
        {"--password", "--input_list"},
        {"--password", "--use_random_inputs"}
    };

    m_commond_param.model_path           = ParseStringOptions(options, "-m",                "--model",            0, "");
    m_commond_param.model_container_path = ParseStringOptions(options, "-nb_model",         "--nb_model",         0, "");
    m_commond_param.password             = ParseStringOptions(options, "-w",                "--password",         0, "");
    m_commond_param.input_list           = ParseStringOptions(options, "-d",                "--input_list",       0, "");
    m_commond_param.src_type             = ParseStringOptions(options, "-input_data_type",  "--input_data_type",  0, "");
    m_commond_param.dst_type             = ParseStringOptions(options, "-output_data_type", "--output_data_type", 0, "");

    if (m_commond_param.model_path.empty() && m_commond_param.model_container_path.empty())
    {
        AURA_LOGE(m_ctx.get(), AURA_TAG, "--model or --nb_model param is empty.\n");
        return Status::ERROR;
    }

    if (!m_commond_param.model_path.empty() && !m_commond_param.model_container_path.empty())
    {
        AURA_LOGE(m_ctx.get(), AURA_TAG, "--model and --nb_model param should not be set both.\n");
        return Status::ERROR;
    }

    // check optianal options
    m_commond_param.output_path       = ParseStringOptions(options, "-output_path",        "--output_path",        0, "");
    m_commond_param.profiling_path    = ParseStringOptions(options, "-profiling_path",     "--profiling_path",     0, "");
    m_commond_param.dump_path         = ParseStringOptions(options, "-dump_path",          "--dump_path",          0, "");
    m_commond_param.htp_mem_step_size = ParseStringOptions(options, "-htp_mem_step_size",  "--htp_mem_step_size",  0, "");
    m_commond_param.snpe_unsigned_pd  = ParseStringOptions(options, "-snpe_unsigned_pd",   "--snpe_unsigned_pd",   0, "");
    m_commond_param.qnn_graph_ids     = ParseStringOptions(options, "-qnn_graph_ids",      "--qnn_graph_ids",      0, "");
    m_commond_param.qnn_graph_ids     = ParseStringOptions(options, "-qnn_udo_path",       "--qnn_udo_path",       0, "");
    m_commond_param.mnn_dump_layers   = ParseStringOptions(options, "-mnn_dump_layers",    "--mnn_dump_layers",    0, "");
    m_commond_param.mnn_tuning        = ParseStringOptions(options, "-mnn_tuning",         "--mnn_tuning",         0, "");
    m_commond_param.mnn_clmem         = ParseStringOptions(options, "-mnn_clmem",          "--mnn_clmem",          0, "");
    m_commond_param.profiling_level   = ParseStringOptions(options, "-profiling_level",    "--profiling_level",    0, "");
    m_commond_param.log_level         = ParseStringOptions(options, "-log_level",          "--log_level",          0, "");
    m_commond_param.perf_level        = ParseStringOptions(options, "-perf_level",         "--perf_level",         0, "");
    m_commond_param.backend           = ParseStringOptions(options, "-backend",            "--backend",            0, "");
    m_commond_param.mnn_precision     = ParseStringOptions(options, "-mnn_precision",      "--mnn_precision",      0, "");
    m_commond_param.mnn_memory        = ParseStringOptions(options, "-mnn_memory",         "--mnn_memory",         0, "");
    m_commond_param.qnn_graph_ids     = ParseStringOptions(options, "-qnn_graph_ids",      "--qnn_graph_ids",      0, "");
    m_commond_param.minn_name         = ParseStringOptions(options, "-minn_name",          "--minn_name",          0, "");

    std::string loop_type_str      = ParseStringOptions(options, "-loop_type",          "--loop_type",          0, "");
    std::string loop_num_str       = ParseStringOptions(options, "-loop_num",           "--loop_num",           0, "");
    std::string loop_duration_str  = ParseStringOptions(options, "-inference_interval", "--inference_interval", 0, "");

    if (m_commond_param.output_path.empty())
    {
        m_commond_param.output_path = "./out";
    }

    struct stat stat_buf;
    if (stat(m_commond_param.output_path.c_str(), &stat_buf) != 0)
    {
        if (MakeDirectories(m_commond_param.output_path) != Status::OK)
        {
            AURA_LOGE(m_ctx.get(), AURA_TAG, "Failed to create output folder %s\n", m_commond_param.output_path.c_str());
            return Status::ERROR;
        }
    }

    if (!m_commond_param.profiling_path.empty())
    {
        if (stat(m_commond_param.profiling_path.c_str(), &stat_buf) != 0)
        {
            if (MakeDirectories(m_commond_param.profiling_path) != Status::OK)
            {
                AURA_LOGE(m_ctx.get(), AURA_TAG, "Failed to create profiling folder %s\n", m_commond_param.profiling_path.c_str());
                return Status::ERROR;
            }
        }
    }

    if (!m_commond_param.dump_path.empty())
    {
        if (stat(m_commond_param.dump_path.c_str(), &stat_buf) != 0)
        {
            if (MakeDirectories(m_commond_param.dump_path) != Status::OK)
            {
                AURA_LOGE(m_ctx.get(), AURA_TAG, "Failed to create dump folder %s\n", m_commond_param.dump_path.c_str());
                return Status::ERROR;
            }
        }
    }

    m_commond_param.loop_type          = (loop_type_str.empty())                                      ? 0        : std::stoi(loop_type_str);
    m_commond_param.loop_num           = (loop_num_str.empty())                                       ? 1        : std::stoi(loop_num_str);
    m_commond_param.inference_interval = (loop_duration_str.empty())                                  ? 0        : std::stoi(loop_duration_str);
    m_commond_param.memory_tag_info    = (check_opt.find("--memory_tag_info")   == std::string::npos) ? DT_FALSE : DT_TRUE;
    m_commond_param.show_time          = (check_opt.find("--show_time_info")    == std::string::npos) ? DT_FALSE : DT_TRUE;
    m_commond_param.get_tensor_info    = (check_opt.find("--get_tensor_info")   == std::string::npos) ? DT_FALSE : DT_TRUE;
    m_commond_param.get_minb_info      = (check_opt.find("--get_minb_info")     == std::string::npos) ? DT_FALSE : DT_TRUE;
    m_commond_param.register_mem       = (check_opt.find("--register_mem")      == std::string::npos) ? DT_FALSE : DT_TRUE;
    m_commond_param.use_random_inputs  = (check_opt.find("--use_random_inputs") == std::string::npos) ? DT_FALSE : DT_TRUE;

    if (m_commond_param.memory_tag_info)
    {
        m_commond_param.func_memProfiler_trace_start = (MemProfilerTrace)dlsym(RTLD_NEXT, "MemProfilerTraceStart");
        m_commond_param.func_memProfiler_trace_end   = (MemProfilerTrace)dlsym(RTLD_NEXT, "MemProfilerTraceEnd");
        if ((NULL == m_commond_param.func_memProfiler_trace_start) || (NULL == m_commond_param.func_memProfiler_trace_end))
        {
            AURA_LOGE(m_ctx.get(), AURA_TAG, "can not locate func_memProfiler_trace_start or func_memProfiler_trace_end\n");
            return Status::ERROR;
        }
    }

    // param check
    if ((!m_commond_param.input_list.empty()) && (m_commond_param.use_random_inputs))
    {
        AURA_LOGE(m_ctx.get(), AURA_TAG, "'--input_list' and '--use_random_inputs' cant set together\n");
        return Status::ERROR;
    }

    // set loop
    m_commond_param.loop_num_inner = 0;
    m_commond_param.loop_num_outer = 0;

    // if net run, set loop
    if (CheckParamValid(check_opt, command_set))
    {
        if (0 == m_commond_param.loop_type)
        {
            m_commond_param.loop_num_inner = m_commond_param.loop_num;
            m_commond_param.loop_num_outer = 1;
        }
        else
        {
            m_commond_param.loop_num_inner = 1;
            m_commond_param.loop_num_outer = m_commond_param.loop_num;
        }

        if (m_commond_param.src_type.empty())
        {
            AURA_LOGE(m_ctx.get(), AURA_TAG, "src_type is invalid, '--input_data_type' is required\n");
            return Status::ERROR;
        }

        if (m_commond_param.dst_type.empty())
        {
            AURA_LOGE(m_ctx.get(), AURA_TAG, "dst_type is invalid, '--output_data_type' is required\n");
            return Status::ERROR;
        }
    }

    if (!m_commond_param.model_path.empty())
    {
        if (ParsePlatform(m_ctx.get(), m_commond_param.model_path, m_commond_param.platform) == Status::ERROR)
        {
            AURA_LOGE(m_ctx.get(), AURA_TAG, "parse platform failed.\n");
            return Status::ERROR;
        }
    }
    else if (!m_commond_param.model_container_path.empty())
    {
        if (!m_commond_param.minn_name.empty())
        {
            if (ParsePlatform(m_ctx.get(), m_commond_param.model_container_path, m_commond_param.minn_name, m_commond_param.platform) == Status::ERROR)
            {
                AURA_LOGE(m_ctx.get(), AURA_TAG, "parse platform failed.\n");
                return Status::ERROR;
            }
        }
    }

    return Status::OK;
}

Status AuraNnRun::Run()
{
    Status ret = Status::ERROR;

    if (m_commond_param.memory_tag_info)
    {
        m_commond_param.func_memProfiler_trace_start("AURA_NN_total");
    }

    if (m_commond_param.get_minb_info)
    {
        ret = GetMinbInfo(m_ctx.get(), m_commond_param);
        if (ret != Status::OK)
        {
            AURA_LOGE(m_ctx.get(), AURA_TAG, "GetMinbInfo failed!\n");
            return ret;
        }
    }

    if (m_commond_param.get_tensor_info)
    {
        ret = GetTensorInfo(m_ctx.get(), m_commond_param);
        if (ret != Status::OK)
        {
            AURA_LOGE(m_ctx.get(), AURA_TAG, "GetTensorInfo failed!\n");
            return ret;
        }
    }

    for (DT_S32 i = 0; i < m_commond_param.loop_num_outer; i++)
    {
        ret = NetRun(m_ctx.get(), m_commond_param);
        if (ret != Status::OK)
        {
            AURA_LOGE(m_ctx.get(), AURA_TAG, "NetRun failed!\n");
            return ret;
        }
    }

    if (m_commond_param.memory_tag_info)
    {
        m_commond_param.func_memProfiler_trace_end("AURA_NN_total");
    }

    return ret;
}