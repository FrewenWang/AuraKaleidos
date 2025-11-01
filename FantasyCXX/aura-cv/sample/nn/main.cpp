#include "aura/runtime/nn.h"                                    // import NN module
#include "aura/runtime/logger.h"                                // import logger module
#include <unordered_map>                                        // import unordered map

#define NN_TAG "NN_SAMPLE"                                      // define nn tag

static std::string help_info = R"(
Usage:
    Usage: sample_nn [ModelType]

Example usage:
    Usage: ./sample_nn snpe

ModelType:
    This sample contains models of aura supported, for example:

    minn_mnn             run minn mnn  sample.
    minn_np              run minn np   sample.
    minn_qnn             run minn qnn  sample.
    minn_snpe            run minn snpe sample.
    minn_xnn             run minn xnn  sample.
    minb_qnn             run minb qnn  sample.
    minb_snpe            run minb snpe sample.
)";

struct ModelInfo                                                // configuration struct for nn model
{
    std::string model_file;                                     // file path of model
    std::string key;                                            // decrypte key for model
    std::vector<std::string> input_node;                        // input node name
    std::vector<std::string> output_node;                       // output node name
    std::string input_file;                                     // file path of test input
};

struct ModelFileInfo
{
    std::string file_path;
    std::string minn_name;
};

static const std::unordered_map<std::string, ModelFileInfo> g_model_map =
{
    {
        "minn_mnn",  {"/data/local/tmp/aura/data/nn/mnn/inception_v3_mnn_gpu_v271.minn"}
    },
    {
        "minn_np",   {"/data/local/tmp/aura/data/nn/np/inception_v3_np_npu_v7.minn"},
    },
    {
        "minn_qnn",  {"/data/local/tmp/aura/data/nn/qnn/inception_v3_qnn_npu_v213.minn"},
    },
    {
        "minn_snpe", {"/data/local/tmp/aura/data/nn/snpe/inception_v3_snpe_npu_v213.minn"},
    },
    {
        "minn_xnn",  {"/data/local/tmp/aura/data/nn/xnn/inception_v3_xnn_npu_v052.minn"},
    },
    {
        "minb_snpe", {"/data/local/tmp/aura/data/nn/minb/inception_v3_snpev224_qnnv224.minb", "inception_v3_snpe_npu_v224"},
    },
    {
        "minb_qnn",  {"/data/local/tmp/aura/data/nn/minb/inception_v3_snpev224_qnnv224.minb", "inception_v3_qnn_npu_v224"},
    }
};

static aura::Status InputParser(MI_S32 argc, MI_CHAR *argv[], std::string &framework)
{
    if (argc != 2)
    {
        std::cout << help_info << std::endl;
        return aura::Status::ERROR;
    }

    std::string name = std::string(argv[1]);
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);  // find sample model
    auto it = g_model_map.find(name);
    if (it == g_model_map.end())
    {
        std::cout << help_info << std::endl;
        return aura::Status::ERROR;
    }

    framework = it->first;
    return aura::Status::OK;
}

static std::shared_ptr<aura::Context> CreateNNContext()
{
    aura::Config config;
    config.SetNNConf(MI_TRUE);                                  // enbale nn module

    std::shared_ptr<aura::Context> ctx = std::make_shared<aura::Context>(config); // create context object
    if (MI_NULL == ctx)
    {
        return MI_NULL;
    }

    if (ctx->Initialize() != aura::Status::OK)                  // initialize context
    {
        AURA_LOGE(ctx, NN_TAG, "failed to initialize context\n");
        return MI_NULL;
    }

    return ctx;
}

static aura::Status Validate(aura::Context *ctx, const aura::Mat &mat)
{
    if (!mat.IsValid())                                         // validate mat
    {
        AURA_LOGE(ctx, NN_TAG, "invalid mat\n");
        return aura::Status::ERROR;
    }
    MI_S32 id    = 0;                                           // retrive label id
    MI_F32 max_p = mat.Ptr<MI_F32>(0)[0];                       // probability of first class

    for (MI_S32 i = 1; i < mat.GetSizes().m_width; i++)
    {
        MI_F32 p = mat.Ptr<MI_F32>(0)[i];
        if (p > max_p)
        {
            max_p = p;
            id    = i;
        }
    }

    if (id != 413)                                               // validate label id
    {
        AURA_LOGE(ctx, NN_TAG, "the label should be 413(n02747177, trash bin), but it was given %d\n", id);
        return aura::Status::ERROR;
    }

    return aura::Status::OK;
}

static aura::Status Sample(aura::Context *ctx, const std::string &model_type)
{
    ModelInfo model;                                            // set nn info
    model.model_file  = g_model_map.at(model_type).file_path;
    model.key         = "abcdefg";
    model.input_node  = {"input"};
    model.output_node = {"InceptionV3/Predictions/Reshape_1"};
    model.input_file  = "/data/local/tmp/aura/data/nn/trash_1x299x299x3_f32.bin";

    aura::NNEngine *nn_engine = ctx->GetNNEngine();             // get nn engine
    if (!nn_engine)
    {
        AURA_LOGE(ctx, NN_TAG, "failed to get nn engine: %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return aura::Status::ERROR;
    }

    std::shared_ptr<aura::NNExecutor> nn_executor = MI_NULL;

    if (model_type.find("minb") != std::string::npos)
    {
        nn_executor = nn_engine->CreateNNExecutor(model.model_file, g_model_map.at(model_type).minn_name, model.key);   // creat nn executor from minb model
    }
    else
    {
        nn_executor = nn_engine->CreateNNExecutor(model.model_file, model.key);                                           // creat nn executor from minn model
    }

    if (!nn_executor)
    {
        AURA_LOGE(ctx, NN_TAG, "failed to create nn executor: %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return aura::Status::ERROR;
    }

    if (nn_executor->Initialize() != aura::Status::OK)          // initialize nn executor
    {
        AURA_LOGE(ctx, NN_TAG, "failed to initialize nn executor: %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return aura::Status::ERROR;
    }

    aura::Mat src(ctx, aura::ElemType::F32, {299,  299, 3});    // create input Mat object
    aura::Mat dst(ctx, aura::ElemType::F32, {  1, 1001, 1});    // create output Mat object
    if (!src.IsValid() || !dst.IsValid())                       // vaildate mat
    {
        AURA_LOGE(ctx, NN_TAG, "failed to create mat: %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return aura::Status::ERROR;
    }

    if (src.Load(model.input_file) != aura::Status::OK)             // load data from file to mat
    {
        AURA_LOGE(ctx, NN_TAG, "failed to load data from %s to mat\n", model.input_file.c_str());
        return aura::Status::ERROR;
    }

    aura::MatMap input  = {{model.input_node[0],  &src}};           // create input MatMap object
    aura::MatMap output = {{model.output_node[0], &dst}};           // create output MatMap object

    if (nn_executor->Forward(input, output) != aura::Status::OK)    // execute forward inference
    {
        AURA_LOGE(ctx, NN_TAG, "failed to execute forward inference: %s\n", ctx->GetLogger()->GetErrorString().c_str());
        return aura::Status::ERROR;
    }

    if (Validate(ctx, dst) != aura::Status::OK)                 // validate result
    {
        AURA_LOGE(ctx, NN_TAG, "failed to validate result\n");
        return aura::Status::ERROR;
    }

    AURA_LOGE(ctx, NN_TAG, "success\n");

    return aura::Status::OK;
}

MI_S32 main(MI_S32 argc, MI_CHAR *argv[])
{
    if (!aura::Context::IsPlatformSupported())
    {
        return -1;
    }

    std::shared_ptr<aura::Context> ctx = CreateNNContext();       // create context for sample
    if (MI_NULL == ctx)
    {
        return -1;
    }

    std::string model_type;
    if (InputParser(argc, argv, model_type) != aura::Status::OK) // parse input
    {
        AURA_LOGE(ctx, NN_TAG, "failed to parse input\n");
        return -1;
    }

    return Sample(ctx.get(), model_type) != aura::Status::OK ? 1 : 0;
}