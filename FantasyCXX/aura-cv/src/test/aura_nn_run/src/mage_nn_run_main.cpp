#include "aura_nn_run.hpp"

static std::string g_logo_str = R"(
  __  __            _____  ______   ___    __  ___    __       _____  _    _  ___    __
 |  \/  |    /\    / ____||  ____| |   \   | ||   \   | |     / __  || |  | ||   \   | |
 | \  / |   /  \  | |  __ | |__    | |\ \  | || |\ \  | |    / |__| || |  | || |\ \  | |
 | |\/| |  / /\ \ | | |_ ||  __|   | | \ \ | || | \ \ | |   /      / | |  | || | \ \ | |
 | |  | | / ____ \| |__| || |____  | |  \ \| || |  \ \| |  /   /\  \ | |__| || |  \ \| |
 |_|  |_|/_/    \_\\_____||______| |_|   \_ _||_|   \_ _| /_ _/  \ _\ \____/ |_|   \_ _|
)";

static std::string help_info = R"(
USAGE:
-----
    Usage: aura_nn_run [options] [args]

REQUIRED OPTIONS:
----------------
    --model                  <minn model>                                         Minn model file path
    --nb_model               <minb model>                                         Minb model file path
    --input_list             <input_list.txt>                                     Specify model input data path
                                                                                    1. only one input tensor               : tensor_name  := input_path
                                                                                    2. multy input tensors                 : tensor_name0 := input_path0  tensor_name1 := input_path1
                                                                                    3. multy input tensors for multy files : tensor_name0 := input_path00 tensor_name1 := input_path01
                                                                                                                             tensor_name0 := input_path10 tensor_name1 := input_path11
    --password               <model password>                                     Encryption password when converting the minn model.
    --input_data_type        <input_data_type/input_data_type.txt>                Specify model input data type.
                                                                                    1. valid values                               : u8/s8/u16/s16/u32/s32/f16/f32
                                                                                    2. specify model each input data type in txt  : tensor_name0 := f32  tensor_name1 := u8
    --output_data_type       <output_data_type/output_data_type.txt>              Specify model output data type.
                                                                                    1. valid values                               : u8/s8/u16/s16/u32/s32/f16/f32
                                                                                    2. specify model each output data type in txt : tensor_name0 := f32  tensor_name1 := u8

OPTIONAL OPTIONS:
----------------
    --help                                                                        Show help info for aura-nn-run
    --get_tensor_info                                                             Get tensor info, default: off
                                                                                    containing: elem_type, shape, scale, zero point
    --get_minb_info                                                               Get minb info, default: off
                                                                                    each minn info containing: size byte, backend, framework, framework version, user version
    --show_time_info                                                              If set this param, print net init/run time, default: off
    --loop_num               [n]                                                  Number of inference loops, default: 1
    --loop_type              [0/1]                                                Inference loop type, default: 0
                                                                                    0: model only init onece, model run multy loops specified by --loop_num
                                                                                    1: model init + model run multy loops specified by --loop_num
    --inference_interval     [n]                                                  Inference interval time, unit milliseconds, default: 0
    --perf_level             [PERF_HIGH/PERF_NORMAL/PERF_LOW]                     comm config, performance levels
                                                                                  supported   platform: qcom(snpe/qnn), mtk(np), xiaomi(xnn)
                                                                                  unsupported platform: mnn
    --profiling_level        [PROFILING_OFF/PROFILING_BASIC/PROFILING_DETAILED]   comm config, profiling levels
                                                                                  supported platform: qcom(snpe/qnn)
                                                                                  unsupported platform: mtk(np), xiaomi(xnn), mnn
    --profiling_path         [profiling path]                                     comm config, path for dump profiling
                                                                                  supported   platform: qcom(snpe/qnn), xiaomi(xnn)
                                                                                  unsupported platform: mtk(np), mnn
    --log_level              [LOG_ERROR/LOG_INFO/LOG_DEBUG]                       comm config, log levels
                                                                                  supported   platform: qcom(snpe/qnn), xiaomi(xnn)
                                                                                  unsupported platform: mtk(np), mnn
    --output_path            [output path]                                        Specify model result output path, default: ./out
    --dump_path              [dump path]                                          Specify the output path of all layer, Only supports qnn/mnn platform, Only supports qnn/mnn platform, default: off
                                                                                    If dump_path is specified, the model results and all layers result are output to this location
    --mnn_precision          [PRECISION_HIGH/PRECISION_NORMAL/PRECISION_LOW]      specialized config, Precision for the MNN executor
                                                                                  supported platform: mnn
    --mnn_memory             [MEMORY_HIGH/MEMORY_NORMAL/MEMORY_LOW]               specialized config, Memory configuration for the MNN executor
                                                                                  supported platform: mnn
    --backend                [CPU/GPU/NPU]                                        comm config, backend type
                                                                                  supported platform: qcom(snpe),  CPU/GPU/NPU (default NPU)
                                                                                                      qcom(qnn),   NPU         (default NPU)
                                                                                                      mtk(np),     NPU         (default NPU)
                                                                                                      xiaomi(xnn), NPU         (default NPU)
                                                                                                      mnn,         CPU/GPU/NPU (default CPU)

    --memory_tag_info                                                             Setting tag for model init and model inference, Used with memoy profiling tool
    --qnn_graph_ids          [n]                                                  specialized config, specify which graph to init
                                                                                  supported platform: qcom(qnn)
                                                                                  format:  1;2;3 (default empty)
    --htp_mem_step_size      [n]                                                  qcom htp config, unit in MB
                                                                                  supported platform: qcom(snpe/qnn)
                                                                                  unsupported platform: mtk(np), xiaomi(xnn), mnn
    --snpe_unsigned_pd       [true/false]                                         specialized config, whether unsigned PD is enabled
                                                                                  supported platform: qcom(snpe)
    --qnn_udo_path                                                                specialized config, Path to the User Defined Operation file
                                                                                  supported platform: qcom(qnn)
    --mnn_dump_layers        [true/false]                                         specialized config, whether dump all layers output
                                                                                  supported platform: mnn
    --mnn_tuning             [GPU_TUNING_NONE/GPU_TUNING_HEAVY/GPU_TUNING_WIDE]   specialized config, Tuning options for the MNN executor
                                                                                  supported platform: mnn
    --mnn_clmem              [GPU_MEMORY_NONE/GPU_MEMORY_BUFFER/GPU_MEMORY_IAURA] specialized config, OpenCL memory type for the MNN executor
    --register_mem                                                                comm config, whether register memory
                                                                                  supported   platform: qcom(snpe/qnn), xiaomi(xnn)
                                                                                  unsupported platform: mnn, mtk(np)
    --use_random_inputs                                                           comm config, speed test mode use random input data instead of '--input_list'
                                                                                  supported platform: qcom(snpe/qnn), mtk(np), xiaomi(xnn), mnn

Example usage:
-------------
    Show help info:
        aura-nn-run --help

    [minn] Get tensor info:
        aura-nn-run --model model.minn --password abcdefg --get_tensor_info

    [minn] Run minn model:
        aura-nn-run --model model.minn --input_list input_list.txt --password abcdefg --input_data_type f32 --output_data_type f32

    [minb] Get minb info:
        aura-nn-run --nb_model model.minb --get_minb_info

    [minb] Get minn tensor info in minb:
        aura-nn-run --nb_model model.minb --minn_name minn_name --password abcdefg --get_tensor_info

    [minb] Run minn model in minb:
        aura-nn-run --nb_model model.minb --minn_name minn_name --input_list input_list.txt --password abcdefg --input_data_type f32 --output_data_type f32
)";

using namespace aura;

AURA_INLINE DT_VOID PrintSplitLine()
{
    std::cout << std::string(86, '=') << std::endl;
}

AURA_INLINE DT_VOID PrintHeading(const std::string &heading)
{
    std::cout << std::endl << std::string(86, '=') << std::endl;
    std::cout << heading << std::endl;
    std::cout << std::string(86, '=') << std::endl;
}

AURA_INLINE DT_VOID PrintHelpInfo()
{
    std::cout << g_logo_str << std::endl;
    PrintHeading("Aura nn run Help Info");
    std::cout << help_info <<std::endl;
    PrintSplitLine();
}

DT_S32 main(DT_S32 argc, DT_CHAR *argv[])
{
    // judge the number of command parameters
    if (1 == argc)
    {
        PrintHelpInfo();
        return 0;
    }

    const std::string command(argv[1]);
    if (("-h" == command) || ("--help" == command))
    {
        PrintHelpInfo();
        return 0;
    }

    AuraNnRun nn_run;

    if (nn_run.Initialize() != Status::OK)
    {
        std::cout << "aura-nn-run Initialize failed." << std::endl;
        return -1;
    }

    if (nn_run.ParseCommandLine(argc, argv) != Status::OK)
    {
        std::cout << "ParseCommandLine failed." << std::endl;
        return -1;
    }

    if (nn_run.Run() != Status::OK)
    {
        std::cout << "NN Run failed." << std::endl;
        return -1;
    }

    return 0;
}