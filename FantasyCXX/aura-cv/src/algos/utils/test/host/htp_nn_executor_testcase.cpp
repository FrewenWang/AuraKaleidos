#include "aura/algos/htp_nn_executor.h"
#include "htp_nn_executor_impl.hpp"
#include "aura/tools/unit_test.h"

#if (defined(AURA_ENABLE_NN) && defined(AURA_ENABLE_HEXAGON))

using namespace aura;

struct ModelList
{
    std::string minn_file;
    std::string ref_file;
};

static Status HtpNNExecutorRun(Context *ctx, Mat &src, Mat &dst, std::string &minn_file, MI_S32 forward_count,
                               MI_S32 &cur_count, MI_S32 total_count)
{
    Status ret = Status::ERROR;

    std::unordered_map<std::string, Mat> input;
    std::unordered_map<std::string, Mat> output;
    NNConfig config;
    Time start_time, create_exe_time, init_time, exe_time, delete_time;

    HexagonEngine *engine = MI_NULL;
    aura::HtpNNExecutor htp_nn_executor(ctx);
    MI_S32 step = forward_count / 3 + 1;
    std::vector<std::string> htp_perf_level = {"perf_low", "perf_normal", "perf_high"};
    std::vector<std::string> hmx_perf_level = {"perf_low", "perf_normal", "perf_high"};

    engine = ctx->GetHexagonEngine();
    if (MI_NULL == engine)
    {
        AURA_LOGE(ctx, AURA_TAG, "engine is null\n");
        return ret;
    }

    engine->SetPower(aura::HexagonPowerLevel::TURBO, MI_FALSE);

    start_time = Time::Now();
    config["perf_level"] = htp_perf_level[cur_count % 3];
    if (htp_nn_executor.Initialize(minn_file, "abcdefg", config) != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "htp_nn_executor Initialize fail\n");
        goto EXIT;
    }

    init_time = Time::Now();
    AURA_LOGI(ctx, AURA_TAG, "nn executor version : %s\n", htp_nn_executor.GetVersion().c_str());
    AURA_LOGD(ctx, AURA_TAG, "backend = %s, Initialize time = %s cur_count = %d total_count = %d\n", "NPU",
              (init_time - start_time).ToString().c_str(), cur_count++, total_count);

    AURA_LOGD(ctx, AURA_TAG, "input info: %s\n", TensorDescMapToString(htp_nn_executor.GetInputs()).c_str());
    AURA_LOGD(ctx, AURA_TAG, "output info: %s\n", TensorDescMapToString(htp_nn_executor.GetOutputs()).c_str());

    input.insert(std::make_pair("input", src));
    output.insert(std::make_pair("InceptionV3/Predictions/Reshape_1", dst));

    for (MI_S32 iter = 0; iter < forward_count; iter++)
    {
        start_time = Time::Now();
        HexagonRpcParam rpc_param(ctx);
        HexagonProfiling profiling;

        rpc_param.ResetBuffer();
        if (rpc_param.Set(input, output, htp_nn_executor.GetDeviceAddr()) != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "rpc_param Set fail\n");
            goto EXIT;
        }

        if (engine->Run(AURA_HTP_NN_EXCUTOR_PACKAGE_NAME, AURA_HTP_NN_EXCUTOR_TEST_RUN, rpc_param, &profiling) != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "engine Run fail\n");
            goto EXIT;
        }

        exe_time = Time::Now() - start_time;

        AURA_LOGD(ctx, AURA_TAG, "backend = %s, htp_perf_level = %s, hmx_perf_level = %s, forward time = %s\n", "NPU",
                  htp_perf_level[(cur_count - 1) % 3].c_str(), hmx_perf_level[iter / step].c_str(), exe_time.ToString().c_str());
        AURA_LOGD(ctx, AURA_TAG, "profiling: %s\n", HexagonProfilingToString(profiling).c_str());
    }

    ret = Status::OK;
EXIT:
    start_time = Time::Now();
    htp_nn_executor.DeInitialize();
    delete_time = Time::Now() - start_time;

    engine->SetPower(aura::HexagonPowerLevel::STANDBY, MI_FALSE);
    AURA_LOGD(ctx, AURA_TAG, "backend = %s, delete time = %s\n", "NPU", delete_time.ToString().c_str());

    return ret;
}

static TestResult HtpNNExecutorTest(ModelList model_list)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MatFactory factory(ctx);
    std::string data_path  = UnitTest::GetInstance()->GetDataPath() + "nn/";
    std::string qnn_path   = data_path + "qnn/";
    std::string input_file = data_path + "trash_1x299x299x3_u8.bin";

    MI_S32 forward_count = 5;

    Mat dst(ctx, ElemType::U8, Sizes3(1, 1, 1001));
    Mat src     = factory.GetFileMat(input_file, ElemType::U8, {299, 299, 3});
    Mat dst_ref = factory.GetFileMat(qnn_path + model_list.ref_file, ElemType::U8, Sizes3(1, 1, 1001));

    TestTime time_val;

    TestResult result;
    MatCmpResult cmp_result;

    MI_S32 cur_count = 0;
    MI_S32 loop_count = UnitTest::GetInstance()->IsStressMode() ? UnitTest::GetInstance()->GetStressCount() : 10;

    Status status_exec = Executor(loop_count, 2, time_val, HtpNNExecutorRun, ctx, src, dst, qnn_path + model_list.minn_file,
                                  forward_count, cur_count, loop_count);
    if (Status::OK == status_exec)
    {
        result.perf_result["npu"] = time_val;
        result.perf_status = TestStatus::PASSED;
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "interface execute fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        result.perf_status = TestStatus::FAILED;
        result.accu_status = TestStatus::FAILED;
        goto EXIT;
    }

    MatCompare(ctx, dst, dst_ref, cmp_result, 1e-2);
    result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
    result.accu_result = cmp_result.ToString();

EXIT:
    return result;
}

NEW_TESTCASE(algos_utils_htp_nn_executor_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v231.minn", "trash_qnn_npu_v230_dst_1x1001_u8.bin"
    };

    TestResult result = HtpNNExecutorTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}
#endif // (defined(AURA_ENABLE_NN) && defined(AURA_ENABLE_HEXAGON))
