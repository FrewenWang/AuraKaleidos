#include "aura/runtime/nn.h"
#include "aura/tools/unit_test.h"

using namespace aura;

struct ModelList
{
    std::string sdk;
    std::string minn_name;
    std::string backend;
    std::string ref_file;
};

struct MinbTestInfo
{
    std::string minb_file;
    std::vector<ModelList> model_list;
};

static Status SnpeRun(Context *ctx, Mat &src, Mat &dst, std::string backend, std::string &minb_file, std::string &minn_name,
                      MI_S32 forward_count, MI_S32 &cur_count, MI_S32 total_count)
{
    Status ret = Status::ERROR;

    NNEngine *nn_engine = MI_NULL;
    std::shared_ptr<NNExecutor> nn_executor = MI_NULL;
    MatMap input;
    MatMap output;
    Time start_time, create_exe_time, init_time, exe_time, delete_time;
    NNConfig config;
    config["log_level"] = "LOG_DEBUG";
    config["backend"] = backend;
    config["htp_async_call"] = "false";

    nn_engine = ctx->GetNNEngine();
    if (MI_NULL== nn_engine)
    {
        AURA_LOGE(ctx, AURA_TAG, "GetNNEngine fail\n");
        goto EXIT;
    }

    start_time = Time::Now();
    nn_executor = nn_engine->CreateNNExecutor(minb_file, minn_name, "abcdefg", config);
    if (MI_NULL == nn_executor)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor is null\n");
        goto EXIT;
    }

    AURA_LOGI(ctx, AURA_TAG, "nn executor version : %s\n", nn_executor->GetVersion().c_str());

    create_exe_time = Time::Now();
    AURA_LOGD(ctx, AURA_TAG, "backend = %s create time = %s cur_count = %d total_count = %d\n", backend.c_str(),
              (create_exe_time - start_time).ToString().c_str(), cur_count++, total_count);

    if (nn_executor->Initialize() != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor Initialize fail\n");
        goto EXIT;
    }

    init_time = Time::Now();
    AURA_LOGD(ctx, AURA_TAG, "backend = %s init_time = %s\n", backend.c_str(), (init_time - create_exe_time).ToString().c_str());
    AURA_LOGD(ctx, AURA_TAG, "input info: %s\n", TensorDescMapToString(nn_executor->GetInputs()).c_str());
    AURA_LOGD(ctx, AURA_TAG, "output info: %s\n", TensorDescMapToString(nn_executor->GetOutputs()).c_str());

    input.insert(std::make_pair("input", &src));
    output.insert(std::make_pair("InceptionV3/Predictions/Reshape_1", &dst));

    for (MI_S32 i = 0; i < forward_count; i++)
    {
        start_time = Time::Now();
        ret = nn_executor->Forward(input, output);
        if (ret != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "nn_executor Forward fail\n");
            goto EXIT;
        }
        exe_time = Time::Now() - start_time;
        AURA_LOGD(ctx, AURA_TAG, "backend = %s forward time = %s\n", backend.c_str(), exe_time.ToString().c_str());
    }

    ret = Status::OK;
EXIT:
    start_time = Time::Now();
    nn_executor.reset();
    delete_time = Time::Now() - start_time;

    AURA_LOGD(ctx, AURA_TAG, "backend = %s, delete time = %s\n", backend.c_str(), delete_time.ToString().c_str());
    return ret;
}

static TestResult SnpeTest(std::string &minb_file, std::string &minn_name, std::string &backend, std::string &ref_file)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MatFactory factory(ctx);
    std::string data_path  = UnitTest::GetInstance()->GetDataPath() + "nn/";
    std::string snpe_path  = data_path + "snpe/";
    std::string minb_path  = data_path + "minb/";
    std::string input_file = data_path + "trash_1x299x299x3_u8.bin";

    MI_U32 forward_count = 5;

    Mat src, dst_ref;
    Mat dst(ctx, ElemType::U8, {1, 1001}, AURA_MEM_DMA_BUF_HEAP);
    src = factory.GetFileMat(input_file, ElemType::U8, {299, 299, 3});

    TestResult result;
    TestTime time_val;
    MatCmpResult cmp_result;
    MI_F64 cmp_eps = ("GPU" == backend) ? 1 : 1e-5;

    dst_ref = factory.GetFileMat(snpe_path + ref_file, ElemType::U8, {1, 1001});

    MI_S32 cur_count = 0;
    MI_S32 loop_count = UnitTest::GetInstance()->IsStressMode() ? UnitTest::GetInstance()->GetStressCount() : 10;

    Status status_exec = Executor(loop_count, 2, time_val, SnpeRun, ctx, src, dst, backend, minb_path + minb_file, minn_name,
                                  forward_count, cur_count, loop_count);
    if (Status::OK == status_exec)
    {
        result.perf_result[backend] = time_val;
        result.perf_status = TestStatus::PASSED;
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "interface execute fail, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        result.perf_status = TestStatus::FAILED;
        result.accu_status = TestStatus::FAILED;
        goto EXIT;
    }

    MatCompare(ctx, dst, dst_ref, cmp_result, cmp_eps);
    result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
    result.accu_result = cmp_result.ToString();

EXIT:
    return result;
}

static Status QnnRun(Context *ctx, Mat &src, Mat &dst, std::string &minn_file, std::string &minn_name, MI_S32 forward_count,
                    MI_S32 &cur_count, MI_S32 total_count, MI_F32 &forward_time, MI_BOOL do_register = MI_FALSE)
{
    Status ret = Status::ERROR;

    NNEngine *nn_engine = MI_NULL;
    std::shared_ptr<NNExecutor> nn_executor = MI_NULL;
    MatMap input;
    MatMap output;
    NNConfig config;
    MI_S32 step = forward_count / 3 + 1;
    Time start_time, create_exe_time, init_time, exe_time, delete_time;

    std::vector<std::string> htp_perf_level = {"perf_low", "perf_normal", "perf_high"};
    std::vector<std::string> hmx_perf_level = {"perf_low", "perf_normal", "perf_high"};

    nn_engine = ctx->GetNNEngine();
    if (MI_NULL == nn_engine)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_engine is null\n");
        return ret;
    }

    start_time = Time::Now();

    config["perf_level"] = htp_perf_level[cur_count % 3];

    nn_executor = nn_engine->CreateNNExecutor(minn_file, minn_name, "abcdefg", config);

    if (MI_NULL == nn_executor)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor is null\n");
        goto EXIT;
    }

    AURA_LOGI(ctx, AURA_TAG, "nn executor version : %s\n", nn_executor->GetVersion().c_str());

    create_exe_time = Time::Now();
    AURA_LOGD(ctx, AURA_TAG, "backend = %s, create time = %s cur_count = %d total_count = %d\n", "NPU",
              (create_exe_time - start_time).ToString().c_str(), cur_count++, total_count);

    if (nn_executor->Initialize() != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor Initialize fail\n");
        goto EXIT;
    }

    init_time = Time::Now();
    AURA_LOGD(ctx, AURA_TAG, "backend = %s, init_time = %s\n", "NPU", (init_time - create_exe_time).ToString().c_str());
    AURA_LOGD(ctx, AURA_TAG, "input info: %s\n", TensorDescMapToString(nn_executor->GetInputs()).c_str());
    AURA_LOGD(ctx, AURA_TAG, "output info: %s\n", TensorDescMapToString(nn_executor->GetOutputs()).c_str());

    input.insert(std::make_pair("input", &src));
    output.insert(std::make_pair("InceptionV3/Predictions/Reshape_1", &dst));

    if (MI_TRUE == do_register)
    {
        AnyParams params;
        params["input_matmap"] = input;
        params["output_matmap"] = output;
        ret = nn_executor->Update("register_mem", params);
        if (ret != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "qnn register failed\n");
            goto EXIT;
        }
    }

    for (MI_S32 iter = 0; iter < forward_count; iter++)
    {
        AnyParams params;
        params["perf_level"]         = std::ref(htp_perf_level[(cur_count - 1) % 3]);
        params["qnn_hmx_perf_level"] = std::ref(hmx_perf_level[iter / step]);
        ret = nn_executor->Update("update_perf", params);
        if (ret != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "Update update_perf fail\n");
            goto EXIT;
        }

        start_time = Time::Now();
        ret = nn_executor->Forward(input, output);
        if (ret != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "nn_executor Forward fail\n");
            goto EXIT;
        }
        exe_time = Time::Now() - start_time;

        forward_time += (MI_F32)exe_time.AsMilliSec();
        AURA_LOGD(ctx, AURA_TAG, "backend = %s, htp_perf_level = %s, hmx_perf_level = %s, forward time = %s\n", "NPU",
                  htp_perf_level[(cur_count - 1) % 3].c_str(), hmx_perf_level[iter / step].c_str(), exe_time.ToString().c_str());

    }

    if (MI_TRUE == do_register)
    {
        AnyParams params;
        params["input_matmap"] = input;
        params["output_matmap"] = output;
        ret = nn_executor->Update("deregister_mem", params);
        if (ret != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "qnn deregister failed\n");
            goto EXIT;
        }
    }

    ret = Status::OK;
EXIT:
    start_time = Time::Now();
    nn_executor.reset();
    delete_time = Time::Now() - start_time;

    AURA_LOGD(ctx, AURA_TAG, "backend = %s, delete time = %s\n", "NPU", delete_time.ToString().c_str());

    return ret;
}

static TestResult QnnTest(std::string &minb_file, std::string &minn_name, std::string &ref_file)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MatFactory factory(ctx);
    std::string data_path  = UnitTest::GetInstance()->GetDataPath() + "nn/";
    std::string qnn_path   = data_path + "qnn/";
    std::string minb_path  = data_path + "minb/";
    std::string input_file = data_path + "trash_1x299x299x3_u8.bin";

    MI_S32 forward_count = 5;

    Mat dst(ctx, ElemType::U8, Sizes3(1, 1, 1001), AURA_MEM_DMA_BUF_HEAP);
    Mat src     = factory.GetFileMat(input_file, ElemType::U8, {299, 299, 3});
    Mat dst_ref = factory.GetFileMat(qnn_path + ref_file, ElemType::U8, Sizes3(1, 1, 1001));

    TestTime time_val;

    MI_F32 time_do_register = 0;
    MI_F32 time_no_register = 0;

    TestResult result;
    MatCmpResult cmp_result;

    MI_S32 cur_count = 0;
    MI_S32 loop_count = UnitTest::GetInstance()->IsStressMode() ? UnitTest::GetInstance()->GetStressCount() : 10;

    Status status_exec = Executor(loop_count, 2, time_val, QnnRun, ctx, src, dst, minb_path + minb_file, minn_name,
                                  forward_count, cur_count, loop_count, time_no_register, MI_FALSE);
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

    status_exec = Executor(loop_count, 2, time_val, QnnRun, ctx, src, dst, minb_path + minb_file, minn_name,
                                forward_count, cur_count, loop_count, time_do_register, MI_TRUE);
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

    if (time_do_register > time_no_register)
    {
        AURA_LOGE(ctx, AURA_TAG, "time_do_register=%f time_no_register=%f\n", time_do_register, time_no_register);
        result.perf_status = TestStatus::FAILED;
    }

EXIT:
    return result;
}

NEW_TESTCASE(runtime_nn_minb_test)
{
    MinbTestInfo minb_test_info =
    {
        "inception_v3_snpev224_qnnv224.minb",
        {
            ModelList{"snpe", "inception_v3_snpe_npu_v224", "NPU", "trash_snpe_npu_v224_dst_1x1001_u8.bin"},
            ModelList{"qnn",  "inception_v3_qnn_npu_v224",  "NPU", "trash_qnn_npu_v224_dst_1x1001_u8.bin"}
        }
    };

    MI_U32 test_num = minb_test_info.model_list.size();
    for (MI_U32 idx = 0; idx < test_num; idx++)
    {
        TestResult result;
        if ("snpe" == minb_test_info.model_list[idx].sdk)
        {
            result = SnpeTest(minb_test_info.minb_file, minb_test_info.model_list[idx].minn_name, minb_test_info.model_list[idx].backend, minb_test_info.model_list[idx].ref_file);
        }
        else if ("qnn" == minb_test_info.model_list[idx].sdk)
        {
            result = QnnTest(minb_test_info.minb_file, minb_test_info.model_list[idx].minn_name, minb_test_info.model_list[idx].ref_file);
        }
        AddTestResult(result.accu_status && result.perf_status, result);
    }
}
