#include "aura/runtime/nn.h"
#include "aura/tools/unit_test.h"

using namespace aura;

struct ModelList
{
    std::string minn_file;
    std::string ref_file;
};

static Status QnnRun(Context *ctx, Mat &src, Mat &dst, std::string &minn_file, DT_S32 forward_count,
                     DT_S32 &cur_count, DT_S32 total_count, DT_F32 &forward_time, DT_BOOL do_register = DT_FALSE)
{
    Status ret = Status::ERROR;

    NNEngine *nn_engine = DT_NULL;
    std::shared_ptr<NNExecutor> nn_executor = DT_NULL;
    FILE *fp = DT_NULL;
    DT_U8 *minn_data = DT_NULL;
    size_t minn_size = 0;
    MatMap input;
    MatMap output;
    NNConfig config;
    DT_S32 step = forward_count / 3 + 1;
    Time start_time, create_exe_time, init_time, exe_time, delete_time;

    std::vector<std::string> htp_perf_level = {"perf_low", "perf_normal", "perf_high"};
    std::vector<std::string> hmx_perf_level = {"perf_low", "perf_normal", "perf_high"};

    if (cur_count > total_count / 2)
    {
        fp = fopen(minn_file.c_str(), "rb");
        if (DT_NULL == fp)
        {
            AURA_LOGE(ctx, AURA_TAG, "fopen %s failed\n", minn_file.c_str());
            return ret;
        }

        fseek(fp, 0, SEEK_END);
        minn_size = ftell(fp);

        minn_data = static_cast<DT_U8*>(AURA_ALLOC(ctx, minn_size));
        if (DT_NULL == minn_data)
        {
            AURA_LOGE(ctx, AURA_TAG, "alloc %ld memory failed\n", minn_size);
            fclose(fp);
            return ret;
        }

        fseek(fp, 0, SEEK_SET);
        size_t read_size = fread(minn_data, 1, minn_size, fp);
        if (read_size < minn_size)
        {
            AURA_LOGE(ctx, AURA_TAG, "read file fail, file size:%d, read size:%d\n", minn_size, read_size);
            AURA_FREE(ctx, minn_data);
            fclose(fp);
            return ret;
        }

        fclose(fp);
    }

    nn_engine = ctx->GetNNEngine();
    if (DT_NULL == nn_engine)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_engine is null\n");
        return ret;
    }

    start_time = Time::Now();

    config["perf_level"] = htp_perf_level[cur_count % 3];

    if (cur_count > total_count / 2)
    {
        config["htp_async_call"] = "false";
        nn_executor = nn_engine->CreateNNExecutor(minn_data, minn_size, "abcdefg", config);
    }
    else
    {
        config["qnn_budget"] = "2";
        nn_executor = nn_engine->CreateNNExecutor(minn_file, "abcdefg", config);
    }

    if (DT_NULL == nn_executor)
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

    if (DT_TRUE == do_register)
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

    for (DT_S32 iter = 0; iter < forward_count; iter++)
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

        forward_time += (DT_F32)exe_time.AsMilliSec();
        AURA_LOGD(ctx, AURA_TAG, "backend = %s, htp_perf_level = %s, hmx_perf_level = %s, forward time = %s\n", "NPU",
                  htp_perf_level[(cur_count - 1) % 3].c_str(), hmx_perf_level[iter / step].c_str(), exe_time.ToString().c_str());

    }

    if (DT_TRUE == do_register)
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

    AURA_FREE(ctx, minn_data);
    AURA_LOGD(ctx, AURA_TAG, "backend = %s, delete time = %s\n", "NPU", delete_time.ToString().c_str());

    return ret;
}

static TestResult QnnTest(ModelList model_list)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MatFactory factory(ctx);
    std::string data_path  = UnitTest::GetInstance()->GetDataPath() + "nn/";
    std::string qnn_path   = data_path + "qnn/";
    std::string input_file = data_path + "trash_1x299x299x3_u8.bin";

    DT_S32 forward_count = 5;

    Mat dst(ctx, ElemType::U8, Sizes3(1, 1, 1001));
    Mat src     = factory.GetFileMat(input_file, ElemType::U8, {299, 299, 3});
    Mat dst_ref = factory.GetFileMat(qnn_path + model_list.ref_file, ElemType::U8, Sizes3(1, 1, 1001));

    TestTime time_val;

    DT_F32 time_do_register = 0;
    DT_F32 time_no_register = 0;

    TestResult result;
    MatCmpResult cmp_result;

    DT_S32 cur_count = 0;
    DT_S32 loop_count = UnitTest::GetInstance()->IsStressMode() ? UnitTest::GetInstance()->GetStressCount() : 10;

    Status status_exec = Executor(loop_count, 2, time_val, QnnRun, ctx, src, dst, qnn_path + model_list.minn_file,
                                  forward_count, cur_count, loop_count, time_no_register, DT_FALSE);
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

    status_exec = Executor(loop_count, 2, time_val, QnnRun, ctx, src, dst, qnn_path + model_list.minn_file,
                           forward_count, cur_count, loop_count, time_do_register, DT_TRUE);
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

NEW_TESTCASE(runtime_nn_qnn_v213_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v213.minn", "trash_qnn_npu_v213_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v214_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v214.minn", "trash_qnn_npu_v214_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v217_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v217.minn", "trash_qnn_npu_v217_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v219_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v219.minn", "trash_qnn_npu_v219_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v220_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v220.minn", "trash_qnn_npu_v220_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v221_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v221.minn", "trash_qnn_npu_v221_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v222_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v222.minn", "trash_qnn_npu_v222_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v223_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v223.minn", "trash_qnn_npu_v223_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v224_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v224.minn", "trash_qnn_npu_v224_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v225_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v225.minn", "trash_qnn_npu_v225_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v226_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v226.minn", "trash_qnn_npu_v226_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v227_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v227.minn", "trash_qnn_npu_v227_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v228_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v228.minn", "trash_qnn_npu_v228_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v229_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v229.minn", "trash_qnn_npu_v229_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v230_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v230.minn", "trash_qnn_npu_v230_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v231_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v231.minn", "trash_qnn_npu_v231_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_qnn_v233_test)
{
    ModelList model_list =
    {
        "inception_v3_qnn_npu_v233.minn", "trash_qnn_npu_v233_dst_1x1001_u8.bin"
    };

    TestResult result = QnnTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}