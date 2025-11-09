#include "aura/runtime/nn.h"
#include "aura/tools/unit_test.h"

using namespace aura;
struct ModelList
{
    std::string minn_file;
    std::string quant_type;
    std::string ref_file;
};

static Status NpRun(Context *ctx, Mat &src, Mat &dst, const std::string &minn_file, const std::string &type_str,
                    DT_S32 forward_count, DT_S32 &cur_count, DT_S32 total_count)
{
    Status ret = Status::ERROR;

    NNEngine *nn_engine = DT_NULL;
    std::shared_ptr<NNExecutor> nn_executor;
    FILE *fp = DT_NULL;
    DT_U8 *minn_data = DT_NULL;
    size_t minn_size = 0;
    MatMap input;
    MatMap output;
    DT_S32 step = forward_count / 3 + 1;
    Time start_time, create_exe_time, init_time, exe_time, delete_time;

    std::vector<std::string> perf_level = {"perf_low", "perf_normal", "perf_high"};

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
        AURA_LOGE(ctx, AURA_TAG, "GetNNEngine failed\n");
        goto EXIT;
    }

    start_time = Time::Now();

    if (cur_count > total_count / 2)
    {
        nn_executor = nn_engine->CreateNNExecutor(minn_data, minn_size, "abcdefg");
    }
    else
    {
        nn_executor = nn_engine->CreateNNExecutor(minn_file, "abcdefg");
    }

    if (DT_NULL == nn_executor)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor is null\n");
        goto EXIT;
    }

    create_exe_time = Time::Now();
    AURA_LOGD(ctx, AURA_TAG, "quantization type = %s create time = %s cur_count = %d total_count = %d\n", type_str.c_str(),
             (create_exe_time - start_time).ToString().c_str(), cur_count++, total_count);

    if (nn_executor->Initialize() != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor Initialize fail\n");
        goto EXIT;
    }

    init_time = Time::Now();
    AURA_LOGD(ctx, AURA_TAG, "quantization type = %s, init time = %s\n", type_str.c_str(), (init_time - create_exe_time).ToString().c_str());
    AURA_LOGI(ctx, AURA_TAG, "nn executor version : %s\n", nn_executor->GetVersion().c_str());

    AURA_LOGD(ctx, AURA_TAG, "input info: %s\n", TensorDescMapToString(nn_executor->GetInputs()).c_str());
    AURA_LOGD(ctx, AURA_TAG, "output info: %s\n", TensorDescMapToString(nn_executor->GetOutputs()).c_str());

    input.insert(std::make_pair("input", &src));
    output.insert(std::make_pair("InceptionV3/Predictions/Reshape_1", &dst));

    for (DT_S32 i = 0; i < forward_count; i++)
    {
        AnyParams params;
        params["perf_level"] = std::ref(perf_level[i / step]);
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
        AURA_LOGD(ctx, AURA_TAG, "quantization type = %s, perf_level = %s, forward time = %s\n", type_str.c_str(), perf_level[i / step].c_str(), exe_time.ToString().c_str());
    }

    ret = Status::OK;

EXIT:
    start_time = Time::Now();
    nn_executor.reset();
    delete_time = Time::Now() - start_time;

    AURA_FREE(ctx, minn_data);
    AURA_LOGD(ctx, AURA_TAG, "quantization type = %s, delete time = %s\n", type_str.c_str(), delete_time.ToString().c_str());

    return ret;
}

static TestResult NpTest(ModelList &model_list)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MatFactory factory(ctx);
    std::string data_path  = UnitTest::GetInstance()->GetDataPath() + "nn/";
    std::string np_path    = data_path + "np/";
    std::string input_file = data_path + "trash_1x299x299x3_u8.bin";

    DT_S32 forward_count = 5;

    Mat src = factory.GetFileMat(input_file, ElemType::U8, {299, 299, 3}, AURA_MEM_HEAP);
    Mat dst(ctx, ElemType::U8, {1, 1001}, AURA_MEM_HEAP);
    Mat ref = factory.GetFileMat(np_path + model_list.ref_file, ElemType::U8, {1, 1001});

    TestTime time_val;
    MatCmpResult cmp_result;
    TestResult result;

    DT_S32 cur_count = 0;
    DT_S32 loop_count = UnitTest::GetInstance()->IsStressMode() ? UnitTest::GetInstance()->GetStressCount() : 10;
    Status status_exec = Executor(loop_count, 2, time_val, NpRun, ctx, src, dst, np_path + model_list.minn_file, model_list.quant_type,
                                  forward_count, cur_count, loop_count);
    if (Status::OK == status_exec)
    {
        result.perf_result[model_list.quant_type] = time_val;
        result.perf_status = TestStatus::PASSED;
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "interface execute failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
        result.perf_status = TestStatus::FAILED;
        result.accu_status = TestStatus::FAILED;
        return result;
    }

    MatCompare(ctx, dst, ref, cmp_result, 1e-5);
    result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
    result.accu_result = cmp_result.ToString();

    return result;
}

NEW_TESTCASE(runtime_nn_np_v7_test)
{
    ModelList model_list =
    {
        "inception_v3_np_npu_v7.minn", "uint8", "trash_np_npu_v7_dst_1x1001_u8.bin",
    };

    TestResult result = NpTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}

NEW_TESTCASE(runtime_nn_np_v8_test)
{
    ModelList model_list =
    {
        "inception_v3_np_npu_v8.minn", "uint8", "trash_np_npu_v8_dst_1x1001_u8.bin",
    };

    TestResult result = NpTest(model_list);

    AddTestResult(result.accu_status && result.perf_status, result);
}