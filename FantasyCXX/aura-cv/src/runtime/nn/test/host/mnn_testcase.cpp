#include "aura/runtime/nn.h"
#include "aura/tools/unit_test.h"

using namespace aura;

static Status MnnTest(Context *ctx, Mat &src, Mat &dst, const std::string &minn_file,
                      DT_S32 forward_count, DT_S32 &cur_count, DT_S32 total_count)
{
    Status ret = Status::ERROR;

    NNEngine *nn_engine = DT_NULL;
    std::shared_ptr<NNExecutor> nn_executor = DT_NULL;
    FILE *fp = DT_NULL;
    DT_U8 *minn_data = DT_NULL;
    size_t minn_size = 0;
    MatMap input;
    MatMap output;
    Time start_time, create_exe_time, init_time, exe_time, delete_time;

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
        return ret;
    }

    start_time = Time::Now();

    NNConfig config;
    config["backend"]       = "GPU";
    config["mnn_precision"] = "PRECISION_HIGH";
    config["mnn_memory"]    = "MEMORY_NORMAL";
    config["mnn_tuning"]    = "GPU_TUNING_WIDE";
    config["mnn_clmem"]     = "GPU_MEMORY_IAURA";

    if (cur_count > total_count / 2)
    {
        nn_executor = nn_engine->CreateNNExecutor(minn_data, minn_size, "abcdefg", config);
    }
    else
    {
        nn_executor = nn_engine->CreateNNExecutor(minn_file, "abcdefg", config);
    }

    if (DT_NULL == nn_executor)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor is null\n");
        goto EXIT;
    }

    AURA_LOGI(ctx, AURA_TAG, "nn executor version : %s\n", nn_executor->GetVersion().c_str());

    create_exe_time = Time::Now();
    AURA_LOGD(ctx, AURA_TAG, "create time = %s cur_count = %d total_count = %d\n", (create_exe_time - start_time).ToString().c_str(),
              cur_count++, total_count);

    if (nn_executor->Initialize() != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "nn_executor Initialize fail\n");
        goto EXIT;
    }

    init_time = Time::Now();
    AURA_LOGD(ctx, AURA_TAG, "init time = %s\n", (init_time - create_exe_time).ToString().c_str());
    AURA_LOGD(ctx, AURA_TAG, "input info: %s\n", TensorDescMapToString(nn_executor->GetInputs()).c_str());
    AURA_LOGD(ctx, AURA_TAG, "output info: %s\n", TensorDescMapToString(nn_executor->GetOutputs()).c_str());

    input.insert(std::make_pair("input", &src));
    output.insert(std::make_pair("InceptionV3/Predictions/Reshape_1", &dst));

    for (DT_S32 i = 0; i < forward_count; i++)
    {
        start_time = Time::Now();
        ret = nn_executor->Forward(input, output);
        if (ret != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "nn_executor Forward fail\n");
            return Status::ERROR;
        }
        exe_time = Time::Now() - start_time;
        AURA_LOGD(ctx, AURA_TAG, "forward time = %s\n", exe_time.ToString().c_str());
    }

    ret = Status::OK;
EXIT:
    start_time = Time::Now();
    nn_executor.reset();
    delete_time = Time::Now() - start_time;

    AURA_FREE(ctx, minn_data);
    AURA_LOGD(ctx, AURA_TAG, "delete time = %s\n", delete_time.ToString().c_str());

    return ret;
}

NEW_TESTCASE(runtime_nn_mnn_test)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();

    MatFactory factory(ctx);
    std::string data_path   = UnitTest::GetInstance()->GetDataPath() + "nn/";
    std::string input_file  = data_path + "trash_1x299x299x3_f32.bin";
    std::string model_names = "inception_v3_mnn_gpu_v271.minn";
    std::string minn_file   = data_path + "mnn/" + model_names;

    DT_S32 forward_count = 5;

    Mat src = factory.GetFileMat(input_file, ElemType::F32, {299, 299, 3});
    Mat dst(ctx, ElemType::F32, {1, 1, 1001});
    TestResult result;

    {
        TestTime time_val;
        DT_S32 cur_count = 0;
        DT_S32 loop_count  = UnitTest::GetInstance()->IsStressMode() ? UnitTest::GetInstance()->GetStressCount() : 10;
        Status status_exec = Executor(loop_count, 2, time_val, MnnTest, ctx, src, dst, minn_file, forward_count, cur_count, loop_count);
        if (Status::OK == status_exec)
        {
            result.perf_result["BackendType::MNN_GPU"] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "interface execute failed, %s\n", ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        DT_S32 id    = -1;
        DT_F32 max_p = 0;
        DT_F32 sum_p = 0;
        for (DT_S32 i = 0; i < 1001; i++)
        {
            sum_p += exp(dst.Ptr<DT_F32>(0)[i]);
        }
        for (DT_S32 i = 0; i < 1001; i++)
        {
            DT_F32 p = exp(dst.Ptr<DT_F32>(0)[i]) / sum_p;
            if (max_p < p)
            {
                max_p = p;
                id    = i;
            }
        }

        // trash label is 413
        result.accu_status = (413 == id) ? TestStatus::PASSED : TestStatus::FAILED;
        if (result.accu_status != TestStatus::PASSED)
        {
            AURA_LOGE(ctx, AURA_TAG, "result = %d\n", id);
        }
    }

EXIT:
    AddTestResult(result.accu_status && result.perf_status, result);
}