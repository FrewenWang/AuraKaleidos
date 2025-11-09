#ifndef AURA_OPS_MATRIX_TRANSPOSE_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_TRANSPOSE_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(TransposeParam,
                ElemType,  elem_type,
                MatSize,   mat_size,
                OpTarget,  target);

AURA_INLINE Status CvTranspose(Mat &src, Mat &dst)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    if (cv_src.channels() > 4 || cv_dst.channels() > 4)
    {
        return Status::ERROR;
    }
    cv::transpose(cv_src, cv_dst);
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
#endif

    return Status::OK;
}

class MatrixTranspose : public TestBase<TransposeParam::TupleTable, TransposeParam::Tuple>
{
public:
    MatrixTranspose(Context *ctx, TransposeParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgba",  ElemType::U8, {512, 512, 4});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in MatrixTranspose\n");
        }
    }

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // get next param set
        TransposeParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());

        Sizes3 src_sz = run_param.mat_size.m_sizes;
        Sizes3 dst_sz(src_sz.m_width, src_sz.m_height, src_sz.m_channel);

        // Create src mat
        Mat src = m_factory.GetDerivedMat(1, 0, run_param.elem_type, src_sz,
                                               AURA_MEM_DEFAULT, run_param.mat_size.m_strides);

        // Create dst mat
        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, dst_sz,
                                              AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.elem_type, dst_sz,
                                              AURA_MEM_DEFAULT, run_param.mat_size.m_strides);

        DT_S32 loop_count = stress_count ? stress_count : (TargetType::NONE == run_param.target.m_type ? 5 : 10);
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = std::string();
        result.input  = src_sz.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = dst_sz.ToString();

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, ITranspose, m_ctx, src, dst, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            result.accu_benchmark = "OpenCV::Transpose";
            status_exec = Executor(10, 2, time_val, CvTranspose, src, ref);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvTranspose execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = ITranspose(m_ctx, src, ref, OpTarget::None());
            result.accu_benchmark = "Transpose(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (MatCompare(m_ctx, dst, ref, cmp_result, 0) == Status::OK)
        {
            result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
            result.accu_result = cmp_result.ToString();
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail\n");
        }

EXIT:
        test_case->AddTestResult(result.accu_status, result);
        // release mat
        m_factory.PutMats(src, dst, ref);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_TRANSPOSE_UNIT_TEST_HPP__