#ifndef AURA_OPS_MATRIX_CONVERT_TO_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_CONVERT_TO_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(ConvertToParam,
                ElemType,  src_type,
                ElemType,  dst_type,
                MatSize,   mat_size,
                MI_F32,    alpha,
                MI_F32,    beta,
                OpTarget,  target);

AURA_INLINE Status CvConvertTo(Mat &src, Mat &dst, MI_F32 alpha, MI_F32 beta)
{

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);

    const Sizes3 sz = src.GetSizes();
    MI_S32 dst_type = ElemTypeToOpencv(dst.GetElemType(), sz.m_channel);
    cv_src.convertTo(cv_dst, dst_type, alpha, beta);
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(alpha);
    AURA_UNUSED(beta);
#endif

    return Status::OK;
}

class MatrixConvertToTest : public TestBase<ConvertToParam::TupleTable, ConvertToParam::Tuple>
{
public:
    MatrixConvertToTest(Context *ctx, ConvertToParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // Get next param set
        ConvertToParam run_param(GetParam((index)));
#if !defined(AURA_ENABLE_NEON_FP16)
        if (TargetType::NEON == run_param.target.m_type && (ElemType::F16 == run_param.src_type || 
                                                              ElemType::F16 == run_param.dst_type))
        {
            return 0;
        }
#endif

        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());

        MI_S32 mem_type = AURA_MEM_DEFAULT;
        Mat src = m_factory.GetRandomMat(0, 512, run_param.src_type, run_param.mat_size.m_sizes, mem_type, run_param.mat_size.m_strides);
        Mat dst = m_factory.GetEmptyMat(run_param.dst_type, run_param.mat_size.m_sizes, mem_type, run_param.mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.dst_type, run_param.mat_size.m_sizes, mem_type, run_param.mat_size.m_strides);

        MI_S32 loop_count = stress_count ? stress_count : (TargetType::NONE == run_param.target.m_type ? 5 : 10);
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = std::string("(alpha:" ) + std::to_string(run_param.alpha) + std::string("beta:") + std::to_string(run_param.beta) + std::string(")");
        result.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.src_type);
        result.output = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.dst_type);

        // run interface

        Status status_exec = Executor(loop_count, 2, time_val, IConvertTo, m_ctx, src, dst,
                                      run_param.alpha, run_param.beta, run_param.target);

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
            result.accu_benchmark = "OpenCV::ConvertTo";
            status_exec = Executor(10, 2, time_val, CvConvertTo, src, ref, run_param.alpha, run_param.beta);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvConvertTo execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            result.accu_benchmark = "ConvertTo(target::none)";
            status_exec = IConvertTo(m_ctx, src, ref, run_param.alpha, run_param.beta, OpTarget::None());

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (MatCompare(m_ctx, dst, ref, cmp_result, 1.0, 0.2) == Status::OK)
        {
            result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
            result.accu_result = cmp_result.ToString();
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail\n");
        }

EXIT:
        test_case->AddTestResult(result.perf_status && result.accu_status, result);
        m_factory.PutAllMats();

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_CONVERT_TO_UNIT_TEST_HPP__