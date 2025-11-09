#ifndef AURA_OPS_MATRIX_MUL_SPECTRUMS_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_MUL_SPECTRUMS_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(MulSpectrumsParam,
                MatSize,  mat_size,
                DT_BOOL,  conj_src1,
                OpTarget, target);

AURA_INLINE Status OpenCVMulSpectrums(Mat &src0, Mat &src1, Mat &dst, DT_BOOL conj_src1)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src0 = MatToOpencv(src0);
    cv::Mat cv_src1 = MatToOpencv(src1);
    cv::Mat cv_dst  = MatToOpencv(dst);
    cv::mulSpectrums(cv_src0, cv_src1, cv_dst, 0, conj_src1);
#else
    AURA_UNUSED(src0);
    AURA_UNUSED(src1);
    AURA_UNUSED(dst);
#endif

    return Status::OK;
}

struct MixedDiff
{
    DT_F64 operator()(const DT_F64 val0, const DT_F64 val1) const
    {
        RelativeDiff relative_diff;
        AbsDiff abs_diff;

        return Min(abs_diff(val0, val1), relative_diff(val0, val1));
    }

    static std::string ToString()
    {
        return "MixedDiff";
    }
};

class MatrixMulSpectrumsTest : public TestBase<MulSpectrumsParam::TupleTable, MulSpectrumsParam::Tuple>
{
public:
    MatrixMulSpectrumsTest(Context *ctx, MulSpectrumsParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // Get next param set
        MulSpectrumsParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());

        // Create src mat
        Mat src0 = m_factory.GetRandomMat(-1000, 1000, ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat src1 = m_factory.GetRandomMat(-1000, 1000, ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);

        // Create dst mat
        Mat dst = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);

        DT_BOOL conj_src1 = run_param.conj_src1;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        result.param  = "";
        result.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(ElemType::F32);
        result.output = run_param.mat_size.ToString() + " " + ElemTypesToString(ElemType::F32);

        // run interface
        DT_S32 loop_count = stress_count ? stress_count : 10;
        Status status_exec = Executor(loop_count, 2, time_val, IMulSpectrums, m_ctx, src0, src1, dst, conj_src1, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
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
            result.accu_benchmark = "OpenCV::MulSpectrums";
            status_exec = Executor(10, 2, time_val, OpenCVMulSpectrums, src0, src1, ref, conj_src1);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCV OpenCVMulSpectrums execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            result.accu_benchmark = "MulSpectrums(target::none)";
            status_exec = IMulSpectrums(m_ctx, src0, src1, ref, conj_src1, OpTarget::None());

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        status_exec = MatCompare<MixedDiff>(m_ctx, dst, ref, cmp_result, 0.1);

        if (status_exec == Status::OK)
        {
            result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
            result.accu_result = cmp_result.ToString();
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail\n");
        }

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        // release mat
        m_factory.PutAllMats();

        return 0;
    }
private:
    Context   *m_ctx;
    MatFactory m_factory;
};


#endif // AURA_OPS_MATRIX_MUL_SPECTRUMS_UNIT_TEST_HPP__
