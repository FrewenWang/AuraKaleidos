#ifndef AURA_OPS_MATRIX_GEMM_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_GEMM_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct GemmTestParam
{
    GemmTestParam()
    {}

    GemmTestParam(MatSize size_src1, MatSize size_src2, MatSize size_dst) : size_src1(size_src1), size_src2(size_src2), size_dst(size_dst)
    {}

    friend std::ostream& operator<<(std::ostream &os, const GemmTestParam &gemm_test_param)
    {
        os << "size_src1:" << gemm_test_param.size_src1 << " | size_src2:" << gemm_test_param.size_src2 << " | size_dst:" << gemm_test_param.size_dst;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    MatSize size_src1;
    MatSize size_src2;
    MatSize size_dst;
};

AURA_TEST_PARAM(GemmParam,
                ElemType,      src_type,
                GemmTestParam, size_param,
                OpTarget,      target);

AURA_INLINE Status CvGemm(Mat &src1, Mat &src2, double alpha, Mat &src3, double beta, Mat &dst, MI_S32 flags = 0)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src1 = MatToOpencv(src1);
    cv::Mat cv_src2 = MatToOpencv(src2);
    cv::Mat cv_src3 = src3.IsValid() ? MatToOpencv(src3) : cv::Mat();
    cv::Mat cv_dst  = MatToOpencv(dst);

    cv::gemm(cv_src1, cv_src2, alpha, cv_src3, beta, cv_dst, flags);
#else
    AURA_UNUSED(src1);
    AURA_UNUSED(src2);
    AURA_UNUSED(alpha);
    AURA_UNUSED(src3);
    AURA_UNUSED(beta);
    AURA_UNUSED(dst);
    AURA_UNUSED(flags);
#endif

    return Status::OK;
}

class MatrixGemmTest : public TestBase<GemmParam::TupleTable, GemmParam::Tuple>
{
public:
    MatrixGemmTest(Context *ctx, GemmParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // Get next param set
        GemmParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());

        MatSize size_src1 = run_param.size_param.size_src1;
        MatSize size_src2 = run_param.size_param.size_src2;
        MatSize size_dst = run_param.size_param.size_dst;

        // Create src mat
        Mat src0 = m_factory.GetRandomMat(0, 65535, run_param.src_type, size_src1.m_sizes, AURA_MEM_DEFAULT, size_src1.m_strides);
        Mat src1 = m_factory.GetRandomMat(0, 65535, run_param.src_type, size_src2.m_sizes, AURA_MEM_DEFAULT, size_src2.m_strides);

        // Create dst mat
        Mat dst = m_factory.GetEmptyMat(run_param.src_type, size_dst.m_sizes, AURA_MEM_DEFAULT, size_dst.m_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.src_type, size_dst.m_sizes, AURA_MEM_DEFAULT, size_dst.m_strides);

        MI_S32 loop_count = (stress_count > 0) ? stress_count : (TargetType::NONE == run_param.target.m_type ? 5 : 10);
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.input  = size_src1.ToString() + " " + ElemTypesToString(run_param.src_type) + "x" + size_src2.ToString() + " " + ElemTypesToString(run_param.src_type);
        result.output = size_dst.ToString() + " " + ElemTypesToString(run_param.src_type);

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, IGemm, m_ctx, src0, src1, dst, run_param.target);

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
            result.accu_benchmark = "OpenCV::Gemm";
            status_exec = Executor(10, 2, time_val, CvGemm, src0, src1, 1, Mat(), 0, ref, 0);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCV Gemm execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            result.accu_benchmark = "Gemm(target::none)";
            status_exec = IGemm(m_ctx, src0, src1, ref, OpTarget::None());
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (MatCompare<RelativeDiff>(m_ctx, dst, ref, cmp_result, 1e-5, 0, 1e-10) == Status::OK)
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
        m_factory.PutMats(src0, src1, dst, ref);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_GEMM_UNIT_TEST_HPP__