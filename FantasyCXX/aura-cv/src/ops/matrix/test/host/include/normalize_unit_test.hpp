#ifndef AURA_OPS_MATRIX_NORMALIZE_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_NORMALIZE_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;
struct NormalizeTestParam
{
    NormalizeTestParam()
    {}
    NormalizeTestParam(DT_F32 a, DT_F32 b, NormType t) : alpha(a), beta(b), type(t)
    {}

    friend std::ostream& operator << (std::ostream &os, const NormalizeTestParam &param)
    {
        os << "type: " << NormTypeToString(param.type) << " alpha: " << param.alpha << " beta: " << param.beta;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    DT_F32 alpha;
    DT_F32 beta;
    NormType type;
};

AURA_TEST_PARAM(NormalizeParam,
                ElemType,           elem_type,
                MatSize,            mat_size,
                NormalizeTestParam, extra_param,
                OpTarget,           target);

#if !defined(AURA_BUILD_XPLORER)
AURA_INLINE DT_S32 NormTypeToOpenCV(const NormType &type)
{
    switch (type)
    {
        case NormType::NORM_MINMAX:
        {
            return cv::NormTypes::NORM_MINMAX;
        }
        case NormType::NORM_L1:
        {
            return cv::NormTypes::NORM_L1;
        }
        case NormType::NORM_L2:
        {
            return cv::NormTypes::NORM_L2;
        }
        case NormType::NORM_INF:
        {
            return cv::NormTypes::NORM_INF;
        }
        default:
        {
            return -1;
        }
    }
}
#endif

AURA_INLINE Status CvNormalize(Mat &src, Mat &dst, DT_F32 alpha, DT_F32 beta, NormType type)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    cv::normalize(cv_src, cv_dst, alpha, beta, NormTypeToOpenCV(type));
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(alpha);
    AURA_UNUSED(beta);
    AURA_UNUSED(type);
#endif

    return Status::OK;
}

class MatrixNormalizeTest : public TestBase<NormalizeParam::TupleTable, NormalizeParam::Tuple>
{
public:
    MatrixNormalizeTest(Context *ctx, NormalizeParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // Get next param set
        NormalizeParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());
        // Create mats
        DT_S32 mem_type = AURA_MEM_DEFAULT;
        Mat src = m_factory.GetRandomMat(-655350, 655350, run_param.elem_type, run_param.mat_size.m_sizes, mem_type, run_param.mat_size.m_strides);
        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_size.m_sizes, mem_type, run_param.mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_size.m_sizes, mem_type, run_param.mat_size.m_strides);

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        result.param  = run_param.extra_param.ToString();
        result.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_type);

        // run interface
        DT_S32 loop_count = stress_count ? stress_count : 10;
        Status status_exec = Executor(loop_count, 2, time_val, INormalize, m_ctx, src, dst, run_param.extra_param.alpha,
                                      run_param.extra_param.beta, run_param.extra_param.type, run_param.target);

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
            result.accu_benchmark = "OpenCV::Normalize";
            status_exec = Executor(10, 2, time_val, CvNormalize, src, ref, run_param.extra_param.alpha, run_param.extra_param.beta, run_param.extra_param.type);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvNormalize execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            result.accu_benchmark = "Normalize(target::none)";
            status_exec = INormalize(m_ctx, src, ref, run_param.extra_param.alpha, run_param.extra_param.beta, run_param.extra_param.type, OpTarget::None());

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (MatCompare(m_ctx, dst, ref, cmp_result, 1.0) == Status::OK)
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

#endif // AURA_OPS_MATRIX_NORMALIZE_UNIT_TEST_HPP__
