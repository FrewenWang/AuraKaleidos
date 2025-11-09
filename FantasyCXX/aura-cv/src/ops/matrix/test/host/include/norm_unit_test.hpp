#ifndef AURA_OPS_MATRIX_NORM_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_NORM_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(NormParam,
                ElemType,       elem_type,
                MatSize,        mat_size,
                NormType,       norm_type,
                OpTarget,       target);

#if !defined(AURA_BUILD_XPLORER)
AURA_INLINE cv::NormTypes NormTypeConvert(NormType type)
{
    switch (type)
    {
        case NormType::NORM_INF:
        {
            return cv::NormTypes::NORM_INF;
        }
        case NormType::NORM_L1:
        {
            return cv::NormTypes::NORM_L1;
        }
        case NormType::NORM_L2:
        {
            return cv::NormTypes::NORM_L2;
        }
        case NormType::NORM_L2SQR:
        {
            return cv::NormTypes::NORM_L2SQR;
        }
        default:
        {
            return cv::NormTypes::NORM_L2;
        }
    }
}
#endif

AURA_INLINE Status CvNorm(Mat &src, DT_F64 &result, NormType type)
{
    if (ElemType::U32 == src.GetElemType() || ElemType::F16 == src.GetElemType())
    {
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    result = cv::norm(cv_src, NormTypeConvert(type));
#else
    AURA_UNUSED(result);
    AURA_UNUSED(type);
#endif

    return Status::OK;
}

class NormTest : public TestBase<NormParam::TupleTable, NormParam::Tuple>
{
public:
    NormTest(Context *ctx, NormParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // Get next param set
        NormParam run_param(GetParam((index)));
        ElemType elem_type = run_param.elem_type;
        MatSize  mat_size  = run_param.mat_size;

        // Print param info
        AURA_LOGI(m_ctx, AURA_TAG, "\n\n######################### Norm param: %s\n", run_param.ToString().c_str());

        // Create src mats
        Mat src_mat = m_factory.GetRandomMat(-655350, 655350, elem_type, mat_size.m_sizes);

        TestTime time_val;
        TestResult result;
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.param  = NormTypeToString(run_param.norm_type);
        result.output = "DT_F64";

        // Execute result variables
        Status ret = Status::OK;
        DT_F64 res_norm = 0.0;
        DT_F64 ref_norm = 0.0;

        // Run interface
        DT_S32 loop_count = stress_count ? stress_count : 10;

        Status status_exec = Executor(loop_count, 2, time_val, INorm, m_ctx, src_mat, &res_norm, run_param.norm_type, run_param.target);
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status                                              = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // Run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            result.accu_benchmark = std::string("OpenCV::Norm");
            ret = Executor(10, 2, time_val, CvNorm, src_mat, ref_norm, run_param.norm_type);

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCV::Norm execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            result.accu_benchmark = std::string("Norm");
            ret = INorm(m_ctx, src_mat, &ref_norm, run_param.norm_type, OpTarget::None());

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // Compare
        {
            DT_F64 relative_diff = Abs(res_norm - ref_norm) / Max(Abs(res_norm), Abs(ref_norm));
            result.accu_result = std::string("dst norm: ") + std::to_string(res_norm) + std::string(" ref norm: ") + std::to_string(ref_norm);
            result.accu_status = (relative_diff < 1E-4) ? TestStatus::PASSED : TestStatus::FAILED;
            if (TestStatus::FAILED == result.accu_status)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "compare failed: %s \n", result.accu_result.c_str());
                goto EXIT;
            }
        }

EXIT:
        test_case->AddTestResult(result.accu_status, result);
        m_factory.PutAllMats();
        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_NORM_UNIT_TEST_HPP__