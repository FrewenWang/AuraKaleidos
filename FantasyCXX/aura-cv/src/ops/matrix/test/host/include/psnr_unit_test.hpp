#ifndef AURA_OPS_MATRIX_PSNR_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_PSNR_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(PsnrParam,
                ElemType,  elem_type,
                MatSize,   mat_size,
                OpTarget,  target);

AURA_INLINE Status CvPSNR(Mat &src0, Mat &src1, DT_F64 &result)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src0 = MatToOpencv(src0);
    cv::Mat cv_src1 = MatToOpencv(src1);
    result = cv::PSNR(cv_src0, cv_src1, 255);
#else
    AURA_UNUSED(src0);
    AURA_UNUSED(src1);
    AURA_UNUSED(result);
#endif

    return Status::OK;
}

class PsnrTest : public TestBase<PsnrParam::TupleTable, PsnrParam::Tuple>
{
public:
    PsnrTest(Context *ctx, PsnrParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // Get next param set
        PsnrParam run_param(GetParam((index)));
        ElemType elem_type = run_param.elem_type;
        MatSize  mat_size  = run_param.mat_size;

        // Print param info
        AURA_LOGI(m_ctx, AURA_TAG, "\n\n######################### Psnr param: %s\n", run_param.ToString().c_str());

        // Create src mats
        Mat src_mat0 = m_factory.GetRandomMat(-655350, 655350, elem_type, mat_size.m_sizes);
        Mat src_mat1 = m_factory.GetRandomMat(-655350, 655350, elem_type, mat_size.m_sizes);

        TestTime time_perf;
        TestTime cv_time;
        TestResult result;
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = "DT_F64";

        // Execute result variables
        Status ret = Status::OK;
        DT_F64 res_psnr = 0.0;
        DT_F64 ref_psnr = 0.0;
        // Run interface
        DT_S32 loop_count = stress_count ? stress_count : 10;
        ret = Executor(loop_count, 2, time_perf, IPsnr, m_ctx, src_mat0, src_mat1, 255.0, &res_psnr, run_param.target);

        if (Status::OK == ret)
        {
            result.perf_result["None"] = time_perf;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // Run benchmark
        result.accu_benchmark = std::string("OpenCV::PSNR");
        ret = Executor(10, 2, cv_time, CvPSNR, src_mat0, src_mat1, ref_psnr);

        if (ret != Status::OK)
        {
            AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCV::PSNR execute fail\n");
            result.accu_status = TestStatus::UNTESTED;
            goto EXIT;
        }

        result.perf_result["OpenCV"] = cv_time;

        // Compare
        {
            DT_F64 relative_diff = Abs(res_psnr - ref_psnr) / Max(Abs(res_psnr), Abs(ref_psnr));
            result.accu_result = std::string("dst psnr: ") + std::to_string(res_psnr) + std::string(" ref norm: ") + std::to_string(ref_psnr);
            result.accu_status = (relative_diff < 1E-5) ? TestStatus::PASSED : TestStatus::FAILED;
            if (TestStatus::FAILED == result.accu_status)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "compare failed: %s \n", result.accu_result.c_str());
                goto EXIT;
            }
        }

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        m_factory.PutAllMats();
        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_PSNR_UNIT_TEST_HPP__