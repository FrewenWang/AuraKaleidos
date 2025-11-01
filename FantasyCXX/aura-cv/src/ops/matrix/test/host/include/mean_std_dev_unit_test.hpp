#ifndef AURA_OPS_MATRIX_MEAN_STD_DEV_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_MEAN_STD_DEV_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(MeanStdDevParam,
                ElemType,  elem_type,
                MatSize,   mat_size,
                OpTarget,  target);

AURA_INLINE Status CvMeanStdDev(Mat &mat, Scalar &means, Scalar &std_devs)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(mat);
    cv::Scalar cv_mean = cv::Scalar::all(0.0);
    cv::Scalar cv_std_dev = cv::Scalar::all(0.0);

    cv::meanStdDev(cv_src, cv_mean, cv_std_dev);

    for (MI_S32 i = 0; i < 4; ++i)
    {
        means.m_val[i] = cv_mean[i];
        std_devs.m_val[i] = cv_std_dev[i];
    }
#else
    AURA_UNUSED(mat);
    AURA_UNUSED(means);
    AURA_UNUSED(std_devs);
#endif

    return Status::OK;
}

class MeanStdDevTest : public TestBase<MeanStdDevParam::TupleTable, MeanStdDevParam::Tuple>
{
public:
    MeanStdDevTest(Context *ctx, MeanStdDevParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // Get next param set
        MeanStdDevParam run_param(GetParam((index)));
        ElemType elem_type = run_param.elem_type;
        MatSize  mat_size  = run_param.mat_size;

        // Print param info
        AURA_LOGI(m_ctx, AURA_TAG, "\n\n######################### MeanStdDev param: %s\n", run_param.ToString().c_str());

        // Create src mats
        MI_S32 mem_type = AURA_MEM_DEFAULT;
        Mat src_mat = m_factory.GetRandomMat(-655350, 655350, elem_type, mat_size.m_sizes, mem_type, mat_size.m_strides);

        TestTime time_val;
        TestResult result;
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = "MI_F64, MI_F64";

        // Execute result variables
        Scalar res_means    = Scalar::All(0.0);
        Scalar res_std_devs = Scalar::All(0.0);
        Scalar ref_means    = Scalar::All(0.0);
        Scalar ref_std_devs = Scalar::All(0.0);

        // used for final compare
        std::vector<Scalar> res_scalars;
        std::vector<Scalar> ref_scalars;
        // Compare result
        ScalarCmpResult cmp_result;

        // Run interface
        MI_S32 loop_count  = stress_count ? stress_count : 10;
        Status status_exec = Executor(loop_count, 2, time_val, IMeanStdDev, m_ctx, src_mat, res_means, res_std_devs, run_param.target);

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
            result.accu_benchmark = std::string("OpenCV::MeanStdDev");
            status_exec = Executor(10, 2, time_val, CvMeanStdDev, src_mat, ref_means, ref_std_devs);

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCV::MeanStdDev execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IMeanStdDev(m_ctx, src_mat, ref_means, ref_std_devs, TargetType::NONE);

            result.accu_benchmark = "MeanStdDev(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // Compare
        res_scalars.push_back(res_means);
        res_scalars.push_back(res_std_devs);
        ref_scalars.push_back(ref_means);
        ref_scalars.push_back(ref_std_devs);
        if (ScalarCompare<RelativeDiff>(m_ctx, ref_scalars, res_scalars, cmp_result, 1e-5) == Status::OK)
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
        m_factory.PutAllMats();
        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_MEAN_STD_DEV_UNIT_TEST_HPP__