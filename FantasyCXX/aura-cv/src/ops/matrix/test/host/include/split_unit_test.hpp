#ifndef AURA_OPS_MATRIX_SPLIT_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_SPLIT_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(SplitParam,
                ElemType,             elem_type,
                std::vector<MatSize>, mat_sizes,
                OpTarget,             target);

// Multi-mat split
AURA_INLINE Status CvSplit(Mat &src, std::vector<Mat> &dst)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::Mat src_mat = MatToOpencv(src);

    std::vector<cv::Mat> dst_mats;
    for (size_t n = 0; n < dst.size(); ++n)
    {
        dst_mats.emplace_back(MatToOpencv(dst[n]));
    }
    cv::split(src_mat, dst_mats);
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
#endif

    return Status::OK;
}

class SplitMultiMatTest : public TestBase<SplitParam::TupleTable, SplitParam::Tuple>
{
public:
    SplitMultiMatTest(Context *ctx, SplitParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // get next param set
        SplitParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());

        TestTime time_val;
        TestResult result;

        const auto& mat_sizes = run_param.mat_sizes;

        Sizes3 src_sz  = mat_sizes[0].m_sizes;

        MI_S32 total_ch_count = 0;
        for (size_t i = 0; i < run_param.mat_sizes.size(); ++i)
        {
            total_ch_count += run_param.mat_sizes[i].m_sizes.m_channel;
        }
        src_sz.m_channel = total_ch_count;

        Mat src = m_factory.GetRandomMat(0, 65535, run_param.elem_type, src_sz);
        std::vector<Mat> dsts;
        std::vector<Mat> refs;

        for (size_t i = 0; i < mat_sizes.size(); ++i)
        {
            dsts.emplace_back(m_factory.GetEmptyMat(run_param.elem_type, mat_sizes[i].m_sizes, AURA_MEM_DEFAULT, mat_sizes[i].m_strides));
            refs.emplace_back(m_factory.GetEmptyMat(run_param.elem_type, mat_sizes[i].m_sizes, AURA_MEM_DEFAULT, mat_sizes[i].m_strides));
        }

        result.param  = "multi-mat";
        result.input = src_sz.ToString() + " " + ElemTypesToString(run_param.elem_type);
        for (size_t i = 0; i < mat_sizes.size(); ++i)
        {
            result.output += run_param.mat_sizes[i].ToString();
        }

        // run interface
        MI_S32 loop_count = stress_count ? stress_count : 10;
        Status status_exec = Executor(loop_count, 2, time_val, ISplit, m_ctx, src, dsts, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result["None"] = time_val;
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
            result.accu_benchmark = "OpenCV::Split";
            status_exec = Executor(10, 2, time_val, CvSplit, src, refs);

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvSplit execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            result.accu_benchmark = "Split(target::none)";
            status_exec = ISplit(m_ctx, src, refs, TargetType::NONE);

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        result.accu_status = TestStatus::PASSED;
        for (size_t i = 0 ; i < dsts.size(); ++i)
        {
            MatCmpResult cmp_result;
            if (MatCompare(m_ctx, dsts[i], refs[i], cmp_result, 0) == Status::OK)
            {
                result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
                result.accu_result = cmp_result.ToString();
            }
            else
            {
                AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail\n");
            }
        }

EXIT:
        test_case->AddTestResult(result.accu_status, result);
        // Release mats
        m_factory.PutMats(src);
        for (size_t i = 0; i < dsts.size(); ++i)
        {
            m_factory.PutMats(dsts[i], refs[i]);
        }

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_SPLIT_UNIT_TEST_HPP__