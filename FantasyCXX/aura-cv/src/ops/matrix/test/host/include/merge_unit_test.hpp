#ifndef AURA_OPS_MATRIX_MERGE_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_MERGE_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(MergeParam,
                ElemType,             elem_type,
                std::vector<MatSize>, mat_sizes,
                OpTarget,             target);

// Multi-mat merge
AURA_INLINE Status CvMerge(std::vector<Mat> &src, Mat &dst)
{
#if !defined(AURA_BUILD_XPLORER)
    std::vector<cv::Mat> src_mats;
    for (size_t n = 0; n < src.size(); ++n)
    {
        src_mats.emplace_back(MatToOpencv(src[n]));
    }
    cv::Mat cv_dst = MatToOpencv(dst);

    cv::merge(src_mats, cv_dst);
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
#endif

    return Status::OK;
}

class MergeMultiMatTest : public TestBase<MergeParam::TupleTable, MergeParam::Tuple>
{
public:
    MergeMultiMatTest(Context *ctx, MergeParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // get next param set
        MergeParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = "multi_mat";
        result.input  = run_param.mat_sizes[0].ToString() + " " + ElemTypesToString(run_param.elem_type);

        const auto& mat_sizes = run_param.mat_sizes;

        Sizes3 src_sz = mat_sizes[0].m_sizes;
        Sizes3 dst_sz;

        dst_sz.m_width  = src_sz.m_width;
        dst_sz.m_height = src_sz.m_height;

        DT_S32 total_ch_count = 0;
        for (size_t n = 0; n < mat_sizes.size(); ++n)
        {
            total_ch_count += mat_sizes[n].m_sizes.m_channel;
        }
        dst_sz.m_channel = total_ch_count;

        result.output = dst_sz.ToString();

        std::vector<Mat> srcs;

        for (size_t i = 0; i < mat_sizes.size(); ++i)
        {
            srcs.emplace_back(m_factory.GetRandomMat(0, 65535, run_param.elem_type, mat_sizes[i].m_sizes, AURA_MEM_DEFAULT, mat_sizes[i].m_strides));
        }
        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, dst_sz);
        Mat ref = m_factory.GetEmptyMat(run_param.elem_type, dst_sz);

        // run interface
        DT_S32 loop_count = stress_count ? stress_count : 10;
        Status status_exec = Executor(loop_count, 2, time_val, IMerge, m_ctx, srcs, dst, run_param.target);

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
        result.accu_benchmark = "OpenCV::Merge";
        status_exec = Executor(10, 2, time_val, CvMerge, srcs, ref);
        if (status_exec != Status::OK)
        {
            AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvMerge execute fail\n");
            result.accu_status = TestStatus::UNTESTED;
            goto EXIT;
        }
        result.perf_result["OpenCV"] = time_val;

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
        for (size_t i = 0; i < srcs.size(); ++i)
        {
            m_factory.PutMats(srcs[i]);
        }

        m_factory.PutMats(dst, ref);
        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_MERGE_UNIT_TEST_HPP__