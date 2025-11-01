#ifndef AURA_OPS_MATRIX_FLIP_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_FLIP_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(FlipParam,
                ElemType,  elem_type,
                MatSize,   mat_size,
                FlipType,  flip_type,
                OpTarget,  target);

AURA_INLINE MI_S32 MatrixFlipTypeToOpenCV(FlipType type)
{
    MI_S32 ret = 0;

    switch (type)
    {
        case FlipType::VERTICAL:
        {
            ret = 0;
            break;
        }
        case FlipType::HORIZONTAL:
        {
            ret = 1;
            break;
        }
        case FlipType::BOTH:
        {
            ret = -1;
            break;
        }
        default:
        {
            ret = 0;
            break;
        }
    }

    return ret;
}

AURA_INLINE Status CvFlip(Mat &src, Mat &dst, FlipType type)
{
#if !defined(AURA_BUILD_XPLORER)
    MI_S32 cv_type = MatrixFlipTypeToOpenCV(type);

    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    cv::flip(cv_src, cv_dst, cv_type);
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(type);
#endif

    return Status::OK;
}

class MatrixFlipTest : public TestBase<FlipParam::TupleTable, FlipParam::Tuple>
{
public:
    MatrixFlipTest(Context *ctx, FlipParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // get next param set
        FlipParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());
        // Create mats
        Mat src = m_factory.GetRandomMat(0, 65535, run_param.elem_type, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.elem_type, run_param.mat_size.m_sizes, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);

        MI_S32 loop_count = stress_count ? stress_count : (TargetType::NONE == run_param.target.m_type ? 5 : 10);
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = "type(" + FlipTypeToString(run_param.flip_type) + ")";
        result.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = run_param.mat_size.ToString();

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, IFlip, m_ctx, src, dst, run_param.flip_type, run_param.target);

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
            result.accu_benchmark = "OpenCV::Flip";
            status_exec = Executor(10, 2, time_val, CvFlip, src, ref, run_param.flip_type);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvFlip execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IFlip(m_ctx, src, ref, run_param.flip_type, OpTarget::None());

            result.accu_benchmark = "Flip(target::none)";

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

#endif // AURA_OPS_MATRIX_FLIP_UNIT_TEST_HPP__