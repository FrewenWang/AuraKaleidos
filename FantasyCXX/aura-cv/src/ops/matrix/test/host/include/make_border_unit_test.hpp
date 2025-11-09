#ifndef AURA_OPS_MATRIX_MAKE_BORDER_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_MAKE_BORDER_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(MakeBorderRunParam,
                ElemType,   elem_type,
                MatSize,    mat_size,
                BorderType, border_type,
                BorderSize, border_size,
                OpTarget,   target);

static Status CvMakeBorder(Mat &src, Mat &dst, BorderType type,
                                 const BorderSize &bsize, const Scalar &scalar)
{
#if !defined(AURA_BUILD_XPLORER)
    DT_S32 cv_type = BorderTypeToOpencv(type);
    cv::Scalar cv_scalar = {scalar.m_val[0], scalar.m_val[1], scalar.m_val[2], scalar.m_val[3]};

    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);

    cv::copyMakeBorder(cv_src, cv_dst, bsize.top, bsize.bottom, bsize.left, bsize.right, cv_type, cv_scalar);
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(type);
    AURA_UNUSED(bsize);
    AURA_UNUSED(scalar);
#endif

    return Status::OK;
}

class MakeBorderTest : public TestBase<MakeBorderRunParam::TupleTable, MakeBorderRunParam::Tuple>
{
public:
    MakeBorderTest(Context *ctx, MakeBorderRunParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        /// get next param set
        MakeBorderRunParam run_param(GetParam((index)));
        ElemType elem_type     = run_param.elem_type;
        MatSize src_mat_size   = run_param.mat_size;
        BorderType border_type = run_param.border_type;
        BorderSize border_size = run_param.border_size;

        // get src mat sizes
        Sizes3 src_sizes   = src_mat_size.m_sizes;
        Sizes  src_strides = src_mat_size.m_strides;
        DT_S32 top  = border_size.top;
        DT_S32 bot  = border_size.bottom;
        DT_S32 left = border_size.left;
        DT_S32 righ = border_size.right;

        /// Get dst mat sizes
        Sizes3 dst_sizes   = src_sizes   + Sizes3(top + bot, left + righ, 0);
        Sizes  dst_strides = src_strides + Sizes(top + bot, (left + righ) * 8);

        // set border value
        Scalar border_value = (src_sizes.m_channel > 4) ? Scalar(1, 1, 1, 1) : Scalar(1, 2, 3, 4);

        /// Print param info
        AURA_LOGI(m_ctx, AURA_TAG, "\n\n######################### MakeBorder param: %s\n", run_param.ToString().c_str());

        /// Create src mats
        DT_S32 mem_type = AURA_MEM_DEFAULT;
        Mat src_mat = m_factory.GetRandomMat(0, 1000, elem_type, src_sizes, mem_type, src_strides);
        Mat dst_mat = m_factory.GetEmptyMat(elem_type, dst_sizes, mem_type, dst_strides);
        Mat ref_mat = m_factory.GetEmptyMat(elem_type, dst_sizes, mem_type, dst_strides);

        TestResult result;
        result.param  = "type(" + BorderTypeToString(border_type) + ")";
        result.input  = src_mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = MatSize(dst_sizes).ToString();

        MatCmpResult cmp_result;
        TestTime time_val;
        DT_S32 loop_count  = stress_count ? stress_count : 5;
        Status status_exec = Executor(loop_count, 2, time_val, IMakeBorder, m_ctx, src_mat, dst_mat, run_param.border_size.top, run_param.border_size.bottom,
                              run_param.border_size.left, run_param.border_size.right, run_param.border_type, border_value, run_param.target);
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

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            result.accu_benchmark = "OpenCV::MakeBorder";
            status_exec           = Executor(5, 2, time_val, CvMakeBorder, src_mat, ref_mat, border_type, border_size, border_value);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvMakeBorder execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IMakeBorder(m_ctx, src_mat, ref_mat, run_param.border_size.top, run_param.border_size.bottom,
                                      run_param.border_size.left, run_param.border_size.right, run_param.border_type, 
                                      border_value, OpTarget::None());

            result.accu_benchmark = "MakeBorder(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (MatCompare(m_ctx, dst_mat, ref_mat, cmp_result, 1) == Status::OK)
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

#endif // AURA_OPS_MATRIX_MAKE_BORDER_UNIT_TEST_HPP__