#ifndef AURA_OPS_MATRIX_ROTATE_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_ROTATE_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(RotateParam,
                ElemType,   elem_type,
                MatSize,    mat_size,
                RotateType, rot_type,
                OpTarget,   target);

#if !defined(AURA_BUILD_XPLORER)
AURA_INLINE MI_S32 RotateTypeToOpenCV(RotateType type)
{
    MI_S32 flag;

    switch (type)
    {
        case RotateType::ROTATE_90:
        {
            flag = cv::ROTATE_90_CLOCKWISE;
            break;
        }
        case RotateType::ROTATE_180:
        {
            flag = cv::ROTATE_180;
            break;
        }
        case RotateType::ROTATE_270:
        {
            flag = cv::ROTATE_90_COUNTERCLOCKWISE;
            break;
        }
        default:
        {
            flag = -1;
            break;
        }
    }

    return flag;
}
#endif

AURA_INLINE Status CvRotate(Mat &src, Mat &dst, RotateType type)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    if (cv_src.channels() > 4 || cv_dst.channels() > 4)
    {
        return Status::ERROR;
    }
    cv::rotate(cv_src, cv_dst, RotateTypeToOpenCV(type));
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(type);
#endif

    return Status::OK;
}

class MatrixRotate : public TestBase<RotateParam::TupleTable, RotateParam::Tuple>
{
public:
    MatrixRotate(Context *ctx, RotateParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // Get next param set
        RotateParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());

        // Create src mat
        Mat src = m_factory.GetRandomMat(0, 65535, run_param.elem_type, run_param.mat_size.m_sizes,
                                               AURA_MEM_DEFAULT, run_param.mat_size.m_strides);

        // Create dst mat
        Sizes3 src_sz = run_param.mat_size.m_sizes;
        Sizes3 dst_sz;

        if (run_param.rot_type == RotateType::ROTATE_180)
        {
            dst_sz = {src_sz.m_height, src_sz.m_width, src_sz.m_channel};
        }
        else
        {
            dst_sz = {src_sz.m_width, src_sz.m_height, src_sz.m_channel};
        }

        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, dst_sz);
        Mat ref = m_factory.GetEmptyMat(run_param.elem_type, dst_sz);

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = RotateTypeToString(run_param.rot_type);
        result.input  = src_sz.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = dst_sz.ToString();

        // run interface
        MI_S32 loop_count = stress_count ? stress_count : 10;
        Status status_exec = Executor(loop_count, 2, time_val, IRotate, m_ctx, src, dst, run_param.rot_type, run_param.target);

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
            result.accu_benchmark = "OpenCV::Rotate";
            status_exec = Executor(10, 2, time_val, CvRotate, src, ref, run_param.rot_type);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvRotate execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IRotate(m_ctx, src, ref, run_param.rot_type, OpTarget::None());

            result.accu_benchmark = "Rotate(target::none)";

            if (status_exec != Status::OK)
            {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
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

#endif // AURA_OPS_MATRIX_ROTATE_UNIT_TEST_HPP__