#ifndef AURA_OPS_MATRIX_ARITHMETIC_SCALAR_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_ARITHMETIC_SCALAR_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;
using divide_func = Status (*)(Context *, MI_F32, const Mat &, Mat &, const OpTarget &);

AURA_TEST_PARAM(ArithmScalarTestParam,
                ElemType,  elem_type_src,
                ElemType,  elem_type_dst,
                MatSize,   mat_size,
                MI_F32,    scalar,
                OpTarget,  target);

static Status CvScalarDivideByMat(MI_F32 scalar, Mat &src, Mat &dst)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src  = MatToOpencv(src);
    cv::Mat cv_dst  = MatToOpencv(dst);
    MI_S32 cv_depth = GetCVDepth(dst.GetElemType());

    cv::divide(scalar, cv_src, cv_dst, cv_depth);
#else
    AURA_UNUSED(scalar);
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
#endif

    return Status::OK;
}

class ArithmScalarTest : public TestBase<ArithmScalarTestParam::TupleTable, ArithmScalarTestParam::Tuple>
{
public:
    ArithmScalarTest(Context *ctx, ArithmScalarTestParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        AURA_UNUSED(test_case);

        /// get next param set
        ArithmScalarTestParam run_param(GetParam((index)));
        ElemType elem_type_src = run_param.elem_type_src;
        ElemType elem_type_dst = run_param.elem_type_dst;
        MatSize mat_size       = run_param.mat_size;
        OpTarget target        = run_param.target;

        // get src mat sizes
        Sizes3 sizes   = mat_size.m_sizes;
        Sizes  strides = mat_size.m_strides;

        /// Print param info
        AURA_LOGI(m_ctx, AURA_TAG, "\n\n######################### ArithmScalar Test Param: %s\n", run_param.ToString().c_str());

        /// Create src mats
        MI_S32 mem_type = AURA_MEM_DEFAULT;
        Mat src = m_factory.GetRandomMat(1, 1024, elem_type_src, sizes, mem_type, strides);
        Mat dst = m_factory.GetEmptyMat(elem_type_dst, sizes, mem_type, strides);
        Mat ref = m_factory.GetEmptyMat(elem_type_dst, sizes, mem_type, strides);

        MI_S32 loop_count = stress_count ? stress_count : (TargetType::NONE == run_param.target.m_type ? 5 : 10);

        // run function
        TestResult result;
        MatCmpResult cmp_result;
        TestTime time_val;
        result.param  = "scalar(" + std::to_string(run_param.scalar) + ")";
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type_src);
        result.output = mat_size.ToString() + " " + ElemTypesToString(elem_type_dst);

        // run interface
        Status status_exec = Executor<divide_func>(loop_count, 2, time_val, IDivide,
                                                   m_ctx, run_param.scalar, src, dst, run_param.target);
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Arithmetical Operator failed with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.accu_status = TestStatus::FAILED;
            result.perf_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            result.accu_benchmark = "OpenCV::divide";
            status_exec = Executor(10, 2, time_val, CvScalarDivideByMat, run_param.scalar, src, ref);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvConvertTo execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            result.accu_benchmark = "ScalarDivide(target::none)";
            status_exec           = IDivide(m_ctx, run_param.scalar, src, ref);

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        MatCompare(m_ctx, dst, ref, cmp_result, 1);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        m_factory.PutAllMats();
        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_ARITHMETIC_SCALAR_UNIT_TEST_HPP__