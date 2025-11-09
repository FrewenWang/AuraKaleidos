#ifndef AURA_OPS_MATRIX_MINMAX_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_MINMAX_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(MinMaxTestParam,
                ElemType,  elem_type,
                MatSize,   mat_size,
                BinaryOpType, op,
                OpTarget,  target);

using AuraMinMaxFunc = std::function<Status(Context *, const Mat &, const Mat &, Mat &, const OpTarget &)>;
static AuraMinMaxFunc GetAuraMinMaxFunc(BinaryOpType op)
{
    switch (op)
    {
        case BinaryOpType::MIN:
        {
            return IMin;
        }
        case BinaryOpType::MAX:
        {
            return IMax;
        }
        default:
        {
            return DT_NULL;
        }
    }
}

static Status CvBinary(Mat &src0, Mat &src1, Mat &dst, BinaryOpType op)
{
    if (src0.GetElemType() != src1.GetElemType() || src0.GetElemType() != dst.GetElemType() ||
        ElemType::U32 == src0.GetElemType() || ElemType::F16 == src0.GetElemType())
    {
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src0 = MatToOpencv(src0);
    cv::Mat cv_src1 = MatToOpencv(src1);
    cv::Mat cv_dst  = MatToOpencv(dst);

    switch (op)
    {
        case BinaryOpType::MIN:
        {
            cv::min(cv_src0, cv_src1, cv_dst);
            break;
        }
        case BinaryOpType::MAX:
        {
            cv::max(cv_src0, cv_src1, cv_dst);
            break;
        }
        default:
        {
            return Status::ERROR;
        }
    }
#else
    AURA_UNUSED(op);
#endif

    return Status::OK;
}

class MinMaxTest : public TestBase<MinMaxTestParam::TupleTable, MinMaxTestParam::Tuple>
{
public:
    MinMaxTest(Context *ctx, MinMaxTestParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        /// get next param set
        MinMaxTestParam run_param(GetParam((index)));
        ElemType elem_type = run_param.elem_type;
        MatSize mat_size   = run_param.mat_size;
        BinaryOpType op    = run_param.op;
        OpTarget target    = run_param.target;

        // get src mat sizes
        Sizes3 sizes   = mat_size.m_sizes;
        Sizes  strides = mat_size.m_strides;

        /// Print param info
        AURA_LOGI(m_ctx, AURA_TAG, "\n\n######################### MinMax Test Param: %s\n", run_param.ToString().c_str());

        /// Create src mats
        DT_S32 mem_type = AURA_MEM_DEFAULT;
        Mat src0 = m_factory.GetRandomMat((DT_F32)INT32_MIN, (DT_F32)INT32_MAX, elem_type, sizes, mem_type, strides);
        Mat src1 = m_factory.GetRandomMat((DT_F32)INT32_MIN, (DT_F32)INT32_MAX, elem_type, sizes, mem_type, strides);
        Mat dst  = m_factory.GetEmptyMat(elem_type, sizes, mem_type, strides);
        Mat ref  = m_factory.GetEmptyMat(elem_type, sizes, mem_type, strides);

        TestResult result;
        result.param  = BinaryOpTypeToString(op);
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(elem_type);

        // run function
        MatCmpResult cmp_result;
        TestTime time_val;
        Status status_exec = Status::ERROR;
        DT_S32 loop_count = stress_count ? stress_count : 5;

        AuraMinMaxFunc aura_func = GetAuraMinMaxFunc(op);
        if (DT_NULL == aura_func)
        {
            AURA_LOGE(m_ctx, AURA_TAG, "function pointer not found\n");
            result.perf_status = TestStatus::FAILED;
            goto EXIT;
        }

        status_exec = Executor(loop_count, 2, time_val, aura_func, m_ctx, src0, src1, dst, run_param.target);
        if (status_exec != Status::OK)
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Binary Operator failed with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            goto EXIT;
        }
        result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
        result.perf_status = TestStatus::PASSED;

        // run benchmark
        if (TargetType::NONE == target.m_type)
        {
            status_exec = Executor(5, 2, time_val, CvBinary, src0, src1, ref, op);
            if (status_exec != Status::OK)
            {
                AURA_LOGI(m_ctx, AURA_TAG, "OpenCV not support\n");
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
            result.accu_benchmark = "OpenCV";
        }
        else
        {
            if (aura_func(m_ctx, src0, src1, ref, OpTarget::None()) != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "Binary None failed with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
            result.accu_benchmark = "None";
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
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        m_factory.PutAllMats();
        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_MINMAX_UNIT_TEST_HPP__