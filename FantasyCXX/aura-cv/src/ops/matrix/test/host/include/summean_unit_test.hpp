#ifndef AURA_OPS_MATRIX_SUMMEAN_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_SUMMEAN_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

using AuraSumMeanFunc = std::function<Status(Context *, const Mat &, Scalar &, const OpTarget &)>;

enum class SumMeanOpType
{
    SUM = 0,
    MEAN,
};

static AuraSumMeanFunc GetAuraSumMeanFunc(SumMeanOpType op)
{
    switch (op)
    {
        case SumMeanOpType::SUM: 
        {
            return ISum;
        }
        case SumMeanOpType::MEAN: 
        {
            return IMean;
        }
        default: 
        {
            return MI_NULL;
        }
    }
}

static std::ostream& operator<<(std::ostream &os,const SumMeanOpType &op)
{
    switch (op)
    {
        case SumMeanOpType::SUM: 
        {
            os << "SUM";
            break;
        }
        case SumMeanOpType::MEAN: 
        {
            os << "MEAN";
            break;
        }
        default: 
        {
            os << "UNDEFINED OP";
            break;
        }
    }

    return os;
}

static std::string UnitTestBinaryOpsToString(const SumMeanOpType &op)
{
    std::ostringstream oss;
    oss << op;
    return oss.str();
}

AURA_TEST_PARAM(SumMeanTestParam,
                ElemType,       elem_type,
                MatSize,        mat_size,
                SumMeanOpType,  op,
                OpTarget,       target);

static Status CvSumMean(Mat &src, Scalar &scalar, SumMeanOpType op)
{
    if (ElemType::U32 == src.GetElemType() || ElemType::F16 == src.GetElemType())
    {
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src = MatToOpencv(src);
    cv::Scalar cv_scalar;

    switch (op)
    {
        case SumMeanOpType::SUM:
        {
            cv_scalar = cv::sum(cv_src);
            break;
        }
        case SumMeanOpType::MEAN:
        {
            cv_scalar = cv::mean(cv_src);
            break;
        }
        default:
        {
            return Status::ERROR;
        }
    }

    scalar.m_val[0] = cv_scalar[0];
    scalar.m_val[1] = cv_scalar[1];
    scalar.m_val[2] = cv_scalar[2];
    scalar.m_val[3] = cv_scalar[3];
#else
    AURA_UNUSED(scalar);
    AURA_UNUSED(op);
#endif

    return Status::OK;
}

class SumMeanTest : public TestBase<SumMeanTestParam::TupleTable, SumMeanTestParam::Tuple>
{
public:
    SumMeanTest(Context *ctx, SumMeanTestParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        SumMeanTestParam run_param(GetParam((index)));
        ElemType elem_type = run_param.elem_type;
        MatSize  mat_size  = run_param.mat_size;
        SumMeanOpType op   = run_param.op;
        OpTarget target    = run_param.target;

        // get src mat sizes
        Sizes3 sizes   = mat_size.m_sizes;
        Sizes  strides = mat_size.m_strides;

        /// Print param info
        AURA_LOGI(m_ctx, AURA_TAG, "\n\n######################### SumMean Test Param: %s\n", run_param.ToString().c_str());

        /// Create src mats
        MI_S32 mem_type = AURA_MEM_DEFAULT;
        Mat src = m_factory.GetRandomMat(-1000, 1100, elem_type, sizes, mem_type, strides);

        TestResult result;
        result.param  = UnitTestBinaryOpsToString(op);
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(elem_type);

        // run function
        ScalarCmpResult cmp_result;
        TestTime time_val;
        Scalar dst_scalar;
        Scalar ref_scalar;
        std::stringstream ss;
        Status status_exec;
        MI_S32 loop_count = stress_count ? stress_count : 5;

        AuraSumMeanFunc aura_func = GetAuraSumMeanFunc(op);
        if (MI_NULL == aura_func)
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Aura Binary Function get failed \n");
            result.perf_status = TestStatus::FAILED;
            goto EXIT;
        }

        status_exec = Executor(loop_count, 2, time_val, aura_func, m_ctx, src, dst_scalar, run_param.target);
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
            result.accu_benchmark = "OpenCV::SumMean";
            status_exec           = Executor(5, 2, time_val, CvSumMean, src, ref_scalar, op);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvSumMean execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec           = aura_func(m_ctx, src, ref_scalar, OpTarget::None());
            result.accu_benchmark = "SumMean(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (ScalarCompare<RelativeDiff>(m_ctx, dst_scalar, ref_scalar, cmp_result, 1e-5) == Status::OK)
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

#endif // AURA_OPS_MATRIX_SUMMEAN_UNIT_TEST_HPP__