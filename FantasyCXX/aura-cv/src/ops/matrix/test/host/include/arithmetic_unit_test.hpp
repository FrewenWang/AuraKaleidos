#ifndef AURA_OPS_MATRIX_ARITHMETIC_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_ARITHMETIC_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct ArithmeticTestParam
{
    ArithmeticTestParam()
    {}

    ArithmeticTestParam(ArithmOpType op) : op(op)
    {}

    friend std::ostream& operator<<(std::ostream &os, const ArithmeticTestParam arithmetic_test_param)
    {
        switch (arithmetic_test_param.op)
        {
            case ArithmOpType::ADD:
            {
                os << "ADD";
                break;
            }
            case ArithmOpType::SUB:
            {
                os << "SUBTRACT";
                break;
            }
            case ArithmOpType::MUL:
            {
                os << "MULTIPLY";
                break;
            }
            case ArithmOpType::DIV:
            {
                os << "DIVIDE";
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

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    ArithmOpType op;
};

AURA_TEST_PARAM(ArithmeticParam,
                ElemType,          elem_type_src,
                ElemType,          elem_type_dst,
                MatSize,           mat_size,
                ArithmOpType,      op,
                OpTarget,          target);

static Status CvArithm(Mat &src0, Mat &src1, Mat &dst, ArithmOpType op)
{
    if (ElemType::F16 == src0.GetElemType() || ElemType::U32 == src0.GetElemType() ||
        ElemType::F16 == src1.GetElemType() || ElemType::U32 == src1.GetElemType() ||
        ElemType::F16 ==  dst.GetElemType() || ElemType::U32 ==  dst.GetElemType())
    {
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat cv_src0 = MatToOpencv(src0);
    cv::Mat cv_src1 = MatToOpencv(src1);
    cv::Mat cv_dst  = MatToOpencv(dst);
    MI_S32 cv_depth = GetCVDepth(dst.GetElemType());

    cv::Mat mask = cv::Mat();
    switch (op)
    {
        case ArithmOpType::ADD:
        {
            cv::add(cv_src0, cv_src1, cv_dst, mask, cv_depth);
            break;
        }
        case ArithmOpType::SUB:
        {
            cv::subtract(cv_src0, cv_src1, cv_dst, mask, cv_depth);
            break;
        }
        case ArithmOpType::MUL:
        {
            cv::multiply(cv_src0, cv_src1, cv_dst, 1.0, cv_depth);
            break;
        }
        case ArithmOpType::DIV:
        {
            cv::divide(cv_src0, cv_src1, cv_dst, 1, cv_depth);
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

using AuraArithmFunc = std::function<Status(Context*, const Mat&, const Mat&, Mat&, const OpTarget&)>;

#define AURA_ARITHM_FUNC_CAST(func_name) \
    static_cast<Status(*)(Context*, const Mat&, const Mat&, Mat&, const OpTarget&)>(&func_name)

static AuraArithmFunc GetAuraArithmOpFunc(ArithmOpType op)
{
    switch (op)
    {
        case ArithmOpType::ADD:
        {
            return AURA_ARITHM_FUNC_CAST(IAdd);
        }
        case ArithmOpType::SUB:
        {
            return AURA_ARITHM_FUNC_CAST(ISubtract);
        }
        case ArithmOpType::MUL:
        {
            return AURA_ARITHM_FUNC_CAST(IMultiply);
        }
        case ArithmOpType::DIV:
        {
            return AURA_ARITHM_FUNC_CAST(IDivide);
        }
        default:
        {
            return MI_NULL;
        }
    }
}

static Status CheckNeonIsSupport(ArithmOpType op, ElemType elem_type_src, ElemType elem_type_dst)
{
    MI_S32 pattern = AURA_MAKE_PATTERN(elem_type_src, elem_type_dst);

    switch (op)
    {
        case ArithmOpType::ADD:
        case ArithmOpType::MUL:
        {
            switch (pattern)
            {
                case AURA_MAKE_PATTERN(ElemType::U8,  ElemType::U8):
                case AURA_MAKE_PATTERN(ElemType::S8,  ElemType::S8):
                case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
                case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
                case AURA_MAKE_PATTERN(ElemType::U32, ElemType::U32):
                case AURA_MAKE_PATTERN(ElemType::S32, ElemType::S32):
                case AURA_MAKE_PATTERN(ElemType::U8,  ElemType::U16):
                case AURA_MAKE_PATTERN(ElemType::S8,  ElemType::S16):
                case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U32):
                case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S32):
                case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
                case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
                {
                    return Status::OK;
                    break;
                }

                default:
                {
                    return Status::ERROR;
                }
            }
            break;
        }

        case ArithmOpType::SUB:
        {
            switch (pattern)
            {
                case AURA_MAKE_PATTERN(ElemType::U8,  ElemType::U8):
                case AURA_MAKE_PATTERN(ElemType::S8,  ElemType::S8):
                case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
                case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
                case AURA_MAKE_PATTERN(ElemType::U32, ElemType::U32):
                case AURA_MAKE_PATTERN(ElemType::S32, ElemType::S32):
                case AURA_MAKE_PATTERN(ElemType::U8,  ElemType::S16):
                case AURA_MAKE_PATTERN(ElemType::S8,  ElemType::S16):
                case AURA_MAKE_PATTERN(ElemType::U16, ElemType::S32):
                case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S32):
                case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
                case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
                {
                    return Status::OK;
                    break;
                }

                default:
                {
                    return Status::ERROR;
                }
            }
            break;
        }

        case ArithmOpType::DIV:
        {
            switch (pattern)
            {
                case AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16):
                case AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32):
                {
                    return Status::OK;
                    break;
                }

                default:
                {
                    return Status::ERROR;
                }
            }
            break;
        }

        default:
            break;
    }

    return Status::ERROR;
}

class ArithmeticTest : public TestBase<ArithmeticParam::TupleTable, ArithmeticParam::Tuple>
{
public:
    ArithmeticTest(Context *ctx, ArithmeticParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    Status CheckParam(MI_S32 index) override
    {
        ArithmeticParam run_param(GetParam((index)));
        if (TargetType::HVX == run_param.target.m_type)
        {
            MI_S32 pattern = AURA_MAKE_PATTERN(run_param.op, run_param.elem_type_src, run_param.elem_type_dst);
            switch (pattern)
            {
                case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::U8,  ElemType::U8):
                case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::U8,  ElemType::U16):
                case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::S8,  ElemType::S8):
                case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::S8,  ElemType::S16):
                case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::U16, ElemType::U16):
                case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::U16, ElemType::U32):
                case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::S16, ElemType::S16):
                case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::S16, ElemType::S32):
                case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::U32, ElemType::U32):
                case AURA_MAKE_PATTERN(ArithmOpType::ADD, ElemType::S32, ElemType::S32):
                case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::U8,  ElemType::U8):
                case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::U8,  ElemType::S16):
                case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::S8,  ElemType::S8):
                case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::S8,  ElemType::S16):
                case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::U16, ElemType::U16):
                case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::U16, ElemType::S32):
                case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::S16, ElemType::S16):
                case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::S16, ElemType::S32):
                case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::U32, ElemType::U32):
                case AURA_MAKE_PATTERN(ArithmOpType::SUB, ElemType::S32, ElemType::S32):
                case AURA_MAKE_PATTERN(ArithmOpType::MUL, ElemType::U8,  ElemType::U16):
                case AURA_MAKE_PATTERN(ArithmOpType::MUL, ElemType::S8,  ElemType::S16):
                case AURA_MAKE_PATTERN(ArithmOpType::MUL, ElemType::U16, ElemType::U32):
                case AURA_MAKE_PATTERN(ArithmOpType::MUL, ElemType::S16, ElemType::S32):
                {
                    return Status::OK;
                }

                default:
                {
                    return Status::ERROR;
                }
            }
        }

        return Status::OK;
    }

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        /// get next param set
        ArithmeticParam run_param(GetParam((index)));
        ElemType elem_type_src               = run_param.elem_type_src;
        ElemType elem_type_dst               = run_param.elem_type_dst;
        MatSize mat_size                     = run_param.mat_size;
        ArithmOpType op                      = run_param.op;
        OpTarget target                      = run_param.target;
        ArithmeticTestParam arithmetic_param = op;

        // check neon is support
        if (TargetType::NEON == target.m_type && CheckNeonIsSupport(op, elem_type_src, elem_type_dst) != Status::OK)
        {
            return 0;
        }

        // get src mat sizes
        Sizes3 sizes   = mat_size.m_sizes;
        Sizes  strides = mat_size.m_strides;

        /// Print param info
        AURA_LOGI(m_ctx, AURA_TAG, "\n\n######################### Arithmetic Test Param: %s\n", run_param.ToString().c_str());

        /// Create src mats
        MI_S32 mem_type = AURA_MEM_DEFAULT;
        Mat src0 = m_factory.GetRandomMat(1, 1024, elem_type_src, sizes, mem_type, strides);
        Mat src1 = m_factory.GetRandomMat(1, 1024, elem_type_src, sizes, mem_type, strides);
        Mat dst  = m_factory.GetEmptyMat(elem_type_dst, sizes, mem_type, strides);
        Mat ref  = m_factory.GetEmptyMat(elem_type_dst, sizes, mem_type, strides);

        MI_S32 loop_count = stress_count ? stress_count : (TargetType::NONE == run_param.target.m_type ? 5 : 10);
        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        Status status_exec;

        result.param  = arithmetic_param.ToString();
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type_src);
        result.output = mat_size.ToString() + " " + ElemTypesToString(elem_type_dst);

        AuraArithmFunc aura_func = GetAuraArithmOpFunc(op);
        if (MI_NULL == aura_func)
        {
            AURA_LOGE(m_ctx, AURA_TAG, "function pointer not found\n");
            result.perf_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run interface
        status_exec = Executor<AuraArithmFunc>(loop_count, 2, time_val, aura_func, m_ctx,
                                               src0, src1, dst, run_param.target);
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == target.m_type)
        {
            status_exec = Executor(5, 2, time_val, CvArithm, src0, src1, ref, op);
            if (status_exec != Status::OK)
            {
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
            result.accu_benchmark = "OpenCV::Arithmetic";
        }
        else
        {
            status_exec           = aura_func(m_ctx, src0, src1, ref, TargetType::NONE);
            result.accu_benchmark = "Arithmetic(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "Arithm None failed with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (ArithmOpType::MUL == run_param.op && ElemType::F16 == run_param.elem_type_dst)
        {
            MatCompare<RelativeDiff>(m_ctx, dst, ref, cmp_result, 1e-3, 0, 1e-6);
        }
        else
        {
            MatCompare(m_ctx, dst, ref, cmp_result, 1);
        }
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

#endif // AURA_OPS_MATRIX_ARITHMETIC_UNIT_TEST_HPP__