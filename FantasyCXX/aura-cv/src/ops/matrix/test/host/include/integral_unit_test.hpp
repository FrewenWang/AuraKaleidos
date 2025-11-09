#ifndef AURA_OPS_MATRIX_INTEGRAL_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_INTEGRAL_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct MatElemIntegral
{
    MatElemIntegral()
    {}

    MatElemIntegral(ElemType first, ElemType second, ElemType third,
                    ElemType fourth, ElemType fifth)
                    : first(first), second(second), third(third),
                      fourth(fourth), fifth(fifth)
    {}

    friend std::ostream& operator<<(std::ostream &os, const MatElemIntegral &sz)
    {
        os << "src elem type : " << sz.first << " dst0 elem type : " << sz.second << " dst1 elem type : " << sz.third << std::endl;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << "src elem type : " << first << " dst0 elem type : " << second << " dst1 elem type : " << third;
        return sstream.str();
    }

    ElemType first;  //src elem type
    ElemType second; //dst elem type
    ElemType third;  //dst_sq elem type
    ElemType fourth; //cv_dst elem type
    ElemType fifth;  //cv_dst_sq elem type
};

AURA_TEST_PARAM(IntegralParam,
                MatElemIntegral,  elem_integral,
                MatSize,          mat_size,
                OpTarget,         target);

AURA_INLINE Status CvIntegral(Context *ctx, Mat &src, Mat &dst0, Mat &dst1)
{
    AURA_UNUSED(ctx);
    Status ret = Status::OK;

#if !defined(AURA_BUILD_XPLORER)
    if (src.GetElemType() != ElemType::S8 && dst0.GetElemType() != ElemType::U32 && 
        (dst1.GetElemType() == ElemType::F64) &&
        !(src.GetElemType() == ElemType::U16 && dst0.GetElemType() == ElemType::S32) &&
        !(src.GetElemType() == ElemType::S16 && dst0.GetElemType() == ElemType::S32) &&
        !(src.GetElemType() == ElemType::U16 && dst0.GetElemType() == ElemType::F32) &&
        !(src.GetElemType() == ElemType::S16 && dst0.GetElemType() == ElemType::F32))
    {
        DT_S32 dst0_cv_type = ElemTypeToOpencv(dst0.GetElemType(), dst0.GetSizes().m_channel);
        DT_S32 dst1_cv_type = ElemTypeToOpencv(dst1.GetElemType(), dst1.GetSizes().m_channel);

        cv::Mat cv_src    = MatToOpencv(src);
        cv::Mat cv_dst    = MatToOpencv(dst0);
        cv::Mat cv_dst_sq = MatToOpencv(dst1);

        cv::integral(cv_src, cv_dst, cv_dst_sq, dst0_cv_type, dst1_cv_type);
    }
    else
    {
        //AURA_LOGE(ctx, AURA_TAG, "CV Integral not support\n");
        ret = Status::ERROR;
    }
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst0);
    AURA_UNUSED(dst1);
#endif

    return ret;
}

AURA_INLINE Status CvIntegralNorm(Context *ctx, Mat &src, Mat &dst, DT_S32 type)
{
    AURA_UNUSED(ctx);
    Status ret = Status::OK;

#if !defined(AURA_BUILD_XPLORER)
    if (0 == type)
    {
        if (src.GetElemType() != ElemType::S8 && dst.GetElemType() != ElemType::U32 &&
            !(src.GetElemType() == ElemType::U16 && dst.GetElemType() == ElemType::S32) &&
            !(src.GetElemType() == ElemType::S16 && dst.GetElemType() == ElemType::S32) &&
            !(src.GetElemType() == ElemType::U16 && dst.GetElemType() == ElemType::F32) &&
            !(src.GetElemType() == ElemType::S16 && dst.GetElemType() == ElemType::F32))
        {
            DT_S32 dst0_cv_type = ElemTypeToOpencv(dst.GetElemType(), dst.GetSizes().m_channel);

            cv::Mat cv_src = MatToOpencv(src);
            cv::Mat cv_dst = MatToOpencv(dst);

            cv::integral(cv_src, cv_dst, dst0_cv_type);
        }
        else
        {
            //AURA_LOGE(ctx, AURA_TAG, "CV Integral not support\n");
            ret = Status::ERROR;
        }
    }
    else if (1 == type)
    {
        if ((src.GetElemType() != ElemType::S8) && (dst.GetElemType() == ElemType::F64))
        {
            Mat dst_temp = dst.Clone();

            DT_S32 dst1_cv_type = ElemTypeToOpencv(dst.GetElemType(), dst.GetSizes().m_channel);

            cv::Mat cv_src    = MatToOpencv(src);
            cv::Mat cv_dst    = MatToOpencv(dst_temp);
            cv::Mat cv_dst_sq = MatToOpencv(dst);

            cv::integral(cv_src, cv_dst, cv_dst_sq, dst1_cv_type, dst1_cv_type);
        }
        else
        {
            AURA_LOGE(ctx, AURA_TAG, "CvIntegralNorm input type error.\n");
            ret = Status::ERROR;
        }
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CvIntegralNorm input type error.\n");
        ret = Status::ERROR;
    }
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(type);
#endif

    return ret;
}

static Status MatCompareIntegral(Context *ctx, const Mat &src, const Mat &ref,
                                 MatCmpResult &result, DT_F32 tolerate = 1, DT_F32 step = 1, DT_F64 cmp_eps = 1e-6)
{
    Status ret = Status::OK;

    if (src.GetElemType() != ref.GetElemType())
    {
        AURA_LOGE(ctx, AURA_TAG, "elem type not match\n", DT_TRUE);
        result.status = DT_FALSE;
        return Status::ERROR;
    }
    if ((ElemType::F32 == src.GetElemType()) || (ElemType::F64 == src.GetElemType()))
    {
        ret = MatCompare<RelativeDiff>(ctx, src, ref, result, 1e-5f, step, cmp_eps);
    }
    else
    {
        ret = MatCompare<AbsDiff>(ctx, src, ref, result, tolerate, step, cmp_eps);
    }

    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "MatCompareIntegral failed\n");
    }

    return ret;
}

static Status MatCompareIntegralBorder(Context *ctx, const Mat &dst, const Mat &ref, MatCmpResult &result)
{
    Status ret = Status::OK;

    if (dst.GetElemType() != ref.GetElemType())
    {
        AURA_LOGE(ctx, AURA_TAG, "elem type not match\n", DT_TRUE);
        result.status = DT_FALSE;
        return Status::ERROR;
    }

    DT_F32 tolerate = 1;
    DT_F32 step = 1;
    DT_F64 cmp_eps = 1e-6;

    Sizes3 size = dst.GetSizes() + Sizes3(1, 1, 0);
    Mat pad_mat(ctx, dst.GetElemType(), size);
    if (IMakeBorder(ctx, dst, pad_mat, 1, 0, 1, 0, BorderType::CONSTANT, Scalar(0, 0, 0, 0), OpTarget::None()) != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "IMakeBorder error\n", DT_FALSE);
        result.status = DT_FALSE;
        return Status::ERROR;
    }

    if ((ElemType::F32 == pad_mat.GetElemType()) || (ElemType::F64 == pad_mat.GetElemType()))
    {
        ret = MatCompare<RelativeDiff>(ctx, pad_mat, ref, result, 1e-5f, step, cmp_eps);
    }
    else
    {
        ret = MatCompare<AbsDiff>(ctx, pad_mat, ref, result, tolerate, step, cmp_eps);
    }

    if (ret != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "MatCompareIntegral failed\n");
    }

    return ret;
}

class MatrixIntegralTest : public TestBase<IntegralParam::TupleTable, IntegralParam::Tuple>
{
public:
    MatrixIntegralTest(Context *ctx, IntegralParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // get nex param set
        IntegralParam run_param(GetParam((index)));
        AURA_LOGI(m_ctx, AURA_TAG, "Run param: %s\n", run_param.ToString().c_str());

        Sizes3 src_sz = run_param.mat_size.m_sizes;
        Sizes3 dst_sz = src_sz;
        Sizes3 ref_sz = src_sz;
        if (TargetType::NONE == run_param.target.m_type) //ref mat is used by opencv, need border
        {
            ref_sz = ref_sz + Sizes3(1, 1, 0);
        }

        // creat iauras
        Mat src = m_factory.GetRandomMat(-32768, 65535, run_param.elem_integral.first, src_sz, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        // create dst and ref mat
        Mat dst0 = m_factory.GetEmptyMat(run_param.elem_integral.second, dst_sz, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat dst1 = m_factory.GetEmptyMat(run_param.elem_integral.third, dst_sz, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat ref0 = m_factory.GetEmptyMat(run_param.elem_integral.fourth, ref_sz, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);
        Mat ref1 = m_factory.GetEmptyMat(run_param.elem_integral.fifth, ref_sz, AURA_MEM_DEFAULT, run_param.mat_size.m_strides);

        TestTime time_val0;
        MatCmpResult cmp_result0;
        MatCmpResult cmp_result1;
        TestResult result;

        result.input  = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_integral.first);
        result.output = run_param.mat_size.ToString() + " " + ElemTypesToString(run_param.elem_integral.second)  + " " +
                        ElemTypesToString(run_param.elem_integral.second);

        // run normal interface
        using integral_func = Status (*)(Context *, const Mat &, Mat &, Mat &, const OpTarget &);
        DT_S32 loop_count   = stress_count ? stress_count : 10;
        Status status_exec  = Status::OK;

        if (dst1.IsValid() && (!dst0.IsValid())) // Sequare mode, less data type
        {
            if ((ElemType::F64 == dst1.GetElemType()) ||
               (ElemType::U32 == dst1.GetElemType() && (ElemType::U8 == src.GetElemType() || ElemType::S8 == src.GetElemType())))
            {
                Executor<integral_func>(loop_count, 2, time_val0, IIntegral, m_ctx, src, dst0, dst1, run_param.target);
            }
            else //Other types are not supported
            {
                goto EXIT;
            }
        }
        else
        {
            Executor<integral_func>(loop_count, 2, time_val0, IIntegral, m_ctx, src, dst0, dst1, run_param.target);
        }

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val0;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "IntegralExecutor failed.\n");
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            result.accu_benchmark = "OpenCV::Integral";
            if (dst0.IsValid() && dst1.IsValid())
            {
                status_exec = Executor(10, 2, time_val0, CvIntegral, m_ctx, src, ref0, ref1);
            }
            else if (dst0.IsValid())
            {
                status_exec = Executor(10, 2, time_val0, CvIntegralNorm, m_ctx, src, ref0, 0);
            }
            else if (dst1.IsValid())
            {
                status_exec = Executor(10, 2, time_val0, CvIntegralNorm, m_ctx, src, ref1, 1);
            }

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvIntegral execute fail, type not support\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            else
            {
                result.perf_result["OpenCV"] = time_val0;
            }
        }
        else
        {
            result.accu_benchmark = "Integral(target::none)";
            status_exec = IIntegral(m_ctx, src, ref0, ref1, OpTarget::None());

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (TargetType::NONE == run_param.target.m_type)
        {
            if (dst0.IsValid() && dst1.IsValid())
            {
                if ((MatCompareIntegralBorder(m_ctx, dst0, ref0, cmp_result0) == Status::OK) && (MatCompareIntegralBorder(m_ctx, dst1, ref1, cmp_result1) == Status::OK))
                {
                    result.accu_status = (cmp_result0.status && cmp_result1.status) ? TestStatus::PASSED : TestStatus::FAILED;
                    result.accu_result = cmp_result0.ToString() + std::string(" sq_result: ") + cmp_result1.ToString();
                }
            }
            else if (dst0.IsValid()) //Normal
            {
                if ((MatCompareIntegralBorder(m_ctx, dst0, ref0, cmp_result0) == Status::OK))
                {
                    result.accu_status = (cmp_result0.status) ? TestStatus::PASSED : TestStatus::FAILED;
                    result.accu_result = cmp_result0.ToString();
                }
            }
            else if (dst1.IsValid()) //Sequare
            {
                if ((MatCompareIntegralBorder(m_ctx, dst1, ref1, cmp_result1) == Status::OK))
                {
                    result.accu_status = (cmp_result1.status) ? TestStatus::PASSED : TestStatus::FAILED;
                    result.accu_result = cmp_result1.ToString();
                }
            }
            else
            {
                AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail\n");
            }
        }
        else
        {
            if (dst0.IsValid() && dst1.IsValid())
            {
                if ((MatCompareIntegral(m_ctx, dst0, ref0, cmp_result0) == Status::OK) && (MatCompareIntegral(m_ctx, dst1, ref1, cmp_result1) == Status::OK))
                {
                    result.accu_status = (cmp_result0.status && cmp_result1.status) ? TestStatus::PASSED : TestStatus::FAILED;
                    result.accu_result = cmp_result0.ToString() + std::string(" sq_result: ") + cmp_result1.ToString();
                }
            }
            else if (dst0.IsValid())
            {
                if (MatCompareIntegral(m_ctx, dst0, ref0, cmp_result0) == Status::OK)
                {
                    result.accu_status = (cmp_result0.status) ? TestStatus::PASSED : TestStatus::FAILED;
                    result.accu_result = cmp_result0.ToString();
                }
            }
            else if (dst1.IsValid())
            {
                if (MatCompareIntegral(m_ctx, dst1, ref1, cmp_result1) == Status::OK)
                {
                    result.accu_status = (cmp_result1.status) ? TestStatus::PASSED : TestStatus::FAILED;
                    result.accu_result = cmp_result1.ToString();
                }
            }
            else
            {
                AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail\n");
            }
        }

EXIT:
        test_case->AddTestResult(result.perf_status && result.accu_status, result);
        // release mat
        m_factory.PutAllMats();

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};
#endif // AURA_OPS_MATRIX_INTEGRAL_UNIT_TEST_HPP__