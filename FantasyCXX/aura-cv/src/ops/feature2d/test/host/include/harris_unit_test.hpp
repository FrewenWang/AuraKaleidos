#ifndef AURA_OPS_HARRIS_UINT_TEST_HPP__
#define AURA_OPS_HARRIS_UINT_TEST_HPP__

#include "aura/ops/feature2d.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct HarrisTestParam
{
    HarrisTestParam()
    {}

    HarrisTestParam(MI_S32 block_size, MI_S32 k_size, MI_F64 k) : block_size(block_size), k_size(k_size), k(k)
    {}

    friend std::ostream& operator<<(std::ostream &os, HarrisTestParam harris_test_param)
    {
        os << "block_size:" << harris_test_param.block_size << " | k_size:" << harris_test_param.k_size << " | k:" << harris_test_param.k << std::endl;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    MI_S32 block_size;
    MI_S32 k_size;
    MI_F64 k;
};

AURA_TEST_PARAM(HarrisParam,
                ElemType,        elem_type,
                MatSize,         mat_sizes,
                HarrisTestParam, param,
                BorderType,      border_type,
                OpTarget,        target);

static Status CvCornerHarris(Context *ctx, Mat &src, Mat &dst, const HarrisTestParam &param, BorderType border_type)
{
    Status ret = Status::OK;
#if !defined(AURA_BUILD_XPLORER)
    if ((src.GetElemType() != ElemType::U8 && src.GetElemType() != ElemType::F32)
         || src.GetSizes().m_channel != 1)
    {
        AURA_LOGE(ctx, AURA_TAG, "CV cornerHarris not support\n");
        return Status::ERROR;
    }

    if ((dst.GetElemType() != ElemType::F32) || dst.GetSizes().m_channel != 1)
    {
        AURA_LOGE(ctx, AURA_TAG, "CV cornerHarris not support\n");
        return Status::ERROR;
    }

    MI_S32 src_cv_type = ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    MI_S32 dst_cv_type = ElemTypeToOpencv(dst.GetElemType(), dst.GetSizes().m_channel);
    MI_S32 cv_type = 0;

    if ((CV_8UC1 != src_cv_type && CV_32FC1 != src_cv_type) || CV_32FC1 != dst_cv_type)
    {
        cv_type = -1;
    }

    if (cv_type != -1)
    {
        cv::Mat cv_src = MatToOpencv(src);
        cv::Mat cv_dst = MatToOpencv(dst);

        MI_S32 cv_border_type = BorderTypeToOpencv(border_type);
        cv::cornerHarris(cv_src, cv_dst, param.block_size, param.k_size, param.k, cv_border_type);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV cornerHarris not support\n");
        ret = Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(param);
    AURA_UNUSED(border_type);
#endif

    return ret;
}

class HarrisTest : public TestBase<HarrisParam::TupleTable, HarrisParam::Tuple>
{
public:
    HarrisTest(Context *ctx, HarrisParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in ResizeTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        HarrisParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (BorderType::REFLECT_101 == run_param.border_type &&
                    run_param.mat_sizes.m_sizes.m_width  < 800 &&
                    run_param.mat_sizes.m_sizes.m_height < 600)
                {
                    return Status::OK;
                }
                else
                {
                    return Status::ERROR;
                }
            }
        }
        return Status::OK;
    }

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // get next param set
        HarrisParam run_param(GetParam((index)));

        // creat iauras
        MI_F32 alpha = run_param.elem_type == ElemType::U8 ? 1.0f : 1 / 255.f;
        Mat src = m_factory.GetDerivedMat(alpha, 0.0f, run_param.elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        Mat dst = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        Mat ref = m_factory.GetEmptyMat(ElemType::F32, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        // run interface
        Scalar border_value;
        Status status_exec = Executor(loop_count, 2, time_val, IHarris, m_ctx, src, dst, run_param.param.block_size,
                                      run_param.param.k_size, run_param.param.k, run_param.border_type, border_value, run_param.target);

        result.param  = run_param.param.ToString() + " | border_type:" + BorderTypeToString(run_param.border_type);
        result.input  = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = run_param.mat_sizes.ToString() + " " + ElemTypesToString(ElemType::F32);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
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
        if (TargetType::NONE == run_param.target.m_type)
        {
            status_exec = Executor(10, 2, time_val, CvCornerHarris, m_ctx, src, ref, run_param.param, run_param.border_type);
            result.accu_benchmark = "OpenCV::cornerHarris";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvCornerHarris execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IHarris(m_ctx, src, ref, run_param.param.block_size, run_param.param.k_size, run_param.param.k, run_param.border_type, border_value, TargetType::NONE);
            result.accu_benchmark = "CornerHarris(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        MatCompare(m_ctx, dst, ref, cmp_result, 1e-5, 1e-5);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(src, dst, ref);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_HARRIS_UINT_TEST_HPP__