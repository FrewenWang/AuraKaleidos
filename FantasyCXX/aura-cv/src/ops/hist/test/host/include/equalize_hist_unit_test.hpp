/** @brief      : equalize hist unit test head for aura
 *  @file       : equalize_hist_unit_test.hpp
 *  @author     : zhangqiongqiong@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : July. 15, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_HIST_UNIT_TEST_HPP__
#define AURA_OPS_HIST_UNIT_TEST_HPP__

#include "aura/ops/hist.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(EqualizeHistParam,
                ElemType, elem_type,
                MatSize,  mat_sizes,
                OpTarget, target);

static Status CvEqualizeHist(Context *ctx, Mat &src, Mat &dst)
{
#if !defined(AURA_BUILD_XPLORER)
    MI_S32 src_cv_type = ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    MI_S32 dst_cv_type = ElemTypeToOpencv(dst.GetElemType(), dst.GetSizes().m_channel);
    if (src_cv_type != -1 && dst_cv_type != -1)
    {
        cv::Mat cv_src = MatToOpencv(src);
        cv::Mat cv_dst = MatToOpencv(dst);

        cv::equalizeHist(cv_src, cv_dst);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "mat type not support\n");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
#endif

    return Status::OK;
}

class EqualizeHistTest : public TestBase<EqualizeHistParam::TupleTable, EqualizeHistParam::Tuple>
{
public:
    EqualizeHistTest(Context *ctx, EqualizeHistParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status = m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in EqualizeHistTest\n");
        }
    }

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        EqualizeHistParam run_param(GetParam((index)));

        Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, ElemType::U8, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        Mat dst = m_factory.GetEmptyMat(ElemType::U8, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        Mat ref = m_factory.GetEmptyMat(ElemType::U8, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = std::string();
        result.input  = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);

        Status status_exec = Executor(loop_count, 2, time_val, IEqualizeHist, m_ctx, src, dst, run_param.target);
        
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail\n");
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        if (TargetType::NONE == run_param.target.m_type)
        {
            status_exec = Executor(10, 2, time_val, CvEqualizeHist, m_ctx, src, ref);
            result.accu_benchmark = "OpenCV::EqualizeHist";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvEqualizeHist execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IEqualizeHist(m_ctx, src, ref, TargetType::NONE);
            result.accu_benchmark = "EqualizeHist(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        MatCompare(m_ctx, dst, ref, cmp_result, 1);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        m_factory.PutMats(src, dst, ref);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_HIST_UNIT_TEST_HPP__