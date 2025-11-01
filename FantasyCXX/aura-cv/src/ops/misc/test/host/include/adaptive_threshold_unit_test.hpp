/** @brief      : adaptive threshold uint test head for aura
 *  @file       : adaptive threshold_unit_test.hpp
 *  @author     : zhanghong16@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : April. 11, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_ADAPTIVE_THRESHOLD_UINT_TEST_HPP__
#define AURA_OPS_ADAPTIVE_THRESHOLD_UINT_TEST_HPP__

#include "aura/ops/misc.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct AdaptiveThresholdTestParam
{
    AdaptiveThresholdTestParam()
    {}

    AdaptiveThresholdTestParam(MI_F32 max_val, AdaptiveThresholdMethod method, MI_S32 type, MI_S32 block_size, MI_F32 delta)
     : max_val(max_val), method(method), type(type), block_size(block_size), delta(delta)
    {}

    friend std::ostream& operator<<(std::ostream &os, AdaptiveThresholdTestParam test_param)
    {
        os << "max_val:" << test_param.max_val << " block_size:" << test_param.block_size << " delta:" << test_param.delta << std::endl;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    MI_F32 max_val;
    AdaptiveThresholdMethod method;
    MI_S32 type;
    MI_S32 block_size;
    MI_F32 delta;
};

AURA_TEST_PARAM(MiscAdaptiveThresholdParam,
                MatSize,                    mat_sizes,
                AdaptiveThresholdTestParam, param,
                OpTarget,                   target);

#if !defined(AURA_BUILD_XPLORER)
static MI_S32 AdaptiveThresholdMethodToOpencv(AdaptiveThresholdMethod method)
{
    MI_S32 cv_type = -1;

    switch (method)
    {
        case AdaptiveThresholdMethod::ADAPTIVE_THRESH_MEAN_C:
        {
            cv_type = cv::ADAPTIVE_THRESH_MEAN_C;
            break;
        }

        case AdaptiveThresholdMethod::ADAPTIVE_THRESH_GAUSSIAN_C:
        {
            cv_type = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
            break;
        }

        default:
        {
            break;
        }
    }

    return cv_type;
}

static MI_S32 AdaptiveThresholdTypeToOpencv(MI_S32 type)
{
    MI_S32 cv_type = -1;

    switch (type)
    {
        case AURA_THRESH_BINARY:
        {
            cv_type = cv::THRESH_BINARY;
            break;
        }

        case AURA_THRESH_BINARY_INV:
        {
            cv_type = cv::THRESH_BINARY_INV;
            break;
        }

        default:
        {
            break;
        }
    }

    return cv_type;
}
#endif

static Status CvAdaptiveThreshold(Context *ctx, Mat &src, Mat &dst, MI_F32 max_val, AdaptiveThresholdMethod method,
                                        MI_S32 type, MI_S32 block_size, MI_F32 delta)
{
#if defined(ANDROID)
#  if !defined(__aarch64__)
    // there exists differences on rounding operation of convertTo Function in OPENCV and Aura projects,
    // differences usually occur when rounding data whose decimal is .5 (**.5)
    return Status::ERROR;
#  endif // __aarch64__
#endif // ANDROID

    Status ret = Status::OK;

#if !defined(AURA_BUILD_XPLORER)
    MI_S32 cv_method = AdaptiveThresholdMethodToOpencv(method);
    MI_S32 cv_th_type = AdaptiveThresholdTypeToOpencv(type);

    MI_S32 cv_type = ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    MI_S32 cn = src.GetSizes().m_channel;

    if (cv_type != CV_8UC(cn))
    {
        cv_type = -1;
        cv_method = -1;
        cv_type = -1;
    }

    if (cv_type != -1 && cv_method != -1 && cv_th_type != -1)
    {
        cv::Mat cv_src = MatToOpencv(src);
        cv::Mat cv_dst = MatToOpencv(dst);

        cv::adaptiveThreshold(cv_src, cv_dst, max_val, cv_method, cv_th_type, block_size, delta);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV threshold not support\n");
        ret = Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(max_val);
    AURA_UNUSED(method);
    AURA_UNUSED(type);
    AURA_UNUSED(block_size);
    AURA_UNUSED(delta);
#endif

    return ret;
}

class AdaptiveThresholdTest : public TestBase<MiscAdaptiveThresholdParam::TupleTable, MiscAdaptiveThresholdParam::Tuple>
{
public:
    AdaptiveThresholdTest(Context *ctx, MiscAdaptiveThresholdParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status ret = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        ret |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});

        if (ret != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in ThresholdTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        MiscAdaptiveThresholdParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (run_param.mat_sizes.m_sizes.m_width  < 800 &&
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
        MiscAdaptiveThresholdParam run_param(GetParam((index)));
        MI_F32 max_val = run_param.param.max_val;
        MI_F32 delta   = run_param.param.delta;

        // creat iauras
        Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, ElemType::U8, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);

        MatSize sz = run_param.mat_sizes;
        Mat dst    = m_factory.GetEmptyMat(ElemType::U8, sz.m_sizes, AURA_MEM_DEFAULT, sz.m_strides);
        Mat ref    = m_factory.GetEmptyMat(ElemType::U8, sz.m_sizes, AURA_MEM_DEFAULT, sz.m_strides);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = AdaptiveThresholdMethodToString(run_param.param.method) + " | " + ThresholdTypeToString(run_param.param.type);
        result.input  = run_param.mat_sizes.ToString() + " " + ElemTypesToString(ElemType::U8);
        result.output = sz.ToString();

        AURA_LOGD(m_ctx, AURA_TAG, "MiscAdaptiveThreshold param detail: mat_size(%s), param(%s)\n",
                  run_param.mat_sizes.ToString().c_str(), run_param.param.ToString().c_str());

        // run interface
        Status ret = Executor(loop_count, 2, time_val, IAdaptiveThreshold, m_ctx, src, dst, max_val, run_param.param.method, run_param.param.type,
                              run_param.param.block_size, delta, run_param.target);

        if (Status::OK == ret)
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

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            // because of the division of boxfilter has an error of 1, bypass compare.
            if (run_param.param.method == aura::AdaptiveThresholdMethod::ADAPTIVE_THRESH_MEAN_C)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvThreshold execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            ret = Executor(10, 2, time_val, CvAdaptiveThreshold, m_ctx, src, ref, run_param.param.max_val, run_param.param.method, run_param.param.type,
                                 run_param.param.block_size, run_param.param.delta);
            result.accu_benchmark = "OpenCV::AdaptiveThreshold";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvThreshold execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            ret = IAdaptiveThreshold(m_ctx, src, ref, max_val, run_param.param.method, run_param.param.type,
                                          run_param.param.block_size, delta, TargetType::NONE);
            result.accu_benchmark = "aura::Threshold(target::none)";

            if (ret != Status::OK)
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

        // release mat
        m_factory.PutMats(src, dst, ref);

        return 0;
    }

private:
    Context     *m_ctx;
    MatFactory   m_factory;
};

#endif // AURA_OPS_ADAPTIVE_THRESHOLD_UINT_TEST_HPP__