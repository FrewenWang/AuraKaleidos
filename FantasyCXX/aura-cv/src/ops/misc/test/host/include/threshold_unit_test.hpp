/** @brief      : threshold uint test head for aura
 *  @file       : threshold_unit_test.hpp
 *  @author     : zhanghong16@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : April. 11, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_THRESHOLD_UINT_TEST_HPP__
#define AURA_OPS_THRESHOLD_UINT_TEST_HPP__

#include "aura/ops/misc.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct ThresholdTestParam
{
    ThresholdTestParam()
    {}

    ThresholdTestParam(DT_F32 thresh, DT_F32 max_val, DT_S32 type) : thresh(thresh), max_val(max_val), type(type)
    {}

    friend std::ostream& operator<<(std::ostream &os, ThresholdTestParam threshold_test_param)
    {
        os << "thresh:" << threshold_test_param.thresh << " max_val:" << threshold_test_param.max_val << std::endl;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    DT_F32 thresh;
    DT_F32 max_val;
    DT_S32 type;
};

AURA_TEST_PARAM(MiscThresholdParam,
                aura::ElemType,     elem_type,
                aura::MatSize,      mat_sizes,
                ThresholdTestParam, param,
                aura::OpTarget,     target);

#if !defined(AURA_BUILD_XPLORER)
static DT_S32 ThresholdTypeToOpencv(DT_S32 method)
{
    DT_S32 cv_type = 0;

    switch (method & AURA_THRESH_MASK_HIGH)
    {
        case AURA_THRESH_OTSU:
        {
            cv_type = cv::THRESH_OTSU;
            break;
        }

        case AURA_THRESH_TRIANGLE:
        {
            cv_type = cv::THRESH_TRIANGLE;
            break;
        }

        default:
        {
            break;
        }
    }

    switch (method & AURA_THRESH_MASK_LOW)
    {
        case AURA_THRESH_BINARY:
        {
            cv_type |= cv::THRESH_BINARY;
            break;
        }

        case AURA_THRESH_BINARY_INV:
        {
            cv_type |= cv::THRESH_BINARY_INV;
            break;
        }

        case AURA_THRESH_TRUNC:
        {
            cv_type |= cv::THRESH_TRUNC;
            break;
        }

        case AURA_THRESH_TOZERO:
        {
            cv_type |= cv::THRESH_TOZERO;
            break;
        }

        case AURA_THRESH_TOZERO_INV:
        {
            cv_type |= cv::THRESH_TOZERO_INV;
            break;
        }

        default:
        {
            cv_type = -1;
            break;
        }
    }

    return cv_type;
}
#endif

static aura::Status CvThreshold(aura::Context *ctx, aura::Mat &src, aura::Mat &dst, DT_F32 thresh, DT_S32 max_val, DT_S32 type)
{
    aura::Status ret = aura::Status::OK;

#if !defined(AURA_BUILD_XPLORER)
    DT_S32 cv_method = ThresholdTypeToOpencv(type);

    DT_S32 cv_type = aura::ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    DT_S32 cn = src.GetSizes().m_channel;

    if (CV_32SC(cn) == cv_type || CV_8SC(cn) == cv_type)
    {
        cv_type = -1;
        cv_method = -1;
    }

    if (cv_type != -1 && cv_method != -1)
    {
        cv::Mat cv_src = aura::MatToOpencv(src);
        cv::Mat cv_dst = aura::MatToOpencv(dst);

        cv::threshold(cv_src, cv_dst, thresh, max_val, cv_method);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV threshold not support\n");
        ret = aura::Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(thresh);
    AURA_UNUSED(max_val);
    AURA_UNUSED(type);
#endif

    return ret;
}

class ThresholdTest : public aura::TestBase<MiscThresholdParam::TupleTable, MiscThresholdParam::Tuple>
{
public:
    ThresholdTest(aura::Context *ctx, MiscThresholdParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        aura::Status ret = aura::Status::OK;

        std::string data_file = aura::UnitTest::GetInstance()->GetDataPath() + "comm/";

        ret |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", aura::ElemType::U8, {487, 487});
        ret |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  aura::ElemType::U8, {512, 512, 3});
        ret |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", aura::ElemType::U8, {256, 256, 2});

        if (ret != aura::Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in ThresholdTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        MiscThresholdParam run_param(GetParam((index)));
        if (aura::UnitTest::GetInstance()->IsStressMode())
        {
            if (aura::TargetType::NONE == run_param.target.m_type)
            {
                if (2 == run_param.mat_sizes.m_sizes.m_channel &&
                    run_param.mat_sizes.m_sizes.m_width  < 800 &&
                    run_param.mat_sizes.m_sizes.m_height < 600)
                {
                    return aura::Status::OK;
                }
                else
                {
                    return aura::Status::ERROR;
                }
            }
        }
        return aura::Status::OK;
    }

    DT_S32 RunOne(DT_S32 index, aura::TestCase *test_case, DT_S32 stress_count) override
    {
        // get next param set
        MiscThresholdParam run_param(GetParam((index)));
        DT_F32 thresh = run_param.param.thresh;
        DT_F32 max_val = run_param.param.max_val;

        // creat iauras
        aura::Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, run_param.elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        aura::Mat src_cv = m_factory.GetDerivedMat(1.0f, 0.0f, aura::ElemType::F16 == run_param.elem_type ? aura::ElemType::F32 : run_param.elem_type,
                                                   run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);

        aura::MatSize sz = run_param.mat_sizes;
        aura::Mat dst = m_factory.GetEmptyMat(run_param.elem_type, sz.m_sizes, AURA_MEM_DEFAULT, sz.m_strides);
        aura::Mat ref = m_factory.GetEmptyMat(((aura::ElemType::F16 == run_param.elem_type) && (aura::TargetType::NONE == run_param.target.m_type))
                                              ? aura::ElemType::F32 : run_param.elem_type, sz.m_sizes, AURA_MEM_DEFAULT, sz.m_strides);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        aura::TestTime time_val;
        aura::MatCmpResult cmp_result;
        aura::TestResult result;

        result.param  = aura::ThresholdTypeToString(run_param.param.type);
        result.input  = run_param.mat_sizes.ToString() + " " + aura::ElemTypesToString(run_param.elem_type);
        result.output = sz.ToString();

        // run interface
        aura::Status ret = aura::Executor(loop_count, 2, time_val, aura::IThreshold, m_ctx, src, dst, thresh, max_val, run_param.param.type,
                                          run_param.target);

        if (aura::Status::OK == ret)
        {
            result.perf_result[aura::TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = aura::TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail, %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = aura::TestStatus::FAILED;
            result.accu_status = aura::TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (aura::TargetType::NONE == run_param.target.m_type)
        {
            ret = aura::Executor(10, 2, time_val, CvThreshold, m_ctx, src_cv, ref, run_param.param.thresh, run_param.param.max_val, run_param.param.type);
            result.accu_benchmark = "OpenCV::Threshold";

            if (ret != aura::Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvThreshold execute fail\n");
                result.accu_status = aura::TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            ret = aura::IThreshold(m_ctx, src, ref, thresh, max_val, run_param.param.type, aura::TargetType::NONE);
            result.accu_benchmark = "aura::Threshold(target::none)";

            if (ret != aura::Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = aura::TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        aura::MatCompare(m_ctx, dst, ref, cmp_result, 1);
        result.accu_status = cmp_result.status ? aura::TestStatus::PASSED : aura::TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(src, dst, ref, src_cv);

        return 0;
    }

private:
    aura::Context     *m_ctx;
    aura::MatFactory   m_factory;
};

#endif // AURA_OPS_THRESHOLD_UINT_TEST_HPP__