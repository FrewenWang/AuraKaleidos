/** @brief      : calc hist unit test head for aura
 *  @file       : calc_hist_unit_test.hpp
 *  @author     : zhangqiongqiong@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : July. 15, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_CALC_HIST_UNIT_TEST_HPP__
#define AURA_OPS_CALC_HIST_UNIT_TEST_HPP__

#include "aura/ops/hist.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct CalcHistTestParam
{
    CalcHistTestParam()
    {}

    CalcHistTestParam(ElemType elem_type, DT_S32 channel, DT_S32 hist_size, Scalar range, DT_BOOL accumulate, DT_BOOL use_mask) : 
                      elem_type(elem_type), channel(channel), hist_size(hist_size), range(range), accumulate(accumulate), use_mask(use_mask)
    {}

    friend std::ostream& operator<<(std::ostream &os, const CalcHistTestParam &calc_hist_test_param)
    {
        os << "channel:" << calc_hist_test_param.channel << " | hist_size:" << calc_hist_test_param.hist_size 
           << " | ranges:[" << static_cast<DT_S32>(calc_hist_test_param.range.m_val[0]) 
           << " " << static_cast<DT_S32>(calc_hist_test_param.range.m_val[1])
           << "] | accumulate:" << calc_hist_test_param.accumulate << " | use_mask:" << calc_hist_test_param.use_mask;
        return os;
    }

    std::string ToString()
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    ElemType elem_type;
    DT_S32   channel;
    DT_S32   hist_size;
    Scalar   range;
    DT_BOOL  accumulate;
    DT_BOOL  use_mask;
};

AURA_TEST_PARAM(CalcHistParam,
                MatSize,           mat_sizes,
                CalcHistTestParam, param,
                OpTarget,          target);

static Status CvCalcHist(Context *ctx, Mat &src, DT_S32 channel, std::vector<DT_U32> &dst, DT_S32 hist_size, 
                         Scalar &range, Mat &mask, DT_BOOL accumulate)
{
#if !defined(AURA_BUILD_XPLORER)
    DT_S32 src_cv_type = ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    if (src_cv_type != -1)
    {
        cv::Mat cv_src  = MatToOpencv(src);
        cv::Mat cv_dst  = cv::Mat(hist_size, 1, CV_32SC1, dst.data());
        cv::Mat cv_mask = mask.IsValid() ? MatToOpencv(mask) : cv::Mat();

        DT_F32 cv_range[]        = {static_cast<DT_F32>(range.m_val[0]), static_cast<DT_F32>(range.m_val[1])};
        const DT_F32 *hist_range = {cv_range};
        cv::Mat result;
        cv::calcHist(&cv_src, 1, &channel, cv_mask, result, 1, &hist_size, &hist_range, true, accumulate);
        result.convertTo(cv_dst, cv_dst.type());
        if ((DT_U32 *)cv_dst.data != dst.data())
        {
            memcpy(dst.data(), cv_dst.data, cv_dst.total() * cv_dst.elemSize());
        }
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "mat type not support\n");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(channel);
    AURA_UNUSED(dst);
    AURA_UNUSED(hist_size);
    AURA_UNUSED(range);
    AURA_UNUSED(mask);
    AURA_UNUSED(accumulate);
#endif

    return Status::OK;
}

class CalcHistTest : public TestBase<CalcHistParam::TupleTable, CalcHistParam::Tuple>
{
public:
    CalcHistTest(Context *ctx, CalcHistParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status  = m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in CalcHistTest\n");
        }
    }

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        CalcHistParam run_param(GetParam((index)));
        ElemType elem_type = run_param.param.elem_type;
        MatSize mat_sizes  = run_param.mat_sizes;
        DT_S32 channel     = run_param.param.channel;
        DT_S32 hist_size   = run_param.param.hist_size;
        Scalar range       = run_param.param.range;
        DT_BOOL accumulate = run_param.param.accumulate;
        DT_BOOL use_mask   = run_param.param.use_mask;

        Sizes3 mask_size  = {mat_sizes.m_sizes.m_height, mat_sizes.m_sizes.m_width, 1};
        Mat src  = m_factory.GetDerivedMat(1.0f, 0.0f, elem_type, mat_sizes.m_sizes, AURA_MEM_DEFAULT, mat_sizes.m_strides);
        Mat mask = use_mask ? m_factory.GetRandomMat(0, 255, ElemType::U8, mask_size, AURA_MEM_DEFAULT, src.GetStrides()) : Mat();
        std::vector<DT_U32> dst(hist_size, 0);
        std::vector<DT_U32> ref(hist_size, 0);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        ArrayCmpResult cmp_result;
        TestResult result;

        result.param  = run_param.param.ToString();
        result.input  = mat_sizes.ToString() + " " + ElemTypesToString(elem_type);
        result.output = std::to_string(hist_size) + " " + ElemTypesToString(ElemType::U32);

        Status status_exec = Executor(loop_count, 2, time_val, ICalcHist, m_ctx, src, channel, dst, hist_size,
                                                  range, mask, accumulate, run_param.target);

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
            status_exec = Executor(10, 2, time_val, CvCalcHist, m_ctx, src, channel, ref, hist_size, range, mask, accumulate);
            result.accu_benchmark = "OpenCV::CalcHist";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvCalcHist execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = ICalcHist(m_ctx, src, channel, ref, hist_size, range, mask, accumulate, TargetType::NONE);
            result.accu_benchmark = "CalcHist(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        if (dst.size() != ref.size())
        {
            AURA_LOGE(m_ctx, AURA_TAG, "dst size(%zu) != ref size(%zu)\n", dst.size(), ref.size());
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        ArrayCompare<decltype(dst.begin()), RelativeDiff>(m_ctx, dst.begin(), ref.begin(), hist_size, cmp_result, 1e-10);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        m_factory.PutMats(src, mask);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_CALC_HIST_UNIT_TEST_HPP__