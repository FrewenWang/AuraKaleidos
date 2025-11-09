/** @brief      : hough circles uint test head for aura
 *  @file       : hough_circles_unit_test.hpp
 *  @author     : zhanghong16@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : June. 29, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_HOUGH_CIRCLES_UINT_TEST_HPP__
#define AURA_OPS_HOUGH_CIRCLES_UINT_TEST_HPP__

#include "aura/ops/misc.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct HoughCirclesTestParam
{
    HoughCirclesTestParam()
    {}

    HoughCirclesTestParam(HoughCirclesMethod type, DT_F64 dp, DT_F64 min_dist, DT_F64 canny_thresh, DT_F64 acc_thresh,
                          DT_S32 min_radius, DT_S32 max_radius)
                          : type(type), dp(dp), min_dist(min_dist), canny_thresh(canny_thresh), acc_thresh(acc_thresh), min_radius(min_radius),
                          max_radius(max_radius)
    {}

    friend std::ostream& operator<<(std::ostream &os, HoughCirclesTestParam hough_circles_test_param)
    {
        os << "HoughCircles | type:" << hough_circles_test_param.type << " | dp:" << hough_circles_test_param.dp
           << " | min_dist:" << hough_circles_test_param.min_dist << " | canny_thresh:" << hough_circles_test_param.canny_thresh
           << " | acc_thresh:" << hough_circles_test_param.acc_thresh << " | min_radius:" << hough_circles_test_param.min_radius
           << " | max_radius:" << hough_circles_test_param.max_radius << std::endl;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    HoughCirclesMethod type;
    DT_F64 dp;
    DT_F64 min_dist;
    DT_F64 canny_thresh;
    DT_F64 acc_thresh;
    DT_S32 min_radius;
    DT_S32 max_radius;
};

AURA_TEST_PARAM(HoughCirclesParam,
                ElemType,        elem_type,
                MatSize,         mat_sizes,
                HoughCirclesTestParam, param,
                OpTarget,        target);

#if !defined(AURA_BUILD_XPLORER)
static DT_S32 HoughCirclesTypeToOpencv(HoughCirclesMethod type)
{
    DT_S32 cv_type = -1;

    switch (type)
    {
        case HoughCirclesMethod::HOUGH_GRADIENT:
        {
            cv_type = cv::HOUGH_GRADIENT;
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

static Status CvHoughCircles(Context *ctx, Mat &mat, std::vector<Scalar> &circles, HoughCirclesMethod type,
                                   DT_F64 dp, DT_F64 min_dist, DT_F64 canny_thresh, DT_F64 acc_thresh, DT_S32 min_radius, DT_S32 max_radius)
{
    Status ret = Status::OK;

#if !defined(AURA_BUILD_XPLORER)
    circles.clear();

    DT_S32 cv_method = HoughCirclesTypeToOpencv(type);

    if (mat.GetElemType() != ElemType::U8)
    {
        AURA_LOGE(ctx, AURA_TAG, "CV CvHoughCircles only support type u8\n");
        return Status::ERROR;
    }

    DT_S32 mat_cv_type = ElemTypeToOpencv(mat.GetElemType(), mat.GetSizes().m_channel);
    DT_S32 cv_type = 0;

    if (CV_8UC1 != mat_cv_type)
    {
        cv_type = -1;
    }

    if (cv_type != -1)
    {
        cv::Mat mat_src = MatToOpencv(mat);

        std::vector<cv::Vec4f> cv_circles;
        cv::HoughCircles(mat_src, cv_circles, cv_method, dp, min_dist, canny_thresh, acc_thresh, min_radius, max_radius);

        for (DT_U64 i = 0; i < cv_circles.size(); i++)
        {
            circles.emplace_back(cv_circles[i][0], cv_circles[i][1], cv_circles[i][2], cv_circles[i][3]);
        }
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV HoughCircles not support\n");
        ret = Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(mat);
    AURA_UNUSED(circles);
    AURA_UNUSED(type);
    AURA_UNUSED(dp);
    AURA_UNUSED(min_dist);
    AURA_UNUSED(canny_thresh);
    AURA_UNUSED(acc_thresh);
    AURA_UNUSED(min_radius);
    AURA_UNUSED(max_radius);
#endif

    return ret;
}

static DT_VOID HoughCirclesConvert(const std::vector<Scalar> &circles, std::vector<Scalar> &cvt_circles)
{
    for (auto it = circles.begin(); it != circles.end(); ++it)
    {
        cvt_circles.emplace_back(it->m_val[0], it->m_val[1], it->m_val[2], 0.f);
    }
}

class HoughCirclesTest : public TestBase<HoughCirclesParam::TupleTable, HoughCirclesParam::Tuple>
{
public:
    HoughCirclesTest(Context *ctx, HoughCirclesParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status = m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in HoughCirclesTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        HoughCirclesParam run_param(GetParam((index)));
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

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // get next param set
        HoughCirclesParam run_param(GetParam((index)));

        // creat iauras
        Mat mat = m_factory.GetDerivedMat(1.0f, 0.0f, run_param.elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);

        std::vector<Scalar> circles_dst;
        std::vector<Scalar> circles_ref;
        std::vector<Scalar> cvt_circles_dst;
        std::vector<Scalar> cvt_circles_ref;
        circles_dst.reserve(256);
        circles_ref.reserve(256);
        cvt_circles_dst.reserve(256);
        cvt_circles_ref.reserve(256);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        UnorderedCmpResult<Scalar> cmp_result;
        TestResult result;

        // run interface
        Status ret = Executor(loop_count, 2, time_val, IHoughCircles, m_ctx, mat, circles_dst, run_param.param.type,
                                          run_param.param.dp, run_param.param.min_dist, run_param.param.canny_thresh, run_param.param.acc_thresh,
                                          run_param.param.min_radius, run_param.param.max_radius, run_param.target);

        result.param = run_param.param.ToString();
        result.input = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = "circles num:" + std::to_string(circles_dst.size());

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
            ret = Executor(10, 2, time_val, CvHoughCircles, m_ctx, mat, circles_ref, run_param.param.type, run_param.param.dp,
                                 run_param.param.min_dist, run_param.param.canny_thresh, run_param.param.acc_thresh, run_param.param.min_radius,
                                 run_param.param.max_radius);
            result.accu_benchmark = "OpenCV::HoughCircles";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvHoughCircles failed\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            ret = IHoughCircles(m_ctx, mat, circles_ref, run_param.param.type, run_param.param.dp, run_param.param.min_dist,
                                     run_param.param.canny_thresh, run_param.param.acc_thresh, run_param.param.min_radius,
                                     run_param.param.max_radius, TargetType::NONE);
            result.accu_benchmark = "HoughCircles(target::none)";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        HoughCirclesConvert(circles_dst, cvt_circles_dst);
        HoughCirclesConvert(circles_ref, cvt_circles_ref);
        UnorderedCompare<Scalar>(m_ctx, cvt_circles_dst, cvt_circles_ref, cmp_result,
                               Tolerate<Scalar>(0.95, Scalar(1e-5f, 1e-5f, 1.f, 1e-5f)));

        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(mat);

        return 0;
    }

private:
    Context     *m_ctx;
    MatFactory   m_factory;
};

#endif // AURA_OPS_HOUGH_CIRCLES_UINT_TEST_HPP__