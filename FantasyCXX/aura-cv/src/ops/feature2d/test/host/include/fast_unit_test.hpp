#ifndef AURA_OPS_FAST_UINT_TEST_HPP__
#define AURA_OPS_FAST_UINT_TEST_HPP__

#include "aura/ops/feature2d.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(FastParam,
                ElemType,         elem_type,
                MatSize,          mat_sizes,
                DT_S32,           threshold,
                DT_BOOL,          nonmax_suppression,
                FastDetectorType, type,
                OpTarget,         target);

#if !defined(AURA_BUILD_XPLORER)
static cv::FastFeatureDetector::DetectorType FastDetectorTypeToOpencv(Context *ctx, FastDetectorType type)
{
    cv::FastFeatureDetector::DetectorType cv_type;

    switch(type)
    {
        case FastDetectorType::FAST_5_8:
        {
            cv_type = cv::FastFeatureDetector::TYPE_5_8;
            break;
        }

        case FastDetectorType::FAST_7_12:
        {
            cv_type = cv::FastFeatureDetector::TYPE_7_12;
            break;
        }

        case FastDetectorType::FAST_9_16:
        {
            cv_type = cv::FastFeatureDetector::TYPE_9_16;
            break;
        }

        default:
        {
            AURA_LOGE(ctx, AURA_TAG, "fast detector type not supported, change to default type 9_16\n");
            cv_type = cv::FastFeatureDetector::TYPE_9_16;
            break;
        }
    }

    return cv_type;
}
#endif

static Status CvFast(Context *ctx, Mat &src, std::vector<KeyPoint> &keypoints, DT_S32 threshold, DT_BOOL nonmax_suppression, FastDetectorType type)
{
#if !defined(AURA_BUILD_XPLORER)
    cv::FastFeatureDetector::DetectorType cv_fast_type = FastDetectorTypeToOpencv(ctx, type);

    keypoints.clear();

    if ((src.GetElemType() != ElemType::U8))
    {
        AURA_LOGE(ctx, AURA_TAG, "CV FAST not support\n");
        return Status::ERROR;
    }

    DT_S32 src_cv_type = ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    DT_S32 cv_type = 0;

    if ((src_cv_type != CV_8UC1))
    {
        cv_type = -1;
    }

    if (cv_type != -1)
    {
        cv::Mat cv_src = MatToOpencv(src);
        std::vector<cv::KeyPoint> cv_keypoints;

        cv::FAST(cv_src, cv_keypoints, threshold, nonmax_suppression, cv_fast_type);

        for (size_t n = 0; n < cv_keypoints.size(); n++)
        {
            KeyPoint tmp(cv_keypoints[n].pt.x, cv_keypoints[n].pt.y, cv_keypoints[n].size, cv_keypoints[n].angle, cv_keypoints[n].response, cv_keypoints[n].octave, cv_keypoints[n].class_id);
            keypoints.push_back(tmp);
        }
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV fast not support\n");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(keypoints);
    AURA_UNUSED(threshold);
    AURA_UNUSED(nonmax_suppression);
    AURA_UNUSED(type);
#endif

    return Status::OK;
}

class FastTest : public TestBase<FastParam::TupleTable, FastParam::Tuple>
{
public:
    FastTest(Context *ctx, FastParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;
        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in ResizeTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        FastParam run_param(GetParam((index)));
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
        FastParam run_param(GetParam((index)));

        // creat iauras
        Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, run_param.elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);

        std::vector<KeyPoint> keypoints_dst;
        std::vector<KeyPoint> keypoints_ref;
        keypoints_dst.reserve(1024);
        keypoints_ref.reserve(1024);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        UnorderedCmpResult<KeyPoint> cmp_result;
        TestResult result;

        // run interface
        DT_S32 max_num_corners = 3000;
        Status ret = Executor(loop_count, 2, time_val, IFast, m_ctx, src, keypoints_dst, run_param.threshold,
                              run_param.nonmax_suppression, run_param.type, max_num_corners,  run_param.target);

        result.param  = FastDetectorTypeToString(run_param.type) + " | threshold:" + std::to_string(run_param.threshold) +
                                                 " | nonmax_suppression:" + std::to_string(run_param.nonmax_suppression);
        result.input  = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = "keypoint num:" + std::to_string(keypoints_dst.size());

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
            ret = Executor(10, 2, time_val, CvFast, m_ctx, src, keypoints_ref, run_param.threshold, run_param.nonmax_suppression, run_param.type);
            result.accu_benchmark = "OpenCV::Fast";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvFast execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            ret = IFast(m_ctx, src, keypoints_ref, run_param.threshold, run_param.nonmax_suppression, run_param.type, max_num_corners, TargetType::NONE);
            result.accu_benchmark = "Fast(target::none)";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        UnorderedCompare<KeyPoint>(m_ctx, keypoints_dst, keypoints_ref, cmp_result);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(src);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_FAST_UINT_TEST_HPP__