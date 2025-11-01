#ifndef AURA_OPS_TOMASI_UINT_TEST_HPP__
#define AURA_OPS_TOMASI_UINT_TEST_HPP__

#include "aura/ops/feature2d.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct TomasiTestParam
{
    TomasiTestParam()
    {}

    TomasiTestParam(MI_S32 max_corners, MI_F64 quality_level, MI_F64 min_distance,
                    MI_S32 block_size, MI_S32 gradient_size, MI_BOOL use_harris, MI_F64 harris_k)
                    : max_corners(max_corners), quality_level(quality_level), min_distance(min_distance),
                      block_size(block_size), gradient_size(gradient_size), use_harris(use_harris), harris_k(harris_k)
    {}

    friend std::ostream& operator<<(std::ostream &os, TomasiTestParam tomasi_test_param)
    {
        os << "max_corners:" << tomasi_test_param.max_corners << " | quality_level:" << tomasi_test_param.quality_level
           << " | min_distance:" << tomasi_test_param.min_distance << " | block_size:" << tomasi_test_param.block_size << " | gradient_size:" << tomasi_test_param.gradient_size
           << " | use_harris:" << tomasi_test_param.use_harris << " | harris_k:" << tomasi_test_param.harris_k;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    MI_S32 max_corners;
    MI_F64 quality_level;
    MI_F64 min_distance;
    MI_S32 block_size;
    MI_S32 gradient_size;
    MI_BOOL use_harris;
    MI_F64 harris_k;
};

AURA_TEST_PARAM(TomasiParam,
                ElemType,        elem_type,
                MatSize,         mat_sizes,
                TomasiTestParam, param,
                OpTarget,        target);

static Status CvGoodFeaturesToTrack(Context *ctx, Mat &src, std::vector<KeyPoint> &key_points, const TomasiTestParam &param)
{
    Status ret = Status::OK;
#if !defined(AURA_BUILD_XPLORER)
    if ((src.GetElemType() != ElemType::U8 && src.GetElemType() != ElemType::F32)
         || src.GetSizes().m_channel != 1)
    {
        AURA_LOGE(ctx, AURA_TAG, "CV goodFeaturesToTrack not support\n");
        return Status::ERROR;
    }

    key_points.clear();

    MI_S32 src_cv_type = ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    MI_S32 cv_type = 0;

    if (CV_8UC1 != src_cv_type && CV_32FC1 != src_cv_type)
    {
        cv_type = -1;
    }

    if (cv_type != -1)
    {
        cv::Mat cv_src = MatToOpencv(src);
        std::vector<cv::Point2f> cv_keypoints;

        cv::goodFeaturesToTrack(cv_src, cv_keypoints, param.max_corners, param.quality_level, param.min_distance, cv::Mat(), param.block_size, param.gradient_size, param.use_harris, param.harris_k);
        for (size_t n = 0; n < cv_keypoints.size(); n++)
        {
            KeyPoint tmp(cv_keypoints[n].x, cv_keypoints[n].y, 0.f, -1.f, 0.f, 0, -1);
            key_points.push_back(tmp);
        }
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV goodFeaturesToTrack not support\n");
        ret = Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(key_points);
    AURA_UNUSED(param);
#endif

    return ret;
}

class TomasiTest : public TestBase<TomasiParam::TupleTable, TomasiParam::Tuple>
{
public:
    TomasiTest(Context *ctx, TomasiParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
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
        TomasiParam run_param(GetParam((index)));
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
        TomasiParam run_param(GetParam((index)));
        // creat iauras
        MI_F32 alpha = run_param.elem_type == ElemType::U8 ? 1.0f : 1 / 255.f;
        Mat src = m_factory.GetDerivedMat(alpha, 0.0f, run_param.elem_type, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);

        std::vector<KeyPoint> dst;
        std::vector<KeyPoint> ref;

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        UnorderedCmpResult<KeyPoint> cmp_result;
        TestResult result;

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, GoodFeaturesToTrack, m_ctx, src, dst, run_param.param.max_corners,
                                      run_param.param.quality_level, run_param.param.min_distance, run_param.param.block_size, run_param.param.gradient_size,
                                      run_param.param.use_harris, run_param.param.harris_k, run_param.target);

        result.param  = run_param.param.ToString();
        result.input  = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = "keyPoint num:" + std::to_string(dst.size());
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
            status_exec = Executor(10, 2, time_val, CvGoodFeaturesToTrack, m_ctx, src, ref, run_param.param);
            result.accu_benchmark = "OpenCV::goodFeaturesToTrack";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvCornerTomasi execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = GoodFeaturesToTrack(m_ctx, src, ref, run_param.param.max_corners, run_param.param.quality_level, run_param.param.min_distance,
                                              run_param.param.block_size, run_param.param.gradient_size, run_param.param.use_harris, run_param.param.harris_k, TargetType::NONE);
            result.accu_benchmark = "GoodFeaturesToTrack(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        UnorderedCompare<KeyPoint>(m_ctx, dst, ref, cmp_result, Tolerate<KeyPoint>(0.99f));
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

#endif // AURA_OPS_TOMASI_UINT_TEST_HPP__