#ifndef AURA_OPS_CANNY_UINT_TEST_HPP__
#define AURA_OPS_CANNY_UINT_TEST_HPP__

#include "aura/ops/feature2d.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(CannyPixelParam,
                ElemType, elem_type,
                MatSize,  imat_sizes,
                MI_F64,   low_thresh,
                MI_F64,   high_thresh,
                MI_S32,   aperture_size,
                MI_BOOL,  l2_gradient,
                OpTarget, target);

AURA_TEST_PARAM(CannyGradientParam,
                ElemType, imat_elem_type,
                ElemType, omat_elem_type,
                MatSize,  imat_sizes,
                MI_F64,   low_thresh,
                MI_F64,   high_thresh,
                MI_BOOL,  l2_gradient,
                OpTarget, target);

static Status CvCanny(Context *ctx, Mat &src, Mat &dst, MI_F64 low_thresh, MI_F64 high_thresh, MI_S32 aperture_size, MI_BOOL l2_gradient)
{
    if ((src.GetElemType() != ElemType::U8) || (dst.GetElemType() != ElemType::U8))
    {
        AURA_LOGE(ctx, AURA_TAG, "CV CvCanny only support type u8\n");
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    MI_S32 src_cv_type = ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    MI_S32 dst_cv_type = ElemTypeToOpencv(dst.GetElemType(), dst.GetSizes().m_channel);
    MI_S32 src_cn = src.GetSizes().m_channel;
    MI_S32 dst_cn = dst.GetSizes().m_channel;
    MI_S32 cv_type = 0;

    if ((CV_8UC(src_cn) != src_cv_type) || (CV_8UC(dst_cn) != dst_cv_type))
    {
        cv_type = -1;
    }

    if (cv_type != -1)
    {
        cv::Mat cv_src = MatToOpencv(src);
        cv::Mat cv_dst = MatToOpencv(dst);

        cv::Canny(cv_src, cv_dst, low_thresh, high_thresh, aperture_size, l2_gradient);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV canny not support\n");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(low_thresh);
    AURA_UNUSED(high_thresh);
    AURA_UNUSED(aperture_size);
    AURA_UNUSED(l2_gradient);
#endif

    return Status::OK;
}

static Status CvCanny(Context *ctx, Mat &dx, Mat &dy, Mat &dst, MI_F64 low_thresh, MI_F64 high_thresh, MI_BOOL l2_gradient)
{
    if ((dx.GetElemType() != ElemType::S16) || (dy.GetElemType() != ElemType::S16) || (dst.GetElemType() != ElemType::U8))
    {
        AURA_LOGE(ctx, AURA_TAG, "CV CvCanny not support, only support s16 input and u8 output\n");
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    MI_S32 dx_cv_type  = ElemTypeToOpencv(dx.GetElemType(), dx.GetSizes().m_channel);
    MI_S32 dy_cv_type  = ElemTypeToOpencv(dy.GetElemType(), dy.GetSizes().m_channel);
    MI_S32 dst_cv_type = ElemTypeToOpencv(dst.GetElemType(), dst.GetSizes().m_channel);
    MI_S32 dx_cn   = dx.GetSizes().m_channel;
    MI_S32 dy_cn   = dy.GetSizes().m_channel;
    MI_S32 dst_cn  = dst.GetSizes().m_channel;
    MI_S32 cv_type = 0;

    if ((CV_16SC(dx_cn) != dx_cv_type) || (CV_16SC(dy_cn) != dy_cv_type) || (CV_8UC(dst_cn) != dst_cv_type))
    {
        cv_type = -1;
    }

    if (cv_type != -1)
    {
        cv::Mat cv_dx  = MatToOpencv(dx);
        cv::Mat cv_dy  = MatToOpencv(dy);
        cv::Mat cv_dst = MatToOpencv(dst);

        cv::Canny(cv_dx, cv_dy, cv_dst, low_thresh, high_thresh, l2_gradient);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "CV canny not support\n");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(low_thresh);
    AURA_UNUSED(high_thresh);
    AURA_UNUSED(l2_gradient);
#endif

    return Status::OK;
}

class CannyPixelTest : public TestBase<CannyPixelParam::TupleTable, CannyPixelParam::Tuple>
{
public:
    CannyPixelTest(Context *ctx, CannyPixelParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status ret = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        ret = m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        ret |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});
        if (ret != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in ResizeTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        CannyPixelParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (1 == run_param.imat_sizes.m_sizes.m_channel &&
                    run_param.imat_sizes.m_sizes.m_width  < 800 &&
                    run_param.imat_sizes.m_sizes.m_height < 600)
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
        using AuraCannyFunc = Status(*)(Context*, const Mat&, Mat&, MI_F64, MI_F64, MI_S32, MI_BOOL, const OpTarget&);
        using CvCannyFunc   = Status(*)(Context*, Mat&, Mat&, MI_F64, MI_F64, MI_S32, MI_BOOL);

        // get next param set
        CannyPixelParam run_param(GetParam((index)));

        // creat iauras
        Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, run_param.elem_type, run_param.imat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.imat_sizes.m_strides);

        Sizes3 omat_size = run_param.imat_sizes.m_sizes;
        omat_size.m_channel = 1;
        Sizes omat_strides = run_param.imat_sizes.m_strides;

        MatSize omat_sizes(omat_size, omat_strides);
        Mat dst = m_factory.GetEmptyMat(run_param.elem_type, omat_size, AURA_MEM_DEFAULT, omat_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.elem_type, omat_size, AURA_MEM_DEFAULT, omat_strides);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = "CannyPixel | low_thresh:" + std::to_string(run_param.low_thresh) + " | high_thresh:" + std::to_string(run_param.high_thresh)
                     + " | aperture_size:" + std::to_string(run_param.aperture_size) + " | l2_gradient:" + std::to_string(run_param.l2_gradient);
        result.input  = run_param.imat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = omat_sizes.ToString();

        // run interface
        Status ret = Executor(loop_count, 2, time_val, AuraCannyFunc(ICanny), m_ctx, src, dst, run_param.low_thresh,
                              run_param.high_thresh, run_param.aperture_size, run_param.l2_gradient, run_param.target);

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
            ret = Executor(10, 2, time_val, CvCannyFunc(CvCanny), m_ctx, src, ref, run_param.low_thresh,
                           run_param.high_thresh, run_param.aperture_size, run_param.l2_gradient);
            result.accu_benchmark = "OpenCV::CannyPixel";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvCanny execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            ret = ICanny(m_ctx, src, ref, run_param.low_thresh, run_param.high_thresh, run_param.aperture_size, run_param.l2_gradient, TargetType::NONE);
            result.accu_benchmark = "CannyPixel(target::none)";

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

class CannyGradientTest : public TestBase<CannyGradientParam::TupleTable, CannyGradientParam::Tuple>
{
public:
    CannyGradientTest(Context *ctx, CannyGradientParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});
        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in ResizeTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        CannyGradientParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (1 == run_param.imat_sizes.m_sizes.m_channel &&
                    run_param.imat_sizes.m_sizes.m_width  < 800 &&
                    run_param.imat_sizes.m_sizes.m_height < 600)
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
        using AuraCannyFunc = Status(*)(Context*, const Mat&, const Mat&, Mat&, MI_F64 , MI_F64, MI_BOOL, const OpTarget&);
        using CvCannyFunc   = Status(*)(Context*, Mat&, Mat&, Mat&, MI_F64, MI_F64, MI_BOOL);

        // get next param set
        CannyGradientParam run_param(GetParam((index)));

        // creat iauras
        Mat dx = m_factory.GetDerivedMat(1.0f, 0.0f, run_param.imat_elem_type, run_param.imat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.imat_sizes.m_strides);
        Mat dy = m_factory.GetDerivedMat(1.0f, 0.0f, run_param.imat_elem_type, run_param.imat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.imat_sizes.m_strides);

        Sizes3 omat_size = run_param.imat_sizes.m_sizes;
        omat_size.m_channel = 1;
        Sizes omat_strides = run_param.imat_sizes.m_strides;

        MatSize omat_sizes(omat_size, omat_strides);
        Mat dst = m_factory.GetEmptyMat(run_param.omat_elem_type, omat_size, AURA_MEM_DEFAULT, omat_strides);
        Mat ref = m_factory.GetEmptyMat(run_param.omat_elem_type, omat_size, AURA_MEM_DEFAULT, omat_strides);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = "CannyGradient | low_thresh:" + std::to_string(run_param.low_thresh) + " | high_thresh:" + std::to_string(run_param.high_thresh)
                     + " | l2_gradient:" + std::to_string(run_param.l2_gradient);
        result.input  = run_param.imat_sizes.ToString() + " " + ElemTypesToString(run_param.imat_elem_type);
        result.output = omat_sizes.ToString() + " " + ElemTypesToString(run_param.omat_elem_type);

        // run interface
        Status ret = Executor(loop_count, 2, time_val, AuraCannyFunc(ICanny), m_ctx, dx, dy, dst,
                              run_param.low_thresh, run_param.high_thresh, run_param.l2_gradient, run_param.target);

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
            ret = Executor(10, 2, time_val, CvCannyFunc(CvCanny), m_ctx, dx, dy, ref, run_param.low_thresh,
                           run_param.high_thresh, run_param.l2_gradient);
            result.accu_benchmark = "OpenCV::CannyGradient";

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvCanny execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            ret = ICanny(m_ctx, dx, dy, ref, run_param.low_thresh, run_param.high_thresh, run_param.l2_gradient, TargetType::NONE);
            result.accu_benchmark = "CannyGradient(target::none)";

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
        m_factory.PutMats(dx, dy, dst, ref);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_CANNY_UINT_TEST_HPP__