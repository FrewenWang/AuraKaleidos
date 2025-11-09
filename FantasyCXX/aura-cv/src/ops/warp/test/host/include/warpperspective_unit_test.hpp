/** @brief      : warp uint test head for aura
 *  @file       : warp_unit_test.hpp
 *  @author     : liuxiaokun1@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : June. 21, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_WARP_UINT_TEST_HPP__
#define AURA_OPS_WARP_UINT_TEST_HPP__

#include "aura/ops/warp.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(WarpPerspectiveParam,
                ElemType,   elem_type,
                MatSize,    mat_size,
                InterpType, interp_type,
                BorderType, border_type,
                OpTarget,   target);

static Status CvWarpPerspective(Mat &src, Mat &dst, Mat &mat,
                                InterpType interp_type, BorderType border_type)
{
#if defined(ANDROID)
#  if !defined(__aarch64__)
    // there exists differences on rounding operation of convertTo Function in OPENCV and Aura projects,
    // differences usually occur when rounding data whose decimal is .5 (**.5)
    return Status::ERROR;
#  endif // __aarch64__
#endif   // ANDROID
#if !defined(AURA_BUILD_XPLORER)
    if ((src.GetElemType() == ElemType::S8) ||
        (src.GetElemType() == ElemType::U32) ||
        (src.GetElemType() == ElemType::S32))
    {
        return Status::ERROR;
    }

    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    cv::Mat cv_mat = MatToOpencv(mat);

    cv::warpPerspective(cv_src, cv_dst, cv_mat, cv::Size(cv_dst.cols, cv_dst.rows), static_cast<DT_S32>(interp_type), BorderTypeToOpencv(border_type));
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(mat);
    AURA_UNUSED(interp_type);
    AURA_UNUSED(border_type);
#endif

    return Status::OK;
}

static DT_VOID GetPerspectivePoints(std::vector<Point2> &src_points, std::vector<Point2> &dst_points, const aura::Sizes3 &sizes)
{
    DT_S32 width  = sizes.m_width;
    DT_S32 height = sizes.m_height;

    src_points[0].m_x = 0.f;
    src_points[0].m_y = 0.f;
    src_points[1].m_x = (DT_F32)(width - 1);
    src_points[1].m_y = 0.f;
    src_points[2].m_x = 0.f;
    src_points[2].m_y = (DT_F32)(height - 1);
    src_points[3].m_x = (DT_F32)(width  - 1);
    src_points[3].m_y = (DT_F32)(height - 1);

    dst_points[0].m_x = 0.f;
    dst_points[0].m_y = 0.f;
    dst_points[1].m_x = (DT_F32)(width - 1);
    dst_points[1].m_y = 0.f;
    dst_points[2].m_x = 0.f;
    dst_points[2].m_y = (DT_F32)(height - 1);
    dst_points[3].m_x = (DT_F32)(width  - 1);
    dst_points[3].m_y = (DT_F32)(height - 1);

    return;
}

class WarpPerspectiveTest : public TestBase<WarpPerspectiveParam::TupleTable, WarpPerspectiveParam::Tuple>
{
public:
    WarpPerspectiveTest(Context *ctx, WarpPerspectiveParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb", ElemType::U8, {512, 512, 3});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in WarpPerspectiveTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        WarpPerspectiveParam run_param(GetParam(index));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (BorderType::REFLECT_101 == run_param.border_type &&
                    run_param.mat_size.m_sizes.m_width < 800 &&
                    run_param.mat_size.m_sizes.m_height < 600)
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
        WarpPerspectiveParam run_param(GetParam(index));
        ElemType             src_elem_type = run_param.elem_type;
        ElemType             dst_elem_type = run_param.elem_type;
        MatSize              mat_size      = run_param.mat_size;

        AURA_LOGD(m_ctx, AURA_TAG, "run param: %s\n", run_param.ToString().c_str());

        // creat iauras
        Mat    src = m_factory.GetDerivedMat(1.f, 0, src_elem_type, mat_size.m_sizes);
        Mat    dst = m_factory.GetEmptyMat(dst_elem_type, mat_size.m_sizes);
        Mat    ref = m_factory.GetEmptyMat(((ElemType::F16 == run_param.elem_type) && (TargetType::NONE == run_param.target.m_type)) ? ElemType::F32 : run_param.elem_type, run_param.mat_size.m_sizes);
        Scalar border_value(0, 0, 0, 0);

        std::vector<Point2> src_points(4);
        std::vector<Point2> dst_points(4);
        GetPerspectivePoints(src_points, dst_points, mat_size.m_sizes);
        Mat mat = GetPerspectiveTransform(m_ctx, src_points, dst_points);

        DT_S32       loop_count = stress_count ? stress_count : 10;
        TestTime     time_val;
        MatCmpResult cmp_result;
        TestResult   result;

        result.param = InterpTypeToString(run_param.interp_type) + " | " +
                       BorderTypeToString(run_param.border_type);
        result.input  = mat_size.ToString() + " " + ElemTypesToString(src_elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(dst_elem_type);

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, IWarpPerspective, m_ctx, src, mat, dst,
                                      run_param.interp_type, run_param.border_type, border_value, run_param.target);

        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status                                              = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            Mat src_cv;
            src_cv = m_factory.GetDerivedMat(1.0f, 0.0f, (ElemType::F16 == run_param.elem_type ? ElemType::F32 : run_param.elem_type),
                                             run_param.mat_size.m_sizes);

            status_exec = Executor(loop_count, 2, time_val, CvWarpPerspective, src_cv, ref, mat, run_param.interp_type, run_param.border_type);

            m_factory.PutMats(src_cv);
            result.accu_benchmark = "OpenCV::warpPerspective";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark warpPerspective execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec           = IWarpPerspective(m_ctx, src, mat, ref, run_param.interp_type, run_param.border_type,
                                                     border_value, TargetType::NONE);
            result.accu_benchmark = "IWarpPerspective(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (Status::OK == MatCompare(m_ctx, dst, ref, cmp_result, 1, 0.5))
        {
            if (cmp_result.status || (cmp_result.total - cmp_result.hist[cmp_result.hist.size() - 1].second) / (DT_F64)cmp_result.total <= 0.00003)
            {
                result.accu_status = TestStatus::PASSED;
            }
            else
            {
                result.accu_status = TestStatus::FAILED;
            }
            result.accu_result = cmp_result.ToString();
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
        }

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(src, mat, dst, ref, mat);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_WARP_UINT_TEST_HPP__
