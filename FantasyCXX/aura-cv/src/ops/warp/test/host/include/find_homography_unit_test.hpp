/** @brief      : homography uint test head for aura
 *  @file       : homography_unit_test.hpp
 *  @author     : yangsen7@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Jan. 22, 2024
 *  @Copyright  : Copyright 2024 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_HOMOGRAPHY_UINT_TEST_HPP__
#define AURA_OPS_HOMOGRAPHY_UINT_TEST_HPP__

#include "aura/ops/warp.h"
#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(FindHomographyParam,
                MatSize,    iaura_size,
                DT_S32,     point_num);

#if !defined(AURA_BUILD_XPLORER)
AURA_INLINE DT_S32 PointToOpencv(const std::vector<Point2> &src_points, std::vector<cv::Point2f> &ref_points)
{
    DT_S32 size = (DT_S32)src_points.size();

    for (DT_S32 i = 0; i < size; i++)
    {
        ref_points[i].x = src_points[i].m_x;
        ref_points[i].y = src_points[i].m_y;
    }

    return 0;
}
#endif

static Status AuraFindHomography(Context *ctx, const std::vector<Point2> &src_points, const std::vector<Point2> &dst_points, Mat &h_mat)
{
    Mat h = FindHomography(ctx, src_points, dst_points);
    if (!h.IsValid())
    {
        return Status::ERROR;
    }
    h_mat = h;

    return Status::OK;
}

static Status CvFindHomography(const std::vector<Point2> &src_points, const std::vector<Point2> &dst_points, Mat &h_mat)
{
#if !defined(AURA_BUILD_XPLORER)
    std::vector<cv::Point2f> cv_src_points(src_points.size());
    std::vector<cv::Point2f> cv_dst_points(dst_points.size());
    PointToOpencv(src_points, cv_src_points);
    PointToOpencv(dst_points, cv_dst_points);
    cv::Mat cv_h = MatToOpencv(h_mat);

    cv_h = cv::findHomography(cv_src_points, cv_dst_points, cv::RANSAC);

    for (DT_S32 i = 0; i < 3; i++)
    {
        DT_F64 *href_row = (DT_F64*)h_mat.Ptr<DT_F64>(i);
        for (DT_S32 j = 0; j < 3; j++)
        {
            href_row[j] = (DT_F64)(cv_h.at<double>(i, j));
        }
    }
#else
    AURA_UNUSED(src_points);
    AURA_UNUSED(dst_points);
    AURA_UNUSED(h_mat);
#endif

    return Status::OK;
}

static Status CreateInputData(Context *ctx, std::vector<Point2> &src_points, std::vector<Point2> &dst_points, Sizes3 iaura_size)
{
    std::uniform_real_distribution<DT_F32> distributer(0.f, 1.f);
    std::mt19937_64 engine(std::mt19937_64::default_seed);

    DT_S32 iaura_width  = iaura_size.m_width;
    DT_S32 iaura_height = iaura_size.m_height;
    DT_S32 points_num   = (DT_S32)src_points.size();

    Mat src_mat_3d = Mat(ctx, ElemType::F32, Sizes3(3, points_num, 1), AURA_MEM_HEAP);
    Mat dst_mat_3d = Mat(ctx, ElemType::F32, Sizes3(3, points_num, 1), AURA_MEM_HEAP);
    for (DT_S32 i = 0; i < points_num; i++)
    {
        DT_F32 cx = SaturateCast<DT_F32>(distributer(engine) * iaura_width);
        DT_F32 cy = SaturateCast<DT_F32>(distributer(engine) * iaura_height);

        src_mat_3d.At<DT_F32>(0, i) = cx;
        src_mat_3d.At<DT_F32>(1, i) = cy;
        src_mat_3d.At<DT_F32>(2, i) = 1.0f;

        src_points[i] = Point2(cx, cy);
    }

    Mat h_mat_f32 = Mat(ctx, ElemType::F32, Sizes3(3, 3, 1), AURA_MEM_HEAP);
    DT_F64 angle  = SaturateCast<DT_F64>(distributer(engine) * 2 * AURA_PI);
    DT_F64 tx     = SaturateCast<DT_F64>(distributer(engine) * sqrt(iaura_width));
    DT_F64 ty     = SaturateCast<DT_F64>(distributer(engine) * sqrt(iaura_height));

    h_mat_f32.At<DT_F32>(0, 0) = Cos(angle); h_mat_f32.At<DT_F32>(0, 1) = -Sin(angle); h_mat_f32.At<DT_F32>(0, 2) = tx;
    h_mat_f32.At<DT_F32>(1, 0) = Sin(angle); h_mat_f32.At<DT_F32>(1, 1) =  Cos(angle); h_mat_f32.At<DT_F32>(1, 2) = ty;
    h_mat_f32.At<DT_F32>(2, 0) = 0.f;        h_mat_f32.At<DT_F32>(2, 1) =  0.f;        h_mat_f32.At<DT_F32>(2, 2) = 1.0f;

    aura::IGemm(ctx, h_mat_f32, src_mat_3d, dst_mat_3d, TargetType::NONE);

    for (DT_S32 i = 0; i < points_num; i++)
    {
        DT_F32 cx = dst_mat_3d.At<DT_F32>(0, i) / dst_mat_3d.At<DT_F32>(2, i);
        DT_F32 cy = dst_mat_3d.At<DT_F32>(1, i) / dst_mat_3d.At<DT_F32>(2, i);

        dst_points[i] = Point2(cx, cy);
    }

    return Status::OK;
}

class FindHomographyTest : public TestBase<FindHomographyParam::TupleTable, FindHomographyParam::Tuple>
{
public:
    FindHomographyTest(Context *ctx, FindHomographyParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // get next param set
        FindHomographyParam run_param(GetParam(index));
        MatSize matrix_size = run_param.iaura_size;
        DT_S32 point_num    = run_param.point_num;

        AURA_LOGD(m_ctx, AURA_TAG, "run param: %s\n", run_param.ToString().c_str());

        // creat inputs
        std::vector<Point2> src_points(point_num);
        std::vector<Point2> dst_points(point_num);
        CreateInputData(m_ctx, src_points, dst_points, matrix_size.m_sizes);
        Mat dst_matrix = m_factory.GetEmptyMat(ElemType::F64, Sizes3(3, 3, 1));
        Mat ref_matrix = m_factory.GetEmptyMat(ElemType::F64, Sizes3(3, 3, 1));

        DT_S32       loop_count = stress_count ? stress_count : 10;
        TestTime     time_val;
        MatCmpResult cmp_result;
        TestResult   result;

        result.param  = matrix_size.ToString() + " | point_num: " + std::to_string(point_num);
        result.input  = std::to_string(point_num) + " " + ElemTypesToString(ElemType::F32);
        result.output = MatSize(Sizes3(3, 3, 1)).ToString()+ " " + ElemTypesToString(ElemType::F64);

        // run interface
        Status status_exec = Executor(loop_count, 2, time_val, AuraFindHomography, m_ctx, src_points, dst_points, dst_matrix);
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(TargetType::NONE)] = time_val;
            result.perf_status                                       = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // run benchmark
        status_exec = Executor(loop_count, 2, time_val, CvFindHomography, src_points, dst_points, ref_matrix);
        result.accu_benchmark = "OpenCV::findHomography";
        if (status_exec != Status::OK)
        {
            AURA_LOGE(m_ctx, AURA_TAG, "benchmark FindHomography execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.accu_status = TestStatus::UNTESTED;
            goto EXIT;
        }
        result.perf_result["OpenCV"] = time_val;

        // compare
        if (Status::OK == MatCompare(m_ctx, dst_matrix, ref_matrix, cmp_result, 1e-12))
        {
            result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
            result.accu_result = cmp_result.ToString();
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
        }

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(dst_matrix, ref_matrix);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_HOMOGRAPHY_UINT_TEST_HPP__