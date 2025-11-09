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
#include "warp_impl.hpp"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(WarpAffineParam,
                ElemType,   elem_type,
                MatSize,    mat_size,
                InterpType, interp_type,
                BorderType, border_type,
                OpTarget,   target);

static Status CvWarpAffine(Mat &src, Mat &dst, Mat &matrix, InterpType interp_type, BorderType border_type)
{
#if !defined(AURA_BUILD_XPLORER)
    if ((src.GetElemType() == ElemType::S8) ||
        (src.GetElemType() == ElemType::U32) ||
        (src.GetElemType() == ElemType::S32))
    {
        return Status::ERROR;
    }

    cv::Mat cv_src = MatToOpencv(src);
    cv::Mat cv_dst = MatToOpencv(dst);
    cv::Mat cv_mat = MatToOpencv(matrix);

    cv::warpAffine(cv_src, cv_dst, cv_mat, cv::Size(cv_dst.cols, cv_dst.rows), static_cast<DT_S32>(interp_type), BorderTypeToOpencv(border_type));
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(matrix);
    AURA_UNUSED(interp_type);
    AURA_UNUSED(border_type);
#endif

    return Status::OK;
}

class WarpAffineTest : public TestBase<WarpAffineParam::TupleTable, WarpAffineParam::Tuple>
{
public:
    WarpAffineTest(Context *ctx, WarpAffineParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb", ElemType::U8, {512, 512, 3});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in WarpAffineTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        WarpAffineParam run_param(GetParam(index));
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
        WarpAffineParam run_param(GetParam(index));
        ElemType        src_elem_type = run_param.elem_type;
        ElemType        dst_elem_type = run_param.elem_type;

        MatSize mat_size = run_param.mat_size;
        Sizes3  coord_size(mat_size.m_sizes.m_height, mat_size.m_sizes.m_width, 2);

        AURA_LOGD(m_ctx, AURA_TAG, "run param: %s\n", run_param.ToString().c_str());

        // creat iauras
        Mat src = m_factory.GetDerivedMat(1.f, 0, src_elem_type, mat_size.m_sizes);
        Mat dst = m_factory.GetEmptyMat(dst_elem_type, mat_size.m_sizes);
        Mat ref = m_factory.GetEmptyMat(((ElemType::F16 == dst_elem_type) && (TargetType::NONE == run_param.target.m_type))
                                        ? ElemType::F32 : dst_elem_type, mat_size.m_sizes);

        Point2 center = Point2(mat_size.m_sizes.m_width / 2, mat_size.m_sizes.m_height / 2);
        Mat    matrix = GetRotationMatrix2D(m_ctx, center, -10.0, 1.1);

        DT_S32 coord_stride = ((coord_size.m_width + 128) & (-128)) * coord_size.m_channel * ElemTypeSize(ElemType::S16);
        Mat    dst_coord    = m_factory.GetEmptyMat(ElemType::S16, coord_size, AURA_MEM_DEFAULT, aura::Sizes(coord_size.m_height, coord_stride));
        Mat    ref_coord    = m_factory.GetEmptyMat(ElemType::S16, coord_size, AURA_MEM_DEFAULT, aura::Sizes(coord_size.m_height, coord_stride));

        Scalar border_value(0, 0, 0, 0);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime     time_val;
        MatCmpResult cmp_result;
        TestResult   result;

        result.param = InterpTypeToString(run_param.interp_type) + " | " +
                       BorderTypeToString(run_param.border_type);
        result.input  = mat_size.ToString() + " " + ElemTypesToString(src_elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(dst_elem_type);

        // run interface
        Status status_exec = Status::OK;

        if (TargetType::HVX == run_param.target.m_type)
        {
#if defined(AURA_ENABLE_HEXAGON)
            RealTimeInfo rt_info;
            if (m_ctx->GetHexagonEngine()->QueryRTInfo(HexagonRTQueryType::VTCM_INFO, rt_info) != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "Get VTCM info failed\n");
                goto EXIT;
            }

            if (rt_info.vtcm_layout.total_vtcm_size * 1024 < src.GetTotalBytes())
            {
                goto EXIT;
            }
#endif
            status_exec = Executor(loop_count, 2, time_val, IWarpAffine, m_ctx, src, matrix, dst,
                                   run_param.interp_type, run_param.border_type, border_value, run_param.target);

            // run coord interface
            status_exec |= Executor(1, 0, time_val, WarpCoord, m_ctx, matrix, dst_coord, WarpType::AFFINE, run_param.target);
        }
        else
        {
            status_exec = Executor(loop_count, 2, time_val, IWarpAffine, m_ctx, src, matrix, dst,
                                   run_param.interp_type, run_param.border_type, border_value, run_param.target);
        }

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

            status_exec = Executor(loop_count, 2, time_val, CvWarpAffine, src_cv, ref, matrix, run_param.interp_type, run_param.border_type);

            m_factory.PutMats(src_cv);
            result.accu_benchmark = "OpenCV::warpAffine";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark warpAffine execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = IWarpAffine(m_ctx, src, matrix, ref, run_param.interp_type, run_param.border_type,
                                      border_value, TargetType::NONE);

            if (TargetType::HVX == run_param.target.m_type)
            {
                // run coord interface
                status_exec |= WarpCoord(m_ctx, matrix, ref_coord, WarpType::AFFINE, TargetType::NONE);
            }

            result.accu_benchmark = "IWarpAffine(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (TargetType::HVX == run_param.target.m_type)
        {
            if (Status::OK == MatCompare(m_ctx, dst_coord, ref_coord, cmp_result, 1, 1))
            {
                result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
                result.accu_result = cmp_result.ToString();
            }
            else
            {
                AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            }
        }
        else
        {
            if (Status::OK == MatCompare(m_ctx, dst, ref, cmp_result, 1, 0.5))
            {
                result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
                result.accu_result = cmp_result.ToString();
            }
            else
            {
                AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            }
        }

    EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(src, dst, ref, dst_coord, ref_coord);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_WARP_UINT_TEST_HPP__
