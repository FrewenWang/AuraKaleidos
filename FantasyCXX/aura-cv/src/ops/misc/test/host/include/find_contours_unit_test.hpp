/** @brief      : find contours uint test head for aura
 *  @file       : find_contours_unit_test.hpp
 *  @author     : wangyisi@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Aug. 10, 2023
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_FIND_CONTOURS_UNIT_TEST_HPP__
#define AURA_OPS_FIND_CONTOURS_UNIT_TEST_HPP__

#include "aura/ops/misc.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;
#if !defined(AURA_BUILD_XPLORER)
static MI_S32 ModeToOpencv(ContoursMode mode)
{
    switch (mode)
    {
        case ContoursMode::RETR_EXTERNAL:
        {
            return 0;
        }
        case ContoursMode::RETR_LIST:
        {
            return 1;
        }
        default:
        {
            return -1;
        }
    }
}

static MI_S32 MethodToOpencv(ContoursMethod method)
{
    switch (method)
    {
        case ContoursMethod::CHAIN_APPROX_NONE:
        {
            return 1;
        }
        case ContoursMethod::CHAIN_APPROX_SIMPLE:
        {
            return 2;
        }
        default:
        {
            return -1;
        }
    }
}
#endif

static Status ResultCompare(Context *ctx, const std::vector<std::vector<Point2i>> &dst,
                            const std::vector<std::vector<Point2i>> &ref, MatCmpResult &cmp_result, MI_F32 tolerate)
{
    if ((dst.size() == 0) || (ref.size() == 0))
    {
        AURA_ADD_ERROR_STRING(ctx, "dst or ref Point2i vector size is 0...");
        return Status::ERROR;
    }
    if (dst.size() != ref.size())
    {
        AURA_ADD_ERROR_STRING(ctx, "dst and ref Point2i vector size not same...");
        return Status::ERROR;
    }

    cmp_result.Clear();

    MI_S32 vec_size = (MI_S32)(dst.size());
    MI_S32 points_size = 0;

    for (MI_S32 i = 0; i < vec_size; i++)
    {
        if (dst[i].size() != ref[i].size())
        {
            std::string error_info = "contours [" + std::to_string(i) + "] points size not same..";
            AURA_ADD_ERROR_STRING(ctx, error_info.c_str());
            return Status::ERROR;
        }
        points_size += (MI_S32)dst[i].size();
    }

    const Sizes3 size_mat = Sizes3(1, 2 * points_size, 1);
    Mat dst_mat = Mat(ctx, aura::ElemType::S32, size_mat);
    Mat ref_mat = Mat(ctx, aura::ElemType::S32, size_mat);
    MI_S32 *dst_data = (MI_S32 *)dst_mat.GetData();
    MI_S32 *ref_data = (MI_S32 *)ref_mat.GetData();

    for (MI_S32 i = 0; i < vec_size; i++)
    {
        for (MI_S32 j = 0; j < (MI_S32)dst[i].size(); j++)
        {
            dst_data[2 * j]     = dst[i][j].m_x;
            dst_data[2 * j + 1] = dst[i][j].m_y;
            ref_data[2 * j]     = ref[i][j].m_x;
            ref_data[2 * j + 1] = ref[i][j].m_y;
        }
    }

    // compare
    if (MatCompare(ctx, dst_mat, ref_mat, cmp_result, tolerate) != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "mat compare execute fail\n");
        return Status::ERROR;
    }

    return Status::OK;
}

AURA_TEST_PARAM(FindContoursParam,
                ElemType,              elem_type,
                MatSize,               mat_size,
                ContoursMode,          mode,
                ContoursMethod,        method,
                OpTarget,              target);

static Status CvFindContours(Context *ctx, Mat &src, std::vector<std::vector<Point2i>> &dst, ContoursMode mode, ContoursMethod method)
{
    AURA_UNUSED(ctx);

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat src_mat   = MatToOpencv(src);
    MI_S32  cv_mode   = ModeToOpencv(mode);
    MI_S32  cv_method = MethodToOpencv(method);
    std::vector<std::vector<cv::Point>> ref_contours;

    cv::findContours(src_mat, ref_contours, cv_mode, cv_method, cv::Point(0, 0));

    dst.resize(ref_contours.size());
    for (MI_S32 i = 0; i < (MI_S32)(dst.size()); i++)
    {
        dst[i].resize(ref_contours[i].size());
        for (MI_S32 j = 0; j < (MI_S32)(dst[i].size()); j++)
        {
            dst[i][j].m_x = ref_contours[i][j].x;
            dst[i][j].m_y = ref_contours[i][j].y;
        }
    }
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(mode);
    AURA_UNUSED(method);
#endif

    return Status::OK;
}

class FindContoursTest : public TestBase<FindContoursParam::TupleTable, FindContoursParam::Tuple>
{
public:
    FindContoursTest(Context *ctx, FindContoursParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status = m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        if (status != Status::OK)
        {
            AURA_LOGD(m_ctx, AURA_TAG, "LoadBaseMat failed in FindContoursTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        FindContoursParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (1 == run_param.mat_size.m_sizes.m_channel &&
                    run_param.mat_size.m_sizes.m_width  < 800 &&
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

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // get next param set
        FindContoursParam run_param(GetParam((index)));
        ElemType elem_type    = run_param.elem_type;
        MatSize &mat_size     = run_param.mat_size;
        ContoursMode   mode   = run_param.mode;
        ContoursMethod method = run_param.method;
        OpTarget target       = run_param.target;
        Point2i offset        = {0, 0};
        AURA_LOGD(m_ctx, AURA_TAG, "find_contours param detail: elem_type(%s), mat_size(%d, %d, %d), mode(%s), method(%s)\n",
                  ElemTypesToString(elem_type).c_str(), mat_size.m_sizes.m_channel, mat_size.m_sizes.m_height, mat_size.m_sizes.m_width,
                  FindContoursModeToString(mode).c_str(), FindContoursMethodToString(method).c_str());
                  
        // usually, FindContours is used after threshold binary mat.
        Mat src_ori = m_factory.GetDerivedMat(1.f, 0.f, elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat src     = src_ori.Clone();
        std::vector<std::vector<Point2i>> dst;
        std::vector<std::vector<Point2i>> ref;
        std::vector<Scalari> hierarchy;

        MI_S32 loop_count = stress_count > 0 ? stress_count : 5;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        // run interface
        result.param  = FindContoursModeToString(mode) + " | " + FindContoursMethodToString(method);
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(elem_type);

        Status status_exec;
        Status ret = aura::IThreshold(m_ctx, src_ori, src, 128, 255, AURA_THRESH_BINARY, OpTarget::None());
        if (ret != Status::OK)
        {
            AURA_LOGD(m_ctx, AURA_TAG, "Threshold in test case failed with error info: \n %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            goto EXIT;
        }

        status_exec = Executor(loop_count, 2, time_val, IFindContours, m_ctx, src, dst, hierarchy, mode, method, offset, target);
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString(run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
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
            status_exec           = Executor(10, 2, time_val, CvFindContours, m_ctx, src, ref, mode, method);
            result.accu_benchmark = "OpenCV::FindContours";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvFindContours execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            Point2i offset = {0, 0};
            status_exec = IFindContours(m_ctx, src, ref, hierarchy, mode, method, offset);

            result.accu_benchmark = "FindContours(target::none)";
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (Status::OK == ResultCompare(m_ctx, dst, ref, cmp_result, 1.f))
        {
            result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
            result.accu_result = cmp_result.ToString();
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "result compare execute fail\n");
        }
        
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

#endif // AURA_OPS_FIND_CONTOURS_UNIT_TEST_HPP__