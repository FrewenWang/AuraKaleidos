#ifndef AURA_OPS_MATRIX_MIN_MAX_LOC_UNIT_TEST_HPP__
#define AURA_OPS_MATRIX_MIN_MAX_LOC_UNIT_TEST_HPP__

#include "aura/ops/matrix.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(MinMaxLocParam,
                ElemType,  elem_type,
                MatSize,   mat_size,
                OpTarget,  target);

AURA_INLINE Status CvMinMaxLoc(Mat &src, MI_F64 &ref_min, MI_F64 &ref_max, Point3i &min_pos, Point3i &max_pos)
{
    if (ElemType::U32 == src.GetElemType() || ElemType::F16 == src.GetElemType())
    {
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    MI_F64 min = 0.0;
    MI_F64 max = 0.0;

    cv::Mat cv_src = MatToOpencv(src);

    Sizes3 sz = src.GetSizes();
    MI_S32 min_p[3] = {-1, -1, -1};
    MI_S32 max_p[3] = {-1, -1, -1};

    if (1 == sz.m_channel)
    {
        cv::minMaxIdx(cv_src, &min, &max, min_p, max_p);
    }
    else
    {
        cv::minMaxIdx(cv_src, &min, &max, MI_NULL, MI_NULL);
    }

    ref_min = min;
    ref_max = max;
    min_pos.m_y = min_p[0];
    min_pos.m_x = min_p[1];
    min_pos.m_z = min_p[2];

    max_pos.m_y = max_p[0];
    max_pos.m_x = max_p[1];
    max_pos.m_z = max_p[2];
#else
    AURA_UNUSED(ref_min);
    AURA_UNUSED(ref_max);
    AURA_UNUSED(min_pos);
    AURA_UNUSED(max_pos);
#endif

    return Status::OK;
}

class MinMaxLocTest : public TestBase<MinMaxLocParam::TupleTable, MinMaxLocParam::Tuple>
{
public:
    MinMaxLocTest(Context *ctx, MinMaxLocParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {}

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        // Get next param set
        MinMaxLocParam run_param(GetParam((index)));
        ElemType elem_type = run_param.elem_type;
        MatSize  mat_size  = run_param.mat_size;

        // Print param info
        AURA_LOGI(m_ctx, AURA_TAG, "\n\n######################### MinMaxLoc param: %s\n", run_param.ToString().c_str());

        // Create src mats
        Mat src_mat = m_factory.GetRandomMat((MI_F32)INT32_MIN, (MI_F32)INT32_MAX, elem_type, mat_size.m_sizes);

        TestTime time_val;
        TestResult result;
        result.input  = mat_size.ToString() + " " + ElemTypesToString(elem_type);
        result.output = "MI_F64, MI_F64";

        Status ret = Status::OK;

        MI_F64 res_min = 0.0;
        MI_F64 res_max = 0.0;
        MI_F64 ref_min = 0.0;
        MI_F64 ref_max = 0.0;

        Point3i res_min_pos = {-1, -1, -1};
        Point3i res_max_pos = {-1, -1, -1};

        Point3i ref_min_pos = {-1, -1, -1};
        Point3i ref_max_pos = {-1, -1, -1};

        // Run interface
        MI_S32 loop_count = stress_count ? stress_count : 10;

        ret = Executor(loop_count, 2, time_val, IMinMaxLoc, m_ctx, src_mat, &res_min, &res_max, &res_min_pos, &res_max_pos, run_param.target);

        if (Status::OK == ret)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
            result.perf_status = TestStatus::PASSED;
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "Interface execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
            result.perf_status = TestStatus::FAILED;
            result.accu_status = TestStatus::FAILED;
            goto EXIT;
        }

        // Run benchmark
        if (TargetType::NONE == run_param.target.m_type)
        {
            TestTime cv_time;
            result.accu_benchmark = std::string("OpenCV::minMaxIdx");
            ret = Executor(10, 2, cv_time, CvMinMaxLoc, src_mat, ref_min, ref_max, ref_min_pos, ref_max_pos);

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark OpenCV::minMaxIdx execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = cv_time;
        }
        else
        {
            result.accu_benchmark = std::string("MinMaxLoc");
            ret = IMinMaxLoc(m_ctx, src_mat, &ref_min, &ref_max, &ref_min_pos, &ref_max_pos, TargetType::NONE);

            if (ret != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // Compare
        result.accu_status = TestStatus::PASSED;

        if (TargetType::NONE == run_param.target.m_type)
        {
            res_min_pos.m_z = res_max_pos.m_z = -1;
            if (src_mat.GetSizes().m_channel > 1)
            {
                res_min_pos.m_x = res_max_pos.m_x = -1;
                res_min_pos.m_y = res_max_pos.m_y = -1;
            }
        }

        // set compare string
        {
            std::ostringstream oss_dst;
            oss_dst << "[dst]  min:" << res_min << " at " << res_min_pos << " max:" << res_max << " at " << res_max_pos;
            std::ostringstream oss_ref;
            oss_ref << "[ref]  min:" << ref_min << " at " << ref_min_pos << " max:" << ref_max << " at " << ref_max_pos;

            result.accu_result = oss_dst.str() + " " + oss_ref.str();
        }

        if (res_min != ref_min || res_max != ref_max || res_min_pos != ref_min_pos || res_max_pos != ref_max_pos)
        {
            result.accu_status = TestStatus::FAILED;
            std::cout << result.accu_result << std::endl;
            AURA_LOGE(m_ctx, AURA_TAG, "MinMaxLoc failed: %s \n", result.accu_result.c_str());
        }

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);
        m_factory.PutAllMats();
        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_MATRIX_MIN_MAX_LOC_UNIT_TEST_HPP__