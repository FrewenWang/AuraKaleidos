/** @brief      : creat clahe unit test head for aura
 *  @file       : creat_clahe_unit_test.hpp
 *  @author     : zhangqiongqiong@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : July. 15, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_CREAT_CLAHE_UNIT_TEST_HPP__
#define AURA_OPS_CREAT_CLAHE_UNIT_TEST_HPP__

#include "aura/ops/hist.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct CreatCLAHETestParam
{
    CreatCLAHETestParam()
    {}

    CreatCLAHETestParam(MI_F64 clip_limit, Sizes tile_grid_size) : clip_limit(clip_limit), tile_grid_size(tile_grid_size)
    {}

    friend std::ostream& operator<<(std::ostream &os, const CreatCLAHETestParam &clahe_test_param)
    {
        os << "clip_limit:" << clahe_test_param.clip_limit << " | hist_size: " << clahe_test_param.tile_grid_size.m_height
           << " " << clahe_test_param.tile_grid_size.m_width;
        return os;
    }

    std::string ToString()
    {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    MI_F64 clip_limit;
    Sizes tile_grid_size;
};

AURA_TEST_PARAM(CreatCLAHEParam,
                ElemType,            elem_type,
                MatSize,             mat_sizes,
                CreatCLAHETestParam, param,
                OpTarget,            target);

static Status CvCreateClAHE(Context *ctx, Mat &src, Mat &dst,
                                  MI_F64 clip_limit, Sizes tile_grid_size)
{
#if !defined(AURA_BUILD_XPLORER)
    MI_S32 src_cv_type = ElemTypeToOpencv(src.GetElemType(), src.GetSizes().m_channel);
    MI_S32 dst_cv_type = ElemTypeToOpencv(dst.GetElemType(), dst.GetSizes().m_channel);
    if (src_cv_type != -1 && dst_cv_type != -1)
    {
        cv::Mat cv_src = MatToOpencv(src);
        cv::Mat cv_dst = MatToOpencv(dst);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(clip_limit);
        clahe->setTilesGridSize(cv::Size(tile_grid_size.m_width, tile_grid_size.m_height));
        clahe->apply(cv_src, cv_dst);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "mat type not support\n");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(clip_limit);
    AURA_UNUSED(tile_grid_size);
#endif

    return Status::OK;
}

class CreatCLAHETest : public TestBase<CreatCLAHEParam::TupleTable, CreatCLAHEParam::Tuple>
{
public:
    CreatCLAHETest(Context *ctx, CreatCLAHEParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx, 512)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status = m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in CreatCLAHETest\n");
        }
    }

    MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 stress_count) override
    {
        CreatCLAHEParam run_param(GetParam((index)));

        Mat src = m_factory.GetDerivedMat(1.0f, 0.0f, ElemType::U8, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        Mat dst = m_factory.GetEmptyMat(ElemType::U8, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);
        Mat ref = m_factory.GetEmptyMat(ElemType::U8, run_param.mat_sizes.m_sizes, AURA_MEM_DEFAULT, run_param.mat_sizes.m_strides);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;

        result.param  = run_param.param.ToString();
        result.input  = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);
        result.output = run_param.mat_sizes.ToString() + " " + ElemTypesToString(run_param.elem_type);

        AURA_LOGD(m_ctx, AURA_TAG, "CreatCLAHETest param detail: elem_type(%s),  mat_size(%s), param(%s)\n",
                  ElemTypesToString(run_param.elem_type).c_str(), run_param.mat_sizes.ToString().c_str(),
                  run_param.param.ToString().c_str());

        Status status_exec = Executor(loop_count, 0, time_val, ICreateClAHE, m_ctx, src, dst,
                                                  run_param.param.clip_limit, run_param.param.tile_grid_size, run_param.target);

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
            status_exec = Executor(10, 2, time_val, CvCreateClAHE, m_ctx, src, ref,
                                         run_param.param.clip_limit, run_param.param.tile_grid_size);
            result.accu_benchmark = "OpenCV::CreateClAHE";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvCreateClAHE execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = ICreateClAHE(m_ctx, src, ref, run_param.param.clip_limit, run_param.param.tile_grid_size, TargetType::NONE);
            result.accu_benchmark = "CreateClAHE(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark none execute fail\n");
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }

        }

        MatCompare(m_ctx, dst, ref, cmp_result, 1);
        result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
        result.accu_result = cmp_result.ToString();

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        m_factory.PutMats(src, dst, ref);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_CREAT_CLAHE_UNIT_TEST_HPP__