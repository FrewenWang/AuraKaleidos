/** @brief      : laplacian uint test head for aura
 *  @file       : laplacian_unit_test.hpp
 *  @author     : liuguangxin1@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : April. 21, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_LAPLACIAN_UINT_TEST_HPP__
#define AURA_OPS_LAPLACIAN_UINT_TEST_HPP__

#include "aura/ops/filter.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

AURA_TEST_PARAM(LaplacianParam,
                MatElemPair, elem_type,
                MatSize,     mat_size,
                DT_S32,      ksize,
                BorderType,  border_type,
                OpTarget,    target);

static Status CvLaplacian(Context *ctx, Mat &src, Mat &dst, LaplacianParam &param)
{
    if (ElemType::F16 == src.GetElemType())
    {
        AURA_LOGE(ctx, AURA_TAG, "CvLaplacian don't support F16\n");
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    cv::Mat src_mat = MatToOpencv(src);
    cv::Mat ref_mat = MatToOpencv(dst);
    DT_S32 ddepth   = ElemTypeToOpencv(dst.GetElemType(), 1);
    cv::Laplacian(src_mat, ref_mat, ddepth, param.ksize, 1.0, 0.0, BorderTypeToOpencv(param.border_type));
#else
    AURA_UNUSED(dst);
    AURA_UNUSED(param);
#endif

    return Status::OK;
}

class LaplacianTest : public TestBase<LaplacianParam::TupleTable, LaplacianParam::Tuple>
{
public:
    LaplacianTest(Context *ctx, LaplacianParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in LaplacianTest\n");
        }
    }

    Status CheckParam(DT_S32 index) override
    {
        LaplacianParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (BorderType::REFLECT_101 == run_param.border_type &&
                    2 == run_param.mat_size.m_sizes.m_channel &&
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

    DT_S32 RunOne(DT_S32 index, TestCase *test_case, DT_S32 stress_count) override
    {
        // get next param set
        LaplacianParam run_param(GetParam((index)));
        ElemType src_elem_type = run_param.elem_type.first;
        ElemType dst_elem_type = run_param.elem_type.second;
        MatSize  mat_size = run_param.mat_size;
        AURA_LOGD(m_ctx, AURA_TAG, "laplacian param detail: elem_type(%s, %s), mat_size(%d, %d, %d), ksize(%d), bordertype(%s) \n",
                  ElemTypesToString(src_elem_type).c_str(), ElemTypesToString(dst_elem_type).c_str(),
                  mat_size.m_sizes.m_channel, mat_size.m_sizes.m_height, mat_size.m_sizes.m_width,
                  run_param.ksize, BorderTypeToString(run_param.border_type).c_str());

        if (TargetType::OPENCL == run_param.target.m_type && run_param.ksize >= 7 && run_param.mat_size.m_sizes.m_channel > 1)
        {
            AURA_LOGD(m_ctx, AURA_TAG, "opencl impl cannot support channel > 1 when ksize >= 7! \n");
            return 0;
        }

        // creat iauras
        Mat src = m_factory.GetDerivedMat(.02f, 0.f, src_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat dst = m_factory.GetEmptyMat(dst_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(dst_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);

        Scalar border_value = Scalar(0, 0, 0, 0);

        DT_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        DT_F64 tolerate = 1.f;

        // run interface
        result.param  = BorderTypeToString(run_param.border_type) + " | ksize:" + std::to_string(run_param.ksize);
        result.input  = mat_size.ToString() + " " + ElemTypesToString(src_elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(dst_elem_type);

        Status status_exec = Executor(loop_count, 2, time_val, ILaplacian, m_ctx, src, dst, run_param.ksize,
                                      run_param.border_type, border_value, run_param.target);
        if (Status::OK == status_exec)
        {
            result.perf_result[TargetTypeToString (run_param.target.m_type)] = time_val;
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
            status_exec = Executor(10, 2, time_val, CvLaplacian, m_ctx, src, ref, run_param);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvLaplacian execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
            result.accu_benchmark = "OpenCV::Laplacian";
        }
        else
        {
            status_exec = ILaplacian(m_ctx, src, ref, run_param.ksize, run_param.border_type, border_value, TargetType::NONE);
            result.accu_benchmark = "Laplacian(target::none)";

            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark execute fail with error: %s\n", m_ctx->GetLogger()->GetErrorString().c_str());
                result.accu_status = TestStatus::FAILED;
                goto EXIT;
            }
        }

        // compare
        if (MatCompare(m_ctx, dst, ref, cmp_result, tolerate) == Status::OK)
        {
            result.accu_status = cmp_result.status ? TestStatus::PASSED : TestStatus::FAILED;
            result.accu_result = cmp_result.ToString();
        }
        else
        {
            AURA_LOGE(m_ctx, AURA_TAG, "mat compare execute fail\n");
        }

EXIT:
        test_case->AddTestResult(result.accu_status && result.perf_status, result);

        // release mat
        m_factory.PutMats(src, dst, ref);

        return 0;
    }

private:
    Context    *m_ctx;
    MatFactory m_factory;
};

#endif // AURA_OPS_LAPLACIAN_UINT_TEST_HPP__
