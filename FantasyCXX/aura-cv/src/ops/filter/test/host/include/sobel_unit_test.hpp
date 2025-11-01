/** @brief      : sobel uint test head for aura
 *  @file       : sobel_unit_test.hpp
 *  @author     : liuguangxin1@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : May. 5, 2022
 *  @Copyright  : Copyright 2022 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_OPS_SOBEL_UINT_TEST_HPP__
#define AURA_OPS_SOBEL_UINT_TEST_HPP__

#include "aura/ops/filter.h"
#include "aura/tools/unit_test.h"
#if !defined(AURA_BUILD_XPLORER)
#  include "opencv_helper.hpp"
#endif

using namespace aura;

struct SobelTestParam
{
    SobelTestParam()
    {}

    SobelTestParam(MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale) : dx(dx), dy(dy), ksize(ksize), scale(scale)
    {}

    friend std::ostream& operator<<(std::ostream &os, const SobelTestParam &sobel_test_param)
    {
        os << "ksize:" << sobel_test_param.ksize << " | dx:" << sobel_test_param.dx << " | dy:" << sobel_test_param.dy
           << " | scale:" << sobel_test_param.scale;
        return os;
    }

    std::string ToString()
    {
        std::stringstream sstream;
        sstream << *this;
        return sstream.str();
    }

    MI_S32 dx;
    MI_S32 dy;
    MI_S32 ksize;
    MI_F32 scale;
};

AURA_TEST_PARAM(SobelParam,
                MatElemPair,    elem_type,
                MatSize,        mat_size,
                SobelTestParam, sobel_test_param,
                BorderType,     border_type,
                OpTarget,       target);

static Status CvSobel(Context *ctx, Mat &src, Mat &dst, SobelParam &param)
{
    AURA_UNUSED(ctx);

#if !defined(AURA_BUILD_XPLORER)
    SobelTestParam &sobel_test_param = param.sobel_test_param;

    cv::Mat src_mat = MatToOpencv(src);
    cv::Mat ref_mat = MatToOpencv(dst);
    MI_S32 ddepth   = ElemTypeToOpencv(dst.GetElemType(), 1);
    cv::Sobel(src_mat, ref_mat, ddepth, sobel_test_param.dx, sobel_test_param.dy,
              sobel_test_param.ksize, sobel_test_param.scale, 0.f, BorderTypeToOpencv(param.border_type));
#else
    AURA_UNUSED(src);
    AURA_UNUSED(dst);
    AURA_UNUSED(param);
#endif

    return Status::OK;
}

class SobelTest : public TestBase<SobelParam::TupleTable, SobelParam::Tuple>
{
public:
    SobelTest(Context *ctx, SobelParam::TupleTable &table) : TestBase(table), m_ctx(ctx), m_factory(ctx)
    {
        Status status = Status::OK;

        std::string data_file = UnitTest::GetInstance()->GetDataPath() + "comm/";

        status |= m_factory.LoadBaseMat(data_file + "cameraman_487x487.gray", ElemType::U8, {487, 487});
        status |= m_factory.LoadBaseMat(data_file + "lena_256x256x2.uv", ElemType::U8, {256, 256, 2});
        status |= m_factory.LoadBaseMat(data_file + "baboon_512x512.rgb",  ElemType::U8, {512, 512, 3});

        if (status != Status::OK)
        {
            AURA_LOGD(ctx, AURA_TAG, "LoadBaseMat failed in SobelTest\n");
        }
    }

    Status CheckParam(MI_S32 index) override
    {
        SobelParam run_param(GetParam((index)));
        if (UnitTest::GetInstance()->IsStressMode())
        {
            if (TargetType::NONE == run_param.target.m_type)
            {
                if (BorderType::REFLECT_101 == run_param.border_type &&
                    2 == run_param.mat_size.m_sizes.m_channel)
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
        SobelParam run_param(GetParam((index)));
        ElemType src_elem_type = run_param.elem_type.first;
        ElemType dst_elem_type = run_param.elem_type.second;
        MatSize  mat_size = run_param.mat_size;
        SobelTestParam sobel_test_param = run_param.sobel_test_param;
        AURA_LOGD(m_ctx, AURA_TAG, "sobel param detail: elem_type(%s, %s), mat_size(%d, %d, %d), test_param(%s), bordertype(%s) \n",
                  ElemTypesToString(src_elem_type).c_str(), ElemTypesToString(dst_elem_type).c_str(),
                  mat_size.m_sizes.m_channel, mat_size.m_sizes.m_height, mat_size.m_sizes.m_width,
                  sobel_test_param.ToString().c_str(), BorderTypeToString(run_param.border_type).c_str());

        // creat iauras
        MI_BOOL promote_fp16 = ElemType::F16 == src_elem_type && TargetType::NONE == run_param.target.m_type;
        ElemType ref_elem_type = promote_fp16 ? ElemType::F32 : dst_elem_type;
        Mat src = m_factory.GetDerivedMat(1.f, 0.f, src_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat dst = m_factory.GetEmptyMat(dst_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);
        Mat ref = m_factory.GetEmptyMat(ref_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);

        Scalar border_value = Scalar(0, 0, 0, 0);

        MI_S32 loop_count = stress_count ? stress_count : 10;

        TestTime time_val;
        MatCmpResult cmp_result;
        TestResult result;
        MI_F64 tolerate = promote_fp16 ? 2.f : 1.f;

        // run interface
        result.param  = BorderTypeToString(run_param.border_type) + " | " + sobel_test_param.ToString();
        result.input  = mat_size.ToString() + " " + ElemTypesToString(src_elem_type);
        result.output = mat_size.ToString() + " " + ElemTypesToString(dst_elem_type);

        Status status_exec = Executor(loop_count, 2, time_val, ISobel, m_ctx, src, dst, sobel_test_param.dx, sobel_test_param.dy,
                                      sobel_test_param.ksize, sobel_test_param.scale, run_param.border_type, border_value, run_param.target);
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
            ElemType cv_src_elem_type = promote_fp16 ? ElemType::F32 : src_elem_type;
            Mat src_cv = m_factory.GetDerivedMat(1.f, 0.f, cv_src_elem_type, mat_size.m_sizes, AURA_MEM_DEFAULT, mat_size.m_strides);

            status_exec = Executor(10, 2, time_val, CvSobel, m_ctx, src_cv, ref, run_param);
            result.accu_benchmark = "OpenCV::Sobel";
            m_factory.PutMats(src_cv);
            if (status_exec != Status::OK)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "benchmark CvSobel execute fail\n");
                result.accu_status = TestStatus::UNTESTED;
                goto EXIT;
            }
            result.perf_result["OpenCV"] = time_val;
        }
        else
        {
            status_exec = ISobel(m_ctx, src, ref, sobel_test_param.dx, sobel_test_param.dy, sobel_test_param.ksize,
                                 sobel_test_param.scale, run_param.border_type, border_value, TargetType::NONE);
            result.accu_benchmark = "Sobel(target::none)";
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

#endif // AURA_OPS_SOBEL_UINT_TEST_HPP__
